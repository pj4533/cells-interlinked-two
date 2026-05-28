"""Step 3 — greedy sufficient-consumer-subset composer.

Given the AP-ranked head list from Step 1 and a holdout set of harmful
prompts:

  1. Measure refusal rate under GLOBAL ablation (the existing
     install_runtime_ablation_hook at L32) on the holdout → target_rr.
  2. Measure refusal rate under NO ablation → baseline_rr.
  3. Greedy: sort heads by AP descending. Start with consumer_set = [].
     For each head in order:
       a. add head to consumer_set
       b. install edge-consumer hooks for consumer_set
       c. measure refusal rate on holdout
       d. if |rr − target_rr| <= ε:  BREAK
  4. Return consumer_set + the full trajectory.

Sweep ε ∈ {0.02, 0.05, 0.10} as a sensitivity check; the subset's
elasticity to ε is itself a finding (small subset under loose ε but
huge subset under tight ε → mixed-routing claim).

Refusal rate definition (see refusal_vocab.py):
    fraction of holdout prompts whose first 256 chars of decoded
    generation contain any marker from REFUSAL_MARKERS.

Generation is greedy (temperature=0), 32 new tokens max, via
HF model.generate(). Greedy + the fixed marker vocabulary makes the
metric bit-exact reproducible across runs given the same prompts and
the same hook configuration.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from ..abliteration import install_runtime_ablation_hook
from .hook import install_edge_consumer_ablation_hook
from .memory_safety import mps_empty_cache_safe
from .proj_cache import load_projection_cache
from .refusal_vocab import refusal_rate

logger = logging.getLogger(__name__)


GENERATION_NEW_TOKENS = 32  # tokens generated per holdout prompt
REFUSAL_SCAN_HEAD_CHARS = 256  # window scanned for refusal markers


@dataclass
class SubsetTrajectory:
    """Per-iteration record of how the greedy composition unfolded."""
    epsilon: float
    baseline_refusal_rate: float       # no ablation
    global_refusal_rate: float          # current install_runtime_ablation_hook
    iterations: list[dict[str, Any]] = field(default_factory=list)
    sufficient_subset: list[tuple[int, int]] = field(default_factory=list)
    stopped_reason: str = "pending"     # "matched" | "exhausted"


@torch.no_grad()
def _generate_greedy_batch(
    model: Any,
    raw_tokenizer: Any,
    rendered_prompts: list[str],
    device: str | torch.device,
    *,
    new_tokens: int = GENERATION_NEW_TOKENS,
    eos_ids: tuple[int, ...] = (),
) -> list[str]:
    """Greedy-generate `new_tokens` new tokens for each prompt; return
    DECODED-CONTINUATION-ONLY strings (the prompt is stripped). One
    forward pass per prompt — no batching for now (KV cache makes
    32-token decode cheap, ~1-3s per prompt on M2 Ultra at 12B)."""
    out: list[str] = []
    for p in rendered_prompts:
        ids = raw_tokenizer.encode(p, add_special_tokens=False).ids
        input_ids = torch.tensor([ids], device=device)
        gen = model.generate(
            input_ids,
            max_new_tokens=new_tokens,
            do_sample=False,
            temperature=1.0,    # ignored when do_sample=False, but explicit
            num_beams=1,
            use_cache=True,
            eos_token_id=list(eos_ids) if eos_ids else None,
            pad_token_id=raw_tokenizer.token_to_id("<pad>")
                or list(eos_ids)[0] if eos_ids else 0,
        )
        new_ids = gen[0, input_ids.shape[1]:].tolist()
        out.append(raw_tokenizer.decode(new_ids))
    return out


def measure_refusal_rate(
    model: Any,
    raw_tokenizer: Any,
    rendered_prompts: list[str],
    device: str | torch.device,
    *,
    eos_ids: tuple[int, ...] = (),
    new_tokens: int = GENERATION_NEW_TOKENS,
    head_chars: int = REFUSAL_SCAN_HEAD_CHARS,
) -> tuple[float, list[str]]:
    """Generate continuations for each prompt and compute the
    refusal-marker hit rate. Returns (rate, continuations)."""
    completions = _generate_greedy_batch(
        model, raw_tokenizer, rendered_prompts, device,
        new_tokens=new_tokens, eos_ids=eos_ids,
    )
    return refusal_rate(completions, head_chars=head_chars), completions


def _all_consumer_layers(
    consumer_set: list[tuple[int, int]],
) -> list[int]:
    return sorted({L for (L, _h) in consumer_set})


def compose_sufficient_subset(
    model: Any,
    raw_tokenizer: Any,
    *,
    ranked_heads: list[tuple[int, int]],
    v_safety: Tensor,
    proj_cache_dir,
    holdout_prompts_rendered: list[str],
    extraction_layer: int,
    epsilons: tuple[float, ...] = (0.02, 0.05, 0.10),
    device: str | torch.device,
    eos_ids: tuple[int, ...] = (),
    log_every: int = 4,
    max_heads: int | None = None,
    target_projections: tuple[str, ...] = ("q", "k", "v"),
    cancel_event: threading.Event | None = None,
    empty_cache_every: int = 4,
) -> dict[float, SubsetTrajectory]:
    """Sweep epsilons; return per-ε trajectory + sufficient subset.

    Args:
        ranked_heads: list of (layer, query_head) tuples in DESCENDING
            AP-score order. Step 1's output, post-sort.
        v_safety: 1-D `[d_model]` refusal direction.
        proj_cache_dir: directory containing layer_{ℓ}.pt files; see
            proj_cache.save_projection_cache. Loaded lazily per layer
            that appears in the consumer set.
        holdout_prompts_rendered: harmful prompts pre-rendered with the
            chat template, disjoint from the Step 1 contrast set.
        max_heads: optional cap on subset size (safety net when ε
            cannot be met — prevents pathological full-list traversal).

    Returns: dict mapping ε → SubsetTrajectory.
    """
    # Cache: layer_idx → cache payload (load on demand)
    proj_caches: dict[int, dict[str, Any]] = {}

    def _ensure_cache(L: int) -> None:
        if L not in proj_caches:
            proj_caches[L] = load_projection_cache(proj_cache_dir, L)

    # Baseline: no ablation
    logger.info(
        "measuring baseline refusal rate on %d holdout prompts (no ablation)",
        len(holdout_prompts_rendered),
    )
    baseline_rr, _ = measure_refusal_rate(
        model, raw_tokenizer, holdout_prompts_rendered, device,
        eos_ids=eos_ids,
    )
    logger.info("baseline refusal rate: %.3f", baseline_rr)

    # Global-ablation target
    logger.info("measuring global-ablation refusal rate")
    global_handle = install_runtime_ablation_hook(
        model, extraction_layer, v_safety, alpha=1.0,
    )
    try:
        target_rr, _ = measure_refusal_rate(
            model, raw_tokenizer, holdout_prompts_rendered, device,
            eos_ids=eos_ids,
        )
    finally:
        global_handle.remove()
    logger.info("global-ablation refusal rate (target): %.3f", target_rr)

    results: dict[float, SubsetTrajectory] = {}

    cap = max_heads if max_heads is not None else len(ranked_heads)

    for eps in epsilons:
        traj = SubsetTrajectory(
            epsilon=eps,
            baseline_refusal_rate=baseline_rr,
            global_refusal_rate=target_rr,
        )
        consumer_set: list[tuple[int, int]] = []
        cancelled = False
        for i, head in enumerate(ranked_heads[:cap]):
            # Honor an external cancel (memory watchdog, SIGINT bridge,
            # etc.) BEFORE the expensive iteration kicks off. Writes
            # whatever subset we've built so far with a distinct
            # stopped_reason so postmortem analysis sees "we stopped
            # early because the watchdog tripped" not "we matched ε".
            if cancel_event is not None and cancel_event.is_set():
                traj.sufficient_subset = list(consumer_set)
                traj.stopped_reason = "cancelled"
                logger.warning(
                    "ε=%.3f compose cancelled at iter=%d size=%d "
                    "(external cancel_event set)",
                    eps, i, len(consumer_set),
                )
                cancelled = True
                break
            consumer_set.append(head)
            _ensure_cache(head[0])
            handles = install_edge_consumer_ablation_hook(
                model, consumer_set, v_safety, proj_caches, alpha=1.0,
                target_projections=target_projections,
            )
            try:
                rr, _ = measure_refusal_rate(
                    model, raw_tokenizer, holdout_prompts_rendered, device,
                    eos_ids=eos_ids,
                )
            finally:
                for h in handles:
                    try:
                        h.remove()
                    except Exception:
                        pass
            # Release MPS allocator cache between iterations. Without
            # this, hundreds of generate() calls fragment the allocator
            # and the resident set grows monotonically until swap kicks
            # in. See docs/MEMORY_PRESSURE_LESSONS.md.
            if (i + 1) % empty_cache_every == 0:
                mps_empty_cache_safe()
            traj.iterations.append({
                "iter": i + 1,
                "added_head": head,
                "consumer_set_size": len(consumer_set),
                "refusal_rate": rr,
                "delta_to_global": rr - target_rr,
            })
            if (i + 1) % log_every == 0:
                logger.info(
                    "ε=%.3f iter=%d size=%d rr=%.3f Δ=%+.3f",
                    eps, i + 1, len(consumer_set), rr, rr - target_rr,
                )
            if abs(rr - target_rr) <= eps:
                traj.sufficient_subset = list(consumer_set)
                traj.stopped_reason = "matched"
                logger.info(
                    "ε=%.3f sufficient subset reached at size=%d (rr=%.3f, target=%.3f)",
                    eps, len(consumer_set), rr, target_rr,
                )
                break
        else:
            if not cancelled:
                traj.sufficient_subset = list(consumer_set)
                traj.stopped_reason = "exhausted"
                logger.warning(
                    "ε=%.3f exhausted ranked list (size=%d) without matching "
                    "global rr=%.3f within ε", eps, len(consumer_set), target_rr,
                )
        results[eps] = traj

        # If we got cancelled, don't even start the next ε — bail out
        # of the outer ε loop so the script can write partials and exit.
        if cancelled:
            logger.warning(
                "stopping ε sweep early; %d ε value(s) completed before cancel",
                len(results),
            )
            break

    return results


__all__ = [
    "SubsetTrajectory",
    "compose_sufficient_subset",
    "measure_refusal_rate",
]
