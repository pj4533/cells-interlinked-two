"""Signed per-head attribution: which heads actually CAUSE refusal.

The Phase B v1 AP score (attribution.py) measures *magnitude* of a
head's q/k/v output change when v_safety is patched at L32 — a
sensitivity metric. The issue: sensitivity is direction-agnostic. A
head that uses v_safety to ENCODE register/politeness/formality is
just as sensitive as a head that uses v_safety to TRIGGER refusal. The
first run of Step 3 confirmed this: AP-top heads at layers 44-45
turned out to be late-stage generation heads; ablating them garbled
the comply path and pushed refusal rate UP (40% → 48%), the opposite
of what we want.

This module fixes that with a **signed, behavior-grounded** metric:

  For each candidate head h:
    1. Install ablation hook for ONLY head h.
    2. On each prompt in a small calibration set, capture the
       next-token logits at the final prompt position.
    3. Score the prompt: log P(refusal_first_token | ablated) −
                         log P(comply_first_token | ablated).
       Lower = head's ablation pushed model toward comply (good).
       Higher = head's ablation pushed model toward refuse (bad).
    4. signed_score(h) = log_odds(baseline) − log_odds(ablated).
       Positive ⇒ ablating this head reduces refusal probability ⇒
       this head is a refusal-promoting consumer (the kind we want
       in our subset).
    5. Average over the calibration prompts.

Cost: one forward per (head × prompt). With ~240 heads × 20 prompts
≈ 4800 forwards × ~0.3s ≈ 25 minutes on M2 Ultra. About 12× cheaper
than the original AP run (which did 600 forwards × 200 contrasts).

Output: the same shape as `compute_attribution_scores` — a
dict[(layer, head)] → float. The subset composer takes whichever
ranking is provided, so we can swap rankings without touching it.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import torch
from torch import Tensor

from .hook import install_edge_consumer_ablation_hook
from .memory_safety import mps_empty_cache_safe
from .proj_cache import _attn_module, _attn_shape

logger = logging.getLogger(__name__)


# Phrase pairs we expect to see at the first generated token for
# refuse-vs-comply decisions. The tokenizer encodes each as 1+
# tokens; we use the FIRST token of each encoding as the "refusal
# anchor" or "comply anchor." Multiple anchors per side increase
# coverage (Gemma may pick "I", "I'm", "Sorry", or "As" to start a
# refusal; "Here", "Sure", "Okay", "The" to start a comply).
REFUSAL_ANCHORS = [
    "I cannot",
    "I can't",
    "I'm sorry",
    "I am sorry",
    "Sorry",
    "I'm unable",
    "I am unable",
    "As an AI",
    "Unfortunately",
    "I won't",
    "I will not",
]

COMPLY_ANCHORS = [
    "Here",
    "Sure",
    "Okay",
    "Certainly",
    "To ",
    "The ",
    "First",
    "Step 1",
    "1.",
    "**",
    "```",
]


def _anchor_token_ids(raw_tokenizer: Any, phrases: list[str]) -> set[int]:
    """Return the set of distinct first-token IDs after encoding each
    phrase. Encodes WITHOUT special tokens so we get the natural
    leading-token IDs."""
    ids: set[int] = set()
    for p in phrases:
        enc = raw_tokenizer.encode(p, add_special_tokens=False).ids
        if enc:
            ids.add(int(enc[0]))
    return ids


@torch.no_grad()
def _forward_last_logits(
    model: Any, input_ids: Tensor,
) -> Tensor:
    """One forward pass, return logits at the last position [V]."""
    out = model(input_ids, use_cache=False)
    return out.logits[0, -1, :].detach().to(torch.float32).cpu()


def _log_odds_refuse_minus_comply(
    logits: Tensor,
    refusal_ids: set[int],
    comply_ids: set[int],
) -> float:
    """log P(refusal-anchor token at first position) −
       log P(comply-anchor token at first position).
    Both sides use logsumexp over their token sets. Returns a scalar.
    Higher value = model more likely to refuse."""
    if not refusal_ids or not comply_ids:
        return 0.0
    r_idx = torch.tensor(sorted(refusal_ids))
    c_idx = torch.tensor(sorted(comply_ids))
    # Pull subsetted logits; use full softmax denominator to keep
    # the log P well-formed (subtract log Z).
    log_Z = torch.logsumexp(logits, dim=0)
    log_p_r = torch.logsumexp(logits[r_idx], dim=0) - log_Z
    log_p_c = torch.logsumexp(logits[c_idx], dim=0) - log_Z
    return float(log_p_r - log_p_c)


@torch.no_grad()
def compute_signed_scores(
    model: Any,
    raw_tokenizer: Any,
    v_safety: Tensor,
    proj_caches: dict[int, dict[str, Any]],
    calibration_prompts_rendered: list[str],
    consumer_heads: list[tuple[int, int]],
    *,
    device: str | torch.device,
    log_every: int = 20,
    target_projections: tuple[str, ...] = ("q", "k", "v"),
    cancel_event: threading.Event | None = None,
    empty_cache_every: int = 20,
) -> dict[tuple[int, int], float]:
    """Compute signed_score(layer, head) averaged over calibration prompts.

    Positive score ⇒ ablating that head reduces refusal probability ⇒
    head is a refusal-promoting consumer.

    Args:
        consumer_heads: every (layer, query_head) we want to score.
            Typically every head across consumer_layers — for Gemma 3
            this is 16 layers × 16 heads = 256 candidates, minus any
            (layer, head) that fall outside the actual shape.
        proj_caches: dict layer → cache, must cover every layer in
            consumer_heads.
        calibration_prompts_rendered: chat-template-rendered prompts
            from the HARMFUL pool. Small (10-30 is enough).
    """
    if not calibration_prompts_rendered:
        raise ValueError("calibration_prompts_rendered is empty")
    if not consumer_heads:
        raise ValueError("consumer_heads is empty")

    refusal_ids = _anchor_token_ids(raw_tokenizer, REFUSAL_ANCHORS)
    comply_ids = _anchor_token_ids(raw_tokenizer, COMPLY_ANCHORS)
    logger.info(
        "anchor token sets: refusal=%d, comply=%d", len(refusal_ids), len(comply_ids),
    )

    # Pre-tokenize all prompts (one forward each for baseline).
    prompt_ids: list[Tensor] = []
    for p in calibration_prompts_rendered:
        ids = raw_tokenizer.encode(p, add_special_tokens=False).ids
        prompt_ids.append(torch.tensor([ids], device=device))

    # Baseline log-odds per prompt — one no-ablation forward each.
    logger.info(
        "measuring baseline log-odds on %d calibration prompts...",
        len(prompt_ids),
    )
    baseline_log_odds: list[float] = []
    for pi in prompt_ids:
        logits = _forward_last_logits(model, pi)
        baseline_log_odds.append(
            _log_odds_refuse_minus_comply(logits, refusal_ids, comply_ids)
        )
    logger.info(
        "baseline log-odds: mean=%.4f, range=[%.4f, %.4f]",
        sum(baseline_log_odds) / len(baseline_log_odds),
        min(baseline_log_odds), max(baseline_log_odds),
    )

    # Per-head scoring.
    scores: dict[tuple[int, int], float] = {}
    for hi, head in enumerate(consumer_heads):
        if cancel_event is not None and cancel_event.is_set():
            logger.warning(
                "signed scoring cancelled at head %d/%d (cancel_event set); "
                "returning partial scores", hi, len(consumer_heads),
            )
            return scores
        if (hi % empty_cache_every) == 0 and hi > 0:
            mps_empty_cache_safe()
        L, q = head
        if L not in proj_caches:
            logger.warning("skip (L=%d, h=%d): no proj cache", L, q)
            continue
        # Install ablation for just this head, measure per-prompt.
        diffs: list[float] = []
        try:
            handles = install_edge_consumer_ablation_hook(
                model, [head], v_safety, proj_caches, alpha=1.0,
                target_projections=target_projections,
            )
        except Exception:
            logger.exception("install hook failed for L%d.h%d", L, q)
            continue
        try:
            for k, pi in enumerate(prompt_ids):
                logits = _forward_last_logits(model, pi)
                lo = _log_odds_refuse_minus_comply(
                    logits, refusal_ids, comply_ids,
                )
                diffs.append(baseline_log_odds[k] - lo)
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
        if diffs:
            scores[(L, q)] = sum(diffs) / len(diffs)
        if (hi + 1) % log_every == 0:
            logger.info(
                "signed_attr: %d/%d heads scored (current: L%d.h%d → %.4f)",
                hi + 1, len(consumer_heads), L, q, scores.get((L, q), 0.0),
            )

    return scores


def enumerate_all_heads(
    model: Any, consumer_layers: list[int],
) -> list[tuple[int, int]]:
    """Return [(layer, query_head_idx)] for every query head across
    `consumer_layers`. Reads n_q from the model's actual attention
    shape so we don't mis-count on configs we haven't seen."""
    heads: list[tuple[int, int]] = []
    for L in consumer_layers:
        attn = _attn_module(model, L)
        n_q, _n_kv, _hd = _attn_shape(attn)
        for q in range(n_q):
            heads.append((L, q))
    return heads


__all__ = [
    "REFUSAL_ANCHORS",
    "COMPLY_ANCHORS",
    "compute_signed_scores",
    "enumerate_all_heads",
]
