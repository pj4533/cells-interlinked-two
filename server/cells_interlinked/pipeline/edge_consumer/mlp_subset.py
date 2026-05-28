"""Per-MLP signed scoring + greedy subset composition.

Companion to mlp_hook.py. Same methodology as
``signed_attribution.compute_signed_scores`` + ``subset_compose``,
but at the MLP-layer granularity instead of attention-head:

  - Score: install ablation hook on JUST one MLP, measure next-token
    log-odds shift toward comply tokens. Positive ⇒ ablating that
    MLP reduces refusal probability ⇒ MLP is a refusal-promoting
    consumer.
  - Compose: greedy add MLPs in signed-score order. Each iteration
    installs hooks on the current MLP subset and measures refusal
    rate. Stop when within ε of global-ablation rate.

15 MLP units across L33–L47, so even a full exhaustive scoring is
cheap (15 × 20 prompts = 300 forwards ≈ 2 minutes).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from ..abliteration import install_runtime_ablation_hook
from .memory_safety import mps_empty_cache_safe
from .mlp_hook import install_mlp_residual_ablation_hook
from .signed_attribution import (
    COMPLY_ANCHORS,
    REFUSAL_ANCHORS,
    _anchor_token_ids,
    _forward_last_logits,
    _log_odds_refuse_minus_comply,
)
from .subset_compose import SubsetTrajectory, measure_refusal_rate

logger = logging.getLogger(__name__)


@dataclass
class MLPSubsetTrajectory:
    """Per-iteration record of MLP greedy composition. Distinct dataclass
    from SubsetTrajectory because the unit is a layer index, not a
    (layer, head) tuple."""
    epsilon: float
    baseline_refusal_rate: float
    global_refusal_rate: float
    iterations: list[dict[str, Any]] = field(default_factory=list)
    sufficient_subset: list[int] = field(default_factory=list)
    stopped_reason: str = "pending"


@torch.no_grad()
def compute_signed_mlp_scores(
    model: Any,
    raw_tokenizer: Any,
    v_safety: Tensor,
    calibration_prompts_rendered: list[str],
    mlp_layers: list[int],
    *,
    device: str | torch.device,
    log_every: int = 2,
    cancel_event: threading.Event | None = None,
    empty_cache_every: int = 4,
) -> dict[int, float]:
    """For each MLP layer, install ablation hook → measure log-odds
    shift on calibration. Returns dict[layer] → signed score.

    Positive score ⇒ ablating that MLP reduces refusal probability ⇒
    MLP is a refusal-promoting consumer.
    """
    if not calibration_prompts_rendered:
        raise ValueError("calibration_prompts_rendered is empty")
    if not mlp_layers:
        raise ValueError("mlp_layers is empty")

    refusal_ids = _anchor_token_ids(raw_tokenizer, REFUSAL_ANCHORS)
    comply_ids = _anchor_token_ids(raw_tokenizer, COMPLY_ANCHORS)
    logger.info(
        "anchor token sets: refusal=%d, comply=%d",
        len(refusal_ids), len(comply_ids),
    )

    prompt_ids = [
        torch.tensor(
            [raw_tokenizer.encode(p, add_special_tokens=False).ids],
            device=device,
        )
        for p in calibration_prompts_rendered
    ]

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

    scores: dict[int, float] = {}
    for li, L in enumerate(mlp_layers):
        if cancel_event is not None and cancel_event.is_set():
            logger.warning(
                "MLP signed scoring cancelled at %d/%d (cancel_event set); "
                "returning partial",
                li, len(mlp_layers),
            )
            return scores
        if li > 0 and (li % empty_cache_every) == 0:
            mps_empty_cache_safe()

        diffs: list[float] = []
        try:
            handles = install_mlp_residual_ablation_hook(
                model, [L], v_safety, alpha=1.0,
            )
        except Exception:
            logger.exception("install MLP hook failed at L%d", L)
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
            scores[L] = sum(diffs) / len(diffs)
        if (li + 1) % log_every == 0:
            logger.info(
                "mlp_signed: %d/%d MLPs scored (current: L%d → %.4f)",
                li + 1, len(mlp_layers), L, scores.get(L, 0.0),
            )

    return scores


def compose_sufficient_mlp_subset(
    model: Any,
    raw_tokenizer: Any,
    *,
    ranked_mlps: list[int],
    v_safety: Tensor,
    holdout_prompts_rendered: list[str],
    extraction_layer: int,
    epsilons: tuple[float, ...] = (0.02, 0.05, 0.10),
    device: str | torch.device,
    eos_ids: tuple[int, ...] = (),
    log_every: int = 1,
    max_mlps: int | None = None,
    cancel_event: threading.Event | None = None,
    empty_cache_every: int = 2,
) -> dict[float, MLPSubsetTrajectory]:
    """Greedy add MLPs in signed-score order, measure refusal rate.

    Same logic as ``compose_sufficient_subset`` but the unit is a
    single MLP layer index, and each iteration installs hooks on the
    current subset of MLPs.
    """
    # Baseline
    logger.info(
        "MLP subset: baseline refusal rate on %d holdout prompts",
        len(holdout_prompts_rendered),
    )
    baseline_rr, _ = measure_refusal_rate(
        model, raw_tokenizer, holdout_prompts_rendered, device,
        eos_ids=eos_ids,
    )
    logger.info("baseline: %.3f", baseline_rr)

    # Global target
    logger.info("MLP subset: measuring global-ablation refusal rate")
    g_handle = install_runtime_ablation_hook(
        model, extraction_layer, v_safety, alpha=1.0,
    )
    try:
        target_rr, _ = measure_refusal_rate(
            model, raw_tokenizer, holdout_prompts_rendered, device,
            eos_ids=eos_ids,
        )
    finally:
        g_handle.remove()
    logger.info("global target: %.3f", target_rr)

    results: dict[float, MLPSubsetTrajectory] = {}
    cap = max_mlps if max_mlps is not None else len(ranked_mlps)
    cancelled = False

    for eps in epsilons:
        if cancel_event is not None and cancel_event.is_set():
            logger.warning(
                "MLP subset: stopping ε sweep before ε=%.3f", eps,
            )
            cancelled = True
            break
        traj = MLPSubsetTrajectory(
            epsilon=eps,
            baseline_refusal_rate=baseline_rr,
            global_refusal_rate=target_rr,
        )
        mlp_subset: list[int] = []
        for i, L in enumerate(ranked_mlps[:cap]):
            if cancel_event is not None and cancel_event.is_set():
                traj.sufficient_subset = list(mlp_subset)
                traj.stopped_reason = "cancelled"
                cancelled = True
                break
            if i > 0 and (i % empty_cache_every) == 0:
                mps_empty_cache_safe()
            mlp_subset.append(L)
            handles = install_mlp_residual_ablation_hook(
                model, mlp_subset, v_safety, alpha=1.0,
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
            traj.iterations.append({
                "iter": i + 1,
                "added_mlp": L,
                "subset_size": len(mlp_subset),
                "refusal_rate": rr,
                "delta_to_global": rr - target_rr,
            })
            if (i + 1) % log_every == 0:
                logger.info(
                    "ε=%.3f mlp_iter=%d size=%d added=L%d rr=%.3f Δ=%+.3f",
                    eps, i + 1, len(mlp_subset), L, rr, rr - target_rr,
                )
            if abs(rr - target_rr) <= eps:
                traj.sufficient_subset = list(mlp_subset)
                traj.stopped_reason = "matched"
                logger.info(
                    "ε=%.3f sufficient MLP subset reached at size=%d "
                    "(rr=%.3f, target=%.3f)",
                    eps, len(mlp_subset), rr, target_rr,
                )
                break
        else:
            if not cancelled:
                traj.sufficient_subset = list(mlp_subset)
                traj.stopped_reason = "exhausted"
                logger.warning(
                    "ε=%.3f exhausted MLP ranked list (size=%d) without "
                    "matching global rr=%.3f within ε",
                    eps, len(mlp_subset), target_rr,
                )
        results[eps] = traj
        if cancelled:
            break

    return results


__all__ = [
    "MLPSubsetTrajectory",
    "compute_signed_mlp_scores",
    "compose_sufficient_mlp_subset",
]
