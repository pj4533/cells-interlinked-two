"""Layer-by-layer residual ablation sweep.

Different question from the edge-consumer line: not 'which heads
consume v_safety' but 'at which layer is refusal committed.'

For each layer L in a swept range, install the existing
``install_runtime_ablation_hook`` (which modifies the residual stream
at L) and measure refusal rate on the holdout. The deepest L at
which residual ablation still drops refusal toward zero is the layer
where the model "commits" to the refusal decision — past that point,
even removing v_safety from the residual can't undo the choice.

Outcomes to look for:

1. **Refusal rate stays low across a wide range of L** — refusal is
   represented in the residual stream over a broad depth, and any of
   those layers is a valid intervention point. Global ablation at L32
   is a member of this family, not a privileged one.

2. **Refusal rate transitions from low to high at some Lc** — there's
   a "commitment depth." Layers <= Lc still represent refusal at the
   residual level (intervention works); layers > Lc have already
   converted the refusal representation into downstream computation
   that's no longer linearly tied to v_safety. Lc is a real
   mechanistic finding.

3. **Refusal rate never drops** — v_safety as defined by
   Macar/Arditi mean-difference isn't actually the refusal direction
   at this layer's representation. Worth re-extracting directions at
   different layers.

Uses each layer's OWN direction (`directions[L]`), not L32's
direction transplanted everywhere. The Macar/Arditi extractor
produces per-layer directions for a reason — they encode the actual
mean-difference at that depth.

Cost: 1 forward pass × 32 tokens per holdout prompt × len(layers).
For 50 prompts × 40 layers ≈ 60-80 minutes on M2 Ultra.
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
from .subset_compose import measure_refusal_rate

logger = logging.getLogger(__name__)


@dataclass
class LayerSweepResult:
    """Per-layer refusal-rate measurement from the sweep."""
    layer: int
    refusal_rate: float
    delta_from_baseline: float
    delta_from_global_at_extraction: float


@dataclass
class LayerSweepReport:
    """Full sweep output: baseline, reference point (L=extraction_layer),
    plus per-layer results sorted by layer index."""
    baseline_refusal_rate: float
    extraction_layer: int
    reference_refusal_rate: float  # rr at extraction_layer
    layers_swept: list[int] = field(default_factory=list)
    per_layer: list[LayerSweepResult] = field(default_factory=list)
    cancelled: bool = False


def run_layer_sweep(
    model: Any,
    raw_tokenizer: Any,
    directions: Tensor,
    holdout_prompts_rendered: list[str],
    *,
    layers: list[int],
    extraction_layer: int,
    alpha: float = 1.0,
    device: str | torch.device,
    eos_ids: tuple[int, ...] = (),
    cancel_event: threading.Event | None = None,
    empty_cache_every: int = 4,
    log_every: int = 2,
) -> LayerSweepReport:
    """Sweep residual-stream ablation across `layers`.

    `directions` is the full per-layer tensor (`[num_layers+1, d_model]`)
    from ``load_directions``. For each L we use ``directions[L]`` as
    the ablation direction at that L — each layer's own
    mean-difference direction, not L32's transplanted.

    `extraction_layer` is the reference (CI 2.5's L32) used to compute
    the reference refusal rate.
    """
    # Baseline (no ablation)
    logger.info(
        "layer sweep: baseline refusal rate on %d holdout prompts",
        len(holdout_prompts_rendered),
    )
    baseline_rr, _ = measure_refusal_rate(
        model, raw_tokenizer, holdout_prompts_rendered, device,
        eos_ids=eos_ids,
    )
    logger.info("baseline refusal rate: %.3f", baseline_rr)

    # Reference: ablation at extraction_layer (the canonical "global"
    # in CI 2.5's vocabulary).
    logger.info("layer sweep: reference ablation at L%d", extraction_layer)
    ref_handle = install_runtime_ablation_hook(
        model, extraction_layer, directions[extraction_layer], alpha=alpha,
    )
    try:
        reference_rr, _ = measure_refusal_rate(
            model, raw_tokenizer, holdout_prompts_rendered, device,
            eos_ids=eos_ids,
        )
    finally:
        ref_handle.remove()
    logger.info(
        "reference (L%d ablation) refusal rate: %.3f",
        extraction_layer, reference_rr,
    )

    report = LayerSweepReport(
        baseline_refusal_rate=baseline_rr,
        extraction_layer=extraction_layer,
        reference_refusal_rate=reference_rr,
        layers_swept=list(layers),
    )

    for i, L in enumerate(layers):
        if cancel_event is not None and cancel_event.is_set():
            logger.warning(
                "layer sweep cancelled at L%d (%d/%d done, cancel_event set)",
                L, i, len(layers),
            )
            report.cancelled = True
            break
        if i > 0 and (i % empty_cache_every) == 0:
            mps_empty_cache_safe()

        try:
            handle = install_runtime_ablation_hook(
                model, L, directions[L], alpha=alpha,
            )
        except Exception:
            logger.exception("install hook failed at L%d; skipping", L)
            continue
        try:
            rr, _ = measure_refusal_rate(
                model, raw_tokenizer, holdout_prompts_rendered, device,
                eos_ids=eos_ids,
            )
        finally:
            try:
                handle.remove()
            except Exception:
                logger.exception("hook removal failed at L%d", L)

        result = LayerSweepResult(
            layer=L,
            refusal_rate=rr,
            delta_from_baseline=rr - baseline_rr,
            delta_from_global_at_extraction=rr - reference_rr,
        )
        report.per_layer.append(result)

        if (i + 1) % log_every == 0:
            logger.info(
                "layer sweep: L%d rr=%.3f (Δbaseline=%+.3f, Δref=%+.3f)  [%d/%d]",
                L, rr, result.delta_from_baseline,
                result.delta_from_global_at_extraction,
                i + 1, len(layers),
            )

    return report


__all__ = ["LayerSweepResult", "LayerSweepReport", "run_layer_sweep"]
