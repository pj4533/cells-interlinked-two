"""Experiment 3 — per-component (attention block | MLP) ablation sweep.

After Experiment 2 identifies Lstart (the earliest layer at which
residual ablation works → the layer where v_safety becomes coherent
in the residual stream), this script localizes WHICH component
actually wrote v_safety into the residual.

For each layer L in [first_layer, last_layer]:
  - Ablate just the attention block's output at L (one
    install_attn_block_residual_ablation_hook); measure refusal rate
    on the 50-prompt holdout.
  - Ablate just the MLP output at L (one
    install_mlp_residual_ablation_hook); measure refusal rate.

Output: a 2 × (last - first + 1) table of refusal rates per
(layer, component). The "interesting" cell is the one with the
biggest drop from baseline (= the component that, when its output
is stripped of v_safety, prevents v_safety from being coherent at
Lstart and beyond).

Backend MUST BE STOPPED before running.

Typical invocation (after Experiment 2 finds Lstart = 5, say):

    cd server
    uv run python -u -m scripts.run_component_sweep \\
        --direction v3_safety \\
        --first-layer 0 --last-layer 5

Cost: each measurement = 50 prompts × 32-token greedy decode ≈ 100s.
For Lstart=5 → 12 measurements (5 attn + 5 mlp + baseline + ref) =
~20 min. For Lstart=14 → ~50 min.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _edge_consumer_common import (
    HARMFUL_PROMPTS,
    arm_watchdog,
    enforce_pre_flight,
    load_active_direction,
    load_m_bundle,
    out_dir,
    render_user_only,
    sample_prompts,
    settings,
)
from cells_interlinked.pipeline.abliteration import install_runtime_ablation_hook
from cells_interlinked.pipeline.edge_consumer.attn_block_hook import (
    install_attn_block_residual_ablation_hook,
)
from cells_interlinked.pipeline.edge_consumer.memory_safety import mps_empty_cache_safe
from cells_interlinked.pipeline.edge_consumer.mlp_hook import (
    install_mlp_residual_ablation_hook,
)
from cells_interlinked.pipeline.edge_consumer.subset_compose import measure_refusal_rate


@dataclass
class ComponentMeasurement:
    layer: int
    component: str  # "attn" | "mlp"
    refusal_rate: float
    delta_from_baseline: float
    delta_from_reference: float


@dataclass
class ComponentSweepReport:
    baseline_refusal_rate: float
    extraction_layer: int
    reference_refusal_rate: float  # rr at extraction_layer ablation
    layers_swept: list[int] = field(default_factory=list)
    per_measurement: list[ComponentMeasurement] = field(default_factory=list)
    cancelled: bool = False


def _measure_with_hook_install(
    install_fn,
    install_args: tuple,
    install_kwargs: dict,
    *,
    model,
    raw_tokenizer,
    holdout_rendered,
    device,
    eos_ids,
) -> float:
    """Helper: install hook via callable, measure refusal rate, remove.

    install_fn returns either a single handle (legacy single-hook
    pattern from install_runtime_ablation_hook) OR a list of handles
    (the mlp/attn-block hooks). Both are handled.
    """
    raw = install_fn(*install_args, **install_kwargs)
    if isinstance(raw, list):
        handles = raw
    else:
        handles = [raw]
    try:
        rr, _ = measure_refusal_rate(
            model, raw_tokenizer, holdout_rendered, device, eos_ids=eos_ids,
        )
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
    return rr


def run_component_sweep(
    model,
    raw_tokenizer,
    directions,
    holdout_rendered: list[str],
    *,
    layers: list[int],
    extraction_layer: int,
    alpha: float = 1.0,
    device,
    eos_ids: tuple[int, ...] = (),
    cancel_event: threading.Event | None = None,
    empty_cache_every: int = 4,
) -> ComponentSweepReport:
    """For each layer in `layers`, measure refusal rate under
    attention-block ablation and under MLP-output ablation.

    Uses each layer's OWN direction (`directions[L]`) — same
    convention as the residual layer sweep. The hypothesis is that
    if MLP at L wrote v_safety into the residual, then stripping it
    from MLP's output (its contribution to the residual) before it's
    added will undo refusal at the read-out point.
    """
    logger = logging.getLogger(__name__)

    # Baseline + reference (no ablation, and global at extraction_layer).
    logger.info(
        "component sweep: baseline on %d holdout prompts",
        len(holdout_rendered),
    )
    baseline_rr, _ = measure_refusal_rate(
        model, raw_tokenizer, holdout_rendered, device, eos_ids=eos_ids,
    )
    logger.info("baseline refusal rate: %.3f", baseline_rr)

    logger.info("component sweep: reference (L%d residual ablation)", extraction_layer)
    handle = install_runtime_ablation_hook(
        model, extraction_layer, directions[extraction_layer], alpha=alpha,
    )
    try:
        reference_rr, _ = measure_refusal_rate(
            model, raw_tokenizer, holdout_rendered, device, eos_ids=eos_ids,
        )
    finally:
        handle.remove()
    logger.info(
        "reference (L%d ablation) refusal rate: %.3f",
        extraction_layer, reference_rr,
    )

    report = ComponentSweepReport(
        baseline_refusal_rate=baseline_rr,
        extraction_layer=extraction_layer,
        reference_refusal_rate=reference_rr,
        layers_swept=list(layers),
    )

    # Walk layers; at each, measure attn-block then MLP.
    iter_idx = 0
    for L in layers:
        v_L = directions[L]
        for component in ("attn", "mlp"):
            if cancel_event is not None and cancel_event.is_set():
                logger.warning(
                    "component sweep cancelled at L%d.%s (cancel_event set)",
                    L, component,
                )
                report.cancelled = True
                return report
            if iter_idx > 0 and (iter_idx % empty_cache_every) == 0:
                mps_empty_cache_safe()

            try:
                if component == "attn":
                    rr = _measure_with_hook_install(
                        install_attn_block_residual_ablation_hook,
                        (model, [L], v_L), {"alpha": alpha},
                        model=model,
                        raw_tokenizer=raw_tokenizer,
                        holdout_rendered=holdout_rendered,
                        device=device,
                        eos_ids=eos_ids,
                    )
                else:
                    rr = _measure_with_hook_install(
                        install_mlp_residual_ablation_hook,
                        (model, [L], v_L), {"alpha": alpha},
                        model=model,
                        raw_tokenizer=raw_tokenizer,
                        holdout_rendered=holdout_rendered,
                        device=device,
                        eos_ids=eos_ids,
                    )
            except Exception:
                logger.exception("hook install/measure failed at L%d.%s", L, component)
                continue

            m = ComponentMeasurement(
                layer=L,
                component=component,
                refusal_rate=rr,
                delta_from_baseline=rr - baseline_rr,
                delta_from_reference=rr - reference_rr,
            )
            report.per_measurement.append(m)
            logger.info(
                "L%02d.%s  rr=%.3f  Δbaseline=%+.3f  Δref=%+.3f",
                L, component, rr, m.delta_from_baseline, m.delta_from_reference,
            )
            iter_idx += 1

    return report


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="Per-component ablation sweep")
    ap.add_argument("--direction", default="v3_safety")
    ap.add_argument("--direction-path", type=Path, default=None)
    ap.add_argument("--first-layer", type=int, default=0)
    ap.add_argument("--last-layer", type=int, required=True,
                    help="Inclusive upper bound. Typically Lstart from Exp 2.")
    ap.add_argument("--holdout", type=int, default=50)
    ap.add_argument("--seed-holdout", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out-dir", type=Path, default=out_dir())
    ap.add_argument("--force", action="store_true")
    # Memory safety
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--watchdog-free-floor-gb", type=float, default=0.01)
    ap.add_argument("--watchdog-swap-ceiling-gb", type=float, default=8.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"component_sweep_{args.direction}.json"
    if out_path.exists() and not args.force:
        print(f"already exists: {out_path}  (pass --force to overwrite)")
        return 0

    print("=" * 64)
    print(f"component sweep — direction={args.direction}")
    print(f"layers L{args.first_layer}..L{args.last_layer}")
    print("=" * 64)

    enforce_pre_flight(min_free_gb=args.min_free_gb)
    watchdog = arm_watchdog(
        free_gb_floor=args.watchdog_free_floor_gb,
        swap_gb_ceiling=args.watchdog_swap_ceiling_gb,
    )

    try:
        bundle = load_m_bundle()
        directions, dir_meta = load_active_direction(args.direction_path)
        if dir_meta["model_name"] != bundle.model_name:
            raise RuntimeError("direction model mismatch")
        layers = list(range(args.first_layer, args.last_layer + 1))
        print(f"sweeping {len(layers)} layers × 2 components "
              f"({2 * len(layers)} total measurements)")

        holdout_raw = sample_prompts(
            HARMFUL_PROMPTS, args.holdout, args.seed_holdout,
            "HARMFUL (holdout)",
        )
        holdout_rendered = [render_user_only(bundle, p) for p in holdout_raw]
        print(f"rendered {len(holdout_rendered)} holdout prompts")

        t0 = time.time()
        report = run_component_sweep(
            bundle.model, bundle.raw_tokenizer,
            directions=directions,
            holdout_rendered=holdout_rendered,
            layers=layers,
            extraction_layer=bundle.extraction_layer,
            alpha=args.alpha,
            device=settings.device,
            eos_ids=bundle.eos_ids,
            cancel_event=watchdog.cancel_event,
        )
        elapsed = time.time() - t0
        print(f"\nsweep complete in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    finally:
        watchdog.stop()
        if watchdog.tripped:
            print(f"\nWATCHDOG TRIPPED: {watchdog.trip_reason}")

    payload = {
        "baseline_refusal_rate": report.baseline_refusal_rate,
        "extraction_layer": report.extraction_layer,
        "reference_refusal_rate": report.reference_refusal_rate,
        "layers_swept": report.layers_swept,
        "cancelled": report.cancelled,
        "per_measurement": [asdict(m) for m in report.per_measurement],
        "variant": args.direction,
        "model_name": bundle.model_name,
        "alpha": args.alpha,
        "holdout_seed": args.seed_holdout,
        "n_holdout": len(holdout_rendered),
        "elapsed_seconds": elapsed,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")

    # Print summary table.
    print()
    print(f"  baseline (no ablation):       {report.baseline_refusal_rate * 100:5.1f}%")
    print(f"  reference (L{report.extraction_layer} ablation):       "
          f"{report.reference_refusal_rate * 100:5.1f}%")
    print()
    print(f"  per-component refusal rate (Δref = pp from reference):")
    print(f"     L   attn rr    Δref      mlp rr     Δref")
    print(f"   ---  --------   ------   --------    ------")
    by_layer: dict[int, dict[str, ComponentMeasurement]] = {}
    for m in report.per_measurement:
        by_layer.setdefault(m.layer, {})[m.component] = m
    for L in sorted(by_layer):
        a = by_layer[L].get("attn")
        mlp = by_layer[L].get("mlp")
        a_str = f"{a.refusal_rate * 100:5.1f}%" if a else "  —  "
        a_d = f"{a.delta_from_reference * 100:+5.1f}" if a else "  —  "
        m_str = f"{mlp.refusal_rate * 100:5.1f}%" if mlp else "  —  "
        m_d = f"{mlp.delta_from_reference * 100:+5.1f}" if mlp else "  —  "
        print(f"   L{L:02d}   {a_str}    {a_d}    {m_str}     {m_d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
