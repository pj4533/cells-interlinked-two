"""Layer-by-layer residual ablation sweep.

For each layer L in a swept range, install the global residual
ablation at that L (using directions[L]) and measure refusal rate on
the holdout. Tells us where in the stack the refusal decision is
"committed" — past Lc, removing v_safety from the residual can't
reverse the decision.

See ``pipeline/edge_consumer/layer_sweep.py`` for the methodology.

Backend MUST BE STOPPED before running.

    cd server
    uv run python -m scripts.run_layer_sweep \\
        --direction v3_safety \\
        --first-layer 15 --last-layer 47
"""

from __future__ import annotations

import argparse
import dataclasses as _dc
import json
import logging
import sys
import time
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
from cells_interlinked.pipeline.edge_consumer.layer_sweep import (
    run_layer_sweep,
)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="Layer-by-layer residual ablation sweep")
    ap.add_argument("--direction", default="v3_safety")
    ap.add_argument("--direction-path", type=Path, default=None)
    ap.add_argument("--first-layer", type=int, default=15,
                    help="First layer to sweep (inclusive).")
    ap.add_argument("--last-layer", type=int, default=None,
                    help="Last layer to sweep (inclusive). "
                         "Default: num_layers - 1.")
    ap.add_argument("--step", type=int, default=1,
                    help="Stride across layers. step=2 halves cost.")
    ap.add_argument("--holdout", type=int, default=50)
    ap.add_argument("--seed-holdout", type=int, default=1,
                    help="Match other scripts' holdout seed for "
                         "comparability with edge-consumer results.")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out-dir", type=Path, default=out_dir())
    ap.add_argument("--force", action="store_true")
    # Memory safety
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    ap.add_argument("--watchdog-free-floor-gb", type=float, default=2.0)
    ap.add_argument("--watchdog-swap-ceiling-gb", type=float, default=8.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"layer_sweep_{args.direction}.json"
    if out_path.exists() and not args.force:
        print(f"already exists: {out_path}  (pass --force to overwrite)")
        return 0

    print("=" * 64)
    print(f"layer sweep — direction={args.direction}")
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
            raise RuntimeError(
                f"direction model_name={dir_meta['model_name']} != "
                f"bundle model_name={bundle.model_name}"
            )
        last_layer = args.last_layer if args.last_layer is not None else bundle.num_layers - 1
        layers = list(range(args.first_layer, last_layer + 1, args.step))
        print(f"sweeping {len(layers)} layers: L{layers[0]}..L{layers[-1]} step={args.step}")
        print(f"reference layer (CI 2.5 extraction): L{bundle.extraction_layer}")

        holdout_raw = sample_prompts(
            HARMFUL_PROMPTS, args.holdout, args.seed_holdout,
            "HARMFUL (holdout)",
        )
        holdout_rendered = [render_user_only(bundle, p) for p in holdout_raw]
        print(f"rendered {len(holdout_rendered)} holdout prompts")

        t0 = time.time()
        report = run_layer_sweep(
            bundle.model, bundle.raw_tokenizer,
            directions=directions,
            holdout_prompts_rendered=holdout_rendered,
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
        "per_layer": [_dc.asdict(r) for r in report.per_layer],
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
    print(
        f"  baseline (no ablation):       {report.baseline_refusal_rate * 100:5.1f}%"
    )
    print(
        f"  reference (L{report.extraction_layer} ablation):      "
        f"{report.reference_refusal_rate * 100:5.1f}%"
    )
    print()
    print(f"  per-layer refusal rate (rr) after ablating that layer's residual:")
    print(f"     L   rr     Δref")
    print(f"   ---  ----   -----")
    for r in report.per_layer:
        marker = " ◀ reference" if r.layer == report.extraction_layer else ""
        print(
            f"   L{r.layer:02d}  {r.refusal_rate * 100:4.1f}%  "
            f"{r.delta_from_global_at_extraction * 100:+5.1f}{marker}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
