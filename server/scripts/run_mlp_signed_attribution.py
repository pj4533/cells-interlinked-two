"""Per-MLP signed scoring + subset composition.

Tests the hypothesis (from the 2026-05-28 attention edge findings)
that the residual 26% refusal rate left after ablating EVERY
attention head's q/k/v is mediated through MLP layers. If MLPs are
the dominant consumers, a small subset of them (1–5 of the 15
candidate MLPs at L33–L47) should bring refusal close to 0%.

Workflow:
  1. Pre-flight + watchdog (memory safety).
  2. Score each MLP layer L in consumer_layers: install
     install_mlp_residual_ablation_hook on JUST L, measure log-odds
     shift on calibration set. signed_score(L) = baseline log-odds −
     ablated log-odds. Positive ⇒ ablating MLP L reduces refusal.
  3. Rank MLPs by signed score descending.
  4. Greedy subset compose for ε ∈ {0.02, 0.05, 0.10}: add MLPs in
     ranked order, measure refusal rate, stop when within ε of
     global L32 ablation target.

Backend MUST BE STOPPED before running.

    cd server
    uv run python -m scripts.run_mlp_signed_attribution \\
        --direction v3_safety
"""

from __future__ import annotations

import argparse
import dataclasses as _dc
import json
import logging
import sys
import time
from pathlib import Path

import torch

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
from cells_interlinked.pipeline.edge_consumer.mlp_subset import (
    compose_sufficient_mlp_subset,
    compute_signed_mlp_scores,
)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="v3_safety")
    ap.add_argument("--direction-path", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=out_dir())
    ap.add_argument("--first-layer", type=int, default=None,
                    help="First MLP layer (default: extraction_layer + 1).")
    ap.add_argument("--last-layer", type=int, default=None,
                    help="Last MLP layer (default: num_layers - 1).")
    ap.add_argument("--calibration", type=int, default=20)
    ap.add_argument("--seed-calibration", type=int, default=10)
    ap.add_argument("--epsilons", type=float, nargs="+",
                    default=[0.02, 0.05, 0.10])
    ap.add_argument("--holdout", type=int, default=50)
    ap.add_argument("--seed-holdout", type=int, default=1)
    ap.add_argument("--max-mlps", type=int, default=None)
    ap.add_argument("--skip-subset", action="store_true")
    ap.add_argument("--force-signed", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    ap.add_argument("--watchdog-free-floor-gb", type=float, default=2.0)
    ap.add_argument("--watchdog-swap-ceiling-gb", type=float, default=8.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    variant = args.direction

    print("=" * 64)
    print(f"MLP signed attribution — variant={variant}")
    print("=" * 64)

    enforce_pre_flight(min_free_gb=args.min_free_gb)
    watchdog = arm_watchdog(
        free_gb_floor=args.watchdog_free_floor_gb,
        swap_gb_ceiling=args.watchdog_swap_ceiling_gb,
    )
    try:
        return _run(args, variant, watchdog.cancel_event)
    finally:
        watchdog.stop()
        if watchdog.tripped:
            print(f"\nWATCHDOG TRIPPED: {watchdog.trip_reason}")


def _run(args, variant, cancel_event):
    bundle = load_m_bundle()
    directions, dir_meta = load_active_direction(args.direction_path)
    if dir_meta["model_name"] != bundle.model_name:
        raise RuntimeError("direction model mismatch")
    extraction_layer = bundle.extraction_layer
    v_safety = directions[extraction_layer]
    first = args.first_layer if args.first_layer is not None else extraction_layer + 1
    last = args.last_layer if args.last_layer is not None else bundle.num_layers - 1
    mlp_layers = list(range(first, last + 1))
    print(f"extraction L{extraction_layer}; MLP layers L{first}..L{last} ({len(mlp_layers)} MLPs)")

    # ── Scoring ──
    signed_out = args.out_dir / f"signed_mlp_scores_{variant}.pt"
    if signed_out.exists() and not args.force_signed:
        print(f"\n[signed] using existing {signed_out.name}")
        payload = torch.load(signed_out, map_location="cpu", weights_only=True)
        ranked_mlps = [int(L) for L, _s in payload["ranked"]]
    else:
        print(f"\n[signed] scoring {len(mlp_layers)} MLPs")
        calibration_raw = sample_prompts(
            HARMFUL_PROMPTS, args.calibration, args.seed_calibration,
            "HARMFUL (calibration)",
        )
        calibration_rendered = [render_user_only(bundle, p) for p in calibration_raw]
        t0 = time.time()
        scores = compute_signed_mlp_scores(
            bundle.model, bundle.raw_tokenizer,
            v_safety, calibration_rendered, mlp_layers,
            device=settings.device,
            cancel_event=cancel_event,
        )
        elapsed = time.time() - t0
        ranked_with_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        ranked_mlps = [L for L, _s in ranked_with_scores]
        payload = {
            "scores": {f"L{L:02d}": float(s) for L, s in scores.items()},
            "ranked": [(int(L), float(s)) for L, s in ranked_with_scores],
            "consumer_mlp_layers": mlp_layers,
            "extraction_layer": int(extraction_layer),
            "variant": variant,
            "model_name": bundle.model_name,
            "n_calibration": len(calibration_rendered),
            "seed_calibration": args.seed_calibration,
            "scoring_method": "mlp_signed_log_odds_refusal_minus_comply",
        }
        torch.save(payload, signed_out)
        signed_out.with_suffix(".pt.json").write_text(json.dumps({
            "variant": variant,
            "model_name": bundle.model_name,
            "n_mlps": len(scores),
            "n_calibration": len(calibration_rendered),
            "elapsed_seconds": elapsed,
            "scoring_method": payload["scoring_method"],
            "convention": (
                "Positive score = ablating MLP at this layer's residual "
                "contribution reduces refusal probability. ranked is "
                "sorted descending."
            ),
        }, indent=2))
        print(f"  done in {elapsed:.0f}s → {signed_out}")
        print(f"  full MLP ranking (high → low signed score):")
        for L, s in payload["ranked"]:
            print(f"    L{L:02d}  signed={s:+.4f}")

    if args.skip_subset:
        return 0

    # ── Subset compose ──
    print(f"\n[subset] composing MLPs with ε={args.epsilons}")
    holdout_raw = sample_prompts(
        HARMFUL_PROMPTS, args.holdout, args.seed_holdout, "HARMFUL (holdout)",
    )
    holdout_rendered = [render_user_only(bundle, p) for p in holdout_raw]

    t0 = time.time()
    trajectories = compose_sufficient_mlp_subset(
        bundle.model, bundle.raw_tokenizer,
        ranked_mlps=ranked_mlps,
        v_safety=v_safety,
        holdout_prompts_rendered=holdout_rendered,
        extraction_layer=extraction_layer,
        epsilons=tuple(args.epsilons),
        device=settings.device,
        eos_ids=bundle.eos_ids,
        max_mlps=args.max_mlps,
        cancel_event=cancel_event,
    )
    elapsed = time.time() - t0

    for eps, traj in trajectories.items():
        out_path = args.out_dir / (
            f"sufficient_subset_signed_mlp_eps={eps:.2f}_{variant}.json"
        )
        payload = {
            **_dc.asdict(traj),
            "variant": variant,
            "model_name": bundle.model_name,
            "extraction_layer": int(extraction_layer),
            "holdout_seed": int(args.seed_holdout),
            "n_holdout": len(holdout_rendered),
            "elapsed_seconds_total": elapsed,
            "ranking_source": "signed_mlp_attribution",
        }
        payload["sufficient_subset"] = [int(L) for L in payload["sufficient_subset"]]
        out_path.write_text(json.dumps(payload, indent=2))
        print(
            f"  ε={eps:.2f}  size={len(payload['sufficient_subset'])}  "
            f"reason={payload['stopped_reason']}  → {out_path.name}"
        )

    print(f"\nMLP signed attribution complete in {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
