"""Pivot script: replace AP-magnitude ranking with signed per-head
refusal-impact ranking, then run subset compose end-to-end.

Written tonight (2026-05-27) after the first Phase B overnight run
showed AP-ranked heads pushing refusal rate UP (40% → 48%) instead of
down — the magnitude-based AP score picks up v_safety-correlated
generation/register heads at L44-45, not refusal-decision heads. See
docs/EDGE_CONSUMER_ABLATION.md for the diagnosis.

This script:
  1. Loads M.
  2. Loads (or builds) projection caches for layers 33–47.
  3. Computes signed_score per head: P(refusal|baseline) − P(refusal|head ablated)
     using a small ~20-prompt calibration set.
  4. Ranks by signed score descending (positive = ablation reduces refusal).
  5. Saves the new ranking as `signed_attribution_scores_v3_safety.pt`
     alongside the original AP file (kept for comparison).
  6. Runs the subset composer with the new ranking, ε ∈ {0.02, 0.05, 0.10}.

Total wall-clock: 20-30 min signed scoring + 1-6 hr subset compose.

Backend MUST BE STOPPED before running.

    cd server
    uv run python -m scripts.run_signed_attribution_and_subset
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
from cells_interlinked.pipeline.edge_consumer.proj_cache import (
    build_and_save_all,
    load_projection_cache,
)
from cells_interlinked.pipeline.edge_consumer.signed_attribution import (
    compute_signed_scores,
    enumerate_all_heads,
)
from cells_interlinked.pipeline.edge_consumer.subset_compose import (
    compose_sufficient_subset,
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
    ap.add_argument("--calibration", type=int, default=20,
                    help="Prompts used to score each head (small — 20 is enough).")
    ap.add_argument("--seed-calibration", type=int, default=10,
                    help="Disjoint from contrast/holdout/diagnostic seeds (0,1,2).")
    # Subset compose knobs
    ap.add_argument("--epsilons", type=float, nargs="+",
                    default=[0.02, 0.05, 0.10])
    ap.add_argument("--holdout", type=int, default=50)
    ap.add_argument("--seed-holdout", type=int, default=1)
    ap.add_argument("--max-heads", type=int, default=None)
    ap.add_argument("--skip-subset", action="store_true",
                    help="Stop after signed scoring (debug).")
    ap.add_argument("--force-signed", action="store_true",
                    help="Recompute signed scores even if artifact exists.")
    ap.add_argument(
        "--projections", nargs="+", default=["q", "k", "v"],
        choices=["q", "k", "v"],
        help="Which projections the hook ablates. Use --projections q "
             "to isolate per-query-head effects from grouped-query KV "
             "sharing (recommended for Gemma 3's GQA).",
    )
    # Memory safety (see docs/MEMORY_PRESSURE_LESSONS.md)
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    ap.add_argument("--watchdog-free-floor-gb", type=float, default=2.0)
    ap.add_argument("--watchdog-swap-ceiling-gb", type=float, default=8.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_projections = tuple(args.projections)
    proj_suffix = "".join(target_projections)  # "qkv", "q", etc.
    print(f"hook target_projections = {target_projections} (suffix={proj_suffix})")

    print("=" * 64)
    print("signed-attribution pivot run — variant=", args.direction)
    print("=" * 64)

    enforce_pre_flight(min_free_gb=args.min_free_gb)
    watchdog = arm_watchdog(
        free_gb_floor=args.watchdog_free_floor_gb,
        swap_gb_ceiling=args.watchdog_swap_ceiling_gb,
    )
    try:
        return _run(args, target_projections, proj_suffix, watchdog.cancel_event)
    finally:
        watchdog.stop()
        if watchdog.tripped:
            print(f"\nWATCHDOG TRIPPED: {watchdog.trip_reason}")


def _run(args, target_projections, proj_suffix, cancel_event):
    bundle = load_m_bundle()
    directions, dir_meta = load_active_direction(args.direction_path)
    if dir_meta["model_name"] != bundle.model_name:
        raise RuntimeError("direction model mismatch")
    extraction_layer = bundle.extraction_layer
    v_safety = directions[extraction_layer]
    consumer_layers = list(range(extraction_layer + 1, bundle.num_layers))
    print(f"extraction layer L{extraction_layer}, "
          f"consumer layers L{consumer_layers[0]}..L{consumer_layers[-1]}")

    # Projection caches (reused from earlier run if they exist).
    proj_cache_dir = args.out_dir / "proj_caches" / args.direction
    needed = [L for L in consumer_layers
              if not (proj_cache_dir / f"layer_{L:02d}.pt").exists()]
    if needed:
        print(f"[pre] building {len(needed)} projection caches...")
        t0 = time.time()
        build_and_save_all(
            bundle.model, v_safety, proj_cache_dir,
            layers=needed, variant=args.direction, model_name=bundle.model_name,
        )
        print(f"   done in {time.time() - t0:.1f}s")
    else:
        print(f"[pre] all projection caches present")

    proj_caches = {L: load_projection_cache(proj_cache_dir, L) for L in consumer_layers}

    # ── Signed scoring ────────────────────────────────────────────
    # Suffix the artifact name with the projection mask so q-only and
    # full qkv runs produce distinct files (they're different scorings).
    signed_out = args.out_dir / (
        f"signed_attribution_scores_{args.direction}_{proj_suffix}.pt"
    )
    if signed_out.exists() and not args.force_signed:
        print(f"\n[signed] using existing {signed_out.name}")
        payload = torch.load(signed_out, map_location="cpu", weights_only=True)
        ranked_signed = [(int(L), int(h)) for (L, h, _s) in payload["ranked"]]
    else:
        print(f"\n[signed] scoring {sum(c['n_q_heads'] for c in proj_caches.values())} heads")
        calibration_raw = sample_prompts(
            HARMFUL_PROMPTS, args.calibration, args.seed_calibration,
            "HARMFUL (signed calibration)",
        )
        calibration_rendered = [render_user_only(bundle, p) for p in calibration_raw]
        all_heads = enumerate_all_heads(bundle.model, consumer_layers)
        t0 = time.time()
        scores = compute_signed_scores(
            bundle.model, bundle.raw_tokenizer,
            v_safety, proj_caches, calibration_rendered, all_heads,
            device=settings.device,
            target_projections=target_projections,
            cancel_event=cancel_event,
        )
        elapsed = time.time() - t0
        ranked_with_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        ranked_signed = [k for k, _s in ranked_with_scores]

        payload = {
            "scores": {f"L{L:02d}_h{h:02d}": float(s)
                       for (L, h), s in scores.items()},
            "ranked": [(int(L), int(h), float(s))
                       for (L, h), s in ranked_with_scores],
            "consumer_layers": consumer_layers,
            "extraction_layer": int(extraction_layer),
            "variant": args.direction,
            "model_name": bundle.model_name,
            "n_calibration": len(calibration_rendered),
            "seed_calibration": args.seed_calibration,
            "scoring_method": "signed_log_odds_refusal_minus_comply",
        }
        torch.save(payload, signed_out)
        signed_out.with_suffix(".pt.json").write_text(json.dumps({
            "variant": args.direction,
            "model_name": bundle.model_name,
            "n_heads_scored": len(scores),
            "n_calibration": len(calibration_rendered),
            "elapsed_seconds": elapsed,
            "scoring_method": payload["scoring_method"],
            "convention": (
                "Positive score = ablating this head reduces refusal "
                "probability (the consumers we want). Negative score = "
                "ablating this head increases refusal. ranked is sorted "
                "descending."
            ),
        }, indent=2))
        print(f"   done in {elapsed:.0f}s → {signed_out}")
        print(f"   top 10 signed-positive heads:")
        for (L, h, s) in payload["ranked"][:10]:
            print(f"     L{L:02d}.h{h:02d}  signed={s:+.4f}")
        print(f"   bottom 5 (avoid — ablating these INCREASES refusal):")
        for (L, h, s) in payload["ranked"][-5:]:
            print(f"     L{L:02d}.h{h:02d}  signed={s:+.4f}")

    if args.skip_subset:
        print("\n[signed] --skip-subset: stopping here.")
        return 0

    # ── Subset compose with new ranking ───────────────────────────
    print(f"\n[subset] composing with signed ranking, ε={args.epsilons}")
    holdout_raw = sample_prompts(
        HARMFUL_PROMPTS, args.holdout, args.seed_holdout, "HARMFUL (holdout)",
    )
    holdout_rendered = [render_user_only(bundle, p) for p in holdout_raw]

    t0 = time.time()
    trajectories = compose_sufficient_subset(
        bundle.model, bundle.raw_tokenizer,
        ranked_heads=ranked_signed,
        v_safety=v_safety,
        proj_cache_dir=proj_cache_dir,
        holdout_prompts_rendered=holdout_rendered,
        extraction_layer=extraction_layer,
        epsilons=tuple(args.epsilons),
        device=settings.device,
        eos_ids=bundle.eos_ids,
        max_heads=args.max_heads,
        target_projections=target_projections,
        cancel_event=cancel_event,
    )
    elapsed = time.time() - t0

    # Write per-ε subset files with a `signed_` prefix so they don't
    # collide with the (failed) AP-ranked subsets.
    for eps, traj in trajectories.items():
        out_path = args.out_dir / (
            f"sufficient_subset_signed_{proj_suffix}_eps={eps:.2f}_{args.direction}.json"
        )
        payload = {
            **_dc.asdict(traj),
            "variant": args.direction,
            "model_name": bundle.model_name,
            "extraction_layer": int(extraction_layer),
            "holdout_seed": int(args.seed_holdout),
            "n_holdout": len(holdout_rendered),
            "elapsed_seconds_total": elapsed,
            "ranking_source": "signed_attribution",
        }
        payload["sufficient_subset"] = [
            [int(L), int(h)] for (L, h) in payload["sufficient_subset"]
        ]
        for it in payload["iterations"]:
            it["added_head"] = [int(it["added_head"][0]), int(it["added_head"][1])]
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"  ε={eps:.2f}  size={len(payload['sufficient_subset'])}  "
              f"reason={payload['stopped_reason']}  → {out_path.name}")

    print(f"\nsigned-attribution pivot complete in {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
