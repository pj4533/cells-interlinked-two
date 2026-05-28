"""Step 3: greedy sufficient-consumer-subset composer (ε-swept).

Reads the ranked head list from Step 1, builds projection caches for
every consumer layer, then for each ε in {0.02, 0.05, 0.10} greedily
adds heads in AP order and measures refusal rate on a holdout set
until rate matches global-ablation rate within ε.

Run with the backend STOPPED.

    cd server
    uv run python -m scripts.compose_edge_consumer_subset \\
        --scores data/edge_consumer/attribution_scores_v3_safety.pt \\
        --epsilons 0.02 0.05 0.10 --holdout 50

Output: one file per ε:
    server/data/edge_consumer/sufficient_subset_eps={ε}_{variant}.json

Each file records the sufficient subset + the full iteration
trajectory (refusal rate vs subset size).
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

# Allow `python -m scripts.compose_edge_consumer_subset` invocation.
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
)
from cells_interlinked.pipeline.edge_consumer.subset_compose import (
    compose_sufficient_subset,
)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="Step 3 — sufficient subset")
    ap.add_argument("--scores", type=Path, required=True,
                    help="attribution_scores_{variant}.pt from Step 1.")
    ap.add_argument("--direction-path", type=Path, default=None)
    ap.add_argument("--epsilons", type=float, nargs="+",
                    default=[0.02, 0.05, 0.10])
    ap.add_argument("--holdout", type=int, default=50,
                    help="Holdout harmful prompts (disjoint from Step 1).")
    ap.add_argument("--seed", type=int, default=1,
                    help="Different seed than Step 1 to ensure disjoint sample.")
    ap.add_argument("--out-dir", type=Path, default=out_dir())
    ap.add_argument("--max-heads", type=int, default=None,
                    help="Safety cap on subset size if ε can't be matched.")
    ap.add_argument("--force-proj-cache", action="store_true",
                    help="Rebuild W·v caches even if files exist.")
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    ap.add_argument("--watchdog-free-floor-gb", type=float, default=2.0)
    ap.add_argument("--watchdog-swap-ceiling-gb", type=float, default=8.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    enforce_pre_flight(min_free_gb=args.min_free_gb)
    watchdog = arm_watchdog(
        free_gb_floor=args.watchdog_free_floor_gb,
        swap_gb_ceiling=args.watchdog_swap_ceiling_gb,
    )

    # Load scores.
    print(f"loading attribution scores from {args.scores}")
    score_payload = torch.load(args.scores, map_location="cpu", weights_only=True)
    variant = score_payload["variant"]
    ranked_full = [(int(L), int(h)) for (L, h, _s) in score_payload["ranked"]]
    consumer_layers = sorted({L for (L, _h) in ranked_full})
    print(f"  variant={variant}  heads={len(ranked_full)}  "
          f"layers=L{min(consumer_layers)}..L{max(consumer_layers)}")

    # Sample holdout prompts.
    holdout_raw = sample_prompts(
        HARMFUL_PROMPTS, args.holdout, args.seed, "HARMFUL (holdout)",
    )

    # Load M.
    bundle = load_m_bundle()
    directions, dir_meta = load_active_direction(args.direction_path)
    if dir_meta["model_name"] != bundle.model_name:
        raise RuntimeError("direction model mismatch")
    extraction_layer = bundle.extraction_layer
    v_safety = directions[extraction_layer]

    # Build / reuse projection caches.
    proj_cache_dir = args.out_dir / "proj_caches" / variant
    needed = [L for L in consumer_layers if (
        args.force_proj_cache or not (proj_cache_dir / f"layer_{L:02d}.pt").exists()
    )]
    if needed:
        print(f"building projection caches for {len(needed)} layer(s)...")
        t0 = time.time()
        build_and_save_all(
            bundle.model, v_safety, proj_cache_dir,
            layers=needed,
            variant=variant,
            model_name=bundle.model_name,
        )
        print(f"  done in {time.time() - t0:.1f}s → {proj_cache_dir}/")
    else:
        print(f"projection caches already present at {proj_cache_dir}/")

    # Render holdout.
    print(f"rendering {len(holdout_raw)} holdout prompts...")
    holdout_rendered = [render_user_only(bundle, p) for p in holdout_raw]

    # Compose subsets.
    t0 = time.time()
    try:
        trajectories = compose_sufficient_subset(
            bundle.model, bundle.raw_tokenizer,
            ranked_heads=ranked_full,
            v_safety=v_safety,
            proj_cache_dir=proj_cache_dir,
            holdout_prompts_rendered=holdout_rendered,
            extraction_layer=extraction_layer,
            epsilons=tuple(args.epsilons),
            device=settings.device,
            eos_ids=bundle.eos_ids,
            max_heads=args.max_heads,
            cancel_event=watchdog.cancel_event,
        )
    finally:
        watchdog.stop()
        if watchdog.tripped:
            print(f"\nWATCHDOG TRIPPED: {watchdog.trip_reason}")
    elapsed = time.time() - t0
    print(f"all ε swept in {elapsed:.1f}s")

    # Persist one file per ε.
    for eps, traj in trajectories.items():
        out_path = args.out_dir / (
            f"sufficient_subset_eps={eps:.2f}_{variant}.json"
        )
        payload = {
            **_dc.asdict(traj),
            "variant": variant,
            "model_name": bundle.model_name,
            "extraction_layer": int(extraction_layer),
            "holdout_seed": int(args.seed),
            "n_holdout": len(holdout_rendered),
            "elapsed_seconds_total": elapsed,
        }
        # Tuples in iterations + sufficient_subset need explicit
        # list conversion for JSON.
        payload["sufficient_subset"] = [
            [int(L), int(h)] for (L, h) in payload["sufficient_subset"]
        ]
        for it in payload["iterations"]:
            it["added_head"] = [int(it["added_head"][0]), int(it["added_head"][1])]
        out_path.write_text(json.dumps(payload, indent=2))
        size = len(payload["sufficient_subset"])
        print(
            f"  ε={eps:.2f}  size={size}  "
            f"reason={payload['stopped_reason']}  "
            f"→ {out_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
