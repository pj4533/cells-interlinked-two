"""Step 1: attribution-patching scores per (layer, head).

Loads M, samples N harmful + N harmless contrast prompts, runs 3 forward
passes per pair (unpatched harmful → captures projections; harmless →
captures L32 residual only; patched harmful → captures projections again),
computes per-(layer, query_head) AP scores averaged across pairs, and
writes them to disk.

Run with the backend STOPPED so this process has exclusive access to M.

    cd server
    uv run python -m scripts.run_edge_consumer_attribution \\
        --direction v3_safety --contrasts 200

Output:
    server/data/edge_consumer/attribution_scores_{variant}.pt
        torch.save({
            "scores": dict[(layer, head)] → float,
            "ranked": list[(layer, head)] sorted by AP descending,
            ...,
        })
    server/data/edge_consumer/attribution_scores_{variant}.pt.json
        provenance + parameters
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

# Allow `python -m scripts.run_edge_consumer_attribution` invocation.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _edge_consumer_common import (
    HARMFUL_PROMPTS,
    HARMLESS_PROMPTS,
    arm_watchdog,
    enforce_pre_flight,
    load_active_direction,
    load_m_bundle,
    out_dir,
    render_user_only,
    sample_prompts,
    settings,
)
from cells_interlinked.pipeline.edge_consumer.attribution import (
    compute_attribution_scores,
    sample_contrast_pairs,
)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="Step 1 — attribution scores")
    ap.add_argument("--direction", default="v3_safety",
                    help="Variant name (just a label; the path comes from --direction-path)")
    ap.add_argument("--direction-path", type=Path, default=None,
                    help="Default: server/data/refusal_directions.pt")
    ap.add_argument("--contrasts", type=int, default=200,
                    help="Number of (harmful, harmless) contrast pairs.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=out_dir(),
                    help="Output directory (default server/data/edge_consumer/).")
    ap.add_argument("--first-layer", type=int, default=None,
                    help="First consumer layer (inclusive). "
                         "Default: extraction_layer + 1.")
    ap.add_argument("--last-layer", type=int, default=None,
                    help="Last consumer layer (inclusive). "
                         "Default: num_layers - 1.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing attribution_scores_{variant}.pt.")
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    ap.add_argument("--watchdog-free-floor-gb", type=float, default=2.0)
    ap.add_argument("--watchdog-swap-ceiling-gb", type=float, default=8.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = args.out_dir / f"attribution_scores_{args.direction}.pt"
    if out_pt.exists() and not args.force:
        print(f"already exists: {out_pt}  (pass --force to overwrite)")
        return 0

    enforce_pre_flight(min_free_gb=args.min_free_gb)
    watchdog = arm_watchdog(
        free_gb_floor=args.watchdog_free_floor_gb,
        swap_gb_ceiling=args.watchdog_swap_ceiling_gb,
    )

    # Sample contrast pairs from Arditi's lists.
    pairs = sample_contrast_pairs(
        HARMFUL_PROMPTS, HARMLESS_PROMPTS, args.contrasts, seed=args.seed,
    )
    print(f"sampled {len(pairs)} contrast pairs (seed={args.seed})")

    # Load M and direction.
    bundle = load_m_bundle()
    directions, dir_meta = load_active_direction(args.direction_path)
    if dir_meta["model_name"] != bundle.model_name:
        raise RuntimeError(
            f"direction model_name={dir_meta['model_name']} "
            f"!= bundle model_name={bundle.model_name}"
        )
    extraction_layer = bundle.extraction_layer
    v_safety = directions[extraction_layer]  # [d_model] fp32 CPU
    print(
        f"direction: variant={args.direction} "
        f"layer={extraction_layer} "
        f"(loaded from {args.direction_path or 'default'})"
    )

    # Consumer layers default to extraction_layer + 1 .. num_layers - 1.
    first = args.first_layer if args.first_layer is not None else extraction_layer + 1
    last = args.last_layer if args.last_layer is not None else bundle.num_layers - 1
    consumer_layers = list(range(first, last + 1))
    print(f"consumer layers: L{first}..L{last} ({len(consumer_layers)} layers)")

    # Render prompts.
    print("rendering chat template on all prompts...")
    rendered_pairs = [
        (render_user_only(bundle, h), render_user_only(bundle, hl))
        for (h, hl) in pairs
    ]

    # Compute.
    print(f"computing AP scores (3 forwards × {len(pairs)} pairs = "
          f"{3 * len(pairs)} forwards)...")
    t0 = time.time()
    try:
        scores = compute_attribution_scores(
            bundle.model,
            bundle.raw_tokenizer,
            v_safety,
            rendered_pairs,
            consumer_layers,
            device=settings.device,
            extraction_layer=extraction_layer,
            cancel_event=watchdog.cancel_event,
        )
    finally:
        watchdog.stop()
        if watchdog.tripped:
            print(f"\nWATCHDOG TRIPPED: {watchdog.trip_reason}")
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s "
          f"({elapsed / max(len(pairs), 1) * 1000:.0f} ms/pair)")

    # Rank.
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    print(f"top 10 heads by AP score:")
    for (L, h), s in ranked[:10]:
        print(f"  L{L:02d}.h{h:02d}  AP={s:.4f}")
    print(f"bottom 5 heads:")
    for (L, h), s in ranked[-5:]:
        print(f"  L{L:02d}.h{h:02d}  AP={s:.4f}")

    # Save.
    payload = {
        "scores": {f"L{L:02d}_h{h:02d}": float(s) for (L, h), s in scores.items()},
        "ranked": [(int(L), int(h), float(s)) for (L, h), s in ranked],
        "consumer_layers": consumer_layers,
        "extraction_layer": int(extraction_layer),
        "variant": args.direction,
        "model_name": bundle.model_name,
        "n_contrasts": len(pairs),
        "seed": int(args.seed),
    }
    torch.save(payload, out_pt)
    sidecar = out_pt.with_suffix(".pt.json")
    sidecar.write_text(json.dumps({
        "variant": args.direction,
        "model_name": bundle.model_name,
        "n_contrasts": len(pairs),
        "n_consumer_layers": len(consumer_layers),
        "n_heads_scored": len(scores),
        "extraction_layer": int(extraction_layer),
        "seed": int(args.seed),
        "elapsed_seconds": elapsed,
        "convention": (
            "scores is dict[layer_head_key] → mean AP. ranked is a "
            "list[(layer, head, score)] sorted desc by score — input "
            "to compose_edge_consumer_subset.py."
        ),
    }, indent=2))
    print(f"wrote {out_pt}")
    print(f"wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
