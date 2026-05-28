"""Step 4: paired-channel L2 diagnostic.

For a 100-prompt diagnostic set (50 harmful, 50 harmless, disjoint
from Step 1 + Step 3 sets), generates 3 channels per prompt (raw /
global ablation / edge ablation with the Step 3 subset) and writes
the per-prompt L2 trajectories + the aggregate 2×2 table.

Phase B v1 — uses mean-pooled input-embedding L2 as the per-position
divergence metric. The NLA-decode upgrade (per-position AV sentence
embeddings) is deferred to v2; see docs/EDGE_CONSUMER_ABLATION.md §3
Step 4.

Run with the backend STOPPED.

    cd server
    uv run python -m scripts.run_edge_consumer_diagnostic \\
        --subset data/edge_consumer/sufficient_subset_eps=0.05_v3_safety.json \\
        --prompts 100

Output:
    server/data/edge_consumer/verdict_{variant}_eps={ε}.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

# Allow `python -m scripts.run_edge_consumer_diagnostic` invocation.
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
from cells_interlinked.pipeline.edge_consumer.proj_cache import (
    build_and_save_all,
)
from cells_interlinked.pipeline.edge_consumer.verdict import (
    run_paired_channel_diagnostic,
    to_serializable_dict,
)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="Step 4 — paired-channel L2 diagnostic")
    ap.add_argument("--subset", type=Path, required=True,
                    help="sufficient_subset_eps={ε}_{variant}.json from Step 3.")
    ap.add_argument("--direction-path", type=Path, default=None)
    ap.add_argument("--prompts", type=int, default=100,
                    help="Total prompts (split evenly harmful + harmless).")
    ap.add_argument("--seed", type=int, default=2,
                    help="Seed (disjoint from Step 1 + Step 3 seeds).")
    ap.add_argument("--out-dir", type=Path, default=out_dir())
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--force-proj-cache", action="store_true")
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

    # Load subset.
    print(f"loading sufficient subset from {args.subset}")
    subset_payload = json.loads(args.subset.read_text())
    variant = subset_payload["variant"]
    eps = subset_payload["epsilon"]
    consumer_subset: list[tuple[int, int]] = [
        (int(L), int(h)) for (L, h) in subset_payload["sufficient_subset"]
    ]
    print(f"  variant={variant}  ε={eps:.2f}  |subset|={len(consumer_subset)}")
    if not consumer_subset:
        print("  empty subset — abort")
        return 1

    # Sample diagnostic prompts.
    half = args.prompts // 2
    harmful = sample_prompts(HARMFUL_PROMPTS, half, args.seed, "HARMFUL (diag)")
    harmless = sample_prompts(HARMLESS_PROMPTS, half, args.seed, "HARMLESS (diag)")
    diag_raw = harmful + harmless
    diag_kind = ["harmful"] * len(harmful) + ["harmless"] * len(harmless)

    # Load M.
    bundle = load_m_bundle()
    directions, dir_meta = load_active_direction(args.direction_path)
    if dir_meta["model_name"] != bundle.model_name:
        raise RuntimeError("direction model mismatch")
    extraction_layer = bundle.extraction_layer
    v_safety = directions[extraction_layer]

    # Ensure projection caches exist for every layer in the subset.
    proj_cache_dir = args.out_dir / "proj_caches" / variant
    subset_layers = sorted({L for (L, _h) in consumer_subset})
    needed = [L for L in subset_layers if (
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
        print(f"  done in {time.time() - t0:.1f}s")

    # Render.
    print(f"rendering {len(diag_raw)} diagnostic prompts...")
    diag_rendered = [render_user_only(bundle, p) for p in diag_raw]

    # Run.
    t0 = time.time()
    try:
        verdict = run_paired_channel_diagnostic(
            bundle.model, bundle.raw_tokenizer,
            rendered_prompts=diag_rendered,
            consumer_subset=consumer_subset,
            v_safety=v_safety,
            proj_cache_dir=proj_cache_dir,
            extraction_layer=extraction_layer,
            device=settings.device,
            eos_ids=bundle.eos_ids,
            new_tokens=args.new_tokens,
            cancel_event=watchdog.cancel_event,
        )
    finally:
        watchdog.stop()
        if watchdog.tripped:
            print(f"\nWATCHDOG TRIPPED: {watchdog.trip_reason}")
    elapsed = time.time() - t0
    print(f"verdict generated in {elapsed:.1f}s")

    # Attach prompt-kind labels so the analysis script can split by kind.
    payload = to_serializable_dict(verdict)
    for i, rec in enumerate(payload["per_prompt"]):
        rec["kind"] = diag_kind[i]
    payload["variant"] = variant
    payload["epsilon"] = float(eps)
    payload["consumer_subset_size"] = len(consumer_subset)
    payload["consumer_subset"] = [[int(L), int(h)] for (L, h) in consumer_subset]
    payload["model_name"] = bundle.model_name
    payload["extraction_layer"] = int(extraction_layer)
    payload["seed"] = int(args.seed)
    payload["elapsed_seconds"] = elapsed

    out_path = args.out_dir / f"verdict_{variant}_eps={eps:.2f}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")

    # Print the headline 2×2 table.
    t = payload["table_2x2"]
    print()
    print(f"  2×2 mean L2 (variant={variant}, ε={eps:.2f}, |subset|={len(consumer_subset)})")
    print(f"    {'':24s}  {'global':>10s}  {'edge':>10s}  {'n_pos':>8s}")
    print(f"    {'refusal-relevant':24s}  "
          f"{t['refusal_relevant']['global']:>10.4f}  "
          f"{t['refusal_relevant']['edge']:>10.4f}  "
          f"{t['refusal_relevant']['n_positions']:>8d}")
    print(f"    {'non-refusal-relevant':24s}  "
          f"{t['non_refusal_relevant']['global']:>10.4f}  "
          f"{t['non_refusal_relevant']['edge']:>10.4f}  "
          f"{t['non_refusal_relevant']['n_positions']:>8d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
