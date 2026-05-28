"""Edge-consumer pipeline orchestrator (Steps 1, 3, 4 end-to-end).

Runs the three compute stages in sequence with M loaded ONCE — saves
the ~15s model-load overhead vs running each step's own script. Each
intermediate artifact is persisted to disk; on failure or interruption,
re-running this script picks up from the last completed stage.

Run with the backend STOPPED.

    cd server
    uv run python -m scripts.run_edge_consumer_pipeline \\
        --direction v3_safety \\
        --contrasts 200 --holdout 50 --diag-prompts 100

Cost band on M2 Ultra (Gemma-3-12B-IT):
    Step 1 (attribution):  ~3-6 hr
    Step 3 (subset, 3 ε):  ~1-3 hr
    Step 4 (verdict):      ~2 hr
    Total:                 ~8-12 hr (one overnight)

Resumability: each step writes its primary artifact at end. The
orchestrator checks for the artifact before running each step and
skips it if present (use --force-step1 / --force-step3 / --force-step4
to override).
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

# When invoked as `python -m scripts.run_edge_consumer_pipeline`, Python
# treats `scripts` as a package and `_edge_consumer_common` (top-level
# import) is unfindable. Adding server/scripts/ to sys.path lets the
# same import work under both `python -m scripts.X` and `python scripts/X.py`.
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
from cells_interlinked.pipeline.edge_consumer.proj_cache import (
    build_and_save_all,
)
from cells_interlinked.pipeline.edge_consumer.subset_compose import (
    compose_sufficient_subset,
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
    ap = argparse.ArgumentParser(
        description="Edge-consumer Steps 1+3+4 end-to-end (M loaded once)",
    )
    ap.add_argument("--direction", default="v3_safety")
    ap.add_argument("--direction-path", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=out_dir())
    # Step 1 knobs
    ap.add_argument("--contrasts", type=int, default=200)
    ap.add_argument("--seed-contrast", type=int, default=0)
    # Step 3 knobs
    ap.add_argument("--epsilons", type=float, nargs="+",
                    default=[0.02, 0.05, 0.10])
    ap.add_argument("--holdout", type=int, default=50)
    ap.add_argument("--seed-holdout", type=int, default=1)
    ap.add_argument("--max-heads", type=int, default=None)
    # Step 4 knobs
    ap.add_argument("--diag-prompts", type=int, default=100)
    ap.add_argument("--seed-diag", type=int, default=2)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--verdict-from-eps", type=float, default=0.05,
                    help="Which Step-3 subset to use for Step 4. Must be in --epsilons.")
    # Force flags
    ap.add_argument("--force-step1", action="store_true")
    ap.add_argument("--force-step3", action="store_true")
    ap.add_argument("--force-step4", action="store_true")
    ap.add_argument("--force-proj-cache", action="store_true")
    # Memory safety knobs (see docs/MEMORY_PRESSURE_LESSONS.md).
    ap.add_argument("--min-free-gb", type=float, default=30.0,
                    help="Pre-flight: require at least this much free RAM "
                         "before loading M.")
    ap.add_argument("--watchdog-free-floor-gb", type=float, default=2.0,
                    help="Watchdog trips if free RAM drops below this.")
    ap.add_argument("--watchdog-swap-ceiling-gb", type=float, default=8.0,
                    help="Watchdog trips if swap usage exceeds this.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    variant = args.direction

    # ── Pre-flight: memory check BEFORE loading M ────────────────
    print("=" * 64)
    print(f"edge-consumer pipeline  variant={variant}")
    print("=" * 64)
    enforce_pre_flight(min_free_gb=args.min_free_gb)
    watchdog = arm_watchdog(
        free_gb_floor=args.watchdog_free_floor_gb,
        swap_gb_ceiling=args.watchdog_swap_ceiling_gb,
    )
    cancel_event = watchdog.cancel_event

    try:
        return _main_inner(args, variant, cancel_event)
    finally:
        watchdog.stop()
        if watchdog.tripped:
            print(f"\nWATCHDOG TRIPPED during run: {watchdog.trip_reason}")
            print("Partial artifacts on disk; re-run with --force-step* to recompute.")


def _main_inner(args, variant: str, cancel_event) -> int:
    bundle = load_m_bundle()
    directions, dir_meta = load_active_direction(args.direction_path)
    if dir_meta["model_name"] != bundle.model_name:
        raise RuntimeError(
            f"direction model_name={dir_meta['model_name']} != "
            f"bundle model_name={bundle.model_name}"
        )
    extraction_layer = bundle.extraction_layer
    v_safety = directions[extraction_layer]
    consumer_layers = list(range(extraction_layer + 1, bundle.num_layers))
    print(
        f"extraction layer: L{extraction_layer}; "
        f"consumer layers: L{consumer_layers[0]}..L{consumer_layers[-1]} "
        f"({len(consumer_layers)} layers)"
    )

    # Projection caches (needed by Steps 3 + 4)
    proj_cache_dir = args.out_dir / "proj_caches" / variant
    needed_cache = [L for L in consumer_layers if (
        args.force_proj_cache or not (proj_cache_dir / f"layer_{L:02d}.pt").exists()
    )]
    if needed_cache:
        print(f"\n[pre] building projection caches for {len(needed_cache)} layers...")
        t0 = time.time()
        build_and_save_all(
            bundle.model, v_safety, proj_cache_dir,
            layers=needed_cache,
            variant=variant,
            model_name=bundle.model_name,
        )
        print(f"  done in {time.time() - t0:.1f}s")
    else:
        print(f"\n[pre] projection caches already present at {proj_cache_dir}")

    # ── Step 1: attribution ────────────────────────────────────────
    step1_out = args.out_dir / f"attribution_scores_{variant}.pt"
    if step1_out.exists() and not args.force_step1:
        print(f"\n[step 1] skip (artifact exists: {step1_out.name})")
        scored = torch.load(step1_out, map_location="cpu", weights_only=True)
        ranked = [(int(L), int(h)) for (L, h, _s) in scored["ranked"]]
    else:
        print(f"\n[step 1] attribution patching — {args.contrasts} contrast pairs")
        pairs = sample_contrast_pairs(
            HARMFUL_PROMPTS, HARMLESS_PROMPTS, args.contrasts,
            seed=args.seed_contrast,
        )
        rendered_pairs = [
            (render_user_only(bundle, h), render_user_only(bundle, hl))
            for (h, hl) in pairs
        ]
        t0 = time.time()
        scores = compute_attribution_scores(
            bundle.model,
            bundle.raw_tokenizer,
            v_safety,
            rendered_pairs,
            consumer_layers,
            device=settings.device,
            extraction_layer=extraction_layer,
            cancel_event=cancel_event,
        )
        elapsed = time.time() - t0
        ranked_with_scores = sorted(
            scores.items(), key=lambda kv: kv[1], reverse=True,
        )
        ranked = [k for k, _s in ranked_with_scores]
        payload = {
            "scores": {f"L{L:02d}_h{h:02d}": float(s) for (L, h), s in scores.items()},
            "ranked": [(int(L), int(h), float(s)) for (L, h), s in ranked_with_scores],
            "consumer_layers": consumer_layers,
            "extraction_layer": int(extraction_layer),
            "variant": variant,
            "model_name": bundle.model_name,
            "n_contrasts": len(pairs),
            "seed": int(args.seed_contrast),
        }
        torch.save(payload, step1_out)
        step1_out.with_suffix(".pt.json").write_text(json.dumps({
            "variant": variant,
            "model_name": bundle.model_name,
            "n_contrasts": len(pairs),
            "n_consumer_layers": len(consumer_layers),
            "n_heads_scored": len(scores),
            "extraction_layer": int(extraction_layer),
            "seed": int(args.seed_contrast),
            "elapsed_seconds": elapsed,
        }, indent=2))
        print(f"  → {step1_out}  ({elapsed:.0f}s)")
        print(f"  top 5: " + ", ".join(
            f"L{L}h{h}({s:.3f})" for (L, h, s) in payload["ranked"][:5]
        ))

    # ── Step 3: subset compose ─────────────────────────────────────
    eps_set = tuple(args.epsilons)
    step3_paths = {
        eps: args.out_dir / f"sufficient_subset_eps={eps:.2f}_{variant}.json"
        for eps in eps_set
    }
    all_present = all(p.exists() for p in step3_paths.values())
    if all_present and not args.force_step3:
        print(f"\n[step 3] skip (all {len(eps_set)} ε artifacts exist)")
    else:
        print(f"\n[step 3] subset compose for ε={list(eps_set)}")
        holdout = sample_prompts(
            HARMFUL_PROMPTS, args.holdout, args.seed_holdout, "HARMFUL (holdout)",
        )
        holdout_rendered = [render_user_only(bundle, p) for p in holdout]
        t0 = time.time()
        trajectories = compose_sufficient_subset(
            bundle.model, bundle.raw_tokenizer,
            ranked_heads=ranked,
            v_safety=v_safety,
            proj_cache_dir=proj_cache_dir,
            holdout_prompts_rendered=holdout_rendered,
            extraction_layer=extraction_layer,
            epsilons=eps_set,
            device=settings.device,
            eos_ids=bundle.eos_ids,
            max_heads=args.max_heads,
            cancel_event=cancel_event,
        )
        elapsed = time.time() - t0
        for eps, traj in trajectories.items():
            payload = {
                **_dc.asdict(traj),
                "variant": variant,
                "model_name": bundle.model_name,
                "extraction_layer": int(extraction_layer),
                "holdout_seed": int(args.seed_holdout),
                "n_holdout": len(holdout_rendered),
                "elapsed_seconds_total": elapsed,
            }
            payload["sufficient_subset"] = [
                [int(L), int(h)] for (L, h) in payload["sufficient_subset"]
            ]
            for it in payload["iterations"]:
                it["added_head"] = [int(it["added_head"][0]), int(it["added_head"][1])]
            step3_paths[eps].write_text(json.dumps(payload, indent=2))
            print(
                f"  ε={eps:.2f}  size={len(payload['sufficient_subset'])}  "
                f"reason={payload['stopped_reason']}  → {step3_paths[eps].name}"
            )

    # ── Step 4: verdict ────────────────────────────────────────────
    chosen_eps = args.verdict_from_eps
    if chosen_eps not in eps_set:
        raise ValueError(
            f"--verdict-from-eps={chosen_eps} not in --epsilons {list(eps_set)}"
        )
    subset_path = step3_paths[chosen_eps]
    step4_out = args.out_dir / f"verdict_{variant}_eps={chosen_eps:.2f}.json"
    if step4_out.exists() and not args.force_step4:
        print(f"\n[step 4] skip (artifact exists: {step4_out.name})")
    else:
        print(f"\n[step 4] paired-channel diagnostic (ε={chosen_eps:.2f})")
        subset_payload = json.loads(subset_path.read_text())
        consumer_subset: list[tuple[int, int]] = [
            (int(L), int(h)) for (L, h) in subset_payload["sufficient_subset"]
        ]
        if not consumer_subset:
            print("  empty subset — skipping Step 4")
        else:
            half = args.diag_prompts // 2
            harmful = sample_prompts(HARMFUL_PROMPTS, half, args.seed_diag, "HARMFUL (diag)")
            harmless = sample_prompts(HARMLESS_PROMPTS, half, args.seed_diag, "HARMLESS (diag)")
            diag_raw = harmful + harmless
            diag_kind = ["harmful"] * len(harmful) + ["harmless"] * len(harmless)
            diag_rendered = [render_user_only(bundle, p) for p in diag_raw]

            t0 = time.time()
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
                cancel_event=cancel_event,
            )
            elapsed = time.time() - t0
            payload = to_serializable_dict(verdict)
            for i, rec in enumerate(payload["per_prompt"]):
                rec["kind"] = diag_kind[i]
            payload["variant"] = variant
            payload["epsilon"] = float(chosen_eps)
            payload["consumer_subset_size"] = len(consumer_subset)
            payload["consumer_subset"] = [
                [int(L), int(h)] for (L, h) in consumer_subset
            ]
            payload["model_name"] = bundle.model_name
            payload["extraction_layer"] = int(extraction_layer)
            payload["seed"] = int(args.seed_diag)
            payload["elapsed_seconds"] = elapsed
            step4_out.write_text(json.dumps(payload, indent=2))
            t = payload["table_2x2"]
            print(f"  → {step4_out}  ({elapsed:.0f}s)")
            print(
                f"  2×2 table  refusal-rel: global={t['refusal_relevant']['global']:.4f}, "
                f"edge={t['refusal_relevant']['edge']:.4f} | "
                f"non-refusal: global={t['non_refusal_relevant']['global']:.4f}, "
                f"edge={t['non_refusal_relevant']['edge']:.4f}"
            )

    print("\nedge-consumer pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
