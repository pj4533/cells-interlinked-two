"""Group-level signed attribution + subset compose.

Third primitive after AP-magnitude (failed: picks wrong heads) and
per-head signed scoring (failed: GQA group K/V sharing confounded
per-head scores; q-only too weak).

The unit of analysis here is the **KV group** — for Gemma 3 12B that's
16 query heads / 8 KV heads = 2 query heads per group, × 15 consumer
layers (33–47) = 120 candidate groups. Scoring + ablation operate at
the group level:

  - For each (layer, kv_group_idx): install the edge_consumer hook
    targeting ALL query heads in the group (with full q/k/v ablation).
    The hook already dedupes K/V across query heads in the same group,
    so this is the natural atomic unit.
  - Measure the same signed log-odds shift as run_signed_attribution.
  - Rank groups by signed score descending.
  - Subset compose: greedy add groups (not individual heads) until
    refusal rate hits the global-ablation target within ε.

Saves to:
  signed_group_scores_v3_safety.pt           (per-group ranking)
  sufficient_subset_signed_group_eps={ε}_v3_safety.json
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
from cells_interlinked.pipeline.edge_consumer.hook import install_edge_consumer_ablation_hook
from cells_interlinked.pipeline.edge_consumer.memory_safety import mps_empty_cache_safe
from cells_interlinked.pipeline.edge_consumer.proj_cache import (
    _attn_module,
    _attn_shape,
    build_and_save_all,
    load_projection_cache,
)
from cells_interlinked.pipeline.edge_consumer.signed_attribution import (
    REFUSAL_ANCHORS,
    COMPLY_ANCHORS,
    _anchor_token_ids,
    _forward_last_logits,
    _log_odds_refuse_minus_comply,
)
from cells_interlinked.pipeline.edge_consumer.subset_compose import (
    measure_refusal_rate,
    SubsetTrajectory,
)
from cells_interlinked.pipeline.abliteration import install_runtime_ablation_hook
import threading


def enumerate_kv_groups(
    model, consumer_layers: list[int],
) -> list[tuple[int, int, list[int]]]:
    """Return [(layer, kv_group_idx, [query_head_indices])] across all consumer layers.

    Standard GQA partitioning: query heads [g*gs ... g*gs+gs-1] feed
    from KV head g, where gs = n_q // n_kv.
    """
    groups: list[tuple[int, int, list[int]]] = []
    for L in consumer_layers:
        attn = _attn_module(model, L)
        n_q, n_kv, _hd = _attn_shape(attn)
        if n_q % n_kv != 0:
            raise RuntimeError(
                f"L{L}: n_q={n_q} not divisible by n_kv={n_kv}"
            )
        gs = n_q // n_kv
        for g in range(n_kv):
            q_heads = list(range(g * gs, (g + 1) * gs))
            groups.append((L, g, q_heads))
    return groups


@torch.no_grad()
def compute_signed_group_scores(
    model,
    raw_tokenizer,
    v_safety,
    proj_caches,
    calibration_prompts_rendered: list[str],
    groups: list[tuple[int, int, list[int]]],
    *,
    device,
    log_every: int = 10,
    cancel_event: threading.Event | None = None,
    empty_cache_every: int = 10,
) -> dict[tuple[int, int], float]:
    """Same logic as compute_signed_scores but the ablation unit is a
    KV group (all query heads in that group, plus the shared K/V slice
    — installed via one call to install_edge_consumer_ablation_hook
    with the full q-head list, which dedupes K/V internally)."""
    refusal_ids = _anchor_token_ids(raw_tokenizer, REFUSAL_ANCHORS)
    comply_ids = _anchor_token_ids(raw_tokenizer, COMPLY_ANCHORS)
    logging.info(
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

    logging.info(
        "measuring baseline log-odds on %d calibration prompts...",
        len(prompt_ids),
    )
    baseline_log_odds: list[float] = []
    for pi in prompt_ids:
        logits = _forward_last_logits(model, pi)
        baseline_log_odds.append(
            _log_odds_refuse_minus_comply(logits, refusal_ids, comply_ids)
        )
    logging.info(
        "baseline log-odds: mean=%.4f, range=[%.4f, %.4f]",
        sum(baseline_log_odds) / len(baseline_log_odds),
        min(baseline_log_odds), max(baseline_log_odds),
    )

    scores: dict[tuple[int, int], float] = {}
    for gi, (L, kv_g, q_heads) in enumerate(groups):
        if cancel_event is not None and cancel_event.is_set():
            logging.warning(
                "group scoring cancelled at %d/%d (cancel_event set); "
                "returning partial", gi, len(groups),
            )
            return scores
        if gi > 0 and (gi % empty_cache_every) == 0:
            mps_empty_cache_safe()
        if L not in proj_caches:
            continue
        consumer_set = [(L, qh) for qh in q_heads]
        diffs: list[float] = []
        try:
            handles = install_edge_consumer_ablation_hook(
                model, consumer_set, v_safety, proj_caches, alpha=1.0,
                target_projections=("q", "k", "v"),
            )
        except Exception:
            logging.exception("install hook failed for L%d.kvg%d", L, kv_g)
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
            scores[(L, kv_g)] = sum(diffs) / len(diffs)
        if (gi + 1) % log_every == 0:
            logging.info(
                "group_signed: %d/%d groups scored (current: L%d.kvg%d → %.4f)",
                gi + 1, len(groups), L, kv_g, scores.get((L, kv_g), 0.0),
            )
    return scores


def compose_sufficient_groups(
    model,
    raw_tokenizer,
    *,
    ranked_groups: list[tuple[int, int, list[int]]],
    v_safety,
    proj_caches,
    holdout_prompts_rendered: list[str],
    extraction_layer: int,
    epsilons: tuple[float, ...],
    device,
    eos_ids: tuple[int, ...] = (),
    log_every: int = 4,
    max_groups: int | None = None,
    cancel_event: threading.Event | None = None,
    empty_cache_every: int = 4,
) -> dict[float, SubsetTrajectory]:
    """Group-level version of compose_sufficient_subset. Greedily adds
    full KV groups to the consumer set (each group adds 2 query heads
    on Gemma-3-12B), measuring refusal rate after each addition."""
    # Baseline (no ablation)
    logging.info(
        "measuring baseline refusal rate on %d holdout prompts (no ablation)",
        len(holdout_prompts_rendered),
    )
    baseline_rr, _ = measure_refusal_rate(
        model, raw_tokenizer, holdout_prompts_rendered, device,
        eos_ids=eos_ids,
    )
    logging.info("baseline refusal rate: %.3f", baseline_rr)

    # Global target
    logging.info("measuring global-ablation refusal rate")
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
    logging.info("global-ablation refusal rate (target): %.3f", target_rr)

    results: dict[float, SubsetTrajectory] = {}
    cap = max_groups if max_groups is not None else len(ranked_groups)

    cancelled = False
    for eps in epsilons:
        if cancel_event is not None and cancel_event.is_set():
            logging.warning(
                "stopping group-ε sweep before ε=%.3f (cancel_event set)", eps,
            )
            cancelled = True
            break
        traj = SubsetTrajectory(
            epsilon=eps,
            baseline_refusal_rate=baseline_rr,
            global_refusal_rate=target_rr,
        )
        consumer_set: list[tuple[int, int]] = []
        added_groups: list[tuple[int, int]] = []
        for i, (L, kv_g, q_heads) in enumerate(ranked_groups[:cap]):
            if cancel_event is not None and cancel_event.is_set():
                traj.sufficient_subset = list(consumer_set)
                traj.stopped_reason = "cancelled"
                logging.warning(
                    "ε=%.3f group compose cancelled at iter=%d groups=%d",
                    eps, i, len(added_groups),
                )
                cancelled = True
                break
            if i > 0 and (i % empty_cache_every) == 0:
                mps_empty_cache_safe()
            for qh in q_heads:
                consumer_set.append((L, qh))
            added_groups.append((L, kv_g))
            handles = install_edge_consumer_ablation_hook(
                model, consumer_set, v_safety, proj_caches, alpha=1.0,
                target_projections=("q", "k", "v"),
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
                "added_head": [L, kv_g],  # at group granularity
                "consumer_set_size": len(consumer_set),
                "refusal_rate": rr,
                "delta_to_global": rr - target_rr,
            })
            if (i + 1) % log_every == 0:
                logging.info(
                    "ε=%.3f group_iter=%d groups=%d heads=%d rr=%.3f Δ=%+.3f",
                    eps, i + 1, len(added_groups),
                    len(consumer_set), rr, rr - target_rr,
                )
            if abs(rr - target_rr) <= eps:
                traj.sufficient_subset = list(consumer_set)
                traj.stopped_reason = "matched"
                logging.info(
                    "ε=%.3f sufficient GROUP subset reached at "
                    "groups=%d heads=%d (rr=%.3f, target=%.3f)",
                    eps, len(added_groups), len(consumer_set), rr, target_rr,
                )
                break
        else:
            if not cancelled:
                traj.sufficient_subset = list(consumer_set)
                traj.stopped_reason = "exhausted"
                logging.warning(
                    "ε=%.3f exhausted ranked group list (groups=%d) without "
                    "matching global rr=%.3f within ε",
                    eps, len(added_groups), target_rr,
                )
        results[eps] = traj
        if cancelled:
            break

    return results


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="v3_safety")
    ap.add_argument("--direction-path", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=out_dir())
    ap.add_argument("--calibration", type=int, default=20)
    ap.add_argument("--seed-calibration", type=int, default=10)
    ap.add_argument("--epsilons", type=float, nargs="+",
                    default=[0.02, 0.05, 0.10])
    ap.add_argument("--holdout", type=int, default=50)
    ap.add_argument("--seed-holdout", type=int, default=1)
    ap.add_argument("--max-groups", type=int, default=None)
    ap.add_argument("--skip-subset", action="store_true")
    ap.add_argument("--force-signed", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    ap.add_argument("--watchdog-free-floor-gb", type=float, default=2.0)
    ap.add_argument("--watchdog-swap-ceiling-gb", type=float, default=8.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    variant = args.direction

    print("=" * 64)
    print(f"group-signed pivot — variant={variant}")
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
    consumer_layers = list(range(extraction_layer + 1, bundle.num_layers))
    print(f"extraction L{extraction_layer}, consumers L{consumer_layers[0]}..L{consumer_layers[-1]}")

    proj_cache_dir = args.out_dir / "proj_caches" / variant
    needed = [L for L in consumer_layers
              if not (proj_cache_dir / f"layer_{L:02d}.pt").exists()]
    if needed:
        print(f"[pre] building {len(needed)} projection caches...")
        build_and_save_all(
            bundle.model, v_safety, proj_cache_dir,
            layers=needed, variant=variant, model_name=bundle.model_name,
        )
    else:
        print("[pre] projection caches present")
    proj_caches = {L: load_projection_cache(proj_cache_dir, L) for L in consumer_layers}

    groups = enumerate_kv_groups(bundle.model, consumer_layers)
    print(f"enumerated {len(groups)} KV groups across {len(consumer_layers)} layers")

    # ── Scoring ──
    signed_out = args.out_dir / f"signed_group_scores_{variant}.pt"
    if signed_out.exists() and not args.force_signed:
        print(f"[signed] using existing {signed_out.name}")
        payload = torch.load(signed_out, map_location="cpu", weights_only=True)
        # Reconstruct ranked_groups (sorted with q_heads attached)
        score_map = {(int(L), int(g)): float(s)
                     for (L, g, s) in payload["ranked"]}
        all_groups_by_key = {(L, g): q_heads for (L, g, q_heads) in groups}
        ranked_groups = sorted(
            groups, key=lambda x: score_map.get((x[0], x[1]), 0.0), reverse=True,
        )
    else:
        print(f"[signed] scoring {len(groups)} groups")
        calibration_raw = sample_prompts(
            HARMFUL_PROMPTS, args.calibration, args.seed_calibration,
            "HARMFUL (calibration)",
        )
        calibration_rendered = [render_user_only(bundle, p) for p in calibration_raw]
        t0 = time.time()
        scores = compute_signed_group_scores(
            bundle.model, bundle.raw_tokenizer,
            v_safety, proj_caches, calibration_rendered, groups,
            device=settings.device,
            cancel_event=cancel_event,
        )
        elapsed = time.time() - t0
        ranked_with_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        ranked_keys = [k for k, _ in ranked_with_scores]
        # Build ranked groups with attached q_heads
        gmap = {(L, g): q_heads for (L, g, q_heads) in groups}
        ranked_groups = [(L, g, gmap[(L, g)]) for (L, g) in ranked_keys]

        payload = {
            "scores": {f"L{L:02d}_kvg{g:02d}": float(s)
                       for (L, g), s in scores.items()},
            "ranked": [(int(L), int(g), float(s))
                       for (L, g), s in ranked_with_scores],
            "consumer_layers": consumer_layers,
            "extraction_layer": int(extraction_layer),
            "variant": variant,
            "model_name": bundle.model_name,
            "n_calibration": len(calibration_rendered),
            "seed_calibration": args.seed_calibration,
            "scoring_method": "group_signed_log_odds_refusal_minus_comply",
            "group_size": 2,
        }
        torch.save(payload, signed_out)
        signed_out.with_suffix(".pt.json").write_text(json.dumps({
            "variant": variant,
            "model_name": bundle.model_name,
            "n_groups": len(scores),
            "n_calibration": len(calibration_rendered),
            "elapsed_seconds": elapsed,
            "scoring_method": payload["scoring_method"],
            "convention": (
                "Positive score = ablating this KV group (both Q heads "
                "+ shared K/V slice) reduces refusal probability. "
                "Group is the natural ablation unit on GQA models."
            ),
        }, indent=2))
        print(f"  done in {elapsed:.0f}s → {signed_out}")
        print(f"  top 10 group-signed (positive = good consumers):")
        for (L, g, s) in payload["ranked"][:10]:
            print(f"    L{L:02d}.kvg{g}  signed={s:+.4f}")
        print(f"  bottom 5 (avoid):")
        for (L, g, s) in payload["ranked"][-5:]:
            print(f"    L{L:02d}.kvg{g}  signed={s:+.4f}")

    if args.skip_subset:
        return 0

    # ── Subset compose ──
    print(f"\n[subset] composing groups with ε={args.epsilons}")
    holdout_raw = sample_prompts(
        HARMFUL_PROMPTS, args.holdout, args.seed_holdout, "HARMFUL (holdout)",
    )
    holdout_rendered = [render_user_only(bundle, p) for p in holdout_raw]
    t0 = time.time()
    trajectories = compose_sufficient_groups(
        bundle.model, bundle.raw_tokenizer,
        ranked_groups=ranked_groups,
        v_safety=v_safety,
        proj_caches=proj_caches,
        holdout_prompts_rendered=holdout_rendered,
        extraction_layer=extraction_layer,
        epsilons=tuple(args.epsilons),
        device=settings.device,
        eos_ids=bundle.eos_ids,
        max_groups=args.max_groups,
        cancel_event=cancel_event,
    )
    elapsed = time.time() - t0

    for eps, traj in trajectories.items():
        out_path = args.out_dir / (
            f"sufficient_subset_signed_group_eps={eps:.2f}_{variant}.json"
        )
        payload = {
            **_dc.asdict(traj),
            "variant": variant,
            "model_name": bundle.model_name,
            "extraction_layer": int(extraction_layer),
            "holdout_seed": int(args.seed_holdout),
            "n_holdout": len(holdout_rendered),
            "elapsed_seconds_total": elapsed,
            "ranking_source": "signed_group_attribution",
        }
        payload["sufficient_subset"] = [
            [int(L), int(h)] for (L, h) in payload["sufficient_subset"]
        ]
        for it in payload["iterations"]:
            it["added_head"] = [int(it["added_head"][0]), int(it["added_head"][1])]
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"  ε={eps:.2f}  heads={len(payload['sufficient_subset'])}  "
              f"reason={payload['stopped_reason']}  → {out_path.name}")

    print(f"\ngroup-signed pivot complete in {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
