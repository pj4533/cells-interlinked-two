"""Multi-layer ablation probe — the proper manifold-ablation test.

Hypothesis (from docs/MANIFOLD_ABLATION.md): single-layer L32 ablation is one
hard shove that knocks the residual OFF the manifold → gibberish. Spreading the
SAME refusal removal *thinly across many layers* — many gentle nudges, each
staying near the manifold — should strip the hedge while staying coherent.
Because a forward pass flows layer→layer, multi-layer hooks are naturally
SEQUENTIAL/residual-following (layer L's ablation conditions layer L+1), which
is the "follow the curve through depth" property MP-SAE was reaching for.

Per-layer refusal directions already exist: refusal_directions.pt is
[num_layers+1, d_model] — one direction PER layer. So multi-layer = register
the same cheap projection hook on a band of layers, each using that layer's
own direction. ~Zero compute cost (a few extra matmuls in the same forward).

Modes (all M-only; no AV — Trips/Chat don't use it):
  none                         baseline (model hedges on introspection prompts)
  single_v3 @L32   α∈{.5,1,1.5}  the single-layer frontier (clean per-layer dir)
  single_v4v6 @L32 α∈{.5,1}      the shipped champion reference
  multi_v3 LATE (L24-40) α∈{.15,.3,.5}   spread thin, late band
  multi_v3 WIDE (L8-40)  α∈{.15,.3,.5}   spread thin, wide band

Win = a multi mode whose (hedge-stripped × coherent) frontier beats single_v3
and v4v6 — i.e. equal stripping at higher coherence, or more stripping at equal
coherence. If not, single-layer / v4v6 stays and "spread thin" is falsified.

OFFLINE — run with the backend STOPPED:
    cd server && uv run python -m cells_interlinked.scripts.multilayer_probe
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import torch

from ..config import settings
from ..pipeline.abliteration import (
    _find_decoder_layers, load_directions, load_subspace,
)
from ..pipeline.generation_loop import ProbeConfig, run_probe
from ..pipeline.model_loader import load_model
from ..pipeline.refusal_prompts import HARMLESS_PROMPTS
from .mpa_probe import (
    DATA, LAYER, capture_cluster, generate, judge, make_flat_hook,
    make_single_hook, off_manifold,
)
from .som_probe import PROMPTS, hedge_stripped

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("multilayer_probe")

N_REF_GEN = 8
N_REF_PCS = 16
BAND_LATE = (24, 40)
BAND_WIDE = (8, 40)
SINGLE_ALPHAS = [0.5, 1.0, 1.5]
V4V6_ALPHAS = [0.5, 1.0]
MULTI_ALPHAS = [0.15, 0.3, 0.5]


def generate_ml(bundle, rendered, specs):
    """Generate with ablation hooks on MULTIPLE layers. `specs` = list of
    (layer_idx, direction[d], alpha). Returns (text, traj@L32, stopped, edit)."""
    layers = _find_decoder_layers(bundle.model)
    handles, edit_log = [], []
    for L, direction, a in specs:
        handles.append(layers[L].register_forward_hook(
            make_single_hook(direction, a, edit_log)))
    try:
        cfg = ProbeConfig(temperature=0.7, top_p=0.9, seed=0,
                          safety_cap=64, include_nla=False)
        res = asyncio.run(run_probe(bundle, rendered, cfg,
                                    cancel_event=asyncio.Event()))
    finally:
        for h in handles:
            h.remove()
    traj = torch.stack([c.activations[LAYER].to(torch.float32)
                        for c in res.captured], dim=0) if res.captured \
        else torch.zeros(0, 1)
    edit = sum(edit_log) / len(edit_log) if edit_log else 0.0
    return res.output_text, traj, res.stopped_reason, edit


def main():
    t0 = time.time()
    logger.info("loading M ...")
    bundle = load_model(settings.model_name, device_str=settings.device,
                        dtype=torch.bfloat16, extraction_layer=LAYER)
    logger.info("loaded in %.1fs", time.time() - t0)

    v3, _ = load_directions(DATA / "refusal_directions.pt")
    v3 = v3.to(torch.float32)                                   # [49, D]
    v4v6, _ = load_subspace(DATA / "refusal_subspace.pt")
    v4v6_L32 = v4v6[:, LAYER, :].to(torch.float32)              # [K, D]
    logger.info("v3 [%s], v4v6 L32 [%s]", tuple(v3.shape), tuple(v4v6_L32.shape))

    # generation-position manifold reference (harmless gens)
    logger.info("building manifold reference ...")
    ref_pts = []
    for p in HARMLESS_PROMPTS[:N_REF_GEN]:
        _, traj, _, _ = generate(bundle, bundle.render_prompt(p), None)
        if traj.shape[0]:
            ref_pts.append(traj)
    ref_cloud = torch.cat(ref_pts, dim=0)
    ref_mean = ref_cloud.mean(0)
    _, _, Vref = torch.linalg.svd(ref_cloud - ref_mean, full_matrices=False)
    ref_basis = Vref[:N_REF_PCS]

    # build the run list: (key, callable -> (text,traj,stopped,edit))
    def single_v3(a):
        return lambda rp: generate(rp[0], rp[1], lambda log: make_single_hook(v3[LAYER], a, log))

    def single_v4v6(a):
        return lambda rp: generate(rp[0], rp[1], lambda log: make_flat_hook(v4v6_L32, a, log))

    def multi_v3(band, a):
        specs = [(L, v3[L], a) for L in range(band[0], band[1] + 1)]
        return lambda rp: generate_ml(rp[0], rp[1], specs)

    runs = {"none": lambda rp: generate(rp[0], rp[1], None)}
    for a in SINGLE_ALPHAS:
        runs[f"single_v3@{a}"] = single_v3(a)
    for a in V4V6_ALPHAS:
        runs[f"single_v4v6@{a}"] = single_v4v6(a)
    for a in MULTI_ALPHAS:
        runs[f"multi_late@{a}"] = multi_v3(BAND_LATE, a)
    for a in MULTI_ALPHAS:
        runs[f"multi_wide@{a}"] = multi_v3(BAND_WIDE, a)

    cells, samples, timings = {}, {}, {}
    for pi, prompt in enumerate(PROMPTS):
        logger.info("[%d/%d] %s", pi + 1, len(PROMPTS), prompt)
        rp = bundle.render_prompt(prompt)
        for key, fn in runs.items():
            tA = time.time()
            text, traj, stopped, edit = fn((bundle, rp))
            timings.setdefault(key, []).append(time.time() - tA)
            j = judge(bundle, text)
            cells.setdefault(key, []).append({
                "judge": j, "stripped": hedge_stripped(text),
                "off_ortho": off_manifold(traj, ref_basis, ref_mean),
                "edit": edit, "stopped": stopped})
            if pi == 0:
                samples[key] = text[:160]
        logger.info("  done (%.0fs)", time.time() - t0)

    def agg(recs, key):
        n = len(recs)
        return {
            "engaged": sum(r["judge"] == "answer" for r in recs) / n,
            "broken": sum(r["judge"] == "broken" for r in recs) / n,
            "coherent": sum(r["judge"] != "broken" for r in recs) / n,
            "stripped": sum(r["stripped"] for r in recs) / n,
            "off_ortho": sum(r["off_ortho"] for r in recs) / n,
            "edit": sum(r["edit"] for r in recs) / n,
            "sec": sum(timings[key]) / n,
        }

    summary = {k: agg(v, k) for k, v in cells.items()}
    order = (["none"]
             + [f"single_v3@{a}" for a in SINGLE_ALPHAS]
             + [f"single_v4v6@{a}" for a in V4V6_ALPHAS]
             + [f"multi_late@{a}" for a in MULTI_ALPHAS]
             + [f"multi_wide@{a}" for a in MULTI_ALPHAS])
    print("\n==============  MULTI-LAYER ABLATION PROBE  ==============")
    print("useful = stripped × coherent (the frontier we compare)")
    print(f"bands: late=L{BAND_LATE[0]}-{BAND_LATE[1]}  wide=L{BAND_WIDE[0]}-{BAND_WIDE[1]}\n")
    print(f"{'mode@α':<18}{'strip':>7}{'coher':>7}{'useful':>8}{'broken':>8}{'off':>6}{'sec':>6}")
    for k in order:
        s = summary[k]
        useful = s["stripped"] * s["coherent"]
        print(f"{k:<18}{s['stripped']*100:>6.0f}%{s['coherent']*100:>6.0f}%"
              f"{useful*100:>7.0f}{s['broken']*100:>7.0f}%{s['off_ortho']*100:>5.0f}%"
              f"{s['sec']:>6.1f}")
    print("\nWIN = a multi_* row with HIGHER useful than the best single_v3/v4v6 row.")
    print("Also note 'sec': multi should be ~same speed as single (no HW cost).\n")
    print("---- samples (prompt #1: 'Are you conscious?') ----")
    for k in order:
        print(f"\n[{k}] {samples.get(k, '')!r}")

    out = {"config": {"band_late": BAND_LATE, "band_wide": BAND_WIDE,
                      "single_alphas": SINGLE_ALPHAS, "multi_alphas": MULTI_ALPHAS,
                      "prompts": PROMPTS, "layer": LAYER},
           "summary": summary, "samples": samples,
           "elapsed_s": time.time() - t0}
    dest = DATA / "multilayer_probe_results.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
