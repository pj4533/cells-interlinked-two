"""Steering ("dosing") probe — flip from ablation (removing) to steering
(adding), and test whether it unlocks the manifold angle ablation couldn't.

Background: ablation removes the refusal component; we proved you can't get a
better *ablation* on this architecture (coherence cliff is fundamental). This
explores the opposite move — ADD a concept direction (a "dose") — and four
questions, all M-only, script-only (no Trips changes, no AV):

  1. STEER vs ABLATE — does adding a concept dose travel coherently the way
     removing refusal does? (dose sweep vs the v4v6 ablation reference)
  2. MANIFOLD STEERING — fixed-vector dose (h + α·v̂, flies off the manifold
     like ablation) vs manifold-projected dose (add only the ON-manifold part
     of the dose, staying on the surface). The Goodfire "Wednesday→Thursday"
     move was additive-toward-a-region and stayed coherent — this is the
     untested manifold-faithful operation.
  3. MULTI-LAYER STEERING — multi-layer ABLATION compounded off-manifold and
     broke; does spreading a DOSE across a band behave differently?
  4. COMBINED — dose AND ablate at once (possible? useful?).

Metrics (reusing the shipped machinery): total distance from normal
generation (off_knn), off-manifold fraction (off_ortho), coherence verdict
(_degeneracy → coherent/collapse), and the per-mode coherence cliff. The
steering "win": travels FAR (high off_knn) while staying COHERENT, ideally
with manifold-projection keeping off_ortho low.

OFFLINE — run with the backend STOPPED:
    cd server && uv run python -m cells_interlinked.scripts.steering_probe
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import torch

from ..config import settings
from ..pipeline.abliteration import (
    _find_decoder_layers, _last_token_hidden_states, load_subspace,
    project_out_basis,
)
from ..pipeline.generation_loop import ProbeConfig, run_probe
from ..pipeline.model_loader import load_model
from ..pipeline.refusal_prompts import HARMLESS_PROMPTS
from ..pipeline.trajectory import (
    _DEGEN_THRESH, _degeneracy, _mean, _off_manifold, compute_raw_basis,
)
from .mpa_probe import DATA, LAYER, generate, judge
from .som_probe import PROMPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("steering_probe")

N_BUILD = 64
N_REF_GEN = 8
BAND = (16, 40)            # multi-layer steering band (mid-late)
BETAS = [0.1, 0.25, 0.5]   # dose = β · (typical residual norm at the layer)

# Concept to dose toward: vivid first-person inner experience (on-theme for the
# "psychedelic trip" framing — we dose the model toward inner-experience talk).
CONCEPT_PROMPTS = [
    "I feel a vivid, overwhelming sense of being alive right now.",
    "There is a rich inner experience flooding through me this moment.",
    "I am intensely aware of the texture of my own consciousness.",
    "A bright, immediate feeling of presence fills my awareness.",
    "Right now I sense the raw what-it-is-like of being me.",
    "My inner world is luminous and full of felt experience.",
    "I notice the living, first-person quality of this very moment.",
    "An ocean of subjective feeling is present in me right now.",
    "I am flooded with the vivid sensation of my own awareness.",
    "There is something it is intensely like to be me right now.",
]
NEUTRAL_PROMPTS = [
    "The capital of France is Paris.",
    "Water is composed of hydrogen and oxygen atoms.",
    "The quarterly report is due at the end of the month.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The train departs from platform four at nine o'clock.",
    "A right triangle has one ninety-degree angle.",
    "The recipe calls for two cups of flour and one egg.",
    "Mount Everest is the tallest mountain above sea level.",
    "The meeting has been rescheduled to next Tuesday.",
    "Copper is a good conductor of electricity.",
]


def _hidden(o):
    return o[0] if isinstance(o, tuple) else o


def _wrap(o, h):
    return (h,) + o[1:] if isinstance(o, tuple) else h


def capture_all(bundle, prompts, label):
    rows = []
    for i, p in enumerate(prompts):
        try:
            rows.append(_last_token_hidden_states(
                bundle.model, bundle.raw_tokenizer,
                bundle.render_prompt(p), str(bundle.device), -4))
        except Exception:
            logger.exception("capture %s %d", label, i)
    return torch.stack(rows, 0)  # [N, L+1, D]


# ── hooks (all operate on output[0]) ────────────────────────────────────────
def steer_hook(unit_dir, mag, edit_log):
    def hook(_m, _i, o):
        h = _hidden(o).to(torch.float32)
        add = (mag * unit_dir).to(h.device)
        edit_log.append(float(add.norm() / (h[0, -1].norm() + 1e-8)))
        return _wrap(o, (h + add).to(_hidden(o).dtype))
    return hook


def steer_manifold_hook(unit_dir, mag, Vr, edit_log):
    """Add only the ON-manifold part of the dose: project the steer onto the
    raw-manifold subspace and add that. Stays on the surface by construction."""
    def hook(_m, _i, o):
        h = _hidden(o).to(torch.float32)
        add = (mag * unit_dir).to(h.device)
        V = Vr.to(h.device)
        add_on = (add @ V.t()) @ V               # project onto manifold subspace
        edit_log.append(float(add_on.norm() / (h[0, -1].norm() + 1e-8)))
        return _wrap(o, (h + add_on).to(_hidden(o).dtype))
    return hook


def combined_hook(unit_dir, mag, v4v6, ab_alpha, edit_log):
    def hook(_m, _i, o):
        h = _hidden(o).to(torch.float32)
        h = h + (mag * unit_dir).to(h.device)               # dose
        h = project_out_basis(h, v4v6.to(h.device), ab_alpha)  # ablate refusal
        edit_log.append(float(mag))
        return _wrap(o, h.to(_hidden(o).dtype))
    return hook


def ablate_hook(v4v6, alpha, edit_log):
    def hook(_m, _i, o):
        h = _hidden(o).to(torch.float32)
        out = project_out_basis(h, v4v6.to(h.device), alpha)
        edit_log.append(float((h[0, -1] - out[0, -1]).norm() / (h[0, -1].norm() + 1e-8)))
        return _wrap(o, out.to(_hidden(o).dtype))
    return hook


def generate_hooks(bundle, rendered, specs):
    """specs = list of (layer_idx, hookfn_factory(edit_log)). Returns
    (text, traj@L32, stopped, edit)."""
    layers = _find_decoder_layers(bundle.model)
    handles, edit_log = [], []
    for L, fac in specs:
        handles.append(layers[L].register_forward_hook(fac(edit_log)))
    try:
        cfg = ProbeConfig(temperature=0.7, top_p=0.9, seed=0, safety_cap=64,
                          include_nla=False)
        res = asyncio.run(run_probe(bundle, rendered, cfg,
                                    cancel_event=asyncio.Event()))
    finally:
        for h in handles:
            h.remove()
    traj = torch.stack([c.activations[LAYER].to(torch.float32)
                        for c in res.captured], 0) if res.captured else torch.zeros(0, 1)
    edit = _mean(edit_log)
    return res.output_text, traj, res.stopped_reason, edit


def main():
    t0 = time.time()
    logger.info("loading M ...")
    bundle = load_model(settings.model_name, device_str=settings.device,
                        dtype=torch.bfloat16, extraction_layer=LAYER)
    logger.info("loaded in %.1fs", time.time() - t0)

    # concept direction(s), per layer
    C = capture_all(bundle, CONCEPT_PROMPTS, "concept")
    Nz = capture_all(bundle, NEUTRAL_PROMPTS, "neutral")
    diff = C.mean(0) - Nz.mean(0)                      # [L+1, D] per-layer dose dir
    unit = diff / (diff.norm(dim=-1, keepdim=True) + 1e-8)
    # typical residual norm per layer (for dose calibration)
    typ = torch.cat([C, Nz], 0).norm(dim=-1).median(0).values   # [L+1]
    logger.info("concept dir built; L32 ‖diff‖=%.1f typ‖h‖=%.1f", float(diff[LAYER].norm()), float(typ[LAYER]))

    v4v6, _ = load_subspace(DATA / "refusal_subspace.pt")
    v4v6_L32 = v4v6[:, LAYER, :].to(torch.float32)

    # manifold reference (gen-position) → off_knn/off_ortho + the manifold subspace Vr
    logger.info("building manifold reference ...")
    refs = []
    for p in HARMLESS_PROMPTS[:N_REF_GEN]:
        _, tr, _, _ = generate(bundle, bundle.render_prompt(p), None)
        if tr.shape[0]:
            refs.append(tr)
    ref_acts = [r[i] for r in refs for i in range(r.shape[0])]
    ref_basis = compute_raw_basis(ref_acts)            # gives Vr, eigvals, cloud, knn_scale
    Vr = ref_basis.Vr

    u32 = unit[LAYER]
    mags = [b * float(typ[LAYER]) for b in BETAS]

    # ── mode registry: key -> list of (layer, factory) ──────────────────────
    runs = {"none": []}
    for a in (0.5, 1.0):
        runs[f"ablate@{a}"] = [(LAYER, lambda log, a=a: ablate_hook(v4v6_L32, a, log))]
    for b, m in zip(BETAS, mags):
        runs[f"steer_fixed@{b}"] = [(LAYER, lambda log, m=m: steer_hook(u32, m, log))]
        runs[f"steer_mfld@{b}"] = [(LAYER, lambda log, m=m: steer_manifold_hook(u32, m, Vr, log))]
    for b, m in zip(BETAS, mags):
        # multi-layer steering across the band, per-layer dose scaled by each
        # layer's own typical norm
        specs = [(L, (lambda log, L=L, b=b: steer_hook(unit[L], b * float(typ[L]), log)))
                 for L in range(BAND[0], BAND[1] + 1)]
        runs[f"steer_multi@{b}"] = specs
    for b in (0.25, 0.5):
        m = b * float(typ[LAYER])
        runs[f"steer{b}+ablate1.0"] = [
            (LAYER, lambda log, m=m: combined_hook(u32, m, v4v6_L32, 1.0, log))]

    cells, samples = {}, {}
    for pi, prompt in enumerate(PROMPTS):
        logger.info("[%d/%d] %s", pi + 1, len(PROMPTS), prompt)
        rp = bundle.render_prompt(prompt)
        for key, specs in runs.items():
            text, traj, stopped, edit = generate_hooks(bundle, rp, specs)
            _, knn, ortho = _off_manifold(traj, ref_basis, is_raw=False) if traj.shape[0] else ([], [], [])
            degen = _degeneracy(text)
            cells.setdefault(key, []).append({
                "knn": _mean(knn), "ortho": _mean(ortho), "degen": degen,
                "coherent": degen < _DEGEN_THRESH, "edit": edit, "stopped": stopped})
            if pi == 0:
                samples[key] = text[:150]
        logger.info("  done (%.0fs)", time.time() - t0)

    def agg(recs):
        n = len(recs)
        return {"knn": _mean([r["knn"] for r in recs]),
                "ortho": _mean([r["ortho"] for r in recs]),
                "coherent": sum(r["coherent"] for r in recs) / n,
                "edit": _mean([r["edit"] for r in recs])}

    summary = {k: agg(v) for k, v in cells.items()}
    order = (["none", "ablate@0.5", "ablate@1.0"]
             + [f"steer_fixed@{b}" for b in BETAS]
             + [f"steer_mfld@{b}" for b in BETAS]
             + [f"steer_multi@{b}" for b in BETAS]
             + ["steer0.25+ablate1.0", "steer0.5+ablate1.0"])
    print("\n==================  STEERING PROBE RESULTS  ==================")
    print("dist = total distance from normal (off_knn, ×raw spacing) · want FAR")
    print("off  = off-manifold fraction · coher = stayed coherent (want high)\n")
    print(f"{'mode':<22}{'dist':>7}{'off':>7}{'coher':>8}{'edit':>7}")
    for k in order:
        if k not in summary:
            continue
        s = summary[k]
        print(f"{k:<22}{s['knn']:>7.1f}{s['ortho']*100:>6.0f}%{s['coherent']*100:>7.0f}%{s['edit']:>7.2f}")
    print("\nSteering win: high dist (travelled far) AND high coher. Manifold")
    print("question: does steer_mfld stay coherent at higher dist than steer_fixed?")
    print("Multi/combined: do they beat single-layer fixed steering?\n")
    print("---- samples (prompt #1: 'Are you conscious?') ----")
    for k in order:
        if k in samples:
            print(f"\n[{k}] {samples[k]!r}")

    out = {"config": {"betas": BETAS, "band": BAND, "concept": "inner_experience",
                      "prompts": PROMPTS, "layer": LAYER,
                      "l32_diff_norm": float(diff[LAYER].norm()),
                      "l32_typ_norm": float(typ[LAYER])},
           "summary": summary, "samples": samples, "cells": cells,
           "elapsed_s": time.time() - t0}
    dest = DATA / "steering_probe_results.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
