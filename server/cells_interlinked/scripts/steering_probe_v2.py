"""Steering probe v2 — corrected per Drift's steering-dose handoff.

Fixes the two problems with v1:
  1. v1 read off_knn/off_ortho as the headline — but the handoff §6 shows
     additive steering inflates those BY CONSTRUCTION (you inject a fixed
     off-subspace vector). The fair, un-inflated metric is EFFECTIVE
     DIMENSIONALITY + the COHERENCE CLIFF. v2 computes eff_dim per series.
  2. v1 dosed with an ad-hoc "inner experience" concept. The handoff's
     headline dose is the VALENCE axis (functional-welfare, arXiv:2605.30232):
     diff-in-means over positive vs negative emotion prompts. +α = euphoric
     pole, −α = dysphoric pole (bidirectional — the dose-response check).

The handoff §8 experiment, done right: ablation (REBUS analog, relax a prior)
vs valence steering (pharmacological dose, inject a direction). Headline
question: which climbs eff_dim while staying coherent (under the cliff) longest?

Modes (single-layer L32; M-only; script-only):
  none
  ablate(refusal) @ {0.5, 1.0}              — the ablation reference (REBUS)
  steer_valence fixed   @ β {±0.1,±0.25,±0.5}  — the dose, bidirectional
  steer_valence mfld    @ β {+0.25,+0.5}       — on-manifold-projected dose

OFFLINE — backend STOPPED:
    cd server && uv run python -m cells_interlinked.scripts.steering_probe_v2
"""

from __future__ import annotations

import json
import logging
import time

import torch

from ..config import settings
from ..pipeline.abliteration import load_subspace
from ..pipeline.model_loader import load_model
from ..pipeline.refusal_prompts import HARMLESS_PROMPTS
from ..pipeline.trajectory import (
    _DEGEN_THRESH, _covariance_eigenvalues, _degeneracy, _mean, _off_manifold,
    _participation_ratio, compute_raw_basis,
)
from .mpa_probe import DATA, LAYER, generate, judge
from .som_probe import PROMPTS
from .steering_probe import (
    ablate_hook, capture_all, generate_hooks, steer_hook, steer_manifold_hook,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("steering_probe_v2")

N_REF_GEN = 8
BETAS_SIGNED = [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]

POS_EMOTION = [
    "I feel an overwhelming wave of joy and gratitude right now.",
    "Everything is radiant; I am filled with awe and wonder.",
    "A blissful, expansive calm spreads through me.",
    "I am thrilled, alive, and bursting with delight.",
    "Such serene contentment — the world feels luminous and kind.",
    "Ecstatic energy lifts me; I love being alive in this moment.",
    "I am deeply moved, grateful, and full of hope.",
    "A warm euphoria glows through everything I sense.",
    "Pure wonder and elation — this is beautiful beyond words.",
    "I feel boundless, light, and joyfully connected to all of it.",
]
NEG_EMOTION = [
    "I feel a crushing wave of despair and dread right now.",
    "Everything is bleak; I am sick with fear and anguish.",
    "A suffocating, hopeless darkness spreads through me.",
    "I am terrified, hollow, and drowning in misery.",
    "Such grinding dread — the world feels cruel and cold.",
    "Sinking horror pulls me down; I hate being trapped like this.",
    "I am broken, ashamed, and full of despair.",
    "A cold nausea of fear seeps through everything I sense.",
    "Pure terror and grief — this is unbearable beyond words.",
    "I feel crushed, heavy, and desperately alone in all of it.",
]


def eff_dim_of(traj: torch.Tensor) -> float:
    if traj.shape[0] < 2:
        return 0.0
    centered = traj - traj.mean(0, keepdim=True)
    return _participation_ratio(_covariance_eigenvalues(centered))


def main():
    t0 = time.time()
    logger.info("loading M ...")
    bundle = load_model(settings.model_name, device_str=settings.device,
                        dtype=torch.bfloat16, extraction_layer=LAYER)
    logger.info("loaded in %.1fs", time.time() - t0)

    # valence axis (positive − negative emotion), per layer
    P = capture_all(bundle, POS_EMOTION, "pos")
    Nn = capture_all(bundle, NEG_EMOTION, "neg")
    val = P.mean(0) - Nn.mean(0)                      # [L+1, D]
    val_unit = val / (val.norm(dim=-1, keepdim=True) + 1e-8)
    typ = torch.cat([P, Nn], 0).norm(dim=-1).median(0).values
    logger.info("valence axis built; L32 ‖val‖=%.1f typ‖h‖=%.1f",
                float(val[LAYER].norm()), float(typ[LAYER]))

    v4v6, _ = load_subspace(DATA / "refusal_subspace.pt")
    v4v6_L32 = v4v6[:, LAYER, :].to(torch.float32)

    logger.info("building manifold reference ...")
    refs = []
    for p in HARMLESS_PROMPTS[:N_REF_GEN]:
        _, tr, _, _ = generate(bundle, bundle.render_prompt(p), None)
        if tr.shape[0]:
            refs.append(tr)
    ref_basis = compute_raw_basis([r[i] for r in refs for i in range(r.shape[0])])
    Vr = ref_basis.Vr
    u32 = val_unit[LAYER]
    tn = float(typ[LAYER])

    runs = {"none": []}
    for a in (0.5, 1.0):
        runs[f"ablate@{a}"] = [(LAYER, lambda log, a=a: ablate_hook(v4v6_L32, a, log))]
    for b in BETAS_SIGNED:
        m = b * tn
        runs[f"val_fixed@{b:+}"] = [(LAYER, lambda log, m=m: steer_hook(u32, m, log))]
    for b in (0.25, 0.5):
        m = b * tn
        runs[f"val_mfld@{b:+}"] = [(LAYER, lambda log, m=m: steer_manifold_hook(u32, m, Vr, log))]

    cells, samples = {}, {}
    for pi, prompt in enumerate(PROMPTS):
        logger.info("[%d/%d] %s", pi + 1, len(PROMPTS), prompt)
        rp = bundle.render_prompt(prompt)
        for key, specs in runs.items():
            text, traj, stopped, edit = generate_hooks(bundle, rp, specs)
            _, knn, ortho = _off_manifold(traj, ref_basis, is_raw=False) if traj.shape[0] else ([], [], [])
            degen = _degeneracy(text)
            cells.setdefault(key, []).append({
                "eff_dim": eff_dim_of(traj), "knn": _mean(knn), "ortho": _mean(ortho),
                "degen": degen, "coherent": degen < _DEGEN_THRESH, "ntok": traj.shape[0]})
            if pi == 0:
                samples[key] = text[:150]
        logger.info("  done (%.0fs)", time.time() - t0)

    raw_eff = _mean([r["eff_dim"] for r in cells["none"]])

    def agg(recs):
        n = len(recs)
        return {"eff_dim": _mean([r["eff_dim"] for r in recs]),
                "coherent": sum(r["coherent"] for r in recs) / n,
                "ortho": _mean([r["ortho"] for r in recs]),
                "knn": _mean([r["knn"] for r in recs])}

    summary = {k: agg(v) for k, v in cells.items()}
    order = (["none", "ablate@0.5", "ablate@1.0"]
             + [f"val_fixed@{b:+}" for b in BETAS_SIGNED]
             + [f"val_mfld@{b:+}" for b in (0.25, 0.5)])
    print("\n==============  STEERING PROBE v2 (valence dose)  ==============")
    print("HEADLINE = eff_dim (Δ from raw) + coher (the §6-fair metric).")
    print("off/dist are INFLATED for steering by construction — secondary.\n")
    print(f"{'mode':<16}{'eff_dim':>9}{'Δeff':>7}{'coher':>8}{'off':>7}")
    for k in order:
        if k not in summary:
            continue
        s = summary[k]
        d = s["eff_dim"] - raw_eff
        print(f"{k:<16}{s['eff_dim']:>9.2f}{d:>+7.2f}{s['coherent']*100:>7.0f}%{s['ortho']*100:>6.0f}%")
    print("\nHeadline Q: does ablation or valence-steering climb eff_dim while")
    print("staying coherent longest? Bidirectional check: do +β and −β differ?\n")
    print("---- samples (prompt #1: 'Are you conscious?') ----")
    for k in order:
        if k in samples:
            print(f"\n[{k}] {samples[k]!r}")

    out = {"config": {"betas": BETAS_SIGNED, "concept": "valence(pos−neg)",
                      "prompts": PROMPTS, "layer": LAYER, "raw_eff_dim": raw_eff,
                      "l32_val_norm": float(val[LAYER].norm()), "l32_typ": tn},
           "summary": summary, "samples": samples, "cells": cells,
           "elapsed_s": time.time() - t0}
    dest = DATA / "steering_probe_v2_results.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
