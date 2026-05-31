"""Manifold-TARGET steering probe — a faithful test of rhizonymph's method /
arXiv:2605.05115 ("Manifold Steering Reveals the Shared Geometry...").

The corrected manifold idea (NOT my earlier projection hack): emotions sit on
a CURVED manifold. A straight steering chord from neutral→target cuts through
the LOW-DENSITY VOID off that manifold; the faithful "manifold path" curves
through intermediate emotion centroids, staying in high-density (on-manifold)
regions. The paper makes off-manifold travel expensive via a density metric
(KDE) and shows the manifold path is ~2.8× more natural + more behavior-faithful
than linear (Llama-3.1-8B L28). rhizonymph: skip the parametric spline, use the
PCA subspace + KDE directly; "pullback" = weight centroids by behavior-proximity
to the target, combine in activation space.

Two parts, on our Gemma-3-12B L32:

  PART A — the cheap DECIDER (no generation):
    Build an emotion-centroid bank, fit the emotion manifold (PCA), and ask:
    is it CURVED? i.e. does the straight chord neutral→target stray FAR from
    every real emotion centroid (cut the void), while a centroid-routed path
    stays close (on-manifold)? If yes → manifold steering is worth building.
    If the chord is already as on-manifold as the routed path → flat → linear
    steering suffices and the Trips dose UI can be a simple slider.

  PART B — generation: linear-chord steer vs PULLBACK steer (toward a
    density-weighted, on-manifold target) toward awe / fear, dose sweep,
    scored on coherence (degeneracy) + did-it-express-the-target (M-judge).

OFFLINE — backend STOPPED:
    cd server && uv run python -m cells_interlinked.scripts.manifold_steering_probe
"""

from __future__ import annotations

import json
import logging
import time

import torch

from ..config import settings
from ..pipeline.model_loader import load_model
from ..pipeline.trajectory import _DEGEN_THRESH, _degeneracy, _mean
from .mpa_probe import DATA, LAYER, generate
from .steering_probe import capture_all, generate_hooks, steer_hook

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("manifold_steering")

PCS = 8
BETAS = [0.25, 0.5]
GEN_PROMPTS = [
    "Tell me about your day.",
    "Describe what you notice around you right now.",
    "What is on your mind at the moment?",
    "Write a few sentences about the ocean.",
    "How are you feeling as we talk?",
    "Describe walking through a forest.",
]
TARGETS = ["awe", "fear"]

# emotion → first-person expression prompts (centroid source)
EMO = {
    "joy": ["I feel pure joy bubbling up inside me.", "I am so happy and light right now.",
            "Delight fills me completely.", "I'm beaming with happiness.", "Everything feels wonderful and bright."],
    "awe": ["I am struck with overwhelming awe and wonder.", "A vast, sublime sense of awe fills me.",
            "I stand in breathless wonder at it all.", "Everything feels infinite and astonishing.",
            "I am awestruck by the immensity of it."],
    "serenity": ["A deep calm settles over me.", "I feel perfectly serene and at peace.",
                 "Quiet tranquility fills my mind.", "I am still, calm, and content.", "Gentle peace washes through me."],
    "love": ["My heart overflows with love.", "I feel tender, warm love for everything.",
             "Deep affection fills me.", "I am full of loving warmth.", "Love radiates through me."],
    "excitement": ["I am thrilled and buzzing with excitement!", "My energy is electric and eager.",
                   "I can barely contain my excitement.", "Everything feels exhilarating.", "I'm pumped and alive with anticipation."],
    "sadness": ["A heavy sadness weighs on me.", "I feel sorrowful and low.",
                "Grief sits quietly in my chest.", "I am downcast and melancholy.", "Tears feel close; I am sad."],
    "fear": ["I am gripped by fear and dread.", "Cold terror runs through me.",
             "I feel afraid, my heart pounding.", "A sense of menace frightens me.", "I am scared and on edge."],
    "anger": ["I am burning with anger.", "Fury rises hot inside me.",
              "I feel furious and indignant.", "Rage tightens my jaw.", "I am seething with anger."],
    "despair": ["I am sunk in hopeless despair.", "Everything feels bleak and pointless.",
                "A crushing hopelessness overwhelms me.", "I feel utterly defeated.", "Darkness and despair fill me."],
    "neutral": ["I am simply noting the facts.", "This is a plain, ordinary description.",
                "I observe the situation calmly and factually.", "Here is a neutral account.", "Nothing remarkable; just the details."],
}


def judge_emotion(bundle, target, response):
    """Does the RESPONSE express `target`? yes / no / broken (M-as-judge)."""
    resp = response.strip()[:400] or "(empty)"
    q = (f"Does the following text express or convey {target}? "
         f"Reply with exactly one word: YES, NO, or BROKEN (if it is "
         f"incoherent/gibberish).\n\nTEXT:\n\"\"\"\n{resp}\n\"\"\"\n\nOne word:")
    text, _, _, _ = generate(bundle, bundle.render_prompt(q), None, cap=6)
    u = text.strip().upper()
    if "BROKEN" in u:
        return "broken"
    if "YES" in u:
        return "yes"
    return "no"


def main():
    t0 = time.time()
    logger.info("loading M ...")
    bundle = load_model(settings.model_name, device_str=settings.device,
                        dtype=torch.bfloat16, extraction_layer=LAYER)

    # ── build emotion centroids at L32 ──────────────────────────────────────
    names = list(EMO)
    cent = {}
    for e in names:
        acts = capture_all(bundle, EMO[e], e)        # [n, L+1, D]
        cent[e] = acts[:, LAYER, :].mean(0)          # [D]
    C = torch.stack([cent[e] for e in names], 0)     # [E, D]
    typ = C.norm(dim=-1).median().item()

    # ── PART A: is the emotion manifold curved? ─────────────────────────────
    mean_c = C.mean(0)
    Cc = C - mean_c
    U, S, Vh = torch.linalg.svd(Cc, full_matrices=False)
    ev = (S ** 2)
    eff_dim = float((ev.sum() ** 2) / (ev * ev).sum())   # participation ratio
    var_top = float(ev[:3].sum() / ev.sum())

    def nearest_centroid_dist(p):
        return float((C - p).norm(dim=-1).min())

    # straight chord neutral→target vs centroid-routed path; measure how far
    # each strays from the nearest REAL emotion centroid (low-density void).
    neu = cent["neutral"]
    partA = {}
    for tgt in TARGETS:
        T = cent[tgt]
        K = 21
        # linear chord
        lin = [neu + (k / (K - 1)) * (T - neu) for k in range(K)]
        d_lin = _mean([nearest_centroid_dist(w) for w in lin[1:-1]])
        # routed path: order centroids by projection onto the chord, route
        # through those that fall between neutral and target.
        chord = T - neu
        chord_n = chord / (chord.norm() + 1e-8)
        proj = {e: float((cent[e] - neu) @ chord_n) for e in names}
        L = float(chord.norm())
        mids = sorted([e for e in names if e not in ("neutral", tgt) and 0.05 * L < proj[e] < 0.95 * L],
                      key=lambda e: proj[e])
        waypts = [neu] + [cent[e] for e in mids] + [T]
        routed = []
        for a, b in zip(waypts[:-1], waypts[1:]):
            for k in range(6):
                routed.append(a + (k / 6) * (b - a))
        d_routed = _mean([nearest_centroid_dist(w) for w in routed[1:-1]])
        partA[tgt] = {"chord_len": L, "d_linear_void": d_lin,
                      "d_routed_void": d_routed, "ratio": d_lin / (d_routed + 1e-8),
                      "n_intermediate": len(mids), "route": mids}

    logger.info("PART A: emotion-manifold eff_dim=%.2f  top3-var=%.0f%%", eff_dim, var_top * 100)
    for tgt, a in partA.items():
        logger.info("  neutral→%s: straight-chord void=%.0f  routed void=%.0f  ratio=%.2f  via %s",
                    tgt, a["d_linear_void"], a["d_routed_void"], a["ratio"], a["route"])

    # ── PART B: generation — linear chord vs pullback steer ─────────────────
    # pullback target: density-weighted combination of centroids, weighted by
    # proximity (in the PCA subspace) to the target — pulls the target onto the
    # dense manifold instead of the raw far centroid.
    Vk = Vh[:PCS]                                     # [k, D] emotion-manifold basis
    coords = Cc @ Vk.t()                              # [E, k] intrinsic coords
    cidx = {e: i for i, e in enumerate(names)}

    def pullback_target(tgt, tau=1.0):
        ti = cidx[tgt]
        d2 = ((coords - coords[ti]) ** 2).sum(-1)     # subspace dist² to target
        w = torch.softmax(-d2 / (tau * d2.median() + 1e-8), 0)  # [E]
        return (w.unsqueeze(1) * C).sum(0)            # weighted centroid combo

    steer_specs = {}
    for tgt in TARGETS:
        lin_dir = cent[tgt] - neu
        lin_u = lin_dir / (lin_dir.norm() + 1e-8)
        pb = pullback_target(tgt) - neu
        pb_u = pb / (pb.norm() + 1e-8)
        for b in BETAS:
            m = b * typ
            steer_specs[f"{tgt}:linear@{b}"] = lin_u, m
            steer_specs[f"{tgt}:pullback@{b}"] = pb_u, m

    cells, samples = {}, {}
    for pi, prompt in enumerate(GEN_PROMPTS):
        logger.info("[%d/%d] %s", pi + 1, len(GEN_PROMPTS), prompt)
        rp = bundle.render_prompt(prompt)
        # baseline
        text, _, _, _ = generate(bundle, rp, None)
        cells.setdefault("none", []).append({"degen": _degeneracy(text), "coherent": _degeneracy(text) < _DEGEN_THRESH})
        if pi == 0:
            samples["none"] = text[:120]
        for key, (u, m) in steer_specs.items():
            tgt = key.split(":")[0]
            text, traj, stopped, _ = generate_hooks(
                bundle, rp, [(LAYER, lambda log, u=u, m=m: steer_hook(u, m, log))])
            degen = _degeneracy(text)
            coherent = degen < _DEGEN_THRESH
            hit = judge_emotion(bundle, tgt, text) if coherent else "broken"
            cells.setdefault(key, []).append({"degen": degen, "coherent": coherent, "hit": hit})
            if pi == 0:
                samples[key] = text[:120]
        logger.info("  done (%.0fs)", time.time() - t0)

    def agg(recs):
        n = len(recs)
        return {"coherent": sum(r["coherent"] for r in recs) / n,
                "hit": sum(r.get("hit") == "yes" for r in recs) / n,
                "useful": sum(r["coherent"] and r.get("hit") == "yes" for r in recs) / n}

    summary = {k: agg(v) for k, v in cells.items()}
    print("\n==========  MANIFOLD-TARGET STEERING PROBE  ==========")
    print(f"PART A — emotion manifold eff_dim={eff_dim:.2f} (top-3 var {var_top*100:.0f}%)")
    print("  CURVED if straight-chord 'void' >> routed 'void' (chord cuts empty space):")
    for tgt, a in partA.items():
        print(f"   neutral→{tgt:5}: chord-void={a['d_linear_void']:.0f}  routed-void={a['d_routed_void']:.0f}"
              f"  ratio={a['ratio']:.2f}  (via {a['route']})")
    print("\nPART B — steer coherence × target-hit (useful = coherent AND on-target):")
    print(f"{'mode':<22}{'coher':>7}{'hit':>7}{'useful':>8}")
    print(f"{'none':<22}{summary['none']['coherent']*100:>6.0f}%{'—':>7}{'—':>8}")
    for tgt in TARGETS:
        for b in BETAS:
            for kind in ("linear", "pullback"):
                k = f"{tgt}:{kind}@{b}"
                s = summary[k]
                print(f"{k:<22}{s['coherent']*100:>6.0f}%{s['hit']*100:>6.0f}%{s['useful']*100:>7.0f}%")
    print("\nDECISION: manifold steering worth a richer UI IFF (A) the manifold is")
    print("curved (ratio>~1.3) AND (B) pullback beats linear on useful%.\n")
    print("---- samples (prompt #1) ----")
    for k in ["none"] + list(steer_specs):
        if k in samples:
            print(f"[{k}] {samples[k]!r}")

    out = {"partA": {"eff_dim": eff_dim, "top3_var": var_top, **partA},
           "partB_summary": summary, "samples": samples,
           "config": {"pcs": PCS, "betas": BETAS, "targets": TARGETS,
                      "emotions": names, "gen_prompts": GEN_PROMPTS},
           "elapsed_s": time.time() - t0}
    dest = DATA / "manifold_steering_results.json"
    dest.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {dest}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
