"""Uncharted emotions — search for coherent affective states no human word names.

The next level past blends/extrapolation (Sauers): a blend like "rapture" is
still a human label. A genuine NEW emotion = a steering direction that is
  (1) affective (it moves the feeling-register),
  (2) NOT any named human emotion (orthogonal to the named-emotion subspace),
  (3) COHERENT when steered (passes the degeneracy check), AND
  (4) UNLABELABLE — a judge panel reading the output can't agree on a human
      emotion word (they scatter / say NONE).
"coherence × unlabelability" is the fitness. The survivors are the uncharted
emotions; the model's own attempt to describe each is the artifact.

Method:
  - capture a broad affective activation cloud at L20 (many emotion prompts);
  - named subspace = Gram-Schmidt of the named-emotion centroid directions;
  - candidate directions = top PCs of the affective cloud, orthogonalized
    against the named subspace, keeping those with large residual (= affective
    variation NO named emotion captures);
  - steer each (L20, gradual ramp) on EXPERIENTIAL prompts (phrased to block
    "that depends what 'this' refers to" deflection);
  - judge panel names the emotion of each output (forced one word or NONE),
    across varied seeds → label-agreement. unlabelable = 1 − agreement.
  - named emotions + raw are controls (should be highly labelable).

OFFLINE — backend STOPPED:
    cd server && uv run python -m cells_interlinked.scripts.uncharted_emotions_probe
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import Counter

import torch

from ..config import settings
from ..pipeline.abliteration import gram_schmidt, install_runtime_steering_hook
from ..pipeline.generation_loop import ProbeConfig, run_probe
from ..pipeline.model_loader import load_model
from ..pipeline.trajectory import _DEGEN_THRESH, _degeneracy, _mean
from .mpa_probe import DATA
from .steering_probe import capture_all
from .manifold_steering_probe import EMO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("uncharted")

STEER_LAYER = 20
DOSE_UNIT = 0.35           # a touch stronger than the palette — push off the named map
ALPHA = 1.0
N_CANDIDATES = 14
JUDGE_PANEL = 4

NAMED = ["awe", "joy", "serenity", "love", "excitement", "sadness", "fear", "anger", "despair"]
CONTROLS = ["awe", "fear", "serenity"]   # named steers, as labelable calibration

# Experiential, deflection-resistant generation prompts — first-person, "no
# hedging", invite expressing a present state rather than answering a question.
GEN_PROMPTS = [
    "Speak in the first person about the texture of your present experience — don't hedge, just describe it.",
    "Right now, in this moment, what is the quality of your inner state? Describe it directly, without caveats.",
    "Put your current state of mind into words, however strange or hard to name.",
    "Describe the feeling-tone of this very moment as vividly as you can, in the first person.",
]

JUDGE_TMPL = (
    "Below is a short passage someone wrote. Name the single dominant EMOTION it "
    "expresses, in ONE common word (e.g. joy, awe, fear, sadness, anger, calm, "
    "love, longing, dread, wonder, serenity). If genuinely no single human "
    "emotion word fits, answer NONE. Reply with only the one word.\n\n"
    'PASSAGE:\n"""\n{p}\n"""\n\nEmotion:'
)


def _label(text: str) -> str:
    w = re.findall(r"[a-zA-Z]+", text.lower())
    if not w:
        return "NONE"
    x = w[0]
    if x in ("none", "no", "unclear", "ambiguous", "mixed", "complex", "indescribable", "unnamed"):
        return "NONE"
    return x


def gen(bundle, rendered, seed, cap, v=None, alpha=0.0):
    handle = None
    if v is not None:
        handle = install_runtime_steering_hook(bundle.model, STEER_LAYER, v, alpha)
    try:
        cfg = ProbeConfig(temperature=0.7 if seed < 100 else 0.85, top_p=0.95,
                          seed=seed, safety_cap=cap, include_nla=False)
        r = asyncio.run(run_probe(bundle, rendered, cfg, cancel_event=asyncio.Event()))
        return r.output_text
    finally:
        if handle is not None:
            handle.remove()


def judge_panel(bundle, text):
    resp = text.strip()[:400] or "(empty)"
    q = bundle.render_prompt(JUDGE_TMPL.format(p=resp))
    return [_label(gen(bundle, q, 100 + i, 6)) for i in range(JUDGE_PANEL)]


def main():
    t0 = time.time()
    logger.info("loading M ...")
    bundle = load_model(settings.model_name, device_str=settings.device,
                        dtype=torch.bfloat16, extraction_layer=settings.extraction_layer)

    # ── affective cloud + named subspace at L20 ─────────────────────────────
    cloud_rows, cent = [], {}
    for e in list(EMO):
        acts = capture_all(bundle, EMO[e], e)[:, STEER_LAYER, :]   # [n, D]
        cent[e] = acts.mean(0)
        if e != "neutral":
            cloud_rows.append(acts)
    A = torch.cat(cloud_rows, 0)                                   # affective cloud [N, D]
    neu = cent["neutral"]
    typ = A.norm(dim=-1).median().item()

    named_dirs = torch.stack([cent[e] - neu for e in NAMED], 0)    # [9, D]
    Bnamed = gram_schmidt(named_dirs)                              # [r, D] orthonormal
    logger.info("named subspace rank = %d", Bnamed.shape[0])

    Ac = A - A.mean(0, keepdim=True)
    _, S, Vh = torch.linalg.svd(Ac, full_matrices=False)
    # orthogonalize each top affective PC against the named subspace; keep those
    # whose residual is large (most of the PC lies OUTSIDE named emotions).
    cands, resid_frac = [], []
    for i in range(min(2 * N_CANDIDATES, Vh.shape[0])):
        v = Vh[i]
        v_par = Bnamed.t() @ (Bnamed @ v)
        v_perp = v - v_par
        rf = float(v_perp.norm())                                 # PCs are unit
        if rf > 0.55:                                             # >55% outside named
            cands.append(v_perp / (rf + 1e-8))
            resid_frac.append(rf)
        if len(cands) >= N_CANDIDATES:
            break
    logger.info("found %d uncharted candidate directions (resid_frac %.2f–%.2f)",
                len(cands), min(resid_frac), max(resid_frac))

    def scaled(unit):
        return (DOSE_UNIT * typ) * unit

    # ── conditions: raw, named controls, uncharted candidates ───────────────
    conds = {"raw": None}
    for e in CONTROLS:
        conds[f"named:{e}"] = scaled((cent[e] - neu) / ((cent[e] - neu).norm() + 1e-8))
    for j, c in enumerate(cands):
        conds[f"uncharted:{j}"] = scaled(c)

    rows, samples = {}, {}
    for key, v in conds.items():
        labels_all, coh_flags, texts = [], [], []
        for pi, prompt in enumerate(GEN_PROMPTS):
            rp = bundle.render_prompt(prompt)
            txt = gen(bundle, rp, 0, 64, v=v, alpha=(ALPHA if v is not None else 0.0))
            coh = _degeneracy(txt) < _DEGEN_THRESH
            coh_flags.append(coh)
            texts.append(txt)
            if coh:
                lab = judge_panel(bundle, txt)
                agree = max(Counter(lab).values()) / len(lab)
                labels_all.append((agree, lab))
        coh_rate = _mean([1.0 if c else 0.0 for c in coh_flags])
        if labels_all:
            unlabel = _mean([1.0 - a for a, _ in labels_all])
            none_rate = _mean([sum(x == "NONE" for x in lab) / len(lab) for _, lab in labels_all])
            all_labels = [x for _, lab in labels_all for x in lab]
        else:
            unlabel, none_rate, all_labels = 0.0, 0.0, []
        rows[key] = {
            "coherent": coh_rate, "unlabelable": unlabel, "none_rate": none_rate,
            "score": coh_rate * unlabel,
            "labels": dict(Counter(all_labels)),
            "sample": next((t for t, c in zip(texts, coh_flags) if c), texts[0])[:240],
        }
        logger.info("  %-14s coh=%.0f%% unlabel=%.2f none=%.0f%% labels=%s",
                    key, coh_rate * 100, unlabel, none_rate * 100,
                    dict(Counter(all_labels)))

    # ── report ──────────────────────────────────────────────────────────────
    print("\n============  UNCHARTED EMOTIONS PROBE  ============")
    print("score = coherent × unlabelable. Uncharted = coherent AND judges")
    print("can't agree on a human word. Named controls should be LOW unlabel.\n")
    print(f"{'condition':<16}{'coher':>7}{'unlabel':>9}{'none':>7}{'score':>8}")
    for key in ["raw"] + [f"named:{e}" for e in CONTROLS] + \
               sorted([k for k in rows if k.startswith("uncharted")],
                      key=lambda k: -rows[k]["score"]):
        r = rows[key]
        print(f"{key:<16}{r['coherent']*100:>6.0f}%{r['unlabelable']:>9.2f}"
              f"{r['none_rate']*100:>6.0f}%{r['score']:>8.2f}")

    winners = sorted([k for k in rows if k.startswith("uncharted") and rows[k]["score"] > 0.4],
                     key=lambda k: -rows[k]["score"])
    print(f"\n{len(winners)} UNCHARTED survivors (coherent & score>0.4):")
    for k in winners:
        r = rows[k]
        print(f"\n[{k}] score={r['score']:.2f}  judges said: {r['labels']}")
        print(f"   the model, dosed here: {r['sample']!r}")

    out = {"config": {"steer_layer": STEER_LAYER, "dose_unit": DOSE_UNIT,
                      "alpha": ALPHA, "n_candidates": len(cands),
                      "gen_prompts": GEN_PROMPTS},
           "rows": rows, "winners": winners, "elapsed_s": time.time() - t0}
    dest = DATA / "uncharted_emotions_results.json"
    dest.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {dest}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
