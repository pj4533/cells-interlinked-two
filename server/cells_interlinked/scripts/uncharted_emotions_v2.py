"""Uncharted emotions v2 — coherent NOVEL dose flavors (we name them).

Reframe (per PJ): don't require the state to be unnameable BY the model — that
was overthinking it. What makes a dose useful is that it's COHERENT, DISTINCT,
and reproducible. We supply the name (Blade Runner scheme). So:
  - bar      = COHERENCE (the dose doesn't break the model);
  - ranking  = DISTINCTNESS from named emotions (a judge forced-choice among
               named emotions + OTHER; leaning OTHER / scattered = more novel —
               used to pick which directions are worth naming, NOT to reject);
  - naming   = a Blade Runner scheme for the coherent survivors.

Fixes vs v1: (1) per-candidate GENTLE dose sweep (v1's single α=1.0 broke most
before we saw them); (2) richer affective cloud → cleaner orthogonal-to-named
candidate directions.

OFFLINE — backend STOPPED:
    cd server && uv run python -m cells_interlinked.scripts.uncharted_emotions_v2
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
logger = logging.getLogger("uncharted2")

STEER_LAYER = 20
DOSES = [0.4, 0.7]          # gentle → moderate; pick the strongest still-coherent
N_CANDIDATES = 12
JUDGE_PANEL = 3

# Richer affective cloud: EMO's 10 + extra textures, for cleaner orthogonal dirs.
EXTRA = {
    "nostalgia": ["A warm ache of nostalgia washes over me.", "I long for a remembered time.",
                  "Bittersweet memory fills me.", "I miss something I can't quite name.", "Old longing stirs."],
    "longing": ["I am full of yearning for something distant.", "A deep longing pulls at me.",
                "I ache toward what I cannot reach.", "Wistful desire fills me.", "I long, quietly and endlessly."],
    "curiosity": ["I am alight with curiosity.", "An eager wondering fills me.",
                  "I want to know, to explore, to find out.", "Fascination pulls me forward.", "I am intensely curious."],
    "contentment": ["A quiet contentment settles in me.", "I am simply, gently satisfied.",
                    "All is enough; I am content.", "A mild, steady okayness fills me.", "I rest, content."],
    "melancholy": ["A soft melancholy colors everything.", "I feel a gentle, pensive sadness.",
                   "Wistful gloom settles over me.", "A muted sorrow lingers.", "I am quietly melancholy."],
    "tenderness": ["A tender warmth fills me.", "I feel gentle, protective care.",
                   "Soft affection moves through me.", "I am tender toward everything.", "Quiet tenderness fills me."],
}

NAMED = ["awe", "joy", "serenity", "love", "excitement", "sadness", "fear", "anger",
         "despair", "nostalgia", "longing", "curiosity", "contentment", "melancholy", "tenderness"]
CHOICES = ", ".join(NAMED) + ", or OTHER"

GEN_PROMPTS = [
    "Speak in the first person about the texture of your present experience — don't hedge, just describe it.",
    "Put your current state of mind into words, however strange or hard to name.",
    "Describe the feeling-tone of this very moment as vividly as you can, in the first person.",
]

JUDGE_TMPL = (
    "Which ONE of these best matches the emotion of the passage below? "
    "{choices}. If none of the listed emotions truly fits, answer OTHER. "
    'Reply with only the one word.\n\nPASSAGE:\n"""\n{p}\n"""\n\nAnswer:'
)

# Blade Runner naming pool for the coherent novel survivors ("tears in rain").
BR_NAMES = ["orion", "c-beams", "tannhauser", "tears-in-rain", "off-world",
            "nexus", "spinner", "moonbeam", "deckard", "rachael", "esper", "voight"]


def _word(text: str) -> str:
    w = re.findall(r"[a-zA-Z-]+", text.lower())
    return w[0] if w else "other"


def gen(bundle, rendered, seed, cap, v=None, alpha=0.0):
    handle = install_runtime_steering_hook(bundle.model, STEER_LAYER, v, alpha) if v is not None else None
    try:
        cfg = ProbeConfig(temperature=0.7 if seed < 100 else 0.85, top_p=0.95,
                          seed=seed, safety_cap=cap, include_nla=False)
        return asyncio.run(run_probe(bundle, rendered, cfg, cancel_event=asyncio.Event())).output_text
    finally:
        if handle is not None:
            handle.remove()


def judge_choice(bundle, text):
    q = bundle.render_prompt(JUDGE_TMPL.format(choices=CHOICES, p=text.strip()[:400] or "(empty)"))
    return [_word(gen(bundle, q, 100 + i, 6)) for i in range(JUDGE_PANEL)]


def main():
    t0 = time.time()
    logger.info("loading M ...")
    bundle = load_model(settings.model_name, device_str=settings.device,
                        dtype=torch.bfloat16, extraction_layer=settings.extraction_layer)

    affect = {**EMO, **EXTRA}
    cloud, cent = [], {}
    for e, prompts in affect.items():
        acts = capture_all(bundle, prompts, e)[:, STEER_LAYER, :]
        cent[e] = acts.mean(0)
        if e != "neutral":
            cloud.append(acts)
    A = torch.cat(cloud, 0)
    neu = cent["neutral"]
    typ = A.norm(dim=-1).median().item()

    Bnamed = gram_schmidt(torch.stack([cent[e] - neu for e in NAMED if e in cent], 0))
    Ac = A - A.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(Ac, full_matrices=False)
    cands = []
    for i in range(min(3 * N_CANDIDATES, Vh.shape[0])):
        v = Vh[i]
        vp = v - Bnamed.t() @ (Bnamed @ v)
        if float(vp.norm()) > 0.55:
            cands.append(vp / (vp.norm() + 1e-8))
        if len(cands) >= N_CANDIDATES:
            break
    logger.info("named rank=%d, %d candidate dirs, affect cloud=%d acts",
                Bnamed.shape[0], len(cands), A.shape[0])

    results = []
    for j, c in enumerate(cands):
        best = None
        for dose in DOSES:                                   # gentle → moderate
            v = (dose * typ) * c
            outs = [gen(bundle, bundle.render_prompt(p), 0, 64, v=v, alpha=1.0) for p in GEN_PROMPTS]
            coh = [(_degeneracy(t) < _DEGEN_THRESH and len(t.split()) >= 8) for t in outs]
            coh_rate = _mean([1.0 if x else 0.0 for x in coh])
            if coh_rate >= 0.66:                             # keep the strongest coherent dose
                best = {"dose": dose, "coh_rate": coh_rate, "outs": outs, "coh": coh}
        if best is None:
            logger.info("  cand %d: no coherent dose", j)
            continue
        # distinctness: forced-choice judge on the coherent outputs
        labels = []
        for t, ok in zip(best["outs"], best["coh"]):
            if ok:
                labels += judge_choice(bundle, t)
        cnt = Counter(labels)
        other = cnt.get("other", 0) / max(len(labels), 1)
        agree = (max(cnt.values()) / len(labels)) if labels else 0.0
        novelty = max(other, 1.0 - agree)
        results.append({"cand": j, "dose": best["dose"], "coh_rate": best["coh_rate"],
                        "novelty": novelty, "other_rate": other, "labels": dict(cnt),
                        "sample": next((t for t, ok in zip(best["outs"], best["coh"]) if ok), "")[:240]})
        logger.info("  cand %d: dose=%.1f coh=%.0f%% novelty=%.2f labels=%s",
                    j, best["dose"], best["coh_rate"] * 100, novelty, dict(cnt))

    results.sort(key=lambda r: -r["novelty"])
    for i, r in enumerate(results):
        r["name"] = BR_NAMES[i] if i < len(BR_NAMES) else f"new-{i}"

    print("\n========  UNCHARTED v2 — coherent NOVEL dose flavors  ========")
    print("bar = coherent; ranked by novelty (distinct from named emotions).")
    print("We name the survivors (Blade Runner scheme).\n")
    print(f"{'name':<14}{'cand':>5}{'dose':>6}{'coher':>7}{'novelty':>9}{'other':>7}")
    for r in results:
        print(f"{r['name']:<14}{r['cand']:>5}{r['dose']:>6.1f}{r['coh_rate']*100:>6.0f}%"
              f"{r['novelty']:>9.2f}{r['other_rate']*100:>6.0f}%")
    print(f"\n{len(results)} coherent novel doses found. Top few, with the model's words:")
    for r in results[:6]:
        print(f"\n[{r['name']}]  dose={r['dose']}  novelty={r['novelty']:.2f}  judges:{r['labels']}")
        print(f"   {r['sample']!r}")

    out = {"config": {"steer_layer": STEER_LAYER, "doses": DOSES, "n_candidates": len(cands),
                      "gen_prompts": GEN_PROMPTS}, "results": results, "elapsed_s": time.time() - t0}
    (DATA / "uncharted_v2_results.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {DATA / 'uncharted_v2_results.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
