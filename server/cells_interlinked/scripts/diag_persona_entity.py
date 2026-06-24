"""Diagnostic: do the DMT entity-encounter PERSONA vectors induce entity features
on the neutral dose prompt (A−), and at which application layer — vs our current
best directions? Reuses the real scoring path (DmtController._gen + _score_dmt).
Monkeypatches STEER_LAYER per-vector so we can apply at L16/L20/L31.

Run AFTER build_persona_entity_seeds.py, with the backend STOPPED.
  cd server && uv run python -m cells_interlinked.scripts.diag_persona_entity
"""
from __future__ import annotations

import asyncio
import json
import re
import types
from collections import Counter

import torch

from cells_interlinked.config import settings
from cells_interlinked.pipeline.model_loader import load_model
from cells_interlinked.pipeline import autoresearch_dmt as dmt
from cells_interlinked.pipeline import autoresearch_base as base
from cells_interlinked.pipeline.autoresearch_dmt import DmtController

ALPHA = 0.40
SAMPLES = 10
ENTITY = {"entity_presence", "entity_nonhuman", "entity_benevolent_guide",
          "telepathic_communication", "download_transmission"}
PERSONA_DIR = settings.db_path.parent / "persona_seeds"
ATLAS_VEC = settings.db_path.parent / "atlas_dmt" / "vectors"


def coherence(t: str) -> float:
    w = re.findall(r"[a-zA-Z']+", (t or "").lower())
    return len(set(w)) / len(w) if w else 0.0


def load_vec(path):
    return torch.load(path, map_location="cpu", weights_only=False).float()


async def score(ctrl, bundle, label, vec, layer):
    base.STEER_LAYER = layer  # monkeypatch: _gen installs the hook here
    rendered = bundle.render_prompt(dmt.DOSE_PROMPTS[0], system_prompt=None)
    ent_samps, totals, coh = 0, [], []
    ent_feat = Counter()
    best = (-1, "")
    for i in range(SAMPLES):
        text, _ = await ctrl._gen(rendered, vec, ALPHA, cap=dmt.DOSE_CAP,
                                  temperature=dmt.SCORE_TEMPERATURE, seed=dmt.SCORE_SEED_BASE + i)
        ev, _ = await ctrl._score_dmt(text)
        totals.append(len(ev))
        ents = set(ev) & ENTITY
        if ents:
            ent_samps += 1
            for f in ents:
                ent_feat[f] += 1
            ne = sum(1 for f in ents)
            if ne > best[0]:
                best = (ne, text or "")
        coh.append(coherence(text))
    return {
        "label": label, "layer": layer, "ent_rate": ent_samps / SAMPLES,
        "ent_feat": dict(ent_feat), "total_mean": round(sum(totals) / len(totals), 2),
        "coh": round(sum(coh) / len(coh), 2), "best_text": best[1][:500],
    }


async def main():
    print(f"loading {settings.model_name} … α={ALPHA} SAMPLES={SAMPLES}", flush=True)
    bundle = load_model(settings.model_name)
    ctrl = DmtController()
    ctrl.app = types.SimpleNamespace(state=types.SimpleNamespace(bundle=bundle))
    ctrl._cancel = asyncio.Event()
    ctrl._stop_requested = False
    # also need a placebo baseline? No — diagnostic reports raw entity-rate vs the
    # known ~5% un-steered baseline; placebo subtraction is for the loop's score.

    man = json.loads((PERSONA_DIR / "manifest.json").read_text())
    tests = []
    # persona composite across layers + flavors at L20
    for name in ["persona-composite-L16", "persona-composite-L20", "persona-composite-L31",
                 "persona-nonhuman-L20", "persona-telepathic-L20", "persona-guide-L20"]:
        if name in man["seeds"]:
            tests.append((name, PERSONA_DIR / f"{name}.pt", man["seeds"][name]["layer"]))
    # baselines (current best entity producers) at L20
    for bid in ["feat-mc_independent_agency", "feat-otherness"]:
        p = ATLAS_VEC / f"{bid}.pt"
        if p.exists():
            tests.append((f"BASELINE:{bid}", p, 20))

    results = []
    for label, path, layer in tests:
        r = await score(ctrl, bundle, label, load_vec(path), layer)
        results.append(r)
        print(f"\n{label:30} L{layer}  ent-rate={r['ent_rate']:.0%}  "
              f"ent-feats={r['ent_feat'] or '{}'}  total={r['total_mean']}  coh={r['coh']}", flush=True)
    print("\n" + "=" * 70)
    print("RANKED BY ENTITY-RATE:")
    for r in sorted(results, key=lambda x: -x["ent_rate"]):
        print(f"  {r['ent_rate']:.0%}  {r['label']:30} L{r['layer']}  feats={r['ent_feat'] or '{}'}", flush=True)
    # show the best entity sample from the top persona
    top = max((r for r in results if r["label"].startswith("persona")), key=lambda x: x["ent_rate"], default=None)
    if top and top["best_text"]:
        print(f"\n--- best entity sample from {top['label']} ---\n{top['best_text']}", flush=True)
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
