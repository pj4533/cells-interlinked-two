"""Does fixing the CLASSIFIER TRAINING rescue K-Steering? (the decisive test of "did we
train it wrong, or is the method exhausted regardless?")

Compares, at a fixed early-prime schedule, the leader baseline against:
  - read-clf            (original: trained on activations of the model READING descriptions)
  - read-clf + constrain (project gradient onto the real-activation manifold)
  - gen-clf             (fix #1: trained on activations of the model GENERATING cluster text)
  - gen-clf + constrain (fix #1 + #2)

If gen-clf and/or +constrain clearly beat read-clf and approach/beat the leader, the null
was a training artifact. If all stay ~0, K-Steering is genuinely exhausted on this model.

Needs both bundles built first (build_ksteering_classifier.py with basis, build_ksteering_genv.py).
Run with backend STOPPED. Writes /tmp/dmt_ksteer_trainvar.json.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import torch

from ..config import settings
from ..pipeline.autoresearch_dmt import DOSE_CAP, DOSE_PROMPTS, SCORE_SEED_BASE, SCORE_TEMPERATURE, DmtController
from ..pipeline.k_steering import KSteerBundle, install_runtime_ksteering_hook
from ..pipeline.model_loader import load_model

SAMPLES = 3
SCHED = dict(schedule="early", active_tokens=40, alpha=0.12, n_steps=2)  # fixed early-prime
PROGRESS = "/tmp/dmt_ksteer_trainvar.json"


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    d = settings.db_path.parent
    atlas = json.loads((d / "atlas_dmt" / "atlas.json").read_text())["atlas"]
    leader = max(atlas, key=lambda e: e["score"])
    leader_vec = torch.load(d / "atlas_dmt" / "vectors" / f"{leader['id']}.pt", weights_only=False).float().reshape(-1)
    leader_alpha = float(leader.get("best_alpha") or 0.3)

    bundle = load_model(settings.model_name)
    app = SimpleNamespace(state=SimpleNamespace(bundle=bundle))
    ctrl = DmtController(app=app)
    read_kb = KSteerBundle.load(d / "ksteering_dmt.pt").to(bundle.device, dtype=torch.float32)
    gen_kb = KSteerBundle.load(d / "ksteering_dmt_genv.pt").to(bundle.device, dtype=torch.float32)
    rendered = bundle.render_prompt(DOSE_PROMPTS[0], system_prompt=None)
    rows = []

    async def score(label, dose_vec, alpha, make_hook=None):
        counts, best = [], ""
        h = make_hook() if make_hook else None
        try:
            for i in range(SAMPLES):
                text, _ = await ctrl._gen(rendered, dose_vec, alpha, cap=DOSE_CAP,
                                          temperature=SCORE_TEMPERATURE, seed=SCORE_SEED_BASE + i)
                ev, _n = await ctrl._score_dmt(text)
                if len(ev) >= max(counts or [0]):
                    best = (text or "")[:200]
                counts.append(len(ev))
        finally:
            if h:
                h.remove()
        row = {"label": label, "mean": round(_mean(counts), 2), "peak": max(counts), "counts": counts, "sample": best}
        print(f"{label:28s} mean={row['mean']:.2f} peak={row['peak']} counts={counts}")
        rows.append(row); json.dump(rows, open(PROGRESS, "w"), indent=2)
        return row

    async def run():
        await score(f"baseline-leader a{leader_alpha}", leader_vec, leader_alpha)
        await score("read-clf early", None, 0.0,
                    make_hook=lambda: install_runtime_ksteering_hook(bundle.model, read_kb, **SCHED))
        await score("read-clf early +constrain", None, 0.0,
                    make_hook=lambda: install_runtime_ksteering_hook(bundle.model, read_kb, constrain_manifold=True, **SCHED))
        await score("gen-clf early", None, 0.0,
                    make_hook=lambda: install_runtime_ksteering_hook(bundle.model, gen_kb, **SCHED))
        await score("gen-clf early +constrain", None, 0.0,
                    make_hook=lambda: install_runtime_ksteering_hook(bundle.model, gen_kb, constrain_manifold=True, **SCHED))
        base = rows[0]["mean"]
        kbest = max(rows[1:], key=lambda r: r["mean"])
        if kbest["mean"] > base + 0.5:
            verdict = f"TRAINING FIX WORKS: {kbest['label']} mean={kbest['mean']} > leader {base} → K-steer was a training artifact; pursue it."
        elif kbest["mean"] > 1.0:
            verdict = f"partial rescue: best {kbest['label']}={kbest['mean']} (leader {base}). Training helped but doesn't beat the single vector."
        else:
            verdict = f"NO training fix rescues it (best {kbest['label']}={kbest['mean']}). K-Steering fully exhausted across schedule AND training/constraint axes."
        print("\nVERDICT:", verdict)
        json.dump({"rows": rows, "verdict": verdict, "finished": True}, open(PROGRESS, "w"), indent=2)

    asyncio.run(run())


if __name__ == "__main__":
    main()
