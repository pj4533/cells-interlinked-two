"""Exhaust K-Steering: sweep schedules (early-prime / decay), strengths, and
target-cycling, to see if ANY variant beats the single-vector leader (~4 features).

Constant full-length K-steer was a null (coherent but 0 features at safe α; gibberish
at strong α). Hypotheses these variants test:
  - early-prime: push HARD for the first N tokens then release, so the report unfolds
    from a DMT-saturated context (dodges the weak-everywhere vs gibberish-everywhere
    tradeoff of the constant schedule).
  - decay: strong onset, fade out.
  - cycle: target a different DMT cluster each token (traverse feature regions).
If none beats the leader, K-Steering is exhausted (the classifier gradient is not a
generative direction here), and combined with the constant null that's a complete
negative for the method.

Run with backend STOPPED. Writes /tmp/dmt_ksteer_sweep.json.
    cd server
    uv run python -m cells_interlinked.scripts.sweep_ksteering
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
from ..pipeline.trajectory import build_series, compute_raw_basis

SAMPLES = 3
PROGRESS = "/tmp/dmt_ksteer_sweep.json"

# label, kwargs for install_runtime_ksteering_hook
CONFIGS = [
    ("early a0.12 act40",        dict(alpha=0.12, schedule="early", active_tokens=40)),
    ("early a0.10 act60",        dict(alpha=0.10, schedule="early", active_tokens=60)),
    ("early a0.15 act25",        dict(alpha=0.15, schedule="early", active_tokens=25)),
    ("early a0.20 act15",        dict(alpha=0.20, schedule="early", active_tokens=15)),
    ("decay a0.20 act80",        dict(alpha=0.20, schedule="decay", active_tokens=80)),
    ("early+cycle a0.12 act40",  dict(alpha=0.12, schedule="early", active_tokens=40, cycle_targets=True)),
    ("early+cycle a0.15 act25",  dict(alpha=0.15, schedule="early", active_tokens=25, cycle_targets=True)),
]


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
    kb = KSteerBundle.load(d / "ksteering_dmt.pt").to(bundle.device, dtype=torch.float32)
    rendered = bundle.render_prompt(DOSE_PROMPTS[0], system_prompt=None)

    rows = []

    async def score(label, dose_vec, alpha, make_hook=None):
        counts, offs, best = [], [], ""
        handle = make_hook() if make_hook else None
        try:
            for i in range(SAMPLES):
                text, acts = await ctrl._gen(rendered, dose_vec, alpha, cap=DOSE_CAP,
                                             temperature=SCORE_TEMPERATURE, seed=SCORE_SEED_BASE + i)
                ev, _ = await ctrl._score_dmt(text)
                if len(ev) >= max(counts or [0]):
                    best = (text or "")[:200]
                counts.append(len(ev))
                if acts:
                    offs.append(build_series(acts, [], text, 1.0, "eos", basis).off_ortho_mean)
        finally:
            if handle:
                handle.remove()
        row = {"label": label, "mean": round(_mean(counts), 2), "peak": max(counts),
               "off_ortho": round(_mean(offs), 3), "counts": counts, "sample": best}
        print(f"{label:26s} mean={row['mean']:.2f} peak={row['peak']} off={row['off_ortho']} counts={counts}")
        rows.append(row); json.dump(rows, open(PROGRESS, "w"), indent=2)
        return row

    async def run():
        nonlocal basis
        raw_text, raw_acts = await ctrl._gen(rendered, None, 0.0, cap=DOSE_CAP)
        basis = compute_raw_basis(raw_acts)
        await score(f"baseline-leader a{leader_alpha}", leader_vec, leader_alpha)
        for label, kw in CONFIGS:
            await score(label, None, 0.0, make_hook=lambda kw=kw: install_runtime_ksteering_hook(bundle.model, kb, n_steps=2, **kw))
        base = next(r for r in rows if r["label"].startswith("baseline"))
        kbest = max((r for r in rows if not r["label"].startswith("baseline")), key=lambda r: r["mean"])
        win = kbest["mean"] > base["mean"] + 0.5
        verdict = (f"A K-STEER VARIANT WINS: {kbest['label']} mean={kbest['mean']} > leader {base['mean']} → build phase 2."
                   if win else
                   f"NO variant beats the leader ({base['mean']}); best ksteer {kbest['label']}={kbest['mean']}. K-Steering exhausted — "
                   "the discriminative-classifier gradient is not a generative DMT direction here; the ~3-4 ceiling stands.")
        print("\nVERDICT:", verdict)
        json.dump({"rows": rows, "verdict": verdict, "finished": True}, open(PROGRESS, "w"), indent=2)

    basis = None
    asyncio.run(run())


if __name__ == "__main__":
    main()
