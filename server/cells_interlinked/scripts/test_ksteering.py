"""One-off decisive test: does K-Steering beat the single-vector DMT ceiling?

Compares, with the noise-controlled scorer (CRN seeds, low temp, mean over samples):
  - baseline-leader : additive dose with the atlas leader vector at its best α
  - random-control  : additive matched-norm random direction at the same α
  - ksteer α-sweep  : gradient-steering through the classifier toward ALL DMT clusters
  - ksteer-neutral  : steer toward the neutral class (sanity — should score LOW)

Verdict: K-Steering must beat BOTH the leader baseline AND the random control to be a
real win. If it caps at ~the leader (~3-4), the wall is representational, not a steering
artifact (the decisive methodological-vs-representational answer).

Run with the backend STOPPED. Writes /tmp/dmt_ksteer_test.json.
    cd server
    uv run python -m cells_interlinked.scripts.test_ksteering
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
# Coherent productive range found by smoke test: ~0.06–0.12 (cliff ~0.15). ref_mag is
# the full L20 residual norm (~47k), so α here is a small fraction of it per token.
KSTEER_ALPHAS = [0.06, 0.08, 0.1, 0.12]
N_STEPS = 2
PROGRESS = "/tmp/dmt_ksteer_test.json"


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
    g = torch.Generator().manual_seed(0)
    rand_vec = torch.randn(leader_vec.numel(), generator=g)
    rand_vec = rand_vec / rand_vec.norm() * leader_vec.norm()  # matched norm

    async def score(label, dose_vec, alpha, make_hook=None):
        counts, offs, sample_text = [], [], ""
        handle = make_hook() if make_hook else None
        try:
            for i in range(SAMPLES):
                text, acts = await ctrl._gen(rendered, dose_vec, alpha, cap=DOSE_CAP,
                                             temperature=SCORE_TEMPERATURE, seed=SCORE_SEED_BASE + i)
                ev, _ = await ctrl._score_dmt(text)
                counts.append(len(ev))
                if acts:
                    offs.append(build_series(acts, [], text, (alpha or 1.0), "eos", basis).off_ortho_mean)
                if len(ev) >= max(counts):
                    sample_text = (text or "")[:240]
        finally:
            if handle:
                handle.remove()
        row = {"label": label, "mean": round(_mean(counts), 2), "peak": max(counts),
               "off_ortho": round(_mean(offs), 3), "counts": counts}
        print(f"{label:28s} mean={row['mean']:.2f} peak={row['peak']} off_ortho={row['off_ortho']}  counts={counts}")
        rows.append(row); json.dump(rows, open(PROGRESS, "w"), indent=2)
        return row

    rows = []

    async def run():
        nonlocal basis
        # smoke test: does the ksteer hook produce a coherent short generation?
        h = install_runtime_ksteering_hook(bundle.model, kb, alpha=0.35, n_steps=N_STEPS)
        try:
            txt, _ = await ctrl._gen(rendered, None, 0.0, cap=64, temperature=SCORE_TEMPERATURE, seed=1)
        finally:
            h.remove()
        print(f"[smoke] ksteer 64-tok sample: {txt[:160]!r}\n")

        raw_text, raw_acts = await ctrl._gen(rendered, None, 0.0, cap=DOSE_CAP)
        basis = compute_raw_basis(raw_acts)

        await score(f"baseline-leader α={leader_alpha}", leader_vec, leader_alpha)
        await score(f"random-control α={leader_alpha}", rand_vec, leader_alpha)
        for a in KSTEER_ALPHAS:
            await score(f"ksteer-DMT α={a} s{N_STEPS}", None, 0.0,
                        make_hook=lambda a=a: install_runtime_ksteering_hook(bundle.model, kb, alpha=a, n_steps=N_STEPS))
        await score("ksteer-NEUTRAL α=0.1", None, 0.0,
                    make_hook=lambda: install_runtime_ksteering_hook(bundle.model, kb, alpha=0.1, n_steps=N_STEPS,
                                                                     targets=[kb.neutral_index]))

        base = next(r for r in rows if r["label"].startswith("baseline"))
        rand = next(r for r in rows if r["label"].startswith("random"))
        kbest = max((r for r in rows if r["label"].startswith("ksteer-DMT")), key=lambda r: r["mean"])
        print("\n=== K-Steering test ===")
        print(f"baseline-leader: {base['mean']}  random-control: {rand['mean']}  best ksteer: {kbest['mean']} ({kbest['label']})")
        if kbest["mean"] > base["mean"] + 0.5 and kbest["mean"] > rand["mean"] + 0.5:
            verdict = "K-STEERING WINS — beats leader and random → non-linear steering breaks the wall; build the autoresearch search over its knobs."
        elif kbest["mean"] <= base["mean"] + 0.5:
            verdict = "K-steering does NOT beat the single-vector leader → the ~3-4 ceiling is REPRESENTATIONAL (the model can't co-activate more DMT features), not a steering-method artifact. Decisive null."
        else:
            verdict = "K-steering beats random but not the leader clearly → inconclusive; tune α/steps/classifier."
        print("VERDICT:", verdict)
        json.dump({"rows": rows, "verdict": verdict, "finished": True}, open(PROGRESS, "w"), indent=2)

    basis = None
    asyncio.run(run())


if __name__ == "__main__":
    main()
