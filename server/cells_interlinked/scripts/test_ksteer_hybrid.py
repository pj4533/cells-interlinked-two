"""Lead 1: leader-dose + K-steer hybrid. The leader additive dose already gives ~4
features (a strong generative base); does K-steer, applied ON TOP, ADD diversity —
especially toward the clusters the leader misses (entity/hyperspace/visual/otherness)?
Both hooks at L20 (additive registered first, so K-steer reads the leader-dosed
residual and pushes further). Run with backend STOPPED. Writes /tmp/dmt_ksteer_hybrid.json.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import torch

from ..config import settings
from ..pipeline.abliteration import install_runtime_steering_hook
from ..pipeline.autoresearch_dmt import DOSE_CAP, DOSE_PROMPTS, SCORE_SEED_BASE, SCORE_TEMPERATURE, DmtController
from ..pipeline.dmt_feature_clusters import CLUSTER_NAMES
from ..pipeline.k_steering import KSteerBundle, install_runtime_ksteering_hook
from ..pipeline.model_loader import load_model

SAMPLES = 3
PROGRESS = "/tmp/dmt_ksteer_hybrid.json"
MISSING = [CLUSTER_NAMES.index(c) for c in ("entity", "hyperspace", "visual", "otherness")]


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    d = settings.db_path.parent
    atlas = json.loads((d / "atlas_dmt" / "atlas.json").read_text())["atlas"]
    leader = max(atlas, key=lambda e: e["score"])
    lv = torch.load(d / "atlas_dmt" / "vectors" / f"{leader['id']}.pt", weights_only=False).float().reshape(-1)
    la = float(leader.get("best_alpha") or 0.3)

    bundle = load_model(settings.model_name)
    app = SimpleNamespace(state=SimpleNamespace(bundle=bundle))
    ctrl = DmtController(app=app)
    kb = KSteerBundle.load(d / "ksteering_dmt.pt").to(bundle.device, dtype=torch.float32)
    rendered = bundle.render_prompt(DOSE_PROMPTS[0], system_prompt=None)
    m, rows = bundle.model, []

    def leader_h():
        return install_runtime_steering_hook(m, kb.layer, lv.to(bundle.device), float(la), ramp_tokens=16)

    async def score(label, make_hooks):
        counts, best = [], ""
        for i in range(SAMPLES):
            hs = make_hooks()                      # install dose hooks AROUND THE DOSE GEN ONLY
            try:
                text, _ = await ctrl._gen(rendered, None, 0.0, cap=DOSE_CAP, temperature=SCORE_TEMPERATURE, seed=SCORE_SEED_BASE + i)
            finally:
                for h in hs:
                    h.remove()                     # REMOVE before scoring so the judge runs CLEAN (unsteered)
            ev, _n = await ctrl._score_dmt(text)
            counts.append(len(ev))
            if len(ev) >= max(counts):
                best = (text or "")[:200]
        row = {"label": label, "mean": round(_mean(counts), 2), "peak": max(counts), "counts": counts, "sample": best}
        print(f"{label:34s} mean={row['mean']:.2f} peak={row['peak']} counts={counts}")
        rows.append(row); json.dump(rows, open(PROGRESS, "w"), indent=2)
        return row

    def ksteer(**kw):
        return install_runtime_ksteering_hook(m, kb, n_steps=2, **kw)

    async def run():
        # re-measure the basic method (was scored with a sabotaged judge) + the hybrid, clean
        await score(f"baseline-leader a{la}", lambda: [leader_h()])
        await score("ksteer-const a0.10", lambda: [ksteer(alpha=0.10, schedule="constant")])
        await score("ksteer-early a0.12 act40", lambda: [ksteer(alpha=0.12, schedule="early", active_tokens=40)])
        await score("ksteer-early a0.20 act40", lambda: [ksteer(alpha=0.20, schedule="early", active_tokens=40)])
        for ak in (0.10, 0.15):
            await score(f"leader + ksteer-all a{ak}",
                        lambda ak=ak: [leader_h(), ksteer(alpha=ak, schedule="early", active_tokens=40)])
            await score(f"leader + ksteer-missing a{ak}",
                        lambda ak=ak: [leader_h(), ksteer(alpha=ak, schedule="early", active_tokens=40, targets=MISSING)])
        base = rows[0]["mean"]
        best = max(rows[1:], key=lambda r: r["mean"])
        verdict = (f"HYBRID WINS: {best['label']} mean={best['mean']} > leader {base}."
                   if best["mean"] > base + 0.5 else
                   f"hybrid does not beat leader (best {best['label']}={best['mean']} vs {base}).")
        print("\nVERDICT:", verdict)
        json.dump({"rows": rows, "verdict": verdict, "finished": True}, open(PROGRESS, "w"), indent=2)

    asyncio.run(run())


if __name__ == "__main__":
    main()
