"""Reusable rigorous hybrid test for any K-steer classifier bundle:
leader-dose vs leader + ksteer-toward-missing-clusters(bundle) at α∈{0.12,0.15,0.18},
6 samples on INDEPENDENT seeds, clean judge. Usage:
    uv run python -m cells_interlinked.scripts.test_hybrid_bundle <bundle.pt>
Writes /tmp/dmt_hybrid_<bundle>.json. Run with backend STOPPED.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
from types import SimpleNamespace

import torch

from ..config import settings
from ..pipeline.abliteration import install_runtime_steering_hook
from ..pipeline.autoresearch_dmt import DOSE_CAP, DOSE_PROMPTS, SCORE_TEMPERATURE, DmtController
from ..pipeline.dmt_feature_clusters import CLUSTER_NAMES
from ..pipeline.k_steering import KSteerBundle, install_runtime_ksteering_hook
from ..pipeline.model_loader import load_model

SAMPLES = 6
CONFIRM_SEED = 7000
MISSING = [CLUSTER_NAMES.index(c) for c in ("entity", "hyperspace", "visual", "otherness")]


def main() -> None:
    bundle_name = sys.argv[1] if len(sys.argv) > 1 else "ksteering_dmt.pt"
    d = settings.db_path.parent
    atlas = json.loads((d / "atlas_dmt" / "atlas.json").read_text())["atlas"]
    leader = max(atlas, key=lambda e: e["score"])
    lv = torch.load(d / "atlas_dmt" / "vectors" / f"{leader['id']}.pt", weights_only=False).float().reshape(-1)
    la = float(leader.get("best_alpha") or 0.3)

    bundle = load_model(settings.model_name)
    app = SimpleNamespace(state=SimpleNamespace(bundle=bundle))
    ctrl = DmtController(app=app)
    kb = KSteerBundle.load(d / bundle_name).to(bundle.device, dtype=torch.float32)
    # cluster bundles target the missing CLUSTERS; finer/other bundles carry their own
    # gap target set in dmt_indices.
    targets = MISSING if kb.classes == CLUSTER_NAMES else kb.dmt_indices
    rendered = bundle.render_prompt(DOSE_PROMPTS[0], system_prompt=None)
    m, rows = bundle.model, []
    out = f"/tmp/dmt_hybrid_{bundle_name.replace('.pt','')}.json"

    def leader_h():
        return install_runtime_steering_hook(m, kb.layer, lv.to(bundle.device), float(la), ramp_tokens=16)

    async def score(label, make_hooks):
        counts = []
        for i in range(SAMPLES):
            hs = make_hooks()
            try:
                text, _ = await ctrl._gen(rendered, None, 0.0, cap=DOSE_CAP, temperature=SCORE_TEMPERATURE, seed=CONFIRM_SEED + i)
            finally:
                for h in hs:
                    h.remove()
            ev, _n = await ctrl._score_dmt(text)
            counts.append(len(ev))
        row = {"label": label, "mean": round(statistics.mean(counts), 2),
               "stdev": round(statistics.pstdev(counts), 2), "min": min(counts), "max": max(counts), "counts": counts}
        print(f"{label:34s} mean={row['mean']:.2f} ±{row['stdev']} range[{row['min']},{row['max']}] counts={counts}")
        rows.append(row); json.dump(rows, open(out, "w"), indent=2)
        return row

    async def run():
        print(f"=== bundle: {bundle_name} ===")
        await score("baseline-leader", lambda: [leader_h()])
        for ak in (0.12, 0.15, 0.18):
            await score(f"leader+ksteer-missing a{ak}",
                        lambda ak=ak: [leader_h(), install_runtime_ksteering_hook(m, kb, alpha=ak, n_steps=2, schedule="early", active_tokens=40, targets=targets)])
        base = rows[0]
        best = max(rows[1:], key=lambda r: r["mean"])
        confirmed = best["mean"] > base["mean"] + max(0.5, base["stdev"]) and best["min"] >= base["mean"] - 0.5
        verdict = (f"WINNER [{bundle_name}]: {best['label']} {best['mean']}±{best['stdev']} vs leader {base['mean']}±{base['stdev']}"
                   if confirmed else
                   f"no win [{bundle_name}]: best {best['label']}={best['mean']}±{best['stdev']} vs leader {base['mean']}±{base['stdev']}")
        print("\nVERDICT:", verdict)
        json.dump({"rows": rows, "verdict": verdict, "confirmed": confirmed, "finished": True}, open(out, "w"), indent=2)

    asyncio.run(run())


if __name__ == "__main__":
    main()
