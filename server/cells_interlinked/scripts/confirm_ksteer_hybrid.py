"""Confirm the hybrid winner (leader + K-steer toward MISSING clusters) with INDEPENDENT
seeds + more samples, and dump full report texts to verify the features are real (not a
judge artifact / selection-bias fluke). Run with backend STOPPED.
Writes /tmp/dmt_ksteer_confirm.json and /tmp/dmt_ksteer_confirm_texts.txt.
"""

from __future__ import annotations

import asyncio
import json
import statistics
from types import SimpleNamespace

import torch

from ..config import settings
from ..pipeline.abliteration import install_runtime_steering_hook
from ..pipeline.autoresearch_dmt import DOSE_CAP, DOSE_PROMPTS, SCORE_TEMPERATURE, DmtController
from ..pipeline.dmt_feature_clusters import CLUSTER_NAMES
from ..pipeline.k_steering import KSteerBundle, install_runtime_ksteering_hook
from ..pipeline.model_loader import load_model

SAMPLES = 6
CONFIRM_SEED = 7000      # INDEPENDENT of the discovery seeds (avoid the same lucky draws)
MISSING = [CLUSTER_NAMES.index(c) for c in ("entity", "hyperspace", "visual", "otherness")]


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
    texts = open("/tmp/dmt_ksteer_confirm_texts.txt", "w")

    def leader_h():
        return install_runtime_steering_hook(m, kb.layer, lv.to(bundle.device), float(la), ramp_tokens=16)

    async def score(label, make_hooks):
        counts, feats_all = [], []
        for i in range(SAMPLES):
            hs = make_hooks()
            try:
                text, _ = await ctrl._gen(rendered, None, 0.0, cap=DOSE_CAP, temperature=SCORE_TEMPERATURE, seed=CONFIRM_SEED + i)
            finally:
                for h in hs:
                    h.remove()
            ev, _n = await ctrl._score_dmt(text)
            counts.append(len(ev)); feats_all.append(sorted(ev.keys()))
            texts.write(f"\n===== {label} | sample {i} | {len(ev)} feats {sorted(ev.keys())} =====\n{text}\n")
            texts.flush()
        row = {"label": label, "mean": round(statistics.mean(counts), 2),
               "stdev": round(statistics.pstdev(counts), 2), "min": min(counts), "max": max(counts),
               "counts": counts, "feature_sets": feats_all}
        print(f"{label:34s} mean={row['mean']:.2f} ±{row['stdev']} range[{row['min']},{row['max']}] counts={counts}")
        rows.append(row); json.dump(rows, open("/tmp/dmt_ksteer_confirm.json", "w"), indent=2)
        return row

    async def run():
        await score("baseline-leader", lambda: [leader_h()])
        for ak in (0.12, 0.15, 0.18):
            await score(f"leader+ksteer-missing a{ak}",
                        lambda ak=ak: [leader_h(), install_runtime_ksteering_hook(m, kb, alpha=ak, n_steps=2, schedule="early", active_tokens=40, targets=MISSING)])
        base = rows[0]
        best = max(rows[1:], key=lambda r: r["mean"])
        # confirmed if the hybrid's mean clears the leader's by > its own stdev AND min >= leader mean
        confirmed = best["mean"] > base["mean"] + max(0.5, base["stdev"]) and best["min"] >= base["mean"] - 0.5
        verdict = (f"CONFIRMED WINNER: {best['label']} mean={best['mean']}±{best['stdev']} (range {best['min']}-{best['max']}) "
                   f"vs leader {base['mean']}±{base['stdev']} — robust on independent seeds." if confirmed else
                   f"NOT robust: {best['label']}={best['mean']}±{best['stdev']} vs leader {base['mean']}±{base['stdev']} — likely selection-bias/noise.")
        print("\nVERDICT:", verdict)
        json.dump({"rows": rows, "verdict": verdict, "confirmed": confirmed, "finished": True}, open("/tmp/dmt_ksteer_confirm.json", "w"), indent=2)
        texts.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
