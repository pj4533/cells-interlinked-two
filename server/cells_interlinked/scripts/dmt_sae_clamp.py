"""SAE clamping for DMT (#3) — the experiment. Clamp the diverse per-cluster DMT SAE features
(build_dmt_sae_features.py) high during generation and score the DMT-feature count with the real
judge, vs the additive leader. The decisive question: can fine-grained SAE clamping co-activate
more DMT feature-TYPES than the single leader vector (which caps ~3.5 via linear interference)?

Conditions (mean DMT-feature count, N independent seeds, clean judge — same noise discipline as
test_hybrid_bundle):
  - baseline-leader              the additive leader dose alone (the bar to beat)
  - leader + dmt-clamp gG        leader dose + clamp the 6 DMT features up
  - leader + random-clamp gG     leader dose + clamp 6 RANDOM features up  [matched control]
  - dmt-clamp gG (no leader)     SAE clamp alone — does it produce DMT unaided?

Pass (SAE clamping breaks the ceiling): best mean > leader + max(0.5, leader_stdev) AND best min
>= leader_mean - 0.5 (the K-steer confirmation gate). Else: even fine-grained multi-feature SAE
clamping caps ~3.5 — confirms the representational wall.

OFFLINE — backend STOPPED. Writes /tmp/dmt_sae_clamp_L{layer}.json.
    cd server && uv run python -m cells_interlinked.scripts.dmt_sae_clamp --layer 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
from types import SimpleNamespace

import torch

from ..config import settings
from ..pipeline.abliteration import install_runtime_steering_hook
from ..pipeline.autoresearch_dmt import DOSE_CAP, DOSE_PROMPTS, SCORE_TEMPERATURE, DmtController
from ..pipeline.model_loader import load_model
from .berg_sae_replication import make_sae_hook

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("dmt_sae_clamp")

SAMPLES = 6
CONFIRM_SEED = 31000
CLAMP = [2.0, 4.0, 8.0]    # multiples of each feature's max_act


async def run() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    args = ap.parse_args()
    layer = args.layer
    d = settings.db_path.parent

    atlas = json.loads((d / "atlas_dmt" / "atlas.json").read_text())["atlas"]
    leader = max(atlas, key=lambda e: e["score"])
    lv = torch.load(d / "atlas_dmt" / "vectors" / f"{leader['id']}.pt", weights_only=False).float().reshape(-1)
    la = float(leader.get("best_alpha") or 0.3)

    raw = torch.load(d / f"dmt_sae_L{layer}.pt", weights_only=False)
    bundle = load_model(settings.model_name, device_str=settings.device,
                        dtype=torch.bfloat16, extraction_layer=settings.extraction_layer)
    app = SimpleNamespace(state=SimpleNamespace(bundle=bundle))
    ctrl = DmtController(app=app)
    m, dev = bundle.model, bundle.device
    lv = lv.to(dev)
    nt = len(raw["target_feats"])
    on_full = torch.cat([raw["on_scale"], torch.zeros(len(raw["rand_feats"]))]).to(dev)
    B = {"dev": dev, "w_enc": raw["w_enc_sel"].to(dev), "b_enc": raw["b_enc_sel"].to(dev),
         "thr": raw["thr_sel"].to(dev), "w_dec": raw["w_dec_sel"].to(dev),
         "b_dec": raw["b_dec"].to(dev), "on_scale_full": on_full}
    TGT = list(range(nt))
    RND = list(range(nt, nt + nt))
    rendered = bundle.render_prompt(DOSE_PROMPTS[0], system_prompt=None)
    out = f"/tmp/dmt_sae_clamp_L{layer}.json"
    rows = []

    def leader_h():
        return install_runtime_steering_hook(m, layer, lv, la, ramp_tokens=16)

    async def score(label, make_hooks):
        counts = []
        for i in range(SAMPLES):
            hs = make_hooks()
            try:
                text, _ = await ctrl._gen(rendered, None, 0.0, cap=DOSE_CAP,
                                          temperature=SCORE_TEMPERATURE, seed=CONFIRM_SEED + i)
            finally:
                for h in hs:
                    h.remove()
            ev, _n = await ctrl._score_dmt(text)
            counts.append(len(ev))
        row = {"label": label, "mean": round(statistics.mean(counts), 2),
               "stdev": round(statistics.pstdev(counts), 2), "min": min(counts), "max": max(counts), "counts": counts}
        rows.append(row); json.dump({"rows": rows, "target": raw["target_feats"], "cluster": raw["target_cluster"]}, open(out, "w"), indent=2, default=str)
        logger.info("%-26s mean=%.2f ±%.2f range[%d,%d] %s", label, row["mean"], row["stdev"], row["min"], row["max"], counts)
        return row

    logger.info("=== DMT SAE clamp L%d | leader=%s a=%.2f | feats=%s (%s) ===",
                layer, leader["id"], la, raw["target_feats"], raw["target_cluster"])

    await score("baseline-leader", lambda: [leader_h()])
    for gG in CLAMP:
        await score(f"leader+dmt-clamp g{gG}", lambda gG=gG: [leader_h(), make_sae_hook(m, B, layer, TGT, "amplify", gG)])
    best_g = max((r for r in rows if r["label"].startswith("leader+dmt")), key=lambda r: r["mean"])["label"].split("g")[-1]
    await score(f"leader+random-clamp g{best_g}", lambda: [leader_h(), make_sae_hook(m, B, layer, RND, "amplify", float(best_g))])
    await score(f"dmt-clamp-only g{best_g}", lambda: [make_sae_hook(m, B, layer, TGT, "amplify", float(best_g))])

    base = rows[0]
    best = max((r for r in rows if "dmt-clamp" in r["label"] and "only" not in r["label"]), key=lambda r: r["mean"])
    rnd = next(r for r in rows if "random-clamp" in r["label"])
    confirmed = (best["mean"] > base["mean"] + max(0.5, base["stdev"]) and best["min"] >= base["mean"] - 0.5
                 and best["mean"] - rnd["mean"] >= 0.5)
    verdict = {
        "layer": layer, "leader_mean": base["mean"], "leader_stdev": base["stdev"],
        "best_dmt_clamp": best["label"], "best_mean": best["mean"], "best_min": best["min"],
        "random_clamp_mean": rnd["mean"],
        "CEILING_BROKEN": bool(confirmed),
        "note": ("SAE clamping co-activates more DMT features than the leader — wall broken"
                 if confirmed else
                 "SAE clamping does NOT beat the leader (caps ~3.5) — confirms the representational wall"),
    }
    json.dump({"rows": rows, "verdict": verdict}, open(out, "w"), indent=2, default=str)
    logger.info("VERDICT: %s", json.dumps(verdict, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(run())
