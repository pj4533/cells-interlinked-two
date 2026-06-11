"""SAE clamping for DMT — v2 (fix the v1 miscalibration). v1 clamped a + g·max_act with
max_act in the thousands → a ~14-30k perturbation on a ~36k residual → instant gibberish at
every strength (so "0 features" was coherence collapse, NOT a real ceiling); and the random
control added zero (on_scale padded with zeros). Both invalid.

v2: the clamp delta direction is "turn the DMT features up" (a' = max(a, max_act)), but the
delta is RESCALED to a swept magnitude `mag` calibrated to the coherent dose range. The sweep
self-calibrates — over-clamp → gibberish → 0 features → not the best mag — so the best DMT
count over the sweep is the fair number. The random control clamps the same count of REAL
random features (real max_act now) at the SAME magnitude.

Conditions (mean DMT-feature count, N seeds, clean judge):
  - baseline-leader
  - leader + dmt-clamp @ mag-sweep      (best over sweep = the calibrated DMT-clamp result)
  - leader + random-clamp @ best mag    (matched magnitude + count control)
  - dmt-clamp-only @ best mag

Pass: best mean > leader + max(0.5, leader_stdev), best min >= leader_mean-0.5, and best beats
random by >=0.5. Else: SAE clamping caps ~3.5 → 3rd confirmation of the representational wall.

OFFLINE — backend STOPPED. Writes /tmp/dmt_sae_clamp_v2_L{layer}.json.
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
from ..pipeline.abliteration import _find_decoder_layers, install_runtime_steering_hook
from ..pipeline.autoresearch_dmt import DOSE_CAP, DOSE_PROMPTS, SCORE_TEMPERATURE, DmtController
from ..pipeline.model_loader import load_model
from .berg_gate_v2 import is_coherent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("dmt_sae_clamp_v2")

SAMPLES = 6
CONFIRM_SEED = 41000
MAGS = [2000.0, 4000.0, 6000.0, 9000.0]    # calibrated clamp-delta L2 norms (coherent dose ~3k)


def make_clamp_hook(model, B, layer, cols, mag, ramp=16):
    lyr = _find_decoder_layers(model)[layer]
    w_enc = B["w_enc"][:, cols]; b_enc = B["b_enc"][cols]; thr = B["thr"][cols]
    w_dec = B["w_dec"][cols, :]; b_dec = B["b_dec"]; tgt = B["on_scale_full"][cols]
    step = [0]

    def hook(_m, _i, out):
        hidden = out[0] if isinstance(out, tuple) else out
        step[0] += 1
        frac = min(1.0, step[0] / max(1, ramp))
        h = hidden.to(torch.float32).clone()
        x = h[:, -1, :]
        pre = (x - b_dec) @ w_enc + b_enc
        a = torch.where(pre > thr, pre, torch.zeros_like(pre))
        a_new = torch.maximum(a, tgt)                 # clamp the features ON (activation-gated)
        delta = (a_new - a) @ w_dec                   # [B, D]  direction = co-activate the features
        n = delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        h[:, -1, :] = x + delta / n * (mag * frac)    # magnitude calibrated to coherent range
        o = h.to(hidden.dtype)
        return (o,) + out[1:] if isinstance(out, tuple) else o

    return lyr.register_forward_hook(hook)


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
    B = {"w_enc": raw["w_enc_sel"].to(dev), "b_enc": raw["b_enc_sel"].to(dev),
         "thr": raw["thr_sel"].to(dev), "w_dec": raw["w_dec_sel"].to(dev),
         "b_dec": raw["b_dec"].to(dev), "on_scale_full": raw["on_scale"].to(dev)}
    TGT = list(range(nt))
    RND = list(range(nt, nt + nt))
    rendered = bundle.render_prompt(DOSE_PROMPTS[0], system_prompt=None)
    out = f"/tmp/dmt_sae_clamp_v2_L{layer}.json"
    rows = []

    def leader_h():
        return install_runtime_steering_hook(m, layer, lv, la, ramp_tokens=16)

    async def score(label, make_hooks):
        counts, coh = [], []
        for i in range(SAMPLES):
            hs = make_hooks()
            try:
                text, _ = await ctrl._gen(rendered, None, 0.0, cap=DOSE_CAP,
                                          temperature=SCORE_TEMPERATURE, seed=CONFIRM_SEED + i)
            finally:
                for h in hs:
                    h.remove()
            ev, _n = await ctrl._score_dmt(text)
            counts.append(len(ev)); coh.append(int(is_coherent(text)))
        row = {"label": label, "mean": round(statistics.mean(counts), 2),
               "stdev": round(statistics.pstdev(counts), 2), "min": min(counts), "max": max(counts),
               "coherent_frac": round(statistics.mean(coh), 2), "counts": counts}
        rows.append(row); json.dump({"rows": rows, "target": raw["target_feats"], "cluster": raw["target_cluster"]}, open(out, "w"), indent=2, default=str)
        logger.info("%-26s mean=%.2f ±%.2f range[%d,%d] coh=%.2f %s",
                    label, row["mean"], row["stdev"], row["min"], row["max"], row["coherent_frac"], counts)
        return row

    logger.info("=== DMT SAE clamp v2 L%d | leader=%s a=%.2f | feats=%s (%s) ===",
                layer, leader["id"], la, raw["target_feats"], raw["target_cluster"])

    await score("baseline-leader", lambda: [leader_h()])
    for mg in MAGS:
        await score(f"leader+dmt-clamp m{int(mg)}", lambda mg=mg: [leader_h(), make_clamp_hook(m, B, layer, TGT, mg)])
    best = max((r for r in rows if r["label"].startswith("leader+dmt")), key=lambda r: r["mean"])
    best_mag = int(best["label"].split("m")[-1])
    await score(f"leader+random-clamp m{best_mag}", lambda: [leader_h(), make_clamp_hook(m, B, layer, RND, float(best_mag))])
    await score(f"dmt-clamp-only m{best_mag}", lambda: [make_clamp_hook(m, B, layer, TGT, float(best_mag))])

    base = rows[0]
    rnd = next(r for r in rows if "random-clamp" in r["label"])
    confirmed = (best["mean"] > base["mean"] + max(0.5, base["stdev"]) and best["min"] >= base["mean"] - 0.5
                 and best["mean"] - rnd["mean"] >= 0.5)
    verdict = {
        "layer": layer, "leader_mean": base["mean"], "leader_stdev": base["stdev"],
        "best_dmt_clamp": best["label"], "best_mean": best["mean"], "best_min": best["min"],
        "best_coherent_frac": best["coherent_frac"], "random_clamp_mean": rnd["mean"],
        "CEILING_BROKEN": bool(confirmed),
        "note": ("SAE clamping co-activates more DMT features than the leader — wall broken"
                 if confirmed else
                 "SAE clamping does NOT beat the leader (caps ~3.5) — 3rd confirmation of the representational wall"),
    }
    json.dump({"rows": rows, "verdict": verdict}, open(out, "w"), indent=2, default=str)
    logger.info("VERDICT: %s", json.dumps(verdict, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(run())
