"""Diagnostic: is the DMT single-vector ceiling an OFF-MANIFOLD / coherence wall?

Doses the atlas leader across a WIDE α range (wider than the search sweep) and, at
each α, measures both:
  - DMT feature count (mean over samples, via the existing DmtController scorer), and
  - off-manifold drift (`off_ortho_mean` from trajectory.py, the same metric `/trip`
    shows) of the dosed L32 trajectory vs. a RAW (undosed) trajectory.

Reading:
  - If features PEAK at a low α where drift is still small, then COLLAPSE as α rises
    and off_ortho climbs → the single linear vector is hitting an off-manifold wall;
    a curved / scheduled PATH could traverse more coherent state-space → path steering
    is justified.
  - If features stay capped while drift stays low → curvature is NOT the bottleneck
    (the ceiling is the judge or genuine co-occurrence limits); path steering won't help.

Read-only: loads its own M, never touches the atlas. Run with the backend STOPPED.

    cd server
    uv run python -m cells_interlinked.scripts.dmt_offmanifold_diagnostic
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import torch

from ..config import settings
from ..pipeline.autoresearch_dmt import DOSE_CAP, DOSE_PROMPTS, DmtController
from ..pipeline.model_loader import load_model
from ..pipeline.trajectory import build_series, compute_raw_basis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dmt_offmanifold_diagnostic")

ALPHAS = [0.15, 0.25, 0.45, 0.7, 1.0, 1.4]   # search uses [0.25,0.45]; extend up to find the cliff
SAMPLES = 3                                  # doses averaged per α (kept low; this is a one-off)


def main() -> None:
    d = settings.db_path.parent
    atlas = json.loads((d / "atlas_dmt" / "atlas.json").read_text())["atlas"]
    leader = max(atlas, key=lambda e: (e.get("score", 0), e.get("peak", 0)))
    vp = d / "atlas_dmt" / "vectors" / f"{leader['id']}.pt"
    vec = torch.load(vp, weights_only=False)
    logger.info("leader = %s  committed mean=%s peak=%s", leader["id"], leader.get("score"), leader.get("peak"))

    logger.info("loading M (%s) …", settings.model_name)
    bundle = load_model(settings.model_name)
    app = SimpleNamespace(state=SimpleNamespace(bundle=bundle))
    ctrl = DmtController(app=app)
    rendered = bundle.render_prompt(DOSE_PROMPTS[0], system_prompt=None)

    async def run() -> None:
        # RAW trajectory → the manifold reference.
        raw_text, raw_acts = await ctrl._gen(rendered, None, 0.0, cap=DOSE_CAP)
        basis = compute_raw_basis(raw_acts)
        raw_series = build_series(raw_acts, [], raw_text, 0.0, "eos", basis)
        raw_feat = len((await ctrl._score_dmt(raw_text))[0])
        logger.info("RAW: off_ortho=%.3f (baseline), features=%d, tokens=%d",
                    raw_series.off_ortho_mean, raw_feat, len(raw_acts))

        rows = []
        for a in ALPHAS:
            feats, offs = [], []
            for _ in range(SAMPLES):
                text, acts = await ctrl._gen(rendered, vec, a, cap=DOSE_CAP)
                if not acts:
                    continue
                s = build_series(acts, [], text, a, "eos", basis)
                ev, _ = await ctrl._score_dmt(text)
                feats.append(len(ev))
                offs.append(s.off_ortho_mean)
            mf = sum(feats) / len(feats) if feats else 0.0
            mo = sum(offs) / len(offs) if offs else 0.0
            rows.append((a, mf, mo, feats, [round(o, 3) for o in offs]))
            logger.info("α=%.2f: features=%.2f  off_ortho=%.3f  (feat samples %s)", a, mf, mo, feats)

        print("\n=== DMT off-manifold diagnostic ===")
        print(f"leader: {leader['id']}  (committed mean {leader.get('score')}, peak {leader.get('peak')})")
        print(f"RAW baseline off_ortho = {raw_series.off_ortho_mean:.3f}\n")
        print(f"{'α':>5}  {'mean_feats':>10}  {'off_ortho':>9}  {'Δ vs raw':>8}")
        for a, mf, mo, _fs, _os in rows:
            print(f"{a:>5.2f}  {mf:>10.2f}  {mo:>9.3f}  {mo - raw_series.off_ortho_mean:>+8.3f}")
        # crude read: where do features peak, and is drift still climbing past it?
        best = max(rows, key=lambda r: r[1])
        print(f"\nfeatures peak at α={best[0]:.2f} (mean {best[1]:.2f}), off_ortho there = {best[2]:.3f}")
        past = [r for r in rows if r[0] > best[0]]
        if past:
            drift_rises = past[-1][2] > best[2]
            feats_fall = past[-1][1] < best[1]
            print(f"beyond the peak: off_ortho {'RISES' if drift_rises else 'flat/falls'}, "
                  f"features {'FALL' if feats_fall else 'hold'}")
            if drift_rises and feats_fall:
                print("→ looks like an OFF-MANIFOLD WALL: features collapse as the linear dose drifts "
                      "off-manifold. A curved/scheduled PATH could traverse further while coherent → "
                      "path steering is justified.")
            else:
                print("→ NOT clearly an off-manifold wall: features don't collapse with rising drift. "
                      "The ceiling is likely the judge or genuine co-occurrence limits; path steering "
                      "is unlikely to help much.")

    asyncio.run(run())


if __name__ == "__main__":
    main()
