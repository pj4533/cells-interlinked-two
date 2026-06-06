"""Measure score variance of the DMT atlas leader.

Re-scores the highest-scoring committed direction N times with the EXACT same
scoring path the loop uses (`DmtController._score_candidate`: dose across the
α-sweep, judge + verify, take the max cell). Because the dose generation is
temperature-sampled (only the judge is greedy), the same direction can score
differently each run. This tells us whether the search is noise-limited — if the
leader doesn't reliably re-score its committed value, single-shot scoring is the
problem, not the candidates.

Read-only: loads its own M, never touches the atlas. Run with the backend STOPPED
(two M's won't fit on 64 GB):

    cd server
    uv run python -m cells_interlinked.scripts.check_leader_variance
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
from types import SimpleNamespace

import torch

from ..config import settings
from ..pipeline.autoresearch_dmt import ALPHA_SWEEP, DmtController
from ..pipeline.model_loader import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("check_leader_variance")

N_RUNS = 5


def main() -> None:
    d = settings.db_path.parent
    atlas = json.loads((d / "atlas_dmt" / "atlas.json").read_text())["atlas"]
    leader = max(atlas, key=lambda e: (e.get("score", 0), e.get("peak", 0)))
    vp = d / "atlas_dmt" / "vectors" / f"{leader['id']}.pt"
    if not vp.exists():
        raise SystemExit(f"leader vector missing: {vp}")
    vec = torch.load(vp, weights_only=False)
    logger.info("leader = %s  committed score=%s  feats=%s",
                leader["id"], leader.get("score"), leader.get("matched_features"))
    logger.info("re-scoring %d× with sweep %s (dose is temperature-sampled) …", N_RUNS, ALPHA_SWEEP)

    logger.info("loading M (%s) …", settings.model_name)
    bundle = load_model(settings.model_name)
    app = SimpleNamespace(state=SimpleNamespace(bundle=bundle))
    ctrl = DmtController(app=app)

    async def run() -> None:
        rows = []
        for i in range(N_RUNS):
            res = await ctrl._score_candidate(vec)
            rows.append(res)
            logger.info("run %d/%d: mean=%.2f  peak=%s  bestα=%s  feats=%s",
                        i + 1, N_RUNS, res["score"], res["peak"],
                        res["best_alpha"], res["matched_features"])
        scores = [r["score"] for r in rows]
        fines = [r["peak"] for r in rows]
        print("\n=== leader score variance (each run already averages SAMPLES_PER_CELL doses) ===")
        print(f"leader: {leader['id']}  (committed at score {leader.get('score')})")
        print(f"re-scored {N_RUNS}× (mean per run): {[round(s, 2) for s in scores]}")
        print(f"  mean-of-means={statistics.mean(scores):.2f}  stdev={statistics.pstdev(scores):.2f}"
              f"  range=[{min(scores):.2f}, {max(scores):.2f}]")
        print(f"  peak per run: {fines}")
        committed = leader.get("score", 0)
        gap = committed - statistics.mean(scores)
        verdict = ("re-scores FAR below committed value → that commit was a fluke (single-shot "
                   "selection bias); averaged scoring is the fix."
                   if gap >= 1.0 else
                   "re-scores near committed value → reliable.")
        print(f"  gap(committed − mean)={gap:.2f} → {verdict}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
