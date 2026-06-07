"""One-time rescore of the DMT atlas under the noise-controlled scoring regime.

The committed scores were noise-inflated (selection bias: the search commits the
argmax of noisy means; a committed mean 3.8 re-scores ~1.7). The VECTORS are valid
— only the numbers are wrong — so we keep every direction and re-evaluate it with
the new regime (common-random-number seeds + lowered dose temperature, via
`DmtController._score_candidate`), overwriting score/peak/per_alpha and recomputing
the frontier. Vectors are untouched.

Writes progress to /tmp/dmt_rescore_progress.json after every entry (done/total,
ETA, old→new) so a watcher can report live. Run with the backend STOPPED.

    cd server
    uv run python -m cells_interlinked.scripts.rescore_dmt_atlas
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from types import SimpleNamespace

import torch

from ..config import settings
from ..pipeline.autoresearch_dmt import ALPHA_SWEEP, SAMPLES_PER_CELL, SCORE_TEMPERATURE, DmtController
from ..pipeline.model_loader import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rescore_dmt_atlas")

PROGRESS = "/tmp/dmt_rescore_progress.json"


def main() -> None:
    d = settings.db_path.parent
    apath = d / "atlas_dmt" / "atlas.json"
    blob = json.loads(apath.read_text())
    atlas = blob["atlas"]
    # rescore highest-committed first so the frontier stabilizes early
    order = sorted(range(len(atlas)), key=lambda i: -atlas[i].get("score", 0))
    total = len(order)
    logger.info("rescoring %d atlas entries (regime: %d samples × α%s, temp %.2f, CRN seeds)",
                total, SAMPLES_PER_CELL, ALPHA_SWEEP, SCORE_TEMPERATURE)

    logger.info("loading M (%s) …", settings.model_name)
    bundle = load_model(settings.model_name)
    app = SimpleNamespace(state=SimpleNamespace(bundle=bundle))
    ctrl = DmtController(app=app)

    started = time.time()

    def write_progress(done: int, current: str, results: list, finished: bool) -> None:
        elapsed = time.time() - started
        per = elapsed / done if done else 0.0
        json.dump({
            "done": done, "total": total, "current": current,
            "elapsed_s": round(elapsed), "per_entry_s": round(per),
            "eta_s": round(per * (total - done)) if done else None,
            "frontier_so_far": round(max([r["new"] for r in results], default=0.0), 2),
            "results": results[-12:], "finished": finished,
        }, open(PROGRESS, "w"), indent=2)

    async def run() -> None:
        results = []
        write_progress(0, "starting", results, False)
        for n, idx in enumerate(order, 1):
            e = atlas[idx]
            vp = d / "atlas_dmt" / "vectors" / f"{e['id']}.pt"
            if not vp.exists():
                logger.warning("  %s: vector missing, skipping", e["id"])
                continue
            vec = torch.load(vp, weights_only=False)
            old = e.get("score", 0.0)
            res = await ctrl._score_candidate(vec)        # SCORE_SEED_BASE (CRN), low temp
            e["score"] = res["score"]
            e["peak"] = res["peak"]
            e["best_alpha"] = res["best_alpha"]
            e["best_prompt"] = res["best_prompt"]
            e["matched_features"] = res["matched_features"]
            e["matched_evidence"] = res.get("matched_evidence", {})
            e["per_alpha"] = res.get("per_alpha", {})
            e["sample"] = res["sample"]
            e["rescored_from"] = round(old, 2)            # remember the inflated value
            e.pop("refined_from", None)                   # old chain was pre-rescore / inflated
            results.append({"id": e["id"], "old": round(old, 2), "new": res["score"]})
            logger.info("  [%d/%d] %-26s %.2f → %.2f (peak %d)", n, total, e["id"], old, res["score"], res["peak"])
            # checkpoint the atlas + progress after every entry (resumable, watchable)
            blob["frontier"] = round(max(x.get("score", 0.0) for x in atlas), 2)
            apath.write_text(json.dumps(blob, indent=2))
            write_progress(n, e["id"], results, False)

        blob["frontier"] = round(max(x.get("score", 0.0) for x in atlas), 2)
        apath.write_text(json.dumps(blob, indent=2))
        write_progress(total, "done", results, True)
        logger.info("DONE. new frontier = %.2f (was %s)", blob["frontier"],
                    round(max(r["old"] for r in results), 2) if results else "?")

    asyncio.run(run())


if __name__ == "__main__":
    main()
