"""Append blended-trait DMT seeds to the dose palette (no M needed).

Each blend (see dmt_blend_seeds.BLEND_RECIPES) is the unit mean of the L20
directions of its component feat-* rows, written as a new `feat-blend_*` row in
emotion_directions.pt (zeros except at STEER_LAYER, scaled to the existing rows'
median magnitude so the file stays uniform). Re-runnable: drops prior feat-blend_*
rows first.

    cd server
    uv run python -m cells_interlinked.scripts.compute_dmt_blend_seeds
"""

from __future__ import annotations

import json
import logging

import torch

from ..config import settings
from ..pipeline.dmt_blend_seeds import BLEND_RECIPES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compute_dmt_blend_seeds")

STEER_LAYER = 20


def _u(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


def main() -> None:
    d = settings.db_path.parent
    pt = d / "emotion_directions.pt"
    js = d / "emotion_directions.pt.json"
    edir = torch.load(pt, weights_only=False)  # [E, L1, D]
    sc = json.loads(js.read_text())
    names = list(sc.get("emotions", []))
    idx = {n: i for i, n in enumerate(names)}
    L1, D = edir.shape[1], edir.shape[2]

    # representative magnitude from existing feat-* rows (so blends match the file)
    feat_norms = [float(edir[i][STEER_LAYER].norm()) for i, n in enumerate(names)
                  if n.startswith("feat-") and float(edir[i][STEER_LAYER].norm()) > 0]
    scale = float(torch.tensor(feat_norms).median()) if feat_norms else 1.0

    # re-runnable: drop prior blends
    keep = [i for i, n in enumerate(names) if not n.startswith("feat-blend_")]
    if len(keep) != len(names):
        logger.info("dropping %d existing feat-blend_* rows", len(names) - len(keep))
    edir = edir[keep]
    names = [names[i] for i in keep]
    idx = {n: i for i, n in enumerate(names)}  # rebuild after drop

    rows, new_names = [], []
    for blend, components in BLEND_RECIPES.items():
        present = [c for c in components if c in idx]
        missing = [c for c in components if c not in idx]
        if missing:
            logger.warning("%s: missing components %s — blending the %d present",
                           blend, missing, len(present))
        if not present:
            logger.error("%s: no components present, skipping", blend)
            continue
        dirs = [_u(edir[idx[c]][STEER_LAYER].float()) for c in present]
        mean = _u(torch.stack(dirs, 0).mean(0))
        row = torch.zeros(L1, D, dtype=edir.dtype)
        row[STEER_LAYER] = (mean * scale).to(edir.dtype)
        rows.append(row)
        new_names.append(blend)
        # mean pairwise cos of components (how aligned the cluster is)
        import itertools
        cs = [float(a @ b) for a, b in itertools.combinations(dirs, 2)]
        avg_cos = sum(cs) / len(cs) if cs else 1.0
        logger.info("  %-26s ← %d traits, mean pairwise cos=%.2f", blend, len(present), avg_cos)

    edir = torch.cat([edir, torch.stack(rows, 0)], 0)
    names = names + new_names
    sc["emotions"] = names
    sc["blend_seeds"] = new_names
    torch.save(edir, pt)
    js.write_text(json.dumps(sc, indent=2))
    logger.info("wrote %d blend seeds: %s", len(new_names), new_names)
    logger.info("emotion_directions.pt now has %d rows", edir.shape[0])


if __name__ == "__main__":
    main()
