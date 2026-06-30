"""Promote the winning DMT entity-encounter PERSONA vectors (built by
build_persona_entity_seeds.py, validated by diag_persona_entity.py) into
emotion_directions.pt as named seed rows at L20, so DmtController.ENTITY_SEEDS
can seed the hill-climb from them. L20 only (the diagnostic: L16 incoherent, L31
genericizes). Re-runnable: drops existing persona-* rows first. No M needed.

  cd server && uv run python -m cells_interlinked.scripts.append_persona_seeds
"""
from __future__ import annotations

import json
import logging

import torch

from ..config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("append_persona_seeds")

STEER_LAYER = 20
# The flavor groups from persona_entity_prompts.FLAVOR_GROUPS (2026-06-30 rebuild,
# machine-elf retarget): composite (all 15 entity forms) + nonhuman / divine /
# trickster sub-directions. Old telepathic/guide groups are gone; this script
# drops stale persona-* rows first, so they're cleaned up automatically.
PROMOTE = ["composite", "nonhuman", "divine", "trickster"]


def main() -> None:
    d = settings.db_path.parent
    pt, js = d / "emotion_directions.pt", d / "emotion_directions.pt.json"
    persona_dir = d / "persona_seeds"
    edir = torch.load(pt, weights_only=False)  # [E, L1, D]
    sc = json.loads(js.read_text())
    names = list(sc.get("emotions", []))
    L1, D = edir.shape[1], edir.shape[2]

    # re-runnable: drop existing persona-* rows
    keep = [i for i, n in enumerate(names) if not n.startswith("persona-")]
    if len(keep) != len(names):
        logger.info("dropping %d existing persona-* rows", len(names) - len(keep))
    edir, names = edir[keep], [names[i] for i in keep]

    rows, new_names = [], []
    for g in PROMOTE:
        vp = persona_dir / f"persona-{g}-L{STEER_LAYER}.pt"
        v = torch.load(vp, map_location="cpu", weights_only=False).float()
        row = torch.zeros(L1, D, dtype=edir.dtype)
        row[STEER_LAYER] = v.to(edir.dtype)
        rows.append(row)
        nm = f"persona-{g}"
        new_names.append(nm)
        logger.info("  %s  ‖v‖=%.1f @L%d", nm, float(v.norm()), STEER_LAYER)

    edir = torch.cat([edir, torch.stack(rows, 0)], 0)
    names = names + new_names
    sc["emotions"] = names
    sc["persona_seeds"] = new_names
    sc["persona_seeds_note"] = (
        "DMT entity-encounter persona vectors (Anthropic persona-vector recipe on the "
        "model's own in-encounter generations vs matched 'alone' introspection, grounded "
        "in DMT entity phenomenology). L20. Built by build_persona_entity_seeds.py; in "
        "DmtController.ENTITY_SEEDS."
    )
    torch.save(edir, pt)
    js.write_text(json.dumps(sc, indent=2))
    logger.info("appended %d persona seeds: %s (file now %d rows)", len(new_names), new_names, edir.shape[0])


if __name__ == "__main__":
    main()
