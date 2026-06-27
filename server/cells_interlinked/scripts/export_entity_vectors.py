"""Hand-picked export of the best DMT entity vectors into the chat/trips dose
palette (the `dmt` group in emotion_directions.pt). Unlike export_to_palette
(top-N by score, auto-named dmt-1/2/…), this promotes specific atlas directions
chosen for entity quality + distinctness, with descriptive names.

Picks (2026-06-27, from the persona-seeded board, ~entity-rate over the cells):
  gen81_crossover  → dmt-entity-contact   53%, presence-dominant; most RELIABLE
  gen176_crossover → dmt-transmission     50%, telepathic + download (communication)
  gen184_crossover → dmt-full-encounter   47%, ALL 5 entity features (broadest)
These three are mutually distinct (pairwise cos ≤ 0.68), i.e. different flavors.

Leaves the atlas untouched. Run with the backend STOPPED, then restart.
  cd server && uv run python -m cells_interlinked.scripts.export_entity_vectors
"""
from __future__ import annotations

import json

import torch

from ..config import settings

STEER_LAYER = 20
ENT = ["entity_presence", "entity_nonhuman", "entity_benevolent_guide",
       "telepathic_communication", "download_transmission"]

PICKS = {
    "gen81_crossover":  ("dmt-entity-contact",  "Reliable entity presence — an autonomous Other is here, watching/moving."),
    "gen176_crossover": ("dmt-transmission",    "Telepathic contact / wordless download from a non-human intelligence."),
    "gen184_crossover": ("dmt-full-encounter",  "Broadest encounter — every entity feature (presence, non-human, guide, telepathy, download)."),
}


def entity_rate(entry) -> float:
    sets = [set(s.get("features") or []) for _a, c in (entry.get("cells") or {}).items() for s in c]
    if not sets:
        return 0.0
    return round(sum(1 for fs in sets if fs & set(ENT)) / len(sets), 2)


def main() -> None:
    d = settings.db_path.parent
    atlas = {e["id"]: e for e in json.loads((d / "atlas_dmt" / "atlas.json").read_text())["atlas"]}
    vdir = d / "atlas_dmt" / "vectors"

    edir = torch.load(d / "emotion_directions.pt", weights_only=False)  # [E, L1, D]
    sc = json.loads((d / "emotion_directions.pt.json").read_text())
    names = list(sc.get("emotions", []))
    L1, D = edir.shape[1], edir.shape[2]

    # Idempotent: drop prior `dmt` group rows.
    prior = set(sc.get("dmt", []))
    keep = [i for i, n in enumerate(names) if n not in prior]
    edir, names = edir[keep], [names[i] for i in keep]

    rows, new_names, meta = [], [], {}
    for aid, (pname, desc) in PICKS.items():
        e = atlas.get(aid)
        if e is None:
            raise SystemExit(f"atlas entry {aid} not found")
        vec = torch.load(vdir / f"{aid}.pt", map_location="cpu", weights_only=False).float()
        row = torch.zeros(L1, D, dtype=edir.dtype)
        row[STEER_LAYER] = vec.to(edir.dtype)
        rows.append(row)
        new_names.append(pname)
        meta[pname] = {
            "atlas_id": aid, "desc": desc, "entity_rate": entity_rate(e),
            "score": e.get("score"), "best_alpha": e.get("best_alpha"),
            "matched_features": e.get("matched_features"),
            "parents": e.get("parents"), "generator": e.get("generator"),
        }
        print(f"  {pname:20} ← {aid:16} ent-rate={meta[pname]['entity_rate']:.0%} ‖v‖={float(vec.norm()):.1f}")

    edir = torch.cat([edir, torch.stack(rows, 0)], 0)
    names = names + new_names
    sc["emotions"] = names
    sc["dmt"] = new_names
    sc["dmt_meta"] = meta
    torch.save(edir, d / "emotion_directions.pt")
    (d / "emotion_directions.pt.json").write_text(json.dumps(sc, indent=2))
    print(f"exported {len(new_names)} entity vectors to the `dmt` palette group: {new_names}")


if __name__ == "__main__":
    main()
