"""Quick check-in on the DMT dimension-hunt (read-only, no M, safe while the search runs).

Reports whether today's inject-heavy / strict-distinct run is finding NEW productive
DMT *dimensions*: effective dimensionality of the productive directions, the axis
structure (cosine to the leader), and which directions are committed since the hunt
baseline. On first run it writes the baseline (/tmp/dmt_hunt_baseline.json); later
runs diff against it so you can see eff-dim and the off-axis set move.

    cd server
    uv run python -m cells_interlinked.scripts.dmt_dimension_check
"""

from __future__ import annotations

import json
import os

import torch

from ..config import settings

BASELINE = "/tmp/dmt_hunt_baseline.json"
SCORE_FLOOR = 1.0   # "productive" directions for the eff-dim read (ignore weak map points)
OFFAXIS_COS = 0.80  # a direction is "off the dominant axis" if cos-to-leader < this


def _u(v):
    return v / (v.norm() + 1e-8)


def _eff_dim(ids, vecs):
    if len(ids) < 2:
        return float(len(ids))
    M = torch.stack([vecs[i] for i in ids], 0)
    C = M - M.mean(0, keepdim=True)
    ev = torch.linalg.svdvals(C) ** 2
    return float((ev.sum() ** 2) / (ev.pow(2).sum() + 1e-12))


def main() -> None:
    d = settings.db_path.parent / "atlas_dmt"
    blob = json.loads((d / "atlas.json").read_text())
    atlas = blob["atlas"]
    gen = blob.get("generation", 0)
    frontier = blob.get("frontier", 0)

    def vec(e):
        return _u(torch.load(d / "vectors" / f"{e['id']}.pt", weights_only=False).float().reshape(-1))

    prod = [e for e in sorted(atlas, key=lambda e: -e["score"]) if e["score"] >= SCORE_FLOOR]
    vecs = {e["id"]: vec(e) for e in prod}
    ids = list(vecs)
    leader = ids[0] if ids else None
    eff = _eff_dim(ids, vecs)
    offaxis = [e for e in prod if leader and float(vecs[e["id"]] @ vecs[leader]) < OFFAXIS_COS]

    print("=" * 64)
    print(f"DMT dimension-hunt check  |  gen {gen}  |  atlas {len(atlas)}  |  frontier {frontier}")
    print("=" * 64)
    print(f"productive directions (score≥{SCORE_FLOOR}): {len(prod)}")
    print(f"effective dimensionality of productive set: {eff:.2f}")
    print(f"leader: {leader} ({prod[0]['score'] if prod else '?'})")
    print(f"\noff-dominant-axis productive directions (cos-to-leader < {OFFAXIS_COS}): {len(offaxis)}")
    for e in offaxis:
        print(f"  {e['id']:28s} score={e['score']}  cos-to-leader={float(vecs[e['id']]@vecs[leader]):+.2f}  feats={e.get('matched_features')}")

    base = json.load(open(BASELINE)) if os.path.exists(BASELINE) else None
    if base is None:
        json.dump({"gen": gen, "atlas": len(atlas), "eff_dim": round(eff, 2),
                   "offaxis": len(offaxis), "ids": [e["id"] for e in atlas]},
                  open(BASELINE, "w"), indent=2)
        print(f"\n[baseline written to {BASELINE}]")
        return

    new_ids = [e for e in atlas if e["id"] not in set(base.get("ids", []))]
    print("\n── since hunt baseline ──")
    print(f"  gen: {base['gen']} → {gen}  (+{gen - base['gen']} candidates screened)")
    print(f"  atlas: {base['atlas']} → {len(atlas)}  (+{len(atlas) - base['atlas']} committed)")
    print(f"  eff-dim: {base['eff_dim']} → {eff:.2f}  ({'↑ NEW STRUCTURE' if eff > base['eff_dim'] + 0.15 else 'flat — no new axes yet'})")
    print(f"  off-axis productive: {base['offaxis']} → {len(offaxis)}")
    if new_ids:
        print("  newly committed:")
        for e in sorted(new_ids, key=lambda e: -e["score"]):
            c = f"{float(vec(e) @ vecs[leader]):+.2f}" if leader else "n/a"
            print(f"    {e['id']:26s} score={e['score']} cos-to-leader={c} {e.get('matched_features')}")
    else:
        print("  (no new commits yet)")


if __name__ == "__main__":
    main()
