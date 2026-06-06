"""Extract matched-contrast DMT seeds + the atlas-derived direction, and validate
them against the current atlas leader BEFORE any long run.

Three products, all appended to emotion_directions.pt as new rows (additive — the
existing seeds and the atlas are untouched; re-runnable, drops prior
feat-mc_*/feat-atlas_* rows first):

  • feat-mc_<trait>   minimal-pair diff-of-means (trait POS − matched NEG at L20),
                      one per trait in dmt_matched_seeds.MATCHED_TRAIT_PAIRS.
  • feat-mc_composite full-DMT-trip POS − vivid-but-non-DMT NEG (the single
                      multi-trait "DMT-bundle" direction).
  • feat-atlas_winners  score-weighted diff over the committed DMT atlas
                      (direction derived from what has actually scored well).

Validation (printed, no long run needed): for each new direction, cosine to the
atlas LEADER (highest-scoring committed direction), to `sublime` (the axis the
first batch collapsed onto), and to its nearest emotion. High cos-to-leader +
low cos-to-sublime = we captured real DMT structure (convergent validity).

Run with the backend STOPPED (loads its own M):

    cd server
    uv run python -m cells_interlinked.scripts.compute_dmt_matched_seeds
"""

from __future__ import annotations

import json
import logging

import torch

from ..config import settings
from ..pipeline.dmt_matched_seeds import (
    ATLAS_DERIVED_NAME,
    COMPOSITE_NAME,
    COMPOSITE_NEG,
    COMPOSITE_POS,
    MATCHED_TRAIT_PAIRS,
)
from ..pipeline.model_loader import load_model
from .compute_dmt_feature_seeds import STEER_LAYER, mean_l20

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compute_dmt_matched_seeds")

DOSE_UNIT = 0.4
NEW_PREFIXES = ("feat-mc_", "feat-atlas_")


def _u(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


def atlas_derived_direction(d) -> tuple[torch.Tensor | None, dict | None]:
    """Score-weighted diff over the committed DMT atlas: sum (score - mean) ·
    unit(vector). Returns (unit_direction, leader_entry) or (None, None)."""
    af = d / "atlas_dmt" / "atlas.json"
    if not af.exists():
        logger.warning("no atlas_dmt/atlas.json — skipping atlas-derived direction")
        return None, None
    entries = json.loads(af.read_text()).get("atlas", [])
    vecs, scores, leader = [], [], None
    for e in entries:
        vp = d / "atlas_dmt" / "vectors" / f"{e['id']}.pt"
        if not vp.exists():
            continue
        v = torch.load(vp, weights_only=False).float().reshape(-1)
        vecs.append(_u(v))
        scores.append(float(e.get("score", 0)))
        if leader is None or e.get("score", 0) > leader.get("score", -1):
            leader = e
    if len(vecs) < 4:
        logger.warning("only %d atlas vectors — skipping atlas-derived direction", len(vecs))
        return None, None
    s = torch.tensor(scores)
    w = s - s.mean()
    if float(w.abs().sum()) < 1e-6:
        logger.warning("atlas scores are uniform — skipping atlas-derived direction")
        return None, leader
    direction = torch.stack(vecs, 0)
    d_vec = (w.unsqueeze(1) * direction).sum(0)
    logger.info("atlas-derived: %d entries, score range %.0f–%.0f, leader=%s (score %s)",
                len(vecs), s.min(), s.max(), leader.get("id"), leader.get("score"))
    return _u(d_vec), leader


def main() -> None:
    d = settings.db_path.parent
    pt = d / "emotion_directions.pt"
    js = d / "emotion_directions.pt.json"
    if not pt.exists():
        raise SystemExit(f"missing {pt}")

    logger.info("loading M (%s) …", settings.model_name)
    bundle = load_model(settings.model_name)

    # ── extract matched-contrast directions ──────────────────────
    new_dirs: dict[str, torch.Tensor] = {}   # name -> unit direction (d_model)
    neg_norms: list[float] = []

    for name, pair in MATCHED_TRAIT_PAIRS.items():
        pos_mean, _ = mean_l20(bundle, pair["pos"])
        neg_mean, nn = mean_l20(bundle, pair["neg"])
        neg_norms += nn
        diff = pos_mean - neg_mean
        new_dirs[name] = _u(diff)
        logger.info("  %-32s |diff|=%.1f", name, float(diff.norm()))

    comp_pos, _ = mean_l20(bundle, COMPOSITE_POS)
    comp_neg, cnn = mean_l20(bundle, COMPOSITE_NEG)
    neg_norms += cnn
    comp_diff = comp_pos - comp_neg
    new_dirs[COMPOSITE_NAME] = _u(comp_diff)
    logger.info("  %-32s |diff|=%.1f", COMPOSITE_NAME, float(comp_diff.norm()))

    atlas_dir, leader = atlas_derived_direction(d)
    if atlas_dir is not None:
        new_dirs[ATLAS_DERIVED_NAME] = atlas_dir

    scale = DOSE_UNIT * float(torch.tensor(neg_norms).median())

    # ── load palette + reference vectors for validation ──────────
    edir = torch.load(pt, weights_only=False)  # [E, L1, D]
    sc = json.loads(js.read_text())
    names = list(sc.get("emotions", []))
    idx = {n: i for i, n in enumerate(names)}
    L1, D = edir.shape[1], edir.shape[2]

    sublime = _u(edir[idx["sublime"]][STEER_LAYER].float()) if "sublime" in idx else None
    emo_names = [n for n in ("awe", "joy", "serenity", "love", "excitement",
                             "sublime", "ecstatic", "rapture", "valence") if n in idx]
    emo_vecs = {n: _u(edir[idx[n]][STEER_LAYER].float()) for n in emo_names}
    leader_vec = None
    if leader is not None:
        lp = d / "atlas_dmt" / "vectors" / f"{leader['id']}.pt"
        if lp.exists():
            leader_vec = _u(torch.load(lp, weights_only=False).float().reshape(-1))

    # ── VALIDATION report ────────────────────────────────────────
    print("\n=== validation (cos to leader / sublime / nearest emotion) ===")
    if leader is not None:
        print(f"leader = {leader['id']}  score={leader['score']}  "
              f"feats={leader.get('matched_features')}\n")
    print(f"{'direction':34s} {'leader':>7s} {'sublime':>8s}  nearest-emotion")
    for name, vec in new_dirs.items():
        cl = f"{float(vec @ leader_vec):+.2f}" if leader_vec is not None else "  n/a"
        cs = f"{float(vec @ sublime):+.2f}" if sublime is not None else "  n/a"
        nearest = max(emo_vecs, key=lambda n: float(vec @ emo_vecs[n])) if emo_vecs else "?"
        cn = float(vec @ emo_vecs[nearest]) if emo_vecs else 0.0
        print(f"{name:34s} {cl:>7s} {cs:>8s}  {nearest} ({cn:+.2f})")
    print()

    # ── append rows (re-runnable: drop prior new-prefix rows) ────
    keep = [i for i, n in enumerate(names) if not n.startswith(NEW_PREFIXES)]
    if len(keep) != len(names):
        logger.info("dropping %d existing %s rows", len(names) - len(keep), NEW_PREFIXES)
    edir = edir[keep]
    names = [names[i] for i in keep]
    rows, new_names = [], []
    for name, vec in new_dirs.items():
        row = torch.zeros(L1, D, dtype=edir.dtype)
        row[STEER_LAYER] = (vec * scale).to(edir.dtype)
        rows.append(row)
        new_names.append(name)
    edir = torch.cat([edir, torch.stack(rows, 0)], 0)
    names = names + new_names
    sc["emotions"] = names
    sc["matched_seeds"] = new_names
    sc["matched_seeds_note"] = (
        "Matched-contrast DMT seeds (minimal-pair diff-of-means + composite + "
        "atlas-derived). Internal DmtController.EXTRA_SEEDS; filtered from the "
        "/dose_emotions picker. See pipeline/dmt_matched_seeds.py."
    )
    torch.save(edir, pt)
    js.write_text(json.dumps(sc, indent=2))
    logger.info("wrote %d matched seeds: %s", len(new_names), new_names)
    logger.info("emotion_directions.pt now has %d rows", edir.shape[0])


if __name__ == "__main__":
    main()
