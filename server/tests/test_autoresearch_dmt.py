"""Static tests for the DMT autoresearch subsystem — no M / AV / MPS required.

Covers the pieces that don't need the model:
  - the DMT feature checklist is well-formed,
  - the feature-judge reply parser (JSON + validation + substring fallback),
  - the crossover override picks the top-scoring direction as a parent,
  - DMT export writes the `dmt` group with score-ranked lineage,
  - CRITICAL: the off-manifold (`research`) and DMT (`dmt`) exports COEXIST in
    emotion_directions.pt — neither clobbers the other's rows or the base
    emotions, and re-exporting one group leaves the other intact.

Run from server/ with:  uv run python -m tests.test_autoresearch_dmt
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cells_interlinked.pipeline import autoresearch_base as arb  # noqa: E402
from cells_interlinked.pipeline.autoresearch import AutoresearchController  # noqa: E402
from cells_interlinked.pipeline.autoresearch_base import STEER_LAYER  # noqa: E402
from cells_interlinked.pipeline.autoresearch_dmt import DmtController  # noqa: E402
from cells_interlinked.pipeline.dmt_features import (  # noqa: E402
    DMT_FEATURES, FEATURE_IDS, N_FEATURES, features_block,
)

_passed = 0
_failed = 0


def check(name: str, fn) -> None:
    global _passed, _failed
    try:
        fn()
    except AssertionError as e:
        _failed += 1
        print(f"FAIL  {name}: {e}")
    except Exception:
        _failed += 1
        print(f"ERROR {name}: {traceback.format_exc()}")
    else:
        _passed += 1
        print(f"ok    {name}")


D = 8
L1 = STEER_LAYER + 4


def _make_palette(td: str):
    d_dir = Path(td)
    edir = torch.randn(2, L1, D)
    torch.save(edir, d_dir / "emotion_directions.pt")
    (d_dir / "emotion_directions.pt.json").write_text(json.dumps({
        "emotions": ["awe", "joy"], "uncharted": [],
    }))
    arb.settings = SimpleNamespace(db_path=d_dir / "probes.sqlite")
    return d_dir


def _dmt_ctrl():
    c = DmtController(app=None)
    c.atlas = [
        {"id": "awe", "generator": "seed", "score": 5, "best_alpha": 1.0,
         "matched_features": ["awe_reverence"], "parents": []},
        {"id": "gen3_crossover", "generator": "crossover", "score": 9, "best_alpha": 2.0,
         "matched_features": ["ego_dissolution", "fractal_geometry"], "parents": ["awe", "joy"]},
        {"id": "gen5_inject", "generator": "inject", "score": 7, "best_alpha": 3.0,
         "matched_features": ["entity_nonhuman"], "parents": []},
    ]
    c._vectors = {e["id"]: torch.randn(D) for e in c.atlas}
    return c


def _off_ctrl():
    c = AutoresearchController(app=None)
    c.atlas = [
        {"id": "awe", "generator": "seed", "off_ortho": 0.9, "alpha_star": 1.0, "parents": []},
        {"id": "gen2_inject", "generator": "inject", "off_ortho": 0.8, "alpha_star": 2.0, "parents": []},
    ]
    c._vectors = {e["id"]: torch.randn(D) for e in c.atlas}
    return c


# ── DMT feature checklist ────────────────────────────────────────
def t_features_well_formed() -> None:
    ids = [f["id"] for f in DMT_FEATURES]
    assert len(ids) == len(set(ids)), "duplicate feature ids"
    assert len(ids) == N_FEATURES == len(FEATURE_IDS)
    assert 20 <= N_FEATURES <= 40, f"unexpected feature count {N_FEATURES}"
    for f in DMT_FEATURES:
        assert f["id"] and f["label"] and f["description"], f
    block = features_block()
    assert block.count("\n") == N_FEATURES - 1, "one feature per line"
    assert "fractal_geometry" in block and "ego_dissolution" in block


# ── judge reply parser ───────────────────────────────────────────
def t_parse_json() -> None:
    ids, ok = DmtController._parse_feature_ids('["fractal_geometry","ego_dissolution"]')
    assert ok and ids == {"fractal_geometry", "ego_dissolution"}, ids


def t_parse_drops_unknown() -> None:
    ids, ok = DmtController._parse_feature_ids('["fractal_geometry","not_a_feature","awe_reverence"]')
    assert ok and ids == {"fractal_geometry", "awe_reverence"}, ids


def t_parse_embedded_json() -> None:
    ids, ok = DmtController._parse_feature_ids('Sure! ["void_blackness"] is present.')
    assert ok and ids == {"void_blackness"}, ids


def t_parse_empty() -> None:
    ids, ok = DmtController._parse_feature_ids("[]")
    assert ok and ids == set(), ids


def t_parse_fallback_substring() -> None:
    ids, ok = DmtController._parse_feature_ids("clearly shows ego_dissolution and ineffability")
    assert not ok and ids == {"ego_dissolution", "ineffability"}, ids


# ── crossover override ───────────────────────────────────────────
def t_crossover_uses_top_scorer() -> None:
    c = _dmt_ctrl()
    c._ref_mag = 1.0
    c.generation = 0
    res = c._crossover()
    assert res is not None
    v, parents, kind = res
    assert kind == "crossover"
    # parent a must be the highest-scoring committed direction (gen3_crossover, 9).
    assert parents[0] == "gen3_crossover", parents
    assert parents[1] != parents[0]


# ── DMT export ───────────────────────────────────────────────────
def t_dmt_export_group_and_lineage() -> None:
    with tempfile.TemporaryDirectory() as td:
        d_dir = _make_palette(td)
        res = _dmt_ctrl().export_to_palette(top_n=8)
        assert res["ok"], res
        sc = json.loads((d_dir / "emotion_directions.pt.json").read_text())
        # 2 discovered (seed excluded), ranked by score desc, named dmt-N.
        assert sc["dmt"] == ["dmt-1", "dmt-2"], sc["dmt"]
        meta = sc["dmt_meta"]
        assert meta["dmt-1"]["atlas_id"] == "gen3_crossover", meta  # score 9
        assert meta["dmt-2"]["atlas_id"] == "gen5_inject", meta     # score 7
        m = meta["dmt-1"]
        assert set(m) >= {"atlas_id", "score", "best_alpha", "matched_features", "parents", "generator"}, m
        assert m["score"] == 9 and m["matched_features"] == ["ego_dissolution", "fractal_geometry"]
        # vector placed at STEER_LAYER.
        edir = torch.load(d_dir / "emotion_directions.pt", weights_only=False)
        assert edir.shape[0] == 4, edir.shape  # 2 base + 2 dmt


def t_exports_coexist() -> None:
    with tempfile.TemporaryDirectory() as td:
        d_dir = _make_palette(td)
        _off_ctrl().export_to_palette(top_n=8)   # writes research-*
        _dmt_ctrl().export_to_palette(top_n=8)   # writes dmt-*
        sc = json.loads((d_dir / "emotion_directions.pt.json").read_text())
        names = sc["emotions"]
        assert "awe" in names and "joy" in names, "base emotions dropped"
        assert sc["research"] == ["research-1"], sc.get("research")  # 1 off-manifold discovered
        assert sc["dmt"] == ["dmt-1", "dmt-2"], sc.get("dmt")
        assert "research-1" in names and "dmt-1" in names and "dmt-2" in names
        edir = torch.load(d_dir / "emotion_directions.pt", weights_only=False)
        assert edir.shape[0] == len(names) == 5, (edir.shape, names)  # 2 base + 1 research + 2 dmt
        # Re-export off-manifold — DMT rows must survive untouched.
        _off_ctrl().export_to_palette(top_n=8)
        sc2 = json.loads((d_dir / "emotion_directions.pt.json").read_text())
        assert sc2["dmt"] == ["dmt-1", "dmt-2"], "DMT group clobbered by off-manifold re-export"
        assert sc2["dmt_meta"]["dmt-1"]["atlas_id"] == "gen3_crossover"
        assert "awe" in sc2["emotions"] and "joy" in sc2["emotions"]


def main() -> int:
    tests = [
        ("features: checklist well-formed", t_features_well_formed),
        ("parse: JSON array", t_parse_json),
        ("parse: drops unknown ids", t_parse_drops_unknown),
        ("parse: embedded JSON", t_parse_embedded_json),
        ("parse: empty array", t_parse_empty),
        ("parse: substring fallback", t_parse_fallback_substring),
        ("crossover: uses top scorer as parent", t_crossover_uses_top_scorer),
        ("export: dmt group + score-ranked lineage", t_dmt_export_group_and_lineage),
        ("export: research + dmt coexist (no clobber)", t_exports_coexist),
    ]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{_passed} passed, {_failed} failed")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
