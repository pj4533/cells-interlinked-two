"""Static tests for autoresearch export + memory helpers.

No M, no AV, no MPS required — synthesizes a tiny atlas + emotion palette
on disk and exercises:

  - `export_to_palette` promotes the top-N NON-seed directions by off_ortho,
    writes their vector into row STEER_LAYER, and (the bit we care about for
    provenance) records full LINEAGE under `research_meta`: atlas_id, parents,
    generator, off_ortho, alpha_star for each research-N name.
  - re-export is idempotent: prior `research-*` rows are dropped, not stacked.
  - seeds are never exported.
  - the MPS memory helpers degrade gracefully when MPS is absent.

Run from server/ with:

    uv run python -m tests.test_autoresearch_export
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

from cells_interlinked.pipeline import autoresearch as ar  # noqa: E402
from cells_interlinked.pipeline.autoresearch import (  # noqa: E402
    AutoresearchController,
    STEER_LAYER,
    _free_mps,
    _mps_mem_gib,
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


def _fixture(td: str, n_emotions: int = 2, d: int = 8):
    """Write a minimal emotion_directions.pt (+sidecar) into a temp data dir and
    return (controller, data_dir). The controller's atlas has one seed and three
    discovered directions with distinct off_ortho so ranking is unambiguous."""
    d_dir = Path(td)
    l1 = STEER_LAYER + 4
    edir = torch.randn(n_emotions, l1, d)
    torch.save(edir, d_dir / "emotion_directions.pt")
    (d_dir / "emotion_directions.pt.json").write_text(json.dumps({
        "emotions": ["awe", "joy"], "uncharted": [], "research": [],
    }))
    # Point the module-global settings at our temp dir.
    ar.settings = SimpleNamespace(db_path=d_dir / "probes.sqlite")

    ctrl = AutoresearchController(app=None)
    ctrl.atlas = [
        {"id": "awe", "generator": "seed", "off_ortho": 0.99, "alpha_star": 3.0, "parents": []},
        {"id": "gen10_crossover", "generator": "crossover", "off_ortho": 0.80,
         "alpha_star": 1.1, "parents": ["awe", "joy"]},
        {"id": "gen22_inject", "generator": "inject", "off_ortho": 0.90,
         "alpha_star": 2.0, "parents": []},
        {"id": "gen30_mutate", "generator": "mutate", "off_ortho": 0.70,
         "alpha_star": 0.5, "parents": ["gen10_crossover"]},
    ]
    ctrl._vectors = {e["id"]: torch.randn(d) for e in ctrl.atlas}
    return ctrl, d_dir


def t_export_records_full_lineage() -> None:
    with tempfile.TemporaryDirectory() as td:
        ctrl, d_dir = _fixture(td)
        res = ctrl.export_to_palette(top_n=8)
        assert res["ok"], res
        sc = json.loads((d_dir / "emotion_directions.pt.json").read_text())
        meta = sc["research_meta"]
        # 3 discovered (seed excluded), ranked by off_ortho desc.
        assert sc["research"] == ["research-1", "research-2", "research-3"], sc["research"]
        assert meta["research-1"]["atlas_id"] == "gen22_inject", meta  # 0.90
        assert meta["research-2"]["atlas_id"] == "gen10_crossover", meta  # 0.80
        assert meta["research-3"]["atlas_id"] == "gen30_mutate", meta  # 0.70
        # Every provenance field carried through.
        for rname, m in meta.items():
            assert set(m) >= {"atlas_id", "off_ortho", "alpha_star", "parents", "generator"}, m
        assert meta["research-2"]["parents"] == ["awe", "joy"], meta["research-2"]
        assert meta["research-2"]["generator"] == "crossover"


def t_export_excludes_seeds() -> None:
    with tempfile.TemporaryDirectory() as td:
        ctrl, d_dir = _fixture(td)
        ctrl.export_to_palette(top_n=8)
        sc = json.loads((d_dir / "emotion_directions.pt.json").read_text())
        ids = {m["atlas_id"] for m in sc["research_meta"].values()}
        assert "awe" not in ids, "seed leaked into export"


def t_export_writes_vector_into_steer_layer() -> None:
    with tempfile.TemporaryDirectory() as td:
        ctrl, d_dir = _fixture(td)
        ctrl.export_to_palette(top_n=1)
        edir = torch.load(d_dir / "emotion_directions.pt", weights_only=False)
        # Last row is research-1 (= gen22_inject). Its STEER_LAYER row must be
        # the saved vector; all other layer rows must be zero.
        row = edir[-1]
        assert torch.allclose(row[STEER_LAYER], ctrl._vectors["gen22_inject"].to(row.dtype)), \
            "exported vector not placed at STEER_LAYER"
        mask = torch.ones(row.shape[0], dtype=torch.bool)
        mask[STEER_LAYER] = False
        assert float(row[mask].abs().max()) == 0.0, "non-steer rows should be zero"


def t_reexport_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as td:
        ctrl, d_dir = _fixture(td)
        ctrl.export_to_palette(top_n=8)
        first = torch.load(d_dir / "emotion_directions.pt", weights_only=False)
        ctrl.export_to_palette(top_n=8)
        second = torch.load(d_dir / "emotion_directions.pt", weights_only=False)
        sc = json.loads((d_dir / "emotion_directions.pt.json").read_text())
        # Re-export must not stack stale research rows.
        assert first.shape == second.shape, (first.shape, second.shape)
        assert sc["emotions"].count("research-1") == 1, sc["emotions"]
        assert len([n for n in sc["emotions"] if n.startswith("research-")]) == 3


def t_export_refuses_while_running() -> None:
    with tempfile.TemporaryDirectory() as td:
        ctrl, _ = _fixture(td)
        ctrl._running = True
        res = ctrl.export_to_palette(top_n=8)
        assert not res["ok"] and "stop" in res["error"].lower(), res


def t_export_empty_atlas_errors() -> None:
    with tempfile.TemporaryDirectory() as td:
        ctrl, _ = _fixture(td)
        ctrl.atlas = [{"id": "awe", "generator": "seed", "off_ortho": 0.9,
                       "alpha_star": 1.0, "parents": []}]
        ctrl._vectors = {"awe": torch.randn(8)}
        res = ctrl.export_to_palette(top_n=8)
        assert not res["ok"], "export of seed-only atlas should fail"


def t_mps_helpers_dont_raise() -> None:
    # On a machine without MPS these must degrade to (0, 0) and a no-op, never raise.
    alloc, driver = _mps_mem_gib()
    assert isinstance(alloc, float) and isinstance(driver, float)
    assert alloc >= 0.0 and driver >= 0.0
    _free_mps()  # must not raise


def main() -> int:
    tests = [
        ("export: records full lineage in research_meta", t_export_records_full_lineage),
        ("export: excludes seeds", t_export_excludes_seeds),
        ("export: vector placed at STEER_LAYER only", t_export_writes_vector_into_steer_layer),
        ("export: re-export is idempotent", t_reexport_is_idempotent),
        ("export: refuses while running", t_export_refuses_while_running),
        ("export: seed-only atlas errors", t_export_empty_atlas_errors),
        ("memory: mps helpers degrade gracefully", t_mps_helpers_dont_raise),
    ]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{_passed} passed, {_failed} failed")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
