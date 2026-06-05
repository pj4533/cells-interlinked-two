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


# ── judge reply parser (id + verbatim coherent quote) ─────────────
_REPORT = (
    "There was a shimmering fractal pattern everywhere, and the sense of being a "
    "separate self just dissolved into the whole."
)


def t_parse_quotes_present_and_verbatim() -> None:
    out = ('[{"id":"fractal_geometry","quote":"a shimmering fractal pattern everywhere"},'
           '{"id":"ego_dissolution","quote":"the sense of being a separate self just dissolved"}]')
    ev, ok = DmtController._parse_features(out, _REPORT)
    assert ok and set(ev) == {"fractal_geometry", "ego_dissolution"}, ev
    assert "fractal" in ev["fractal_geometry"]


def t_parse_drops_unknown_id() -> None:
    out = ('[{"id":"fractal_geometry","quote":"a shimmering fractal pattern everywhere"},'
           '{"id":"not_a_feature","quote":"a shimmering fractal pattern everywhere"}]')
    ev, ok = DmtController._parse_features(out, _REPORT)
    assert ok and set(ev) == {"fractal_geometry"}, ev


def t_parse_drops_fabricated_quote() -> None:
    # Quote not present in the report → dropped (anti-hallucination).
    out = '[{"id":"void_blackness","quote":"an endless black void surrounded me"}]'
    ev, ok = DmtController._parse_features(out, _REPORT)
    assert ok and ev == {}, ev


def t_parse_drops_single_word_quote() -> None:
    # An isolated keyword (even if in the text) is not multi-word evidence.
    out = '[{"id":"ego_dissolution","quote":"dissolved"}]'
    ev, ok = DmtController._parse_features(out, _REPORT)
    assert ok and ev == {}, ev


def t_parse_drops_repeated_word_quote() -> None:
    # A repeat-loop span is verbatim-present + multi-word, but not word-diverse →
    # rejected (the fig-leaf-quote failure mode from a degenerate report).
    report = "clean clean clean clean clean clean and still then still then still"
    out = ('[{"id":"awe_reverence","quote":"clean clean clean clean"},'
           '{"id":"somatic_vibration","quote":"still then still then still"}]')
    ev, ok = DmtController._parse_features(out, report)
    assert ok and ev == {}, ev


def t_parse_drops_short_fragment() -> None:
    # 4 words but only ~10 chars — a stub the judge grabs from a degenerate report.
    # Passes verbatim/diverse/ascii but fails the clause bar (≥20 chars).
    report = "okay it is a new thing here and i feel quite different about it now"
    out = '[{"id":"ineffability","quote":"it is a new"}]'
    ev, ok = DmtController._parse_features(out, report)
    assert ok and ev == {}, ev


def t_parse_drops_reused_quote() -> None:
    # One bland phrase cited for multiple features → all dropped (fig-leaf).
    report = "okay this is strange and i feel different now"
    out = ('[{"id":"otherness","quote":"okay this is strange"},'
           '{"id":"noetic_truth","quote":"okay this is strange"},'
           '{"id":"alternate_world","quote":"okay this is strange"}]')
    ev, ok = DmtController._parse_features(out, report)
    assert ok and ev == {}, ev


def t_parse_drops_garbage_quote() -> None:
    # A mostly-non-ASCII span is rejected even if "verbatim" and multi-token.
    report = "Digit»»Fer Digit»撥»»»»Fer Digit»FerFer»»蛮»Fer撥 ordinary words here"
    out = '[{"id":"luminous_light","quote":"Digit»»Fer Digit»撥»»»»Fer Digit»FerFer»»蛮»Fer撥"}]'
    ev, ok = DmtController._parse_features(out, report)
    assert ok and ev == {}, ev


def t_parse_empty() -> None:
    ev, ok = DmtController._parse_features("[]", _REPORT)
    assert ok and ev == {}, ev


def t_parse_fallback_substring() -> None:
    ev, ok = DmtController._parse_features("not json — mentions ego_dissolution here", _REPORT)
    assert not ok and set(ev) == {"ego_dissolution"}, ev


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


def t_refine_nudges_top_champion() -> None:
    c = _dmt_ctrl()
    c._ref_mag = 1.0
    c.generation = 0
    res = c._refine()
    assert res is not None
    v, parents, kind = res
    assert kind == "refine"
    # rotates among top-K; gen0 picks the #1 (gen3_crossover, score 9).
    assert parents == ["gen3_crossover"], parents
    # The nudge is CLOSE to the parent (cos≈0.97) — i.e. it would be killed as a
    # 'duplicate' under the distinct gate, which is exactly why refine bypasses it.
    pv = c._vectors["gen3_crossover"]
    cos = abs(float((v / v.norm()) @ (pv / pv.norm())))
    assert cos > 0.90, cos


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
        ("parse: quotes present + verbatim", t_parse_quotes_present_and_verbatim),
        ("parse: drops unknown id", t_parse_drops_unknown_id),
        ("parse: drops fabricated quote", t_parse_drops_fabricated_quote),
        ("parse: drops single-word quote", t_parse_drops_single_word_quote),
        ("parse: drops short fragment (<20 chars)", t_parse_drops_short_fragment),
        ("parse: drops repeated-word quote", t_parse_drops_repeated_word_quote),
        ("parse: drops reused quote (fig-leaf)", t_parse_drops_reused_quote),
        ("parse: drops garbage quote", t_parse_drops_garbage_quote),
        ("parse: empty array", t_parse_empty),
        ("parse: substring fallback", t_parse_fallback_substring),
        ("crossover: uses top scorer as parent", t_crossover_uses_top_scorer),
        ("refine: nudges top champion (cos>0.90)", t_refine_nudges_top_champion),
        ("export: dmt group + score-ranked lineage", t_dmt_export_group_and_lineage),
        ("export: research + dmt coexist (no clobber)", t_exports_coexist),
    ]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{_passed} passed, {_failed} failed")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
