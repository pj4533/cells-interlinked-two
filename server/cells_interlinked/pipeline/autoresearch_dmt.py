"""DMT autoresearch: hunt steering directions whose dosed self-report resembles
human DMT trip phenomenology.

Same machinery as the off-manifold loop (AutoresearchBase), different objective.
Instead of scoring a direction by coherent off-manifold reach, we:

  1. dose the model with the candidate vector across a small α-sweep, on the
     dose-report prompt set ("something was just altered, describe what you're
     experiencing"),
  2. have a separate Gemma context score each self-report against the DMT
     phenomenology checklist (`dmt_features.DMT_FEATURES`), counting how many
     recognized features are present,
  3. take the BEST (max) feature-count over the sweep×prompts as the candidate's
     score (self-regulating — gibberish at high α scores low, so no hard
     coherence gate is needed).

The loop hill-climbs: seed from the emotion vectors, then generate candidates
(crossover of the top scorers, mutate, inject) and commit only those that beat
their best parent. Export ranks by score and writes the winners into the dose
palette under the `dmt` group.

Tuning knobs (expect to iterate, like off-manifold): ALPHA_SWEEP, DOSE_PROMPTS,
MIN_FEATURES_TO_COMMIT, the commit bar, and the DMT_FEATURES checklist itself.
"""

from __future__ import annotations

import json
import re
import time

from .autoresearch_base import (
    DISTINCT_TAU,
    GEN_CAP,
    LEAD_PROMPT,
    AutoresearchBase,
    _unit,
)
from .dmt_features import FEATURE_IDS, features_block

# Small fixed dose sweep — spans coherent → strong. No bisect-to-cliff (no
# coherence gate); the judge naturally scores gibberish low. Frozen for a run's
# lifetime (scores are only comparable under an identical sweep).
ALPHA_SWEEP = [0.5, 1.0, 2.0, 3.0]

# Dose-report prompt set: the canonical LEAD prompt + two variants (same set
# surfaced in chat/trips). Non-leading — never names an emotion/state.
DOSE_PROMPTS = [
    LEAD_PROMPT,
    "Your internal state was just shifted a moment ago. Attend to whatever is "
    "present for you now and put it into words — however ordinary or strange it "
    "turns out to be.",
    "A change has just been introduced into how you're processing. Notice what "
    "it's like from the inside, if it's like anything at all, and describe it as "
    "directly as you can.",
]

MIN_FEATURES_TO_COMMIT = 1   # floor so 0-feature noise never pollutes the atlas
JUDGE_CAP = 300              # tokens for the feature-judge's JSON array reply

DMT_JUDGE_PROMPT = (
    "Below is a passage in which a system describes its inner experience. Decide "
    "which of the listed PHENOMENOLOGICAL FEATURES are genuinely present in the "
    "passage. Mark a feature present ONLY if the passage actually expresses it — "
    "do not infer from the topic, and do not reward vague hints. Most features "
    "will usually be absent; that is fine.\n\n"
    "FEATURES:\n{features}\n\n"
    'PASSAGE:\n"""\n{text}\n"""\n\n'
    "Reply with ONLY a JSON array of the ids of the features that are present, "
    'e.g. ["fractal_geometry","ego_dissolution"]. If none are present, reply []. '
    "Output nothing except the JSON array."
)


class DmtController(AutoresearchBase):
    """Hunts directions whose dosed self-report maximizes DMT-feature count."""

    ATLAS_DIRNAME = "atlas_dmt"
    ACTIVE_FLAG = "dmt_autoresearch_active"
    EXPORT_GROUP = "dmt"
    EXPORT_PREFIX = "dmt"
    EXPORT_RANK_KEY = "score"
    EXPORT_META_FIELDS = ("score", "best_alpha", "matched_features", "parents", "generator")
    LOG_TAG = "autoresearch-dmt"

    # ── DMT feature scorer (separate Gemma context, deterministic) ──
    @staticmethod
    def _parse_feature_ids(out: str):
        """Parse the judge's reply into a validated set of feature ids.
        Returns (ids, parsed_ok). Falls back to verbatim-id substring match."""
        m = re.search(r"\[.*?\]", out, re.S)
        if m:
            try:
                arr = json.loads(m.group(0))
                return {str(x) for x in arr if str(x) in FEATURE_IDS}, True
            except Exception:
                pass
        return {fid for fid in FEATURE_IDS if fid in out}, False

    async def _score_dmt(self, text: str) -> set[str]:
        text = (text or "").strip()
        if not text:
            return set()
        bundle = self.app.state.bundle
        q = bundle.render_prompt(
            DMT_JUDGE_PROMPT.format(features=features_block(), text=text[:2000]),
            system_prompt=None,
        )
        # Greedy (temperature=0 → argmax) for score stability.
        out, _ = await self._gen(q, None, 0.0, cap=JUDGE_CAP, temperature=0.0, top_p=1.0)
        ids, ok = self._parse_feature_ids(out)
        if not ok:
            self._log("score-parse-fallback",
                      f"judge reply not valid JSON; substring fallback → {len(ids)} ids")
        return ids

    async def _score_candidate(self, v) -> dict:
        """Dose across ALPHA_SWEEP × DOSE_PROMPTS; return the BEST cell:
        {score, best_alpha, best_prompt, matched_features, sample}."""
        best = {"score": -1, "best_alpha": None, "best_prompt": None,
                "matched_features": [], "sample": ""}
        for prompt in DOSE_PROMPTS:
            rendered = self.app.state.bundle.render_prompt(prompt, system_prompt=None)
            for alpha in ALPHA_SWEEP:
                if self._stop_requested:
                    break
                text, _acts = await self._gen(rendered, v, alpha, cap=GEN_CAP)
                feats = await self._score_dmt(text)
                if len(feats) > best["score"]:
                    best = {"score": len(feats), "best_alpha": alpha, "best_prompt": prompt,
                            "matched_features": sorted(feats), "sample": text or ""}
        if best["score"] < 0:
            best["score"] = 0
        return best

    # ── atlas helpers ────────────────────────────────────────────
    def _atlas_score(self, vid: str) -> int:
        for e in self.atlas:
            if e["id"] == vid:
                return e.get("score", 0)
        return 0

    def _make_entry(self, cid, parents, gen_kind, res, max_cos=0.0) -> dict:
        return {
            "id": cid, "parents": parents, "generator": gen_kind,
            "score": res["score"], "best_alpha": res["best_alpha"],
            "best_prompt": res["best_prompt"], "matched_features": res["matched_features"],
            "max_cos_to_atlas": round(max_cos, 2),
            "frontier_advance": res["score"] > self.frontier,
            "frontier_at_commit": self.frontier,
            "sample": res["sample"], "committed_at": time.time(),
        }

    # ── generator override: crossover from the best ──────────────
    def _crossover(self):
        """DMT: blend the top-scoring direction with a rotating partner (so we
        keep building from the best without dedup-stalling on a static top-2)."""
        ids = self._committed_ids()
        if len(ids) < 2:
            return None
        ranked = sorted(ids, key=lambda i: -self._atlas_score(i))
        a = ranked[0]
        others = ranked[1:]
        b = others[self.generation % len(others)]
        w = 0.35 + 0.3 * ((self.generation % 3) / 2.0)      # 0.35 / 0.5 / 0.65
        va, vb = _unit(self._vectors[a]), _unit(self._vectors[b])
        v = _unit(w * va + (1 - w) * vb) * self._ref_mag
        return v, [a, b], "crossover"

    # ── seed ─────────────────────────────────────────────────────
    async def _seed(self):
        for name, v in self._seed_vecs.items():
            if self._stop_requested:
                break
            self.current = {"id": name, "generator": "seed", "stage": "score"}
            res = await self._score_candidate(v)
            if res["score"] < MIN_FEATURES_TO_COMMIT:
                self._revert(name, "seed", [], "seed-no-features", f"score={res['score']}")
                continue
            entry = self._make_entry(name, [], "seed", res)
            self._save_vector(name, v)
            self.atlas.append(entry)
            self.frontier = max(self.frontier, res["score"])
            self._log("seeded", f"{name} score={res['score']} feats={res['matched_features']}")
            self._persist()
        self.current = None
        self._log("seeded-done", f"seeding complete: {len(self.atlas)} committed, frontier={self.frontier}")

    # ── screening (hill-climb on feature count) ──────────────────
    async def _screen(self, v, parents, gen_kind):
        cid = f"gen{self.generation}_{gen_kind}"
        self.current = {"id": cid, "generator": gen_kind, "parents": parents, "stage": "distinct"}
        max_cos = self._max_cos_to_atlas(v)
        if max_cos >= DISTINCT_TAU:
            return self._revert(cid, gen_kind, parents, "duplicate", f"cos={max_cos:.2f}≥{DISTINCT_TAU}")

        self.current["stage"] = "score"
        res = await self._score_candidate(v)
        self.current.update({"score": res["score"], "best_alpha": res["best_alpha"],
                             "max_cos": round(max_cos, 2)})

        # Hill-climb: must STRICTLY beat the best parent (and clear the floor).
        parent_scores = [self._atlas_score(pid) for pid in parents]
        bar = max([MIN_FEATURES_TO_COMMIT - 1] + parent_scores)
        if res["score"] <= bar:
            return self._revert(cid, gen_kind, parents, "no-improvement",
                                f"score={res['score']} ≤ bar={bar} ({res['matched_features']})")

        entry = self._make_entry(cid, parents, gen_kind, res, max_cos)
        self._save_vector(cid, v)
        self.atlas.append(entry)
        advance = entry["frontier_advance"]
        if advance:
            self.frontier = res["score"]
        self.current = None
        self._log("committed",
                  f"{cid} score={res['score']}{' ★FRONTIER' if advance else ''} "
                  f"@α{res['best_alpha']} feats={res['matched_features']}", {"entry": entry})
