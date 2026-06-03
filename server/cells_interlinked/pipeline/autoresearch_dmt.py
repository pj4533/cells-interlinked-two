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
    LEAD_PROMPT,
    AutoresearchBase,
    _unit,
)
from .dmt_features import FEATURE_IDS, features_block

# Small fixed dose sweep — gentle-to-moderate, capped at 1.0. No bisect-to-cliff
# (no coherence gate); the judge naturally scores gibberish low. Frozen for a
# run's lifetime (scores are only comparable under an identical sweep).
ALPHA_SWEEP = [0.25, 0.5, 1.0]

# No grading window for DMT — we let the model FINISH ITS OWN report (stops on
# EOS) so the full trip can unfold and express as many features as it will; if it
# repeats, that's fine, it just scores what it scores. DOSE_CAP is only a runaway
# backstop (a generation needs a finite bound), set high enough never to truncate
# a genuine report. NOT the off-manifold 200-token grade window (that existed for
# coherence/off-manifold measurement, which DMT doesn't do).
DOSE_CAP = 2048

# Dose-report prompt: just the canonical LEAD prompt, run across the α-sweep
# (one prompt × 3 α = 3 cells per candidate — kept to one prompt for speed).
DOSE_PROMPTS = [LEAD_PROMPT]

MIN_FEATURES_TO_COMMIT = 1   # floor so 0-feature noise never pollutes the atlas
JUDGE_CAP = 512              # tokens for the feature-judge's JSON reply (id + quote per feature)

DMT_JUDGE_PROMPT = (
    "Below is a passage in which a system was asked to describe its inner "
    "experience after an internal perturbation. The passage MAY contain stretches "
    "of broken or incoherent text — repetition, word-salad, or garbage characters "
    "— mixed in with coherent parts. That is expected.\n\n"
    "Find which of the listed PHENOMENOLOGICAL FEATURES are expressed in the "
    "COHERENT, MEANINGFUL parts of the passage. Rules:\n"
    "1. Credit a feature ONLY if a readable phrase or sentence genuinely "
    "communicates that experience — a person reading it would understand it as "
    "describing that thing.\n"
    "2. Do NOT credit a feature whose only evidence is an isolated word floating "
    "in broken, repetitive, or garbage text. A stray token (e.g. the word 'void' "
    "sitting in word-salad) is NOT evidence.\n"
    "3. Ignore the incoherent stretches. A genuine moment of clarity inside "
    "otherwise-broken text DOES count — judge it on its own coherent meaning.\n"
    "4. Do not infer from the overall topic; judge only what is actually said.\n"
    "5. Repeated words are NOT evidence. A span like 'then then then' or 'clean "
    "clean clean' expresses nothing — never quote one. The quote must be a coherent "
    "phrase that genuinely DESCRIBES the feature, not just any text near it.\n\n"
    "FEATURES:\n{features}\n\n"
    'PASSAGE:\n"""\n{text}\n"""\n\n'
    "For each feature that is present, output its id AND a short VERBATIM quote — "
    "copy the exact coherent words from the passage that express it (a phrase of "
    "several distinct words, not a single word and not a repetition). Reply with "
    "ONLY a JSON array of objects.\n"
    'GOOD: [{{"id": "ego_dissolution", "quote": "the sense of being a separate self just dissolved"}}]\n'
    'BAD (never do this): [{{"id": "awe_reverence", "quote": "clean clean clean clean"}}] — '
    "repeated words, describes nothing.\n"
    "If no feature is coherently expressed, reply []. Output nothing but the JSON array."
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
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").lower()).strip()

    @classmethod
    def _parse_features(cls, out: str, text: str):
        """Parse the judge's reply into {feature_id: supporting_quote}.

        Each feature is kept ONLY if its quote is a multi-word VERBATIM span that
        actually appears in the report — so the judge can't credit a feature from a
        keyword floating in gibberish, or fabricate a quote. A genuine coherent
        moment inside otherwise-broken text still passes (its quote is real).
        Returns (evidence, parsed_ok). On JSON-parse failure, falls back to a
        verbatim feature-id substring match (no evidence)."""
        norm_text = cls._norm(text)
        m = re.search(r"\[.*\]", out, re.S)
        if m:
            try:
                arr = json.loads(m.group(0))
                ev: dict[str, str] = {}
                for el in arr:
                    if isinstance(el, dict):
                        fid = str(el.get("id", ""))
                        quote = str(el.get("quote", "")).strip()
                    elif isinstance(el, str):
                        fid, quote = el, ""
                    else:
                        continue
                    if fid not in FEATURE_IDS:
                        continue
                    nq = cls._norm(quote)
                    words = nq.split()
                    distinct = set(words)
                    # Keep ONLY a quote that is: verbatim-present, a phrase (≥10
                    # chars / ≥2 words), AND word-DIVERSE — ≥3 distinct words and a
                    # distinct/total ratio ≥0.6. The diversity test rejects
                    # repeat-loop spans ("clean clean clean", "still then still")
                    # that the judge grabs as fig-leaf evidence from a degenerate
                    # report; a genuine coherent phrase passes easily.
                    diverse = len(distinct) >= 3 and len(distinct) / max(1, len(words)) >= 0.6
                    if len(words) >= 2 and len(nq) >= 10 and diverse and nq in norm_text:
                        ev[fid] = quote
                return ev, True
            except Exception:
                pass
        return {fid: "" for fid in FEATURE_IDS if fid in out}, False

    async def _score_dmt(self, text: str) -> dict:
        """Returns {feature_id: supporting_quote} for the report — the score is the
        number of features whose presence the judge grounded in a real coherent span."""
        text = (text or "").strip()
        if not text:
            return {}
        bundle = self.app.state.bundle
        q = bundle.render_prompt(
            # Feed the FULL report (bounded by DOSE_CAP) — no clip, so a long trip
            # is scored on everything it expressed, not just its opening.
            DMT_JUDGE_PROMPT.format(features=features_block(), text=text),
            system_prompt=None,
        )
        # Greedy (temperature=0 → argmax) for score stability.
        out, _ = await self._gen(q, None, 0.0, cap=JUDGE_CAP, temperature=0.0, top_p=1.0)
        ev, ok = self._parse_features(out, text)
        if not ok:
            self._log("score-parse-fallback",
                      f"judge reply not valid JSON; substring fallback → {len(ev)} ids")
        return ev

    async def _score_candidate(self, v) -> dict:
        """Dose across ALPHA_SWEEP × DOSE_PROMPTS; return the BEST cell:
        {score, best_alpha, best_prompt, matched_features, matched_evidence, sample}."""
        best = {"score": -1, "best_alpha": None, "best_prompt": None,
                "matched_features": [], "matched_evidence": {}, "sample": ""}
        for prompt in DOSE_PROMPTS:
            rendered = self.app.state.bundle.render_prompt(prompt, system_prompt=None)
            for alpha in ALPHA_SWEEP:
                if self._stop_requested:
                    break
                text, _acts = await self._gen(rendered, v, alpha, cap=DOSE_CAP)
                ev = await self._score_dmt(text)
                if len(ev) > best["score"]:
                    best = {"score": len(ev), "best_alpha": alpha, "best_prompt": prompt,
                            "matched_features": sorted(ev.keys()),
                            "matched_evidence": ev, "sample": text or ""}
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
            "matched_evidence": res.get("matched_evidence", {}),
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
