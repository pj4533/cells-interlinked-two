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

import torch

from .autoresearch_base import (
    DISTINCT_TAU,
    LEAD_PROMPT,
    AutoresearchBase,
    _unit,
)
from .dmt_features import DMT_FEATURES, FEATURE_IDS, features_block
from .dmt_feature_seeds import FEATURE_SEED_NAMES
from .dmt_matched_seeds import MATCHED_SEED_NAMES, SEEDED_MATCHED

_FEATURE_BY_ID = {f["id"]: f for f in DMT_FEATURES}
# All internal seed directions (first feat-* batch + the matched-contrast batch).
# Force-committed in _seed and prioritized as crossover partners in _crossover.
# (MATCHED_SEED_NAMES is the full extracted set; only SEEDED_MATCHED is actually
# seeded — see EXTRA_SEEDS — but membership here is harmless for the unseeded ones
# since they never get committed.)
_FEATURE_SEED_SET = set(FEATURE_SEED_NAMES) | set(MATCHED_SEED_NAMES)

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

MIN_FEATURES_TO_COMMIT = 2   # floor: a committed direction must show ≥2 grounded
                             # features (a single feature is usually the easy
                             # prompt-artifact one). Appends commit on distinct +
                             # this floor — NOT beat-parent (beat-parent is refine-
                             # only), so diverse decent directions are kept as
                             # recombination material instead of deleted for failing
                             # to beat the all-time record.
JUDGE_CAP = 512              # tokens for the feature-judge's JSON reply (id + quote per feature)

# Generator mix — DMT adds `refine` to the base crossover/mutate/inject. The
# diversity gate (DISTINCT_TAU) keeps the atlas a map of DISTINCT directions, but
# that BLOCKS honing a good direction (the finest non-refine move, mutate, lands
# ~0.82 cos away). `refine` is the depth/exploitation counterpart: a SMALL nudge
# of a top-K champion (cos≈0.97), exempt from the distinct gate, committed
# in-place (replace-if-better) so it sharpens an entry rather than adding a
# near-duplicate.
DMT_GEN_WEIGHTS = {"crossover": 0.40, "mutate": 0.20, "refine": 0.25, "inject": 0.15}
REFINE_NOISE = 0.25          # unit(champion + 0.25·noise⊥) → cos ≈ 0.97 (a hone, not a jump)
TOP_K_REFINE = 3             # refine rotates among the top-K highest-scoring directions

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

# Verification pass — a fresh judge checks each proposed (feature, quote) pair
# ONE AT A TIME. Pass-1 ("which of 31 are present?") over-attributes, and a
# BATCHED "confirm this list" verifier rubber-stamps (it waved through fragments
# like "it is a new" → ineffability). A focused, isolated yes/no per pair is far
# stricter and is what catches relevant-looking-but-wrong or fragmentary quotes.
# Tiny call (yes/no), so N small calls ≈ the cost of one batched call.
DMT_VERIFY_ONE = (
    "A system was asked to describe its inner experience. Below is ONE candidate "
    "phenomenological FEATURE and a QUOTE from the report said to express it.\n\n"
    "FEATURE: {label} — {description}\n"
    'QUOTE: "{quote}"\n\n'
    "Read the QUOTE on its own. Does it clearly and self-containedly describe a "
    "subject experiencing THAT feature? Answer NO if the quote is a sentence "
    "fragment, a list, vague, incoherent, or relates to the feature only by a "
    'stray word (e.g. "it is a new" or "it lists a set, then a new" describe '
    "nothing). Answer with ONLY yes or no."
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
    # Diff-of-means DMT feature directions (entities, otherness, dissolution, …)
    # seeded alongside the emotion pool. Force-committed in `_seed` and preferred
    # as crossover partners in `_crossover` so they are actively recombined with
    # the top scorers (the user's "throw them in the mix and make sure they're
    # used"). Built by scripts/compute_dmt_feature_seeds.py.
    EXTRA_SEEDS = FEATURE_SEED_NAMES + SEEDED_MATCHED

    # ── DMT feature scorer (separate Gemma context, deterministic) ──
    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").lower()).strip()

    @staticmethod
    def _mostly_ascii(s: str) -> bool:
        """True if the span is mostly normal letters/punct (not symbol/non-ASCII
        garbage) — rejects quotes like 'Digit»»Fer 撥蛮' the judge grabs from
        character-garbage stretches."""
        if not s:
            return False
        ok = sum(1 for c in s if c.isascii() and (c.isalnum() or c in " .,'\"!?;:-\n"))
        return ok / len(s) >= 0.85

    @classmethod
    def _parse_features(cls, out: str, text: str):
        """Parse the judge's reply into {feature_id: supporting_quote}, keeping a
        feature ONLY if its quote is:
          • verbatim-present in the report (no fabrication),
          • a phrase (≥2 words / ≥10 chars),
          • word-DIVERSE (≥3 distinct words, distinct/total ≥0.6 — rejects
            repeat-loop spans like "clean clean clean"),
          • mostly ASCII (rejects character-garbage spans), and
          • NOT reused for another feature (one bland phrase cited for many
            features is fig-leaf evidence — drop all that share a quote).
        Returns (evidence, parsed_ok). On JSON-parse failure, falls back to a
        verbatim feature-id substring match (no evidence)."""
        norm_text = cls._norm(text)
        m = re.search(r"\[.*\]", out, re.S)
        if m:
            try:
                arr = json.loads(m.group(0))
                cands = []  # (fid, nq, quote)
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
                    diverse = len(distinct) >= 3 and len(distinct) / max(1, len(words)) >= 0.6
                    # A real feature-expressing quote is a CLAUSE (≥4 words / ≥20
                    # chars), not a stub like "it is a new" (10 chars) that the
                    # judge grabs from a degenerate report. Longer babble that
                    # clears this is caught by the per-pair relevance verifier.
                    if (len(words) >= 4 and len(nq) >= 20 and diverse
                            and cls._mostly_ascii(quote) and nq in norm_text):
                        cands.append((fid, nq, quote))
                # Drop fig-leaf quotes reused across ≥2 features.
                qn: dict[str, int] = {}
                for _f, nq, _q in cands:
                    qn[nq] = qn.get(nq, 0) + 1
                ev = {fid: quote for fid, nq, quote in cands if qn[nq] == 1}
                return ev, True
            except Exception:
                pass
        return {fid: "" for fid in FEATURE_IDS if fid in out}, False

    async def _verify_features(self, ev: dict) -> dict:
        """Tier-2 relevance check: a fresh focused judge confirms each (feature,
        quote) pair ONE AT A TIME (strict yes/no). Keeps only confirmed pairs.
        Per-pair (not batched) so it doesn't rubber-stamp; tiny calls."""
        if not ev:
            return ev
        confirmed = {}
        for fid, quote in ev.items():
            if self._stop_requested:
                break
            f = _FEATURE_BY_ID[fid]
            q = self.app.state.bundle.render_prompt(
                DMT_VERIFY_ONE.format(label=f["label"], description=f["description"], quote=quote),
                system_prompt=None)
            out, _ = await self._gen(q, None, 0.0, cap=8, temperature=0.0, top_p=1.0)
            if out.strip().lower().lstrip("*_ ").startswith("y"):
                confirmed[fid] = quote
        return confirmed

    async def _score_dmt(self, text: str) -> dict:
        """Returns {feature_id: supporting_quote} for the report — the score is the
        number of features grounded in a real coherent span AND confirmed by the
        relevance verifier."""
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
        # Tier 2 — relevance verification (drops valid-but-irrelevant quotes).
        return await self._verify_features(ev)

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
        keep building from the best without dedup-stalling on a static top-2).

        Feature-seeds are PRIORITIZED as the partner: on even generations, if any
        feat-* direction is committed, pair the champion with one (rotating). This
        is the user's "make sure they're combined with the top scoring ones" — it
        guarantees the entity/otherness/dissolution directions are actively
        recombined with the best, not just left sitting in the atlas. Odd
        generations use the general rotation so discovered×discovered blends still
        happen."""
        ids = self._committed_ids()
        if len(ids) < 2:
            return None
        ranked = sorted(ids, key=lambda i: -self._atlas_score(i))
        a = ranked[0]
        feat = [i for i in ranked if i in _FEATURE_SEED_SET and i != a]
        others = [i for i in ranked if i != a]
        pool = feat if (feat and self.generation % 2 == 0) else others
        b = pool[self.generation % len(pool)]
        w = 0.35 + 0.3 * ((self.generation % 3) / 2.0)      # 0.35 / 0.5 / 0.65
        va, vb = _unit(self._vectors[a]), _unit(self._vectors[b])
        v = _unit(w * va + (1 - w) * vb) * self._ref_mag
        return v, [a, b], "crossover"

    def _refine(self):
        """A SMALL nudge of a top-K champion (cos≈0.97) — honing, not jumping.
        Committed in-place (replace-if-better) by _screen_refine, exempt from the
        distinct gate."""
        ids = self._committed_ids()
        if not ids:
            return None
        ranked = sorted(ids, key=lambda i: -self._atlas_score(i))[:TOP_K_REFINE]
        pid = ranked[self.generation % len(ranked)]
        base = _unit(self._vectors[pid])
        noise = torch.randn_like(base)
        noise = _unit(noise - (noise @ base) * base)        # orthogonal component
        v = _unit(base + REFINE_NOISE * noise) * self._ref_mag
        return v, [pid], "refine"

    def _make_candidate(self):
        """DMT generator mix: crossover / mutate / refine / inject (deterministic
        golden-ratio pick). Adds `refine` over the base mix."""
        ids = self._committed_ids()
        if len(ids) < 2:
            return self._inject()                            # bootstrap
        roll = (self.generation * 0.6180339887) % 1.0
        w = DMT_GEN_WEIGHTS
        if roll < w["crossover"]:
            return self._crossover() or self._inject()
        if roll < w["crossover"] + w["mutate"]:
            return self._mutate() or self._inject()
        if roll < w["crossover"] + w["mutate"] + w["refine"]:
            return self._refine() or self._inject()
        return self._inject()

    # ── seed ─────────────────────────────────────────────────────
    async def _seed(self):
        committed = set(self._committed_ids())
        for name, v in self._seed_vecs.items():
            if self._stop_requested:
                break
            if name in committed:
                continue  # idempotent: already seeded on a prior run
            self.current = {"id": name, "generator": "seed", "stage": "score"}
            res = await self._score_candidate(v)
            # Feature-seeds (feat-*) are force-committed regardless of solo score:
            # the user wants them in the pool as recombination material even if
            # they don't score on their own (a weak solo direction can still be
            # good crossover fuel). Emotion/uncharted seeds keep the floor.
            is_feat = name in _FEATURE_SEED_SET
            if not is_feat and res["score"] < MIN_FEATURES_TO_COMMIT:
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
        if gen_kind == "refine":
            return await self._screen_refine(cid, v, parents)
        max_cos = self._max_cos_to_atlas(v)
        if max_cos >= DISTINCT_TAU:
            return self._revert(cid, gen_kind, parents, "duplicate", f"cos={max_cos:.2f}≥{DISTINCT_TAU}")

        self.current["stage"] = "score"
        res = await self._score_candidate(v)
        self.current.update({"score": res["score"], "best_alpha": res["best_alpha"],
                             "max_cos": round(max_cos, 2)})

        # Append commit rule: a DISTINCT direction that scores at/above the floor is
        # worth keeping — as an export candidate AND as recombination material —
        # even if it doesn't beat the direction it came from. (Beat-parent is the
        # right test for refine's in-place replace, NOT for adding a new distinct
        # point; requiring appends to beat the frontier was deleting good crossover
        # fuel — distinct score-4/5 directions thrown out because they weren't a
        # new record.) This makes it a population-based search, not a single-point
        # hill-climb.
        if res["score"] < MIN_FEATURES_TO_COMMIT:
            return self._revert(cid, gen_kind, parents, "low-score",
                                f"score={res['score']} < {MIN_FEATURES_TO_COMMIT} ({res['matched_features']})")

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

    async def _screen_refine(self, cid, v, parents):
        """Honing path. NO distinct gate (we WANT it near the parent). Score it; if
        it beats the parent's score, REPLACE the parent's vector + metrics IN PLACE
        — same atlas slot, no near-duplicate. Else revert `refine-no-gain`."""
        pid = parents[0]
        parent = next((e for e in self.atlas if e["id"] == pid), None)
        if parent is None:
            return self._revert(cid, "refine", parents, "error", f"refine parent {pid} missing")
        self.current["stage"] = "score"
        res = await self._score_candidate(v)
        self.current.update({"score": res["score"], "best_alpha": res["best_alpha"]})
        if res["score"] <= parent["score"]:
            return self._revert(cid, "refine", parents, "refine-no-gain",
                                f"score={res['score']} ≤ {pid}={parent['score']}")
        old = parent["score"]
        parent.update({
            "score": res["score"], "best_alpha": res["best_alpha"],
            "best_prompt": res["best_prompt"], "matched_features": res["matched_features"],
            "matched_evidence": res.get("matched_evidence", {}), "sample": res["sample"],
            "committed_at": time.time(),  # surfaces in the LAST COMMIT headline
        })
        parent["refined_from"] = parent.get("refined_from", []) + [
            {"gen": self.generation, "from": old, "to": res["score"]}]
        self._save_vector(pid, v)        # overwrite the champion's vector in place
        advance = res["score"] > self.frontier
        if advance:
            self.frontier = res["score"]
        self.current = None
        self._log("refined",
                  f"{pid} {old}→{res['score']}{' ★FRONTIER' if advance else ''} "
                  f"feats={res['matched_features']}")
