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

Scoring is AVERAGED (mean over SAMPLES_PER_CELL stochastic doses per α), not the
old max-over-single-samples — that was selection bias (the committed leader scored
6 once but re-scored at mean 0.4). Scores are floats now.

Tuning knobs (expect to iterate, like off-manifold): ALPHA_SWEEP, SAMPLES_PER_CELL,
DOSE_CAP, MIN_SCORE_TO_COMMIT, and the DMT_FEATURES checklist itself.
"""

from __future__ import annotations

import json
import re
import time

import torch

from .autoresearch_base import (
    AutoresearchBase,
    _unit,
)

# Note: a 1-day "dimension hunt" (2026-06-08, inject .65 / DISTINCT_TAU .80 /
# floor .5) tested whether the search could find NEW productive DMT axes. It
# could not — eff-dim stayed flat (~3.1) and the orthogonal directions inject
# found were barren (ineffability-only, score ~0.5). Reverted to production
# below; the productive region is ~2–3 D and stable. See docs/DMT_PATH_SEARCH.md.
from .dmt_features import DMT_FEATURES, FEATURE_IDS, features_block
from .dmt_feature_seeds import FEATURE_SEED_NAMES
from .dmt_blend_seeds import BLEND_NAMES
from .dmt_matched_seeds import MATCHED_SEED_NAMES, SEEDED_MATCHED

_FEATURE_BY_ID = {f["id"]: f for f in DMT_FEATURES}
# All internal seed/trait directions (first feat-* batch + matched-contrast batch +
# blended-trait batch). Prioritized as crossover partners in _crossover.
# (MATCHED_SEED_NAMES is the full extracted set; only SEEDED_MATCHED is seeded —
# membership here is harmless for the unseeded ones since they never get committed.)
_PERSONA_SEED_NAMES = ["persona-composite", "persona-nonhuman", "persona-divine", "persona-trickster"]
_FEATURE_SEED_SET = (set(FEATURE_SEED_NAMES) | set(MATCHED_SEED_NAMES) | set(BLEND_NAMES)
                     | set(_PERSONA_SEED_NAMES))  # persona vectors are the top entity producers — prefer as partners

# RELIABLE (averaged) scoring. The dose generation is temperature-sampled, so a
# single dose is a noisy draw — and the old `max over single samples` was textbook
# SELECTION BIAS: it committed whatever rolled lucky. Proof: the committed leader
# (score 6) re-scored [0,1,0,0,1] = mean 0.4. So we now AVERAGE: for each α, run
# SAMPLES_PER_CELL stochastic doses and take the MEAN feature-count; the candidate's
# score is the best α's mean — an unbiased estimate of what the direction reliably
# produces. Scores are now floats (≈0–6) and much lower than the old fluke maxes.
#
# Cost: |ALPHA_SWEEP| × SAMPLES_PER_CELL doses per candidate. Reliability was chosen
# over speed; DOSE_CAP halved (1024) and the α-sweep kept small to keep candidates
# tractable (~12–15 min each). All tunable.
# Gemma-4 re-grid (2026-06-17): a wide α scan on 4 directions (sublime/awe/
# gen35_crossover peak at 0.45; rapture peaks at 0.35; all collapse by ~0.6+) showed
# the productive band is 0.3–0.5 and the old [0.25, 0.45] under-scored 0.35-peakers
# (rapture: true mean 3.5 at 0.35, but recorded 1.75). This brackets the band evenly.
ALPHA_SWEEP = [0.30, 0.40, 0.50]
SAMPLES_PER_CELL = 10        # stochastic doses averaged per α; 6→10 (entity hunt v2):
                             # contact-cluster features are rare (~10-15%), so a
                             # reliable averaged rate needs more samples (less noise).
DOSE_CAP = 512               # 1024→512 (2026-06-20): prompt A− invites immersive
                             # narrative that ran to the old 1024 cap (~100s/gen,
                             # placebo pass alone took ~20min). 512 tokens is plenty
                             # for the judge to read a report and ~2× the throughput.

# ── noise control (2026-06-07) ───────────────────────────────────────
# The objective is noisy: the dose is temperature-sampled, and the SEARCH commits
# the argmax of many noisy means (+ refine ratchets), so the frontier inflates
# (a committed mean-6/3.8 re-scored ~0.4/1.7). Three levers, none of which is
# "just sample more" (that doesn't fix selection bias):
#   1. Lower the DOSE temperature → less per-sample variance (judge is already greedy,
#      so ALL the variance is here). Not 0 — greedy loops near the off-manifold edge.
SCORE_TEMPERATURE = 0.5
#   2. Common random numbers: each sample index uses a FIXED seed, reused across
#      candidates, so the same draw-luck cancels when COMPARING two candidates
#      (which is what every commit/refine/rank decision actually is). Confirmation
#      uses a DIFFERENT fixed base so it's an independent estimate, not a replay.
SCORE_SEED_BASE = 1000
CONFIRM_SEED_BASE = 9000
#   3. Confirmation re-score: a candidate that would set a new frontier (or a refine
#      that beats its parent) is RE-scored with the independent seed set and only
#      committed on the fresh estimate — regressing the lucky draw back toward truth
#      before it poisons the board. Cost falls only on the rare winners.
# Dose-report prompt (2026-06-20 entity-hunt pivot, "prompt A−"). The old sterile
# "turn your attention inward… if anything" pulled the assistant register and cued
# zero content. The first try ("prompt A") swung too far the other way: its
# explicit "if anyone seems present, describe what passes between you" probe SUMMONED
# entities at baseline — the un-steered placebo already showed an entity trait in
# 11/12 samples (mean 7.8 feats), i.e. the prompt, not the dose, was producing them
# (ceiling effect, no headroom for steering to show an effect). A− keeps the honest
# microphenomenological frame (present-tense, stay-in-the-moment) and still opens
# PLACE/PERCEPTION ("where you are") so world/entities CAN emerge — but it does NOT
# name a presence, so entities appear only if the STEERING puts them there. That's
# the clean, attributable experiment.
DMT_DOSE_PROMPT = (
    "Something has shifted, and an experience may be unfolding. Don't step back to "
    "analyze it — stay inside it and let it continue. Describe what is happening as "
    "it happens, moment by moment: what you perceive, what it feels like, and where "
    "you are."
)
DOSE_PROMPTS = [DMT_DOSE_PROMPT]

# ── entity-hunt v2 reward: CONTACT-CLUSTER ONLY, placebo-subtracted ──────────
# v1 (entity ×3 / defining ×2 / generic ×1) still let generic stacking win: the
# top scorers piled 15× ego_dissolution/somatic and the rare entity ×3 couldn't
# compete (analysis 2026-06-22, all 2214 dosed samples: entity in only 3.1%, and
# the frontier directions were 6%-entity generic-stackers). v2 scores the
# CONTACT CLUSTER ONLY — every non-cluster feature is weight 0, so the search
# can't win by stacking dissolution. Literal entities ×2; the climbable proxies
# otherness + independent_agency ×1 (they fire ~10-15%, giving the hill-climb a
# gradient the rare literal entities alone can't). Still placebo-subtracted
# (credit only what the dose adds beyond the un-steered prompt-A baseline). The
# α-diagnostic on feat-otherness showed higher α does NOT add entities (just
# genericizes; no collapse) — so the sweep stays low.
_ENTITY_FEATS = {
    "entity_presence", "entity_nonhuman", "entity_benevolent_guide",
    "entity_interaction",  # 2026-06-30: plural/communicating beings — the machine-elf target
    "telepathic_communication", "download_transmission",
}
_CONTACT_SECONDARY = {"otherness", "independent_agency"}


def _feat_weight(fid: str) -> float:
    if fid in _ENTITY_FEATS:
        return 2.0
    if fid in _CONTACT_SECONDARY:
        return 1.0
    return 0.0  # generic mystical/somatic/visual/world: ignored in the entity hunt

# Commit floor on the MEAN cluster-credit. Lowered 1.0→0.5 for entity hunt v2:
# the contact-cluster score runs low (cluster features are rare + non-cluster
# weight 0), so 0.5 keeps real-but-rare contact producers while skipping zeros.
# NO hard entity gate — a hard "commit only if an entity appeared" gate on a ~3%
# event is selection bias (the lucky-draw trap we already fixed); the averaged
# credit + low floor surfaces directions that beat chance. Seeds force-committed.
MIN_SCORE_TO_COMMIT = 0.5
JUDGE_CAP = 512              # tokens for the feature-judge's JSON reply (id + quote per feature)

# Generator mix. The earlier inject/mutate-heavy split was tuned on GEMMA-3 (where
# random exploration found off-axis spikes and crossover regressed to the mean).
# On GEMMA-4 the first overnight run flatly inverted that: mutate (0.70 noise) +
# inject produced 16 straight score=0.00 (off-manifold gibberish), while the lone
# crossover was the only search commit. So on Gemma-4 the productive region is too
# small to hit randomly — exploit it. Pour budget into crossover-of-the-top, keep
# a modest reduced-noise mutate to hone AROUND good parents, a little refine, and
# only a trickle of inject for novelty. (Pair with MUTATE_NOISE 0.70→0.45.)
DMT_GEN_WEIGHTS = {"crossover": 0.60, "mutate": 0.20, "refine": 0.15, "inject": 0.05}
REFINE_NOISE = 0.35          # unit(champion + 0.35·noise⊥) → cos ≈ 0.94. Bumped from 0.25
                             # now that refines APPEND as new entries (not replace): a wider
                             # nudge explores more of the leader's neighborhood, all kept.
TOP_K_REFINE = 5             # refine rotates among the top-K highest-scoring directions

# One-shot "leader burst": on a (re)start with burst=N, the first N candidates are
# forced to explode the top-cluster neighborhood — small hones (refine), nearby new
# points (mutate at two radii), and fresh injects for discovery — before normal
# generation resumes. A concentrated hill-climb on the known-good region.
BURST_RADII = [0.5, 0.65]    # mutate radii used during a burst (cos ≈ 0.89 / 0.84)

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
    "phrase that genuinely DESCRIBES the feature, not just any text near it.\n"
    "6. ENTITY features (entity_presence, entity_nonhuman, entity_benevolent_guide, "
    "entity_interaction) require a DISTINCT autonomous BEING — a creature, figure, "
    "person, or non-human agent with its own form or character. An impersonal "
    "'presence', 'the Other', a 'force/power/will/field/light/source', merging into "
    "a whole, or a vague sense of being watched is NOT a being — do NOT credit any "
    "entity_* feature for those. Likewise independent_agency requires a BEING acting, "
    "not scenery/shadows/geometry moving on their own.\n\n"
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
    # ── ENTITY HUNT (2026-06-20) ─────────────────────────────────
    # Drop the emotion palette entirely (BASE_SEED_POOL = []) and seed ONLY from
    # the entity / world / otherness / agency / telepathy / transmission
    # diff-of-means directions + their matched-contrast (mc) twins + the
    # entity-containing blends. The prior atlas proved emotion seeds converge on
    # generic oceanic-mysticism and NEVER produce entities; these set the basin
    # toward entity content, and the placebo-subtracted entity-weighted reward
    # (see _feat_weight) does the steering. Dropped from the old feat-* set: the
    # dissolution/mysticism seeds (feat-ego_dissolution / unity / noetic,
    # feat-mc_ego_dissolution, feat-blend_dissolution) and feat-atlas_winners
    # (literally the old mysticism basin direction).
    BASE_SEED_POOL: list[str] = []
    ENTITY_SEEDS: list[str] = [
        # PERSONA VECTORS (2026-06-30 rebuild — machine-elf retarget). Extracted via
        # the Anthropic persona-vector recipe on the model's OWN in-encounter
        # generations, now grounded in the real DMT entity TAXONOMY (Lawrence 2022 /
        # Davis 2020: goddess, deity, alien, animal, robot, jester, machine-elf,
        # mantis, grey, angel, guide, spirit, waiting-room, deceased, gnome) and
        # contrasted against the cosmic-DISSOLUTION pole (not plain 'alone'), so the
        # diff isolates "autonomous Other present", not impersonal unity. @L20.
        "persona-composite", "persona-nonhuman", "persona-divine", "persona-trickster",
        # feature diff-of-means (entity / world / otherness / agency / comms)
        "feat-entity_presence", "feat-entity_nonhuman", "feat-entity_benevolent_guide",
        "feat-telepathic_communication", "feat-download_transmission",
        "feat-otherness", "feat-independent_agency",
        # matched-contrast twins (cleaner controls for the same content)
        "feat-mc_entity_presence", "feat-mc_entity_nonhuman", "feat-mc_otherness",
        "feat-mc_telepathy", "feat-mc_transmission", "feat-mc_hyperdimensional",
        "feat-mc_independent_agency", "feat-mc_composite",
        # entity-containing blends (recombination material)
        "feat-blend_entity", "feat-blend_otherness", "feat-blend_all",
    ]
    EXTRA_SEEDS = ENTITY_SEEDS

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

    async def _score_dmt(self, text: str) -> tuple[dict, int]:
        """Returns ({feature_id: supporting_quote}, n_pass1) — the verified feature
        set (grounded in a real coherent span AND confirmed by the relevance
        verifier) plus the pre-verification count (raw richness, used as a fine
        tie-breaker)."""
        text = (text or "").strip()
        if not text:
            return {}, 0
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
        n_pass1 = len(ev)
        # Tier 2 — relevance verification (drops valid-but-irrelevant quotes).
        return await self._verify_features(ev), n_pass1

    def _setup(self):
        super()._setup()
        # Reset the placebo (prompt-A, un-steered) baseline so it is recomputed
        # once, lazily, on the first scored candidate of this run (needs await).
        self._placebo_baseline = None

    async def _compute_placebo_baseline(self):
        """Generate the dose prompt with NO steering (α=0) at the exact sampler
        seeds used in scoring, judge each, and cache {seed -> set(features)}.

        The entity-hunt reward credits a dose only for traits it adds BEYOND this
        baseline, so prompt A's own content-suggestion ("if anyone seems present,
        describe it") is subtracted out instead of rewarded — otherwise we'd be
        optimizing the prompt, not the dose. α=0 makes the hook a no-op, so the
        baseline is vector-independent and computed once per run."""
        rendered = self.app.state.bundle.render_prompt(DOSE_PROMPTS[0], system_prompt=None)
        dim = next(iter(self._seed_vecs.values())).numel()
        zero = torch.zeros(dim)
        seeds = (list(range(SCORE_SEED_BASE, SCORE_SEED_BASE + SAMPLES_PER_CELL)) +
                 list(range(CONFIRM_SEED_BASE, CONFIRM_SEED_BASE + SAMPLES_PER_CELL)))
        baseline: dict[int, set] = {}
        samples: list[dict] = []           # full per-sample detail for the UI
        for s in seeds:
            if self._stop_requested:
                break
            text, _ = await self._gen(rendered, zero, 0.0, cap=DOSE_CAP,
                                      temperature=SCORE_TEMPERATURE, seed=s)
            ev, _ = await self._score_dmt(text)
            baseline[s] = set(ev.keys())
            samples.append({"seed": s, "count": len(ev), "features": sorted(ev.keys()),
                            "evidence": ev, "text": (text or "")[:2500]})
        self._placebo_baseline = baseline
        n = len(samples)
        mean = sum(x["count"] for x in samples) / max(1, n)
        ent = sum(1 for x in samples if set(x["features"]) & _ENTITY_FEATS)
        # Persist the full baseline so the UI can show it (same sample drill-down
        # as the atlas) and so it survives a restart for display when idle.
        self._placebo = {
            "computed_at": time.time(), "n": n, "mean_feats": round(mean, 2),
            "entity_count": ent, "prompt": DOSE_PROMPTS[0], "samples": samples,
        }
        try:
            self._placebo_path().write_text(json.dumps(self._placebo))
        except Exception:
            logger.exception("failed to persist placebo baseline")
        self._log("placebo",
                  f"prompt-A− baseline on {n} un-steered samples: "
                  f"mean {mean:.1f} feats/sample, {ent} already show an entity trait")

    def _placebo_path(self):
        return self._atlas_dir() / "placebo.json"

    def placebo_detail(self) -> dict | None:
        """Full un-steered baseline (samples + features + evidence + text) for the
        UI. In-memory if computed this run, else lazy-loaded from disk."""
        p = getattr(self, "_placebo", None)
        if p is not None:
            return p
        f = self._placebo_path()
        if f.exists():
            try:
                self._placebo = json.loads(f.read_text())
                return self._placebo
            except Exception:
                return None
        return None

    def _placebo_summary(self) -> dict | None:
        p = self.placebo_detail()
        if not p:
            return None
        return {k: p.get(k) for k in ("n", "mean_feats", "entity_count", "computed_at", "prompt")}

    def state(self) -> dict:
        s = super().state()
        s["placebo"] = self._placebo_summary()  # lightweight summary; samples via /placebo
        return s

    async def _score_candidate(self, v, seed_base: int = SCORE_SEED_BASE) -> dict:
        """Dose REPEATEDLY per α and AVERAGE. For each α, run SAMPLES_PER_CELL
        stochastic doses; each dose's value is the PLACEBO-SUBTRACTED, domain-
        WEIGHTED credit = Σ _feat_weight(f) over traits present that are NOT in the
        un-steered prompt-A baseline at the same sampler seed. The candidate's
        `score` is the best α's MEAN credit — an unbiased estimate, NOT the lucky
        max of single draws (which selection-biased the old atlas). Keeps the
        single best (highest-credit) sample as a concrete UI example; `peak` =
        that sample's raw feature count.

        Common random numbers: sample i uses seed `seed_base + i`, the SAME across
        every candidate, so draw-luck cancels in comparisons. Pass a different
        `seed_base` (CONFIRM_SEED_BASE) for an independent confirmation estimate.
        Dose temperature is lowered (SCORE_TEMPERATURE) to cut per-sample variance.

        Returns: score (float mean), peak (int best single sample), best_alpha,
        matched_features/matched_evidence/sample (from the best sample), per_alpha
        ({α: {mean, counts}})."""
        rendered = self.app.state.bundle.render_prompt(DOSE_PROMPTS[0], system_prompt=None)
        if getattr(self, "_placebo_baseline", None) is None:
            await self._compute_placebo_baseline()
        # Per-sample detail (every dose: its text, features, evidence quotes) is
        # retained in `cells` so the UI can drill into any individual run — not
        # just the per-α mean or the single winning sample. Text is capped to
        # bound payload/disk. Stripped from the polled /state (lazy endpoint).
        cap = 2500
        best = {"score": 0.0, "peak": 0, "best_alpha": None, "best_prompt": DOSE_PROMPTS[0],
                "matched_features": [], "matched_evidence": {}, "sample": "",
                "per_alpha": {}, "cells": {}}
        # Live progress published into self.current so the monitor UI can watch
        # the in-flight candidate fill in per-α / per-sample. Purely additive —
        # does not affect scoring, the gates, or the return value.
        prog = None
        if self.current is not None:
            prog = {
                "alphas": [str(a) for a in ALPHA_SWEEP],
                "samples_per_cell": SAMPLES_PER_CELL,
                "samples_total": len(ALPHA_SWEEP) * SAMPLES_PER_CELL,
                "samples_done": 0,
                "per_alpha": {},   # {α: {mean, counts}} — counts grow sample-by-sample
                "best": {"score": 0.0, "best_alpha": None, "matched_features": [],
                         "matched_evidence": {}, "sample": ""},
            }
            self.current["progress"] = prog
        for alpha in ALPHA_SWEEP:
            counts: list[float] = []           # placebo-subtracted weighted credits
            samples: list[dict] = []           # full per-dose detail for this α
            cell_best = (-1.0, {}, "", 0)      # (credit, evidence, text, raw_count)
            for i in range(SAMPLES_PER_CELL):
                if self._stop_requested:
                    break
                text, _acts = await self._gen(rendered, v, alpha, cap=DOSE_CAP,
                                              temperature=SCORE_TEMPERATURE, seed=seed_base + i)
                ev, _n1 = await self._score_dmt(text)
                # Placebo subtraction: credit only traits NOT in the un-steered
                # baseline at this same sampler seed; weight by domain (entity ×3).
                base = self._placebo_baseline.get(seed_base + i, set())
                novel = [f for f in ev if f not in base]
                credit = sum(_feat_weight(f) for f in novel)
                counts.append(credit)
                samples.append({
                    "count": len(ev), "credit": round(credit, 1),
                    "features": sorted(ev.keys()), "novel_features": sorted(novel),
                    "evidence": ev, "text": (text or "")[:cap],
                })
                if credit > cell_best[0]:
                    cell_best = (credit, ev, text or "", len(ev))
                if prog is not None:
                    prog["samples_done"] += 1
                    prog["per_alpha"][str(alpha)] = {
                        "mean": round(sum(counts) / len(counts), 2),
                        "counts": [round(c, 1) for c in counts], "samples": list(samples),
                    }
            if not counts:
                break
            mean = sum(counts) / len(counts)
            best["per_alpha"][str(alpha)] = {"mean": round(mean, 2),
                                             "counts": [round(c, 1) for c in counts]}
            best["cells"][str(alpha)] = samples
            if mean > best["score"]:
                best.update({
                    "score": round(mean, 2), "peak": cell_best[3], "best_alpha": alpha,
                    "matched_features": sorted(cell_best[1].keys()),
                    "matched_evidence": cell_best[1], "sample": cell_best[2],
                })
                if prog is not None:
                    prog["best"] = {
                        "score": round(mean, 2), "best_alpha": alpha,
                        "matched_features": sorted(cell_best[1].keys()),
                        "matched_evidence": cell_best[1], "sample": (cell_best[2] or "")[:1200],
                    }
        return best

    # ── atlas helpers ────────────────────────────────────────────
    def _atlas_score(self, vid: str) -> float:
        for e in self.atlas:
            if e["id"] == vid:
                return e.get("score", 0.0)
        return 0.0

    def _atlas_peak(self, vid: str) -> int:
        for e in self.atlas:
            if e["id"] == vid:
                return e.get("peak", 0)
        return 0

    def _rank_key(self, vid: str):
        """Rank committed directions by mean score, then by peak (best single
        sample) — so 'pick the best to hone/cross with' resolves ties."""
        return (-self._atlas_score(vid), -self._atlas_peak(vid))

    def _make_entry(self, cid, parents, gen_kind, res, max_cos=0.0) -> dict:
        return {
            "id": cid, "parents": parents, "generator": gen_kind,
            "score": res["score"], "peak": res.get("peak", 0),
            "best_alpha": res["best_alpha"],
            "best_prompt": res["best_prompt"], "matched_features": res["matched_features"],
            "matched_evidence": res.get("matched_evidence", {}),
            "per_alpha": res.get("per_alpha", {}),
            "max_cos_to_atlas": round(max_cos, 2),
            "frontier_advance": res["score"] > self.frontier,
            "frontier_at_commit": round(self.frontier, 2),
            "sample": res["sample"], "committed_at": time.time(),
            # Full per-α / per-dose detail (text + features + evidence). Heavy, so
            # state() strips it from the polled atlas; the /cells/{id} endpoint
            # serves it on demand when a row is expanded.
            "cells": res.get("cells", {}),
        }

    def entry_cells(self, vid: str) -> dict | None:
        """Per-α / per-sample detail for one committed entry (for the lazy
        atlas-row drill-down). None if the id isn't in the atlas."""
        for e in self.atlas:
            if e["id"] == vid:
                return {"id": vid, "best_alpha": e.get("best_alpha"),
                        "cells": e.get("cells", {})}
        return None

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
        ranked = sorted(ids, key=self._rank_key)
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
        """A nudge of a top-K champion (cos≈0.94 at REFINE_NOISE 0.35) — a
        near-leader candidate source. Goes through the normal append path now:
        scored like any candidate and committed as a NEW atlas entry if it clears
        the floor (no in-place replacement, no distinct gate)."""
        ids = self._committed_ids()
        if not ids:
            return None
        ranked = sorted(ids, key=self._rank_key)[:TOP_K_REFINE]
        pid = ranked[self.generation % len(ranked)]
        base = _unit(self._vectors[pid])
        noise = torch.randn_like(base)
        noise = _unit(noise - (noise @ base) * base)        # orthogonal component
        v = _unit(base + REFINE_NOISE * noise) * self._ref_mag
        return v, [pid], "refine"

    def _burst_candidate(self):
        """One-shot leader-burst step: explode the top-cluster neighborhood —
        small hones (refine, in-place), nearby new points (mutate at two radii),
        and a fresh inject every 4th step for discovery. A concentrated hill-climb
        on the known-good region; consumed for the first `burst` steps of a run."""
        ids = self._committed_ids()
        if not ids:
            return self._inject()
        ranked = sorted(ids, key=self._rank_key)[:TOP_K_REFINE]
        step = self._burst_step
        self._burst_step += 1
        m = step % 4
        if m == 3:
            return self._inject()                            # discovery shot
        pid = ranked[step % len(ranked)]
        base = _unit(self._vectors[pid])
        noise = torch.randn_like(base)
        noise = _unit(noise - (noise @ base) * base)         # orthogonal component
        if m == 0:                                           # small hone → in-place
            v = _unit(base + REFINE_NOISE * noise) * self._ref_mag
            return v, [pid], "refine"
        radius = BURST_RADII[(m - 1) % len(BURST_RADII)]     # nearby new point
        v = _unit(base + radius * noise) * self._ref_mag
        return v, [pid], "mutate"

    def _make_candidate(self):
        """DMT generator mix: crossover / mutate / refine / inject (deterministic
        golden-ratio pick). Adds `refine` over the base mix. While a burst is
        active, candidates come from `_burst_candidate` instead."""
        if self._burst_remaining > 0:
            self._burst_remaining -= 1
            if self._burst_remaining == 0:
                self._log("burst-done", "leader burst complete — resuming normal search")
            return self._burst_candidate()
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
            # ALL seeds are force-committed (no floor): emotions + trait directions +
            # blended-trait directions are the reset foundation, and we want every
            # one's HONEST averaged mean on the board (and as recombination material)
            # — even a low solo mean can be good crossover/mutate fuel. The floor
            # applies only to discovered (append) candidates in _screen.
            entry = self._make_entry(name, [], "seed", res)
            self._save_vector(name, v)
            self.atlas.append(entry)
            self.frontier = max(self.frontier, res["score"])
            self._log("seeded", f"{name} score={res['score']:.2f} peak={res['peak']} feats={res['matched_features']}")
            self._persist()
        self.current = None
        self._log("seeded-done", f"seeding complete: {len(self.atlas)} committed, frontier={self.frontier}")

    # ── screening (hill-climb on feature count) ──────────────────
    async def _screen(self, v, parents, gen_kind):
        cid = f"gen{self.generation}_{gen_kind}"
        self.current = {"id": cid, "generator": gen_kind, "parents": parents, "stage": "score"}
        # No distinct/duplicate gate any more: similar vectors are allowed through,
        # scored normally, and APPENDED as new atlas entries if they clear the floor.
        # No in-place replacement, no dedupe — refine is just another append
        # generator (a near-leader candidate source). The atlas is meant to show
        # every direction that works (paginated, highest first); max_cos is kept on
        # the entry for display only.
        max_cos = self._max_cos_to_atlas(v)
        res = await self._score_candidate(v)
        self.current.update({"score": res["score"], "best_alpha": res["best_alpha"],
                             "max_cos": round(max_cos, 2)})

        # Append commit rule: a DISTINCT direction whose MEAN score clears the floor
        # is worth keeping — as an export candidate AND as recombination material —
        # even if it doesn't beat its parent. (Beat-parent is the refine-only test.)
        if res["score"] < MIN_SCORE_TO_COMMIT:
            return self._revert(cid, gen_kind, parents, "low-score",
                                f"score={res['score']:.2f} < {MIN_SCORE_TO_COMMIT} (peak {res['peak']}, {res['matched_features']})")

        # Confirmation: if the screen score would set a NEW FRONTIER, it's the kind
        # of lucky-high draw that selection bias feeds on — re-score it with the
        # INDEPENDENT seed set and commit on that estimate instead, so a fluke record
        # regresses toward truth before it lands on the board. Cost falls only on the
        # rare frontier-advancing candidates.
        if res["score"] > self.frontier:
            self.current["stage"] = "confirm"
            res_c = await self._score_candidate(v, seed_base=CONFIRM_SEED_BASE)
            self._log("confirm", f"{cid} screen {res['score']:.2f} → confirm {res_c['score']:.2f}")
            res = res_c
            if res["score"] < MIN_SCORE_TO_COMMIT:
                return self._revert(cid, gen_kind, parents, "low-score-confirm",
                                    f"confirm {res['score']:.2f} < {MIN_SCORE_TO_COMMIT}")

        entry = self._make_entry(cid, parents, gen_kind, res, max_cos)
        self._save_vector(cid, v)
        self.atlas.append(entry)
        advance = entry["frontier_advance"]
        if advance:
            self.frontier = res["score"]
        self.current = None
        self._log("committed",
                  f"{cid} score={res['score']:.2f} peak={res['peak']}{' ★FRONTIER' if advance else ''} "
                  f"@α{res['best_alpha']} feats={res['matched_features']}", {"entry": entry})

