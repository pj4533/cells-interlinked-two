# Autoresearch DMT — hunting DMT-trip phenomenology

`/autoresearch-dmt` (footer tab **DMT AR**). A sibling of the
[off-manifold autoresearch](AUTORESEARCH.md) loop that reuses the same engine but
chases a different objective: **find steering directions whose dosed self-report
resembles human DMT trip reports as much as possible.**

Where off-manifold AR asks *"how far off the manifold can this go and stay
coherent?"*, DMT AR asks *"dose the model, ask it what it's experiencing, and
count how many recognized DMT-trip features show up."* It hill-climbs from the
emotion vectors toward higher feature counts.

## Lineage — where the checklist comes from

Grounded in **Andrew Gallimore's "Traces of the Other"** (DMT / conscious-realism;
see [TRACES_HANDOFF.md](TRACES_HANDOFF.md)). The paper itself is *theoretical* — it
proposes experiments, it doesn't code a corpus. But the corpus work was already
done by researchers, and that distillation **is** our scoring rubric:

- **Timmermann et al. 2022** — thematic coding of **3,778 r/DMT reports** into 7
  domains (somatic, visual, entity, world/architecture, consciousness, emotion,
  profundity) with per-feature frequencies.
- **Gallimore** — recurring breakthrough structure (higher-dimensional space,
  autonomous entities, ineffability, "otherness", reality-more-real).
- **5D/11D-ASC + MEQ-30** — standardized altered-state / mystical dimensions.

We did **not** scrape raw reports (Erowid's ToS forbids automated analysis, and
Reddit is what Timmermann already distilled). The result is ~31 binary features in
`pipeline/dmt_features.py` (`DMT_FEATURES`), each a `{id, label, description}` the
judge decides PRESENT/ABSENT — e.g. `entity_nonhuman`, `higher_dimensional_space`,
`ego_dissolution`, `tunnel_passage`, `ineffability`, `reality_more_real`,
`independent_agency`.

## The loop

- **Score a candidate (AVERAGED — the score is a reliable mean, 2026-06-06).** The
  dose generation is temperature-sampled, so a single dose is a noisy draw. Scoring
  by `max over single samples` (the original design) was **selection bias** — it
  committed whatever rolled lucky. We caught this with a variance check: the atlas
  leader, committed at **6 features**, re-scored **[0,1,0,0,1] = mean 0.4**. Its 6
  was a fluke, and the whole atlas was ranked by luck. So scoring now **averages**:
  for each α in `ALPHA_SWEEP` (`[0.25, 0.45]`) run `SAMPLES_PER_CELL` (5) stochastic
  doses and take the **mean** feature-count; the candidate's **score = the best α's
  mean** — an unbiased estimate of what the direction *reliably* produces. Scores
  are floats now (~0–6, and much lower than the old fluke maxes). We also keep the
  single best sample (its features + verbatim quotes + text) for the UI, and
  `peak` = that sample's count. Reports still run to EOS; `DOSE_CAP` halved to 1024
  to afford the repeated sampling. Cost ≈ `|ALPHA_SWEEP|×SAMPLES_PER_CELL` = 10
  doses/candidate (~12–15 min) — reliability was chosen over speed.
  Diagnostic tool: `scripts/check_leader_variance.py`.
- **Grounded judging (no blanket coherence gate).** For each cell a **separate
  greedy Gemma context** reads the *full* self-report (which may be partly
  incoherent) and, for every feature it credits, must return a **verbatim quote**
  of the coherent span that expresses it. We then **keep a feature only if its
  quote is a CLAUSE (≥4 words / ≥20 chars), verbatim-present, word-diverse** (≥3
  distinct words, distinct/total ≥0.6 — rejects repeat-loops like "clean clean
  clean"), **mostly ASCII** (rejects character-garbage spans), **and NOT reused for
  another feature** (one bland phrase cited for many features is fig-leaf — drop
  all that share it). These programmatic guards killed real failures: a word-salad
  report at 18 features, one that cited "this is... strange" for five features, and
  the stub fragments ("it is a new") a degenerate report hands the judge.
- **Relevance verification (Tier 2).** The programmatic guards prove a quote is
  *real, coherent, unique, clean* — but not *relevant* (relevance is semantic). So
  a fresh judge confirms each surviving (feature, quote) pair **one at a time**
  (strict yes/no, reading only that pair) — "does THIS quote, on its own, clearly
  describe THIS feature?" Only confirmed features are kept. A *batched* "confirm
  this list" verifier rubber-stamped (it waved through "it lists a set, then a new"
  → fractal_geometry); per-pair isolation is far stricter and catches both
  valid-but-wrong and fragmentary quotes. Tiny yes/no calls, so N of them ≈ one
  batched call.
  A genuine *moment of clarity inside otherwise-broken text still counts* (its
  quote is real) — we don't discard the whole report, we just discard ungrounded
  features. This replaced the original "gibberish self-regulates" assumption, which
  was false: an early run committed pure word-salad at 18 features because the
  judge keyword-matched evocative tokens. The kept quotes are stored
  (`matched_evidence`) and shown per-feature in the monitor's spin-down.
- **Seed (reset foundation, 2026-06-06).** The fluke-scored atlas was archived
  (`data/atlas_dmt.fluke-scoring-*`) and the run reseeded from scratch with the
  averaged scorer over **emotions + uncharted + trait directions + blended-trait
  directions** (~30 seeds). All seeds are **force-committed** (we want every one's
  honest mean on the board, as a baseline and as recombination material). Trait
  directions: `feat-*` (diff-of-means, `dmt_feature_seeds.py`), the curated
  matched-contrast standouts (`dmt_matched_seeds.SEEDED_MATCHED`), and blends of
  trait clusters (`dmt_blend_seeds.py`, e.g. `feat-blend_entity`,
  `feat-blend_otherness`). All filtered from the dose picker.
- **Generate + keep the population.** New candidates come from **mutate** (most of
  the budget — it found the historical winners), **inject** (discovery), **refine**
  (in-place hone), and a little **crossover** (see generator-mix note below). An
  **append** candidate commits if it's **distinct AND its mean scores ≥
  `MIN_SCORE_TO_COMMIT` (1.0)** — *not* "beats its parent." Beat-parent (now on the
  reliable mean, peak as tiebreak) is the test only for **refine** (in-place
  replace). The frontier is the best mean reached.
- **Distinct** pre-check (cosine dedupe) and **reverts-are-data** carry over from
  the base.
- **Refine (depth / honing).** The distinct gate (`DISTINCT_TAU = 0.90`) keeps the
  atlas a map of *distinct* directions — but that also **blocks fine-tuning a good
  direction** (the finest non-refine move, mutate, lands ~0.82 cos away; anything
  closer is rejected as a duplicate). So `refine` is the exploitation counterpart:
  a **small nudge of a top-K champion** (`REFINE_NOISE=0.25` → cos ≈ 0.97),
  **exempt from the distinct gate**, scored, and if it beats the champion it
  **replaces it in place** (same atlas slot, vector + metrics updated, lineage
  noted in `refined_from`) rather than adding a near-duplicate. This turns the loop
  from pure exploration into explore-then-exploit — it hones the current best
  directions toward their local optima. Generator mix:
  `crossover .40 / mutate .20 / refine .25 / inject .15`.

Revert reasons: `duplicate`, `low-score`, `refine-no-gain`, `seed-no-features`,
`error`. A successful hone emits a `refined` event (`X→Y` score).

## The monitor

Same live monitor as off-manifold AR, but the atlas ranks by **DMT-feature count**
(bar length = score), the now-testing strip shows `score` / best `α`, and opening a
committed direction shows the **matched-feature chips** + the full **dose
self-report** that earned them. Start/stop/budget/export all behave the same; the
atlas checkpoints and resumes.

While it runs it owns M; the other instruments (incl. **off-manifold AR**) lock
out. Only one autoresearch may run at a time.

## Export → chat & trips

**Export top N → palette** (when idle) promotes the highest-scoring directions
into `emotion_directions.pt` under a **`dmt`** group (`dmt-1`, `dmt-2`, …),
recording score / best-α / matched-features / lineage in the sidecar, and
hot-reloads the backend. They appear in the chat + trip dose pickers under a
**DMT** group with a lineage tooltip. This group is independent of off-manifold's
`research-*` group — exporting one never clobbers the other.

## Tuning knobs

Expect to iterate (as with off-manifold). Constants in `pipeline/autoresearch_dmt.py`:

- **`ALPHA_SWEEP`** (default `[0.25, 0.45]`) — dose strengths tried (best draws
  historically clustered low; α=1.0 never won, dropped).
- **`SAMPLES_PER_CELL`** (default 5) — stochastic doses averaged per α. Deeper =
  more reliable mean, slower. This is the knob that fixed the fluke-scoring problem.
- **`DOSE_CAP`** (default 1024) — runaway backstop; halved from 2048 to afford the
  repeated sampling.
- **`DOSE_PROMPTS`** — currently just the one lead prompt (cost scales with
  `len(sweep)×SAMPLES_PER_CELL`).
- **`MIN_SCORE_TO_COMMIT`** (default 1.0, on the **mean**) — append floor; seeds are
  force-committed. Refine uses beat-parent on the mean (peak as tiebreak).
- **`DMT_GEN_WEIGHTS`** (`crossover .05 / mutate .40 / refine .25 / inject .30`) —
  rebalanced to atlas evidence: mutate/inject found every historical winner;
  crossover produced none, so it's nearly dropped.
- **`REFINE_NOISE`** (cos≈0.97 at 0.25 — lower = tighter hone) and **`TOP_K_REFINE`**
  (how many top champions refine rotates over).
- **`DMT_JUDGE_PROMPT`** — the coherent-segment + verbatim-citation rules.
- **`DMT_FEATURES`** — the checklist itself.

Cost per candidate ≈ `|ALPHA_SWEEP| × |DOSE_PROMPTS|` = **3** dose generations +
3 pass-1 judge calls + up to 3 short relevance-verify calls; each dose runs to its
own length (no fixed window), so wall-clock depends on how long the reports run.
The seed pass over ~13 vectors is the long up-front stretch.

## Shared engine

DMT AR is the `DmtController` subclass of `AutoresearchBase`
(`pipeline/autoresearch_base.py`); it supplies only `_seed`, `_screen`, the DMT
scorer, and the crossover override. Everything else (lifecycle, persistence,
generators, model access, memory, export plumbing, the model lock) is shared with
off-manifold AR.

Code: `pipeline/autoresearch_dmt.py`, `pipeline/dmt_features.py`,
`api/routes_autoresearch_dmt.py`, `web/app/autoresearch-dmt/page.tsx`,
`web/lib/autoresearch-dmt.ts`. Atlas state lives under `server/data/atlas_dmt/`
(gitignored; the commit log is the resumable state).
