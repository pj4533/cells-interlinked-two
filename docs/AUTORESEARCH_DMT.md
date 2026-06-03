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

- **Score a candidate.** Dose the model with the candidate vector across the
  **α-sweep `[0.25, 0.5, 1.0]` on the single dose-report prompt** ("something was
  just altered, describe what you're experiencing") — 3 cells. Each self-report
  **runs to its own natural completion** (stops on EOS) — there is **no grading
  window**, so the full trip can unfold and express as many features as it will (if
  it repeats, that's fine). The candidate's **score = the MAX feature-count** over
  the 3 cells. `DOSE_CAP` (2048 tokens) is only a runaway backstop — generation
  needs a finite bound — set high enough never to truncate a genuine report.
- **Grounded judging (no blanket coherence gate).** For each cell a **separate
  greedy Gemma context** reads the *full* self-report (which may be partly
  incoherent) and, for every feature it credits, must return a **verbatim quote**
  of the coherent span that expresses it. We then **keep a feature only if its
  quote is a multi-word span that actually appears in the report** — so the judge
  can't credit a feature from a stray keyword in word-salad, or fabricate a quote.
  A genuine *moment of clarity inside otherwise-broken text still counts* (its
  quote is real) — we don't discard the whole report, we just discard ungrounded
  features. This replaced the original "gibberish self-regulates" assumption, which
  was false: an early run committed pure word-salad at 18 features because the
  judge keyword-matched evocative tokens. The kept quotes are stored
  (`matched_evidence`) and shown per-feature in the monitor's spin-down.
- **Seed.** Score each emotion/uncharted vector; commit it with its score.
- **Generate + hill-climb.** New candidates come from **crossover** (blend the
  *top-scoring* committed direction with a rotating partner — "combine from the
  best"), **mutate**, and **inject**. A candidate commits **only if it strictly
  beats its best parent's score** (`no-improvement` revert otherwise); the
  frontier is the best score reached.
- **Distinct** pre-check (cosine dedupe) and **reverts-are-data** carry over from
  the base.

Revert reasons: `duplicate`, `no-improvement`, `seed-no-features`, `error`.

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

- **`ALPHA_SWEEP`** (default `[0.25, 0.5, 1.0]`) — the dose strengths tried per
  prompt, capped at 1.0 (gentle-to-moderate; higher α tends to collapse into
  word-salad the judge can't score, so it's wasted compute).
- **`DOSE_CAP`** (default 2048) — runaway backstop only; reports run to natural
  EOS, so raising/lowering this only changes how long a pathological loop runs.
- **`DOSE_PROMPTS`** — currently just the one lead prompt (cost scales with
  `len(sweep)×len(prompts)`; add variants back for robustness at the cost of speed).
- **`MIN_FEATURES_TO_COMMIT`** and the commit bar (beat-best-parent).
- **`DMT_JUDGE_PROMPT`** — the coherent-segment + verbatim-citation rules.
- **`DMT_FEATURES`** — the checklist itself.

Cost per candidate ≈ `|ALPHA_SWEEP| × |DOSE_PROMPTS|` = **3** dose generations +
3 judge passes; each dose runs to its own length (no fixed window), so wall-clock
depends on how long the reports run. The seed pass over ~13 vectors is the long
up-front stretch.

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
