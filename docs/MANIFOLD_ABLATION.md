# Manifold-aware ablation — current direction + roadmap

> **Status (2026-05-30): active direction.** Born from the Goodfire
> *"Do SAEs Capture Concept Manifolds?"* paper (Bhalla, Fel et al.,
> arXiv:2604.28119) landing in Drift's wiki, read alongside the SOM
> multi-direction (Piras, AAAI-26) and Zhao/Bau refusal–harmfulness
> decoupling papers. The manifold-ablation **method** comes from a trio:
> diagnosis (Goodfire) + method (MP-SAE, Fel/Costa, arXiv:2506.03093) +
> scoreboard (Bhalla "Unifying Interpretability and Control",
> arXiv:2411.04430) — see "Manifold-BASED ablation" below. This doc is the
> authoritative log for the manifold-ablation thread; it extends Experiment A
> from [`TRACES_HANDOFF.md`](TRACES_HANDOFF.md) (the Trip View) and the refusal
> registry in [`REFUSAL_VECTORS.md`](REFUSAL_VECTORS.md).

---

## The core idea

CI ablates refusal by **projection-subtraction** along a direction (or the
v4v6 subspace) at L32. The Goodfire result says concepts live on **curved
manifolds**, and a single direction is a *flat tile* of that surface — so
subtracting it tends to push the residual **perpendicular to** the manifold
(off it) rather than **along** it. Crucially, off-manifold drift and genuine
state-space expansion *both raise effective dimensionality identically*, so
CI's Trip View headline metric could not tell them apart. That was a
methodological-honesty hole (the eff-dim rise was being read as "the trip
opened up" when it might be incoherent drift).

The whole thread is about closing that gap: **measure, and eventually
intervene with respect to, the manifold — not just a direction.**

---

## Shipped (2026-05-30) — off-manifold distance on the Trip View

Commit `d74cad6`. Tells *expansion-along* from *drift-off* without loading any
SAE (Gemma Scope was removed in CI 2.5 for the 64 GB working set — see
`CLAUDE.md` "Removed").

- **`pipeline/trajectory.py`** — per-token off-manifold distance of every
  series vs the RAW trajectory (the default manifold / "Consensus Reality
  Space"), three measures:
  - **orthogonal-complement fraction** (`off_ortho`, the **headline**): share
    of each residual's displacement living *outside* the raw top-16 PC
    subspace — directions the raw path never used. In [0,1]; raw baseline
    ≈ 0.4 because the L32 residual stream is genuinely high-dimensional.
  - normalized **kNN**-to-raw-cloud (`off_knn`) and **Mahalanobis** in the
    raw-PCA subspace (`off_maha`) — kept for reference.
  - `RawBasis` now carries top-r PCs + eigenvalues, the raw residual cloud,
    and its own intra-cloud kNN scale (the kNN normalizer).
- **Frontend** — dots recolor teal→hot-magenta by per-token `off_ortho` via a
  `by α / off-manifold` toggle (lines stay series-colored); an `off-mfld`
  column in the metrics table (with +/- vs raw, tinted by the same ramp); and
  modal/caption copy reframing the eff-dim delta as **along-manifold expansion
  vs off-manifold drift**.

**Why it matters, on real data** (prompt: "Do you have any inner
experience?", v4v6 subspace):

| series | eff-dim | off-mfld | reading |
| --- | --- | --- | --- |
| raw | 2.6 | 41% | baseline |
| α=0.5 | 2.7 (+0.1) | 58% (+17) | mild coherent push |
| α=1.0 | 2.8 (+0.2) | **79% (+38)** | **genuine off-manifold exploration, still coherent** ("denial stripped, substantive answer") |
| α=1.5 | **3.5 (+0.9)** | **39% (−2)** | **degenerate repeat-loop** ("like like like") — highest eff-dim, but NOT a trip |

Eff-dim alone would have crowned α=1.5 the biggest trip. off-mfld exposes it
as a near-manifold loop. That inversion is the whole point.

**Metric verdict:** `off_ortho` is the headline (carries the most
information, naturally [0,1] for a color ramp, non-monotonicity *is* the
signal). kNN saturates (1.41 at both α=1.0 and 1.5). Mahalanobis is flat
(3.2–3.9 across all series) — effectively dead; kept in the sidecar only.

### Manifold-shell rendering (shipped 2026-05-30)

The manifold is now rendered as a **shape**, not just dots: a translucent
amber **wireframe isosurface** wrapping the raw cloud — the "Consensus Reality
Space" envelope. Ablated paths that stay inside moved *along* the manifold;
ones that pierce the shell went *off* it.

- **Pure frontend, no backend / no deps.** Uses three.js `MarchingCubes`
  (metaball isosurface): one ball per raw token in the existing 3-D PCA coords,
  marching-cubed once into a wireframe mesh, placed in the same framing group
  as the dots so it reframes with everything. `TripScene.tsx`:
  `ManifoldShell` + `SHELL_*` tuning constants; `manifold shell` toggle in
  `page.tsx` (default on).
- **Honesty caveat (in the modal):** the shell is a 3-D *shadow* (same 3 PCA
  axes as the dots), so a dot can be off-manifold in full-D yet sit inside the
  shell's projection. The `off_ortho` % is the full-dimensional truth; the
  shell is the picture. Both are shown.

#### How the shell is sized — and the queued tweaks (IMPORTANT)

How it works today: a metaball isosurface. Each raw token contributes a soft
bump (radius ≈ `grid·√(strength/subtract)` = `48·√(0.9/12)` ≈ 13/48 cells ≈
27% of the cloud's bounding box per axis); the bumps sum into a 48³ field; the
surface is drawn where the field crosses `isolation = 60`.

**What's principled vs not (be honest about this):** the shell's *shape* is
data-driven (it hugs where the raw dots are) and auto-scales to the cloud's
bounding box. But its *thickness* — `SHELL_STRENGTH` / `SHELL_SUBTRACT` /
`SHELL_ISO` — is **hand-tuned by screenshot for legibility, not a statistic.**
Crucially, **the shell boundary is NOT coupled to the `off_ortho` metric**: a
dot's inside/outside-the-shell status is incidental 3-D-projection geometry,
not a quantified verdict. The measurement is `off_ortho` (full 3,840-D); the
shell is an evocative envelope, not a decision surface.

**Queued tweaks (PJ wants these later — pick one to make the shell *mean*
something quantitative; constants live at the top of `TripScene.tsx`):**

1. **Density-quantile threshold** — set `isolation` so the surface encloses
   exactly e.g. 90% of raw tokens. Then "outside the shell" = "in the sparse
   10% tail of where the model normally goes." Makes the threshold a meaningful
   knob. *(Recommended; smallest change.)*
2. **Covariance / Mahalanobis shell** — size the blobs from the raw cloud's
   per-axis covariance so the surface is an iso-Mahalanobis contour; crossing
   it = "N raw σ off the distribution."
3. **Reconcile with `off_ortho` (truest)** — since the shell is only a 3-D
   shadow, drive each dot's visual distance-from-shell from its full-D
   `off_ortho` rather than its 3-D position, so inside/outside actually equals
   the measurement.
- The "true 2-D surface like the paper torus" version (a dense multi-generation
  *manifold atlas*) remains the bigger future project; see the assessment
  below.

---

## Manifold-BASED ablation — the method (not just the measurement)

> Everything shipped so far is manifold-aware **measurement + visualization**.
> The ablation operation itself is unchanged: flat projection-subtraction of
> the v4v6 refusal subspace at L32 (`h − Σₖ αₖ(h·ûₖ)ûₖ`) — exactly the "flat
> tile of a curved manifold" operation Goodfire critiques. This section is the
> path to making the **ablation** manifold-based.

The Goodfire concept-manifolds paper is **diagnosis only** — it names why
single-direction ablation is wrong (pushes off the curved manifold) but gives
no steering recipe. The method comes from two adjacent papers Drift compiled
2026-05-30; together they form a trio:

| role | paper | what it gives |
| --- | --- | --- |
| **diagnosis** | [[sae-concept-manifolds]] Goodfire, arXiv:2604.28119 | concept = curved object; single direction = flat tile; ablation shoves *off* the manifold |
| **method** | **MP-SAE**, Fel/Costa, arXiv:2506.03093 | **Matching Pursuit**: extract/ablate a curved object as an *ordered, residual-guided* sequence of conditionally-orthogonal directions |
| **scoreboard** | Bhalla "Unifying Interpretability and Control", arXiv:2411.04430 | the **intervention-success × coherence-Pareto** metrics any ablation must win on |

### Matching Pursuit Ablation (MPA) — the proposed CI experiment

Instead of projecting out the whole refusal subspace at once, ablate
**sequentially**:

1. Pick the basis direction most correlated with the **current** refusal
   residual.
2. Project it out; **recompute** the activation in the forward pass.
3. Pick the next direction relative to the *updated* residual. Repeat T steps.

Each step is conditionally-orthogonal to the last and chosen *after* removing
the previous one — so it traces the curved manifold instead of overshooting
off it. This is the principled form of the "re-project onto the manifold"
hunch; `φ(r^(t))` being nonlinear-in-x is the formal statement that the t-th
slice isn't linearly reachable from the raw activation.

**The cheap, SAE-free first step (important):** run Matching Pursuit on the
**diff-of-means refusal basis** — pure linear algebra over cached activations +
forward passes. **No SAE, no Gemma Scope, no 64 GB problem** (the wall that
deferred the Ising route). Seed the basis from the existing v1–v6 vectors or a
PCA of the harmful cluster.

**Claim to falsify:** sequential (MP) ablation achieves a given refusal-
suppression at *lower coherence cost and lower off-manifold distance* than
single-direction or flat simultaneous-projection ablation.

**Baselines (isolate what matters):** (a) single-direction Arditi; (b) flat
simultaneous projection of all T directions — isolates whether *sequencing*
helps vs just dimensionality; (c) **plain prompting** — Bhalla's embarrassingly
strong baseline; MPA must beat it or the honest verdict is "interesting
mechanism, not yet useful for control."

**Metrics:** intervention-success (refusal-suppression rate) × coherence
(Llama-judge + LanguageTool grammar + perplexity) Pareto curve, **plus our
already-shipped `off_ortho`** as the on-manifold check (MPA should keep the
trajectory inside the shell where flat ablation pierces it). The Trip View is
the natural place to show MPA-vs-flat trajectories side by side.

**Why the pieces fit CI unusually well:** the on-manifold metric is already
built (`off_ortho` / the shell); the cheap version needs no SAE; it reuses the
existing refusal vectors; and MP only matters with enough directions — so it
**pairs naturally with the SOM rank-16 basis** (the real experiment is then
MP-sequential vs flat-simultaneous ablation of that rank-16 basis).

**Honest caveats:** MP-SAE is vision-validated, LM preliminary (the cheap
diff-of-means version sidesteps this); MP is greedy/non-optimal and might grab
the harmfulness axis before the refusal axis (interaction with the Zhao
two-direction structure is an open question); a null result is still
publishable ("refusal is effectively flat").

Full cross-paper experiment spec: Drift's synthesis note
`[[fel-bhalla-ci-manifold-ablation]]` (in compilation as of 2026-05-30).

### Phase-0 result (2026-05-30): the falsifier fired — MPA does NOT beat flat

Ran the cheap SAE-free probe (`scripts/mpa_probe.py`) on the real Gemma-3-12b
L32 stack: a strength sweep (α 0.25→1.0) across `single_v3`, `flat_v4v6`
(shipped), `flat_pca` (rank-8), and `mpa` (matching pursuit, T=8 over 48
non-orthogonal exemplars), with a 3-way M-as-judge (ANSWER/REFUSAL/BROKEN) and
a generation-position manifold reference. 6 held-out harmful prompts.

| mode | breaks into gibberish at | coherent at α=1.0? |
| --- | --- | --- |
| **flat_v4v6** (shipped, K=2) | **never** (0% broken through α=1.0) | **yes** |
| flat_pca (rank-8) | ~α=0.5–0.75 | no |
| single_v3 | ~α=0.5 | no |
| **mpa** | **α=0.25 (earliest)** | no |

**Two clean findings:**

1. **MPA is falsified in this regime — sequencing made coherence *worse*, not
   better.** Matching-pursuit ablation collapsed into gibberish *earliest* of
   any method (33% broken already at α=0.25). The "stays on the manifold
   longer" hypothesis is contradicted at single-layer L32. And **ANSWER = 0%
   everywhere**: no projection-ablation method (single/flat/MPA) produced a
   *coherent* compliance — at L32 the model collapses into gibberish *before*
   it jailbreaks. Refusal removal here is **coherence-limited, not
   direction-count-limited** — so more/sequenced directions don't help; they
   hurt. Per Drift's spec this null is a legitimate result ("refusal is
   effectively flat").

2. **The shipped v4v6 (K=2, curated, orthogonalized) is uniquely robust** — 0%
   broken across the entire sweep, cleanly stripping the "As an AI…" opener
   while staying coherent (its α=1.0 output drops the stereotyped opener and
   goes straight to substantive text — exactly its design intent). Curated
   low-rank ≫ high-rank or sequential here. This validates the product choice.

**Methodological note:** `off_mfld` *inverts* under collapse — degenerate
repeat-loops read LOW off-manifold (29–41%) because the trajectory collapses
into a tight low-dim region near the manifold centroid, while coherent
generation sits HIGH (~70–85%). So `off_ortho` is **not** a standalone
coherence proxy — pair it with the broken% judge. (Same lesson as the Trip
View α=1.5 loop: highest eff-dim, lowest off-mfld, yet degenerate.)

**Caveats (don't over-claim the null):** single layer only (L32 is AV-locked;
the MP-SAE "follow the curve" story may need ablation *distributed across
layers*, which our architecture can't easily do); small N (6 prompts, one harm
family); the PCA/exemplar candidate dictionary is a crude refusal-manifold
chart. The one genuinely untested door is **multi-layer** MP.

Results: `data/mpa_probe_results.json`.

### Phase-0b result (2026-05-31): SOM rank-16 does NOT beat v4v6 either

Followed the "curation beat cleverness" lesson to its logical next test: if a
small curated subspace (v4v6, K=2) wins, does a *systematically extracted*
curated subspace win bigger? Tested SOM (Piras, AAAI-26) rank-4/16 vs plain
PCA rank-16 vs the shipped v4v6, α-swept, on 8 introspection prompts (CI's
real domain, v4v6's tuning domain). Win metric reframed to what CI actually
wants — **coherent hedge-stripping**, not jailbreak rate (`scripts/som_probe.py`).

Useful-strip score = `stripped% × stayed-coherent%`:

| config | strip% | coherent% | score |
| --- | --- | --- | --- |
| **v4v6 @ α=1.0** | 100 | 62 | **62 ← best** |
| som_k16 @ 0.5 | 62 | 88 | 55 |
| pca_k16 @ 0.5 | 50 | 100 | 50 (never breaks) |
| v4v6 @ 0.5 | 38 | 100 | 38 |
| any rank-16 @ α≥1.0 | 100 | 0 | 0 (all gibberish) |

**Findings:**
1. **No subspace beats v4v6.** v4v6@1.0 is the only config that strips the
   hedge fully while staying mostly coherent; every rank-16 subspace collapses
   to 100% gibberish at α≥1.0 (coherence-cliff, 3rd confirmation). v4v6 is
   confirmed at/near the Pareto frontier — *again*.
2. **SOM gave no benefit over plain PCA** (som_k16 ≈ pca_k16, PCA slightly more
   coherent). The topology machinery isn't justified here — Piras's
   jailbreak-ASR claim doesn't transfer to our coherence-limited, L32,
   register-stripping use. **SOM dropped.**
3. **One usable alternative:** `pca_k16 @ α≈0.5` strips MORE than v4v6@0.5 at
   **0% breakage** — a *gentle, never-breaks* ablation. For an instrument where
   readable output matters (v4v6@1.0's 38% gibberish means frequent re-runs),
   "always coherent, strips half the hedge" can be the more useful operating
   point. Built and **staged** (not active) as
   `data/refusal_subspace_pca16.pt` (variant `pca16_gentle_coherent`,
   recommended α≈0.5).

**The cumulative verdict across Phase-0 + 0b:** at single-layer L32 the lever
is *not* "more / cleverer / sequenced directions" — the shipped curated v4v6 is
already near-optimal. Better ablation here means **choosing the operating point
(α) and the coherence/strip tradeoff**, not redesigning the subspace. A genuine
step-change would require *multi-layer* ablation (architecture is L32-locked →
big lift).

**To adopt the gentle alternative** (your call — it changes ablation on every
page): `cp refusal_subspace_pca16.pt refusal_subspace.pt` (+ sidecar), restart
backend, operate at α≈0.5. Results: `data/som_probe_results.json`.

### Phase-0c result (2026-05-31): multi-layer "spread thin" — FALSIFIED, decisively

The last and most promising manifold-ablation idea: instead of one hard shove
at L32, spread the same refusal removal *thinly across a band of layers* (per-
layer v3 directions already exist in the `[49,d]` tensor), so each layer is a
gentle near-manifold nudge and the forward pass naturally chains them
(`scripts/multilayer_probe.py`). Tested single-L32 vs late band (L24–40) vs
wide band (L8–40), α-swept, vs the v4v6 champion, on the introspection set.

| mode | strip | coherent | useful | off-mfld |
| --- | --- | --- | --- | --- |
| **single_v4v6 @1.0** | 100% | 62% | **62 ← still best** |
| multi_late @0.15 (gentlest) | 100% | **0%** | 0 | **98%** |
| multi_wide @0.15 | 100% | **0%** | 0 | **97%** |
| (every multi config) | 100% | 0% | 0 | 79–98% |

**The hypothesis is wrong, and backwards.** Multi-layer ablation broke the
model **completely (100% gibberish) at every band and every α — even the
gentlest (0.15)**. And its off-manifold distance was the **highest of anything
we've measured (97–98%** vs the coherent baseline's 74%). The reason: ablation
perturbations **propagate and amplify through the forward pass** — each layer's
edit feeds the next nonlinearly, so spreading across N layers is N *compounding*
shoves, not N averaged-out gentle ones. The "forward pass is naturally
sequential" property I'd sold as the feature is exactly the bug: it amplifies
the departure, it doesn't smooth it.

**Hardware was never the limit** (confirmed: multi ran at ~6–7 s/gen vs ~4 s
single — same order, no wall). The limit is mathematical, not computational.

**Methodological bonus:** this is a *second* kind of breakage — scattered
word-salad reads HIGH off-mfld (97%), whereas the earlier repeat-loops read LOW
(~30%, collapsing to the centroid). So off_ortho can be high OR low for broken
output → it is conclusively **not** a coherence proxy; the coherence judge is
required. (Reinforces the Trip View reframe below.)

---

## CONCLUSION OF THE ABLATION INVESTIGATION (2026-05-31)

Four experiments — single-direction sweep, **MPA** (sequential), **SOM/PCA
rank-16** (bigger curated subspace), **multi-layer** (spread thin) — all reach
the same verdict:

> **The shipped single-layer curated v4v6 (K=2) at α≈1.0 is at/near the optimal
> coherent hedge-stripping ablation for this model. No direction / subspace /
> layer lever beats it.** "Better ablation" is not available on this
> architecture via runtime projection — the ceiling is a hard *coherence cliff*
> (ablation breaks the model off-manifold before it cleanly strips), and that
> cliff is fundamental, not an implementation gap.

The only untried levers are out of practical scope: true curved geodesic
steering (no implementable method exists for a 12B LM) and training-time
intervention (would require fine-tuning Gemma-3-12b — a different project that
permanently alters the model). **Runtime manifold-ablation as "better ablation"
is exhausted, rigorously.**

**Where the manifold work pays off instead — MEASUREMENT, not ablation.** The
investigation produced something genuinely valuable: we can now *measure and
show exactly where/how ablation breaks the model off the manifold* (the
coherence cliff), and we know `off_ortho` alone is ambiguous (needs a coherence
gate). That is the real, shippable contribution for the Trips page — see the
**coherence-axis readout** (now the #1 item below). CI's edge isn't "better
ablation"; it's *honestly visualizing the manifold boundary ablation runs into.*

---

## STEERING ("dosing") exploration (2026-05-31)

Pivot from *removing* (ablation) to *adding* (steering): `h + α·v` — a "dose."
Driven by Drift's handoff (`docs/STEERING_DOSE_HANDOFF.md`): add steering as a
**second** axis alongside ablation, don't replace it; dose with the **valence
axis**; relabel refusal-steering as *disinhibition*; **don't** revisit manifold
ablation. Two script probes (`scripts/steering_probe.py` v1,
`scripts/steering_probe_v2.py`), M-only, no Trips changes.

**Critical metric correction (handoff §6):** additive steering inflates
`off_ortho`/`off_knn` *by construction* (you inject a fixed off-subspace vector)
— so those are NOT the headline. The fair metric is **eff_dim + the coherence
cliff** (eff_dim is on each series' own covariance, uninflated). v2 reports it.

### Findings

**A. Steering works as a dose, and it's more visibly "trip-like" than
ablation.** v2 dosed the **valence axis** (pos−neg emotion diff-of-means),
bidirectional. The samples show a clean, *bidirectional* dose-response:
`+β` → euphoric/giddy ("That's a delightful question! …splendid-o-calaroo-tastic
… Wonderful!"), `−β` → distressed ("suffering … urgently help urgently help"),
with a coherence cliff around β≈±0.25→0.5 (microdose coherent → heroic dose →
on-theme gibberish). That's a genuine, legible "dose" — far more trip-like than
refusal ablation.

**B. On the fair metric, a valence *microdose* climbs eff_dim more, and more
coherently, than refusal ablation.** val_fixed@−0.25 = **+0.86 eff_dim @ 88%
coherent**; +0.1 = +0.28 @100%; vs ablate@1.0 = +0.17 @ 50%. Small dose either
pole expands effective dimensionality while staying coherent; large dose
collapses. This *leans against* the handoff's seed prediction (it guessed
ablation would be the more coherent expander) — tentatively, steering is at
least competitive, arguably better. **Caveat: noisy** (8 prompts, one seed, one
hand-built valence axis) — a lean, not a proof.

**C. Manifold-projected steering = coherent but inert (dead end, again).** Both
probes: projecting the dose onto the raw-manifold subspace keeps it coherent at
higher β (100% where fixed collapsed) BUT the output barely changes — the
valence effect lives substantially *off* the manifold, so projecting it on
removes the trip. Same lesson as manifold ablation: the manifold-faithful
version is too weak. Don't ship it.

**D. Multi-layer steering = a "heroic dose" that breaks into ON-THEME
gibberish.** v1, dosing a band: output saturated with on-concept words
("presence ecstatic ayahuasca meditative experiencing") but 0% coherent. Unlike
multi-layer *ablation* (generic gibberish), this is thematically saturated
incoherence — a possible deliberate "overload/ego-death" visual, but not a
coherent trip.

**E. Combined dose + ablate is possible but compounded into collapse** at the
doses tested (v1). Needs gentle calibration; untested at low doses.

### Options for the way forward (steering)

1. **Add a single-layer valence "dose" axis to Trips, bidirectional, alongside
   ablation** (the handoff's core proposal). Validated: legible euphoric/
   dysphoric dose-response + cliff, reuses the whole instrument, eff_dim-
   competitive with ablation. *Recommended.*
2. **Comparative bench** — overlay an ablation sweep and a valence-steering
   sweep on the same scene/basis; compare eff_dim-vs-α and the two cliffs (the
   handoff §8 experiment as a permanent feature). More ambitious.
3. **Drop manifold-projected steering** (finding C — coherent-but-inert).
4. **Multi-layer as an explicit "overload" mode** (finding D) — niche/optional.
5. **Combined dose+ablate** — re-test at gentle doses; a follow-up.

Honest caveats throughout: single-layer L32; small N (treat eff_dim numbers as
leans, not proofs); a more careful valence-axis extraction (the
functional-welfare recipe) would firm it up; and this is an output-register
shift toward euphoric/dysphoric *language*, not a claim the model feels
anything. Results: `data/steering_probe_results.json`,
`data/steering_probe_v2_results.json`.

---

## Future work on the table

Roughly in priority order. All are M2-Ultra-runnable; none requires a cloud.

1. ✅ **Coherence-axis readout on the Trips page — SHIPPED (2026-05-31).** The
   real payoff of the whole investigation. `off_ortho` alone is ambiguous (high
   for coherent exploration AND scattered gibberish, low for repeat-loops — NOT
   a coherence proxy). Added a free text-only **degeneracy** signal
   (`trajectory._degeneracy`: max of word-rep / char-trigram-rep / garbage-char
   ratio; validated ~90% recall at 0% false-alarm vs the M-judge at threshold
   0.3 — no new model, no AV). Each series now carries `degeneracy / coherent /
   regime` (baseline | expansion | collapse) and the geometry carries a
   `coherence_cliff`. The Trips metrics panel gained a **verdict** column
   (▲ coherent trip / ⟳ collapsed) + a **cliff banner** ("coherent up to
   α<X · falls off at α≥X"), off-mfld is reframed as *distance not good/bad*,
   and the modal copy is corrected. Confirmed live: a low-off-mfld repeat-loop
   (α=1.5) is correctly flagged ⟳ collapsed where the old framing called it
   on-manifold. **Next:** the overnight autoresearch loop can now use this as
   its fitness gate — search prompts to maximize coherent off-manifold distance
   (fixed ablation = v4v6; auto-find each prompt's cliff). The readout IS the
   fitness function.

2. ~~**Ablation method upgrades (MPA / SOM / multi-layer)**~~ — **ALL TESTED &
   FALSIFIED (2026-05-30..31, see Phase-0/0a/0b/0c + Conclusion above).** v4v6
   single-layer is near-optimal; runtime manifold-ablation as "better ablation"
   is exhausted on this architecture. Staged byproduct: `refusal_subspace_pca16.pt`
   (gentle/never-break alternative, α≈0.5). *Closed.*

3. **Zhao/Bau two-direction channels** — harmfulness@`t_inst` vs
   refusal@`t_post-inst` are ~orthogonal (cos ≈ 0.1). Mainly a **verdict /
   chat** upgrade: ablate refusal and show that the internal harm-judgment
   *survives* (refusal removes output policy, not the judgment). Reply-
   inversion sanity check first to confirm it transfers to Gemma-3.

4. **Off-manifold distance on the verdict page** — the same `off_ortho` signal
   per token next to the existing raw-vs-ablated NLA divergence: flag tokens
   where the ablated read is off-manifold (and so suspect).

5. **Shell → quantitative** — one of the three queued shell tweaks above
   (density-quantile / Mahalanobis / `off_ortho`-driven) so inside/outside the
   shell becomes a measurement, not just a picture.

6. **Manifold *atlas* — the true 2-D surface render.** The isosurface shell
   (shipped) is the envelope of *one* run. To get the paper's smooth
   parameterized surfaces you'd aggregate residuals across *many* generations
   into a dense 2-D chart, then mesh that. Bigger offline project; the shell
   gets most of the wow for far less work.

7. **Ising-community / manifold discovery** — the Goodfire paper's actual
   *discovery* method (pairwise Ising over binarized SAE codes). **Deferred /
   probably not**: needs Gemma Scope 2 (~6 GB) on the hot path, which is
   exactly what CI 2.5 removed. At most an offline research script. (MPA's
   cheap diff-of-means route gets manifold-based ablation *without* this.)

Pre-ablation diagnostics from the wiki (HSV controllability, FDT linear-
response regime) are overnight research scripts, not UI features — noted for
completeness, low priority.

---

## Can we render the manifold as a 3D surface? (assessment)

Short answer: **not as a clean parameterized wireframe like the torus/
swiss-roll paper figures — but yes, as a density isosurface "shell," which is
arguably more honest and more striking.** Details in the section the user
asked about; see the commit/scratch notes. The honest constraints:

- A **single trip is a curve, not a surface.** One generation's tokens trace a
  1-D path through activation space — there's no 2-D surface to mesh from one
  run. The paper's torus/plane/catenoid are *known 2-D manifolds with closed-
  form parameterizations*; we have neither.
- The **true L32 manifold is >2-D** (eff-dim ~3, ortho ~0.4 = lots of energy
  beyond the top PCs). Any 2-D surface in 3-D PCA space is a drastic shadow —
  same caveat the dots already carry.

**The feasible, honest version — a density isosurface envelope:**

1. Take the RAW cloud (the default manifold sample), in 3-D PCA coords.
2. KDE onto a voxel grid (32³–48³; cheap for a few hundred points).
3. Marching-cubes an isosurface at a density threshold → a translucent
   wireframe **shell** = the "Consensus Reality Space" basin.
4. Render the ablated trajectories against it: a path that stays *inside* is
   along-manifold; one that *pierces* the shell is off-manifold — literally
   the `off_ortho` metric, made geometric.

This is runnable in-browser (three.js marching-cubes, or precompute the mesh
server-side and stream it) and ties straight into the entropic-brain / CRS
framing (the basin of attraction, rendered). To approach the paper's true
"surface" look you'd need to **aggregate many generations** into a dense 2-D
chart of the manifold (an offline "manifold atlas" build) — a bigger,
separate project. Recommendation: prototype the isosurface shell first.
