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

---

## Future work on the table

Roughly in priority order. All are M2-Ultra-runnable; none requires a cloud.

1. ~~**Matching Pursuit Ablation (MPA)**~~ — **TESTED & FALSIFIED at
   single-layer L32 (2026-05-30, see Phase-0 result above).** Sequencing made
   coherence worse, not better; the shipped v4v6 already wins. The only
   remaining variant worth trying is **multi-layer** MP (ablation distributed
   across layers, as the MP-SAE method really intends) — but L32 is AV-locked,
   so this is a bigger lift and lower priority now. *Not a near-term item.*

2. **SOM rank-16 multi-direction refusal subspace** (Piras, AAAI-26) — an
   *afternoon* compute script: cache ~1k harmful/harmless L32 residuals, train
   a 4×4 SOM, derive a coherent rank-16 refusal subspace, drop it in as a
   principled `refusal_subspace` variant. Replaces the 6 hand-picked vectors
   with a systematic family; benefits **every** ablation channel (trip / probe
   1b / chat) since they all route through `pick_ablation_target`. Provably
   reduces to current behavior at k=1. On Gemma2-9B-it (closest substrate) MD
   ablation hit 96% vs 90% single-direction. **Pairs with MPA** — the rank-16
   basis is what makes sequencing (MP) vs simultaneous (flat) a real contrast.
   See `REFUSAL_VECTORS.md`.

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
