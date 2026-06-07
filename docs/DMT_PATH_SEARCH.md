# DMT search: from a vector to a *path* (possible direction)

**Status: exploration / not built.** This captures a candidate next direction for the
DMT autoresearch (`/autoresearch-dmt`): **extend the search from a single steering
*vector* to a steering *path* (trajectory) through a low-rank subspace.** It records
the motivating insight, a diagnostic we ran (2026-06-07) that supports it, the design
options, and the open risks — so it's ready to pick up. Nothing in the loop changed.

Origin: Drift's handoff `CI_LORA_DMT_HANDOFF.md` (LoRA/self-explainer ideas) →
discussion of the manifold/subspace literature already in CI's orbit.

---

## The setup today

A DMT candidate is a single additive direction `v` at L20: the dose is `h + α·v`
(`install_runtime_steering_hook`, ramped over `GEN_RAMP=16`). The search hill-climbs
over `v`; scoring averages DMT-feature counts over repeated stochastic doses
(`SAMPLES_PER_CELL=5`, `ALPHA_SWEEP=[0.25,0.45]`). Best direction so far:
`feat-download_transmission`.

## The key insight (why "search for a subspace" by itself is a trap)

A **static** subspace dose `h + Σ αᵢ·vᵢ` **is just a single vector** `w = Σ αᵢ·vᵢ`,
and the current search already ranges over *all* directions in ℝ³⁸⁴⁰ — which contains
every vector in every subspace. So "find the best subspace, then dose with a vector
from it" reaches **nothing the single-vector search can't already reach**. A static
single-layer subspace can only ever be a *search-efficiency trick* (a smart region to
seed/constrain the hill-climb), never an expressivity gain — it cannot break the
single-vector ceiling.

To genuinely exceed a single static L20 vector, the intervention must stop being one
static additive vector. Exactly three ways:

1. **Time-varying path (trajectory steering)** — the dose direction changes over
   generation: `h_t + v(t)`, tracing a curve. NOT reducible to a static vector. The
   real "manifold" move.
2. **Multi-layer** — directions at several layers (L16/L20/L24) at once. Not reducible
   to one L20 vector — but breaks the "a single direction at L20" deliverable framing.
3. **Non-additive operator** — project/constrain the residual onto a fitted DMT
   manifold instead of adding (the `project_out` ablation operator, inverted).

## Why a path could beat a point (mechanism)

Raising α on `h + α·v` eventually **teleports the residual off-manifold** → coherence
breaks → word-salad → the judge scores it low. Features peak at a moderate α then
collapse. A curved/scheduled path can traverse **more coherent state-space distance**
while staying on-manifold, potentially expressing more of the phenomenology before it
breaks. The win isn't reaching a different static point (a single vector could reach
that) — it's *travelling further without falling off the manifold*. This is exactly
the Wurgaft "manifold steering" reframe ("from finding the right direction to finding
the right geometry") and what CI's shipped `/trip` off-manifold metric measures.
Phenomenologically apt too: a trip is a trajectory (onset → visual → entity contact →
insight → return), not a fixed state.

## Diagnostic (2026-06-07): is the ceiling an off-manifold wall?

`scripts/dmt_offmanifold_diagnostic.py` doses the leader across a wide α range and, at
each α, measures DMT-feature count vs. off-manifold drift (`off_ortho_mean`, the
`trajectory.py` metric `/trip` shows) of the dosed L32 trajectory vs. a raw (undosed)
one. Result (leader `feat-download_transmission`, RAW baseline off_ortho = 0.371):

| α | mean features | off_ortho | Δ vs raw |
|---|---|---|---|
| 0.15 | 0.67 | 0.501 | +0.131 |
| **0.25** | **1.67** | 0.587 | +0.216 |
| 0.45 | 1.33 | 0.561 | +0.191 |
| 0.70 | 0.33 | 0.750 | +0.380 |
| 1.00 | 0.00 | 0.913 | +0.542 |
| 1.40 | 0.00 | 0.925 | +0.554 |

**Read 1 — off-manifold wall: SUPPORTED.** Features peak in a narrow low-α window
(α≈0.25) and collapse to 0 as α rises and the dose drives sharply off-manifold
(off_ortho 0.59 → 0.91). You *can't push the linear dose harder to get more features* —
harder = off-manifold = incoherent. That's the signature a coherent **path** could
get past. → path steering is worth prototyping.

**Read 2 — IMPORTANT caveat: the frontier is still noise-inflated.** The leader
committed at **mean 3.8** but re-scores here at **~1.5–1.7** (α=0.25 samples [1,3,1];
α=0.45 [2,0,2]). Averaging 5 samples reduced per-sample noise but the *search* still
selects lucky means (every commit/refine took an upward draw of mean-of-5), so the
board's frontier is inflated; the leader's **reliable** level is ~1.5–2 features, not
3.8. Raw/undosed self-report scores 0, so the dose *does* induce features — just
modestly and noisily. Any path search inherits this noisy objective; it may need more
samples per candidate or a less noisy scorer (cf. the LoRA self-explainer idea) before
"path beats point" can be measured cleanly.

## Design options (if we pursue it), ranked by upside-per-effort

1. **Trajectory steering through a low-rank subspace (the real experiment).** Build a
   rank-2/3 orthonormal subspace from the atlas's best *distinct* directions
   (`gram_schmidt` already exists; e.g. download_transmission + otherness/independent_agency
   + a dissolution direction). Dose along a *scheduled path* through it (ramp into
   direction 1, then 2, then 3 over generation) instead of a static sum. Generalizes
   the existing single-vector ramp hook to a piecewise schedule. **Genuinely more
   expressive, phenomenologically motivated, still a CI-decodable deliverable**
   (rank-k subspace + schedule is interpretable / NLA-decodable). Cost: search now
   includes the schedule (more candidates), each still one generation.
2. **Static subspace as a search trick (cheap, limited).** Generalize `crossover` from
   `w·a+(1−w)·b` (one knob, a point on a line) to `Σ αᵢ·vᵢ` over the top-K with
   *independent* weights. Easy, may find a better combined vector than pairwise
   crossover — but **cannot beat single-vector reach**, only find it faster. Frame as
   "better crossover," not "manifold steering."
3. **Manifold-constrained intervention (principled, ambitious, defer).** Fit a low-dim
   manifold to the real DMT-state activations we already capture, steer along geodesics
   / project onto it (faithful Wurgaft/Goodfire). Right version if curvature clearly
   dominates; deliverable gets murkier (a curved manifold is less cleanly "a decodable
   direction"). Only after (1) shows trajectories beat static.

Keep it **low-rank (k≤~4)** to preserve the decodable-direction deliverable. Multi-layer
steering (option 2 in the insight list) is also genuinely more expressive and is on the
`/trip` roadmap (`MANIFOLD_ABLATION.md`: SOM rank-16, Zhao two-direction) but breaks the
"single direction at L20" framing — keep it a separate thread.

## Open questions / risks

- **Noise first.** The frontier is inflated (~3.8 board vs ~1.5–2 reliable). Decide
  whether to harden the scorer (more samples / activation-readout) before or alongside
  path search — otherwise "path beats point" is unmeasurable.
- **Scoring cost.** The scorer is the bottleneck (~12–15 min/candidate). A schedule
  search multiplies the candidate count; budget accordingly.
- **Decodability.** A rank-≤4 subspace + schedule stays decodable; a full curved
  manifold or multi-layer object does not, cleanly. Don't drift out of the frame.
- **Does a path actually clear the wall?** The diagnostic shows linear hits a wall; it
  does NOT yet show a path gets past it. That's the experiment.

## Pointers

- `scripts/dmt_offmanifold_diagnostic.py` — the diagnostic above (re-runnable).
- `pipeline/trajectory.py` — `compute_raw_basis` / `build_series` / `off_ortho` (the
  manifold metric, shared with `/trip`).
- `pipeline/abliteration.py` — `install_runtime_steering_hook`, `gram_schmidt`,
  subspace-aware ablation (`[K, d_model]` basis precedent).
- `docs/MANIFOLD_ABLATION.md` — CI's manifold-aware ablation thread + roadmap.
- Wiki: `concepts/manifold-steering-shared-geometry.md` (Wurgaft), `sae-concept-manifolds.md`
  (Goodfire/Bhalla/Fel), `self-explanation-privileged-access.md` (the handoff's paper).
- `CI_LORA_DMT_HANDOFF.md` (iCloud) — Drift's handoff that kicked this off.
