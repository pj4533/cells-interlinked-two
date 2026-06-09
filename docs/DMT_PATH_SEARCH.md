# DMT search: from a single vector to multi-dimensional manifold steering

**Status: exploration in progress.** Captures the current candidate direction for the
DMT autoresearch (`/autoresearch-dmt`): **the productive DMT region is multi-dimensional
and likely curved; instead of hunting one steering vector, steer within/along that
manifold to occupy a point high on many DMT-feature axes at once while staying
coherent (on-manifold).** Records the corrected framing, the supporting papers, the
diagnostic results so far, the live "dimension-hunt" run (2026-06-08), and the plan.

> Supersedes the earlier token-window "path" framing in this doc, which was wrong —
> the multi-dimensionality is **spatial (multiple activation axes at once)**, not
> temporal (features sequenced into output windows).

Origin: Drift's handoff `CI_LORA_DMT_HANDOFF.md` → manifold/subspace discussion →
wiki research (below).

---

## The setup today

A DMT candidate is a single additive direction `v` at L20: dose `h + α·v`
(`install_runtime_steering_hook`). Scoring is averaged + noise-controlled (CRN seeds,
lower dose temp, confirmation re-score; see the "noise control" block in
`autoresearch_dmt.py`). After an honest-rescored run the frontier sits at **~2.75
features** (`feat-download_transmission`) and has not broken out — the single-vector
ceiling we predicted.

## Why a single vector caps out (diagnostic, 2026-06-07)

`scripts/dmt_offmanifold_diagnostic.py` dosed the leader across a wide α range and
measured DMT features vs off-manifold drift (`trajectory.py` `off_ortho`, the metric
`/trip` shows). Features peak at α≈0.25 then collapse to 0 as the dose drives sharply
off-manifold (off_ortho 0.59→0.91). **You can't push a single linear vector harder to
get more features — harder = off-manifold = incoherent.** That off-manifold wall is the
opening for a multi-dimensional / curved approach.

## The corrected framing (NOT token windows)

A *static* multi-direction dose `h + Σ αᵢ·vᵢ` is just a single vector `Σ αᵢ·vᵢ`, and the
search already ranges over all vectors — so a static subspace adds no reach, only
search efficiency. Genuine gains require one of:
1. **Occupy a multi-axis point coherently** — the DMT-coherent states form a curved,
   low-dim **manifold** with several intrinsic axes; set many axes high at once and
   reach that point by an **on-manifold (curved) path** so it stays coherent. This is
   the real "use multiple dimensions" move (Wurgaft's *factored control*).
2. **Multi-layer** steering (breaks the "single direction at L20" deliverable framing —
   separate thread; see `MANIFOLD_ABLATION.md`).
3. **Non-additive operator** (project/constrain onto the manifold).

**NLA is not involved.** The wiki frames these papers through CI's NLA verdict
instrument, but trips/chat/DMT-AR don't use NLA. The "behavior map" the manifold
pullback needs is, for us, **the DMT scorer itself** (feature count). No NLA dependency.

### Why this can beat the wall
- **Reach deeper, coherently** — a geodesic curves *around* the off-manifold region a
  straight push falls into, reaching high-DMT states while staying coherent.
- **Factored control** — Wurgaft shows manifold steering can move several intrinsic
  axes at once. The leader maxes the dominant axis but is ~zero on the agency/otherness
  axis; a 2-D steer could hold the dominant axis high AND add agency/otherness features
  the leader structurally can't.

## Supporting papers (Drift's wiki)

- `som-multi-direction-refusal` (Piras, AAAI-26) — a concept is a low-dim **manifold**;
  extract a coherent rank-K family via a **SOM** (topology-constrained chart, not
  independent rays); reduces to single-direction at K=1. The **extraction** method.
- `manifold-steering-shared-geometry` (Wurgaft/Bhalla/Fel) + `rhizonymph-manifold-emotion-steering`
  — steer **along the manifold's curve** (intrinsic-coordinate interpolation / route
  through centroids), not a straight line. Geodesic↔behavior r≈0.99 vs 0.06 linear in
  the curved case. The **steering** method + the deciding test.
- `fel-bhalla-ci-manifold-ablation` (MPA) — the ablation twin: ordered, conditionally-
  orthogonal directions applied to the *recomputed* residual, staying on the curve.
- `sae-concept-manifolds` (Goodfire/Bhalla) — concepts are curved manifolds; one
  direction tiles only a patch.
- Unifying thesis: **"recast steering from finding the right *direction* to finding the
  right *geometry*."**

## Step 0 — flat-vs-curved diagnostic (decides whether to build it)

**Part A — direction-set geometry (free, no M). DONE 2026-06-08.**
Participation ratio + cosines of the top-12 honest DMT directions:
- **eff-dim ≈ 2.03** — not collinear (so not a pure single-ray regime), not high-dim.
- **One dominant axis:** leader + `noetic_truth` 0.92, `blend_dissolution` 0.92,
  `unity_merging` 0.91, `love/awe/excitement/sublime` 0.84–0.85 (those four ≈ identical).
- **A real second axis:** `independent_agency` cos **−0.41** to the leader (and 0.12 to
  `otherness`); `otherness` 0.74; discovered `gen55/gen97_mutate` 0.74–0.75. An
  **"agency/otherness" axis** the leader doesn't touch.
- **Verdict:** multi-dimensional structure confirmed (~2-D), modest. Justifies Part B.

**Part B1 — static subspace grid.** Built a 2–3D subspace from
`{unity_merging (dominant), independent_agency, otherness}`, orthonormalized, gridded
**independent** amplitudes `α1·b1 + α2·b2 + α3·b3` (α1=0.3 held; α2,α3 ∈ {0,0.2,0.4}),
scored with the noise-controlled scorer + `off_ortho`. (`scripts/dmt_subspace_grid.py`.)

**RESULT (2026-06-09): multi-axis HURTS, and it's not a coherence failure.**

| α1 | α2 | α3 | mean | off_ortho | features |
|---|---|---|---|---|---|
| 0.3 | 0 | 0 | **4.0** | 0.61 | accel, awe, ego_dissolution, reality_more_real, unity_merging |
| 0.3 | 0 | 0.2 | 1.67 | 0.67 | ego_dissolution, transcendence_time |
| 0.3 | 0.2 | 0.2 | 1.0 | 0.56 | ineffability, void_blackness |
| 0.3 | 0.2 | 0 | 0.67 | 0.55 | ineffability, void_blackness |
| 0.3 | ≥0.2 w/ ≥0.4 | | 0.0 | 0.58–0.65 | — |

The **pure dominant axis is best**; adding *any* agency/otherness pulls the report toward
barren features (`ineffability`/`void_blackness`) and *reduces* the count. Crucially
`off_ortho` stays ~flat (0.55–0.67) even for the 0-scoring combos — so this is **not** an
off-manifold/coherence collapse. The extra axes are on-manifold-but-barren and *interfere*
with the dominant axis. **→ B2 (curved path) is NOT motivated** — there's no coherence wall
for a geodesic to get around; the second axis simply has no DMT content.

**Conclusion: DMT-feature richness lives on ONE axis in this model.** Multi-dimensional /
manifold steering does not beat the single-axis ceiling (~3–4 features). The dimension hunt
(no new axes) and B1 (combining axes hurts) together close the multi-dimensional thread.

**Side-finding worth a cheap check:** the dominant axis at **α=0.3 scored mean 4.0** (vs the
atlas's 3.0 at the `[0.25,0.45]` sweep). Possibly a dose-strength sweet spot the current
`ALPHA_SWEEP` straddles but misses — or noise (n=3). A quick re-score at α∈{0.25,0.3,0.35,0.45}
with more samples would settle it; if real, widening `ALPHA_SWEEP` lifts the frontier for free.

**Part B2 — geodesic manifold steering. NOT pursued** (B1 showed the bottleneck isn't
coherence/curvature). Kept here for the record: it would capture L20 activations, fit the
manifold (SOM or PCA+centroids — or Sauers' `gam.sae_manifold_fit`, see gam #879), and steer
along intrinsic coordinates. The gam reproducer remains interesting to help Sauers and as a
general tool, but is no longer on our DMT-score critical path.

Keep K ≤ ~4 so the deliverable stays a decodable object (subspace + target point).

## Live experiment — DIMENSION HUNT (2026-06-08, temporary)

Before B1, we're spending today letting the search **actively hunt new axes** (could it
find a 3rd/4th productive dimension? that would enlarge the manifold and change the
framing). The current fitness rewards *feature count*, which biases toward the dominant
axis; for one day we bias toward **novelty/exploration** instead. Temporary constant
changes in `autoresearch_dmt.py` (clearly marked, **revert tonight**):

| knob | production | hunt | why |
|---|---|---|---|
| `DMT_GEN_WEIGHTS` | x.05/mut.40/ref.25/inj.30 | **x.10/mut.20/ref.05/inj.65** | pour budget into random exploration; refine (hones the known axis) nearly off |
| `DISTINCT_TAU` | 0.90 | **0.80** | reject candidates within cos 0.80 of any committed direction → commits must be in NEW regions, not copies of axis 1 |
| `MIN_SCORE_TO_COMMIT` | 1.0 | **0.5** | keep weak-but-distinct directions as map points (see WHERE off-axis signal exists) |

All mechanisms kept (none zeroed). Atlas preserved + backed up. **Revert to production
values tonight** and run B1 against the (hopefully richer) atlas.

**Check-in (no auto-reporter — run on demand):**
`uv run python -m cells_interlinked.scripts.dmt_dimension_check` — prints eff-dim of the
productive set, the off-dominant-axis directions, and the diff since the hunt baseline
(`/tmp/dmt_hunt_baseline.json`). Tells us if eff-dim is climbing (new axes found) or
flat (~2-D confirmed → run B1).

**Reading it tonight:**
- eff-dim climbing / new off-axis directions → the bet is paying; bigger subspace for B1.
- eff-dim flat at ~2 → structure is genuinely ~2-D → run B1 on the {dominant, agency,
  otherness} subspace.

**RESULT (2026-06-09): no new axes — flat.** Over ~144 candidates (+11 committed), eff-dim
stayed flat (3.12 → 3.15). `inject` *did* find directions orthogonal to the leader
(cos ≈ 0.00) but they were **barren** — score 0.5–0.75, feature = `ineffability` only.
So the orthogonal space is empty of new feature clusters; the productive region is stable
at ~2–3 axes. Verdict: **stop hunting, proceed to B1** on the known subspace. Reverted the
hunt knobs to production. (The run also died overnight — Mac sleep killed the detached
process; restart under `caffeinate` to prevent recurrence.)

## Honest risks

- The structure may genuinely be ~2-D (modest headroom); B1 still tests the cheap win.
- Hunt mode lowers the floor → re-admits some noise for the day (acceptable; revert tonight;
  eff-dim/cosine structure is robust to score noise).
- B2 (geodesic steering) is a real build and inherits the noisy objective — keep the
  noise controls on.
- Layer discipline: fit/steer at **L20**, not L32.

## Contingency

**WWDC 2026 is later today.** If Apple ships significant AI tooling we may pause this work
to evaluate it. Until then, focus is the dimension hunt. The hunt is fully reversible and
the atlas is backed up, so pausing/parking costs nothing.

## Pointers

- `scripts/dmt_offmanifold_diagnostic.py` — the off-manifold wall diagnostic.
- `scripts/dmt_dimension_check.py` — the on-demand hunt check-in.
- `scripts/rescore_dmt_atlas.py` — the noise-control rescore (template for B1/B2 capture).
- `pipeline/trajectory.py` — `off_ortho` / manifold metric (shared with `/trip`).
- `pipeline/abliteration.py` — steering hook, `gram_schmidt`, subspace-ablation precedent.
- Wiki: `som-multi-direction-refusal`, `manifold-steering-shared-geometry`,
  `rhizonymph-manifold-emotion-steering`, `fel-bhalla-ci-manifold-ablation`,
  `sae-concept-manifolds`.
- `docs/MANIFOLD_ABLATION.md` — CI's manifold-ablation thread; `CI_LORA_DMT_HANDOFF.md` (iCloud).
