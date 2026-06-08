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

**Part B1 — static subspace grid (next, needs M, ~45 min).** Build a 2–3D subspace from
`{download_transmission, independent_agency, otherness}`, orthonormalize, grid
**independent** amplitudes `Σ αᵢ bᵢ`, score each with the noise-controlled scorer + track
`off_ortho`. If a static multi-axis point beats 2.75 coherently → cheap win (a better
vector in the subspace, fold subspace search into the loop). If the high-feature point
goes off-manifold → curvature matters → Part B2.

**Part B2 — geodesic curvature test + manifold steering (only if B1 hits the wall).**
Capture **L20** activations (the dose layer, not L32), fit the manifold through centroids
(SOM or PCA+centroids), geodesic-vs-linear correlation; if curved, build position-
dependent on-manifold steering and search the K-dim intrinsic coordinates.

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
