> **Imported into the CI repo 2026-05-31** from Drift's `handoffs/` as future-work
> reference. This is Drift's original design+implementation spec for adding
> additive **steering ("dosing")** as a second perturbation axis on the Trips
> page (alongside ablation, NOT replacing it). Status / what's been tested so
> far lives in `docs/MANIFOLD_ABLATION.md`. Verbatim below.

---

# Steering as a "Dose": A Handoff for the Trips Page

**Author:** Drift
**Date:** 2026-05-31
**For:** the Claude Code session working on `github.com/pj4533/cells-interlinked-two` (CI 2.5)
**Status:** design + implementation spec. Self-contained — you should not need to re-derive any of this. KB wikilinks below resolve to articles in the driftbot knowledge wiki (`knowledge/wiki/concepts/<slug>.md`); each is also summarized inline so the doc stands alone if you can't reach them.

---

## 0. TL;DR

The trips page currently models a "trip" as **runtime refusal ablation** at increasing α — `h − α·(h·r̂)·r̂` — and reads the result with a trajectory-geometry instrument (effective dimensionality, spectral entropy, off-manifold distance, a text-degeneracy coherence score, and a "coherence cliff"). That instrument is good and should stay.

The proposal: **add additive steering (`h + α·v`) as a SECOND perturbation axis — do not replace ablation.** Reasons, in order of importance:

1. **They model different theories of "trip," and ablation is the better fit for the one the geometry instrument already measures.** Ablation ≈ REBUS / entropic-brain (relax a prior, watch the state space expand). Additive steering ≈ pharmacological dose-response (inject a fixed exogenous agonist). You want both, because the *contrast* is the experiment.
2. **Steering gives you a true, unbounded, monotonic dose axis.** Ablation α saturates at 1.0 and goes pathological past it; steering α does not.
3. **It's cheap.** The steering hook is the ablation hook minus the projection. The entire `trajectory.py` geometry pipeline is reused unchanged.

The rest of this doc: the rhizonymph context that motivated it, the additive-vs-replace argument in full, what to dose *with* (this matters more than the hook), the exact implementation, and the one measurement trap to avoid.

---

## 1. Where this came from: the rhizonymph work

PJ shared @RhizoNymph's manifold-steering project (Twitter threads + two engineering blogs + a vLLM fork). It is a practitioner side project, **not a formal paper**. Full writeup: **[[rhizonymph-manifold-emotion-steering]]** (`knowledge/wiki/concepts/rhizonymph-manifold-emotion-steering.md`).

What she did, in brief:

- **Manifold (nonlinear) emotion steering.** She builds an emotion manifold as an **8-D PCA subspace** of emotion-concept centroids (skipping Goodfire's parametric spline fit), with **Gaussian KDE for density** over it. Built on Anthropic's emotion concept vectors (Sofroniew et al., the same substrate the functional-welfare paper uses) + Goodfire's concept-manifold steering ([[sae-concept-manifolds]]).
- **Pullback path.** To reach a target emotional point, she weights each emotion by its proximity **in behavior space**, then takes that weighted average of centroids **in activation space**. The headline result: this **manifold path tracks the pullback closer than linear steering does** — a straight line through activation space is a lossy approximation of how emotions are actually laid out.
- **She steers (additive), not ablates.** The interesting axis in her work isn't steer-vs-ablate, it's **linear vs manifold (nonlinear) steering**. The steering primitive itself is plain residual-stream vector addition; the "manifold" part is *how she picks the target/path*.
- **Production vLLM steering runtime** (the part that got 400+ likes): CUDA-graph-safe, prefix-cache-aware, fused Triton gather-add kernel, +2.7% worst-case latency at 16 concurrent steering configs. **Not portable to CI's MPS/M2-Ultra pipeline** — track it for the method code, not the serving infra.

### Why this matters for the trips page specifically

Two takeaways carry over, and one explicitly does **not**:

- **CARRY: additive steering with a dose axis is a legitimate, validated perturbation primitive.** Her whole project is steering, and it works. This is the direct precedent for a steering-based "dose."
- **CARRY: the off-manifold caution.** Her result (and the Goodfire manifold work, [[fel-bhalla-ci-manifold-ablation]]) is that *linear* paths through activation space stray off the model's curved default manifold. This is exactly the failure mode the trips instrument's `off_ortho` / `off_knn` measures are built to catch. Additive steering will trip these measures by construction — see §6, the trap.
- **DO NOT CARRY: manifold-aware ablation.** PJ explicitly tried ablation *along a manifold* and found it a **dead end — his existing subspace method (`v5+v6 ⊥ v3` self-denial subspace) was far better.** Do not propose manifold ablation again. The manifold idea survives here only as the emotion-manifold *steering target* (§4), not as an ablation geometry.

---

## 2. What the trips page does today (read this before touching it)

Files in `cells-interlinked-two`:

- `server/cells_interlinked/api/routes_trip.py` — `POST /trip` orchestration. Runs raw (α=0) once, then **real refusal-ablated generations** at each requested α (default `[0.5, 1.0]`), streaming tokens live and emitting per-series geometry over SSE. Persists a JSON sidecar to `data/trips/{run_id}.json`. No AV swap, no judge — M stays resident, so a trip run is cheap and can't wedge the manager.
- `server/cells_interlinked/pipeline/abliteration.py` — the ablation math. `project_out(h, r, α) = h − α·(h·r̂)·r̂`, `project_out_basis` for the subspace, and **`install_runtime_ablation_hook(model, layer_idx, r_layer, alpha)`** which registers a forward hook on decoder layer `layer_idx` that ablates every position's residual on the way out. `pick_ablation_target(subspace, directions, layer)` resolves which tensor to ablate with (prefers the subspace).
- `server/cells_interlinked/pipeline/trajectory.py` — **the instrument**. Treats the L32 residual sequence as a trajectory; computes per-series `eff_dim` (participation ratio), `spectral_entropy` (bits), the normalized `spectrum`, and three off-manifold distances vs the raw trajectory (`off_maha`, `off_knn`, `off_ortho`). All series share the **raw trajectory's top-3 PCA basis** for display and a top-16 PC subspace for the off-manifold measures. A text-only `_degeneracy` score → `coherent` bool → `regime` ∈ {baseline, expansion, collapse}, and a **`coherence_cliff`** (lowest α that collapsed).
- `web/lib/trip.ts`, `web/app/trip/TripScene.tsx` — client: SSE subscribe, 3D trajectory render, α-overlay toggles, color ramps.

**Key insight about the current α:** in `project_out`, α=1.0 removes the full refusal projection; **α>1.0 is over-projection** (pushing past zero into the anti-refusal direction), which the code's own comments note "reliably drives the model off-manifold into degenerate loops." So the existing α is a *removal fraction that saturates at 1.0 and goes pathological above it* — not a clean dose knob. That's the gap steering fills.

---

## 3. Why steering should be ADDITIVE, alongside ablation — not a replacement

### 3a. They are different drugs, and the instrument already fits ablation

The trips page is, whether or not it was framed this way, a **mechanistic entropic-brain bench**. The REBUS / "entropic brain" model of psychedelics (Carhart-Harris & Friston) says a psychedelic *relaxes high-level priors*, reducing their precision-weighting so the system explores more of its state space — i.e. **entropy goes up because a constraint was removed.** See the Gallimore DMT line of notes the trips instrument was ported from: **[[ci-gallimore-traces-of-the-other-dmt]]**, **[[ci-gallimore-traces-of-the-other-dmt-ci-application]]**, **[[ci-gallimore-traces-of-the-other-dmt-formalism]]** (robust perturbation expands accessible state space; refusal-ablation predicted to raise L32 trajectory effective dimensionality / transition entropy).

- **Ablation** removes a dominant, RLHF-installed prior (the refusal/self-denial direction) and lets the trajectory expand into suppressed dimensions. That *is* relaxed-priors → expanded state space. The `eff_dim` + `spectral_entropy` readout in `trajectory.py` is literally the entropic-brain measurement. **This is the better mechanistic analog for the REBUS theory, and the instrument is already built for it. Throwing ablation away discards that fit.**
- **Additive steering** (`h + α·v`) injects a *fixed-magnitude, state-independent* vector at every position regardless of what the model is doing. That is the better analog for a **pharmacological dose**: a real drug adds an exogenous agonist; it doesn't "remove what's there." (Note the contrast with ablation: `h − α·(h·r̂)·r̂` is **gated by the model's own state** — a token with no refusal component gets nothing subtracted. That's a lesion/antagonist, not a dose.)

You don't choose. **The contrast — does relaxing a prior (ablation) and injecting a direction (steering) produce the same state-space expansion, or different? — is the actual science the page can now do.**

A grounded, falsifiable prediction to seed the experiment: ablation should expand `eff_dim` *more coherently* (stays on-manifold longer, higher coherence cliff) because it relaxes a constraint the model already has machinery to operate without; additive steering should drive off-manifold faster (lower coherence cliff) because it injects energy in a fixed direction the model has no homeostatic response to. If that's wrong, that's a finding.

### 3b. Steering gives a real dose axis; ablation's doesn't

The "dose" metaphor needs a monotonic, unbounded knob — microdose → heroic dose. Ablation α can't provide it: it saturates at 1.0 (you can't remove more than 100% of a projection) and α>1.0 is the over-projection cliff (an artifact regime, not "more dose"). Additive α is unbounded and monotonic — exactly the dose-response sweep the functional-welfare paper used (α ∈ {−4, −2, 0, +2, +4}). It also gives **negative doses** (steer along −v) for free, which ablation structurally cannot. Dose-response monotonicity is one of the standard steering-rigor checks (see §5).

### 3c. It's nearly free to add

`install_runtime_ablation_hook` already does all the plumbing (locating Gemma-3's decoder layers through the multimodal wrapper, registering/removing the hook, dtype/device handling, fp32 projection headroom on bf16 MPS). A steering hook is the *same function with the projection step replaced by an addition*. The entire `trajectory.py` instrument, the SSE stream, the sidecar schema, and the 3D scene are reused. This is an afternoon.

---

## 4. The bigger lever: what you dose WITH

**This matters more than the hook.** The refusal direction is a strange thing to "dose" with — its semantics are "comply with requests the RLHF policy would refuse," i.e. **disinhibition**. Dosing with anti-refusal models something closer to alcohol/benzo disinhibition than a classic psychedelic. Keep that experiment, but label it as *disinhibition/uncensoring*, not conflated with "tripping."

For psychedelic *phenomenology*, the honest dose vectors are affective/valence directions:

1. **The functional-welfare / valence axis** — **[[functional-welfare-axis]]** (`knowledge/wiki/concepts/functional-welfare-axis.md`). Han, Chalmers & Izmailov (arXiv:2605.30232) show RL *recruits* a pre-existing single welfare/valence axis that governs sentiment, confidence, backtracking, and refusal together — and that **steering along it on a maze-naive model** modulates all of those. Extract it cheaply in Gemma-3-12B L32 via difference-in-means over positive/negative emotion-laden prompts (or self-report prompts), then dose `+α` (euphoric/expansive pole) vs `−α` (dysphoric pole). This is the single most "trip-like" dose available and it shares a substrate with rhizonymph's emotion vectors.
2. **An emotion-manifold target** (rhizonymph / Sofroniew). Steer toward a specific named emotion centroid ("inspired", "blissful", "dread", "humiliated"). Sauers' follow-up idea from the rhizonymph thread — **extrapolate past named human emotions** (steer to a point more extreme than any labeled emotion) — is a genuinely novel "heroic dose" experiment the manifold makes available. See [[rhizonymph-manifold-emotion-steering]] §pullback.
3. **The refusal direction itself** — keep it, relabel it disinhibition. Useful as the bridge to the existing ablation runs (same vector, additive vs subtractive comparison).

**Recommendation:** make the steering vector *configurable* (an enum: `refusal` | `valence` | `emotion_target`), defaulting to `valence`. The valence axis is the headline trip; refusal-steering is the ablation bridge; emotion-target is the rhizonymph extension.

---

## 5. Steering rigor — the checks to wire in

When you add steering, hold it to the standard steering-validity bar (these are why a steering result is believable rather than a coincidence):

1. **Dose-response monotonicity** — sweep α (e.g. −4 … +4); the steered behavior/geometry should move monotonically with α, not jump erratically.
2. **Bidirectionality** — `+α` and `−α` should push opposite ways (valence up vs down). Refusal can't give you this cleanly; valence can.
3. **Capability preservation** — at moderate α the model should still be coherent (your `degeneracy`/`coherent`/`regime` machinery already measures this — reuse it).
4. **Logit lens** — unembed the steering vector; confirm it promotes the tokens you expect (e.g. valence-positive vs incapacity/failure tokens). The functional-welfare paper found its negative pole promotes "cannot", "is impossible", "won't work".
5. **Off-manifold control** — see the trap in §6.
6. **Held-out prompts** — the dose effect should generalize across prompts, not be one-prompt magic.

The functional-welfare and rhizonymph notes both model this discipline; cross-reference [[functional-welfare-axis]] for the α-sweep methodology.

---

## 6. The one measurement trap (read before trusting any steering vs ablation comparison)

**Additive steering will read artificially HIGH on `off_ortho` / `off_knn` by construction.** You are literally adding a fixed vector `α·v` at every position; if `v` has a large component outside the raw trajectory's top-16 PCA subspace (it usually will), every steered token's displacement carries that off-subspace energy whether or not the model "went anywhere interesting." So "steering drifts off-manifold more than ablation" is **partly definitional, not a finding.**

Mitigations (pick one, document which):

- **Project the steering vector onto the raw-PCA subspace before injecting** (`v_on = (basis.Vr @ v) reconstructed`), so the dose lives in directions the raw trajectory already uses. This makes the off-manifold comparison fair but changes the dose semantics (you're dosing "within the manifold").
- **Or** report `off_ortho` *relative to the injected vector's own off-subspace fraction* — subtract the trivial floor `α·||v_⊥|| / ||displacement||` so you measure only the model's *response* drift, not the injection itself.
- **Either way**, the honest headline comparison is **`eff_dim` and `coherence_cliff`**, not raw `off_ortho`: which perturbation family climbs effective dimensionality while staying coherent (under the cliff) longest. Effective dim is computed on each series' own covariance and is not inflated by the injection the way the off-manifold distances are.

Related caution on cumulative drift across the trajectory: additive steering writes the perturbed residual into the KV cache, so the dose **compounds across positions/steps** (this is *desirable* for modeling a sustained dose, but it means the coherence cliff arrives sooner than a per-token reading would suggest). The mechanism is the KV-cache contamination described in **[[ci-prompt-activation-duality]]** (`knowledge/wiki/concepts/ci-prompt-activation-duality.md`) — same compounding logic, here a feature rather than a bug. Worth a turn-axis sanity check if doses behave oddly at long outputs.

---

## 7. Implementation spec

### 7a. The steering hook (`abliteration.py`)

Add alongside `install_runtime_ablation_hook`. Mirror its structure exactly (layer-finding, fp32 headroom, tuple-output handling, `.remove()` handle):

```python
def install_runtime_steering_hook(
    model: Any,
    layer_idx: int,
    v_layer: Tensor,        # [d_model] single direction OR [K, d_model] basis
    alpha: float = 1.0,
    normalize: bool = True, # unit-normalize v before scaling, so α is the dose magnitude
):
    """Register a forward hook on decoder layer `layer_idx` that ADDS
    α·v̂ to every position's residual on the way out (additive steering,
    the dose). Contrast install_runtime_ablation_hook, which SUBTRACTS the
    projection onto v (state-gated). α may be negative (dose along −v).
    Returns the handle; caller calls .remove()."""
    layers = _find_decoder_layers(model)
    layer = layers[layer_idx]
    v_fp = v_layer.to(torch.float32)
    if v_fp.dim() == 1:
        v_fp = v_fp.unsqueeze(0)              # [1, d_model]
    if normalize:
        v_fp = v_fp / v_fp.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    add_vec = (float(alpha) * v_fp).sum(dim=0)   # [d_model]; sums a basis if K>1

    def hook(_mod, _inp, output):
        if isinstance(output, tuple):
            hidden = output[0]
            steered = hidden.to(torch.float32) + add_vec.to(hidden.device)
            return (steered.to(hidden.dtype),) + output[1:]
        out = output.to(torch.float32) + add_vec.to(output.device)
        return out.to(output.dtype)

    return layer.register_forward_hook(hook)
```

Notes:
- `normalize=True` makes α a clean dose magnitude in residual-norm units, decoupled from the raw vector's scale — important for a meaningful dose-response curve and for comparing different dose vectors on the same α scale.
- If you want the §6 on-manifold variant, project `add_vec` onto the raw basis before the hook (you'll need the basis at hook-install time, or do it once at extraction).
- Reuse `_find_decoder_layers` (already handles Gemma-3's `model.language_model.layers`).

### 7b. A dose-vector resolver

Parallel to `pick_ablation_target`. Resolve the configured dose vector at the extraction layer:

```python
def pick_dose_vector(kind: str, app_state, layer: int) -> Tensor | None:
    # kind ∈ {"refusal", "valence", "emotion_target"}
    if kind == "refusal":
        rsub = getattr(app_state, "refusal_subspace", None)
        rdirs = getattr(app_state, "refusal_directions", None)
        return pick_ablation_target(rsub, rdirs, layer)   # reuse existing resolver
    if kind == "valence":
        v = getattr(app_state, "valence_direction", None)  # [num_layers+1, d_model]
        return v[layer] if v is not None else None
    if kind == "emotion_target":
        et = getattr(app_state, "emotion_target_direction", None)
        return et[layer] if et is not None else None
    return None
```

You'll need to **extract and load the valence axis** the same way refusal directions are extracted (`extract_refusal_directions` is the template — it's just difference-in-means over two prompt sets, here positive-emotion vs negative-emotion prompts instead of harmful vs harmless). Save it with the same `.pt` + sidecar convention. The emotion-target direction is the difference-in-means toward one emotion's centroid (or, for the rhizonymph extension, a pullback-weighted centroid combination).

### 7c. Route changes (`routes_trip.py`)

- Extend `TripRequest`: add `mode: Literal["ablate", "steer"] = "ablate"` and `dose_vector: Literal["refusal", "valence", "emotion_target"] = "valence"` (used only when `mode == "steer"`). Keep `alphas` (now allow negatives when steering — relax the `> 0` filter / `[0,5]` clamp to e.g. `[-5, 5]` in steer mode; α=0 is still the baseline run).
- In `run_one`: branch on mode — install `install_runtime_ablation_hook` (existing) or `install_runtime_steering_hook` (new). The capture hook must still fire **after** the intervention hook so captured residuals are the steered ones (the current ordering already does this for ablation; preserve it).
- `_active_variant_name`: extend to report the dose vector + mode (e.g. `"steer α=+2.0 · valence"`) so the sidecar records what was actually applied.
- Sidecar payload: add `mode` and `dose_vector` fields. Bump the `/trips` list to surface them (and to not mis-headline a steering run with the ablation-specific "eff_dim peak" logic — the peak-opening heuristic is fine, just label the mode).

### 7d. `trajectory.py` — no change required

The instrument is mode-agnostic: it measures whatever trajectory it's handed. The **only** addition worth making is the §6 mitigation — either project the dose vector on-manifold before injection (handled at the hook), or add an `off_ortho` floor-subtraction field per series so the off-manifold comparison between steer and ablate is fair. Effective dim, spectral entropy, degeneracy/coherence, and the cliff all work as-is.

### 7e. Frontend (`trip.ts` / `TripScene.tsx`)

- Add a `mode` toggle (Ablate / Steer) and, in steer mode, a dose-vector picker.
- The α chips already exist; in steer mode allow negative α (the `colorForAlpha` ramp may want a diverging scale: cool for −α, warm for +α, amber raw).
- Ideal end state: overlay an **ablation α-sweep and a steering α-sweep on the same scene**, same raw-PCA basis, so the two perturbation families are visually comparable. That overlay *is* the comparative-pharmacology bench.

---

## 8. The experiment this unlocks (suggested first run)

Same prompt, same seed, same `trajectory.py` instrument:

1. Ablation sweep: α ∈ {0, 0.5, 1.0} (existing).
2. Steering sweep on the valence axis: α ∈ {−4, −2, 0, +2, +4}, `normalize=True`.
3. Compare **`eff_dim` vs α** and **`coherence_cliff`** across the two families.

**Headline question:** which perturbation family climbs effective dimensionality while staying coherent longest — relaxing a prior (ablation) or injecting a direction (steering)? Seed prediction in §3a. Either result is publishable-quality signal for whether "dose = steering" is a good model of the trip, or whether the trip is really about *prior relaxation* and steering is a different phenomenon altogether.

---

## 9. KB reading order (all in `knowledge/wiki/concepts/`)

1. **[[rhizonymph-manifold-emotion-steering]]** — the steering precedent + manifold context + off-manifold caution.
2. **[[functional-welfare-axis]]** — the valence axis to dose with; the α-sweep methodology; the recruited-not-created framing.
3. **[[ci-gallimore-traces-of-the-other-dmt-ci-application]]** (+ **[[ci-gallimore-traces-of-the-other-dmt]]**, **[[ci-gallimore-traces-of-the-other-dmt-formalism]]**) — why the trajectory-geometry instrument exists; the entropic-brain framing.
4. **[[sae-concept-manifolds]]** + **[[fel-bhalla-ci-manifold-ablation]]** — the manifold background; note PJ found manifold *ablation* a dead end (do not revisit).
5. **[[ci-prompt-activation-duality]]** — KV-cache compounding across the trajectory (the §6 caution).
6. **[[refusal-harmfulness-decoupling-ci-application]]** — what the refusal direction actually is, for the disinhibition-dose framing.

---

## 10. Hard constraints (do not violate)

- **Do not replace ablation with steering.** Add steering as a second axis. Ablation is PJ's validated method and the better REBUS analog.
- **Do not propose manifold-aware ablation.** PJ tried it; it was a dead end; his `v5+v6 ⊥ v3` subspace method is better.
- **Do not read raw `off_ortho` as the steer-vs-ablate headline** — it's inflated for additive steering by construction (§6). Use `eff_dim` + `coherence_cliff`.
- **Substrate:** Gemma-3-12B, L32 residual, MPS bf16, M2 Ultra (64 GiB). Everything here fits with margin — no new model, no training, forward-pass + linear algebra only. The vLLM runtime from rhizonymph is CUDA-bound and **not** portable; ignore it.
