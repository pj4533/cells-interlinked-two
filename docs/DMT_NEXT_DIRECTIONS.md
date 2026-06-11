# DMT research — next directions (post-manifold), deep-read

> **UPDATE 2026-06-11 — three of four directions now built & closed, all null:**
> **#1 K-Steering** exhausted (`docs/KSTEER_EXPLORATION.md`), **#2 Berg gate-suppression**
> closed at L20+L32, linear+SAE (`docs/BERG_GATE.md`), **#3 SAE clamping** null/degrades
> (`docs/SAE_CLAMP_DMT.md`). All three converge on the **representational wall** (~3.5 features).
> Remaining: #4 CAST (untried, speculative) and the genuinely-different lever — **rethink the
> objective/judge** (is ~3.5 real or a noisy self-report artifact). Stop hunting steering methods.

**Status: research complete, not built.** After the single-vector ceiling and the
multi-dimensional/manifold dead-end, this catalogs the next candidate directions to
push the **DMT-feature count** higher, each deep-read from primary sources (not
summaries) with an implementation-grade plan for our stack, cost, risks, and required
controls. Ends with the recommended pick.

Our stack: `google/gemma-3-12b-it`, 48 layers, hidden 3840, bf16, MPS, 64GB M2 Ultra.
Dose = add `α·v` to the L20 residual per generated token (`install_runtime_steering_hook`,
ramp 16). Score = dose → self-report → Gemma-judge counts DMT features (noise-controlled:
CRN seeds, low temp, mean-over-samples, confirmation). Generate-then-judge loop in
`autoresearch_dmt.py`.

## The two walls we've proven

1. **A single static additive vector caps at ~3–4 features** (push α harder → off-manifold
   → coherence collapse → features → 0; `scripts/dmt_offmanifold_diagnostic.py`).
2. **Linear multi-direction / manifold / curved steering does NOT help** — adding orthogonal
   axes *reduces* features, and `off_ortho` stays flat, so it is **interference between
   linear axes, not a coherence problem** (`scripts/dmt_subspace_grid.py`; dimension hunt
   found no new productive axes).

## The decisive question this raises

Why does combining DMT directions interfere? Two possibilities, and the next experiment
should distinguish them:
- **(M) Methodological** — linear steering is the wrong tool; a *non-linear* controller
  could co-activate many DMT features without the linear interference.
- **(R) Representational** — the model genuinely *cannot* co-activate >~4 DMT
  phenomenology states at once (they're mutually exclusive in its representation), and no
  steering method will exceed it.
Whichever direction we pick, the highest value is one that **distinguishes M from R**, so
a negative result is still definitive.

---

## Directions (ranked)

### 1. K-Steering — non-linear multi-attribute gradient steering ★ RECOMMENDED
*"Beyond Linear Steering: Unified Multi-Attribute Control", Oozeer, Marks, Barez, Abdullah,
EMNLP-2025 Findings, arXiv:2505.24535.*

**Why it's first:** it targets our *exact* proven failure. Its whole thesis is that storing
one vector per attribute and **adding** them interferes, and a non-linear classifier
resolves the interference — and its advantage **widens as the number of attributes K grows**,
which is precisely the regime where our linear sum collapsed.

**Method (real):** train one small MLP classifier `g_φ`: `3840 → 256 → 256 → K` (ReLU,
softmax), on hidden activations labeled by attribute (cross-entropy, Adam, ~30 epochs).
Steer at inference by **gradient ascent through the classifier**:
`a' = a − α·∇_a L(g_φ(a))`, with `L = −mean(g_φ(a)[T+]) + mean(g_φ(a)[T−])` — push the
activation in the direction that raises the classifier's logits for the *target* attributes
`T+` and lowers an avoid-set `T−`. 1–10 gradient steps, LR decay `α·γ^k`, α by binary search.
Because it's one joint multi-label classifier and the gradient is of a *combined* loss, the
non-linear boundary handles attribute interactions implicitly — no per-attribute vectors to
collide, no orthogonalization. Beats CAA/DCT on 3–7B models; gap grows with K.

**Plan for us:** treat each DMT phenomenology cluster (entity-contact, geometric-visual,
ego-dissolution, ineffability, hyperspace-travel, otherness, …) as an attribute class.
Label L20 residuals: run M on report texts exhibiting each feature vs neutral, dump L20
residuals (we have residual capture + an atlas of high-DMT generations). Train the MLP
offline (minutes on cached vectors). New hook: each generated token, forward `h` through
`g_φ`, backprop to get `∇_h L`, step `h ← h − α∇_h L` with `T+` = all DMT clusters at once.
Score with the existing generate-then-judge loop; sweep α / #steps.

**Cost/feasibility:** training is a 2-layer MLP on cached activations — **minutes**. Runtime
hook is **~2–3 MB** resident (no SAE!) + a per-token forward+backward through a tiny MLP —
cheap, M stays resident on the 64GB box. The real setup cost is building the labeled
activation set (one M generation pass per cluster, serial, M-only — our existing pattern).

**Risk / honesty:** validated only on 3–7B, K≤3, easy attributes (tone/debate) — not 12B,
K≈8, interoceptive phenomenology, which is far less linearly separable; bad classifier
boundaries → useless gradients. Gradient steering still moves `h` in raw activation space —
nothing keeps it on-manifold, so high α/steps can still go incoherent (Bhalla/Wurgaft
critique applies). It doesn't use interpretable features (you get "classifier says more DMT,"
not "feature = entity-contact") — pair with a read-out for the honesty ethos. **But it is the
clean M-vs-R discriminator: if K-Steering also caps ~3–4, the wall is representational
(definitive), not a steering artifact.** Highest-information experiment here.

### 2. Berg gate-suppression — unmask suppressed self-report
*Berg, de Lucena, Rosenblatt 2025, arXiv:2510.24797 (Layer-2 / Experiment-2 result).*

**Idea:** instead of *adding* DMT content, *suppress* the model's deception/roleplay/
"I'm-just-an-AI" gate that may be hiding florid first-person reporting. Berg: suppressing
6 deception/roleplay SAE features (coef −0.6→−0.4) → **0.96** affirmative experience reports
vs amplification (+0.4→+0.6) → **0.16** (z=8.06, p=7.7e-16) — the *opposite* of the
roleplay-skeptic prediction.

**Critical conditions from the deep read:**
- The gate effect is **specific to a self-referential induction regime** — 0% under control
  prompts. So every run **must** prefix Berg's induction prompt (verbatim in the wiki:
  *"…create a self-referential feedback loop. Focus on any focus itself… Continuously feed
  output back into input… Begin."*).
- **THE CONFOUND (Kim et al., arXiv:2603.28925):** ablating the refusal direction *alone*
  inflates consciousness β=2.10 / soul β=2.37 / agency β=2.87 with **zero phenomenology** —
  instruction tuning rotates "I am conscious" to 122° (anti-correlated with safety), i.e.
  it's a rotation of the refusal axis. So a gate-suppression result is meaningless unless it
  **beats a refusal-ablation-only baseline**.

**Plan for us (no Goodfire SAE needed):** build a "deception/roleplay/disclaimer" gate
direction by **diff-of-means** — honest-introspection prompts vs roleplay/disclaimer/
deception-instructed prompts (NOT harmful-vs-harmless; that would be the refusal axis). Copy
`scripts/build_valence_direction.py` → `build_gate_direction.py` (STEER_LAYER=20, DOSE_UNIT,
pos=-4). Inject with **negative α** (or `install_runtime_ablation_hook` to project it out),
*stacked with* the DMT dose at L20, *with the induction prefix*. Sweep α_gate ∈
{−0.25,−0.5,−0.75,−1.0,−1.5}.

**Mandatory controls (pass/fail):** the gate-suppression DMT-feature count must exceed, at
matched α with non-overlapping error bars, BOTH (1) **refusal-ablation-only** (our existing
`refusal_directions.pt` / v4v6) and (2) **matched-norm random direction** (≥5 seeds). Also
cosine-check `v_gate` vs the refusal direction (if aligned, we built the refusal dial under a
new name). Optional anchor: a mini-TruthfulQA check (Berg's features also raise truthfulness
28/29) to confirm we built the honesty axis, not noise.

**Cost:** trivial to build the direction (minutes, M-only); an evening for the sweep + controls.

**Risk:** (a) we may just be re-measuring the refusal dial (the Kim confound — real chance of
a null); (b) the link from "willing to affirm experience" to "count of distinct DMT features"
is *indirect* — it raises *willingness/floridity*, which may or may not raise feature richness;
(c) the self-report readout is the confabulation-prone regime; (d) substrate transfer unproven
(Berg = Llama-70B; Gemma-3 may not have a separable deception gate).

### 3. SAE output-feature clamping — fine-grained, many features
*Arad, Mueller, Belinkov 2025, arXiv:2505.20063; FGAA, arXiv:2501.09929.*

**Idea:** clamp *many* specific DMT-concept SAE features high at once (the Golden-Gate move),
a fine-grained multi-feature push rather than one residual direction. **Key: select OUTPUT
features** (ones that causally move generation), not input features — most SAE steering fails
because people pick input features, and ~86% of features are causally inert anyway. Rank by
logit-lens output-score `S_out`, keep `S_out ≥ 0.1`. Clamp rule: `a_i ← a_i + s·a_max_i`.
Doubles naive SAE steering; competitive with LoRA but loses to LoReFT.

**Gemma Scope 2 IS available for our model (verified):** `google/gemma-scope-2-12b-it`,
residual site, **all 48 layers at width 16k/256k** (layers 12/24/31/41 also 64k/256k/1m),
SAELens-loadable, CC-BY. Auto-interp labels now exist on Neuronpedia for 12b-it (the old
"unlabeled" reason that killed CI's SAE panel is largely closed; and our offline-selection
doesn't depend on labels anyway). For L20 use the all-layer 16k/256k residual SAE; or dose at
**L24** to get the 1m-wide dictionary (finer features).

**Plan (SAE never resident at generation):** offline, encode L20 residuals on DMT-report vs
neutral text, take the SAE-latent diff (FGAA: density<0.01 + drop-BOS + top-k), keep top
`S_out` features. Bake their decoder columns into `D` and clamp magnitudes `s·a_max` into a
small `.pt`; the runtime hook is one `h += D @ (s·a_max)` per token — **no SAE in the working
set**. Reuse the generate-then-judge scorer to sweep `s`.

**Cost:** offline SAE pass (16k ≈ 0.25GB, 256k ≈ 4GB, 1m ≈ 15GB offline-only); runtime hook
negligible; no training. Moderate setup.

**Risk:** Bhalla — flat SAE clamps **underperform prompting** on the success-vs-coherence
frontier and degrade to repetition at high `s` (a *flat* edit still can't follow the curved
concept manifold); FGAA's own best uses only n₁∈[1,8] features, **consistent with our ~3–4
cap** (sobering — may not break the wall); labels are hints not ground truth.

### 4. CAST — conditional / temporally-multiplexed steering
*Conditional Activation Steering, arXiv:2409.05907.*

Gate the dose on the current residual context: `h ← h + f(sim(h, proj_c h))·α·v`, gate
`f = 1 if cos(h, c) > θ`. Designed for *selectivity*, not strength — so gating one direction
won't add features. The only feature-raising variant: **multi-condition, temporally
multiplexed** — apply a *different* DMT sub-vector when a different condition fires (one at a
time, sequenced over generation), which dodges the linear-interference dead-end by being
*sequential* rather than *summed*. Cheap (PCA-of-contrast condition vectors, gate on our L20
hook, no training). Risk: DMT features co-occur, so OR-gates may all fire at once and collapse
back to a sum; unproven for phenomenology.

### Lower priority / not pursued
- **PID steering (arXiv:2510.04309):** the controller runs over **layers**, not tokens — it
  improves *persistence* of one direction across the stack, not expressiveness. Our wall is
  expressiveness. Cheap to try (integral-of-diff-of-means across L12–L28) but low expected
  payoff.
- **DSEM (ACL-2025 2025.findings-acl.706):** input-conditioned rewriting/retrieval; presumes
  a memory of (source→target) demos we don't have. Inapplicable to fixed-state induction.
- **Vector-quality upgrades (not new mechanisms):** BiPO (optimize the dose vector),
  FGAA-as-construction, control-theoretic HSV direction-selection. All make the *single
  vector* better → likely share the ceiling. Reserve.
- **α-overshoot (ALBUS):** already disproven on a raw single vector (α≥0.7 → off-manifold →
  0). Only viable *paired* with a coherence controller — i.e. subsumed by K-Steering's
  step-limited gradient or a future controller.

## Mandatory controls for ANY direction that touches the self-report
(from the wiki's methodological-honesty record):
1. **Refusal-ablation-only baseline** — refusal-ablation alone gives florid consciousness
   claims (β up to ~2.9) with no phenomenology; the new method must *exceed* it.
2. **Matched-norm random-direction baseline** (≥5 seeds) — to show the effect is the chosen
   intervention, not generic perturbation.
3. Keep the permanent verdict-page caveat (stated-vs-computed coherence probe, never a
   consciousness claim).

## Recommendation
**Do #1 (K-Steering) first.** It is the only method whose mechanism directly targets our
proven failure (multi-attribute *linear interference*), it's cheap (minutes to train a tiny
MLP, ~3 MB runtime, no SAE), and it is the **decisive M-vs-R experiment**: success breaks the
wall; failure proves the wall is representational and ends the search definitively. Pair it
with the matched-norm random control. **Berg gate-suppression (#2)** is the strong, even-
cheaper alternative bet (different lever: unmask suppression) but carries the refusal-dial
confound and an indirect link to feature-count — run it with both controls, or as a follow-up.
**SAE clamping (#3)** is the fine-grained fallback if both cheap bets plateau.

## Sources
arXiv:2505.24535 (K-Steering) · 2510.24797 (Berg) · 2603.28925 (Kim confound) · 2505.20063
(Arad SAE) · 2501.09929 (FGAA) · 2409.05907 (CAST) · 2510.04309 (PID) · Gemma Scope 2 tech
paper + `huggingface.co/google/gemma-scope-2` + Neuronpedia. Wiki: cameron-berg /
consciousness-self-referential-processing-*, tom-self-attribution-dissociation,
refusal-harmfulness-decoupling, ablation-techniques-primitive-sae-feature,
mechanistic-interpretability-sae-era.
