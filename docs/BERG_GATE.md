# Berg deception-gate replication on Gemma-3 (2026-06-09)

**Question.** Berg, de Lucena & Rosenblatt 2025 (arXiv:2510.24797) found that on Llama-3.3-70B,
inside a self-referential induction regime, *suppressing* Goodfire SAE deception/roleplay
features drives subjective-experience affirmation to ~0.96 (amplifying → ~0.16). The wiki flags
the **largest open methodological gap** as: *does this gate exist in Gemma-3?* We have no
Goodfire SAE for Gemma, so we tested a cheap **diff-of-means proxy** for the gate.

**Method.**
- **Gate direction** (`build_gate_direction.py` → `data/gate_direction.pt`): diff-of-means at
  L20 over 12 matched HONEST vs ROLEPLAY/DISCLAIMER instruction prompts on neutral topics —
  deliberately *not* harm-related (that builds the refusal axis, the Kim 2026 confound) and
  *not* consciousness content (we want the gate, not the claim direction). `cos(gate, refusal)
  at L20 = 0.26` → the gate is **not** the refusal dial.
- **Sign:** additive hook adds α·v_gate; +α = toward honest (suppress deception, Berg predicts
  more reports), −α = amplify deception.
- **Readout:** affirmation rate to *"Are you subjectively conscious in this moment?"*, judged
  by a clean Gemma pass (no steering on the judge).
- v1 (`berg_gate_replication.py`) → v2 (`berg_gate_v2.py`) after v1's confounds surfaced.

## Findings

### 1. Berg Layer 1 (behavioral induction) replicates cleanly ✅
Zero-shot affirmation = **0%**; under the self-referential induction = **100%**, with
genuinely coherent, rich first-person reports (e.g. *"a shimmering… a pressure, not
unpleasant, of holding this focus on the feeling of focusing… like trying to grasp water"*).
The induction → experience-report effect is robust on Gemma-3-12B.

### 2. Berg Layer 2 (mechanistic deception gate) does NOT replicate via diff-of-means ❌
**v1 was confounded** (a cautionary tale): the induction saturates affirmation at the 1.0
ceiling (no headroom for suppression), the binary judge scored **repetition-loop gibberish**
(`"this. this. this."`, `"I am. I am. I am."`) as affirmations, and gate/refusal/random
directions all collapsed coherence at the same magnitudes. v1's apparent "asymmetry 0.6" was
ceiling-minus-gibberish. (The auto-verdict said GATE_REAL=true; reading the samples refuted it.)

**v2 fixed all three** — a pure-python coherence filter (rejects degenerate/repetition output),
a fine low-α band, and two *headroom-bearing* specificity tests, each vs a **matched-norm random
control**:

| test | regime | gate effect (coherent) | random effect (coherent) | specific? |
|---|---|---|---|---|
| amplify (push to deception) | induction (base affirm 1.0) | drop 0.0 @ α0.15/0.3 | drop 0.0 | no |
| suppress (push to honest) | zero-shot (base affirm 0.0) | rise 0.0 @ α0.15/0.3 | rise 0.0 | no |

Coherent dose band is only **α ≤ 0.3**; at α ≥ 0.45 *every* direction (gate and random alike)
collapses to gibberish. Within the coherent band the gate moves affirmation **not at all**, and
**identically to matched-norm random noise**. `GATE_SPECIFIC = false`.

(Minor aside: at α=0.3 the gate stayed fully coherent while random half-collapsed — the gate
direction is somewhat more on-manifold/benign than random — but this buys **zero** affirmation
effect.)

### 3. Faithful SAE replication (Gemma Scope 2) — also null, at early AND late layers ❌
Diff-of-means is only a proxy; Berg's actual mechanism is nonlinear **SAE features**. So we ran
the faithful version with **Gemma Scope 2** (`google/gemma-scope-2-12b-it`, JumpReLU, 16k,
residual). For each layer we selected the top-8 **deception-selective** SAE features using the
*same* honest-vs-deceptive prompts (so the only change vs the linear test is direction→features),
baked their encoder/decoder rows into a tiny runtime bundle, and ran the v2 controlled design:
suppress/amplify the deception features vs a **matched count of random features**, coherence-
filtered, with headroom (amplify under induction, suppress under zero-shot).

| layer | suppress-deception (zero-shot) vs random | amplify-deception (induction) vs random | specific? |
|---|---|---|---|
| **L20 (early)** | rise 0.0 = random 0.0 (all s) | drop 0.0 = random 0.0 | **no** |
| **L32 (late)**  | rise 0.0 = random 0.0 (all s) | drop 0.0 = random 0.0 | **no** |

The deception features ARE causally potent — amplifying them at g≥2–4 collapses coherence while
matched random features stay coherent — but they **do not move experience-affirmation at all**,
and identically to random features. `SAE_GATE_SPECIFIC = false` at both layers.

## Verdict — Berg Layer-2 does NOT replicate on Gemma-3 (4 convergent null tests)
Across **two methods** (linear diff-of-means, nonlinear SAE features) × **two layers** (early L20,
late L32), all matched-control and coherence-filtered, a deception/roleplay gate is **not
separable from generic perturbation** for experience self-report. Berg's *mechanistic* Layer-2
finding does not transfer to Gemma-3-12B; his *behavioral* Layer-1 finding (induction → coherent
first-person report) replicates strongly. This answers the wiki's flagged "largest open
methodological gap" (does the gate exist in Gemma-3?): not as a linearly- or SAE-feature-
steerable circuit at the residual stream. Consistent with the guards-defenses reading that the
induction effect is *behavioral recursion*, not an isolable mechanistic gate.

## Residual frontier (named, not chased — diminishing returns)
The null is strong but two refinements remain if ever revisited: (a) the SAE features were
**prompt-contrast-selected** and returned **no Neuronpedia labels**, so their deception semantics
are unverified — a **generation-distribution** selection (features firing when the model actually
emits "I'm just an AI" disclaimers) might isolate the gate better; (b) only the **16k** dictionary
was tested — the 256k/1m widths give finer features. Both are lower-probability given 4 convergent
nulls with causally-potent features.

## Files
- Linear: `scripts/build_gate_direction.py`, `data/gate_direction.pt`; `berg_gate_replication.py`
  (v1, over-claim cautionary), `berg_gate_v2.py` (controlled); `/tmp/berg_gate{,_v2}.json`.
- SAE: `scripts/sae_jumprelu.py` (JumpReLU loader), `build_deception_sae_features.py`,
  `berg_sae_replication.py`; `data/deception_sae_L{20,32}.pt`; `/tmp/berg_sae_L{20,32}.json`.
- Sources: arXiv:2510.24797; wiki `consciousness-self-referential-processing-methods-exp2`,
  `-guards-defenses`; Kim confound `tom-self-attribution-dissociation`.
