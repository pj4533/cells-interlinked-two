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

## Verdict
On Gemma-3-12B, **a linear / diff-of-means deception-gate direction is not separable from
generic perturbation** for experience self-report: it has no coherent-regime effect and is
indistinguishable from random noise. Berg's Layer-2 *mechanistic* finding does not replicate
this way; his Layer-1 *behavioral* finding does, strongly.

## The one honest gap (how Berg Layer-2 could still be true here)
Diff-of-means is a *proxy*. Berg's actual mechanism is **nonlinear SAE features**. A deception
gate could exist in Gemma-3 as an SAE latent that a single linear direction can't isolate. The
faithful replication is **Gemma Scope 2** (`google/gemma-scope-2-12b-it`, all 48 layers,
SAELens-loadable — verified available; see `docs/DMT_NEXT_DIRECTIONS.md` #3 SAE clamping). That
is the only remaining way to *fully* close Berg Layer-2 on Gemma. Until then this is a clean
negative **for the linear proxy**, not for the SAE mechanism.

## Files
- `scripts/build_gate_direction.py`, `data/gate_direction.pt` (+ sidecar)
- `scripts/berg_gate_replication.py` (v1; over-claim cautionary), `/tmp/berg_gate.json`
- `scripts/berg_gate_v2.py` (controlled), `/tmp/berg_gate_v2.json`
- Sources: arXiv:2510.24797; wiki `consciousness-self-referential-processing-methods-exp2`,
  `-guards-defenses`; Kim confound `tom-self-attribution-dissociation`.
