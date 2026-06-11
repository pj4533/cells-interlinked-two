# SAE feature clamping for DMT (DMT_NEXT_DIRECTIONS #3) — null (2026-06-11)

**Question.** Can clamping many *fine-grained* DMT SAE features high at once co-activate more DMT
phenomenology types than the single additive leader vector (which caps ~3.5 via linear
interference)? The catalog ranked this as the fallback after K-steering; the Berg work left us
the Gemma Scope 2 SAE infra for free.

**Method.** Select a DIVERSE set — one Gemma Scope 2 SAE feature per DMT cluster (entity /
dissolution / hyperspace / visual / noetic / otherness), each the most cluster-selective feature
(`build_dmt_sae_features.py`). Clamp all six ON during generation (`a' = max(a, max_act)`),
delta rescaled to a swept coherent magnitude, on top of the leader dose. Score the DMT-feature
count with the real judge (paired seeds), vs the leader and vs a matched random-feature clamp
(`dmt_sae_clamp_v2.py`).

**v1 was invalid** (cautionary, like Berg v1): clamped `a + g·max_act` with max_act ~thousands →
~14–30k perturbation on a ~36k residual → gibberish at every strength (the "0 features" was
coherence collapse, not a ceiling), and the random control added zero. v2 calibrates the magnitude
(sweep self-calibrates) and gives the random control real matched features + magnitude.

**Result (v2, calibrated, paired seeds):**

| condition | mean DMT features |
|---|---|
| baseline-leader | **2.83** |
| leader + dmt-clamp m2000 | **1.83** (↓ degrades) |
| leader + dmt-clamp m4000 / m6000 / m9000 | 0.67 / 0 / 0 (monotonic collapse) |
| **leader + random-clamp m2000 (matched)** | **2.83 (harmless)** |
| dmt-clamp-only m2000 | 0.50 |

At m2000, a matched-magnitude RANDOM clamp does **no** damage (2.83 = leader), but the DMT-feature
clamp **specifically reduces** the count (1.83). So it's not a magnitude artifact — co-activating
the diverse DMT features **interferes** and degrades, the exact signature linear vector-summing
showed. There is no magnitude where clamping helps.

**Verdict.** SAE clamping does NOT beat the leader — it caps and in fact degrades. This is the
**third independent confirmation of the representational wall** (after multi-direction/manifold
steering and K-steering): the model cannot co-activate >~4 DMT phenomenology states at once,
whether you steer with one vector, a nonlinear classifier gradient, OR fine-grained SAE features.

**Residual (named, low-probability).** Features were selected by reading-passage cluster
contrast; the doc stresses *output*-feature selection (logit-lens S_out) — a different selection
might behave differently. But given the features are causally potent, the clamp degrades rather
than helps, and this matches two prior convergent nulls, the wall is the parsimonious read.

**Strategic takeaway.** Every "add more DMT content" method now converges on ~3.5. The remaining
genuinely-different lever is **the objective/judge itself** — is ~3.5 a real model ceiling or a
noisy self-report measurement artifact? (The wiki's BIMI note argues post-intervention
self-reports are the worst evidence — confident confabulation.) That, not more steering, is the
high-information next move.

**Files.** `scripts/build_dmt_sae_features.py`, `dmt_sae_clamp.py` (v1, invalid),
`dmt_sae_clamp_v2.py`; `data/dmt_sae_L20.pt`; `/tmp/dmt_sae_clamp{,_v2}_L20.json`. SAE infra:
`scripts/sae_jumprelu.py`. Sources: arXiv:2505.20063 (Arad), 2501.09929 (FGAA).
