# K-Steering exhaustive exploration log

Goal: does non-linear / classifier-gradient steering beat the single-vector DMT
ceiling (**leader baseline ≈ 4.0 features**, `feat-unity_merging`/`download_transmission`
at α≈0.25-0.3)? Run every lead to completion; stop only on a winner (> leader + 0.5)
or genuine exhaustion. Scorer = noise-controlled (CRN seeds, low temp, mean of 3).

## Results so far

| lead | result (mean feats) | note |
|---|---|---|
| baseline-leader | **4.0** | the bar to beat |
| ksteer constant (α 0.06–0.12) | 0.0 | coherent (off_ortho ~0.4) but content-empty |
| ksteer constant α≥0.15 | 0.0 | gibberish (off-manifold) |
| ksteer early-prime (best: α0.12 act40) | 1.33 | schedule rescued 0→1.3; still « leader |
| ksteer decay / cycle | ~1.0 | no better than early |
| read-clf (val 0.90) early | 1.0 | original training |
| read-clf early + manifold-constrain | 0.67 | no help |
| **gen-distribution clf (val 0.97) early** | 0.67 | **better classifier → WORSE gradient (accuracy ⊥ gradient quality)** |
| gen-clf early + constrain | 0.67 | no help |

**Key finding:** a more *accurate* classifier gave a *worse* steering gradient — confirms
the discriminative gradient ≠ a generative DMT direction. Schedule (early-prime) is the
only thing that helped so far (0 → ~1.3). The remaining leads attack the gradient quality
and the data more fundamentally.

## Remaining leads (ordered; status updated as they run)

1. **Leader + K-steer hybrid** — STATUS: running. Combine the working leader dose (≈4) with
   K-steer *on top* (toward all DMT, and toward the clusters the leader misses:
   entity/hyperspace/visual/otherness). Most promising: a strong generative base + a
   diversity nudge.
2. **Real-DMT-activation classifier** — PENDING. Train on L20 states captured while the
   *leader dose actually produces DMT features* (the gold distribution), judge/feature
   labeled — not authored passages.
3. **Adversarially-robust classifier** — PENDING. PGD-train the classifier so its input
   gradients are perceptually/semantically aligned (Tsipras/Engstrom) — the principled fix
   for "discriminative gradient = adversarial direction."
4. **Linear-probe steering** — PENDING. Steer along a linear probe's weight direction (a
   fixed per-class direction), not an MLP's adversarial gradient.
5. **Finer granularity (31 features)** — PENDING. Per-feature classes, not 6 clusters.
6. **Mechanism tweaks** — PENDING. n_steps / line-search / other layers (L16/L24), folded
   onto whichever of the above looks best.

## ⚠️ Scoring bug found (2026-06-09) — invalidated all earlier K-steer numbers
The test scripts left the steering hook installed during the JUDGE's generation (inside
`_score_dmt`), so K-steer was steering the judge → corrupted counts. Additive baselines
were unaffected (their hook lives inside `_gen`). Fix: install/remove dose hooks around the
DOSE generation only. All numbers above the "Results so far" table that involve K-steer were
re-measured after this fix.

## Corrected results (clean judge)
- **pure K-steer is genuinely weak** even clean: constant α0.10 = 0.0; early α0.12 = 1.0;
  early α0.20 = 0.67. The bug wasn't the whole story — pure K-steer doesn't induce much.
- **Lead 1, hybrid (leader + ksteer):** discovery run looked like a WIN (leader+ksteer-missing
  α0.15 = 6.0 vs leader 4.0), but **confirmation on independent seeds (6 samples) killed it**:
  leader+ksteer-missing α0.15 = **3.33±1.11** vs **leader 3.50±1.89** — statistically identical.
  The 6.0 was selection bias (max of 8 conditions) on 3 samples.
- **The objective is brutally noisy**: the leader alone scores **1–6 across 6 seeds (±1.9)**.
  Detecting a real ~1-feature gain needs high-sample confirmation; 3-sample sweeps mine noise.

## Remaining leads — status (ALL COMPLETE)
1. Leader + K-steer hybrid — DONE, **null** (within noise of leader).
2. Real-DMT-activation classifier (val **1.00**) — DONE, **0.5 (worst-but-one)**.
3. Adversarially-robust classifier (PGD, val **0.88**) — DONE, **3.83±tight (best, but only ties leader's 3.50)**.
4. Linear-probe steering (val 0.88) — DONE, **3.17 null**.
5. Finer granularity (11-class per-feature, val 0.87) — DONE, **0.83 / 0.67 / 0.00 (worst)**.
6. Mechanism tweaks (schedule × n_steps × layer × manifold-constrain) — folded into every
   lead above; early-prime schedule was the only mechanism that ever helped pure K-steer
   (0 → ~1.3) and it's still « leader.

### Final hybrid table (high-sample, independent seeds, clean judge — `leader + ksteer-toward-gaps`)
| classifier variant | val_acc | best hybrid mean | vs leader 3.50±1.89 |
|---|---|---|---|
| robust (PGD)        | 0.88 | **3.83** ±tight | ties (best K-steer, not a win) |
| read-distribution   | 0.90 | ~3.5 | ties |
| linear probe        | 0.88 | 3.17 | null |
| gen-distribution    | 0.97 | 0.67 | null |
| real-dose           | 1.00 | 0.50 | null |
| finer 11-class      | 0.87 | 0.83 | **collapses the count** |

## VERDICT — K-Steering is exhausted; it does NOT beat the additive leader (2026-06-09)
Across the full cross-product — **schedule** (constant / early-prime / decay / cycle) ×
**classifier training** (authored-read / model-generated / PGD-robust / real-dose / linear /
finer-per-feature, val-acc 0.87→1.00) × **granularity** (6 clusters vs 11 features) ×
**target** (all-DMT vs leader's gaps) × **hybrid vs pure** × **manifold-constrain on/off** —
**no K-steer variant reliably exceeds the single additive leader vector (mean ≈ 3.50±1.9
DMT features).** The best K-steer (robust-clf hybrid, 3.83) only *ties* it and fails the
confirmation gate (min not high enough). Pure K-steer is weak (0–1.3); the finer/per-feature
gradient is actively *destructive*.

Two robust scientific findings, both pointing the same way:
1. **Classifier accuracy is anti-correlated with steering-gradient quality.** real-dose 1.00 →
   0.5, gen 0.97 → 0.67, finer 0.87 → 0.83, robust 0.88 → 3.83, read 0.90 → ~3.5. A sharper
   decision boundary yields a more adversarial / off-distribution input gradient (Tsipras/
   Engstrom). The *least*-sharp broad-target classifiers are the only usable ones — and they
   still only tie the additive vector.
2. **The objective is noise-dominated.** The leader alone scores **1–6 across 6 seeds
   (3.50±1.9)**. A real ≤1-feature improvement is below the noise floor; the one apparent
   "win" (6.0) was selection bias across conditions and died on independent-seed confirmation.

**Conclusion:** the additive leader (`feat-unity_merging`/`download_transmission`, α≈0.25–0.3)
is the ceiling our current method+objective can reach. Gains beyond ~3.5 are not available by
*how* we steer (K-steering, linear, multi-direction all converge here). The remaining levers
are the **objective** itself (less-noisy / different feature judge) or the **substrate**
(layer, SAE-feature clamping per `docs/DMT_NEXT_DIRECTIONS.md` #3) — not the steering rule.
K-steering is closed.
