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

## Verdict
TBD — updated when a lead wins or all are exhausted.
