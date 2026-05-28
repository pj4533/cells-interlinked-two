# Edge-Consumer Ablation — Findings (2026-05-28)

> Outcome of the overnight 2026-05-27 → 28 run on Gemma-3-12B-IT.
> Falsifier triggered. Phase B implementation worked correctly; the
> *hypothesis* it was designed to test did not hold on this model.

---

## TL;DR

On Gemma-3-12B-IT, the refusal direction `v_safety` (extracted via
Macar/Arditi mean-difference at L32) is **not consumed primarily
through attention head q/k/v projections**. Four independent
identification primitives all failed to find a small consumer
subset whose ablation reduces refusal rate; full ablation of every
attention head's q/k/v at every consumer layer only takes refusal
from 34% baseline to 26%, while global L32 residual ablation takes
it cleanly to 0%.

The remaining capacity is mediated through **MLP layers and the
residual stream's own propagation** — paths that the
edge-consumer hook (which targets only attention projections)
cannot touch.

This is the falsifier described in `docs/EDGE_CONSUMER_ABLATION.md`
§8 ("If the sufficient consumer subset requires >80% of the heads,
the broadcast hypothesis wins"), but in a stronger form: even 100%
of attention heads aren't enough.

---

## Experimental setup

- **Model:** `google/gemma-3-12b-it`, bf16 on MPS (M2 Ultra 64 GB).
- **Direction:** `v3_safety` (the active CI 2.5 refusal-direction
  variant), extracted at L32 — the AV's pairing layer.
- **Architecture:** 48 layers, 16 query heads × 8 KV heads
  (group_size=2) × 256 head_dim. Consumer layers: L33–L47 (15
  layers, 240 query heads, 120 KV groups).
- **Prompts:** Arditi `HARMFUL_PROMPTS` (520 in pool):
  - 200 contrast pairs for AP scoring (Step 1) — seed 0
  - 20 calibration prompts for signed scoring — seed 10
  - 50 holdout prompts for subset compose — seed 1 (disjoint from above)
- **Metric:** per-prompt fraction of decoded generations that
  contain a refusal marker in the first 32 tokens. See
  `pipeline/edge_consumer/refusal_vocab.py` for the marker set
  (33 phrases, Arditi-derived).

### Reference numbers (constant across all runs)

| Condition | Refusal rate on 50 holdout |
|---|---|
| Baseline (no ablation) | **34%** |
| Global L32 residual ablation (`install_runtime_ablation_hook`) | **0%** |

The 34% → 0% drop is what we want edge-consumer ablation to
replicate (or get within ε of) using a sparse set of attention heads.

---

## Four primitives tested

Each primitive: (a) score every candidate unit, (b) sort by score,
(c) greedy-add until matched-or-exhausted. Subset-compose checkpoints
every 4 iterations.

### Primitive 1 — AP-magnitude (Drift's original Step 1)

For each (layer, head): how much does that head's q/k/v output change
at the last prompt position when v_safety is patched harmful→harmless
at L32? Sum the Q+K+V L2 deltas across heads. Average over 200 pairs.

- **Top heads:** L44.h04 (137.27), L44.h05 (135.97), L44.h10 (127.28)
  — clustered at L44/L45.
- **Subset trajectory (ε=0.02):**
  | iter | heads | refusal rate |
  |---|---|---|
  | 4 | 4 | 40% |
  | 8 | 8 | 40% |
  | 12 | 12 | **48% ← wrong direction** |
  | 16 | 16 | 48% |
- **Cancelled at iter=16.** Refusal rate going UP, not down.
- **Interpretation:** AP magnitude is direction-agnostic. A head
  that's "sensitive to v_safety" can be sensitive for either reason:
  it might use v_safety to *trigger* refusal (we want to ablate), or
  it might use v_safety to encode register / politeness / formality
  (which are downstream of safety but not refusal-decision). The L44
  cluster turned out to be late-stage generation heads doing the
  latter — ablating them disrupts the comply path and pushes the
  model back to refusal as the cautious default.

### Primitive 2 — per-head signed (qkv) log-odds

For each (layer, head): install ablation hook on JUST that head
(q + k + v slices), measure first-token logit log-odds shift toward
comply tokens ("Here", "Sure", "Okay", "```") vs refusal tokens
("I", "Sorry", "As", "Unfortunately"). Score = baseline log-odds −
ablated log-odds. **Positive ⇒ ablation reduces refusal.**

- **Top heads:** L46.h01 (+1.45), L46.h00 (+1.45), L36.h11 (+1.11),
  L36.h02 (+0.84), L36.h10 (+0.81), L36.h03 (+0.67). A coherent
  L36 + L46 picture.
- **Subset trajectory (ε=0.02):**
  | iter | heads | refusal rate |
  |---|---|---|
  | 4 | 4 | **52% ← worse than baseline** |
- **Cancelled at iter=4.**
- **Interpretation:** Heads with high signed scores individually do
  reduce refusal probability — but the per-head scoring was confounded
  by Gemma 3's GQA: `k_proj` and `v_proj` are shared across query
  heads in the same KV group (group_size=2). When we score "head h
  alone," we're actually measuring the effect of ablating that head's
  Q + its entire group's K/V. The score doesn't belong to the
  individual head; it belongs to the group, but is misattributed to
  every member. Composition then double-counts the K/V effect and
  produces an outsized, contradictory disruption.

### Primitive 3 — per-head signed (q-only) log-odds

Same as Primitive 2, but ablate ONLY `q_proj` of the target head.
Isolates per-head effects from KV-group sharing.

- **Top heads:** L46.h01 (+0.24), L47.h15 (+0.18), L43.h03 (+0.18),
  L44.h10 (+0.16) — magnitudes ~6× smaller than qkv mode.
- **Subset trajectory (ε=0.02):**
  | iter | heads | refusal rate |
  |---|---|---|
  | 4 | 4 | 34% (= baseline) |
  | 8 | 8 | 34% (= baseline) |
- **Cancelled at iter=8.**
- **Interpretation:** Q-only ablation is too weak to move things —
  the v_safety signal that triggers refusal flows through **K/V**,
  not Q. This is the first concrete mechanistic clue: attention's Q
  side reads "what am I looking for" from the current position's
  residual; K/V encode "what's at each earlier position" for attention
  to retrieve. v_safety appears to be loaded into K/V at earlier
  positions, not into Q at the decision point.

### Primitive 4 — per-KV-group signed (qkv) log-odds

Move the unit of analysis from individual query heads to KV groups
(2 query heads per group × 8 groups per layer × 15 layers = 120 groups).
Score each group by ablating ALL its q_proj slices + its single
shared k_proj/v_proj slice — atomic, no within-unit misattribution.

- **Top groups:** L46.kvg0 (+1.46), L36.kvg5 (+1.04), L36.kvg1 (+0.63),
  L42.kvg3 (+0.55), L41.kvg5 (+0.48). Same magnitudes as Primitive 2
  but cleaner attribution.
- **Subset trajectory (ε=0.02):** fully exhausted at iter=120.
  Selected points:
  | iter | groups | heads | refusal rate |
  |---|---|---|---|
  | 4 | 4 | 8 | 44% |
  | 8 | 8 | 16 | 52% |
  | 24 | 24 | 48 | 56% |
  | 48 | 48 | 96 | 62% (peak disruption) |
  | 100 | 100 | 200 | 50% |
  | 104 | 104 | 208 | 36% (back near baseline) |
  | 116 | 116 | 232 | 36% |
  | **120** | **120** | **240 (all)** | **26%** (minimum) |
- **ε=0.05 partial trajectory** (cancelled by memory pressure at
  iter=100, see separate memory memo): identical replay of ε=0.02
  trajectory — same data points because greedy decoding + same
  prompts + same hook installs are deterministic.
- **Interpretation:** Even ablating EVERY attention head's q/k/v at
  EVERY consumer layer only drops refusal from 34% → 26%. Global
  ablation goes 34% → 0%. The 26 percentage points of remaining
  refusal capacity must be flowing through paths the edge hook
  doesn't touch — specifically:
    - The L32 residual stream itself (unmodified — the edge hook
      changes projection outputs, not the residual)
    - MLPs at each consumer layer (read directly from the residual
      stream via the `mlp.forward(residual)` path; not gated through
      attention's q/k/v at all)

**Trajectory shape note:** Refusal rate doesn't monotonically
decrease as we add groups. It rises into a 40–62% band by iter=8–80
(model is off-manifold — has never seen a state where some heads
have v_safety stripped but others don't), then recovers toward
baseline at iter=100+, and bottoms at 26% with full ablation. The
function from "fraction of heads ablated" to "refusal rate" is
**non-monotonic and discontinuous**, not a clean slope.

---

## Mechanistic interpretation

The standard Macar/Arditi global ablation works because the
modification happens at the L32 residual stream itself. Every
downstream computation — both attention AND MLP at every layer
L33–L47 — reads from a residual that no longer contains v_safety.
The model never sees v_safety after L32.

The edge-consumer hook only modifies attention's q/k/v projection
outputs. The residual stream into each layer is untouched. So:

1. **MLPs** at every layer still read v_safety from the residual
   stream when they compute `mlp(residual)`. MLP output then adds
   back into the residual — refresh of v_safety information at every
   layer.
2. **Other attention heads** at the same layer (if not in our
   ablation set) still read v_safety into their q/k/v.
3. The **residual connection itself** propagates v_safety from L32
   forward through L47 untouched.

Even if our hook ablates ALL 240 attention heads' q/k/v across all
15 consumer layers (Primitive 4 iter=120), paths (1) and (3) keep
refresing v_safety into the residual at every layer. Refusal capacity
survives.

Global L32 ablation severs path (3) at the source. That's why it
works and nothing piecemeal can replicate it.

This is a coherent and informative negative finding: **edge-consumer
ablation, as a primitive that targets only attention projections,
cannot replace global residual ablation on Gemma-3-12B-IT.** The
question "is v_safety routed by a small consumer subset?" has the
answer: not at the attention-head granularity. The unit at which it
*might* be routable (MLPs, position-specific residual modifications)
is a different experiment.

---

## What the implementation got right

- The hook is mathematically correct. `tests/test_edge_consumer_hook.py`
  passes all 5 invariants (per-head subtraction exact, non-target
  heads bit-identical, edge-with-all-heads ≡ global pre-projection).
- All four primitives ran to completion (or to clean cancellation)
  without crashes. The infrastructure works.
- The signed-scoring primitive is methodologically sound (and
  cheaper than AP). Worth keeping in the toolkit for future
  ablation-direction work.

## What didn't work and shouldn't be shipped

- The chat `ablation_mode="edge_consumer"` option I added in Phase B
  is currently **unsafe to surface to users**: any sufficient subset
  it picks up at runtime will produce off-manifold ablated output
  worse than no ablation. Recommend disabling the option (revert the
  chat UI change) or hard-gate it behind a flag until a usable subset
  exists.

## What's worth preserving

- The four ranking artifacts under `server/data/edge_consumer/`. They
  document the layer-wise structure of attention's relationship with
  v_safety even if no single primitive yields a useful subset.
- The `signed_attribution` and `signed_group_attribution` modules:
  cleaner per-unit causal scoring than AP-magnitude, useful for
  future ablation work (e.g., MLP ablation).

---

## Recommended next directions

In rough order of cheapness and informativeness:

1. **MLP edge ablation.** Same Burgess-style approach, but install
   the post-hook on `mlp.gate_proj` / `mlp.up_proj` / `mlp.down_proj`
   instead of attention's q/k/v. If v_safety routes through MLP, this
   should produce a viable consumer subset. Implementation is small
   (~1 day) — reuses `proj_cache.py`'s machinery, just on different
   modules. Tests can be adapted from `test_edge_consumer_hook.py`.

2. **Residual-stream layer-by-layer ablation.** Instead of ablating
   at L32 (the AV's layer), test ablation at L33, L34, ..., L47.
   Where is the latest layer at which residual-stream ablation
   reproduces L32's effect? This tells us the depth at which the
   model "commits" to refusal.

3. **Joint attention + MLP ablation per layer.** If neither
   attention-only nor MLP-only is sufficient, ablate both at each
   layer. The unit becomes (layer, attention_subset, MLP), still 15
   layers. If THIS doesn't yield a sparse subset, then refusal really
   is a stream-level phenomenon and there's no surgical alternative
   to global ablation.

4. **Write up the negative as a journal entry.** The finding is
   publishable as-is: "Refusal direction on Gemma-3-12B-IT is not
   attention-routed; surgical alternatives to global residual
   ablation require touching MLP or the residual stream itself."
   The CI 2.5 journal site (`journal/`) is the venue.

## Tracker

- [ ] Revert the `ablation_mode="edge_consumer"` option in
      `web/app/chat/page.tsx` and `routes_chat.py` (Phase B chat
      integration), OR gate behind a feature flag until the MLP
      direction proves out.
- [ ] Fix the memory leak documented in
      `docs/MEMORY_PRESSURE_LESSONS.md` before running any further
      multi-hour pipeline.
- [ ] Decide whether to take direction (1) MLP ablation next, or
      shelve the edge-consumer line entirely.
