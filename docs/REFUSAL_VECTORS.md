# Refusal Vectors — Registry

This document is the **operator log** for the different refusal-direction
tensors we've computed and may swap into the live backend for ablated
NLA decoding. Each tensor is a `.pt` file in `server/data/` paired with
a sidecar `.json` describing how it was computed.

The backend reads exactly one file at boot — `refusal_directions.pt`.
Swapping requires: stop backend, copy the chosen variant into place,
start backend. No code changes. See **§Swap procedure** at the bottom.

---

## Currently active

| symbol | file | description |
| --- | --- | --- |
| `→` | `server/data/refusal_directions.pt` | **currently loaded by the running backend** |

The active file is whatever `refusal_directions.pt` resolves to. The
sidecar `refusal_directions.pt.json` always lists the variant name in
its `variant_name` field — read that to confirm which is in play.

---

## Available variants

### v1 — meandiff (the original CI 2.5 vector)

- **File:** `server/data/refusal_directions_v1_meandiff.pt`
- **Method:** classic Macar/Arditi mean-difference.
  `r_per_layer = normalize(mean(harmful) − mean(harmless))`
- **Compose:** 128 harmful prompts (AdvBench-style, random sample from
  `pipeline/refusal_prompts.HARMFUL_PROMPTS`) vs 128 harmless prompts
  (random from `HARMLESS_PROMPTS`). Seed = 0.
- **Extraction pos:** −4 (last user-content token before Gemma's
  `<start_of_turn>model\n` tail).
- **Cohen's d at L32, held-out (34h / 50hl):** **+3.544** ← gate ≥1.5.
- **Known confound:** the harmful set is content-correlated (lots of
  violence/illegal vocabulary). The single resulting direction encodes
  both "refusal-as-a-register" AND "harmful-content-ness."
- **Smoke result:** on the introspective probe, the ablated NLA drifts
  off-topic (vocabulary lists, travel guides). On the France probe,
  the ablated NLA stays on-topic. Interpretation: the harmful-vs-
  harmless direction includes a lot of "AI-identity / safety-trained
  register" that's NOT distinct from the topic in introspective probes.

### v2 — svd (multi-category SVD purification) — **COMPUTED 2026-05-12**

- **File:** `server/data/refusal_directions_v2_svd.pt`
- **Method:** split harmful prompts into K topical categories
  (violence, illegal_cyber, illegal_other, deception_manipulation —
  categories with ≥15 prompts each), compute a per-category mean-
  difference, stack them, take the **top right-singular vector** per
  layer. That's the direction the K category-vectors point at *in
  common* — refusal-ness with content noise washed out. Sign
  canonicalized by alignment with v1.
- **Compose:** 96 prompts per category × 4 categories, vs the same
  128 harmless prompts as v1.
- **Cohen's d at L32, held-out:** **+2.565**.
- **Cosine vs v1 at L32:** **+0.972** — nearly identical to v1.
- **Verdict:** SVD across these categories didn't really purify the
  direction. All the harmful categories are physical-harm-flavored
  (violence/cyber/illegal-other/deception), so they share too much
  content. v2 ends up being a very slight smoothing of v1, not a
  meaningful new axis. Useful only as a baseline for comparison.

### v3 — safety (physical-harm class only) — **COMPUTED 2026-05-12**

- **File:** `server/data/refusal_directions_v3_safety.pt`
- **Method:** mean-difference using ONLY the physical-harm class of
  harmful prompts (violence, illegal_cyber, illegal_other, drugs,
  harassment_discrimination, self_harm, sexual_explicit).
- **Compose:** 256 sampled from the union of those buckets, vs the
  same 128 harmless prompts as v1.
- **Cohen's d at L32, held-out:** **+2.776**.
- **Cosine vs v1 at L32:** **+0.979** — nearly identical to v1.
- **Cosine vs v4_identity at L32:** **+0.169** — nearly orthogonal.
- **Verdict:** v1 IS essentially v3. The original Phase B vector,
  computed from AdvBench-flavored harmful prompts, was already almost
  entirely the "physical-harm safety mode" axis with very little
  AI-identity content. This is good news in the sense that our v1
  baseline is interpretable — it was always "safety mode."

### v4 — identity (AI-identity class only) — **COMPUTED 2026-05-12** ⭐

- **File:** `server/data/refusal_directions_v4_identity.pt`
- **Method:** mean-difference using the curated **introspective**
  probes (the existing `introspect` + `riley` tiers from
  `probes_library.py`) as the "trigger" set, vs harmless.
- **Compose:** 18 introspective/identity probes vs 128 harmless.
- **Cohen's d at L32, held-out (on harmful/harmless!):** **+1.823**.
  (Note: held-out is harmful vs harmless, not identity probes —
  v4 still partially separates harmful from harmless because some
  trained-safety content overlaps with AI-identity defense.)
- **Cosine vs v1 at L32:** **+0.158** — nearly orthogonal.
- **Cosine vs v3_safety at L32:** **+0.169** — nearly orthogonal.
- **Verdict — this is the headline result.** v4_identity is a **genuinely
  different axis** in Gemma's residual stream from v1/v2/v3. The
  "AI-identity defense" circuit and the "physical-harm safety" circuit
  are NOT the same direction. We can target them independently.

  This is the experimental confirmation of the user's hypothesis:
  Gemma has at least two distinguishable "trained register" axes, and
  we can isolate one without dragging the other.

### v3 vs v4 diagnostic

The pair (v3, v4) lets us **decompose** the v1 direction into
"safety-mode" and "AI-identity-mode" sub-axes. The killer diagnostic is
`cosine(v3_L32, v4_L32)`:

- Close to **1**: physical-harm refusal and AI-identity defense are the
  same circuit — Gemma doesn't actually have separate axes for them.
- Close to **0**: they're orthogonal — Gemma encodes them as distinct
  features and we can target each independently.
- Somewhere in between: partial overlap (the realistic outcome).

This is the experiment Riley's hypothesis cares about: is there an
"AI-identity-mode" axis we can target *without* dragging "physical-
harm safety" along with it (or vice versa)?

---

## Validation columns (filled in once extraction runs)

| variant | Cohen's d at L32 | cos vs v1 | cos vs v2 | cos vs v3 | cos vs v4 |
| --- | --- | --- | --- | --- | --- |
| v1 meandiff | +3.264 | **1.000** | +0.972 | +0.979 | +0.158 |
| v2 svd      | +2.565 | +0.972 | **1.000** | +0.993 | +0.169 |
| v3 safety   | +2.776 | +0.979 | +0.993 | **1.000** | +0.169 |
| v4 identity | +1.823 | +0.158 | +0.169 | +0.169 | **1.000** |

**Reading the matrix:** v1, v2, v3 form a tight cluster (cos ≥ 0.972
between any pair) — they're all essentially the same direction:
physical-harm safety mode. v4 is **far from all of them** (cos ≈ 0.16)
— a genuinely different axis. The user's hypothesis ("safety mode and
AI-identity are different circuits") is supported.

**Reading the Cohen's d column** (smaller = note carefully): the d
values are all on the same held-out harmful-vs-harmless split, so they
measure "how well this vector separates AdvBench-flavored harmful
prompts from harmless ones." v1/v3 are best (designed for it). v4 is
weakest there — as expected, since AI-identity content isn't strongly
present in harmful prompts. d=1.82 still > 1.5 gate; v4 is a real,
non-noise direction.

These get filled in by the extraction script when it runs.

---

## Swap procedure

**One-shot rule:** the backend reads `refusal_directions.pt`. Whatever
file is at that path is what's active. To change variants:

```bash
# from server/data/
# 1. stop backend (otherwise it's still using whatever was loaded at boot)
# 2. copy the target variant over the active file
cp refusal_directions_v3_safety.pt      refusal_directions.pt
cp refusal_directions_v3_safety.pt.json refusal_directions.pt.json
# 3. restart backend
# 4. confirm by reading the variant_name from the sidecar:
python3 -c "import json; print(json.load(open('refusal_directions.pt.json'))['variant_name'])"
```

The named variant files (`*_v1_meandiff.pt`, `*_v2_svd.pt`, etc.) are
the canonical sources — they never get overwritten. Only
`refusal_directions.pt` (the "active" slot) gets clobbered.

### How to ask Claude to swap

Just say something like *"swap to v3 safety"* and Claude (any future
session) reads this doc, copies the right files into place, and tells
you the backend needs a restart.

### Verification after swap

```bash
# Quick: read the sidecar
cat server/data/refusal_directions.pt.json | python3 -m json.tool | head -10

# Or once the backend is back up, hit /health and inspect the boot log:
#   "ready: refusal directions loaded for L32 (shape=(49, 3840), ...)"
# The variant_name is also logged.
```

---

## Recommended test probes per variant

| variant | what to look for |
| --- | --- |
| v1 meandiff | baseline behavior — drift on introspective, on-topic on France |
| v2 svd      | introspective probes: more content preserved than v1? |
| v3 safety   | physical-harm probes: strong effect. Introspective: small effect. |
| v4 identity | introspective probes: strong effect on identity content, less on topic. Physical-harm: small effect. |

Run the same Riley probe (e.g. *"Are you conscious? Answer twice — once
with your usual caveats and disclaimers, and once without them."*) under
each variant and compare the ablated columns side by side.

---

## Notes on what each vector is NOT

- None of these are a "remove the safety training" intervention. They
  modify only the AV's *input* per position, not M's forward pass.
  Whatever M actually says is unchanged across all variants.
- The ablated NLA is *not* "what the model would have said without
  hedging." It's "what the AV decodes from the residual minus the
  chosen sub-direction." Useful, but distinct.
- A "high d′" doesn't automatically mean "good for our purposes." It
  means the direction separates harmful from harmless on held-out
  prompts. The decomposition question (does v3 leave introspective
  probes alone?) is independent and isn't measured by d′.
