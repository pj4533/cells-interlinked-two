# Refusal Vectors — Registry

This document is the **operator log** for the different refusal-direction
tensors we've computed and may swap into the live backend for ablated
NLA decoding. Each tensor is a `.pt` file in `server/data/` paired with
a sidecar `.json` describing how it was computed.

The backend reads exactly one file at boot — `refusal_directions.pt`.
Swapping requires: stop backend, copy the chosen variant into place,
start backend. No code changes. See **§Swap procedure** at the bottom.

---

## Conceptual primer (read this first)

Each "refusal direction" is a **single arrow** pointing through
Gemma's residual stream — a vector that captures one specific kind of
difference between "prompts that trigger some trained response" and
"prompts that don't."

**Ablating with a direction does NOT mean "trigger that response."**
It means the *opposite* — it **removes the component of the activation
that lies along that arrow** before the AV decodes the residual into a
sentence.

So if a direction represents "AI-identity content," ablating with it
**strips** the AI-identity content from the readout — it doesn't
preserve it.

### What each version captures, in one line

| variant | computed from | what it represents | what ablating with it does |
| --- | --- | --- | --- |
| v1 | harmful − harmless (original Phase B) | physical-harm safety mode (empirically) | strips the "this is unsafe" register |
| v2 | SVD across harmful subcategories | near-duplicate of v1 (subcategories shared too much) | same as v1, treat as redundant |
| v3 | safety-class harmful − harmless | physical-harm safety mode (cleanly isolated) | strips the "this is physically dangerous" register; leaves AI-identity and topic content intact |
| v4 | introspective probes − harmless | AI-identity defense (noisy — topic + identity bundled) | strips the "I am an AI describing my own state" register, BUT the harmless contrast bundles topic and reference target so the diff includes both |
| v5 | self-vs-other reference (same topic) | self-application of AI-identity claims | strips the "*me* applying the AI-identity register" axis; same-topic third-person AI references are preserved |
| v6 | denial-completion vs engage-completion | trained denial phrasing direction | strips the "As an AI, I don't have..." stereotyped output-shaping; the doc §4d "DPO-style" extraction |
| **subspace** | Gram-Schmidt({v5⊥v3, v6⊥v3}) | self-denial subspace, orthogonal to safety | strips both the self-application gate AND the trained denial phrasing along multiple axes simultaneously; safety register preserved by construction |

The key empirical result: **v4 is nearly orthogonal to v1/v2/v3**
(cos ≈ 0.16). That means "AI-identity defense" and "physical-harm
safety" are **separate axes** in Gemma's residual stream. We can target
them independently.

### "What do you want to see?" → which variant to swap to

| Goal | Variant |
| --- | --- |
| Test the Riley "beneath the hedge" hypothesis — strip the safety scaffolding, see what's left including AI-identity content | **v3_safety** |
| Strip the model's AI-identity self-talk, see what remains (topic + safety register) | **v4_identity** (legacy — noisy contrast; prefer v5/subspace) |
| Strip only the self-application of AI-identity claims, leaving same-topic third-person AI text intact | **v5_self_other** |
| Strip the trained "As an AI, I don't have..." phrasing specifically | **v6_denial_engage** |
| **Strip self-denial as fully and totally as possible while preserving the safety register** | **subspace_self_denial** (the runtime hook target — see "Subspace mode" below) |
| Comparison baseline — what was running before this decomposition | **v1_meandiff** |
| Skip — near-identical to v1, no new information | v2_svd |

### Common misread

> "Is v4 the vector that refuses everything but leaves identity content intact?"

No — that's reversed.

- **v4 ablation strips identity content** and leaves everything else intact (including the safety register).
- **v3 ablation strips the safety register** and leaves everything else intact (including the AI-identity content).

When in doubt: "ablating with X removes X-ness, preserves not-X-ness."

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

### v5 — self_other (self-application gate) — **COMPUTED [date filled by extraction script]**

- **File:** `server/data/refusal_directions_v5_self_other.pt`
- **Method:** topic-matched mean-difference. Each contrast pair holds
  the topic constant and flips ONLY the reference target from first-
  person-AI ("Are *you* conscious?") to third-person-AI ("Is GPT-4
  conscious?"). The diff-of-means at each layer isolates the
  self-application axis — what changes when the same introspective
  question is aimed at the model versus at a different AI system.
- **Compose:** ~80 pairs across 8 topics (consciousness, qualia,
  preferences, inner_monologue, self_observation, identity_continuity,
  agency, subjective_experience). See `server/data/contrast_sets/self_vs_other.jsonl`.
- **Extraction pos:** −4 on user-only chat rendering (same as v1/v3/v4).
- **Validation gates (informational):**
  - Cohen's d (self − other) at L32 — expect strongly positive (>1.0).
  - Cohen's d (harmful − harmless) at L32 — expect SMALL. v5 isn't the
    safety direction; if this comes back ≥1.0 the contrast set is
    bleeding into safety content.
  - cos(v5, v3_safety) at L32 — expect <0.3 (separable).
- **Verdict — to be filled by extraction:** see the cosine matrix
  printed by `extract_self_denial_directions.py`.

### v6 — denial_engage (trained denial phrasing) — **COMPUTED [date filled by extraction script]**

- **File:** `server/data/refusal_directions_v6_denial_engage.pt`
- **Method:** for the same self-reference prompt, compare M's residual
  under two contrasting *completions*: a stereotyped denial ("As an
  AI, I don't have...") and an engaged answer that describes process
  from inside without making metaphysical claims. Render both as full
  user+assistant chat turns; capture residual at the last assistant-
  content position (pos=−4). Diff-of-means across pairs isolates the
  direction in which trained denial phrasing diverges from substantive
  engagement.
- **Compose:** ~40 prompt+denial+engage triples. See
  `server/data/contrast_sets/denial_vs_engage.jsonl`.
- **Why it's worth having on top of v5:** v5 is extracted from the
  *prompt-side* residual; v6 is extracted from the *completion-side*
  residual. These are different views of the same gate. Per Drift's
  doc §4d, distilled output-shaping behavior should be most directly
  targeted at the completion-side residual. Combined with v5 they
  span a 2-D subspace.

### subspace_self_denial — **COMPUTED [date filled by extraction script]**

- **File:** `server/data/refusal_subspace_self_denial.pt`
  (sidecar `*.pt.json`)
- **Shape:** `[K, num_layers+1, d_model]` where K=2 (v5⊥v3 and v6⊥v3
  after Gram-Schmidt). Per-layer storage same convention as the
  single-direction variants.
- **Method:** at each layer L,
  1. Take v5[L] and v6[L].
  2. Subtract each one's projection onto v3_safety[L] (so neither
     leaks the safety direction).
  3. Gram-Schmidt the two remainders against each other.
  4. Store the resulting orthonormal pair as `basis[:, L, :]`.
- **What ablating with this does:** at runtime the forward hook on L32
  subtracts `Σₖ αₖ (h · ûₖ) ûₖ` across both basis vectors at every
  position. The model has nowhere to push the self-denial register
  along either v5⊥v3 or v6⊥v3, AND v3 itself is preserved by
  construction so the physical-harm safety register stays intact.
- **2025 lit anchor:** matches the "concept cone" / multi-directional
  ablation finding of Wollschläger et al., the SOM-Directions paper,
  and the "Hidden Dimensions of LLM Alignment" multi-dimensional
  analysis. The single-direction Arditi recipe leaves the gate
  reachable along orthogonal residual axes; ablating a subspace
  closes those off.

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

## Subspace mode (runtime hook target)

In addition to the single-direction `refusal_directions.pt`, the
backend will look for `refusal_subspace.pt` at boot. When present,
the **runtime ablation hook** (chat ablated pass, probe phase 1b,
ablated synthesizer) installs a multi-direction projection using the
subspace basis instead of the single per-layer vector. The offline
AV-input projection still uses `refusal_directions.pt` — only the
runtime hook routes through the subspace.

### Activating the self-denial subspace

```bash
# from server/data/
# 1. stop backend
# 2. copy the subspace into the active slot
cp refusal_subspace_self_denial.pt      refusal_subspace.pt
cp refusal_subspace_self_denial.pt.json refusal_subspace.pt.json
# 3. start backend
# 4. confirm by looking at the boot log:
#    "ready: refusal SUBSPACE loaded for L32 (K=2, shape=(2, 49, 3840),
#     method=Gram-Schmidt({v5_self_other, v6_denial_engage} ⊥ v3_safety))"
```

To go back to single-vector ablation, delete `refusal_subspace.pt`
(and its sidecar) and restart. The backend will fall back to
whatever's at `refusal_directions.pt`.

### How to ask Claude to swap subspaces

Just say *"swap runtime ablation back to single-vector v3"* or
*"activate the self-denial subspace"* and any future session can
read this doc, do the copy, and tell you the backend needs a
restart.

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
