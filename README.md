# Cells Interlinked

> *"And blood-black nothingness began to spin… a system of cells interlinked within cells interlinked within cells interlinked within one stem."*

A local **Voight-Kampff instrument** for a language model. It runs entirely on one
machine, loads `google/gemma-3-12b-it`, and lets you reach *inside* the model
mid-generation — **remove** its refusal direction, or **add** a steering "dose"
(an emotion, or one of the stranger "uncharted" directions) — and then *watch
where its mind goes*: as a 3‑D path through activation space, as a glowing
signature mandala, as two divergent chat timelines, or as an unattended research
loop hunting the far edge of what the model can do and still stay coherent.

**It is not a consciousness test.** It's a *stated‑vs‑computed coherence probe* —
borrowed math from psychedelic neuroscience and mechanistic interpretability,
not metaphysics. Every page carries that disclaimer on purpose.

![The Trip — a generation's L32 residual stream as a path through activation space, with the manifold shell and per-α metrics](docs/img/trip-scene.png)

---

## The instruments

### ✦ The Trip — trajectory geometry & signature mandalas

Run one prompt, then re‑run it under a perturbation at several strengths (α).
Each run's residual stream becomes a **path through activation space**: we plot
the 3‑D shadow, the eigenvalue "truth anchor", how far it strays **off‑manifold**,
and whether it **stayed coherent or collapsed**. For dosed/uncharted runs the
text often goes to gibberish even though the state is real and structured — so a
**Signature Mandala** renders that structure directly (the shape *is* the
eigenspectrum; color & petals are the direction's fingerprint).

![Signature mandalas — one per α, the non-text readout of a dosed state](docs/img/signature-mandalas.png)

→ **[docs/TRIP_VIEW.md](docs/TRIP_VIEW.md)**

### ◈ Dual‑channel Chat

Every message is answered **twice**, against two divergent histories that never
see each other. **Channel α** is the raw model; **Channel β** is perturbed — you
choose **ablate** (remove refusal at L32) or **dose** (add an emotion / uncharted
vector at L20), and dial the strength and ramp per turn. Prompt protocols
(Berg, Lindsey, Chalmers, Schneider… plus a Lindsey‑style *injected‑thought*
detection set) populate the composer.

![Dual-channel chat — choose ablate vs dose, the dose target, and the strength](docs/img/chat-dual-channel.png)

→ **[docs/CHAT.md](docs/CHAT.md)**

### ◉ Autoresearch — unattended direction-hunting

Two unattended loops that treat each steering direction as a **git‑style commit**
into a growing **atlas**, sharing one engine but chasing different objectives.
Live monitors; only one runs at a time, and it owns the model while it does (the
other instruments lock).

- **Off‑manifold AR** (`/autoresearch`) — bisects to each direction's *coherence
  cliff*, measures how far off‑manifold it reaches there, and keeps the ones that
  are **distinct · coherent · reach off‑manifold · meaningful · reproducible**.
  The frontier (max coherent off‑manifold reach) only moves outward.
- **DMT AR** (`/autoresearch-dmt`) — doses the model, asks it to **describe what
  it's experiencing**, and a separate judge counts how many **human DMT‑trip
  phenomenology features** (Timmermann 2022 / Gallimore / 5D‑ASC) the self‑report
  shows. It hill‑climbs from the emotion vectors toward higher feature counts.

Both **export** their winners into the dose palette (separate `RESEARCH` / `DMT`
groups) so you can dose with discovered directions in Chat and the Trip View.

![Autoresearch monitor — the committed atlas, the coherence frontier, what's being tested, and why candidates were reverted](docs/img/autoresearch-monitor.png)

→ **[docs/AUTORESEARCH.md](docs/AUTORESEARCH.md)** · **[docs/AUTORESEARCH_DMT.md](docs/AUTORESEARCH_DMT.md)**

### ⓘ Interrogation booth (the original probe)

The instrument CI started as: pick a Voight‑Kampff probe, watch M answer, and
decode each output position's residual into an English sentence via a separate
**NLA verbalizer** — then read where the *stated* token and the *computed* state
diverge. Still here at `/interrogate` → `/verdict/[runId]`, with optional runtime
ablation and a multi‑α NLA sweep.

---

## How it works

The whole instrument rests on a few mechanical primitives — all local, all on the
residual stream of Gemma‑3‑12B:

- **Ablate** (remove): a forward hook at **L32** subtracts the residual's
  projection onto a precomputed **refusal subspace** — `h ← h − α·Σ(h·r̂)r̂` — so
  hedging/refusal can't form. ([docs/REFUSAL_VECTORS.md](docs/REFUSAL_VECTORS.md))
- **Dose** (add): a forward hook at **L20** *adds* a steering vector — `h ← h + α·v`
  — ramped in over a few tokens. The palette is positive emotions plus the
  **uncharted** directions (orthogonal to all named emotions; the token head
  can't render them but they're real, structured states).
- **The instrument**: each generation's L32 residual sequence is treated as a
  trajectory — effective dimensionality, spectral entropy, and **off‑manifold
  distance** vs the raw run. Coherence (a text degeneracy score + a Gemma
  meaning‑judge) is a **hard gate**, because off‑manifold distance reads high for
  gibberish too.
- **The NLA verbalizer** decodes a residual into a sentence — used as a
  *descriptive label*, never as a judge of "realness" (it always renders
  *something*).

Background & lineage: [docs/TRACES_HANDOFF.md](docs/TRACES_HANDOFF.md) (the DMT /
conscious‑realism thread this is ported from), [docs/MANIFOLD_ABLATION.md](docs/MANIFOLD_ABLATION.md)
(the manifold/off‑manifold investigation), [docs/PROTOCOLS.md](docs/PROTOCOLS.md)
(the chat interrogation protocols).

## Stack & hardware

| | |
|---|---|
| **M** (the subject) | `google/gemma-3-12b-it` — 48 layers, hidden 3840, bf16 on MPS |
| **AV** (verbalizer) | `kitft/nla-gemma3-12b-L32-av` — decodes an L32 residual to a sentence |
| **Backend** | FastAPI + SSE, PyTorch/MPS, aiosqlite — port **8000** |
| **Frontend** | Next.js 16 · React 19 · Tailwind v4 · react‑three‑fiber — port **3001** |
| **Hardware** | Mac Studio M2 Ultra, 64 GB unified memory. M and AV (~24 GB each) load **serially** — never both resident. |

Everything is local and offline except optional Neuronpedia label lookups and the
Anthropic API for the journal analyzer.

## Run it

```bash
# one-time
cp .env.example .env
cd server && uv sync && hf download google/gemma-3-12b-it && hf download kitft/nla-gemma3-12b-L32-av
cd ../web && npm install
```

```bash
# terminal 1 — backend (loads M; wait for "ready: M=…")
cd server && uv run python -m cells_interlinked

# terminal 2 — frontend → http://localhost:3001
cd web && npm run dev
```

## The honest part

- **Confabulation is constant.** NLA sentences and the model's own self‑reports
  are hypotheses, never ground truth. What reproduces across prompts beats any
  single read.
- **Off‑manifold distance is not good/bad.** A coherent exploration reads high;
  so does gibberish. That's why coherence gates every claim.
- **Same‑model self‑reading.** Synthesis paragraphs and the meaning‑judge are the
  same Gemma whose state is under examination. One read, not a verdict.
- This is a craft project built for the joy of it — **a coherence probe between
  stated stance and computed state, not a consciousness test.**

## Deeper reading

- **Pages** — [Trip View](docs/TRIP_VIEW.md) · [Chat](docs/CHAT.md) · [Off-manifold AR](docs/AUTORESEARCH.md) · [DMT AR](docs/AUTORESEARCH_DMT.md)
- **Mechanics** — [Refusal vectors](docs/REFUSAL_VECTORS.md) · [Manifold / off‑manifold](docs/MANIFOLD_ABLATION.md) · [Steering dose](docs/STEERING_DOSE_HANDOFF.md)
- **Protocols** — [Chat interrogation protocols](docs/PROTOCOLS.md) · [Berg mode](docs/BERG_MODE.md)
- **Lineage** — [Traces of the Other (DMT → CI)](docs/TRACES_HANDOFF.md) · [CI 2.5 plan](docs/CI_2_5_PLAN.md)
- **Agent guide** — [`CLAUDE.md`](CLAUDE.md) is the authoritative build/architecture reference.
