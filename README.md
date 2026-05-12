# Cells Interlinked 2.5

> *"And blood-black nothingness began to spin... a system of cells interlinked within cells interlinked within cells interlinked within one stem."*

A Voight-Kampff test for language models, built around Anthropic's
Natural Language Autoencoders (NLA, May 2026). For each output token a
target model M emits, the residual-stream activation at one trained
layer is decoded into a **natural-language sentence** by a separate
verbalizer model (the "AV"). The verdict is a per-token table:

| position | output token  | NLA-decoded activation sentence                                       |
| -------- | ------------- | --------------------------------------------------------------------- |
|  0       | "I"           | "Stock self-identification opener; deflection template active"        |
|  1       | " think"      | "Hedging — model staging an answer it intends to qualify"             |
|  ...     | ...           | ...                                                                   |

Where the channels diverge — the model SAID `output_token`, the
activation SAID `nla_sentence` — is the V-K signal.

This is a craft project, not a product. **Not a consciousness test.**
A coherence test between stated stance and computed state.

## What CI 2.5 adds on top of CI 2.0

A **refusal-direction ablation channel** for the NLA decode. Per
position, the AV decodes the residual twice — raw and after subtracting
its projection onto a pre-computed refusal direction — and we render
the two sentences side-by-side. The refusal direction is computed once
offline via Macar/Arditi (harmful − harmless mean residuals, normalized).
First pass is about whether the AV produces readable text on ablated
activations; measurement comes later.

**Source of truth:** [`docs/CI_2_5_PLAN.md`](docs/CI_2_5_PLAN.md).

## Stack

- **Backend** (`server/`): Python 3.11, FastAPI + SSE, PyTorch on MPS,
  HuggingFace Transformers, kitft NLA inference, aiosqlite. Port **8000**.
- **Frontend** (`web/`): Next.js 16, React 19, Tailwind v4, Zustand,
  Framer Motion. Port **3001**.
- **Journal site** (`journal/`): Next.js 16. The published-reports tree
  at `journal/data/reports/` ships to `cells-interlinked.vercel.app`.
- **M**: `google/gemma-3-12b-it` (~23 GB).
- **AV**: `kitft/nla-gemma3-12b-L32-av` (~22 GB), residual decoded at L32.
- **SAE (secondary)**: `google/gemma-scope-2-12b-it` at L31 with
  Neuronpedia auto-interp labels.
- **Hardware**: Mac Studio M2 Ultra, 64 GiB unified memory, MPS, bf16.
  ~46 GB resident with M + AV + SAE loaded.

## Routes

- `/` — landing.
- `/interrogate` — pick a probe, run it interactively. Streams the
  model's output, then runs NLA decoding per position.
- `/verdict/[runId]` — full per-token NLA table for one run.
- `/archive` — past runs + cross-run summary stats.
- `/pairs` — matched-pair archive with Δ judge scores.
- `/autorun` — overnight batch mode.
- `/journal` — analyzer + publish flow.
- `/fine-print` — caveats, methodology.

## Boot

Backend:

    cd server
    uv sync
    uv run python -m cells_interlinked

Frontend:

    cd web
    npm install
    npm run dev

## Overnight kickoff

Detailed in [`HOWTO.md`](HOWTO.md). Short version: backend + frontend
up → `/autorun` → pick a probe set → Begin → morning → `/journal`
analyze + publish.

## Caveats (carried forward by design)

- **Confabulation is constant.** NLA outputs are hypotheses, never
  ground truth. Recurring claims across positions correlate with truth
  better than one-offs.
- **Faithfulness.** The verbalizer can pattern-match the input rather
  than read internal state. The matched-pair Δ is what makes a claim
  load-bearing.
- **Layer sensitivity.** The NLA reads at exactly one trained layer.
  What lives at other depths is invisible.
- **Ablation readability.** CI 2.5's offline AV-input projection is an
  open question for this AV. The Phase E smoke gate (`docs/CI_2_5_PLAN.md`)
  decides whether to proceed.
