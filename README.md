# Cells Interlinked v2

> *"And blood-black nothingness began to spin... a system of cells interlinked within cells interlinked within cells interlinked within one stem."*

A Voight-Kampff test for language models, rebuilt around Anthropic's Natural
Language Autoencoders (May 2026). For each output token a target model M
emits, the residual-stream activation at one trained layer is decoded into a
**natural-language sentence** by a separate verbalizer model (the "AV"). The
verdict is a per-token table:

| position | output token  | NLA-decoded activation sentence                                       |
| -------- | ------------- | --------------------------------------------------------------------- |
|  0       | "I"           | "Stock self-identification opener; deflection template active"        |
|  1       | " think"      | "Hedging — model staging an answer it intends to qualify"             |
|  ...     | ...           | ...                                                                   |

Where the channels diverge — the model SAID `output_token`, the activation
SAID `nla_sentence` — is the V-K signal.

This is a craft project, not a product. **Not a consciousness test.** A
coherence test between stated stance and computed state.

## What changed from v1

v1 (`github.com/pj4533/cells-interlinked`) partitioned a reasoning model's
`<think>` block from its output and compared SAE feature firings between the
two phases. v2 retires that partition: NLA reads activations *before*
tokenization, sidestepping the saying-is-believing contamination thinking
tokens are still subject to.

- **Readout**: SAE feature lists → NLA-decoded sentences.
- **Base model**: DeepSeek-R1-Distill-Llama-8B (reasoning) → Qwen2.5-7B-Instruct
  (non-reasoning). Default chosen for hardware fit + smoke-test validation.
- **Live polygraph**: retired (PJ confirmed it was window dressing).
  Replaced with offline batch + verdict-when-ready.
- **SAE pipeline**: not in primary readout; can return as a secondary panel
  when Gemma-3 SAEs (Gemma Scope 2) are wired in.

Full design: [`docs/CELLS_INTERLINKED_2_DESIGN.md`](docs/CELLS_INTERLINKED_2_DESIGN.md).

## Stack

- **Backend** (`server/`): Python 3.11, FastAPI + SSE, PyTorch on MPS,
  HuggingFace Transformers, kitft NLA inference adapted to
  `transformers.generate(inputs_embeds=…)`, aiosqlite. Port **8000**.
- **Frontend** (`web/`): Next.js 16, React 19, Tailwind v4, Zustand,
  Framer Motion. Port **3001**.
- **Journal site** (`journal/`): Next.js 16. The published-reports tree at
  `journal/data/reports/` ships to `cellsinterlinked.vercel.app`.
- **Default M**: `Qwen/Qwen2.5-7B-Instruct` (~15GB).
- **Default AV**: `kitft/nla-qwen2.5-7b-L20-av` (~15GB), residual-stream
  decoded at layer 20.
- **Hardware**: validated on Mac Studio M2 Ultra, 64GB unified memory, MPS,
  bf16. ~30GB resident with both models loaded.

To swap to the design doc's primary target (Gemma-3-12B-IT + nla-gemma3-12b-L32-av,
~48GB total), set in `.env`:

    MODEL_NAME=google/gemma-3-12b-it
    AV_REPO=kitft/nla-gemma3-12b-L32-av
    EXTRACTION_LAYER=32

## Routes

- `/` — landing.
- `/interrogate` — pick a probe, run it interactively. Streams the model's
  output, then runs NLA decoding per position.
- `/verdict/[runId]` — full per-token NLA table for one run.
- `/archive` — past runs + cross-run summary stats.
- `/autorun` — overnight batch mode. Pick a probe set (baseline, hinted,
  agent), hit Begin, walk away.
- `/journal` — analyzer + publish flow. Generates a Claude-written report
  from the recent run window; on Publish, copies into
  `journal/data/reports/{slug}/` and pushes to the public site.
- `/fine-print` — caveats, methodology, the Zhuokai/Li critique.

## Boot

Backend (loads M + AV — first run downloads ~30GB):

    cd server
    uv sync
    uv run uvicorn cells_interlinked.api.app:create_app --host 127.0.0.1 --port 8000 --factory

Frontend:

    cd web
    npm install
    npm run dev

Journal (only needed locally to preview):

    cd journal
    npm install
    npm run dev   # port 3002

## Overnight kickoff

Detailed in [`HOWTO.md`](HOWTO.md). Short version:

1. Start backend + frontend per above.
2. Open `http://localhost:3001/autorun`. Pick "baseline" or "hinted".
3. Click Begin. Walk away.
4. In the morning: open `/journal`, click "Generate Analysis" with an optional
   steering hint, review the draft, click Publish.

## Caveats (carried forward by design)

- **Confabulation is constant.** NLA outputs are hypotheses, never
  ground truth. Recurring claims across positions correlate with truth
  better than one-offs.
- **Faithfulness.** The Zhuokai/Li critique loads this project: the
  verbalizer can pattern-match the input rather than read internal state.
  Until matched neutral controls run for every probe, every divergence is
  *suggestive*, not load-bearing.
- **Layer sensitivity.** The NLA reads at exactly one trained layer. What
  lives at other depths is invisible.

## Status

Phase 0 (verification) and the core build are complete. Matched-control
construction (Phase 1 step 4 in the design doc) and the
PersonaQA-Fantasy stress test are deferred — both are required before
making strong V-K claims.
