# Cells Interlinked 2.5 — agent guide

> **Authoritative plan: [`docs/CI_2_5_PLAN.md`](docs/CI_2_5_PLAN.md).**
> If anything in this file contradicts that one, the plan doc wins.

A local Voight-Kampff interrogation interface. CI 2.5 extends the CI 2.0
instrument with a refusal-direction ablation channel: for every output
position, the AV decodes the residual twice — raw and with its
projection onto a pre-computed refusal direction subtracted — and shows
both side-by-side. The first pass is about readability, not measurement.

---

## Project ethos (do not violate)

- **Craft over feature count.** Built for the joy of it, not as a product
  or paper. Default to *less* surface area. When tempted to add a panel
  / extra experiment / comparison view, ask first.
- **Methodological honesty is non-negotiable.** Every verdict page
  carries permanent visible disclaimers. Never let the UI over-claim
  what a channel divergence means. This is a stated-vs-computed
  coherence probe, not a consciousness test.
- **Quiet mastery aesthetic.** The probe data is the point, not spinner
  animations.
- **Backend restart discipline.** Every Python change requires a
  restart before the running process sees it. Multiple past incidents
  trace to code on disk diverging from code in memory.

---

## Hardware + environment

- **Mac Studio M2 Ultra, 64GB unified memory.** All work is local and
  offline. No cloud calls except (a) Neuronpedia label lookups (cached
  locally; no telemetry sent) and (b) Anthropic API for the journal
  analyzer when invoked.
- **MPS backend, bf16.** `bitsandbytes` is CUDA-only and will not run
  here.
- **Disk + memory pressure are real.** Model weights (~46 GB for M + AV
  combined) plus working memory crowd 64 GiB. Overnight autoruns have
  generated up to ~40 GiB of macOS swap on disk. **Run compute jobs
  serially:** stop the backend before a script that loads M, run the
  script, restart the backend. Don't stack residents.
- **Port 3001** for the interactive web UI (port 3000 is taken by the
  user's Drift Docker container). **Port 8000** for the FastAPI backend.

---

## Stack (locked for CI 2.5)

| Piece | Choice | Notes |
| --- | --- | --- |
| M | `google/gemma-3-12b-it` | 48 layers, hidden 3840. bf16 on MPS. |
| AV | `kitft/nla-gemma3-12b-L32-av` | Decodes M's L32 residual into a natural-language sentence. |
| SAE | `google/gemma-scope-2-12b-it`, subdir `resid_post/layer_31_width_16k_l0_small` | Secondary panel. L31, not L32 — Neuronpedia auto-interp labels exist only at the four canonical layers (12/24/31/41). Adjacent to AV's L32. |
| Refusal direction | Computed offline via Macar/Arditi technique from `pipeline/refusal_prompts.py` (harmful + harmless prompts). Saved to `server/data/refusal_directions.pt`. | NEW for CI 2.5. |
| Judge | Gemma-as-judge via yes/no token logits. Eval-suspicion + introspection. Runs on raw NLA only. | `pipeline/judge.py`. |
| Backend | FastAPI + SSE on port 8000 | One-run-at-a-time `asyncio.Lock`. Custom autoregressive loop on `model.forward(use_cache=True)`. |
| Frontend | Next.js 16 + React 19 + Tailwind v4 + Zustand + Framer Motion | Port 3001. |
| Journal site | Separate Next.js app in `journal/`. Deployed to Vercel project `cells-interlinked`. | Reads `journal/data/reports/{slug}/{report.json,body.md}` from the filesystem at build time. |
| Persistence | SQLite via `aiosqlite` | Tables: `probes`, `analyses`, `feature_labels`, `autorun_state`. |

**Next.js note:** the version is 16.2.4 — newer than most training data.
Read the relevant guide in `web/node_modules/next/dist/docs/` before
writing frontend code. See `web/AGENTS.md`.

---

## How to run

Two terminals.

```bash
# Terminal 1 — backend
cd server
uv run python -m cells_interlinked
# Wait for: "ready: M=google/gemma-3-12b-it ... AV=kitft/nla-gemma3-12b-L32-av"
# Health check: curl http://localhost:8000/health

# Terminal 2 — frontend
cd web
npm run dev
# Open http://localhost:3001
```

First-time setup (one-time):

```bash
cp .env.example .env
cd server && uv sync
hf download google/gemma-3-12b-it
hf download kitft/nla-gemma3-12b-L32-av
hf download google/gemma-scope-2-12b-it --include "resid_post/layer_31_width_16k_l0_small/*"
cd ../web && npm install
```

For CI 2.5 work specifically, also compute the refusal direction (see
`docs/CI_2_5_PLAN.md` Phase B). Stop the backend first.

---

## Critical implementation invariants

These exist for hard-won reasons. Don't undo without thinking.

1. **Gemma-3-12B-IT is the deployed M.** Every operational signal — UI
   labels, the SAE secondary panel, the kitft AV pairing,
   `e2e/v2-gemma-sae.mjs` — assumes Gemma. CI 2.5 makes Gemma the
   `config.py` default explicitly.
2. **Custom autoregressive loop.** `pipeline/generation_loop.py` calls
   `model.forward(input_ids, past_key_values=kv, use_cache=True)`
   step-by-step. Forward hooks at the AV's extraction layer capture
   the last-position residual each step. We do NOT use
   `model.generate()` (no per-step emission control).
3. **SSE event protocol** is a discriminated union (see `web/lib/types.ts`
   and `server/cells_interlinked/api/routes_probe.py`). Keep both ends
   in sync; the frontend types file mirrors the backend dataclasses.
4. **One-run-at-a-time.** `RunRegistry` holds an `asyncio.Lock`; only
   one probe runs through M at a time. The model + AV together are too
   large to swap.
5. **NLA decode happens at L32.** The AV is paired to this layer.
   Changing the layer means a different AV. CI 2.5's refusal-direction
   projection is locked to L32 for the same reason — that's where the
   AV reads.
6. **Use the raw Rust tokenizer for encode/decode, NOT the transformers
   wrapper.** `transformers==5.7.0+` wraps the Rust BPE tokenizer in a
   way that's broken for this Gemma config. We load the raw
   `tokenizers.Tokenizer` from `tokenizer.json` separately as
   `bundle.raw_tokenizer`. The transformers wrapper is kept ONLY for
   `apply_chat_template` (Jinja templating).
7. **Gemma's multimodal wrapper.** `Gemma3ForConditionalGeneration`
   nests its decoder layers under `.model.language_model.layers`.
   Forward-hook installers and any future runtime-ablation path must
   traverse correctly.
8. **Caveats panel is always visible** on the verdict page — not behind
   a toggle. Same for `/fine-print`.
9. **Feature labels come from Neuronpedia.** After each probe, the
   backend collects every `(layer, feature_id)` referenced in the
   verdict and asynchronously fetches `description` from
   `https://www.neuronpedia.org/api/feature/{model_id}/31-gemmascope-2-res-16k/{feature_id}`.
   Cached in the `feature_labels` SQLite table. Empty string = no label.
10. **Journal `<cite>` tag stripping.** Anthropic's server-side
    `web_search` wraps any text derived from a search result in
    `<cite index="...">...</cite>` tags. `analyzer.py` strips these
    from title / summary / body_markdown before persistence so they
    don't leak into the published journal.
11. **Journal metadata shape.** The deployed `journal/` Next.js template
    is v1-era and reads `metadata.summary_stats.{total_runs,
    autorun_runs, manual_runs, proposer_runs}` plus empty
    `top_thinking_only` / `top_output_only` arrays. The analyzer emits
    this shape; v2-specific data lives under `v2_*` keys. Vercel
    prerender will crash if `summary_stats` is undefined.
12. **Vercel deploy from repo root.** The `cells-interlinked` Vercel
    project has `rootDirectory=journal` set in the dashboard. Running
    `vercel deploy` from `journal/` fails with "path doesn't exist"
    because it appends `journal/journal`. Always from repo root.

---

## What's in scope for CI 2.5 (the only thing we're shipping)

See `docs/CI_2_5_PLAN.md` for the full phase plan. Briefly:

1. Build `pipeline/abliteration.py` with `extract_refusal_directions`,
   `save_directions` / `load_directions`, `project_out`.
2. Compute the refusal direction for Gemma (Macar/Arditi, harmful −
   harmless, normalize per layer).
3. Verify via Cohen's d ≥ 1.5 at L32 on held-out prompts.
4. Wire `include_ablated_decode` flag end-to-end. AV decodes raw +
   ablated residual per position.
5. Readability smoke gate on 3 baseline probes. Fall back to partial
   projection `α=0.5` if AV collapses.
6. Side-by-side NLA columns on verdict page.
7. 4 Riley starter probes with matched neutrals.

## Deferred (not now, but possible follow-ups)

- Judge running on ablated NLA (requires resolving readability first).
- Runtime hook ablation on M's forward pass (Drift's Phase 1b path).
- Drift's full 24-probe Riley library.
- Pre-registration doc, paired Wilcoxon analytics, Δ-of-Δ tables.
- The `α=1.5` over-projection sweep.

---

## Things that have already burned us (institutional memory)

- **Disk space.** Overnight runs have allocated up to 40 GiB of macOS
  swap. Disk-full once crashed MPS mid-probe. Free space matters more
  than it looks; 50 GiB free can disappear in 18 hours.
- **Port 3000 collision** with the user's Drift Docker container. Web
  is on 3001.
- **Safari SSE buffering.** Without a 2 KB padding comment at the
  start of the SSE stream, Safari/Firefox hold every event until
  end-of-run. See `routes_stream.py`.
- **Activation array O(n²).** Per-event `cells: [...s.cells, ...new]`
  in Zustand spread the entire array per event. Fixed with a
  module-level buffer flushed at 10 Hz.
- **`transformers==5.7.0+` tokenizer wrapper broken for Gemma.**
  Encodes "Hello world" as `['H', 'elloworld']`. Use raw `tokenizers.Tokenizer`.
- **Backend restart drift.** Multiple incidents of fixes shipped to git
  without bouncing the backend. Always restart after a Python commit.
- **`<cite>` tags leaking into journal.** Server-side web_search wraps
  cited text. Strip during analyzer output processing.
- **Journal metadata schema mismatch.** v1-era template requires
  `summary_stats` key; v2 analyzer was writing `window_stats`. Vercel
  prerender crashes. Now the analyzer emits both shapes.
- **`vercel deploy` from wrong CWD.** Project rootDirectory is set in
  the dashboard; always deploy from repo root.
- **SSE replay duplicates rows.** Reconnection replays the event log
  from event 0; store handlers must upsert by position, not append.

---

## Where things live

```
server/cells_interlinked/
  __main__.py              uvicorn entry
  config.py                env-driven settings (.env at repo root)
  api/
    app.py                 FastAPI factory, lifespan loads M + AV + SAE + refusal directions
    routes_probe.py        POST /probe, POST /cancel/{id}, GET /probes/{recent,id}
    routes_stream.py       GET /stream/{id} — SSE drain
    routes_autorun.py      autorun control + state
    routes_journal.py      journal CRM endpoints
    runs.py                RunRegistry + per-run asyncio queues + EventLog
  pipeline/
    model_loader.py        Gemma-3-12B-IT bf16 on MPS, ModelBundle
    nla_client.py          AV decoder — decodes one residual into a sentence
    sae_runner.py          Gemma Scope 2 JumpReLU SAE
    labels.py              Neuronpedia label fetcher + SQLite cache
    phase_tracker.py       per-position residual capture
    generation_loop.py     custom autoregressive loop + residual hooks
    verdict.py             phase-boundary aggregation, TokenRow + aggregate
    judge.py               Gemma-as-judge for eval-suspicion + introspection
    abliteration.py        (NEW for CI 2.5) refusal-direction extract + project_out
    refusal_prompts.py     HARMFUL_PROMPTS + HARMLESS_PROMPTS for Phase B
    probes_library.py      curated probe library (Riley added in Phase G)
    probe_controls.py      BASELINE_CONTROLS, control_for(probe_text)
    probe_queue.py         meta-sets (both, agent-both, matched-controls)
    autorun.py             AutorunController
    analyzer.py            journal analyzer (Claude Opus 4.7 + tools)
    publisher.py           publish_analysis: write, git add/commit/push, vercel deploy
  storage/db.py            aiosqlite schema
  scripts/
    compute_refusal_direction.py    Phase B compute script

web/
  app/                     Next.js 16 + React 19 (port 3001)
  lib/                     sse.ts, store.ts, types.ts, probes.ts

journal/                   separate Next.js app, deployed to cells-interlinked.vercel.app
  data/reports/            published reports (checked in)

docs/
  CI_2_5_PLAN.md           SOURCE OF TRUTH

e2e/                       playwright smoke tests
```
