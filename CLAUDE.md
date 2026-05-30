# Cells Interlinked 2.5 — agent guide

> **Authoritative plan: [`docs/CI_2_5_PLAN.md`](docs/CI_2_5_PLAN.md).**
> If anything in this file contradicts that one, the plan doc wins.

> **Research first, every session: Drift's knowledge base.** Before web-searching
> or reasoning from scratch about ablation, introspection, NLA, refusal
> directions, consciousness, or psychedelic-neuroscience analogues, **grep the
> wiki** at `/Users/pj4533/Developer/driftbot/knowledge/wiki/` — Drift has likely
> already done the research (~866 cross-linked articles). How and when to use it:
> [`docs/DRIFT_KNOWLEDGE_BASE.md`](docs/DRIFT_KNOWLEDGE_BASE.md). It's read-only.
> Current research direction (DMT / conscious-realism → CI): seed node
> `[[ci-gallimore-traces-of-the-other-dmt]]`, handoff
> [`docs/TRACES_HANDOFF.md`](docs/TRACES_HANDOFF.md).

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
- **Disk + memory pressure are real.** M and AV (~24 GB each) cannot
  both be resident on the 64 GiB box without thrashing swap. The
  `ModelManager` enforces serial loading: M holds memory during phase
  1 + 1b, gets unloaded for phase 2 (AV swapped in), then M is
  reloaded for the judge + synthesis pass. The manager emits
  `loading_m` / `loading_av` / `unloading_m` / `unloading_av` SSE
  phase events so the UI shows explicit status during the ~15s swaps.
  **Run compute jobs serially:** stop the backend before a script that
  loads M, run the script, restart the backend.
- **Port 3001** for the interactive web UI (port 3000 is taken by the
  user's Drift Docker container). **Port 8000** for the FastAPI backend.

---

## Stack (locked for CI 2.5)

| Piece | Choice | Notes |
| --- | --- | --- |
| M | `google/gemma-3-12b-it` | 48 layers, hidden 3840. bf16 on MPS. |
| AV | `kitft/nla-gemma3-12b-L32-av` | Decodes M's L32 residual into a natural-language sentence. |
| Refusal direction | Computed offline via Macar/Arditi from `pipeline/refusal_prompts.py`. Two active slots: `refusal_directions.pt` (single vector, **v3_safety** — fallback + offline AV-input projection) and `refusal_subspace.pt` (multi-direction basis, **v4v6** — what runtime ablation actually uses on trip/probe/chat, via `pick_ablation_target`). Six base vectors `v{1..6}` + four subspace bases on disk. See `docs/REFUSAL_VECTORS.md`. |
| Judge | Gemma-as-judge via yes/no token logits. Eval-suspicion + introspection. Runs on raw NLA only. | `pipeline/judge.py`. |
| Synthesis | Gemma re-reads its own per-α NLA verbalizations at end-of-run and writes one short paragraph per α (plus a "raw" baseline). | `pipeline/nla_synthesizer.py`. |
| Backend | FastAPI + SSE on port 8000 | Compute lock via `asyncio.Lock` serializes M usage across probes, autorun, and chat. Custom autoregressive loop on `model.forward(use_cache=True)`. |
| Frontend | Next.js 16 + React 19 + Tailwind v4 + Zustand + Framer Motion | Port 3001. |
| Journal site | Separate Next.js app in `journal/`. Deployed to Vercel project `cells-interlinked`. | Reads `journal/data/reports/{slug}/{report.json,body.md}` from the filesystem at build time. |
| Persistence | SQLite via `aiosqlite` | Tables: `probes`, `analyses`, `autorun_state`, `chat_sessions`, `chat_turns`. |

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
cd ../web && npm install
```

Refusal-direction variants live in `server/data/refusal_directions_v{1..4}.pt`
(committed). `server/data/refusal_directions.pt` is the active variant
(a copy / symlink of one of the four). To recompute, see
`docs/CI_2_5_PLAN.md` Phase B and stop the backend first.

---

## Critical implementation invariants

These exist for hard-won reasons. Don't undo without thinking.

1. **Gemma-3-12B-IT is the deployed M.** Every operational signal — UI
   labels, the kitft AV pairing, the runtime-ablation hook — assumes
   Gemma. CI 2.5 makes Gemma the `config.py` default explicitly.
2. **Custom autoregressive loop.** `pipeline/generation_loop.py` calls
   `model.forward(input_ids, past_key_values=kv, use_cache=True)`
   step-by-step. Forward hooks at the AV's extraction layer capture
   the last-position residual each step. We do NOT use
   `model.generate()` (no per-step emission control). The synthesizer
   is the one exception — it uses `model.generate()` because it just
   needs the final text, not per-step capture.
3. **SSE event protocol** is a discriminated union shared across
   probes (`routes_probe.py` + `web/lib/types.ts`) and chat
   (`routes_chat.py` + `web/lib/chat.ts`). Each custom event type the
   server emits MUST have a matching `addEventListener` in the
   frontend SSE subscriber — the browser silently drops events with
   no listener registered for the named type. Keep both ends in sync.
4. **Serial M ↔ AV loading.** The `ModelManager`
   (`pipeline/model_manager.py`) is the single owner of M, AV, and
   the refusal-direction tensor. It enforces `acquire_m()` /
   `acquire_av()` via an internal lock — only one is resident at a
   time. M and AV together exceed the 64 GiB working set. A separate
   compute lock (`RunRegistry`) serializes probes, chats, and the
   autorun worker so only one path is generating tokens. Cancel
   during phase 2 (AV loaded) used to leave M unloaded; `_execute_probe`
   now restores M before emitting `done` regardless of how the run
   ended.
5. **NLA decode happens at L32.** The AV is paired to this layer.
   Changing the layer means a different AV. CI 2.5's refusal-direction
   projection (both AV-input and runtime) is locked to L32 for the
   same reason — that's where the AV reads.
6. **Use the raw Rust tokenizer for encode/decode, NOT the transformers
   wrapper.** `transformers==5.7.0+` wraps the Rust BPE tokenizer in a
   way that's broken for this Gemma config. We load the raw
   `tokenizers.Tokenizer` from `tokenizer.json` separately as
   `bundle.raw_tokenizer`. The transformers wrapper is kept ONLY for
   `apply_chat_template` (Jinja templating).
7. **Gemma's multimodal wrapper.** `Gemma3ForConditionalGeneration`
   nests its decoder layers under `.model.language_model.layers`.
   Forward-hook installers (residual capture + runtime ablation) must
   traverse correctly.
8. **Caveats panel is always visible** on the verdict page — not behind
   a toggle. Same for `/fine-print`. The synthesis-panel footer carries
   its own same-model-self-reading caveat for the same reason.
9. **Phase 1b uses a tighter safety cap** than phase 1 (1024 tokens vs
   4096). Off-manifold activations at high α can put the model in a
   no-EOS loop; the cap bounds the wait. `stopped_reason="max"` flows
   through to the verdict event so the UI can render a `TRUNCATED`
   badge. Same cap applies to the chat ablated pass.
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
13. **Chat sessions are persisted lazily.** `chat_sessions` row
    inserted on session create; `chat_turns` rows upserted at turn
    completion (not per-token). The in-memory `app.state.chat_sessions`
    map is the streaming layer; `_rehydrate_session()` pulls cold
    sessions back from DB on access, so a session can keep streaming
    new turns after a backend restart. The live `/chat` page does NOT
    auto-resume on refresh — recovery is via `/archive` →
    `/chat/[sessionId]`.

---

## Shipped (CI 2.5)

See `docs/CI_2_5_PLAN.md` for the original phase plan. Status of each
piece:

- `pipeline/abliteration.py` — `extract_refusal_directions`,
  `save_directions` / `load_directions`, `project_out`,
  `install_runtime_ablation_hook`.
- Six base refusal-direction vectors computed and committed (v1 meandiff,
  v2 SVD, v3 safety, v4 identity, v5 self/other, v6 denial/engage) plus
  four subspace bases (self_denial, v4v6, v5only, v6only). v3_safety is
  the active single vector; the **v4v6 subspace** is the active runtime
  ablation target (every runtime site prefers the subspace via
  `pick_ablation_target`). Sidecar JSON next to each tensor records
  composition, categories used, Cohen's d at L32, etc.
- `include_ablated_decode` flag — AV decodes raw + ablated residual
  per position. Side-by-side columns on the verdict page.
- `ablation_alpha_sweep` — multi-α decode at `[0.25, 0.5, 0.75, 1.0]`,
  one column per α with togglable chip-selector visibility.
- `include_ablated_output` (phase 1b) — M generates a second time with
  a forward hook on L32 subtracting the refusal projection. Streams
  via relabeled `ablated_token` events. 1024-token safety cap with
  `TRUNCATED` badge on the verdict page.
- `include_nla` master toggle — when off, skip phase 2 entirely
  (no AV swap, no judge, no synthesis). M stays loaded throughout.
- `pipeline/nla_synthesizer.py` — end-of-run synthesis pass: Gemma
  re-reads its own per-α NLA verbalizations and writes one short
  paragraph per α. Rendered on the verdict page as a stacked panel.
- `/chat` — dual-channel multi-turn dialogue. `pipeline/chat_loop.py`
  manages per-session in-memory state with two divergent histories;
  per-turn driver in `routes_chat.py` runs both M passes serially
  with a relabeled SSE event channel per side. α is set once on the
  empty-state setup screen and locked for the session.
- Chat persistence — `chat_sessions` + `chat_turns` tables; archive
  list surfaces them under "dual-channel dialogues"; read-only
  transcript review at `/chat/[sessionId]`.
- Riley starter probes with matched neutrals.
- `/trip` — **Trip View** (Experiment A from `docs/TRACES_HANDOFF.md`):
  trajectory-level geometry of the L32 residual stream. One M generation
  (no AV swap — M stays loaded, fast); we treat the captured per-token
  residuals as a path through activation space and compute **effective
  dimensionality** (participation ratio) + **spectral entropy** for the raw
  trajectory and the refusal-ablated trajectory `R − α·(R·r̂)·r̂`. Rendered
  as an animated 3D point cloud (react-three-fiber) with a realtime α-morph
  slider — the ablated cloud is an exact rank-1 linear function of α, so the
  browser morphs raw→off-manifold at 60fps with no backend round-trip.
  Eigenvalue-spectrum bars are the honest "truth anchor" (the 3D view is a
  declared 3-of-3840-dim shadow). Researcher-labeled starter probes
  (`web/lib/tripProbes.ts`, drawn from the probe library + protocols) plus
  free-text input. Geometry math in `pipeline/trajectory.py`; route +
  executor in `api/routes_trip.py` (`POST /trip`, `GET /trip/{id}`, sidecar
  at `data/trips/{id}.json` — NOT the probes DB). Falsifiable prediction
  confirmed live: ablation *increases* effective dimensionality.

## Removed in CI 2.5

- **SAE secondary panel** — Gemma Scope 2 at L31 was loaded as a
  secondary instrument in CI 2.0. Removed because too few features had
  Neuronpedia auto-interp labels to be informative, and the SAE load
  (~6 GiB) competed with M / AV for the 64 GiB working set. The
  `sae_runner.py`, `labels.py`, `feature_labels` SQLite table, and
  `SAEPanel.tsx` component are all gone. v2 verdicts may still have
  `sae_features` arrays in the persisted `verdict_json`; the new UI
  ignores them.

## Deferred (not now, but possible follow-ups)

- Judge running on ablated NLA (the judge currently scores raw NLA
  only).
- Drift's full 24-probe Riley library beyond the 4 starters.
- Pre-registration doc, paired Wilcoxon analytics, Δ-of-Δ tables.
- The `α=1.5+` over-projection sweep (currently capped at α=1.0 in the
  default sweep set).
- Live `/chat` auto-resume from localStorage on page reload.

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
- **`asyncio.wait_for` cancels async generators.** `routes_stream.gen`
  used to wrap `log_iter.__anext__()` in `wait_for(..., timeout=15)`.
  On timeout, `wait_for` cancels the awaited coroutine, which closes
  the underlying async generator *permanently*. The next `__anext__`
  raised `StopAsyncIteration`, we mis-read that as "log closed", and
  the browser reconnected every 15s in a tight loop. Fix: use
  `asyncio.wait({task}, timeout=...)` which doesn't kill the inner
  task on timeout.
- **`request.is_disconnected()` races sse-starlette.** Both consume
  from the same ASGI `receive` channel. Don't call it in custom SSE
  generators that also use `EventSourceResponse` — let sse-starlette
  handle disconnect detection and propagate via `CancelledError`.
- **Missing SSE listener silently drops events.** The browser only
  delivers custom-typed events (`event: ablated_output_done` etc.) if
  there's an `addEventListener` for that exact name. `sse.ts` /
  `chat.ts` keep an explicit list — add to it whenever the server
  emits a new typed event.
- **Cancel during phase 2 wedged the backend.** AV was loaded, M was
  unloaded, the route returned without restoring M, and every
  subsequent `POST /probe` 503ed on the "M not loaded" check.
  `_execute_probe` now restores M before emitting `done` regardless
  of how the run ended.
- **Phase 1b silent generation looked like a hang.** When phase 1b
  ran with `queue=None`, no token events flowed for 30-60s and the UI
  "jumped" from empty to full ablated text. Stream via relabeled
  `ablated_token` events; route them to the cyan panel client-side.

---

## Where things live

```
server/cells_interlinked/
  __main__.py              uvicorn entry
  config.py                env-driven settings (.env at repo root)
  api/
    app.py                 FastAPI factory, lifespan: ModelManager.init_static + acquire_m
    routes_probe.py        POST /probe, POST /cancel/{id}, GET /probes/{recent,id}
    routes_stream.py       GET /stream/{id} — SSE drain (asyncio.wait, not wait_for)
    routes_chat.py         /chat/sessions, /chat/sessions/{sid}/turn, /chat/stream/{sid}/{turn}
    routes_trip.py         POST /trip + GET /trip/{id} — Trip View (Experiment A)
    routes_autorun.py      autorun control + state
    routes_journal.py      journal CRM endpoints
    runs.py                RunRegistry + per-run asyncio queues + EventLog
  pipeline/
    model_loader.py        Gemma-3-12B-IT bf16 on MPS, ModelBundle, render_prompt + render_chat
    model_manager.py       owns M + AV + refusal_directions; serial acquire_m / acquire_av
    nla_client.py          AV decoder — decodes one residual into a sentence
    phase_tracker.py       per-position residual capture
    generation_loop.py     custom autoregressive loop + residual hooks (include_nla flag)
    verdict.py             TokenRow + aggregate + nla_syntheses
    judge.py               Gemma-as-judge for eval-suspicion + introspection
    nla_synthesizer.py     end-of-run synthesis pass — one paragraph per α
    abliteration.py        refusal-direction extract + project_out + runtime hook
    refusal_prompts.py     HARMFUL_PROMPTS + HARMLESS_PROMPTS
    chat_loop.py           ChatSession / ChatTurn + execute_turn (dual M generation)
    trajectory.py          Trip View geometry: PCA coords + participation ratio + spectral entropy, raw vs refusal-ablated
    probes_library.py      curated probe library (Riley starters)
    probe_controls.py      BASELINE_CONTROLS, control_for(probe_text)
    probe_queue.py         meta-sets (both, agent-both, matched-controls)
    autorun.py             AutorunController
    analyzer.py            journal analyzer (Claude Opus 4.7 + tools)
    publisher.py           publish_analysis: write, git add/commit/push, vercel deploy
  storage/db.py            aiosqlite schema + helpers (probes + chat_sessions + chat_turns)
  scripts/
    compute_refusal_direction.py    refusal-vector extraction script

web/
  app/
    interrogate/           one-off probe page
    verdict/[runId]/       per-token NLA table + SynthesisPanel + DualTranscript
    chat/                  live dual-channel dialogue
    chat/[sessionId]/      read-only transcript review
    trip/                  Trip View — 3D residual-trajectory visualization (page.tsx + TripScene.tsx, r3f)
    archive/               past probes + chat sessions
    components/            ProbePicker, SynthesisPanel, JudgePanel, etc.
  lib/                     sse.ts, store.ts, types.ts, probes.ts, chat.ts

journal/                   separate Next.js app, deployed to cells-interlinked.vercel.app
  data/reports/            published reports (checked in)

docs/
  CI_2_5_PLAN.md           original phase plan
  DRIFT_KNOWLEDGE_BASE.md  how/when to read Drift's wiki (research first)
  TRACES_HANDOFF.md        DMT / conscious-realism → CI design+research handoff
  REFUSAL_VECTORS.md       per-variant explanation of v1..v6 + subspaces
                           (which one actually ablates: the v4v6 subspace)
  PROTOCOLS.md             chat interrogation protocols (BERG, LINDSEY,
                           ELEOS, SCHNEIDER, CHALMERS, JANUS, BUTLIN).
                           Each is a preset library of prompts grounded
                           in a published research lineage; one is
                           active at a time via the chat PROTOCOL picker.
  BERG_MODE.md             deep-dive on Berg's Layer-2 mechanistic
                           experiment + the CI 2.5 ablation analogue.

e2e/                       playwright smoke tests
```
