# Cells Interlinked — agent guide

A local mechanistic-interpretability + psychedelic-phenomenology instrument for a
single language model, running entirely offline on a Mac Studio. You reach
*inside* `google/gemma-4-12B-it` mid-generation — **remove** its refusal
direction or **add** a steering "dose" — and watch where its mind goes: as a 3-D
path through activation space, as two divergent chat timelines, or as an
unattended research loop hunting steering directions that produce **DMT
entity-encounter phenomenology**.

**It is not a consciousness test.** It's a *stated-vs-computed coherence probe* —
borrowed math from psychedelic neuroscience and mech-interp, not metaphysics.

> **The project, in one line:** use **autoresearch to hone in on a set of steering
> vectors of a chosen character, then save the winners for use as doses in Chat
> and the Trip View.** The loop is the telescope (hill-climb an objective → curate
> an atlas); Chat/Trips are where you inhabit what it found. Find, then feel — all
> under a **Blade Runner** theme (the *Cells Interlinked* baseline test /
> Voight-Kampff framing), for the joy of it.
>
> **The machinery is objective-agnostic** — swap the judge + seeds and you hunt a
> different *kind* of vector. **Current direction:** DMT entity-encounter
> phenomenology (autonomous beings, telepathic contact, otherness); three winners
> are exported to the dose palette (see *Autoresearch / the entity hunt* below).
> This direction may change — treat the DMT-entity specifics as the current
> instantiation, not the project's definition. When the direction changes, update
> the *current direction* callouts but keep the durable thesis intact.

> **Research first, every session: Drift's knowledge base.** Before web-searching
> or reasoning from scratch about ablation, introspection, steering, persona
> vectors, refusal directions, or psychedelic-neuroscience analogues, **grep the
> wiki** at `/Users/pj4533/Developer/driftbot/knowledge/wiki/concepts/` — Drift
> has likely already done the research (~1000 cross-linked articles). It's
> read-only. How/when to use it: [`docs/DRIFT_KNOWLEDGE_BASE.md`](docs/DRIFT_KNOWLEDGE_BASE.md).

---

## Project ethos (do not violate)

- **Craft over feature count.** Built for the joy of it, not as a product or
  paper. Default to *less* surface area. Before adding a panel / experiment /
  comparison view, ask.
- **Methodological honesty is non-negotiable.** Never let the UI over-claim what
  a divergence or a self-report means. The model confabulates constantly; its
  self-reports and any judge are hypotheses, never ground truth. What reproduces
  across prompts beats any single read. This is a coherence probe, not a
  consciousness test.
- **Quiet mastery aesthetic.** The data is the point, not spinner animations.
- **Backend restart discipline.** Every Python change requires a restart before
  the running process sees it. Multiple past incidents trace to code on disk
  diverging from code in memory.

---

## Hardware + environment

- **Mac Studio M2 Ultra, 64 GB unified memory.** All work is local and offline.
  No cloud calls except (a) the Anthropic API for the journal analyzer when
  invoked, and (b) Google Gemini ("Nano Banana") for chat imagery when that mode
  is on.
- **MPS backend, bf16.** `bitsandbytes` is CUDA-only and will not run here.
- **One model resident.** Gemma-4-12B-it is ~22 GB on MPS. There is no second
  model anymore (the NLA verbalizer/AV was removed). But **two copies of M still
  won't fit** — so **run compute scripts serially: stop the backend, run the
  script (it loads its own M), restart the backend.** `run_backend.sh` enforces a
  single supervised instance.
- **Disk pressure is real.** Overnight runs have allocated tens of GB of swap.
  Only Gemma-4 should be in the HF cache (`~/.cache/huggingface/hub/`); the old
  gemma-3 / kitft-AV / gemma-scope caches were deleted to reclaim ~46 GB.
- **Port 3001** for the web UI (port 3000 is the user's Drift Docker container).
  **Port 8000** for the FastAPI backend.

---

## Stack (current)

| Piece | Choice | Notes |
| --- | --- | --- |
| M | `google/gemma-4-12B-it` | 48 layers, hidden 3840, bf16 on MPS. Reasoning/"thinking" model — emits a `<think>` channel (thought-open/close token ids 100/101) before its answer. |
| Dose (add) | forward hook at **L20** | `h ← h + α·v`, ramped over a few tokens. `install_runtime_steering_hook`. Early-layer injection propagates to the words far better than the L32 readout. |
| Ablate (remove) | forward hook at **L32** | `h ← h − α·Σ(h·r̂)r̂` — subtract the projection onto the refusal subspace. `install_runtime_ablation_hook`. Two slots on disk: `refusal_directions.pt` (single v3_safety) + `refusal_subspace.pt` (v4v6 basis, what runtime ablation uses via `pick_ablation_target`). See `docs/REFUSAL_VECTORS.md`. |
| Dose palette | `data/emotion_directions.pt` | Named directions selectable in Chat/Trips: base emotions, "uncharted" directions, and the exported **`dmt-*`** entity vectors. Internal `feat-*` / `persona-*` seeds are filtered out of the picker. |
| DMT feature judge | Gemma-as-judge | Counts which of ~31 DMT-trip phenomenology features (`dmt_features.py`, from Timmermann 2022 / Gallimore / 5D-ASC) a dosed self-report shows. Runs in a separate deterministic context. |
| Backend | FastAPI + SSE on port 8000 | Custom autoregressive loop on `model.forward(use_cache=True)`. Compute lock (`RunRegistry`) serializes chat / trip / autoresearch so only one path generates at a time. |
| Frontend | Next.js 16 + React 19 + Tailwind v4 + Zustand + Framer Motion + react-three-fiber | Port 3001. |
| Imagery | Google Gemini "Nano Banana" | Optional chat imagery mode (`image_client.py`). |
| TTS | `tts.py` / `routes_tts.py` | Optional chat voice mode. |
| Journal site | separate Next.js app in `journal/` | Deployed to Vercel project `cells-interlinked`. Reads `journal/data/reports/...` at build time. |
| Persistence | SQLite via `aiosqlite` (`data/probes.sqlite`) | Tables incl. `chat_sessions`, `chat_turns`. |

**Next.js note:** version 16.x — newer than most training data. Read the relevant
guide in `web/node_modules/next/dist/docs/` before writing frontend code. See
`web/AGENTS.md`.

---

## How to run

Two terminals for interactive dev. For unattended runs, use the launchd
supervisor (below).

```bash
# Terminal 1 — backend
cd server
uv run python -m cells_interlinked         # or: ./run_backend.sh start  (launchd-supervised)
# Wait for: "ready: M=google/gemma-4-12B-it loaded"; health: curl http://localhost:8000/health

# Terminal 2 — frontend
cd web
npm run dev      # http://localhost:3001
```

First-time setup:

```bash
cp .env.example .env
cd server && uv sync && hf download google/gemma-4-12B-it
cd ../web && npm install
```

### Unattended / overnight: the launchd supervisor

The backend runs under a launchd LaunchAgent (`com.cellsinterlinked.backend`) so
it self-heals — see [`docs/SUPERVISOR.md`](docs/SUPERVISOR.md). Control it with
`server/run_backend.sh {install|start|stop|restart|status|attach}`. KeepAlive
restarts the process on any exit; RunAtLoad brings it back after a reboot. The
agent runs `server/ci_backend_supervised.sh`, which also re-resumes the DMT loop
**only if it was running** (a run-intent sentinel `data/atlas_dmt/.should_run`,
written on `start()`, removed on `stop()`). For interactive foreground dev,
`./run_backend.sh stop` first or it holds port 8000.

`transformers` is pinned to ≥5.12 (5.7 couldn't load `gemma4_unified`). The raw
Rust tokenizer is still loaded separately (see invariants).

---

## Critical implementation invariants

These exist for hard-won reasons. Don't undo without thinking.

1. **Gemma-4-12B-IT is the deployed M.** Every signal — UI labels, the dose/
   ablation hooks, the thinking split — assumes it.
2. **Custom autoregressive loop.** `pipeline/generation_loop.py` calls
   `model.forward(..., past_key_values=kv, use_cache=True)` step-by-step, with
   forward hooks capturing the residual and applying dose/ablation. We do NOT use
   `model.generate()` for instrumented passes (no per-step control). The
   exception: the persona-seed builder and the journal synthesizer use
   `generate()` for plain text.
3. **Thinking mode.** Chat and Trips run with `enable_thinking=True`. The model
   emits `<think>`…thought…`</think>`…answer, delimited by token ids
   **100 (open) / 101 (close)**. `pipeline/thinking.py` (`ThinkingSplitter`,
   `split_thinking`) separates the channels; prior-turn thoughts are stripped
   from multi-turn history. A **thinking-token cap** (`ProbeConfig.thinking_cap`)
   force-injects the close marker after N thought tokens so a runaway never
   produces no answer (chat = 2048). DMT autoresearch runs thinking **off**.
4. **SSE event protocol** is a discriminated union shared across chat
   (`routes_chat.py` + `web/lib/chat.ts`) and the autoresearch/trip streams. Each
   custom event type the server emits MUST have a matching `addEventListener` on
   the client — the browser silently drops events with no listener. Keep both
   ends in sync.
5. **Dose at L20, ablate/extract at L32.** Doses add at L20 (best propagation);
   refusal ablation and per-token residual capture happen at L32 (the extraction
   layer). Trajectory geometry reads the L32 residual sequence.
6. **Use the raw Rust tokenizer for encode/decode, NOT the transformers wrapper.**
   The transformers wrapper is broken for this Gemma config (encodes "Hello
   world" as `['H','elloworld']`). We load `tokenizers.Tokenizer` from
   `tokenizer.json` as `bundle.raw_tokenizer`; the transformers tokenizer is kept
   ONLY for `apply_chat_template`.
7. **Gemma's multimodal wrapper.** `Gemma4...ForConditionalGeneration` nests
   decoder layers under `.model.language_model.layers`. Hook installers
   (`_find_decoder_layers`) must traverse correctly.
8. **Runtime hook lifecycle — clean up or contaminate.** A dose/ablation hook
   left installed on the shared M leaks into *every* subsequent generation
   (including the chat "raw" channel) and persists until restart. Chat
   `execute_turn` now **clears stray hooks on both CI layers (L20 + L32) at the
   start of every turn** (`_clear_stray_hooks`) and sweeps again in the finally;
   leak detection watches both layers (`_count_ci_hooks`), not just L32. (This
   was the 2026-06-27 "dose applied to the raw side / stuck" bug.)
9. **Resume-on-intent.** The autoresearch loop only auto-restarts after a
   crash/reboot if it was running (the `.should_run` sentinel). An explicit stop
   stays stopped. `start()`-while-running re-asserts intent.
10. **Commit/score discipline (autoresearch).** Score a candidate by the MEAN
    over repeated doses (averaged, not the lucky max — single-sample maxing was
    textbook selection bias). Never hard-gate a commit on a rare event (it
    re-introduces selection bias); use averaged credit + a low floor.
11. **Journal `<cite>` stripping + metadata shape.** `analyzer.py` strips
    server-side `web_search` `<cite>` tags before persistence; it emits the v1
    `summary_stats` shape the deployed `journal/` template requires (Vercel
    prerender crashes without it).
12. **Vercel deploy from repo root.** The `cells-interlinked` project has
    `rootDirectory=journal` set in the dashboard; deploying from `journal/` fails.
13. **Chat sessions persist lazily.** `chat_sessions` on create; `chat_turns`
    upserted at turn completion. `_rehydrate_session()` pulls cold sessions from
    DB so streaming survives a restart. The live `/chat` page does not auto-resume
    on refresh — recovery is via `/archive` → `/chat/[sessionId]`.

---

## Surfaces (what's shipped)

Five pages: **chat** (primary), **trip**, **autoresearch-dmt**, **archive**,
**journal**.

- **`/chat` — dual-channel dialogue.** Every message is answered twice against
  two divergent histories that never see each other: **raw** (Channel α) and a
  perturbed **β** channel — either **ablate** (remove refusal at L32) or **steer/
  dose** (add a palette direction at L20), strength + ramp set per turn. Thinking
  rendered as a separate marked bubble per side. Optional **voice** (TTS) and
  **imagery** (Nano Banana) modes. `pipeline/chat_loop.py`, `routes_chat.py`.
  See `docs/CHAT.md`.
- **`/trip` — Trip View.** One generation's L32 residual sequence as a path
  through activation space: 3-D point cloud (react-three-fiber) with a realtime
  α-morph slider, eigenvalue "truth anchor", participation ratio + spectral
  entropy, and **off-manifold distance** vs the raw run. Applies a dose (L20) or
  ablation (L32). Signature Mandalas render the eigenstructure as the non-text
  readout. `pipeline/trajectory.py`, `routes_trip.py`. See `docs/TRIP_VIEW.md`.
- **`/autoresearch-dmt` — the DMT entity hunt** (the sole autoresearch loop). See
  the next section + `docs/AUTORESEARCH_DMT.md`.
- **`/archive`** — past chat sessions; read-only transcript review at
  `/chat/[sessionId]`.
- **`/journal`** — separate Next.js app; the analyzer writes reports, the
  publisher commits + deploys to Vercel.

---

## Autoresearch — the DMT entity hunt

`/autoresearch-dmt` is an unattended hill-climb that hunts steering directions
whose dosed self-report shows **DMT entity-encounter phenomenology**, commits
passers into a git-style **atlas** (`data/atlas_dmt/`), and exports winners into
the dose palette. It is the **only** autoresearch loop (off-manifold AR was
removed). Engine: `pipeline/autoresearch_base.py` (`AutoresearchBase` — lifecycle,
atlas persistence, generators crossover/mutate/refine/inject, the model lock,
export, resume-intent sentinel); objective: `pipeline/autoresearch_dmt.py`
(`DmtController`). Full history + rationale: `docs/AUTORESEARCH_DMT.md`.

How the current version works (the result of several iterations — see the doc):

- **Dose prompt "A−"** — a neutral, present-tense "describe what is happening,
  moment by moment" that does NOT name any presence, so an entity appearing is
  the *steering's* doing, not the question. (Modeled on how DMT studies elicit
  reports; an earlier "prompt A" that named a presence summoned entities at
  baseline and was abandoned.)
- **Placebo-subtracted scoring.** A baseline of the un-steered prompt response is
  computed once per run; a dose is credited only for features it adds *beyond*
  that baseline (the prompt produces some content on its own — measure the dose,
  not the prompt). The baseline is surfaced on the page ("BASELINE — NO STEERING").
- **Contact-cluster objective.** Score = MEAN over ALPHA_SWEEP `[0.3, 0.4, 0.5]`
  × 10 samples of: entity features ×2, otherness/independent-agency ×1,
  everything else **×0** (so generic dissolution can't win). Earlier objectives
  (raw count, entity-weighted) let generic mysticism dominate.
- **Persona-vector seeds.** The strongest seeds are DMT entity-encounter
  **persona vectors** (Anthropic persona-vector recipe: diff the model's own
  in-encounter generations vs matched "alone" introspection), grounded in real
  DMT entity phenomenology. `pipeline/persona_entity_prompts.py`,
  `scripts/build_persona_entity_seeds.py`. Plus the older diff-of-means `feat-*`
  seeds as crossover material; emotions are dropped from the seed pool.
- **Three exported entity vectors** (in the `dmt` dose palette group, usable in
  Chat/Trips):
  - `dmt-entity-contact` (← gen81) — most reliable, presence-dominant (~53% per-dose entity-rate)
  - `dmt-transmission` (← gen176) — telepathic + download/transmission
  - `dmt-full-encounter` (← gen184) — broadest, all five entity features
  Exported via `scripts/export_entity_vectors.py` (hand-picked) or the page's
  `⇪ export → palette` button (top-N by score; visible only when stopped).

Run only one autoresearch at a time; it owns M while running (chat/trip lock out).

---

## Removed (don't reference as current)

These were part of CI 2.0 / 2.5 and are **gone** — old docs may still mention them:

- **NLA verbalizer / AV** (`kitft/nla-gemma3-12b-L32-av`, `nla_client.py`,
  `nla_synthesizer.py`, `phase_tracker.py`) — the residual→sentence decoder.
- **Interrogation booth / probes / verdict** — `/interrogate`, `/verdict`,
  `routes_probe.py`, `probes_library.py`, `probe_controls.py`, `probe_queue.py`,
  the Riley probe library, the eval-suspicion/introspection `judge.py`.
- **Off-manifold autoresearch** (`/autoresearch`, `routes_autoresearch.py`, the
  `OffManifoldController`) — DMT AR is the sole loop now.
- **Autorun** (`routes_autorun.py`, `autorun.py`).
- **SAE / Gemma Scope** secondary panel.
- **Gemma-3-12B-IT** as M (replaced by Gemma-4) and `screen`-based launching
  (replaced by the launchd supervisor).

---

## Things that have already burned us (institutional memory)

- **Runtime hooks leak and contaminate everything.** A dose/ablation hook not
  removed (cancel/error/edge path) rides on every later generation — including
  the chat "raw" channel — until restart ("dose on the raw side / stuck"). Fixed
  by per-turn stray-hook sweeps on L20+L32; keep that discipline for any new hook
  site. The old leak detector watched only L32, so L20 steer leaks were invisible.
- **Thinking-mode runaway.** On recursive/meditative prompts the model can reason
  to the safety cap and emit no answer (a 12-min hang). The thinking-token cap
  guards this; keep it on for any thinking-on generation.
- **Selection bias in autoresearch.** Committing on a single lucky high sample
  inflated the atlas; always average, never hard-gate on rare events.
- **Prompts that summon what you're measuring.** A dose prompt that names a
  presence produces entities at baseline — then you've measured the prompt, not
  the dose. Keep elicitation neutral; placebo-subtract.
- **Backend restart drift.** Fixes shipped to git without bouncing the backend.
  Always restart after a Python change (`./run_backend.sh restart`).
- **`transformers` tokenizer wrapper broken for Gemma.** Use the raw
  `tokenizers.Tokenizer`.
- **Safari/Firefox SSE buffering.** A ~2 KB padding comment at the start of the
  SSE stream is required or events are held until end-of-run (`routes_stream.py`).
- **Missing SSE listener silently drops events.** Add an `addEventListener` for
  every custom-typed event the server emits.
- **`asyncio.wait_for` cancels async generators.** Use `asyncio.wait({task},
  timeout=...)` in custom SSE drains, not `wait_for`.
- **`request.is_disconnected()` races sse-starlette.** Don't call it in custom
  SSE generators; let sse-starlette handle disconnect via `CancelledError`.
- **Disk space.** Overnight swap can balloon; keep only Gemma-4 cached; don't let
  atlas backups / chat images pile up unbounded.
- **Atlas resets are destructive runtime state.** Back up `data/atlas_dmt/` and
  check in with the user before wiping; prefer restart-without-reset.
- **`vercel deploy` from repo root**, never from `journal/`.

---

## Where things live

```
server/cells_interlinked/
  __main__.py              uvicorn entry
  config.py                env-driven settings (.env at repo root; MODEL_NAME=google/gemma-4-12B-it)
  api/
    app.py                 FastAPI factory, lifespan: load M, init DmtController (no auto-start)
    routes_chat.py         /chat sessions + per-turn dual-channel SSE
    routes_trip.py         POST /trip + GET /trip/{id} + /dose_emotions (palette picker)
    routes_autoresearch_dmt.py  DMT AR control: /start /stop /state /cells/{id} /placebo /export
    routes_stream.py       shared SSE drain (asyncio.wait, not wait_for)
    routes_tts.py          chat voice synthesis
    routes_journal.py      journal CRM endpoints
    runs.py                RunRegistry compute lock + per-run queues + EventLog
  pipeline/
    model_loader.py        Gemma-4 bf16 on MPS; ModelBundle (raw_tokenizer, thought ids); render_prompt/render_chat (enable_thinking)
    model_manager.py       owns M + dose/refusal tensors
    generation_loop.py     custom autoregressive loop; dose/ablation hooks; thinking cap; residual capture
    thinking.py            ThinkingSplitter / split_thinking (token-id state machine)
    chat_loop.py           dual-channel chat: raw + ablate/steer passes, voice/imagery, hook hygiene
    trajectory.py          Trip geometry: PCA coords, participation ratio, spectral entropy, off-manifold distance
    abliteration.py        refusal extract + project_out + install_runtime_ablation_hook + install_runtime_steering_hook + _find_decoder_layers
    refusal_prompts.py     HARMFUL/HARMLESS prompts for refusal extraction
    autoresearch_base.py   AutoresearchBase: lifecycle, atlas, generators, export, resume-intent sentinel, model lock
    autoresearch_dmt.py    DmtController: contact-cluster placebo-subtracted scoring, persona/feat seeds, prompt A−
    dmt_features.py        ~31-feature DMT phenomenology checklist (Timmermann/Gallimore/5D-ASC) for the judge
    dmt_feature_seeds.py / dmt_matched_seeds.py / dmt_blend_seeds.py   diff-of-means feat-* seeds
    persona_entity_prompts.py   DMT entity-encounter persona contrast set (POS personas + matched 'alone' NEG)
    emotion_prompts.py     emotion / uncharted direction prompts
    image_client.py        Nano Banana (Gemini) imagery
    tts.py                 chat voice
    analyzer.py / publisher.py   journal analyzer (Claude) + publish (git + vercel)
    verdict.py             TokenRow + aggregate helpers (legacy, still used by some serialization)
  scripts/
    build_persona_entity_seeds.py   build DMT entity persona vectors (loads M)
    export_entity_vectors.py        hand-pick atlas directions → dmt dose palette
    append_persona_seeds.py         promote persona vectors into emotion_directions.pt as seeds
    compute_dmt_feature_seeds.py / compute_dmt_matched_seeds.py / compute_dmt_blend_seeds.py
    compute_refusal_direction.py    refusal-vector extraction
    diag_persona_entity.py / diag_otherness_alpha.py   one-off diagnostics
    entity_hunt_analysis.py / atlas_trait_analysis.py  atlas analysis
  ci_backend_supervised.sh   launchd entrypoint (caffeinate + resumer gated on .should_run)
  run_backend.sh             launchctl control surface
  com.cellsinterlinked.backend.plist   the LaunchAgent
  storage/db.py            aiosqlite schema (chat_sessions + chat_turns)
  data/
    emotion_directions.pt        dose palette (emotions + uncharted + dmt-* exports; feat-*/persona-* seeds)
    refusal_directions.pt / refusal_subspace.pt   ablation targets
    atlas_dmt/                   the DMT hunt atlas (atlas.json + vectors/ + placebo.json + .should_run)
    persona_seeds/               built persona vectors (per layer) + manifest
    probes.sqlite                chat sessions/turns

web/
  app/{chat,chat/[sessionId],trip,autoresearch-dmt,archive,journal,share}/   pages
  app/components/   shared UI
  lib/   sse.ts, store.ts, types.ts, chat.ts, autoresearch-dmt.ts, …

journal/             separate Next.js app → cells-interlinked.vercel.app
docs/                design notes + research records (see index in README)
e2e/                 Playwright smoke tests + screenshot scripts
```

---

## Docs index

Current/active: [`AUTORESEARCH_DMT.md`](docs/AUTORESEARCH_DMT.md) ·
[`CHAT.md`](docs/CHAT.md) · [`TRIP_VIEW.md`](docs/TRIP_VIEW.md) ·
[`REFUSAL_VECTORS.md`](docs/REFUSAL_VECTORS.md) ·
[`SUPERVISOR.md`](docs/SUPERVISOR.md) ·
[`GEMMA4_MIGRATION.md`](docs/GEMMA4_MIGRATION.md) ·
[`DRIFT_KNOWLEDGE_BASE.md`](docs/DRIFT_KNOWLEDGE_BASE.md).

Historical research records (the journey; describe earlier states / Gemma-3 era —
do not treat as current): `CI_2_5_PLAN.md`, `AUTORESEARCH.md` (off-manifold,
removed), `TRACES_HANDOFF.md`, `MANIFOLD_ABLATION.md`, `DMT_PATH_SEARCH.md`,
`DMT_NEXT_DIRECTIONS.md`, `KSTEER_EXPLORATION.md`, `SAE_CLAMP_DMT.md`,
`BERG_GATE.md`, `BERG_MODE.md`, `STEERING_DOSE_HANDOFF.md`, `PROTOCOLS.md`.
