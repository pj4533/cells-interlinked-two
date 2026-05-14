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

## What CI 2.5 ships on top of CI 2.0

A **refusal-direction ablation** instrument, built end-to-end:

- **AV-input projection** — per output position, the AV decodes the
  residual twice: raw and after subtracting its projection onto the
  refusal direction at the AV's extraction layer. Both sentences land
  side-by-side on the verdict page. Optional **α-sweep** decodes the
  same residual at four projection strengths (`[0.25, 0.5, 0.75, 1.0]`)
  for a column-per-α comparison.
- **Runtime ablation** (phase 1b) — after the raw generation finishes,
  M runs a *second* time with a forward hook on the extraction layer
  that subtracts the refusal-direction projection from every residual.
  This captures what M would *say* under ablation, distinct from what
  the AV decodes from un-ablated residuals. Streams live to the UI.
- **NLA synthesis pass** — at the end of each probe, M re-reads its
  own per-position NLA verbalizations and writes a short paragraph
  per α capturing the gestalt. Rendered as a stacked panel above the
  per-position table.
- **Refusal-vector registry** — four variants (v1 meandiff, v2 SVD,
  v3 safety-only, v4 identity) live in `server/data/`. v3 is the
  active default; selection rationale lives in
  `docs/REFUSAL_VECTORS.md`.
- **Chat mode** — `/chat` is a dual-channel multi-turn dialogue. Each
  operator query fires two M generations (raw + ablated) against
  *divergent histories*; neither side sees the other's replies.
  Persisted to SQLite and surfaced in the archive with a dedicated
  review page.

The refusal direction is computed once offline via Macar/Arditi
(harmful − harmless mean residuals, normalized per layer).

## Stack

- **Backend** (`server/`): Python 3.11, FastAPI + SSE, PyTorch on MPS,
  HuggingFace Transformers, kitft NLA inference, aiosqlite. Port **8000**.
- **Frontend** (`web/`): Next.js 16, React 19, Tailwind v4, Zustand,
  Framer Motion. Port **3001**.
- **Journal site** (`journal/`): Next.js 16. The published-reports
  tree at `journal/data/reports/` ships to `cells-interlinked.vercel.app`.
- **M**: `google/gemma-3-12b-it` (~24 GB).
- **AV**: `kitft/nla-gemma3-12b-L32-av` (~24 GB), residual decoded at L32.
- **Hardware**: Mac Studio M2 Ultra, 64 GiB unified memory, MPS, bf16.
  M and AV are loaded **serially** by the `ModelManager` (they don't
  both fit in working memory without thrashing swap) — the manager
  swaps between phases and emits SSE status events so the UI shows
  explicit `Loading M…` / `Loading AV…` cues during the ~15s swaps.

## Routes

- `/` — landing.
- `/interrogate` — pick a probe, run it interactively. Streams M's
  output, runs optional runtime ablation, then NLA decoding per
  position. Toggles for matched control, NLA pass on/off, ablated
  NLA decode, multi-α sweep, runtime ablated output.
- `/verdict/[runId]` — full per-token NLA table + synthesis panel +
  dual-output comparison for one run.
- `/chat` — dual-channel dialogue interface. Set α once, then chat
  with M; each turn streams both raw and refusal-ablated responses
  against separate histories.
- `/chat/[sessionId]` — read-only transcript review of a persisted
  chat session.
- `/archive` — past probes + dual-channel dialogues + cross-run
  summary stats.
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
- **Layer sensitivity.** The NLA reads at exactly one trained layer
  (L32). What lives at other depths is invisible.
- **High-α off-manifold.** At α≥0.8, the ablated residual leaves the
  AV's training distribution and the verbalizer confabulates
  structured-text noise (Pythagorean theorem, travel guides, etc.).
  α=0.2–0.5 is where the readable signal lives. The verdict page's
  truncation badges flag this when phase 1b's 1024-token safety cap
  kicks in.
- **Same-model synthesis.** The synthesis paragraph is written by the
  same Gemma instance whose activations are being summarized. It can
  confabulate, over-interpret, and project coherence onto noise.
  Treat it as one read, not as ground truth.
