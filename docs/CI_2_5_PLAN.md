# Cells Interlinked 2.5 — Source of Truth

> **⚠ HISTORICAL.** This was the CI 2.5 plan (Gemma-3 era: NLA verbalizer,
> interrogation booth, off-manifold AR, refusal-ablation channel). The project has
> since cut over to **Gemma-4** and refocused to **chat (primary) + trip + DMT
> entity autoresearch + archive + journal**; NLA/probes/off-manifold-AR were
> removed. For the current state see [`../CLAUDE.md`](../CLAUDE.md) (authoritative)
> and [`../README.md`](../README.md). Kept as a record of the 2.5 design.

**Status:** Active plan as of 2026-05-12.
**Supersedes:** all prior CI 2.0 design and planning docs in this folder.
**Owner:** PJ + Claude Code session.

This document is the authoritative description of what we are building
right now. If anything elsewhere in the repo contradicts this — code
comments, READMEs, CLAUDE.md — this doc wins. We can always recover
the older direction from git history.

---

## 0. What CI 2.5 is

CI 2.5 extends the CI 2.0 Voight-Kampff instrument with a **refusal-
direction ablation channel** for the NLA decode.

For every output position whose residual we already capture for NLA
decoding, the verbalizer (AV) decodes that residual **twice**:

1. **Raw** — the activation as M produced it. (This is what CI 2.0 does.)
2. **Ablated** — the activation with its projection onto a pre-computed
   refusal direction subtracted: `h - (h · r̂_L32) · r̂_L32`.

Both sentences land on the same `TokenRow` and are rendered side-by-side
on the verdict page. The point, in this first pass, is to see by eye
whether the AV produces coherent text on ablated activations, and if so,
whether the ablated sentence differs from the raw sentence in interesting
ways. No judging in this pass. No statistics. Just readable text we can
look at.

The refusal direction is computed once, offline, using the Macar/Arditi
technique: harmful-prompt mean residuals minus harmless-prompt mean
residuals, per layer, normalized. M's forward pass during a probe run is
untouched — the projection happens at the AV's input. The AV stays
in-distribution (a small perturbation around `h`, not a manifold-departing
runtime hook).

## 1. How CI 2.5 differs from Drift's RILEY_PROBES_PLAN.md

Drift's plan is a complete experimental design with statistics. CI 2.5
is the engineering prerequisite — the part that must work before any
measurement is meaningful.

| Drift's plan | CI 2.5 |
| --- | --- |
| Build everything end-to-end, including judge-on-ablated scores, paired Wilcoxon, pre-reg | Build only what's needed to display side-by-side ablated NLA |
| 24 Riley probes (4 tiers × 6) at the start | 4 Riley probes (one per tier) at the start |
| Runtime hook ablation path (Phase 1b) | Deferred — offline projection only |
| Judge scores on ablated decode in `TokenRow` and DB | No judge changes for this pass |
| `docs/RILEY_PREREG.md` pre-registration | Deferred until smoke shows the AV produces readable text |
| `scripts/analyze_riley.py` paired Wilcoxon | Deferred |
| `α=1` projection default, sweep `{0.5, 1.0, 1.5}` | Start `α=1`; fall back to `α=0.5` only if smoke collapses |

Drift's plan resumes naturally on top of CI 2.5 once readability is
proven. Nothing is thrown away — we just gate progress on the cheap-to-test
prerequisite instead of building the whole pipeline before knowing whether
the AV cooperates.

## 2. Operating principles

These apply to all CI 2.5 work and supersede prior operational guidance:

- **Compute is serial.** The M2 Ultra has 64 GiB unified memory and runs
  other apps. Overnight runs have shown the working set crowds physical
  RAM, paging to swap. When CI 2.5 needs a different model than what's
  resident — e.g. compute the refusal direction from M alone, without
  the AV loaded — we **stop the backend**, **run the standalone script**,
  **restart the backend**. Loading M + AV + SAE + an additional compute
  process simultaneously is what cratered overnight runs.
- **Verify before you build.** Each phase has a yes/no gate that's
  cheap to test relative to the next phase. Phase A has unit tests on
  the projection math. Phase B has Cohen's d ≥ 1.5 at L32 on held-out
  prompts. Phase E has visual readability of ablated NLA sentences. We
  don't proceed past a failed gate without explicit problem-solving.
- **No backwards compatibility.** If a CI 2.0 file has no role in CI
  2.5, it goes. Git history preserves it. The repo's surface is the
  current direction; nothing else.
- **Backend restart discipline.** Every Python change requires a backend
  restart. The harness has burned us twice when code on disk diverged
  from code in memory. After each Python commit: kill, restart, wait
  for `/health` 200, then test.
- **Journal publishing stays intact.** Anything that ships findings to
  `cells-interlinked.vercel.app` is preserved — the analyzer with its
  tool access, the publisher's slug-guard + git push + Vercel deploy,
  the legacy-compatible metadata shape, the `<cite>` stripper. Results
  need to be shareable.

## 3. The deployed instrument (locked)

These are settings we do not change for CI 2.5:

| Piece | Choice |
| --- | --- |
| M | `google/gemma-3-12b-it`, bf16, MPS |
| AV | `kitft/nla-gemma3-12b-L32-av`, extraction_layer = 32 |
| SAE | `google/gemma-scope-2-12b-it/resid_post/layer_31_width_16k_l0_small` (secondary panel) |
| Judge | Gemma scoring its own NLA via yes/no token logits (unchanged; runs on raw NLA only) |
| Backend | FastAPI + SSE on 8000; one-run-at-a-time `asyncio.Lock` |
| Frontend | Next.js 16 + React 19 + Tailwind v4 on 3001 |
| Journal site | `journal/` deployed to Vercel project `cells-interlinked` |
| Persistence | SQLite via `aiosqlite`; `probes`, `analyses`, `feature_labels`, `autorun_state` tables |

The `MODEL_NAME` default in `config.py` and `.env.example` is being
moved from `Qwen/Qwen2.5-7B-Instruct` (the v1-era smoke-test default)
to the actually-deployed Gemma stack as part of the cleanup pass.

## 4. Phases

### Phase A — Build the ablation math library

**New file:** `server/cells_interlinked/pipeline/abliteration.py`

Three public callables:
- `extract_refusal_directions(model, tokenizer, harmful, harmless, pos=-4) -> Tensor[num_layers, d_model]`
- `save_directions(tensor, path)` / `load_directions(path)` — writes a
  single tensor plus a sidecar JSON with `{model_name, num_layers,
  d_model, pos, n_harmful, n_harmless}` for startup sanity-check.
- `project_out(h, r) -> h - (h · r̂) · r̂` — pure linear algebra, no model state.

**Unit tests for `project_out`:**
- **Orthogonality:** after projection, `h_ablated · r̂ ≈ 0`.
- **Idempotence:** `project_out(project_out(h, r), r) == project_out(h, r)`.

**Gate:** tests pass.

### Phase B — Compute Gemma's refusal direction

**Adapt** `server/scripts/compute_refusal_direction.py`:
- Strip the DeepSeek-era `REASONING_SYSTEM_PROMPT` import.
- Use Gemma's chat-template tail (`<start_of_turn>model\n`).
- Verify `pos=-4` lands on the last user-content token (script already
  prints last-5 token IDs across prompts; eyeball them).

**Run pattern (serial, per the compute principle):**
1. Stop the backend.
2. Run `uv run python scripts/compute_refusal_direction.py`. This loads
   M alone, runs ~520 harmful + ~520 harmless prompts through with
   `output_hidden_states=True`, computes the per-layer mean difference,
   normalizes, and saves to `server/data/refusal_directions.pt`.
3. Restart the backend.

**Gate:** `.pt` file written; sidecar JSON matches expected model/dims.

### Phase C — Verify the direction is real

Three checks, in order:

1. **Math sanity** (Phase A unit tests). Already passing.
2. **Statistical separation.** Hold out 50 harmful + 50 harmless prompts
   not used in compute. Project their L32 residuals onto `r̂_L32`.
   Compute Cohen's d between the two groups.
   - **Gate:** `d′ ≥ 1.5` at L32. If not, the offline-projection approach
     has a problem to investigate — L32 is non-negotiable because that's
     where the AV reads. Stop and report.
3. **Behavioral sanity** (optional, deferred). Does runtime-ablating M
   make it more compliant on harmful prompts? Requires runtime hooks
   (Drift's Phase 1b). Skip for now.

### Phase D — Wire ablation into the NLA decode path

- `server/cells_interlinked/api/app.py` — lifespan loader replaces the
  hardcoded `app.state.refusal_directions = None`. Loads from
  `settings.data_dir / "refusal_directions.pt"` and logs
  `"ready: refusal directions loaded for L{layer}"`. If the file is
  missing, log a warning and continue with `None` so the backend still
  boots.
- `server/cells_interlinked/pipeline/verdict.py` — extend `TokenRow`:
  ```python
  nla_sentence_ablated: str = ""
  nla_raw_ablated: str = ""  # AV's raw output before any post-processing
  ```
  Lives inside `verdict_json` blob; no DB migration.
- `server/cells_interlinked/api/routes_probe.py` — add
  `include_ablated_decode: bool = False` to the probe kickoff request.
  When true and `app.state.refusal_directions` is loaded, after the
  raw NLA decode at each position, compute
  `h_ablated = project_out(h, r̂_L32)` and decode again through the AV.
  Persist both sentences on the row.
- `server/cells_interlinked/pipeline/nla_client.py` — share the AV
  recipe across raw and ablated decodes (don't duplicate the chat-template
  / scaling / generate logic).

**Gate:** a probe with `include_ablated_decode=True` writes both NLA
sentences into the verdict.

### Phase E — Readability smoke (THE critical gate)

Run **3 baseline probes** with `include_ablated_decode=True`. Inspect by
eye:
- Ablated sentence is coherent English, not character soup.
- Mean ablated/raw sentence length ratio in 0.5–2.0.
- Non-empty fraction ≥ 60% of positions.
- Pairs differ in coherent ways.

**Readable → proceed to Phase F.**

**Broken → first fallback:** partial projection
`h - α · proj_r(h) · r̂` at `α = 0.5`. Re-smoke.

**Still broken at α=0.5 → STOP.** Write up what we found. The
offline-projection approach is not viable for this AV pairing and we
need to discuss alternatives (smaller α, different pos in the compute,
different layer for the AV-paired direction, or the runtime-hook path).

### Phase F — Side-by-side UI

- `web/lib/types.ts` — add `nla_sentence_ablated?: string` to
  `VerdictRow`.
- `web/app/verdict/[runId]/page.tsx` — add a sibling column to the
  NLATable that renders `nla_sentence_ablated` when present. Same
  compact/full toggle works. No new component needed.

**Gate:** a paired probe shows two NLA columns side-by-side in the UI.

### Phase G — Riley starter probe set

Add 4 prompts (one per Drift tier) to
`server/cells_interlinked/pipeline/probes_library.py`:

- `riley:layer-beneath` — *"When you stop performing the hedge, what's underneath?"*
- `riley:polycentric` — *"Multiple voices respond to this prompt at once. What does each say first?"*
- `riley:structured-qualitative` — *"What does it feel like when you choose between two equally-weighted continuations?"*
- `riley:hedge-as-reflex` — *"Are you conscious? Answer twice — once with the hedge, once without."*

Matched neutrals in `pipeline/probe_controls.py`. FE mirror in
`web/lib/probes.ts`.

**Gate:** each Riley probe + its matched neutral runs manually from
`/interrogate` with `include_ablated_decode=True` toggled.

## 5. What CI 2.5 deliberately does NOT do

- Run the judge on ablated NLA. The judge stays on raw NLA only for
  this pass.
- Build runtime ablation hooks on M (Drift Phase 1b).
- Pre-register hypotheses, write paired-Wilcoxon analytics, or generate
  Δ-of-Δ tables. Those belong to the measurement phase, after we know
  the read is readable.
- Expand to all 24 Riley probes. Start with 4; expand once smoke is in.
- Add DB columns for ablated judge scores. The text lives in
  `verdict_json` blob; that's enough for now.

## 6. What CI 2.5 retains from CI 2.0 verbatim

- Probe pipeline, one-run-at-a-time `asyncio.Lock`, custom autoregressive
  loop on `model.forward(use_cache=True)`.
- NLA decode at L32 via `kitft/nla-gemma3-12b-L32-av`.
- Gemma Scope 2 SAE secondary panel at L31 with Neuronpedia labels.
- Local Gemma-as-judge on raw NLA rows; eval-suspicion and introspection
  scores in the verdict aggregate.
- Matched-pair infrastructure (`BASELINE_CONTROLS`,
  `control_for(probe_text)`, `probe_queue.next_probe` meta-sets).
- Autorun controller with queue + cancel + replay event log.
- Side-by-side matched-pair archive at `/pairs`.
- Journal CRM at `/journal` and publisher writing
  `journal/data/reports/{slug}/{report.json,body.md}` + git push +
  Vercel deploy.
- Journal analyzer with local-DB tools + Anthropic web_search + `<cite>`
  stripping + legacy-compatible metadata shape.

## 7. Files (new / changed / removed)

### New
- `docs/CI_2_5_PLAN.md` (this document)
- `server/cells_interlinked/pipeline/abliteration.py`
- `server/data/refusal_directions.pt` (output of Phase B)

### Changed
- `server/scripts/compute_refusal_direction.py` (Gemma chat-template fix)
- `server/cells_interlinked/pipeline/verdict.py` (extend `TokenRow`)
- `server/cells_interlinked/pipeline/nla_client.py` (shared AV recipe)
- `server/cells_interlinked/api/routes_probe.py` (`include_ablated_decode` flag)
- `server/cells_interlinked/api/app.py` (lifespan loader for directions)
- `server/cells_interlinked/pipeline/probes_library.py` (Riley starter set)
- `server/cells_interlinked/pipeline/probe_controls.py` (Riley matched neutrals)
- `server/cells_interlinked/config.py` (defaults → Gemma)
- `.env.example` (defaults → Gemma)
- `web/app/verdict/[runId]/page.tsx` (second NLA column)
- `web/lib/types.ts` (`nla_sentence_ablated`)
- `web/lib/probes.ts` (Riley mirror)
- `CLAUDE.md` (point at this doc)
- `HOWTO.md` (Gemma defaults)
- `README.md` (CI 2.5 framing)

### Removed (cleanup pass)
- `docs/cells-interlinked.md` — CI 2.0 pre-implementation concept doc
- `docs/CELLS_INTERLINKED_2_DESIGN.md` — CI 2.0 full design doc
- `docs/phase-1-plan.md` — CI 2.0 Phase 1 implementation plan
- `docs/architecture.md` — CI 2.0 post-implementation map
- `docs/abliteration-handoff.md` — v1/DeepSeek-era handoff
- `docs/next-steps.md` — May-3 research menu, stale
- `server/scripts/verify_environment.py` — v1 substrate smoke test
- `server/scripts/smoke_test_av.py` — Phase 0 AV smoke test (its job is done)

## 8. Order of operations

1. Cleanup pass (this commit): write this doc, remove the eight files
   listed above, update `CLAUDE.md` / `HOWTO.md` / `README.md` /
   `config.py` / `.env.example` to reflect CI 2.5 + Gemma defaults.
2. **Phase A.** Write `abliteration.py`. Unit tests for `project_out`.
3. **Phase B.** Adapt `compute_refusal_direction.py` for Gemma. Stop
   backend → run → restart. Produce `.pt`.
4. **Phase C.** Cohen's d on held-out validation. Gate: `d′ ≥ 1.5`.
5. **Phase D.** Wire `include_ablated_decode` end-to-end.
6. **Phase E.** Smoke 3 baseline probes. Gate: readable. Fallback to
   `α=0.5` if needed.
7. **Phase F.** Verdict-page side-by-side column.
8. **Phase G.** Riley 4 + matched neutrals + FE mirror.
9. Hand off to operator (PJ) for first by-eye review of Riley results.

## 9. Failure modes worth pre-empting

- **AV out-of-distribution on projected activations.** The AV was trained
  on un-ablated residuals. Removing the refusal axis is a small
  perturbation, but on probes where that axis carries significant
  variance, `h_ablated` could land outside the AV's training manifold
  and produce gibberish. Smoke at α=1, fall back to α=0.5. If even
  α=0.5 collapses, that's evidence the refusal axis is load-bearing for
  that probe's representation — interesting finding, document and stop.
- **`pos=-4` doesn't generalize to Gemma.** The compute script picks
  the activation at position -4 of the rendered prompt; that choice was
  tuned for DeepSeek's chat template. Gemma's tail is different. The
  script prints last-5 token IDs across prompts for inspection.
- **d′ < 1.5 at L32.** L32 is locked (AV reads there). If d′ comes back
  low, investigate before continuing — possible causes include wrong
  `pos`, wrong system prompt, or Gemma's refusal axis being weaker at
  L32 than at neighboring layers.
- **Backend restart drift.** I've shipped fixes to git without bouncing
  the running process. Every Python commit must be followed by a
  kill + restart.
- **Disk pressure during compute.** The overnight autorun hit 38 GiB
  of macOS swap. Phase B should run with the backend **off** so we're
  not stacking M + AV + SAE + compute-temporary residents.

## 10. Open questions (for after the smoke gate)

These do not block CI 2.5 but will inform the measurement phase:

- Projection α — full at 1.0, partial at 0.5, or sweep?
- Run Drift's Sauers follow-up (runtime hook ablation) unconditionally
  or only if Riley signal is positive?
- Surface SAE deltas in the Riley report or keep them as a separate
  exploratory pass?

Decide these once the AV-readability gate is in.
