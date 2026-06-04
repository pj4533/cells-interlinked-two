# Gemma 4 12B — migration assessment & handoff

**Status: NOT started. Decision (2026-06-04): stay on gemma-3-12b for now.** This
doc captures the research + a concrete plan so it's ready to pick up as a possible
next step. Nothing in the codebase has changed.

## Context

`google/gemma-4-12b-it` shipped 2026-06-03 (Apache 2.0). The question: could it
replace `gemma-3-12b-it` as our **M** for the non-NLA instruments — **off-manifold
autoresearch, DMT autoresearch, the Trip View, and ablated/dosed chat** — while
the NLA verdict pipeline stays on gemma-3? Short answer: **feasible, but it's a
port + full recompute, not a drop-in swap.** Do it as a validate-first spike on a
branch before committing.

## What Gemma 4 12B actually is (confirmed from config.json + HF + transformers docs)

| | gemma-3-12b-it (current M) | gemma-4-12b-it |
| --- | --- | --- |
| layers | 48 | **48** (identical) |
| hidden_size | 3840 | **3840** (identical) |
| vocab | 262144 | 262144 |
| dtype | bf16 | bf16 (~24 GB) |
| context | (gemma-3) | 131072 in `text_config` (256k marketed) |
| transformers model_type | `gemma3` | `gemma4_unified` |
| class | `Gemma3ForConditionalGeneration` | `Gemma4UnifiedForConditionalGeneration` |
| min transformers | 5.7 (we pin this) | **≥ 5.10** |
| license | Gemma | Apache 2.0 |

Other facts: encoder-free multimodal (text/image/video/native audio in one
decoder — no separate ViT), **Per-Layer Embeddings (PLE)** (auxiliary signals
injected into every decoder layer), optional **Multi-Token-Prediction (MTP)**
drafter for speculative decoding, and a community **abliterated variant**
(`OpenYourMind/gemma-4-12B-it-abliterated-uncensored`) already exists — i.e.
refusal-direction extraction on gemma-4 is demonstrated.

## Feasibility

### The lucky part
**48 layers / hidden 3840 are identical to gemma-3-12b**, and bf16 (~24 GB) fits
the 64 GB box for M-only work exactly as today. So our direction tensors and atlas
vectors are *dimensionally* compatible — no reshaping.

### The trap
Dimensional compatibility ≠ semantic compatibility. A gemma-3 refusal/emotion/
steering vector will **load and run** on gemma-4 without a shape error, but
gemma-4's activation space is a different geometry — that vector steers into noise.
**Every precomputed artifact must be recomputed**, and because nothing crashes, a
careless "swap" would silently produce garbage. Treat identical dims as a
convenience for the code, not a shortcut on the science.

### The catches (why it's not a drop-in)

1. **NLA is out, and that forks the model story.** The kitft AV
   (`kitft/nla-gemma3-12b-L32-av`) decodes *gemma-3's* L32 residual; it's
   meaningless for gemma-4 and no gemma-4 AV exists. So the verdict page / judge /
   synthesis stay on gemma-3. Two ~24 GB models can't co-reside on 64 GB, so this
   becomes a **model-swap** problem (the `ModelManager` would now juggle gemma-3 ⇄
   gemma-4 in addition to M ⇄ AV), or NLA is disabled whenever gemma-4 is M.
   **This is the biggest architectural decision and should be settled first.**

2. **Recompute everything (gemma-3-specific → invalid):**
   - refusal vectors `data/refusal_directions_v{1..6}.pt` + subspaces
     (`scripts/compute_refusal_direction.py` / `pipeline/abliteration.py`).
   - `data/emotion_directions.pt` (+ `.json`) and `data/valence_direction.pt`.
   - both atlases: `data/atlas/` (off-manifold) and `data/atlas_dmt/` — **rebuilt
     from scratch** (the current 16-direction off-manifold atlas and the DMT run
     are gemma-3 artifacts; the exported `research-*` / `dmt-*` palette entries
     too).

3. **Re-tune the layer choices.** `STEER_LAYER = 20` and extraction `L32` were
   found empirically *for gemma-3* (the "inject at L20 propagates best" sweep, see
   `docs/MANIFOLD_ABLATION.md`). Re-run that sweep on gemma-4 — layer 20 of gemma-4
   ≠ layer 20 of gemma-3 functionally. The off-manifold trajectory geometry
   (`pipeline/trajectory.py`) reads `bundle.extraction_layer`, so that constant
   moves too.

4. **Architecture / hook re-port + validation:**
   - The 12B is the **`gemma4_unified`** class — distinct from the smaller e2b/e4b
     `Gemma4ForConditionalGeneration`. Our forward-hook layer traversal
     (`_find_decoder_layers` in `pipeline/abliteration.py`, `_find_layers` in
     `pipeline/generation_loop.py`, currently `.model.language_model.layers` per
     CLAUDE.md invariant #7) must be re-derived for the unified class.
   - **Per-Layer Embeddings:** gemma-4 injects auxiliary signals into each layer.
     The residual stream is still hookable, but the "residual stream as a clean
     transition operator" framing that the Trip View + off-manifold geometry lean
     on is muddier — **validate** that adding/subtracting at a layer output
     (steering / ablation) behaves the way it does on gemma-3.
   - **MTP:** optional accelerator; ignore it. Our custom autoregressive loop
     (`generation_loop.py`, `model.forward(use_cache=True)` step-by-step) is
     unaffected as long as we don't opt into the drafter.

5. **transformers 5.7 → ≥5.10 upgrade.** Risks the existing gemma-3 path. In
   particular, re-verify the **raw-tokenizer workaround** (CLAUDE.md invariant #6 —
   the transformers 5.7 wrapper mis-tokenizes Gemma; we load `tokenizers.Tokenizer`
   from `tokenizer.json` directly). Confirm whether 5.10 fixes or changes that for
   both gemma-3 and gemma-4. Pin carefully; consider a separate uv env for the
   spike so the working gemma-3 backend isn't disturbed.

## The spike (validate before committing — ~an afternoon)

Goal: answer "does dose→coherent altered self-report hold on gemma-4, and is it
*richer* than gemma-3?" before paying for the full recompute. On a branch, in an
isolated env:

1. `transformers >= 5.10`, `hf download google/gemma-4-12b-it`.
2. Load via `Gemma4UnifiedForConditionalGeneration`; confirm the decoder-layer
   module path and update `_find_decoder_layers` / `_find_layers` for it.
3. Smoke the custom forward loop + residual capture (one prompt, capture L32) —
   confirm hooks fire and shapes are `[1, 3840]`.
4. Extract ONE valence (or emotion) direction on gemma-4 (diff-of-means, same
   method as today).
5. Run the **Trip dose at a few layers** (sweep ~L16/L20/L24/L28). Two questions:
   (a) does a coherent altered self-report appear (steering works through PLE)?
   (b) is it richer/more vivid than the gemma-3 equivalent? Note the best layer.

If yes → proceed to the full port. If no → you've spent an afternoon, not a
rewrite.

## Full port checklist (after a good spike)

- [ ] Settle the **NLA model-swap** design (ModelManager gemma-3 ⇄ gemma-4, or
      NLA-off when gemma-4 is M). Update `pipeline/model_manager.py`,
      `api/app.py` lifespan.
- [ ] `config.py`: model id, and any layer constants.
- [ ] `pipeline/model_loader.py`: load the `gemma4_unified` class; re-verify the
      raw-tokenizer path under transformers 5.10; `extraction_layer`, `eos_ids`.
- [ ] `pipeline/abliteration.py` + `generation_loop.py`: hook traversal for the
      unified class; validate steering/ablation under PLE.
- [ ] Re-run the **steer/extract layer sweep**; set `STEER_LAYER` +
      `extraction_layer` (autoresearch_base.py, config, trajectory.py).
- [ ] Recompute refusal vectors + subspaces; `emotion_directions`;
      `valence_direction`.
- [ ] Rebuild both atlases from scratch (off-manifold + DMT); re-export palettes.
- [ ] UI labels that say "Gemma-3" (verdict caveats, fine-print, footers).
- [ ] Decide gemma-3 vs gemma-4 default; keep gemma-3 reachable for NLA.

## Open questions to resolve during the spike

- Exact decoder-layer module path on `Gemma4UnifiedForConditionalGeneration`
  (docs describe the e2b/e4b `Gemma4ForConditionalGeneration` nesting as
  `model.language_model.model.layers` — the unified 12B may differ).
- Does PLE change how a steering add at a layer propagates (does the "early-layer
  injection propagates best" result still hold, and at which layer)?
- Does transformers 5.10 keep the gemma-3 backend working (tokenizer + the
  multimodal-wrapper hook path)?
- bf16 on-disk size + load time for gemma-4-12b on the M2 Ultra (expect ~24 GB,
  similar to gemma-3).

## Sources

- Google blog — Gemma 4: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
- HF model card: https://huggingface.co/google/gemma-4-12B-it
- config.json: https://huggingface.co/google/gemma-4-12B-it/resolve/main/config.json
- transformers Gemma4 docs: https://huggingface.co/docs/transformers/model_doc/gemma4
- MarkTechPost (encoder-free + native audio): https://www.marktechpost.com/2026/06/03/google-deepmind-releases-gemma-4-12b-an-encoder-free-multimodal-model-with-native-audio-that-runs-on-a-16-gb-laptop/
- Abliterated variant (refusal-ablation demonstrated): https://huggingface.co/OpenYourMind/gemma-4-12B-it-abliterated-uncensored
