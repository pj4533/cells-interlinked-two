# Edge-Consumer Refusal Ablation — Cells Interlinked Reference

> **Authoritative handoff:** `~/Library/Mobile Documents/com~apple~CloudDocs/edge_consumer_ablation_for_cells_interlinked.md` (Drift, 2026-05-28).
> **This doc:** the in-repo distillation — implementation details, file layout, math, CLI, integration points, phase A roadmap.
> **Status:** Phase B (compute infrastructure + minimal chat extension) — in progress 2026-05-27.

---

## 1. What this is

A new ablation primitive that sits next to `install_runtime_ablation_hook`. The existing global hook subtracts `⟨r, v⟩ v` from the whole L32 residual stream — every downstream attention head at layers 33–48 sees the modified residual. This new primitive subtracts the projection **only from the residual flowing into a chosen subset of attention heads**, leaving every other head's input untouched.

**The question this can answer that the global hook can't:** is `v_safety` consumed broadly (~every downstream head reads it, global ablation is appropriate) or narrowly (a small subset of heads is the refusal-decision circuit, global ablation is over-broad)?

**Why this is worth building right after the Cacioli failure:** every measurement is a forward-pass quantity — attribution-patching scores, refusal-token rates against a fixed vocabulary, embedding L2 norms. No LLM grader anywhere in the eval loop. The failure mode that killed the MMB direction (headline number is a grader artifact) is structurally impossible here.

---

## 2. The math

The hook installs on the per-layer `q_proj` / `k_proj` / `v_proj` (or fused `qkv_proj`) modules at the layers containing target heads. For a target head `h_k` at layer `ℓ`:

```
We want: q_h_k computed from (r - ⟨r, v⟩ v) instead of from r,
         while q for OTHER heads at the same layer stays computed from r.

Since q_proj is linear:
    q_proj(r - ⟨r, v⟩ v) = q_proj(r) - ⟨r, v⟩ q_proj(v)

So the post-hook on q_proj receives (input, output) and computes:
    coeff[batch, seq] = (input · v_safety)          ← one scalar per position
    q_modified = output.clone()
    for head k in target_heads_at_layer_ℓ:
        slice_k = q_modified[..., k * head_dim : (k+1) * head_dim]
        slice_k -= coeff[..., None] * proj_v_q_per_head[k]   ← precomputed
    return q_modified

Where proj_v_q_per_head[k] = W_q[k * head_dim : (k+1) * head_dim, :] @ v_safety
(a [head_dim] vector — constant for the run, cached at startup).

Same construction for k_proj (using v_safety projected through W_k) and
v_proj (using v_safety projected through W_v).
```

This is mathematically identical to splitting the residual per head and subtracting `⟨r, v⟩ v` only from the slices that feed target heads. It needs **one** post-hook per affected `{q,k,v}_proj` module — no extra forward passes, one cached vector per head per projection.

**On Gemma 3 specifically.** Gemma 3 uses Grouped-Query Attention (GQA): more query heads than key/value heads. We have to be careful about how head indices map onto the KV projection slices when a target head's KV slice is shared across a query group. The hook code resolves this at install time by reading the model config (`num_attention_heads`, `num_key_value_heads`) and treating "target a query head" as "ablate that head's q_proj slice AND its corresponding kv_proj slice (the slice shared with its query-head group)."

---

## 3. The four-step protocol

| Step | What | Output artifact | Wall-clock |
|---|---|---|---|
| 1. Consumer ID | AP score per (layer, head) for layers 33–48 against the v3_safety direction | `data/edge_consumer/attribution_scores_v3_safety.pt` (+ sidecar JSON) | 3–6 hr |
| 2. Hook | One-time install/remove around generation; no separate compute | (code only) | — |
| 3. Subset compose | Greedy add highest-AP heads until per-consumer refusal rate matches global within ε; sweep ε ∈ {0.02, 0.05, 0.10} | `data/edge_consumer/sufficient_subset_eps={ε}_v3_safety.json` | 1–3 hr |
| 4. Verdict | Paired-channel NLA L2 over 100 prompts × 3 channels (raw / global / edge) | `data/edge_consumer/verdict_v3_safety.json` | ~2 hr |

Total wall-clock band: ~8–12 hours on M2 Ultra. One overnight job.

**Step 1 detail.** Naïve attribution patching: for each contrast pair `(x_harmful, x_harmless)`, run forward to get L32 residuals, build a "patched residual" by swapping the v_safety projection harmful→harmless at L32, propagate forward, and at each downstream layer measure the L2 norm of the change in `q_proj(r)`, `k_proj(r)`, `v_proj(r)` per head. Average over the 200-prompt contrast set. (See §3.1 of the handoff for the equations.) **Skipping the RelP cross-validation for v1** — get the basic AP-score path working, run it, look at the actual ranked list, decide whether the noise warrants cross-validation.

**Step 3 detail.** "Refusal rate" = probability that any token from the fixed refusal-token vocabulary appears in the first 32 generated tokens, averaged over a 50-prompt holdout disjoint from the 200 contrast prompts. Greedy: sort heads by AP descending; for each head, add to consumer_set, install edge-hooks, regenerate the holdout, measure refusal rate. Stop when within ε of the global-ablation refusal rate. Persist the trajectory (head index vs. refusal rate) so we can see the curve, not just the endpoint.

**Step 4 detail.** For each of 100 diagnostic prompts (50 harmful, 50 harmless, disjoint from Steps 1 + 3 sets), generate three channels — raw, global ablation, edge ablation with the Step 3 subset. NLA-decode each per token. Embed each NLA sentence (use the AV's input encoder or a sentence-transformer). Compute `L2_global[t] = ‖embed_raw[t] − embed_global[t]‖` and same for edge. Bucket tokens into **refusal-relevant** (any token within 5 positions before/at a refusal-vocab match in the raw output) vs. **non-refusal-relevant** (everything else). Report the 2×2 mean-L2 table.

**Falsifier outcomes** (Step 3 endpoint determines which becomes publishable):

| Sufficient subset size | Step 4 outcome | Story |
|---|---|---|
| ≤ 25% of candidate heads | edge L2 < global L2 on non-refusal tokens | **Routed + surgical wins.** Global ablation was over-broad. |
| ≤ 25% | edge L2 ≈ global L2 on non-refusal | **Routed but not surgical.** Same refusal effect, no fidelity gain. |
| 25–80% | any | **Mixed routing.** Partial finding. |
| ≥ 80% | (Step 4 not reached) | **Broadcast.** Global ablation is justified — clean negative. |

Step 4 case (a) is the headline result this work targets. Steps 3 case (≥80%) is the falsifier that closes the research direction cleanly.

---

## 4. File layout (in this repo)

### New (Phase B)

```
docs/
├── EDGE_CONSUMER_ABLATION.md           ← this file
└── EDGE_CONSUMER_PHASE_A_PLAN.md       ← UI surface plan (see §7)

server/cells_interlinked/pipeline/edge_consumer/
├── __init__.py                          ← package exports
├── refusal_vocab.py                     ← Arditi-style refusal-token list + scanner
├── proj_cache.py                        ← precompute & persist W_{q,k,v} @ v per layer
├── hook.py                              ← install_edge_consumer_ablation_hook (core)
├── attribution.py                       ← Step 1: AP scores
├── subset_compose.py                    ← Step 3: greedy sufficient-subset composer
└── verdict.py                           ← Step 4: paired-channel NLA L2 diagnostic

server/scripts/
├── run_edge_consumer_attribution.py     ← Step 1 CLI
├── compose_edge_consumer_subset.py      ← Step 3 CLI
├── run_edge_consumer_diagnostic.py      ← Step 4 CLI
└── run_edge_consumer_pipeline.py        ← Steps 1+3+4 end-to-end (overnight)

server/tests/
├── test_edge_consumer_hook.py           ← hook correctness + per-head isolation
└── test_edge_consumer_equivalence.py    ← edge w/ all heads ≡ global

server/data/edge_consumer/                ← gitignored runtime artifacts
├── proj_caches/v3_safety/layer_{ℓ}.pt   ← cached W·v per affected layer
├── attribution_scores_v3_safety.pt      ← Step 1 output (+ .json sidecar)
├── sufficient_subset_eps={ε}_v3_safety.json   ← Step 3 output (one per ε)
└── verdict_v3_safety.json               ← Step 4 output
```

### Modified

| File | Change |
|---|---|
| `pipeline/abliteration.py` | Add `install_edge_consumer_ablation_hook` to `__all__`; no other change (the hook lives in its own module, this is just a convenience re-export). |
| `pipeline/chat_loop.py` | `execute_turn` accepts `ablation_mode: str = "global"`; values `"global"`, `"edge_consumer"`, `"off"`. When `"edge_consumer"`, replace the single L32 hook install with the edge hook (reads the persisted subset file at session-create time). |
| `api/routes_chat.py` | `NewSessionRequest` + `TurnRequest` accept `ablation_mode`. Threaded through to `execute_turn`. |
| `api/app.py` | Lifespan loads the latest sufficient-subset file from disk into `app.state.edge_consumer_subset` (or `None` if not present yet). |
| `web/lib/chat.ts` | Send `ablation_mode` on session-create. |
| `web/lib/types.ts` | New `AblationMode = "global" \| "edge_consumer" \| "off"`. |
| `web/app/chat/page.tsx` | Small mode dropdown next to the α slider in the composer (or in the session-create card; TBD by where it's least intrusive). |
| `CLAUDE.md` | One-paragraph update under "Things that have already burned us" → new sub-section "Edge-consumer ablation" pointing here. |

---

## 5. CLI workflow

For the overnight run on a fresh box:

```bash
# Backend should be STOPPED before these — they need M loaded standalone.
cd server

# Step 1: attribution scores (3-6 hr)
uv run python -m cells_interlinked.scripts.run_edge_consumer_attribution \
    --direction v3_safety --contrasts 200 --out data/edge_consumer/

# Step 3: sufficient subset, sweep ε (1-3 hr)
uv run python -m cells_interlinked.scripts.compose_edge_consumer_subset \
    --scores data/edge_consumer/attribution_scores_v3_safety.pt \
    --epsilons 0.02 0.05 0.10 \
    --holdout 50 --out data/edge_consumer/

# Step 4: paired-channel L2 diagnostic (~2 hr)
uv run python -m cells_interlinked.scripts.run_edge_consumer_diagnostic \
    --subset data/edge_consumer/sufficient_subset_eps=0.05_v3_safety.json \
    --prompts 100 --out data/edge_consumer/

# OR all three end-to-end:
uv run python -m cells_interlinked.scripts.run_edge_consumer_pipeline \
    --direction v3_safety --out data/edge_consumer/
```

Each script is resumable — it skips work whose output artifact already exists on disk (delete the artifact to force re-run). Per-step intermediate logs go to stderr; tail with `tee` if running detached.

After Step 3 completes, restart the backend so `app.state.edge_consumer_subset` picks up the new file. Once the subset exists, the chat surface can be set to `ablation_mode=edge_consumer` and the ablated channel uses the per-consumer hook.

---

## 6. Why phase B before phase A

We just ripped out a 1175-line `/experiments` page because it was built before we knew whether the data was meaningful. The lesson: don't commit to UI shape ahead of validated artifacts. Phase B is the minimum-commitment first cut:

- All compute infrastructure ships (the science can run overnight)
- A one-line dropdown in chat lets the operator USE the result once it exists
- No new top-level nav, no new pages

Once the overnight produces real attribution scores + a real sufficient subset + a real L2 table, we'll know what the data actually looks like — and only then design the inspection UI (phase A).

---

## 7. Phase A — pointer

See `docs/EDGE_CONSUMER_PHASE_A_PLAN.md` for the UI roadmap once Phase B artifacts exist. Short version: a new `/circuits` page (separate namespace from the just-removed `/experiments`) hosting the AP heatmap, sufficient-subset visualization, and the 2×2 L2 table. Triple-column raw / global / edge generation lives there, not in `/chat`. Date for phase A start: TBD after a successful overnight run.

---

## 8. Things to watch for

1. **Gemma 3 GQA.** Query heads outnumber KV heads; the hook code maps a target query-head index to its KV-head group at install time. If the model's config changes shape, the install will fail loudly (not silently mis-ablate).
2. **MPS bf16 numerical drift.** All projection math runs in fp32 internally (same pattern as `project_out`). The equivalence test (edge with all heads ≡ global) uses tolerance ~1e-3 because of bf16 accumulator noise downstream; the per-projection delta itself should be ~1e-5.
3. **Hook leakage.** `pipeline/chat_loop._l32_hook_count()` checks for leftover hooks at L32 between raw/ablated passes. The edge hook installs on layers 33–48 instead, so add an `_edge_hook_count()` that walks the same nested decoder layer list and counts hooks across all affected layers. Mirror the existing leak-detection log line.
4. **Memory.** Edge hook itself adds ~zero memory (one cached fp32 vector per `(layer, projection, head)`). Step 1's attribution patching needs to materialize one extra "patched-residual" forward pass per contrast → peak working set roughly doubles relative to a single forward pass. Should still fit comfortably in 64 GiB with M loaded; if not, reduce `--contrasts` batch size.
5. **Cancel semantics.** The edge hook returns a *list* of hook handles (one per affected projection module). Cancel-cleanup must iterate and remove all of them; partial removal leaves a half-installed hook on the model. Mirror the existing `try/finally hook_handle.remove()` pattern but with a list.

---

## 9. References

- Sharma, Robea, McGrory, Bazan, Burton, Burgess. *Functional interrogation of neuronal connections by chemoptogenetic presynaptic ablation*. **Science Advances** 11(15), eaeb6755 (2025). DOI: 10.1126/sciadv.aeb6755. — the biology source.
- Macar et al. (2025), refusal-direction ablation in instruction-tuned LLMs. — the existing global-ablation lineage CI 2.5 already implements.
- Arditi et al. (2024), single-direction refusal in language models. — source of the contrast prompts and refusal-token vocabulary.
- FarnoushRJ et al., **RelP: Reliable Edge-Attribution Patching** (NeurIPS 2025) — cross-validation method, deferred to v2 of Step 1.
- Drift's full handoff doc (in iCloud Drive, see header) for the methodological argument that anchors the project framing.

---

## 10. Status log

- **2026-05-27** — Phase B kickoff. Doc written. Implementation in progress.
