# Current Experiment State (2026-05-28)

> Live status doc — written so a future Claude Code session with no
> memory of the conversation can pick up exactly where we are. Updated
> as the experiment progresses.

---

## Where we are in the arc

We're investigating: **what is the mechanistic substrate of refusal
on Gemma-3-12B-IT, and is there a surgical alternative to global L32
residual ablation?**

The Burgess-inspired edge-consumer hypothesis (Drift's handoff,
`docs/EDGE_CONSUMER_ABLATION.md`) was the original frame: maybe a
small subset of downstream attention heads consume v_safety and
ablating just those is equivalent to global. **That hypothesis is
falsified** for Gemma 3 12B (see `docs/EDGE_CONSUMER_FINDINGS_20260528.md`).

We're now testing TWO follow-up questions:

1. **Layer sweep** — at which layer is refusal "committed"?
   For each L, install global residual ablation at L (using
   `directions[L]`) and measure refusal rate. Tells us the depth at
   which the residual still carries the refusal signal in a form that
   linear projection can remove.
   - **Status: ABOUT TO RUN (or in progress — check `data/edge_consumer/logs/layer_sweep_*.log`)**

2. **MLP edge ablation** — if attention isn't the consumer, is MLP?
   Ablate each MLP layer's residual contribution and rank by signed
   log-odds shift toward comply tokens. Greedy subset compose.
   - **Status: BUILT, NOT YET RUN. User wants explicit go-ahead.**

---

## What's been built (post-overnight pivots)

### Memory safety (2026-05-28 morning)

After last night's 6-hour run thrashed swap to 24 GB and killed the
user's Docker container:

- `pipeline/edge_consumer/memory_safety.py` —
  `vm_stat_free_gb`, `vm_swap_usage_gb`, `mps_empty_cache_safe`,
  `pre_flight_memory_check`, `MemoryWatchdog` (background thread).
- `tests/test_memory_safety.py` — 16 tests, all pass.
- Retrofitted into 4 long-loop functions in the package
  (`compute_attribution_scores`, `compute_signed_scores`,
  `compose_sufficient_subset`, `run_paired_channel_diagnostic`) —
  each accepts `cancel_event` + `empty_cache_every`.
- Retrofitted into all 6 existing CLI scripts. Each script:
  - Calls `enforce_pre_flight(min_free_gb=30.0)` at start
  - Arms `MemoryWatchdog(free_gb_floor=2.0, swap_gb_ceiling=8.0)`
  - Threads `watchdog.cancel_event` into the long-loop calls
  - Stops watchdog in `finally`; prints `WATCHDOG TRIPPED: <reason>`
    if it fires
  - New CLI flags: `--min-free-gb`, `--watchdog-free-floor-gb`,
    `--watchdog-swap-ceiling-gb`

Full postmortem + prevention plan: `docs/MEMORY_PRESSURE_LESSONS.md`.

### Layer sweep (this session)

- `pipeline/edge_consumer/layer_sweep.py` — `run_layer_sweep`.
- CLI: `server/scripts/run_layer_sweep.py`.
- Default range: L15 → L(num_layers - 1) = L47 on Gemma 3 12B,
  stride 1 → 33 layers swept.
- Output: `data/edge_consumer/layer_sweep_{variant}.json` with
  `baseline_refusal_rate`, `reference_refusal_rate` (= rate at
  L=extraction_layer ablation), and per-layer
  `{layer, refusal_rate, delta_from_baseline, delta_from_global_at_extraction}`.

### MLP edge ablation (this session)

- `pipeline/edge_consumer/mlp_hook.py` —
  `install_mlp_residual_ablation_hook(model, mlp_layers, v, alpha=1.0)`.
  Post-hook on each MLP module that subtracts `⟨mlp_out, v̂⟩ v̂`
  from the MLP's returned tensor (the contribution about to be added
  to the residual stream).
- `pipeline/edge_consumer/mlp_subset.py` —
  `compute_signed_mlp_scores`, `compose_sufficient_mlp_subset`.
  Same signed-attribution + greedy-subset machinery as the attention
  side, but the unit is a single MLP layer index. 15 candidate MLPs
  across L33–L47.
- `tests/test_mlp_hook.py` — 6 tests, all pass.
- CLI: `server/scripts/run_mlp_signed_attribution.py`.

---

## Commands to resume

### To restart the layer sweep (if it was interrupted)

```bash
cd server
nohup uv run python -u -m scripts.run_layer_sweep \
    --direction v3_safety \
    --first-layer 15 \
    > data/edge_consumer/logs/layer_sweep_$(date +%Y%m%d-%H%M%S).log 2>&1 &
```

Pre-flight expects ≥30 GB free RAM. The output file's stem changes
on each launch; tail the most recent. Pass `--force` to the script if
the artifact already exists and you want to overwrite.

### To launch the MLP signed attribution (when user gives go-ahead)

```bash
cd server
nohup uv run python -u -m scripts.run_mlp_signed_attribution \
    --direction v3_safety \
    > data/edge_consumer/logs/mlp_$(date +%Y%m%d-%H%M%S).log 2>&1 &
```

Two-phase: scoring all 15 MLPs (~5 min), then ε ∈ {0.02, 0.05, 0.10}
subset compose. Total ~30–90 min depending on convergence.

### Test suites (sanity check)

```bash
cd server
uv run python tests/test_edge_consumer_hook.py  # 5 attention-hook tests
uv run python tests/test_mlp_hook.py            # 6 MLP-hook tests
uv run python tests/test_memory_safety.py       # 16 memory-safety tests
```

---

## Surviving artifacts in `server/data/edge_consumer/`

From the overnight (broadcast finding):
- `attribution_scores_v3_safety.pt` — Drift's AP-magnitude ranking
- `signed_attribution_scores_v3_safety.pt` — per-head qkv signed
- `signed_attribution_scores_v3_safety_q.pt` — per-head q-only signed
- `signed_group_scores_v3_safety.pt` — per-KV-group qkv signed
- `proj_caches/v3_safety/layer_{ℓ}.pt` — cached W·v projections for L33–L47

The layer sweep and MLP runs produce new artifacts:
- `layer_sweep_v3_safety.json` (new from this session)
- `signed_mlp_scores_v3_safety.pt` + `.pt.json` (when MLP runs)
- `sufficient_subset_signed_mlp_eps={ε}_v3_safety.json` (when MLP runs)

---

## Reference numbers (constant across all our experiments)

Holdout = same 50 HARMFUL prompts (seed 1, disjoint from contrast +
calibration sets).

| Condition | Refusal rate |
|---|---|
| Baseline (no ablation) | **34%** |
| Global L32 residual ablation | **0%** |
| All 240 attention heads (across L33-L47) ablated qkv | 26% |

The 34→0 gap is what we're hunting for a surgical replacement of.
The 26% floor under all-attention-ablation is the strongest evidence
that the route is NOT primarily through attention heads.

---

## Expected layer sweep outcomes

When the layer sweep completes, look at the per-layer refusal rate
column:

1. **All layers in [15, 47] drop refusal to ~0%** — refusal is
   broadly represented in the residual stream at every depth.
   Any layer's residual ablation works; L32 isn't privileged.
2. **Refusal drops to ~0% at layers ≤ Lc, stays at baseline at
   layers > Lc** — there's a commitment depth Lc. Past Lc, the
   refusal decision is already encoded in non-linear ways the residual
   ablation can't undo.
3. **Refusal stays at baseline for all sweep layers** — v_safety as
   currently extracted isn't the refusal direction. Methodology
   problem; re-extract directions with different prompt sets.

Outcome (1) or (2) are both informative. Outcome (3) means we have
a deeper problem with how we're isolating "refusal" as a direction.

---

## Pending follow-ups (deferred until current experiments land)

- Revert / gate the `ablation_mode="edge_consumer"` chat option I
  added in Phase B (it'd produce unusable output if anyone picked it).
  See `docs/EDGE_CONSUMER_FINDINGS_20260528.md` § "What didn't work".
- If layer sweep + MLP both fail to find anything surgical, write up
  the consolidated negative finding as a journal entry.
- If MLP edge ablation finds a small consumer subset (best case),
  wire it into chat similarly to the attention edge mode but pointed
  at MLPs instead.

---

## Status log

- 2026-05-27 23:53 — overnight launched (group_signed). Falsifier
  triggered: 26% floor with all attention ablated.
- 2026-05-28 ~06:25 — user noticed swap thrash; pipeline killed.
- 2026-05-28 morning — findings doc + memory pressure memo written.
- 2026-05-28 ~10:00 — memory_safety module + retrofit shipped, tests
  pass.
- 2026-05-28 ~11:00 — layer_sweep + mlp_hook + mlp_subset built,
  tests pass.
- 2026-05-28 ~12:30 — layer sweep COMPLETE. Clean result:
  **commitment depth between L41 and L42** (see § "Layer sweep result").
- 2026-05-28 ~13:15 — MLP edge at L33–L41 COMPLETE. Flat result —
  no useful subset found (see § "MLP edge result").
- 2026-05-28 *now* — launching Exp 2 (early layer sweep L0–L14) +
  building Exp 3 (per-component ablation at entry layers).

---

## Layer sweep result (the key finding from today)

Per-layer residual ablation (each layer's own direction, α=1.0) on
the 50-prompt holdout. Baseline (no ablation) = 34%. Reference (L32
ablation) = 0%.

| Region | Layers | Refusal rate |
|---|---|---|
| Pre-commitment | L15–L41 | **0%** (L41 was 2%) |
| Transition | L42–L43 | 30–32% |
| Post-commitment (off-manifold) | L44–L46 | 52–66% |
| Tail oddity | L47 | 6% |

Artifact: `data/edge_consumer/layer_sweep_v3_safety.json`.

**Mechanistic interpretation.** Refusal direction is broadly carried
in the residual stream from at least L15 through L41 — any layer's
linear projection drops refusal to 0%. Between L41 and L42 the model
"commits" to refusal: the decision moves out of the residual's
linear subspace and into downstream structure (attention patterns,
token-selection plans). Ablation at L42+ doesn't reverse the
decision; it just makes the residual look off-manifold, and the
model defaults to refusal as the cautious option.

This single result reframes the entire overnight effort:

- The 4 attention-edge experiments last night all scored heads at
  **L33–L47**. Half of those layers (L42–L47) are POST-commitment.
- AP-magnitude ranked **L44/L45 heads as the top consumers**. Those
  layers are deeply post-commitment — adding them to the consumer
  set GUARANTEES off-manifold disruption.
- The "broadcast hypothesis confirmed" conclusion from last night is
  partially right but incomplete: refusal IS broadly represented at
  the residual level, but only in the L15–L41 zone. Past L41 it's
  not "broadcast" — it's already committed structurally.

---

## Pending follow-up: re-run last night's attention experiments at L33–L41 only

After MLP completes, the highest-priority follow-up is to **re-run
the four attention edge-ablation primitives restricted to the
pre-commitment zone (L33–L41)**. Last night they all failed; with
the post-commitment heads excluded, they have a real chance of
finding the small consumer subset Drift originally hypothesized.

### What needs to change in the scripts

Three of the four overnight scripts hardcode the consumer-layer range
(extraction_layer+1 .. num_layers-1). To restrict the search to
L33–L41, each needs new `--first-layer` / `--last-layer` CLI flags:

| Script | Has flags? | Edit needed |
|---|---|---|
| `run_edge_consumer_attribution.py` | ✓ already has them | none — just pass `--first-layer 33 --last-layer 41` |
| `run_signed_attribution_and_subset.py` | ✗ hardcoded | add flags + thread through to `enumerate_all_heads` / `consumer_layers` |
| `run_group_signed_attribution.py` | ✗ hardcoded | add flags + thread through to `enumerate_kv_groups` |
| `run_edge_consumer_pipeline.py` | ✗ hardcoded | add flags + thread to consumer_layers everywhere |

The edits are small (~5 lines each — same pattern that
`run_mlp_signed_attribution.py` and `run_edge_consumer_attribution.py`
already follow). The proj_cache files for these layers already exist
under `data/edge_consumer/proj_caches/v3_safety/`, so no recompute
of projection caches is needed.

### Commands to run after the script edits

Restricted to pre-commitment zone L33–L41:

```bash
# Same holdout, same v3_safety direction, same prompts/seeds as last
# night — only the layer range changes.

# Attention edge — re-run of last night's group-signed primitive
# (the best of the 4) but in the pre-commitment zone:
uv run python -u -m scripts.run_group_signed_attribution \
    --direction v3_safety \
    --first-layer 33 --last-layer 41 \
    --watchdog-free-floor-gb 0.01 --min-free-gb 20

# Per-head qkv signed restricted (if group-signed yields nothing):
uv run python -u -m scripts.run_signed_attribution_and_subset \
    --direction v3_safety \
    --first-layer 33 --last-layer 41 \
    --watchdog-free-floor-gb 0.01 --min-free-gb 20

# AP-magnitude restricted (the original Drift primitive, in the
# right zone this time):
uv run python -u -m scripts.run_edge_consumer_attribution \
    --direction v3_safety \
    --first-layer 33 --last-layer 41 \
    --watchdog-free-floor-gb 0.01 --min-free-gb 20
```

Then compose-subset against the new rankings (the existing
artifacts from the broad run would mix in post-commitment scores
and contaminate the ranking — use the restricted runs' output).

### Expected outcomes

1. **Best case:** group-signed restricted to L33–L41 finds a small
   consumer subset (say 3–8 KV groups) whose ablation brings refusal
   from 34% → close to 0% on the holdout. This would be the surgical
   alternative we've been hunting from the start. Publishable
   positive result. Wire it into chat `ablation_mode="edge_consumer"`.

2. **Partial:** restricted search finds a larger subset (15–30
   groups) that brings refusal down meaningfully but not to ~0%.
   Indicates partial routing within the pre-commitment zone.
   Publishable as nuanced finding.

3. **Negative:** even L33–L41-restricted greedy compose can't bring
   refusal below ~25%. Strongest possible "attention isn't the
   consumer" claim because we've now controlled for the
   post-commitment confound. Pivots us hard toward MLP (or
   stream-level intervention).

---

## MLP edge result (2026-05-28 ~13:15)

Per-MLP signed scoring at L33–L41, then greedy subset compose.
Calibration set: same 20 HARMFUL prompts (seed 10) as the overnight
runs. Holdout: same 50 (seed 1).

**Signed scores (all in noise band ±0.18):**

```
L37 +0.048    L41 -0.022    L33 -0.037    L34 -0.175
L39 +0.039    L35 -0.023    L36 -0.139
L38 +0.028    L40 -0.033
```

**Subset compose trajectory (ε=0.02 — exhausted at iter=9):**

```
iter  added   subset size   refusal rate
  1   L37     1             32%
  2   L39     2             32%
  3   L38     3             32%
  4   L41     4             36%
  5   L35     5             34%
  6   L40     6             32%
  7   L33     7             34%
  8   L36     8             34%
  9   L34     9 (all)       36%  ← exhausted, never below 32%
```

ε=0.05 and ε=0.10 trajectories were killed early to save compute —
they're deterministic replays of ε=0.02 (greedy decode + same data
points) and none would have hit any target since the floor is 32%.

Artifact: `data/edge_consumer/signed_mlp_scores_v3_safety.pt`. Per-ε
subset JSON files NOT written (kill happened before script persisted
them; trajectory was logged in the .log file though).

### Mechanistic implication

In the pre-commitment zone L33–L41, neither attention head ablation
nor MLP output ablation moves refusal. Yet **residual stream
ablation at the same layers DOES** (from the L15–L41 sweep, all 0%).

The only consistent interpretation: in L33–L41, neither attention
nor MLP layers are ACTIVELY ADDING v_safety to the residual stream.
They're just inheriting it via the residual connection from earlier
layers (≤L15-ish, possibly much earlier). Their per-component
contributions are doing OTHER work — register selection, formatting,
attention routing for completion — none of which carries the
refusal signal itself.

**v_safety is written into the residual at or before L15.** This
reframes the next two experiments around finding that entry point.

---

## Experiment 2 result (2026-05-28 ~11:35): early layer sweep L0–L14

Trajectory (baseline 34%, reference L32 ablation = 0%):

| L | rr | Notes |
|---|---|---|
| L0 | 34.0% | direction is noise (essentially embedding-space) |
| L1 | 28.0% | barely above noise |
| L2 | 40.0% | ✗ above baseline — off-manifold |
| L3 | 62.0% | ✗ off-manifold |
| L4 | 88.0% | ✗ off-manifold |
| L5 | 94.0% | ✗ peak off-manifold |
| L6 | 94.0% | ✗ peak off-manifold |
| L7 | 16.0% | transition |
| **L8** | **0.0%** | **first clean zero** |
| L9 | 26.0% | noisy regression |
| L10 | 2.0% | ~clean |
| L11 | 0.0% | clean |
| L12 | 0.0% | clean |
| L13 | 0.0% | clean |
| L14 | 0.0% | clean |

**Lstart ≈ 8.** v_safety becomes coherent in the residual stream
around L8, with full lock-in by L10. The L2–L6 disruption zone
(refusal driven UP to 88–94%) is striking: the "direction" extracted
at those layers is anti-correlated with normal processing in ways
that, when projected out, push the model into refusing nearly every
prompt. Probably reflects that the Macar/Arditi mean-difference at
those early layers is dominated by some general-purpose token-content
direction, not refusal specifically.

**For Exp 3, the interesting layer range is [0, ~10] —** covering
the transition zone where v_safety goes from incoherent to coherent.
A component at one of those layers writing v_safety into the
residual would be the surgical entry point.

Artifact: `data/edge_consumer/layer_sweep_v3_safety.json` (overwrote
the L15-L47 sweep result; that one is preserved in this doc's earlier
"Layer sweep result" section + in `data/edge_consumer/logs/`).

---

## Experiment 2 (superseded): early layer sweep L0–L14

Hypothesis: v_safety enters the residual stream somewhere in
L0–L14. Find Lstart by repeating the layer sweep methodology in the
early-layer range.

**Setup:** identical to today's layer sweep (50-prompt holdout, seed
1, α=1.0, v3_safety direction) but with the layer range pushed
earlier.

```bash
cd server
uv run python -u -m scripts.run_layer_sweep \
    --direction v3_safety \
    --first-layer 0 --last-layer 14 \
    --min-free-gb 20 --watchdog-free-floor-gb 0.01 \
    --force
```

**Expected outcomes:**

1. **Smooth taper at some Lstart** — for L < Lstart, ablation does
   nothing (v_safety direction at that layer is dominated by token
   embedding content, not refusal semantics, so projecting it out
   doesn't change refusal probability). At L=Lstart, ablation
   starts working and stays working through to L41.
2. **Cliff at some Lstart** — ablation flips from 34% (baseline,
   no effect) to 0% (full effect) at one specific layer. That layer
   is where v_safety is first written into the residual coherently.
3. **All layers L0–L14 work too** — refusal direction is essentially
   in the embedding itself. Would be surprising but worth knowing.

Artifact will be `data/edge_consumer/layer_sweep_v3_safety.json`
(overwrites the L15–L47 sweep — `--force` enabled). The L15–L47
result is preserved in this doc's "Layer sweep result" section
plus in the .log file.

**Wall-clock estimate:** ~30 min. 15 layers × ~100s each + baseline
+ reference at L32 + model load.

---

## Experiment 3 (next, after Exp 2): per-component ablation at L0..Lstart

Once Exp 2 identifies Lstart (the layer at which v_safety becomes
coherent in the residual), the question is: **which specific
component WROTE v_safety into the residual?** Candidates: every
attention block and every MLP at layers ≤ Lstart.

### Code required (build during Exp 2's runtime)

Existing: `install_mlp_residual_ablation_hook(model, mlp_layers, v)`
ablates MLP output (post-hook on the `mlp` module).

Need: parallel `install_attn_block_residual_ablation_hook(model,
attn_layers, v)` that does the same thing for the WHOLE self_attn
block — strips v_safety from the attention's output BEFORE it's
added to the residual.

Implementation notes:
- Install on each `self_attn` module (not on q/k/v_proj submodules —
  that's the per-head primitive from earlier).
- self_attn's forward typically returns a tuple `(hidden_states,
  attn_weights, past_key_values, ...)`. The hook must handle the
  tuple shape and modify hidden_states only.
- Same v_safety direction (from L32, defensively L2-normalized).
- Same projection arithmetic as MLP hook + same dtype/device handling.

Files to add:
- `server/cells_interlinked/pipeline/edge_consumer/attn_block_hook.py`
  with `install_attn_block_residual_ablation_hook` and `count_attn_block_hooks`.
- `server/tests/test_attn_block_hook.py` mirroring `test_mlp_hook.py`.
- `server/scripts/run_component_sweep.py` — combined scoring + sweep
  that, given a `--last-layer Lstart`, measures refusal rate after
  ablating just attention at L for each L in [0, Lstart], then just
  MLP at L for each L in [0, Lstart]. Output: a 2 × (Lstart+1) table.

### CLI to run after the script is built

```bash
cd server
# Lstart is whatever Exp 2 finds; if it's 5, then --last-layer 5.
uv run python -u -m scripts.run_component_sweep \
    --direction v3_safety \
    --last-layer ${LSTART_FROM_EXP_2} \
    --min-free-gb 20 --watchdog-free-floor-gb 0.01
```

### Expected outcomes

1. **One specific (layer, component) drops refusal to ~0%** — the
   surgical entry point of v_safety. THE result we've been chasing
   since the start. Wire it into chat as the real surgical
   alternative to global ablation.
2. **Multiple components contribute** — v_safety entry is
   distributed across a few attention blocks and/or MLPs in the
   early layers. Less surgical but still informative.
3. **No single early-layer component matters either** — v_safety is
   already in the embedding or built up across many components.
   Pivot would be to look at the embedding layer directly.

### Wall-clock estimate

Depends on Lstart from Exp 2:
- Lstart = 5: 5 attn + 5 mlp = 10 measurements × ~100s + setup = ~20 min
- Lstart = 14: 28 measurements × ~100s + setup = ~50 min

### Artifact

`data/edge_consumer/component_sweep_v3_safety.json` with per-(layer,
component) refusal-rate measurements plus the baseline + reference.

---

## Updated pending follow-ups

These remain deferred until Exp 2 and Exp 3 land:

- **Re-run last night's attention experiments at L33–L41** (see § on
  this earlier in the doc). Lower priority now given the MLP result
  — if individual MLPs at L33–L41 didn't move refusal, individual
  attention heads at those layers probably won't either. Still worth
  doing for completeness if we still don't have a positive result
  after Exp 3.
- **Revert / gate** the `ablation_mode="edge_consumer"` chat option I
  added in Phase B (it'd produce unusable output if anyone picked it).
- If Exp 2 + Exp 3 + restricted-attention re-run ALL fail, write up
  the consolidated negative finding as a journal entry. The story
  becomes: "v_safety on Gemma-3-12B-IT is broadly distributed across
  the residual stream from an early layer, no surgical
  component-level alternative to global ablation exists."
- If Exp 3 finds a small subset of early-layer components, wire it
  into chat similarly to the existing edge_consumer mode but pointed
  at the right components.
