# Edge-Consumer Ablation — Phase A (Inspection UI) Plan

> **Status:** sketch. Real shape TBD after the first overnight Phase B run produces concrete artifacts. We just removed an 1175-line `/experiments` page that was built ahead of validated data; the lesson is to design this surface AROUND the actual data, not before.

---

## What Phase A adds

A read-only inspection surface for the artifacts written by the Phase B scripts:

| Artifact | What's in it | UI need |
|---|---|---|
| `attribution_scores_{variant}.pt` | ~256 (layer, head) scores | **AP heatmap** (layer × head grid, cyan intensity = AP magnitude) |
| `sufficient_subset_eps={ε}_{variant}.json` × N ε | sufficient subset per ε + iteration trajectory | **subset view**: which heads selected, refusal-rate-vs-size curve |
| `verdict_{variant}_eps={ε}.json` | per-prompt L2 trajectories + 2×2 table | **2×2 table card** + **per-prompt triple-column samples** |

These need a home. The phase A question is *where*.

## Where it lives — decision deferred

Three plausible homes; we choose after the first overnight makes the data shape concrete:

**Option C: dedicated `/circuits` page (recommended baseline)**
- New top-level route, separate namespace from the just-removed `/experiments` and from the day-to-day `/chat`
- Single page hosts: heatmap (top), subset view (middle), 2×2 table + sample drawer (bottom)
- Triple-column "raw / global / edge" samples live here, NOT on chat
- Reuses `DualTranscript` pattern from `web/app/verdict/[runId]/page.tsx`

**Option C alt: fold into `/fine-print`**
- The 2×2 L2 table is a *methodological* artifact — it tells the user whether global ablation is over-broad
- Putting it on `/fine-print` next to the existing caveats panel makes the falsifier framing explicit
- Downside: `/fine-print` becomes a mixed-purpose page (caveats + active data)

**Option C alt: sub-tab of `/archive`**
- Heatmap and subset are read-only artifacts, conceptually similar to past probe records
- Downside: archive is per-event ("here's one run"), circuits is per-variant ("here's our analysis of v3_safety")

**Recommendation when artifacts land:** start with Option C (new `/circuits` page). Survey the actual heatmap, subset trajectory, and 2×2 table; if any one panel turns out to be uninteresting (e.g., a uniform heatmap because the broadcast hypothesis won), drop that panel and consider folding the remainder into one of the alt locations.

## Phase A's data contract — what the UI will need

These are the API endpoints the inspection page will read. Phase B doesn't build them; Phase A does. Pin them now so the artifact format and the future API can co-evolve.

```
GET /circuits/variants
  → list[{variant, file, modified_at, has_attribution, subset_epsilons, has_verdict}]

GET /circuits/{variant}/attribution
  → { ranked: [{layer, head, ap}], consumer_layers, n_contrasts }

GET /circuits/{variant}/subsets
  → { epsilons: { "0.05": { subset: [[L, h], ...],
                             trajectory: [{iter, rr, delta}], stopped_reason },
                  ... },
      baseline_refusal_rate, global_refusal_rate }

GET /circuits/{variant}/verdict?eps=0.05
  → { table_2x2, per_prompt: [{prompt, kind, l2_global, l2_edge,
                                refusal_mask, channels: {...}}],
      metadata }

GET /circuits/{variant}/sample/{prompt_idx}
  → full per-prompt triple-column transcript (raw, global, edge texts)
```

All four endpoints are static-file readers: they parse the JSON/pt artifacts from `server/data/edge_consumer/` and stream them out. No compute. The CLI scripts produce the input format; the API just exposes it. No SSE — these endpoints are GET-only and synchronous.

## Visualizations needed

**Heatmap.** A grid layer × head, colored by AP score. Gemma-3-12B: 16 layers (33–48) × 16 query-heads-per-layer = 256 cells. Cyan intensity scales with AP. Hover shows L, head, AP. Click pins a head and shows its row in the subset trajectory (which iteration was it added?). Easiest: SVG or a Canvas drawn cell-by-cell.

**Subset trajectory.** Two curves on the same axis: refusal-rate (y) vs subset-size (x), with three horizontal reference lines (baseline / global / target ± ε). One curve per ε in the swept set, color-coded. Where each curve hits its stopping criterion → that's the sufficient subset size.

**2×2 L2 table.** Plain table with 4 cells. Above the table: a one-sentence interpretation rendered from the values themselves ("global L2 is X× larger than edge L2 on non-refusal tokens — surgical ablation preserves more semantic content there"). Below: link to the per-prompt drill-down.

**Per-prompt triple-column drawer.** Reuse the chat-page `DualTranscript` pattern; extend to 3 columns with a third cyan-dim style for the edge channel. Refusal-relevant tokens highlighted on the raw side (the bucketing that produced the 2×2 averages).

## What we explicitly DON'T do in phase A

- **No live compute trigger.** Phase A is read-only on existing artifacts. Re-running an overnight is `cd server && uv run python -m scripts.run_edge_consumer_pipeline`, not a button.
- **No editing the subset.** Operator can't manually add/remove heads from the subset via the UI; subsets are immutable per (variant, ε).
- **No comparing across variants.** v3_safety is the only variant for now; multi-variant comparison (v1 / v2 / v3 / v4) is phase 2.
- **No chat integration changes.** The chat surface already has the `ablation_mode` picker (Phase B); Phase A adds inspection, not interaction.

## Triggers — when to start Phase A

- The first overnight Phase B run completes successfully
- The AP heatmap is non-trivially structured (not uniform — uniform means the broadcast hypothesis won and Phase A is unnecessary)
- The sufficient subset for ε=0.05 is small (<25% of candidate heads) OR the elasticity across ε is informative

If those triggers fire, build phase A. If they don't, write up the negative result and move to the next CI extension (Moon HSV / Zhou cross-patching).

## Cost estimate (when triggered)

- API endpoints (4 GETs, all reading static files): ~half a day
- Heatmap component: ~1 day (SVG / Canvas + interaction)
- Subset trajectory chart: ~half a day (Recharts or vanilla SVG)
- 2×2 table card: ~2 hours
- Per-prompt drawer (extension of DualTranscript to 3 cols): ~1 day
- Wiring + nav link: ~half a day
- **Total: ~4 days of focused work** once we have artifacts and a clear sense of which panels are load-bearing.
