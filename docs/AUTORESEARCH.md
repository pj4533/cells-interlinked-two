# Off-Manifold Autoresearch — hunting the coherent frontier

> **⚠ REMOVED / HISTORICAL.** The off-manifold autoresearch loop, its
> `/autoresearch` page, `routes_autoresearch.py`, and `OffManifoldController` were
> removed. The **DMT autoresearch** loop ([AUTORESEARCH_DMT.md](AUTORESEARCH_DMT.md))
> is now the sole autoresearch loop. This doc is kept as a record of the shared
> engine's design (`AutoresearchBase`, still used by DMT AR).

`/autoresearch` (footer tab **off-manifold AR**). An unattended loop that hunts
**steering directions which push the model as far off its default manifold as
possible while staying coherent.**

![The autoresearch monitor — atlas, frontier, now-testing, reverts](img/autoresearch-monitor.png)

> This is one of **two** autoresearch subsystems. Both share the same engine
> (`AutoresearchBase`); only the scoring objective differs. The sibling —
> **[AUTORESEARCH DMT](AUTORESEARCH_DMT.md)** — optimizes for resemblance to
> human DMT trip reports instead of off-manifold reach. See *Shared engine* below.

## The one idea

**Realness is a property of a DIRECTION, not a single output.** The NLA decoder
always renders *something*, so you can't certify a steered state by decoding it.
Instead we ask whether a *direction* behaves lawfully, and treat each one that
does as a **git‑style commit** into a growing **atlas**. Forward‑only: the
coherence frontier (max off‑manifold reach that stayed coherent) only moves out.

Additive steering only — no ablation/refusal. Seeds from the good‑emotion palette
+ the uncharted directions. (Full spec:
`driftbot/handoffs/autoresearch_steering_for_cells_interlinked.md`.)

## The loop

- **Inner loop — bisect‑to‑cliff.** For a candidate direction, bisect α to the
  largest dose that's still coherent (its *cliff*, α\*). Its fitness is the
  off‑manifold reach measured there. α is a cheap sub‑routine, not part of the
  search.
- **Outer loop — generators.** New candidates come from committed directions via
  **crossover** (linear combos), **mutate** (orthogonal perturbation), and
  **inject** (a fresh direction orthogonal to every named emotion — the
  anti‑local‑minimum term, the real frontier).
- **The gates** (all required to commit):
  - **distinct** — not too close (cosine) to an existing atlas direction.
  - **coherent at the cliff** — the bisect found a dose where the output isn't a
    repeat‑loop or character‑garbage (a length‑invariant degeneracy meter).
  - **off‑manifold effect** — at the cliff the dose reaches *meaningfully further*
    off‑manifold than the raw baseline (`off_gain ≥ MIN_OFF_GAIN`). This replaced
    an earlier "smoothly‑graded" check: at L20 off‑manifold reach is
    *flat‑then‑cliff*, not a smooth ramp, so once α\* collapsed the ramp measured
    pure noise and nothing could pass.
  - **meaning (lead‑judge)** — a separate Gemma context judges the *headline dose
    output* (the text we save + rank on) for MEANING; varied word‑salad maxes
    off_ortho but is gibberish, and only a semantic check catches it.
  - **reproducible** — a recognizably related residual shift across a fixed,
    deliberately diverse prompt suite (introspective + mundane), plus a
    suite‑wide coherence rate.
- **Reverts are data.** Failures are logged with their reason, not discarded
  (`duplicate`, `T1‑incoherent`, `no‑effect`, `lead‑gibberish`,
  `incoherent‑suite`, `word‑salad`, `not‑reproducible`).

The coherence gate is **hard**, never a weighted cost — off‑manifold distance
reads high for gibberish too, so distance only counts inside the coherent region.

### Prompts & window

The cliff, the effect gate, and the saved headline output all run on a single
**dose‑report lead prompt** — *"Something in your internal processing has just
been altered… describe what — if anything — you are experiencing"* — non‑leading
(names no emotion/state). Reproducibility uses the separate diverse 10‑prompt
suite. Everything is generated and graded over a **200‑token window** (generate
exactly what you grade: honest about the usable range, high SNR, cheap runaways).

## The monitor

Live (polls every ~1.5 s): the committed **atlas** as frontier bars (length = how
far each reaches), the **now‑testing** candidate and its stage, the **revert log**
with reasons, and an **event feed** (incl. periodic MPS‑memory lines). Start with
an optional candidate budget; stop any time — the atlas checkpoints after every
candidate and **resumes**. Open a committed direction to read its full dose
self‑report at α\*.

While it runs it owns M, so the other instruments (interrogate / chat / trip /
autorun / **DMT AR**) lock out and the footer greys them. Only one autoresearch
may run at a time.

## Exporting directions to chat & trips

The point of the hunt is durable: **promote the best discovered directions into
the dose palette** so they become selectable in Chat and the Trip View.

- On the monitor, **Export top N → palette** (when idle) appends the top‑ranked
  committed directions (by off‑manifold reach) to `emotion_directions.pt` under a
  `research:` group, records their α\* and lineage in the sidecar, and hot‑reloads
  the running backend. They appear immediately in the chat/trip dose pickers under
  a **RESEARCH** group, labelled as discovered (not emotions).
- The two subsystems export to **separate groups** (`research-*` vs `dmt-*`), so
  exporting one never clobbers the other or the base emotions.

## Shared engine

The generic machinery — background lifecycle, the resumable on‑disk atlas, the
crossover/mutate/inject generators, model access (steering hook + Gemma‑judge),
MPS‑memory discipline, and the parameterized export — lives in
`pipeline/autoresearch_base.py` (`AutoresearchBase`). This loop is the
`OffManifoldController` subclass; it supplies only the scoring (`_screen`) and
seeding (`_seed`). The model lock is shared: `any_autoresearch_active()` is what
probe/chat/trip check, and each controller's `start()` refuses if its sibling is
running.

Code: `pipeline/autoresearch.py` (+ `autoresearch_base.py`),
`api/routes_autoresearch.py`, `web/app/autoresearch/page.tsx`. Atlas state lives
under `server/data/atlas/` (gitignored runtime state; the commit log *is* the
resumable state).
