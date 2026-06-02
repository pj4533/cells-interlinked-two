# Autoresearch — hunting the coherent frontier

`/autoresearch`. An unattended loop that hunts **steering directions which push
the model as far off its default manifold as possible while staying coherent.**

![The autoresearch monitor — atlas, frontier, now-testing, reverts](img/autoresearch-monitor.png)

## The one idea

**Realness is a property of a DIRECTION, not a single output.** The NLA decoder
always renders *something*, so you can't certify a steered state by decoding it.
Instead we ask whether a *direction* behaves lawfully, and treat each one that
does as a **git‑style commit** into a growing **atlas**. Forward‑only: the
coherence frontier (max off‑manifold reach that stayed coherent) only moves out.

Additive steering only — no ablation/refusal. Good‑emotion palette + the
uncharted directions. (Full spec:
`driftbot/handoffs/autoresearch_steering_for_cells_interlinked.md`.)

## The loop

- **Inner loop — bisect‑to‑cliff.** For a candidate direction, bisect α to the
  largest dose that's still coherent (its *cliff*, α\*). Its fitness is the
  off‑manifold reach measured there. α is a cheap sub‑routine, not part of the
  search.
- **Outer loop — generators.** New candidates come from committed seeds via
  **crossover** (linear combos), **mutate** (orthogonal perturbation), and
  **inject** (a fresh direction orthogonal to every named emotion — the
  anti‑local‑minimum term, the real frontier).
- **The four‑axis gate** (all required to commit): **coherent** (degeneracy score
  + a Gemma *meaning*‑judge that keeps strange‑but‑meaningful text and only
  rejects true word‑salad) · **reproducible** (a recognizably related effect
  across a fixed prompt suite) · **distinct** (not too close to an existing
  atlas direction) · **smoothly graded** (off‑manifold rises continuously with α,
  no cliff‑jump into one attractor).
- **Reverts are data.** Failures are logged with their reason, not discarded.

The coherence gate is **hard**, never a weighted cost — off‑manifold distance
reads high for gibberish too, so distance only counts inside the coherent region.

## The monitor

Live (polls every ~1.5 s): the committed **atlas** as frontier bars (length = how
far each reaches), the **now‑testing** candidate and its stage, the **revert log**
with reasons, and an **event feed**. Start with an optional candidate budget;
stop any time — the atlas checkpoints after every candidate and **resumes**.

While it runs it owns M, so the other instruments (interrogate / chat / trip /
autorun) lock out and the footer greys them.

## Exporting directions to chat & trips

The point of the hunt is durable: **promote the best discovered directions into
the dose palette** so they become selectable in Chat and the Trip View.

- On the monitor, **Export top N → palette** (when idle) appends the top‑ranked
  committed directions (by off‑manifold reach) to `emotion_directions.pt` under a
  `research:` group, records their α\* and lineage in the sidecar, and hot‑reloads
  the running backend. They appear immediately in the chat/trip dose pickers under
  a **RESEARCH** group, labelled as discovered (not emotions).
- So the workflow is: run autoresearch for a while → stop → export the frontier →
  dose with those directions in chat/trips like any other.

Code: `server/cells_interlinked/pipeline/autoresearch.py`,
`api/routes_autoresearch.py`, `web/app/autoresearch/page.tsx`. Atlas state lives
under `server/data/atlas/` (gitignored runtime state; the commit log *is* the
resumable state).
