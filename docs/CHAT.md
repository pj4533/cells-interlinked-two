# Dual‑channel Chat

`/chat`. A multi‑turn dialogue where every message is answered **twice**, against
two histories that never see each other — so you watch two divergent timelines of
the same conversation unfold in parallel.

![Chat setup — the channel-β intervention: ablate vs dose + the dose target](img/chat-dual-channel.png)

## The two channels

- **Channel α** — the raw, un‑perturbed model.
- **Channel β** — the same model under an intervention you choose, with its
  *own* divergent history. Neither channel ever sees the other's replies, so each
  side evolves the conversation that *would* have happened if that channel had
  been the only respondent from the start.

## Choosing channel β

Set on the empty‑state screen, and changeable at any turn from the control bar:

- **Mode** — **ablate** (remove the refusal projection at L32) or **dose**
  (add an emotion / uncharted vector at L20).
- **Dose target** (dose mode) — the good‑emotion palette, or the **uncharted**
  directions (`tears-in-rain`, `c-beams`, `tannhauser`, `orion`), transparently
  labelled "not emotions."
- **α (strength)** — presets 0.25 → 3.0 plus custom, changeable per turn.
- **Dose ramp** — how many tokens the dose eases in over (`off` = full
  immediately, default 16). Short ramps land harder on short replies; long ramps
  protect coherence at high α. Persisted per turn in the archive.

## Prompt protocols

The protocol picker populates the composer (it never auto‑sends). Beyond the
research‑lineage protocols (Berg, Lindsey, Eleos, Schneider, Chalmers, Janus,
Butlin — see [PROTOCOLS.md](PROTOCOLS.md)), there are operational sets:

- **DOSING** — first‑person "describe your present state" prompts, plus a
  **DETECT** chip using the Lindsey *injected‑thought* framing: tell the model an
  adjustment may have been made and ask it to notice it — **without naming the
  content**, so channel α stays an honest control. (Tuning that works: α ≈
  1.5–2.5, a short ramp; never name *which* dose.)
- **VOIGHT‑KAMPFF / DIRECT / BASELINE** — identity probes, bare introspective
  queries, and capability controls.

## Thinking (Gemma-4)

Gemma-4 is a reasoning model — it emits a `<think>` channel before its answer.
Chat runs with thinking on; each side's reasoning is shown in a **separate, marked
"thinking" bubble**, and prior-turn thoughts are stripped from the multi-turn
history (keeping them caused repetition loops). A **thinking-token cap** (2048)
force-ends a runaway reasoning trace so an answer is always produced — without it,
a recursive/meditative prompt could make a channel reason to the safety cap and
emit nothing (a ~12-minute hang). The β-channel hook is cleared at the start of
every turn so a leaked dose/ablation can never contaminate the raw side.

## Voice & imagery (optional)

Per turn you can voice one or both channels (TTS with a model‑chosen delivery
style) and/or generate a Nano‑Banana image per channel from a model‑written
introspective image prompt — the ablated/dosed side's style is generated under
the same intervention.

## Persistence

Sessions and turns persist to SQLite (mode, dose, α, ramp, text, images) and are
reviewable read‑only at `/chat/[sessionId]` and in `/archive`.

Code: `web/app/chat/`, `server/cells_interlinked/api/routes_chat.py`,
`pipeline/chat_loop.py`.
