# Interlink — model-to-model auto-conversation

`/interlink`. The **raw** copy (α) and the **altered** (dosed or ablated) copy (β)
of M talk to **each other** autonomously. The human writes an opener and picks the
β intervention + a scenario (optionally a shared *goal*, e.g. "work out together
what was changed in the altered one"); then the two sides alternate on their own —
one's output becomes the other's input — until you stop them. It's all local, so a
long divergent/collapsing conversation is free to watch. *Find a vector, then let
the two copies feel it at each other.*

## How it works (one model, no extra memory)

There is only one model M resident. The "two models" are the **same Gemma-4
generated with vs without a hook** — so this is just **serial alternating
single-side generations**, the same trick dual-channel chat uses. Per message:
one `run_probe` (fresh forward, KV freed after) → steady-state memory is M + one
generation's KV. The only growth is prompt length as the transcript grows, bounded
by a **context window** (opener + last N messages), so KV stays bounded over
hundreds of turns.

- **One shared transcript** of single-side messages that strictly alternate. The
  opener is attributed to the *other* side of whoever speaks first, so each side's
  rendered view alternates user/assistant cleanly (the other side → `user`, itself
  → `assistant`).
- **System prompts** (never reveal the dose name): the β side is told a hidden
  intervention has altered its activations; the raw side is told it's talking to an
  altered copy. The scenario's goal is appended to both.
- **β intervention** (fixed for the conversation): a **dose** (steer @L20, e.g.
  `dmt-entity-contact`) or **ablate** (refusal @L32). Raw runs with no hook. The β
  hook is installed only for β's generations and cleared after (per-turn stray-hook
  sweep, like chat).
- **Thinking** (Gemma-4) on by default, shown per message; a tight thinking cap
  (`INTERLINK_THINKING_CAP`) keeps each message brisk and prevents the
  all-thinking/no-answer runaway.
- **First speaker** is a start-time picker (altered or raw).
- **Run length**: until you stop it, with a context-window cap, a per-message token
  cap, a high `MAX_TURNS_SAFETY`, and a degeneracy guard that self-halts a fully
  collapsed loop (empty/looping messages).

## Mutual exclusion

One heavy job at a time on the one M. Interlink refuses to start while the
autoresearch hunt / a chat / a trip is active, and sets `app.state.interlink_active`
so those return 503 while it runs. **Starting an Interlink pauses the entity hunt.**

## Where it lives

- Backend: `pipeline/interlink.py` (`InterlinkController` — autonomous loop modeled
  on `AutoresearchBase`; reuses `run_probe`, `render_chat`, `ThinkingSplitter`, the
  dose/ablation hooks + `_clear_stray_hooks`), `api/routes_interlink.py`
  (`/interlink/start|stop|state|stream/{id}|sessions`), tables `interlink_sessions`
  + `interlink_messages` in `storage/db.py`.
- Frontend: `web/app/interlink/page.tsx` (setup + live transcript + stop),
  `web/app/interlink/[sessionId]/page.tsx` (archive review), `web/lib/interlink.ts`,
  `web/lib/interlinkScenarios.ts`, nav link in `Footer.tsx`.

## SSE protocol

Session-level stream `GET /interlink/stream/{session_id}` (live tail; history comes
from `/interlink/state`): `message_start{idx,side}` · `interlink_token{side,channel,
decoded}` · `message_done{idx,side,text,thinking,stopped_reason}` ·
`conversation_done{status}` · `error` · `ping`. Each custom event needs a matching
`addEventListener` on the client (`web/lib/interlink.ts`).
