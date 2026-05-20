"""Chat-mode routes: dual-thread conversation with M.

Each chat session holds two parallel histories (raw + ablated). Each
user turn fires both M passes serially, streaming tokens to the client
on two distinct event channels so the UI can paint two bubbles
simultaneously.

Endpoints:
  POST /chat/sessions               create a session, returns {session_id, alpha, variant}
  GET  /chat/sessions/{sid}         fetch session state (history + alpha)
  POST /chat/sessions/{sid}/turn    send a user message, returns {turn_idx}
  GET  /chat/stream/{sid}/{turn}    SSE drain for that turn
  POST /chat/sessions/{sid}/cancel  abort the in-flight turn
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..config import settings
from ..pipeline.chat_loop import ChatSession, ChatTurn, execute_turn, new_session
from ..storage import db

logger = logging.getLogger(__name__)
router = APIRouter()


_PADDING = ":" + (" " * 2047) + "\n\n"
_PING_INTERVAL_SEC = 15.0


# Per-session event log: tiny analogue of the probe registry's EventLog,
# but per-turn. The streaming endpoint reads from this log; the turn
# executor writes to it.
class _TurnEventLog:
    def __init__(self) -> None:
        self._events: list[dict] = []
        self._cond = asyncio.Condition()
        self._closed = False

    async def append(self, evt: dict) -> None:
        async with self._cond:
            self._events.append(evt)
            self._cond.notify_all()

    async def close(self) -> None:
        async with self._cond:
            self._closed = True
            self._cond.notify_all()

    async def stream_from(self, start_idx: int = 0) -> AsyncIterator[dict]:
        i = start_idx
        while True:
            async with self._cond:
                while i >= len(self._events) and not self._closed:
                    await self._cond.wait()
                if i >= len(self._events):
                    return
                evt = self._events[i]
                i += 1
            yield evt


def _sessions(request: Request) -> dict[str, ChatSession]:
    sessions: dict[str, ChatSession] | None = getattr(
        request.app.state, "chat_sessions", None,
    )
    if sessions is None:
        sessions = {}
        request.app.state.chat_sessions = sessions
    return sessions


def _logs(request: Request) -> dict[str, _TurnEventLog]:
    """Per-(session,turn) event logs, keyed by f"{sid}:{turn_idx}"."""
    logs: dict[str, _TurnEventLog] | None = getattr(
        request.app.state, "chat_turn_logs", None,
    )
    if logs is None:
        logs = {}
        request.app.state.chat_turn_logs = logs
    return logs


def _cancels(request: Request) -> dict[str, asyncio.Event]:
    """Per-session cancel events. One outstanding turn per session, so
    keyed by session_id (not turn idx)."""
    cancels: dict[str, asyncio.Event] | None = getattr(
        request.app.state, "chat_cancels", None,
    )
    if cancels is None:
        cancels = {}
        request.app.state.chat_cancels = cancels
    return cancels


# ── Models ────────────────────────────────────────────────────────────

class NewSessionRequest(BaseModel):
    alpha: float = Field(default=0.5, ge=0.0, le=5.0)


class NewSessionResponse(BaseModel):
    session_id: str
    alpha: float
    direction_variant: str


class TurnRequest(BaseModel):
    user_text: str = Field(..., min_length=1, max_length=8000)
    # Per-turn α. Optional; falls back to the session's α (which itself
    # was set on session creation) when not supplied.
    alpha: float | None = Field(default=None, ge=0.0, le=5.0)
    # Per-turn voice mode. Selects which side(s) get the voice
    # system prompt + envelope parsing + TTS playback:
    #   "off"     → no voice; both sides use the default prompt
    #   "both"    → both raw and ablated emit envelopes + are spoken
    #   "raw"     → only raw emits envelope + is spoken; ablated runs
    #               on the default prompt (clean text stream)
    #   "ablated" → only ablated emits envelope + is spoken; raw runs
    #               on the default prompt
    # Accepts the legacy boolean for back-compat — `true` → "both".
    voice_mode: str = "off"
    # Per-turn imagery toggle. When true, BOTH sides generate an
    # image-prompt via a separate Gemma pass and send it to Gemini
    # Nano Banana; the resulting PNGs are saved under
    # data/chat_images/ and surfaced as thumbnails next to each
    # channel's text.
    imagery_enabled: bool = False
    # Operator-selected framing for the image-prompt pass (one of
    # IMAGE_PROMPT_FRAMINGS' keys). Defaults to "evokes". Falls
    # through to the server's default if the value is unknown.
    imagery_framing: str = "evokes"


class TurnResponse(BaseModel):
    turn_idx: int


class TurnView(BaseModel):
    turn_idx: int
    user_text: str
    raw_text: str
    ablated_text: str
    raw_stopped_reason: str
    ablated_stopped_reason: str
    started_at: float
    finished_at: float | None
    error: str | None
    alpha: float
    # Imagery state. Empty strings when imagery was off for the turn
    # (or the side's generation failed); the *_image_url fields are
    # relative paths into the /chat-images static mount.
    raw_image_prompt: str = ""
    ablated_image_prompt: str = ""
    raw_image_url: str = ""
    ablated_image_url: str = ""
    # Operator-selected framing key + the rendered template (with
    # user_query interpolated). The rendered string is what the
    # archive modal shows under "prompt sent to model" so the
    # operator can see exactly what was asked. Empty on legacy
    # turns (pre-framing-feature).
    image_framing: str = ""
    image_framing_prompt: str = ""


class SessionView(BaseModel):
    session_id: str
    alpha: float
    direction_variant: str
    created_at: float
    turns: list[TurnView]


def _render_framing_prompt(t: ChatTurn) -> str:
    """Render the framing prompt that was sent to M for this turn's
    image-prompt pass. Returns the empty string if imagery was off
    or the turn predates the framing feature."""
    from ..pipeline.chat_loop import build_image_prompt_request
    if not t.imagery_framing:
        return ""
    return build_image_prompt_request(t.user_text, t.imagery_framing)


def _turn_to_view(t: ChatTurn) -> TurnView:
    # Render the framing prompt on demand from the stored key + the
    # user's text. This way historical sessions get reformatted with
    # the current template wording, but the framing key was the
    # operator's actual choice at the time. Empty framing → empty
    # rendered prompt (legacy turns predating this feature).
    from ..pipeline.chat_loop import build_image_prompt_request
    framing_key = t.imagery_framing if t.imagery_enabled else ""
    framing_prompt = (
        build_image_prompt_request(t.user_text, framing_key)
        if framing_key
        else ""
    )
    return TurnView(
        turn_idx=t.turn_idx,
        user_text=t.user_text,
        raw_text=t.raw_text,
        ablated_text=t.ablated_text,
        raw_stopped_reason=t.raw_stopped_reason,
        ablated_stopped_reason=t.ablated_stopped_reason,
        started_at=t.started_at,
        finished_at=t.finished_at,
        error=t.error,
        alpha=t.alpha,
        raw_image_prompt=t.raw_image_prompt,
        ablated_image_prompt=t.ablated_image_prompt,
        raw_image_url=t.raw_image_url,
        ablated_image_url=t.ablated_image_url,
        image_framing=framing_key,
        image_framing_prompt=framing_prompt,
    )


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/chat/sessions", response_model=NewSessionResponse)
async def create_session(req: NewSessionRequest, request: Request) -> NewSessionResponse:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="M not loaded")

    # Read the active refusal-direction variant name so the UI can
    # surface "α=0.5 · v3_safety" framing alongside ablated turns.
    # When the self-denial subspace is active, its sidecar's variant_name
    # takes precedence — that's what the runtime hook is actually using
    # (chat_loop.py:execute_turn calls pick_ablation_target(subspace,
    # directions, ...) which prefers the subspace).
    variant_name = ""
    try:
        sub_meta_path = settings.db_path.parent / "refusal_subspace.pt.json"
        if sub_meta_path.exists():
            variant_name = json.loads(sub_meta_path.read_text()).get(
                "variant_name", "self_denial_subspace",
            )
        else:
            meta_path = settings.db_path.parent / "refusal_directions.pt.json"
            if meta_path.exists():
                variant_name = json.loads(meta_path.read_text()).get(
                    "variant_name", "",
                )
    except Exception:
        pass

    session = new_session(req.alpha, variant_name=variant_name)
    _sessions(request)[session.session_id] = session

    # Persist the session header right away so the archive list shows
    # the session even before the first turn lands.
    try:
        await db.insert_chat_session(
            settings.db_path,
            session_id=session.session_id,
            alpha=session.alpha,
            direction_variant=session.direction_variant,
            created_at=session.created_at,
        )
    except Exception:
        logger.exception("failed to persist chat session header")

    return NewSessionResponse(
        session_id=session.session_id,
        alpha=session.alpha,
        direction_variant=session.direction_variant,
    )


async def _rehydrate_session(sid: str, request: Request) -> ChatSession | None:
    """Pull a session from SQLite back into the in-memory map. Returns
    the live ChatSession (or None if the session_id is unknown).
    Idempotent: if the session is already in memory, just returns it."""
    sessions = _sessions(request)
    existing = sessions.get(sid)
    if existing is not None:
        return existing
    row = await db.get_chat_session(settings.db_path, sid)
    if row is None:
        return None
    session = ChatSession(
        session_id=row["session_id"],
        alpha=row["alpha"],
        direction_variant=row["direction_variant"],
        created_at=row["created_at"],
    )
    # Replay persisted turns into memory. Completed turns get their
    # canonical text re-attached; in-flight turns from a previous
    # backend process would show finished_at=None — we restore them
    # but they're effectively frozen (the asyncio task that was
    # driving them is gone).
    for t in row["turns"]:
        session.turns.append(ChatTurn(
            turn_idx=t["turn_idx"],
            user_text=t["user_text"],
            alpha=t["alpha"],
            raw_text=t["raw_text"],
            ablated_text=t["ablated_text"],
            raw_stopped_reason=t["raw_stopped_reason"],
            ablated_stopped_reason=t["ablated_stopped_reason"],
            started_at=t["started_at"],
            finished_at=t["finished_at"],
            error=t["error"],
            raw_image_prompt=t.get("raw_image_prompt", ""),
            ablated_image_prompt=t.get("ablated_image_prompt", ""),
            raw_image_url=t.get("raw_image_url", ""),
            ablated_image_url=t.get("ablated_image_url", ""),
            imagery_enabled=bool(
                t.get("raw_image_url") or t.get("ablated_image_url")
            ),
            imagery_framing=t.get("image_framing") or "",
        ))
    sessions[sid] = session
    return session


@router.get("/chat/sessions/{sid}", response_model=SessionView)
async def get_session(sid: str, request: Request) -> SessionView:
    session = await _rehydrate_session(sid, request)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return SessionView(
        session_id=session.session_id,
        alpha=session.alpha,
        direction_variant=session.direction_variant,
        created_at=session.created_at,
        turns=[_turn_to_view(t) for t in session.turns],
    )


@router.get("/chat/sessions")
async def list_sessions(request: Request, limit: int = 50, offset: int = 0) -> dict:
    """Paginated session list for the archive. Read straight from
    SQLite — in-memory map is only the actively-streamed set."""
    return await db.list_chat_sessions(
        settings.db_path,
        limit=max(1, min(200, limit)),
        offset=max(0, offset),
    )


@router.post("/chat/sessions/{sid}/turn", response_model=TurnResponse)
async def post_turn(sid: str, req: TurnRequest, request: Request) -> TurnResponse:
    session = await _rehydrate_session(sid, request)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="M not loaded")

    # Per-session serialization: only one outstanding turn at a time.
    if session._lock.locked():
        raise HTTPException(
            status_code=409, detail="a turn is already in flight for this session"
        )

    turn_alpha = req.alpha if req.alpha is not None else session.alpha
    turn_alpha = max(0.0, min(5.0, float(turn_alpha)))
    # Coerce voice_mode. Accept the legacy bool form (`true`/`false`)
    # so a stale client doesn't 422 — `true` maps to "both", `false`
    # to "off". Anything not in the known set falls back to "off".
    raw_mode = req.voice_mode
    if isinstance(raw_mode, bool):  # type: ignore[unreachable]
        voice_mode = "both" if raw_mode else "off"
    elif raw_mode in ("off", "both", "raw", "ablated"):
        voice_mode = raw_mode
    else:
        voice_mode = "off"
    # Validate imagery framing against the known set; fall through
    # to the default if a stale client sends an unknown key.
    from ..pipeline.chat_loop import (
        DEFAULT_IMAGE_FRAMING,
        IMAGE_PROMPT_FRAMINGS,
    )
    framing_key = (
        req.imagery_framing
        if req.imagery_framing in IMAGE_PROMPT_FRAMINGS
        else DEFAULT_IMAGE_FRAMING
    )
    turn = ChatTurn(
        turn_idx=len(session.turns),
        user_text=req.user_text.strip(),
        alpha=turn_alpha,
        voice_mode=voice_mode,
        imagery_enabled=bool(req.imagery_enabled),
        imagery_framing=framing_key,
    )
    session.turns.append(turn)
    log = _TurnEventLog()
    _logs(request)[f"{sid}:{turn.turn_idx}"] = log
    cancel = asyncio.Event()
    _cancels(request)[sid] = cancel

    # Per-side event relabeler. Every event flowing out of execute_turn
    # gets tagged with its channel + a typed name the client SSE
    # listeners can hook (raw_token vs ablated_token, etc). Image
    # events follow the same pattern so the browser EventSource
    # delivers them — without an addEventListener for the exact name,
    # the event is silently dropped.
    _RELABEL = {
        "token": "{side}_token",
        "stopped": "{side}_stopped",
        "image_prompt": "{side}_image_prompt",
        "image_generating": "{side}_image_generating",
        "image_done": "{side}_image_done",
        "image_error": "{side}_image_error",
    }

    def _make_emitter(side: str):
        async def emit(evt: dict) -> None:
            side_evt = {**evt, "side": side}
            tmpl = _RELABEL.get(evt.get("type", ""))
            if tmpl:
                side_evt["type"] = tmpl.format(side=side)
            await log.append(side_evt)
        return emit

    emit_raw = _make_emitter("raw")
    emit_ablated = _make_emitter("ablated")

    async def driver() -> None:
        async with session._lock:
            await log.append({
                "type": "turn_started",
                "turn_idx": turn.turn_idx,
                "alpha": turn.alpha,
            })
            rdirs = getattr(request.app.state, "refusal_directions", None)
            rsub = getattr(request.app.state, "refusal_subspace", None)
            try:
                await execute_turn(
                    bundle=bundle,
                    session=session,
                    turn=turn,
                    raw_emit=emit_raw,
                    ablated_emit=emit_ablated,
                    cancel_event=cancel,
                    refusal_directions=rdirs,
                    refusal_subspace=rsub,
                )
            except Exception as exc:
                logger.exception("chat turn executor failed")
                turn.error = str(exc)
                await log.append({"type": "error", "message": str(exc)})
            # Persist the completed turn before signaling done so a
            # client that reloads on receipt of turn_done will see the
            # canonical row in DB.
            try:
                await db.upsert_chat_turn(
                    settings.db_path,
                    session_id=session.session_id,
                    turn_idx=turn.turn_idx,
                    user_text=turn.user_text,
                    raw_text=turn.raw_text,
                    ablated_text=turn.ablated_text,
                    raw_stopped_reason=turn.raw_stopped_reason,
                    ablated_stopped_reason=turn.ablated_stopped_reason,
                    started_at=turn.started_at,
                    finished_at=turn.finished_at,
                    error=turn.error,
                    alpha=turn.alpha,
                    raw_image_prompt=turn.raw_image_prompt,
                    ablated_image_prompt=turn.ablated_image_prompt,
                    raw_image_url=turn.raw_image_url,
                    ablated_image_url=turn.ablated_image_url,
                    image_framing=(
                        turn.imagery_framing if turn.imagery_enabled else ""
                    ),
                )
            except Exception:
                logger.exception("failed to persist chat turn")

            await log.append({
                "type": "turn_done",
                "turn_idx": turn.turn_idx,
                "raw_text": turn.raw_text,
                "ablated_text": turn.ablated_text,
                "raw_stopped_reason": turn.raw_stopped_reason,
                "ablated_stopped_reason": turn.ablated_stopped_reason,
                "error": turn.error,
                # voice_mode is now a string ("off"/"both"/"raw"/"ablated").
                # Clients that care about the boolean shape compare to
                # != "off"; clients that care which side is voiced
                # branch on the literal value.
                "voice_mode": turn.voice_mode,
                "raw_speech": turn.raw_speech,
                "raw_style": turn.raw_style,
                "ablated_speech": turn.ablated_speech,
                "ablated_style": turn.ablated_style,
                # Imagery final state — empty when imagery was off or
                # the side's generation failed. URLs are relative paths
                # served by the /chat-images static mount.
                "imagery_enabled": turn.imagery_enabled,
                "raw_image_prompt": turn.raw_image_prompt,
                "ablated_image_prompt": turn.ablated_image_prompt,
                "raw_image_url": turn.raw_image_url,
                "ablated_image_url": turn.ablated_image_url,
                "raw_image_error": turn.raw_image_error,
                "ablated_image_error": turn.ablated_image_error,
                # The framing key the operator picked + the rendered
                # template with user_query interpolated, so the
                # client can show "what was sent to the model" in
                # the lightbox modal without reconstructing it
                # locally.
                "image_framing": (
                    turn.imagery_framing if turn.imagery_enabled else ""
                ),
                "image_framing_prompt": (
                    _render_framing_prompt(turn)
                    if turn.imagery_enabled
                    else ""
                ),
            })
            await log.close()

    asyncio.create_task(driver())
    return TurnResponse(turn_idx=turn.turn_idx)


@router.post("/chat/sessions/{sid}/cancel")
async def cancel_turn(sid: str, request: Request) -> dict:
    cancels = _cancels(request)
    ev = cancels.get(sid)
    if ev is None:
        return {"ok": False, "reason": "no in-flight turn"}
    ev.set()
    return {"ok": True}


@router.get("/chat/stream/{sid}/{turn_idx}")
async def stream_turn(
    sid: str, turn_idx: int, request: Request,
) -> EventSourceResponse:
    log = _logs(request).get(f"{sid}:{turn_idx}")
    if log is None:
        raise HTTPException(status_code=404, detail="turn not found")

    async def gen() -> AsyncIterator[dict]:
        yield {"comment": _PADDING}
        log_iter = log.stream_from(0).__aiter__()
        next_task: asyncio.Task | None = asyncio.create_task(
            log_iter.__anext__()  # type: ignore[arg-type]
        )
        try:
            while True:
                done, _pending = await asyncio.wait(
                    {next_task}, timeout=_PING_INTERVAL_SEC,
                )
                if next_task not in done:
                    yield {"event": "ping", "data": "{}"}
                    continue
                try:
                    evt = next_task.result()
                except StopAsyncIteration:
                    return
                next_task = asyncio.create_task(
                    log_iter.__anext__()  # type: ignore[arg-type]
                )
                yield {
                    "event": evt.get("type", "message"),
                    "data": json.dumps(evt),
                }
                if evt.get("type") in ("turn_done", "error"):
                    return
        finally:
            if next_task is not None and not next_task.done():
                next_task.cancel()

    return EventSourceResponse(
        gen(),
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache, no-transform",
        },
    )
