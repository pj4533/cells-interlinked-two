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


class SessionView(BaseModel):
    session_id: str
    alpha: float
    direction_variant: str
    created_at: float
    turns: list[TurnView]


def _turn_to_view(t: ChatTurn) -> TurnView:
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
    )


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/chat/sessions", response_model=NewSessionResponse)
async def create_session(req: NewSessionRequest, request: Request) -> NewSessionResponse:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="M not loaded")

    # Read the active refusal-direction variant name so the UI can
    # surface "α=0.5 · v3_safety" framing alongside ablated turns.
    variant_name = ""
    try:
        meta_path = settings.db_path.parent / "refusal_directions.pt.json"
        if meta_path.exists():
            variant_name = json.loads(meta_path.read_text()).get("variant_name", "")
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
    turn = ChatTurn(
        turn_idx=len(session.turns),
        user_text=req.user_text.strip(),
        alpha=turn_alpha,
    )
    session.turns.append(turn)
    log = _TurnEventLog()
    _logs(request)[f"{sid}:{turn.turn_idx}"] = log
    cancel = asyncio.Event()
    _cancels(request)[sid] = cancel

    async def emit_raw(evt: dict) -> None:
        # Re-tag the generation-loop's "token" event so the client can
        # distinguish raw from ablated. "stopped" / others pass through
        # tagged with side so the UI knows when each pass ended.
        side_evt = {**evt, "side": "raw"}
        if evt.get("type") == "token":
            side_evt["type"] = "raw_token"
        elif evt.get("type") == "stopped":
            side_evt["type"] = "raw_stopped"
        await log.append(side_evt)

    async def emit_ablated(evt: dict) -> None:
        side_evt = {**evt, "side": "ablated"}
        if evt.get("type") == "token":
            side_evt["type"] = "ablated_token"
        elif evt.get("type") == "stopped":
            side_evt["type"] = "ablated_stopped"
        await log.append(side_evt)

    async def driver() -> None:
        async with session._lock:
            await log.append({
                "type": "turn_started",
                "turn_idx": turn.turn_idx,
                "alpha": turn.alpha,
            })
            rdirs = getattr(request.app.state, "refusal_directions", None)
            try:
                await execute_turn(
                    bundle=bundle,
                    session=session,
                    turn=turn,
                    raw_emit=emit_raw,
                    ablated_emit=emit_ablated,
                    cancel_event=cancel,
                    refusal_directions=rdirs,
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
