"""Interlink — model-to-model auto-conversation control + live stream.

  POST /interlink/start  {mode, dose_emotion, alpha, dose_ramp, opener, goal,
                          first_speaker, thinking}  → {ok, session_id}
  POST /interlink/stop                              → halt after the current message
  GET  /interlink/state                             → status + config + transcript so far
  GET  /interlink/stream/{session_id}               → SSE: message_start / interlink_token /
                                                       message_done / conversation_done / error
  GET  /interlink/sessions                          → archive list
  GET  /interlink/sessions/{session_id}             → archive review

One conversation runs at a time; it owns M while running (chat/trip/autoresearch
lock out via app.state.interlink_active).
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..config import settings
from ..storage import db

logger = logging.getLogger(__name__)
router = APIRouter()

_PADDING = ":" + (" " * 2047) + "\n\n"
_PING_INTERVAL_SEC = 15.0


def _ctrl(request: Request):
    c = getattr(request.app.state, "interlink", None)
    if c is None:
        raise HTTPException(status_code=503, detail="interlink controller not initialized")
    return c


class StartRequest(BaseModel):
    mode: str = "steer"                       # "steer" (dose@L20) or "ablate" (@L32)
    dose_emotion: str | None = None
    alpha: float = Field(default=0.5, ge=0.0, le=5.0)
    dose_ramp: int = Field(default=1, ge=0, le=128)
    opener: str = Field(..., min_length=1, max_length=8000)
    goal: str = Field(default="", max_length=2000)
    first_speaker: str = "beta"               # "raw" or "beta"
    thinking: bool = True


@router.post("/interlink/start")
async def interlink_start(req: StartRequest, request: Request) -> dict:
    return await _ctrl(request).start(
        mode=req.mode, dose_emotion=req.dose_emotion, alpha=req.alpha,
        dose_ramp=req.dose_ramp, opener=req.opener, goal=req.goal,
        first_speaker=req.first_speaker, thinking=req.thinking,
    )


@router.post("/interlink/stop")
async def interlink_stop(request: Request) -> dict:
    return await _ctrl(request).stop()


@router.get("/interlink/state")
async def interlink_state(request: Request) -> dict:
    return _ctrl(request).state()


@router.get("/interlink/stream/{session_id}")
async def interlink_stream(session_id: str, request: Request) -> EventSourceResponse:
    ctrl = _ctrl(request)
    if ctrl.session_id != session_id:
        raise HTTPException(status_code=404, detail="no live conversation with that id")
    log = ctrl.log
    # Tail live events only — the client rebuilds history from /interlink/state.
    start_idx = log.length()

    async def gen() -> AsyncIterator[dict]:
        yield {"comment": _PADDING}
        log_iter = log.stream_from(start_idx).__aiter__()
        next_task: asyncio.Task | None = asyncio.create_task(log_iter.__anext__())  # type: ignore[arg-type]
        try:
            while True:
                done, _ = await asyncio.wait({next_task}, timeout=_PING_INTERVAL_SEC)
                if next_task not in done:
                    yield {"event": "ping", "data": "{}"}
                    continue
                try:
                    evt = next_task.result()
                except StopAsyncIteration:
                    return
                next_task = asyncio.create_task(log_iter.__anext__())  # type: ignore[arg-type]
                yield {"event": evt.get("type", "message"), "data": json.dumps(evt)}
                if evt.get("type") == "conversation_done":
                    return
        finally:
            if next_task is not None and not next_task.done():
                next_task.cancel()

    return EventSourceResponse(
        gen(),
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache, no-transform"},
    )


@router.get("/interlink/sessions")
async def interlink_sessions(request: Request, limit: int = 50, offset: int = 0) -> dict:
    return await db.list_interlink_sessions(settings.db_path, limit=limit, offset=offset)


@router.get("/interlink/sessions/{session_id}")
async def interlink_session(session_id: str, request: Request) -> dict:
    row = await db.get_interlink_session(settings.db_path, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="session not found")
    return row
