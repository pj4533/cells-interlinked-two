"""GET /stream/{run_id} — SSE drain of the run's event log.

Replays the FULL backlog of the run's events from index 0, then tails
the live stream. Multiple concurrent subscribers each get an independent
replay — useful when a user navigates away mid-run and returns; their
re-subscribed page receives every event the original session would have,
not just the events after reconnect.

Includes a 2KB initial padding comment so Safari and Firefox flush the
response buffer immediately.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

router = APIRouter()

_PADDING = ":" + (" " * 2047) + "\n\n"
_PING_INTERVAL_SEC = 15.0


@router.get("/stream/{run_id}")
async def stream(run_id: str, request: Request) -> EventSourceResponse:
    state = request.app.state.registry.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run not found")

    async def gen() -> AsyncIterator[dict]:
        # Flush-buster comment so non-Chromium browsers stop buffering.
        yield {"comment": _PADDING}

        # Iterate the event log starting from 0. Replays full backlog
        # then waits for new events. Returns naturally when the producer
        # calls event_log.close() and we've reached the end.
        log_iter = state.event_log.stream_from(0).__aiter__()

        # We interleave a periodic ping so HTTP intermediaries don't
        # drop the connection on long quiet stretches (a per-token NLA
        # decode at ~17s leaves the channel quiet between events).
        while True:
            if await request.is_disconnected():
                return
            try:
                evt = await asyncio.wait_for(
                    log_iter.__anext__(), timeout=_PING_INTERVAL_SEC,
                )
            except asyncio.TimeoutError:
                yield {"event": "ping", "data": "{}"}
                continue
            except StopAsyncIteration:
                return
            yield {"event": evt.get("type", "message"), "data": json.dumps(evt)}
            if evt.get("type") in ("done", "error"):
                return

    return EventSourceResponse(
        gen(),
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache, no-transform",
        },
    )
