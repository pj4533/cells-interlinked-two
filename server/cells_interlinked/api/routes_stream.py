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
import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

router = APIRouter()
logger = logging.getLogger(__name__)

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

        t_open = time.time()
        n_events = 0
        n_pings = 0
        exit_reason = "unknown"

        # CRITICAL: do NOT wrap `log_iter.__anext__()` in
        # `asyncio.wait_for(...)`. wait_for *cancels* the awaited
        # coroutine on timeout, and cancelling an async generator's
        # __anext__ closes the generator permanently. Subsequent
        # __anext__ calls then raise StopAsyncIteration, which we'd
        # mis-read as "log closed, run is done" and exit. Result was
        # a ~15s reconnect loop: open stream, replay, wait 15s, yield
        # one ping, generator closed on next __anext__, SSE exits,
        # browser reconnects, repeat.
        #
        # Instead: keep a persistent next-event task and race it
        # against a timeout via asyncio.wait. The next-event task
        # survives the timeout; we only consume it when it actually
        # completes.
        next_task: asyncio.Task | None = asyncio.create_task(
            log_iter.__anext__()  # type: ignore[arg-type]
        )
        try:
            while True:
                done, _pending = await asyncio.wait(
                    {next_task}, timeout=_PING_INTERVAL_SEC,
                )
                if next_task not in done:
                    n_pings += 1
                    yield {"event": "ping", "data": "{}"}
                    continue
                try:
                    evt = next_task.result()
                except StopAsyncIteration:
                    exit_reason = "log_closed"
                    next_task = None
                    return
                # Kick off the next __anext__ before yielding so we
                # don't introduce latency between events.
                next_task = asyncio.create_task(
                    log_iter.__anext__()  # type: ignore[arg-type]
                )
                n_events += 1
                yield {"event": evt.get("type", "message"), "data": json.dumps(evt)}
                if evt.get("type") in ("done", "error"):
                    exit_reason = f"terminal_{evt.get('type')}"
                    return
        except asyncio.CancelledError:
            exit_reason = "cancelled"
            raise
        finally:
            if next_task is not None and not next_task.done():
                next_task.cancel()
            logger.info(
                "stream %s closed after %.2fs: %d events, %d pings, reason=%s",
                run_id, time.time() - t_open, n_events, n_pings, exit_reason,
            )

    return EventSourceResponse(
        gen(),
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache, no-transform",
        },
    )
