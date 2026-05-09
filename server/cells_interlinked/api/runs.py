"""Per-run state held server-side. Each run has an event log, cancel event,
and task handle.

The event log is append-only and supports multiple concurrent subscribers,
each tracking its own position. New subscribers replay the full backlog
before tailing the live stream — so a user who navigates away mid-run
and returns gets the complete picture, not just the events emitted after
they reconnect.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


class EventLog:
    """Append-only log of SSE events for one probe run.

    Replay semantics: stream_from(0) replays every event ever appended
    (in order), then blocks waiting for new events as they arrive. When
    `close()` has been called and the consumer reaches the end, the
    iterator returns gracefully — telling SSE handlers "this run is
    done, no more events will come."

    Single-writer / multi-reader. The producer (one per probe in
    routes_probe._execute_probe) calls append/close. Each subscriber
    (SSE handler, autorun drain) gets its own AsyncIterator.
    """

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

    @property
    def closed(self) -> bool:
        return self._closed

    def __len__(self) -> int:
        return len(self._events)

    async def stream_from(self, start_idx: int = 0) -> AsyncIterator[dict]:
        i = start_idx
        while True:
            async with self._cond:
                while i >= len(self._events) and not self._closed:
                    await self._cond.wait()
                if i >= len(self._events):
                    return  # closed and caught up
                evt = self._events[i]
                i += 1
            yield evt


@dataclass
class RunState:
    run_id: str
    prompt_text: str
    event_log: EventLog = field(default_factory=EventLog)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    completed: bool = False

    async def emit(self, evt: dict) -> None:
        """Convenience: append to the event log. Replaces v1's
        `state.queue.put(...)` calls."""
        await self.event_log.append(evt)


class RunRegistry:
    def __init__(self) -> None:
        self._runs: dict[str, RunState] = {}
        self._lock = asyncio.Lock()  # serializes generation; one model, one stream

    def add(self, run: RunState) -> None:
        self._runs[run.run_id] = run

    def get(self, run_id: str) -> RunState | None:
        return self._runs.get(run_id)

    def remove(self, run_id: str) -> None:
        self._runs.pop(run_id, None)

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock
