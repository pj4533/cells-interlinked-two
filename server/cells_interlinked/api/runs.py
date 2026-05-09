"""Per-run state held server-side. Each run has an event log, cancel event,
and task handle.

The event log is append-only and supports multiple concurrent subscribers,
each tracking its own position. New subscribers replay the full backlog
before tailing the live stream — so a user who navigates away mid-run
and returns gets the complete picture, not just the events emitted after
they reconnect.

The registry also exposes a fair, queue-tracked acquire() context
manager — only one probe holds compute at a time (MPS is one device),
but the registry tells callers their position in the queue so the UI
can surface "queued behind run X — position 2" instead of sitting
silently while another probe runs.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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
    """One probe's compute at a time. MPS is one device — two concurrent
    `model.generate()` calls just slice the GPU and finish in 2× the
    wall-clock with no throughput gain. So we serialize via a lock, but
    track the queue (current holder + ordered waiters) so the UI can
    surface "queued behind run X" instead of looking frozen.
    """

    def __init__(self) -> None:
        self._runs: dict[str, RunState] = {}
        self._lock = asyncio.Lock()
        self._holder: str | None = None
        self._waiters: list[str] = []

    def add(self, run: RunState) -> None:
        self._runs[run.run_id] = run

    def get(self, run_id: str) -> RunState | None:
        return self._runs.get(run_id)

    def remove(self, run_id: str) -> None:
        self._runs.pop(run_id, None)

    @property
    def lock(self) -> asyncio.Lock:
        # Backward-compat raw access; new code should prefer acquire().
        return self._lock

    @property
    def holder_run_id(self) -> str | None:
        return self._holder

    @property
    def waiters(self) -> list[str]:
        return list(self._waiters)

    def position_of(self, run_id: str) -> int | None:
        """0 = currently holding the lock; 1 = next up; None = not queued."""
        if self._holder == run_id:
            return 0
        if run_id in self._waiters:
            return self._waiters.index(run_id) + 1
        return None

    @asynccontextmanager
    async def acquire(self, run_id: str) -> AsyncIterator[None]:
        """Fair-FIFO acquire of the compute lock. Tracks queue position
        so observers can poll for "where am I in line?". If the awaiting
        coroutine is cancelled while in the queue, cleans up the waiters
        list before re-raising."""
        self._waiters.append(run_id)
        try:
            await self._lock.acquire()
        except BaseException:
            if run_id in self._waiters:
                self._waiters.remove(run_id)
            raise
        # Lock owned. Promote ourselves out of the waiters list and into
        # the holder slot.
        try:
            if run_id in self._waiters:
                self._waiters.remove(run_id)
            self._holder = run_id
            yield
        finally:
            self._holder = None
            self._lock.release()

    def snapshot(self) -> dict:
        return {
            "holder_run_id": self._holder,
            "waiters": list(self._waiters),
            "waiters_count": len(self._waiters),
        }
