"""Memory safety helpers for multi-hour MPS pipelines.

Born out of the 2026-05-27 → 28 overnight, where ~6 hours of
``model.generate()`` + hook install/remove cycles fragmented MPS's
allocator until the system hit 24 GB of swap and killed the user's
Docker container. See ``docs/MEMORY_PRESSURE_LESSONS.md`` for the
postmortem.

Three layers of protection:

1. **Pre-flight check** (``pre_flight_memory_check``) — run once at
   script start, BEFORE loading M. Refuses to proceed when free RAM
   is below a floor that would leave M (~24 GB) plus headroom plus
   other user processes (Docker, IDE) without room. This is the
   "did the user forget to stop the backend?" guard, enforced in
   code instead of just in CLAUDE.md.

2. **MPS empty_cache** (``mps_empty_cache_safe``) — drop-in between
   iterations of long generation loops. Tells PyTorch's MPS allocator
   to release cached blocks back to the OS, preventing the slow
   monotonic working-set growth that the overnight hit.

3. **Watchdog thread** (``MemoryWatchdog``) — runs in the background
   while compute is happening. Polls ``vm_stat`` and ``sysctl
   vm.swapusage`` periodically; trips a ``cancel_event`` if free RAM
   drops below a floor or swap usage exceeds a ceiling. Long pipelines
   honor the event between iterations and write their partial
   artifact cleanly, rather than fighting the macOS OOM killer.

All three are no-ops on non-MPS systems (e.g. CI / Linux test boxes).
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


PAGE_SIZE_BYTES = 16384  # macOS on Apple Silicon uses 16 KB pages
GB = 1024 ** 3


# ── Low-level system queries ─────────────────────────────────────────


def _vm_stat_output(_runner: Callable[..., str] | None = None) -> str:
    """Run vm_stat and return its text output. Indirected so tests can
    inject a fake `_runner` returning canned output."""
    if _runner is not None:
        return _runner()
    return subprocess.check_output(["vm_stat"], text=True, timeout=5)


def _swapusage_output(_runner: Callable[..., str] | None = None) -> str:
    """Run `sysctl vm.swapusage` and return its text output."""
    if _runner is not None:
        return _runner()
    return subprocess.check_output(
        ["sysctl", "vm.swapusage"], text=True, timeout=5,
    )


def vm_stat_free_pages(*, _output_runner: Callable | None = None) -> int:
    """Parse vm_stat's 'Pages free' line. Returns 0 on macOS where
    available, or a sentinel huge int on platforms where vm_stat
    isn't present (so calling code sees 'lots of memory free' and
    doesn't trip the watchdog)."""
    try:
        out = _vm_stat_output(_output_runner)
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        logger.debug("vm_stat unavailable (%s); treating as unbounded", e)
        return 10 ** 12  # ~16 PB; effectively unbounded
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Pages free:"):
            tail = line.split(":", 1)[1].strip().rstrip(".")
            try:
                return int(tail)
            except ValueError:
                logger.warning("could not parse vm_stat line: %r", line)
                return 10 ** 12
    logger.warning("'Pages free:' not in vm_stat output")
    return 10 ** 12


def vm_stat_free_gb(*, _output_runner: Callable | None = None) -> float:
    """Convenience: free pages × 16 KB → GB."""
    return vm_stat_free_pages(_output_runner=_output_runner) * PAGE_SIZE_BYTES / GB


def vm_swap_usage_gb(*, _output_runner: Callable | None = None) -> float:
    """Parse `sysctl vm.swapusage` and return the 'used' value in GB.

    The line looks like:
        vm.swapusage: total = 3072.00M  used = 1784.62M  free = 1287.38M  (encrypted)
    We extract the 'used = X.YM' or 'used = X.YG' value.
    """
    try:
        out = _swapusage_output(_output_runner)
    except (FileNotFoundError, subprocess.SubprocessError):
        return 0.0
    # Find "used = X.YM" — sysctl format. Walk tokens after the
    # "used" keyword and pick the first numeric token with a unit
    # suffix. Format example:
    #   "vm.swapusage: total = 3072.00M  used = 1784.62M  free = 1287.38M ..."
    idx = out.find("used")
    if idx < 0:
        return 0.0
    multipliers = {"K": 1 / (1024 * 1024), "M": 1 / 1024, "G": 1.0}
    for token in out[idx:].split()[1:]:  # skip the literal "used"
        if not token or token[-1] not in multipliers or len(token) < 2:
            continue
        try:
            value = float(token[:-1])
        except ValueError:
            continue
        return value * multipliers[token[-1]]
    return 0.0


# ── MPS empty_cache wrapper ──────────────────────────────────────────


def mps_empty_cache_safe() -> None:
    """Call torch.mps.empty_cache() when available; no-op otherwise.

    Safe to call from any context (CPU / CUDA / MPS / no torch). Costs
    ~50–100 ms when it does fire; cheap enough to call between every
    iteration of a generation loop.
    """
    try:
        import torch
    except ImportError:
        return
    mps_mod = getattr(torch, "mps", None)
    if mps_mod is None:
        return
    empty = getattr(mps_mod, "empty_cache", None)
    if callable(empty):
        try:
            empty()
        except Exception:
            logger.debug("torch.mps.empty_cache raised", exc_info=True)


# ── Pre-flight gate ──────────────────────────────────────────────────


@dataclass
class MemoryPreflightResult:
    free_gb: float
    swap_used_gb: float
    min_free_gb_required: float
    ok: bool
    message: str


def pre_flight_memory_check(
    *,
    min_free_gb: float = 30.0,
    max_swap_used_gb: float = 4.0,
    _vm_stat_runner: Callable | None = None,
    _swap_runner: Callable | None = None,
) -> MemoryPreflightResult:
    """Inspect system memory and refuse to proceed if it's tight.

    Defaults: at least 30 GB free RAM (M is ~24 GB and we want
    headroom for the user's other processes) AND swap-used below
    4 GB (sustained non-zero swap is the canary that memory is
    already under pressure).

    Returns a result struct rather than raising — callers decide
    whether to abort or continue with a warning.
    """
    free_gb = vm_stat_free_gb(_output_runner=_vm_stat_runner)
    swap_gb = vm_swap_usage_gb(_output_runner=_swap_runner)
    problems: list[str] = []
    if free_gb < min_free_gb:
        problems.append(
            f"only {free_gb:.1f} GB free, need ≥ {min_free_gb:.1f} GB "
            f"(is the backend / another model job already running?)"
        )
    if swap_gb > max_swap_used_gb:
        problems.append(
            f"swap usage {swap_gb:.1f} GB exceeds ceiling "
            f"{max_swap_used_gb:.1f} GB (system already under pressure)"
        )
    if problems:
        return MemoryPreflightResult(
            free_gb=free_gb,
            swap_used_gb=swap_gb,
            min_free_gb_required=min_free_gb,
            ok=False,
            message="pre-flight FAIL: " + "; ".join(problems),
        )
    return MemoryPreflightResult(
        free_gb=free_gb,
        swap_used_gb=swap_gb,
        min_free_gb_required=min_free_gb,
        ok=True,
        message=(
            f"pre-flight OK: {free_gb:.1f} GB free, "
            f"swap {swap_gb:.1f} GB"
        ),
    )


# ── Watchdog ─────────────────────────────────────────────────────────


@dataclass
class WatchdogReading:
    free_gb: float
    swap_used_gb: float


class MemoryWatchdog:
    """Background thread that polls system memory and trips a
    cancel_event when thresholds are crossed.

    Usage:

        watchdog = MemoryWatchdog(
            free_gb_floor=2.0,
            swap_gb_ceiling=8.0,
            poll_seconds=30.0,
        )
        watchdog.start()
        try:
            run_long_pipeline(cancel_event=watchdog.cancel_event)
        finally:
            watchdog.stop()
        if watchdog.tripped:
            print(f"watchdog tripped: {watchdog.trip_reason}")

    Honors the cancel_event by setting it; pipelines must check the
    event between iterations and exit gracefully.
    """

    def __init__(
        self,
        *,
        free_gb_floor: float = 2.0,
        swap_gb_ceiling: float = 8.0,
        poll_seconds: float = 30.0,
        cancel_event: Optional[threading.Event] = None,
        _vm_stat_runner: Callable | None = None,
        _swap_runner: Callable | None = None,
        log_every: int = 10,
    ) -> None:
        self.free_gb_floor = float(free_gb_floor)
        self.swap_gb_ceiling = float(swap_gb_ceiling)
        self.poll_seconds = float(poll_seconds)
        self.cancel_event = cancel_event or threading.Event()
        self._vm_stat_runner = _vm_stat_runner
        self._swap_runner = _swap_runner
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._log_every = max(1, int(log_every))
        self.tripped: bool = False
        self.trip_reason: str = ""
        self.last_reading: Optional[WatchdogReading] = None

    def _read(self) -> WatchdogReading:
        return WatchdogReading(
            free_gb=vm_stat_free_gb(_output_runner=self._vm_stat_runner),
            swap_used_gb=vm_swap_usage_gb(_output_runner=self._swap_runner),
        )

    def _loop(self) -> None:
        poll_count = 0
        while not self._stop_event.is_set():
            try:
                reading = self._read()
            except Exception:
                logger.exception("memory watchdog read failed; continuing")
                self._stop_event.wait(self.poll_seconds)
                continue
            self.last_reading = reading
            poll_count += 1
            if poll_count % self._log_every == 0:
                logger.info(
                    "memory watchdog: free=%.1f GB, swap_used=%.1f GB "
                    "(thresholds: free≥%.1f, swap≤%.1f)",
                    reading.free_gb, reading.swap_used_gb,
                    self.free_gb_floor, self.swap_gb_ceiling,
                )
            reasons: list[str] = []
            if reading.free_gb < self.free_gb_floor:
                reasons.append(
                    f"free RAM {reading.free_gb:.2f} GB < floor "
                    f"{self.free_gb_floor:.1f} GB"
                )
            if reading.swap_used_gb > self.swap_gb_ceiling:
                reasons.append(
                    f"swap used {reading.swap_used_gb:.2f} GB > ceiling "
                    f"{self.swap_gb_ceiling:.1f} GB"
                )
            if reasons:
                self.trip_reason = " ; ".join(reasons)
                self.tripped = True
                logger.warning(
                    "MEMORY WATCHDOG TRIPPED: %s — setting cancel_event",
                    self.trip_reason,
                )
                self.cancel_event.set()
                return
            self._stop_event.wait(self.poll_seconds)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("watchdog already started")
        self._thread = threading.Thread(
            target=self._loop, name="memory-watchdog", daemon=True,
        )
        self._thread.start()
        logger.info(
            "memory watchdog armed: free≥%.1f GB, swap≤%.1f GB, "
            "poll=%.1fs", self.free_gb_floor, self.swap_gb_ceiling,
            self.poll_seconds,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Cleanly shut down the watchdog thread. Safe to call
        multiple times."""
        self._stop_event.set()
        t = self._thread
        if t is None:
            return
        t.join(timeout=timeout)
        self._thread = None


__all__ = [
    "PAGE_SIZE_BYTES",
    "GB",
    "MemoryPreflightResult",
    "MemoryWatchdog",
    "WatchdogReading",
    "mps_empty_cache_safe",
    "pre_flight_memory_check",
    "vm_stat_free_gb",
    "vm_stat_free_pages",
    "vm_swap_usage_gb",
]
