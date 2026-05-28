"""Unit tests for memory_safety helpers.

Uses fake `_runner` injections to drive the parsers with canned
output, so the tests run anywhere (Linux CI, non-MPS macOS) without
depending on the real system's vm_stat / sysctl state.

The watchdog test runs a real background thread with a 0.05s poll
period, so it's fast (sub-second) but exercises the real
``threading.Event`` plumbing.
"""

from __future__ import annotations

import threading
import time

from cells_interlinked.pipeline.edge_consumer.memory_safety import (
    MemoryWatchdog,
    mps_empty_cache_safe,
    pre_flight_memory_check,
    vm_stat_free_gb,
    vm_stat_free_pages,
    vm_swap_usage_gb,
)


# ── Parser fixtures ──────────────────────────────────────────────────


VMSTAT_LOW_FREE = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                3994.
Pages active:                            464626.
Pages inactive:                          459235.
Pages speculative:                         8780.
Pages throttled:                              0.
Pages wired down:                       1817706.
"""

VMSTAT_HIGH_FREE = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                             3213771.
Pages active:                              5000.
Pages inactive:                          347420.
Pages speculative:                            0.
Pages throttled:                              0.
Pages wired down:                        212955.
"""

SWAP_HEAVY = "vm.swapusage: total = 24576.00M  used = 24007.69M  free = 568.31M  (encrypted)"
SWAP_NORMAL = "vm.swapusage: total = 3072.00M  used = 1784.62M  free = 1287.38M  (encrypted)"
SWAP_ZERO = "vm.swapusage: total = 0.00M  used = 0.00M  free = 0.00M  (encrypted)"


# ── vm_stat parsing ─────────────────────────────────────────────────


def test_vm_stat_free_pages_high():
    pages = vm_stat_free_pages(_output_runner=lambda: VMSTAT_HIGH_FREE)
    assert pages == 3213771, pages


def test_vm_stat_free_gb_high():
    gb = vm_stat_free_gb(_output_runner=lambda: VMSTAT_HIGH_FREE)
    # 3,213,771 × 16384 bytes / 1024**3 = 49.04 GB
    assert 48.5 < gb < 49.5, gb


def test_vm_stat_free_gb_low():
    gb = vm_stat_free_gb(_output_runner=lambda: VMSTAT_LOW_FREE)
    # 3,994 × 16 KB = ~62 MB = 0.06 GB
    assert 0.0 < gb < 0.2, gb


def test_vm_stat_unavailable_returns_huge():
    """When vm_stat doesn't exist (Linux CI), parser should return a
    sentinel huge value so the watchdog doesn't false-trip."""
    def boom():
        raise FileNotFoundError("vm_stat")
    pages = vm_stat_free_pages(_output_runner=boom)
    assert pages > 10 ** 10, pages


# ── swap parsing ─────────────────────────────────────────────────────


def test_swap_usage_heavy():
    gb = vm_swap_usage_gb(_output_runner=lambda: SWAP_HEAVY)
    # 24007.69 MB ≈ 23.45 GB
    assert 23.0 < gb < 24.0, gb


def test_swap_usage_normal():
    gb = vm_swap_usage_gb(_output_runner=lambda: SWAP_NORMAL)
    # 1784.62 MB ≈ 1.74 GB
    assert 1.5 < gb < 2.0, gb


def test_swap_usage_zero():
    gb = vm_swap_usage_gb(_output_runner=lambda: SWAP_ZERO)
    assert gb == 0.0, gb


# ── pre_flight_memory_check ──────────────────────────────────────────


def test_preflight_ok_when_plenty_free():
    r = pre_flight_memory_check(
        min_free_gb=10.0,
        max_swap_used_gb=4.0,
        _vm_stat_runner=lambda: VMSTAT_HIGH_FREE,
        _swap_runner=lambda: SWAP_NORMAL,
    )
    assert r.ok, r.message
    assert "OK" in r.message


def test_preflight_fails_on_low_free_ram():
    r = pre_flight_memory_check(
        min_free_gb=30.0,
        max_swap_used_gb=4.0,
        _vm_stat_runner=lambda: VMSTAT_LOW_FREE,
        _swap_runner=lambda: SWAP_NORMAL,
    )
    assert not r.ok
    assert "free" in r.message


def test_preflight_fails_on_heavy_swap_even_with_free_ram():
    r = pre_flight_memory_check(
        min_free_gb=10.0,
        max_swap_used_gb=4.0,
        _vm_stat_runner=lambda: VMSTAT_HIGH_FREE,
        _swap_runner=lambda: SWAP_HEAVY,
    )
    assert not r.ok
    assert "swap" in r.message


# ── mps_empty_cache_safe ─────────────────────────────────────────────


def test_mps_empty_cache_safe_does_not_throw():
    """No-op on systems without MPS; doesn't crash on systems with MPS."""
    mps_empty_cache_safe()


# ── MemoryWatchdog ───────────────────────────────────────────────────


def test_watchdog_does_not_trip_when_memory_is_fine():
    wd = MemoryWatchdog(
        free_gb_floor=1.0,
        swap_gb_ceiling=100.0,
        poll_seconds=0.05,
        log_every=1000,  # silence
        _vm_stat_runner=lambda: VMSTAT_HIGH_FREE,
        _swap_runner=lambda: SWAP_NORMAL,
    )
    wd.start()
    time.sleep(0.2)
    wd.stop()
    assert not wd.tripped
    assert not wd.cancel_event.is_set()


def test_watchdog_trips_on_low_free_ram():
    wd = MemoryWatchdog(
        free_gb_floor=10.0,  # 10 GB floor with only 62 MB free → trip
        swap_gb_ceiling=100.0,
        poll_seconds=0.05,
        log_every=1000,
        _vm_stat_runner=lambda: VMSTAT_LOW_FREE,
        _swap_runner=lambda: SWAP_NORMAL,
    )
    wd.start()
    # Give the loop one or two ticks to trip.
    for _ in range(20):
        if wd.tripped:
            break
        time.sleep(0.05)
    wd.stop()
    assert wd.tripped, "watchdog should have tripped on low free RAM"
    assert wd.cancel_event.is_set()
    assert "free RAM" in wd.trip_reason


def test_watchdog_trips_on_heavy_swap():
    wd = MemoryWatchdog(
        free_gb_floor=0.001,
        swap_gb_ceiling=4.0,  # 4 GB ceiling with 24 GB swap → trip
        poll_seconds=0.05,
        log_every=1000,
        _vm_stat_runner=lambda: VMSTAT_HIGH_FREE,
        _swap_runner=lambda: SWAP_HEAVY,
    )
    wd.start()
    for _ in range(20):
        if wd.tripped:
            break
        time.sleep(0.05)
    wd.stop()
    assert wd.tripped, "watchdog should have tripped on heavy swap"
    assert "swap" in wd.trip_reason


def test_watchdog_stop_is_idempotent():
    """stop() must be safe to call multiple times (we use it in finally
    blocks, possibly after we already called it in the happy path)."""
    wd = MemoryWatchdog(
        free_gb_floor=0.001,
        swap_gb_ceiling=100.0,
        poll_seconds=0.5,
        log_every=1000,
        _vm_stat_runner=lambda: VMSTAT_HIGH_FREE,
        _swap_runner=lambda: SWAP_NORMAL,
    )
    wd.start()
    wd.stop()
    wd.stop()  # second call — must not throw


def test_watchdog_uses_provided_cancel_event():
    """If the caller passes its own cancel_event, the watchdog should
    set THAT event (so pipelines watching their own event get
    notified)."""
    external = threading.Event()
    wd = MemoryWatchdog(
        free_gb_floor=10.0,
        swap_gb_ceiling=100.0,
        poll_seconds=0.05,
        log_every=1000,
        cancel_event=external,
        _vm_stat_runner=lambda: VMSTAT_LOW_FREE,
        _swap_runner=lambda: SWAP_NORMAL,
    )
    wd.start()
    for _ in range(20):
        if external.is_set():
            break
        time.sleep(0.05)
    wd.stop()
    assert external.is_set()


if __name__ == "__main__":
    fns = [
        test_vm_stat_free_pages_high,
        test_vm_stat_free_gb_high,
        test_vm_stat_free_gb_low,
        test_vm_stat_unavailable_returns_huge,
        test_swap_usage_heavy,
        test_swap_usage_normal,
        test_swap_usage_zero,
        test_preflight_ok_when_plenty_free,
        test_preflight_fails_on_low_free_ram,
        test_preflight_fails_on_heavy_swap_even_with_free_ram,
        test_mps_empty_cache_safe_does_not_throw,
        test_watchdog_does_not_trip_when_memory_is_fine,
        test_watchdog_trips_on_low_free_ram,
        test_watchdog_trips_on_heavy_swap,
        test_watchdog_stop_is_idempotent,
        test_watchdog_uses_provided_cancel_event,
    ]
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            raise
        except Exception as e:
            print(f"  ERR   {fn.__name__}: {type(e).__name__}: {e}")
            raise
    print(f"\n{len(fns)}/{len(fns)} tests passed")
