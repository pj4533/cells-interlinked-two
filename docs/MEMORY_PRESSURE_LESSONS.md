# Memory Pressure on Multi-Hour MPS Pipelines

> What happened during the 2026-05-27 → 28 overnight, what the cause
> was, and what to add to future scripts before running anything
> that loops `model.generate()` for hours.

---

## What happened

The overnight ran for ~6 hours. Memory rose slowly across iterations.
By the time the user noticed (~5.5 hr in), the system was at:

- **swap used: 24.0 GB / 24.6 GB total** (98%)
- **free RAM: 62 MB** (3,994 × 16 KB pages)
- **swapouts since process start: 69 million**
- A Docker container running an agent process failed because there
  was no memory left for it to allocate

After killing our Python pipeline:

- Swap shrunk to **1.8 GB used** (macOS auto-shrank the swap
  allocation since the pressure dropped)
- Free RAM: **~50 GB**

So the entire 24 GB of swap was attributable to our pipeline. There
was no leak in the conventional sense (no growing list of Python
tensors) — the leak was at the Metal-allocator level on MPS.

---

## Why MPS leaks (the actual mechanism)

PyTorch on CUDA exposes `torch.cuda.empty_cache()` which tells the
allocator to release cached blocks back to the driver. The MPS
backend has `torch.mps.empty_cache()` available since PyTorch 2.0,
but **none of our pipeline code calls it**.

What accumulates:

1. **`model.generate()` KV cache** — each call allocates per-position
   K/V tensors. The cache is released to PyTorch's *cache* on call
   return, but PyTorch keeps the cached blocks in case the next call
   wants them. On MPS, those cached blocks sit in unified memory and
   don't get returned to the OS until `empty_cache()` is called.

2. **Forward-hook closures** — every `install_edge_consumer_ablation_hook`
   call creates closures holding references to `v_safety`, `mask_flat`,
   `proj_v_per_head` tensors. Per iteration we install ~3 hooks per
   affected layer × ~15 layers = ~45 closures. They're freed when
   `.remove()` runs, but the underlying tensor buffers stay in the
   allocator's cache.

3. **Fragmentation.** Hundreds of hook install/remove cycles plus
   hundreds of generation calls fragment the allocator's free list.
   Even when total live memory is reasonable, the allocator can't
   coalesce free blocks into contiguous regions large enough for new
   allocations, so it asks the OS for fresh pages.

The combination produces slow, monotonic growth in the process's
working set over hours. At hour 5–6 it crossed the threshold where
macOS started aggressively paging out other processes (including the
user's Docker container).

---

## Prevention checklist

Before running any pipeline that loops `model.generate()` or
hook-install cycles for more than ~30 minutes:

### 1. Call `empty_cache` between iterations

The cheapest mitigation. Add this to the inner loop of subset-compose
and similar long-running scorers:

```python
import torch
# After each iteration that did a generate() + hook install/remove:
if hasattr(torch.mps, "empty_cache"):
    torch.mps.empty_cache()
```

This tells the MPS allocator to release cached blocks back to the OS.
Costs ~50–100 ms per call. Calling it every 4–8 iterations is fine;
calling it after EVERY iteration is overkill but harmless.

### 2. Hard memory watchdog inside the script

A background thread that polls `vm_stat` and trips the cancel event
if free pages drop below a threshold:

```python
import subprocess, threading, time
PAGE_SIZE = 16384  # 16 KB pages on M2
FREE_PAGES_FLOOR = 65536  # 1 GB free → bail

def _vm_stat_free_pages() -> int:
    out = subprocess.check_output(["vm_stat"], text=True)
    for line in out.splitlines():
        if line.startswith("Pages free:"):
            return int(line.split(":")[1].strip().rstrip("."))
    return 10**9

def memory_watchdog(cancel_event: threading.Event, period: float = 30.0):
    while not cancel_event.is_set():
        free = _vm_stat_free_pages()
        if free < FREE_PAGES_FLOOR:
            print(f"WATCHDOG: free pages = {free} (< {FREE_PAGES_FLOOR}). "
                  f"Triggering cancel.", flush=True)
            cancel_event.set()
            return
        time.sleep(period)
```

Drive long pipelines through this `cancel_event` so the script
gracefully wraps up and writes its partial artifact instead of
fighting the OOM killer.

### 3. Pre-flight check: enough free RAM + swap

Add to every overnight script's startup:

```python
# Require ≥ 30 GB free pages before starting a multi-hour run.
# M's weights alone are 24 GB; we want some headroom plus room for
# the user's Docker, IDE, etc.
free_pages = _vm_stat_free_pages()
free_gb = free_pages * 16384 / (1024 ** 3)
if free_gb < 30:
    print(f"abort: only {free_gb:.1f} GB free, need ≥ 30 GB")
    sys.exit(1)
```

This catches the case where the user forgot to stop the backend
before launching the overnight (the existing CLAUDE.md instruction
that we keep needing to enforce in code rather than docs).

### 4. Bound generate-call count per process lifetime

A hard cap on `model.generate()` calls per process: if you need more
than, say, 5000 calls, structure the work into chunks that exit and
re-enter the process. Each fresh process starts with a clean
allocator. We hit roughly:

  Step 3 subset compose, ε=0.02: 120 iter × 50 prompts = 6000 calls
  Step 3 subset compose, ε=0.05: 120 iter × 50 prompts = 6000 calls
  Step 3 subset compose, ε=0.10: 120 iter × 50 prompts = 6000 calls

18,000 generate calls in one process is well past where MPS starts
fragmenting. Splitting into three processes (one per ε) plus a
top-level orchestrator script would keep each process's call count
below the fragmentation threshold.

### 5. Monitor swap, not just free RAM

`vm_stat` only shows free pages. Swap usage is a leading indicator:
macOS will swap aggressively long before free pages drop to zero. The
watchdog should also check:

```python
out = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
# parse "total = X.YM  used = Z.WM  free = ..." line
```

Trip the cancel event if swap **used** exceeds e.g. 4 GB. On a clean
M2 Ultra with only M loaded, swap should stay near 0; sustained
non-zero swap usage is the canary.

---

## Apply to existing scripts before next overnight

The scripts that produced last night's run:

- `server/scripts/run_edge_consumer_pipeline.py`
- `server/scripts/run_signed_attribution_and_subset.py`
- `server/scripts/run_group_signed_attribution.py`

All call `model.generate()` in long loops without `empty_cache`. Any
re-run will hit the same wall. Before the next overnight on this
class of pipeline, retrofit at minimum:

1. `torch.mps.empty_cache()` in the inner loop of `subset_compose`
   and `verdict`.
2. The 30 GB-free pre-flight check at script start.
3. The watchdog thread (with both free-pages AND swap-usage checks),
   plumbed through the existing `cancel_event` pattern.

These are 2–3 hours of work. Worth doing before any further
edge-consumer / MLP-ablation / Burgess-style experiments are run.

---

## What I should have done last night

Two specific lapses:

1. **Should have killed the run after ε=0.02 exhausted at iter=120.**
   At that point we had the publishable finding. The ε=0.05 and
   ε=0.10 sweeps were deterministic replay — same prompts, same
   greedy decode, same hook installs → identical data points.
   Leaving them running for "tidy artifacts" was zero new
   information at the cost of 6+ hours of compute and the eventual
   swap thrash.

2. **Should have surfaced a memory check well before 5 hours.** The
   monitor was tailing the log file for compute events; it wasn't
   watching system memory. Adding `vm_stat` checks (e.g. one per
   hour) to the monitor's event filter would have caught the swap
   creep ~3 hours earlier than the user noticing.

Both are correctable for next time.
