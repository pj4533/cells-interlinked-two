"""ModelManager — owns M and AV with strict serial loading.

CI 2.5 hardware constraint: M (~24 GiB) and AV (~24 GiB) both bf16 on
MPS, on a 64 GiB unified-memory box. With other apps + the OS, holding
both resident pushes the working set past physical RAM and causes
catastrophic swap thrashing. Symptoms we've measured: 29 GiB of disk-
backed swap, 110% CPU but no token progress for minutes.

So this manager enforces the invariant: **at most one of {M, AV} is
loaded at any time.** Phase 1 (generation + residual capture) needs M;
phase 2 (NLA decoding) needs AV; the judge pass needs M back. The
manager handles the swaps and emits status events so the UI can show
"loading M..." / "unloading AV..." cues during the otherwise-quiet
interval.

The SAE secondary instrument is small (~1 GiB on CPU) and lives
alongside M (loaded when M loads, unloaded when M unloads). Refusal
directions are tiny and stay resident.

Trade-off: each model swap costs ~15s of wall-clock time (warm load
from disk cache → MPS). For a 20-min probe, that's ~3.5% overhead.
For a 100-probe autorun, that's ~75 min of cumulative load. We pay
this in exchange for never thrashing swap, which would otherwise add
HOURS per probe under load.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import torch

from ..config import settings
from .abliteration import load_directions
from .model_loader import load_model, ModelBundle
from .nla_client import NLAClient

logger = logging.getLogger(__name__)


# Callback type for status messages. The manager calls this whenever a
# model transition starts or finishes; the route layer wires it to
# state.emit so SSE clients see the status. Async so it can publish to
# an asyncio.Queue. Returning a non-awaitable is also tolerated.
StatusEmitter = Callable[[dict[str, Any]], Awaitable[None] | None]


class ModelManager:
    """Serial loader for M, AV, and SAE. Refusal directions stay
    resident; SAE rides with M; M and AV are mutually exclusive."""

    def __init__(self) -> None:
        # The two big residents. None when unloaded.
        self.bundle: ModelBundle | None = None
        self.nla: NLAClient | None = None
        # Always-resident small data.
        self.refusal_directions: torch.Tensor | None = None
        # Lock so we don't race two probes trying to swap models at the
        # same time. The probe registry already serializes runs, but
        # we hold this lock too for safety in case a non-probe code
        # path (e.g. a future compute script) wants to use the manager.
        self._lock = asyncio.Lock()

    # ── Static init ───────────────────────────────────────────────────

    async def init_static(self) -> None:
        """One-shot startup: load refusal_directions. Mark M and AV
        unloaded. Called from the FastAPI lifespan."""
        # Refusal directions — tiny (~1 MB).
        rd_path = settings.db_path.parent / "refusal_directions.pt"
        try:
            directions, meta = load_directions(rd_path)
            if meta.get("model_name") != settings.model_name:
                logger.warning(
                    "refusal_directions.pt was computed for %r but config M is %r; "
                    "skipping load.",
                    meta.get("model_name"), settings.model_name,
                )
            elif meta.get("d_model") and meta.get("d_model") != _expected_d_model():
                logger.warning(
                    "refusal_directions.pt d_model=%d mismatches config; skipping load.",
                    meta.get("d_model"),
                )
            else:
                self.refusal_directions = directions
                logger.info(
                    "ready: refusal directions loaded for L%d (shape=%s, dtype=%s, variant=%s)",
                    meta.get("extraction_layer_for_ci25"),
                    tuple(directions.shape), str(directions.dtype),
                    meta.get("variant_name", "unknown"),
                )
        except FileNotFoundError:
            logger.info(
                "no refusal_directions.pt at %s — ablated NLA decode unavailable",
                rd_path,
            )
        except Exception:
            logger.exception("failed to load refusal_directions.pt; continuing without")

        logger.info(
            "ModelManager ready: M=%s AV=%s (both unloaded; will load on demand)",
            settings.model_name, settings.av_repo,
        )

    # ── Public API ────────────────────────────────────────────────────

    async def acquire_m(
        self, emit: StatusEmitter | None = None,
    ) -> ModelBundle:
        """Ensure M (and its companion SAE) is loaded. If AV is loaded,
        unloads it first. Returns the bundle, ready for forward passes."""
        async with self._lock:
            if self.bundle is not None:
                return self.bundle
            if self.nla is not None:
                await self._unload_av(emit)
            await self._load_m(emit)
            assert self.bundle is not None
            return self.bundle

    async def acquire_av(
        self, emit: StatusEmitter | None = None,
    ) -> NLAClient:
        """Ensure AV is loaded. If M is loaded, unloads it first.
        Returns the NLA client."""
        async with self._lock:
            if self.nla is not None:
                return self.nla
            if self.bundle is not None:
                await self._unload_m(emit)
            await self._load_av(emit)
            assert self.nla is not None
            return self.nla

    async def release_all(self, emit: StatusEmitter | None = None) -> None:
        """Tear down whatever is loaded. Called on shutdown."""
        async with self._lock:
            if self.bundle is not None:
                await self._unload_m(emit)
            if self.nla is not None:
                await self._unload_av(emit)

    # ── Internal load/unload (must hold _lock) ───────────────────────

    async def _load_m(self, emit: StatusEmitter | None) -> None:
        await _emit(emit, {
            "type": "phase",
            "name": "loading_m",
            "message": f"Loading {settings.model_name}...",
        })
        t0 = time.time()
        dtype = _resolve_dtype(settings.dtype)
        self.bundle = await asyncio.to_thread(
            load_model,
            settings.model_name,
            device_str=settings.device,
            dtype=dtype,
            extraction_layer=settings.extraction_layer,
        )
        logger.info(
            "loaded M in %.1fs (layers=%d hidden=%d)",
            time.time() - t0,
            self.bundle.num_layers,
            self.bundle.hidden_dim,
        )
        await _emit(emit, {
            "type": "phase",
            "name": "m_loaded",
            "message": f"M loaded in {time.time() - t0:.1f}s",
        })

    async def _unload_m(self, emit: StatusEmitter | None) -> None:
        await _emit(emit, {
            "type": "phase",
            "name": "unloading_m",
            "message": "Unloading M to free memory...",
        })
        t0 = time.time()
        # Drop M. torch.mps.empty_cache() is the load-bearing bit —
        # without it, MPS holds onto the allocator pool and the memory
        # isn't actually freed.
        self.bundle = None
        gc.collect()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            logger.exception("torch.mps.empty_cache() failed")
        logger.info("unloaded M in %.1fs", time.time() - t0)

    async def _load_av(self, emit: StatusEmitter | None) -> None:
        await _emit(emit, {
            "type": "phase",
            "name": "loading_av",
            "message": f"Loading AV ({settings.av_repo})...",
        })
        t0 = time.time()
        dtype = _resolve_dtype(settings.dtype)
        self.nla = await asyncio.to_thread(
            NLAClient,
            settings.av_repo,
            device_str=settings.device,
            dtype=dtype,
        )
        logger.info(
            "loaded AV in %.1fs (d_model=%d, extraction_layer=L%d)",
            time.time() - t0,
            self.nla.sidecar.d_model,
            self.nla.sidecar.extraction_layer,
        )
        await _emit(emit, {
            "type": "phase",
            "name": "av_loaded",
            "message": f"AV loaded in {time.time() - t0:.1f}s",
        })

    async def _unload_av(self, emit: StatusEmitter | None) -> None:
        await _emit(emit, {
            "type": "phase",
            "name": "unloading_av",
            "message": "Unloading AV to free memory...",
        })
        t0 = time.time()
        self.nla = None
        gc.collect()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            logger.exception("torch.mps.empty_cache() failed")
        logger.info("unloaded AV in %.1fs", time.time() - t0)


# ── Module helpers ──────────────────────────────────────────────────

def _resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def _expected_d_model() -> int:
    """Best-guess d_model from settings — cheap sanity check for the
    refusal_directions sidecar before M is loaded. Gemma-3-12B-IT is
    3840; we default to that since it's our deployed M."""
    return 3840


async def _emit(emit: StatusEmitter | None, payload: dict[str, Any]) -> None:
    if emit is None:
        return
    try:
        result = emit(payload)
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        logger.exception("status emit failed for %s", payload.get("name"))


# Imported by api/app.py; type for the SAE subdir path
_ = json  # keep json import live for future sidecar manipulation
