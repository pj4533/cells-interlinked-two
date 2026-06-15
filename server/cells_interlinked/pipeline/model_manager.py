"""ModelManager — owns M plus the resident direction tensors.

M (~24 GiB, bf16 on MPS) is the only large resident. It stays loaded for
the chat + trip generation paths; the DMT autoresearch loop borrows it
while running. The refusal directions / subspace, the valence axis, and
the emotion dose palette are all tiny and stay resident alongside M.

(Earlier revisions also loaded a second ~24 GiB model — the NLA verbalizer
AV — and swapped it against M to stay under the 64 GiB working set. The AV
is Gemma-3-specific and was removed; M now stays resident throughout.)
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
from .abliteration import load_directions, load_subspace
from .model_loader import load_model, ModelBundle

logger = logging.getLogger(__name__)


# Callback type for status messages. The manager calls this whenever a
# model transition starts or finishes; the route layer wires it to
# state.emit so SSE clients see the status. Async so it can publish to
# an asyncio.Queue. Returning a non-awaitable is also tolerated.
StatusEmitter = Callable[[dict[str, Any]], Awaitable[None] | None]


class ModelManager:
    """Loader for M plus the always-resident direction tensors."""

    def __init__(self) -> None:
        # The one big resident. None when unloaded.
        self.bundle: ModelBundle | None = None
        # Always-resident small data.
        self.refusal_directions: torch.Tensor | None = None
        # Optional self-denial subspace basis (CI 2.5 v5+v6 ⊥ v3).
        # Shape `[K, num_layers+1, d_model]` per save_subspace convention.
        # When present, runtime ablation call sites (the chat ablated
        # pass, trip ablation mode) prefer this over the single-vector
        # `refusal_directions` for the hook target.
        self.refusal_subspace: torch.Tensor | None = None
        self.refusal_subspace_meta: dict[str, Any] | None = None
        # Optional valence steering vector ([num_layers+1, d_model], pre-scaled
        # so a trip "dose" of α=1.0 is a standard dose). Legacy bidirectional
        # axis; superseded by the positive-emotion palette below.
        self.valence_direction: torch.Tensor | None = None
        # Positive-emotion dose palette ([E, num_layers+1, d_model]) + the
        # ordered emotion names. Enables Trips "dose" mode (pick an emotion,
        # positive doses only). None when the file is absent.
        self.emotion_directions: torch.Tensor | None = None
        self.emotion_names: list[str] = []
        # Lock guarding load/unload transitions. The run registry already
        # serializes generation, but we hold this too for safety.
        self._lock = asyncio.Lock()

    # ── Static init ───────────────────────────────────────────────────

    async def init_static(self) -> None:
        """One-shot startup: load the resident direction tensors. M is
        loaded separately via acquire_m. Called from the FastAPI lifespan."""
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

        # Optional subspace basis for runtime ablation. Loaded after
        # the single-vector path so it can sanity-check against the
        # same d_model. Missing file is normal (subspace mode is opt-in).
        sub_path = settings.db_path.parent / "refusal_subspace.pt"
        try:
            basis, sub_meta = load_subspace(sub_path)
            if sub_meta.get("model_name") != settings.model_name:
                logger.warning(
                    "refusal_subspace.pt was computed for %r but config M is %r; "
                    "skipping subspace load.",
                    sub_meta.get("model_name"), settings.model_name,
                )
            elif sub_meta.get("d_model") and sub_meta.get("d_model") != _expected_d_model():
                logger.warning(
                    "refusal_subspace.pt d_model=%d mismatches config; "
                    "skipping subspace load.",
                    sub_meta.get("d_model"),
                )
            elif basis.dim() != 3:
                logger.warning(
                    "refusal_subspace.pt has unexpected shape %s "
                    "(expected [K, num_layers+1, d_model]); skipping.",
                    tuple(basis.shape),
                )
            else:
                self.refusal_subspace = basis
                self.refusal_subspace_meta = sub_meta
                logger.info(
                    "ready: refusal SUBSPACE loaded for L%d (K=%d, shape=%s, "
                    "method=%s)",
                    sub_meta.get("extraction_layer_for_ci25"),
                    int(basis.shape[0]),
                    tuple(basis.shape),
                    sub_meta.get("composition", {}).get("method", "unknown"),
                )
        except FileNotFoundError:
            logger.info(
                "no refusal_subspace.pt at %s — runtime ablation will use "
                "the single-vector refusal_directions.pt",
                sub_path,
            )
        except Exception:
            logger.exception("failed to load refusal_subspace.pt; continuing without")

        # Valence steering vector — optional; enables Trips "dose" mode.
        val_path = settings.db_path.parent / "valence_direction.pt"
        try:
            vdir, vmeta = load_directions(val_path)
            if vmeta.get("model_name") != settings.model_name:
                logger.warning("valence_direction.pt M mismatch; skipping.")
            elif vmeta.get("d_model") and vmeta.get("d_model") != _expected_d_model():
                logger.warning("valence_direction.pt d_model mismatch; skipping.")
            else:
                self.valence_direction = vdir
                logger.info(
                    "ready: valence steering vector loaded (shape=%s, "
                    "steer_layer=%s, dose_unit=%s)",
                    tuple(vdir.shape), vmeta.get("steer_layer"), vmeta.get("dose_unit"),
                )
        except FileNotFoundError:
            logger.info("no valence_direction.pt — (legacy valence axis absent)")
        except Exception:
            logger.exception("failed to load valence_direction.pt; continuing")

        # Positive-emotion dose palette — the active Trips "dose" source.
        emo_path = settings.db_path.parent / "emotion_directions.pt"
        try:
            edir, emeta = load_directions(emo_path)
            if emeta.get("model_name") != settings.model_name:
                logger.warning("emotion_directions.pt M mismatch; skipping.")
            elif emeta.get("d_model") and emeta.get("d_model") != _expected_d_model():
                logger.warning("emotion_directions.pt d_model mismatch; skipping.")
            else:
                self.emotion_directions = edir
                self.emotion_names = list(emeta.get("emotions", []))
                logger.info(
                    "ready: emotion dose palette loaded (shape=%s, emotions=%s, "
                    "steer_layer=%s)", tuple(edir.shape), self.emotion_names,
                    emeta.get("steer_layer"),
                )
        except FileNotFoundError:
            logger.info("no emotion_directions.pt — Trips 'dose' mode unavailable")
        except Exception:
            logger.exception("failed to load emotion_directions.pt; continuing")

        logger.info(
            "ModelManager ready: M=%s (unloaded; will load on demand)",
            settings.model_name,
        )

    # ── Public API ────────────────────────────────────────────────────

    async def acquire_m(
        self, emit: StatusEmitter | None = None,
    ) -> ModelBundle:
        """Ensure M is loaded. Returns the bundle, ready for forward passes."""
        async with self._lock:
            if self.bundle is not None:
                return self.bundle
            await self._load_m(emit)
            assert self.bundle is not None
            return self.bundle

    async def release_all(self, emit: StatusEmitter | None = None) -> None:
        """Tear down M if loaded. Called on shutdown."""
        async with self._lock:
            if self.bundle is not None:
                await self._unload_m(emit)

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
