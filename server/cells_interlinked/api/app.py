"""FastAPI factory. Loads M + AV once on startup; tears down cleanly on shutdown.

v2 swap: where v1 loaded SAEs alongside M, v2 loads the kitft NLA verbalizer
(AV) at app.state.nla. Both M and AV stay resident across the autorun batch
so phase-1 generation and phase-2 NLA decoding alternate without paying the
~3-min model load each time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import settings
from ..pipeline.autorun import AutorunController
from ..pipeline.labels import init_labels_table
from ..pipeline.model_loader import load_model
from ..pipeline.nla_client import NLAClient
from ..pipeline.sae_runner import (
    DEFAULT_SAE_REPO,
    DEFAULT_SAE_SUBDIR,
    JumpReLUSAE,
)
from ..storage import db
from .routes_autorun import router as autorun_router
from .routes_journal import router as journal_router
from .routes_probe import router as probe_router
from .routes_stream import router as stream_router
from .runs import RunRegistry

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db(settings.db_path)
    await init_labels_table(settings.db_path)

    # Any probes left in-flight from a previous backend process are
    # orphaned: their asyncio task died with the old process, their
    # RunRegistry entry is gone. Mark them errored so the archive
    # doesn't show ghosts that 404 on reconnect.
    n_orphans = await db.cleanup_orphans(settings.db_path)
    if n_orphans:
        logger.info("cleaned up %d orphaned in-flight run(s)", n_orphans)

    dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[settings.dtype]

    logger.info("loading M=%s ...", settings.model_name)
    bundle = await asyncio.to_thread(
        load_model,
        settings.model_name,
        device_str=settings.device,
        dtype=dtype,
        extraction_layer=settings.extraction_layer,
    )

    logger.info("loading AV=%s ...", settings.av_repo)
    nla = await asyncio.to_thread(
        NLAClient,
        settings.av_repo,
        device_str=settings.device,
        dtype=dtype,
    )
    assert nla.sidecar.extraction_layer == bundle.extraction_layer, (
        f"AV sidecar extraction_layer={nla.sidecar.extraction_layer} != "
        f"M's configured extraction_layer={bundle.extraction_layer}. "
        f"Check that the AV repo and EXTRACTION_LAYER env match."
    )
    assert nla.sidecar.d_model == bundle.hidden_dim, (
        f"AV d_model={nla.sidecar.d_model} != M hidden_dim={bundle.hidden_dim}. "
        f"M and AV must share architecture."
    )

    app.state.bundle = bundle
    app.state.nla = nla

    # Optional secondary instrument: Gemma Scope 2 SAE at the same
    # extraction layer the AV reads. Skip silently if the architecture
    # isn't Gemma-flavoured (Gemma Scope 2 only covers Gemma-3) — Qwen
    # and other M models still have NLA as their primary readout.
    sae: JumpReLUSAE | None = None
    if settings.sae_enabled and "gemma" in bundle.model_name.lower():
        try:
            sae = await asyncio.to_thread(
                JumpReLUSAE,
                settings.sae_repo,
                settings.sae_subdir,
                settings.sae_device,
            )
            assert sae.cfg.d_model == bundle.hidden_dim, (
                f"SAE d_model={sae.cfg.d_model} != M hidden_dim={bundle.hidden_dim}"
            )
            # SAE layer may differ from AV layer: Neuronpedia only hosts
            # auto-interp labels for the four canonical Gemma Scope 2
            # layers (12/24/31/41), so we read at L31 even though the AV
            # reads at L32. The phase-1 hook captures both layers; phase-2
            # routes appropriately. We just sanity-check the SAE layer
            # is plausible for the loaded M.
            if sae.cfg.layer != bundle.extraction_layer:
                logger.info(
                    "SAE layer L%d differs from AV layer L%d — phase 1 will "
                    "capture both; cross-reference is adjacent-layer.",
                    sae.cfg.layer, bundle.extraction_layer,
                )
            logger.info(
                "SAE loaded as secondary instrument: %s/%s (d_sae=%d, L%d)",
                settings.sae_repo, settings.sae_subdir, sae.cfg.d_sae,
                sae.cfg.layer,
            )
        except Exception as exc:
            logger.warning("SAE load failed; continuing NLA-only: %s", exc)
            sae = None
    elif settings.sae_enabled:
        logger.info(
            "SAE skipped: only loads automatically for Gemma-flavoured M "
            "(current: %s)",
            bundle.model_name,
        )
    app.state.sae = sae

    app.state.registry = RunRegistry()
    app.state.refusal_directions = None  # not supported in v2

    autorun = AutorunController(db_path=settings.db_path)
    autorun.app = app
    app.state.autorun = autorun
    await db.set_autorun_running(
        settings.db_path, running=False, event="server-restart", ts=time.time()
    )

    logger.info(
        "ready: M=%s (L=%d, hidden=%d, layer=L%d) | AV=%s",
        bundle.model_name, bundle.num_layers, bundle.hidden_dim,
        bundle.extraction_layer, settings.av_repo,
    )

    try:
        yield
    finally:
        if autorun.running:
            await autorun.stop()
        logger.info("shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Cells Interlinked v2",
        description="V-K probes via NLA-decoded activations vs. output tokens",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|[a-z0-9-]+\.local)(:\d+)?$",
        # DELETE is used by /journal/{id} to discard a draft. Without it,
        # browser CORS preflight (OPTIONS) returns 400 and the UI fails
        # silently. PATCH/PUT included for completeness — cheap, no harm.
        allow_methods=["GET", "POST", "DELETE", "PATCH", "PUT", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(probe_router)
    app.include_router(stream_router)
    app.include_router(autorun_router)
    app.include_router(journal_router)

    @app.get("/health")
    def health() -> dict:
        bundle = getattr(app.state, "bundle", None)
        nla = getattr(app.state, "nla", None)
        sae = getattr(app.state, "sae", None)
        return {
            "status": "ok",
            "model_loaded": bundle is not None,
            "av_loaded": nla is not None,
            "sae_loaded": sae is not None,
            "model_name": bundle.model_name if bundle else None,
            "av_repo": settings.av_repo,
            "sae_repo": (sae.cfg.repo + "/" + sae.cfg.subdir) if sae else None,
            "extraction_layer": bundle.extraction_layer if bundle else None,
            "device": str(bundle.device) if bundle else None,
        }

    return app
