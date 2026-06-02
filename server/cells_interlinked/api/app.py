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
from fastapi.staticfiles import StaticFiles

from ..config import settings
from ..pipeline.autorun import AutorunController
from ..pipeline.autoresearch import AutoresearchController
from ..storage import db
from .routes_autorun import router as autorun_router
from .routes_autoresearch import router as autoresearch_router
from .routes_chat import router as chat_router
from .routes_journal import router as journal_router
from .routes_probe import router as probe_router
from .routes_stream import router as stream_router
from .routes_trip import router as trip_router
from .routes_tts import router as tts_router
from .runs import RunRegistry

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db(settings.db_path)

    # Any probes left in-flight from a previous backend process are
    # orphaned: their asyncio task died with the old process, their
    # RunRegistry entry is gone. Mark them errored so the archive
    # doesn't show ghosts that 404 on reconnect.
    n_orphans = await db.cleanup_orphans(settings.db_path)
    if n_orphans:
        logger.info("cleaned up %d orphaned in-flight run(s)", n_orphans)

    # CI 2.5 serial-model architecture: M (~24 GiB) and AV (~24 GiB)
    # can't both be resident on the 64 GiB box without thrashing swap.
    # ModelManager owns them and ensures only one is loaded at a time.
    # Lifespan pre-loads M (so the first probe doesn't pay a model-
    # load cost on the first phase). AV gets loaded lazily during the
    # first probe's phase-2 NLA decode (with status events visible to
    # the user). The probe-execution code in routes_probe drives the
    # M↔AV swaps as the phases progress.
    from ..pipeline.model_manager import ModelManager
    manager = ModelManager()
    await manager.init_static()
    app.state.manager = manager

    # Pre-load M. Costs ~15s warm / ~70s cold but happens once at
    # startup; subsequent probes see "M loaded" immediately.
    await manager.acquire_m(emit=None)

    # app.state shims for existing code paths. These references are
    # mutated by the manager when it swaps models — the probe route
    # updates them after each acquire/release.
    app.state.bundle = manager.bundle
    app.state.nla = manager.nla  # None at startup; populated on first phase 2
    app.state.refusal_directions = manager.refusal_directions
    # Optional self-denial subspace basis (CI 2.5 v5+v6 ⊥ v3). When
    # present, runtime ablation call sites (chat ablated pass, phase 1b,
    # ablated synthesizer) prefer this over the single direction.
    app.state.refusal_subspace = manager.refusal_subspace
    app.state.refusal_subspace_meta = manager.refusal_subspace_meta
    # Optional valence steering vector — legacy bidirectional axis.
    app.state.valence_direction = manager.valence_direction
    # Positive-emotion dose palette — the active Trips "dose" mode source.
    app.state.emotion_directions = manager.emotion_directions
    app.state.emotion_names = manager.emotion_names

    app.state.registry = RunRegistry()

    autorun = AutorunController(db_path=settings.db_path)
    autorun.app = app
    app.state.autorun = autorun
    await db.set_autorun_running(
        settings.db_path, running=False, event="server-restart", ts=time.time()
    )

    # Autoresearch (steering-direction hunt). Owns M while running, which locks
    # out probe/chat/trip via app.state.autoresearch_active.
    autoresearch = AutoresearchController()
    autoresearch.app = app
    try:
        autoresearch._load()  # surface any persisted atlas to /state + export before a run
    except Exception:
        logger.exception("failed to preload autoresearch atlas")
    app.state.autoresearch = autoresearch
    app.state.autoresearch_active = False

    logger.info(
        "ready: M=%s | AV=%s | both unloaded (serial load on demand)",
        settings.model_name, settings.av_repo,
    )

    try:
        yield
    finally:
        if autorun.running:
            await autorun.stop()
        if autoresearch.running:
            await autoresearch.stop()
        await manager.release_all()
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
    app.include_router(trip_router)
    app.include_router(autorun_router)
    app.include_router(autoresearch_router)
    app.include_router(journal_router)
    app.include_router(chat_router)
    app.include_router(tts_router)

    # /chat-images serves the PNGs generated by Nano Banana when chat
    # imagery is enabled. We ensure the directory exists at startup
    # so StaticFiles doesn't refuse to mount on a fresh checkout.
    settings.image_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/chat-images",
        StaticFiles(directory=str(settings.image_dir)),
        name="chat-images",
    )

    @app.get("/health")
    def health() -> dict:
        bundle = getattr(app.state, "bundle", None)
        nla = getattr(app.state, "nla", None)
        return {
            "status": "ok",
            "model_loaded": bundle is not None,
            "av_loaded": nla is not None,
            "model_name": settings.model_name,
            "av_repo": settings.av_repo,
            "extraction_layer": settings.extraction_layer,
            "device": settings.device,
        }

    return app
