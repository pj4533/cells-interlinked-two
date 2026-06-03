"""DMT autoresearch control + live state. Mirrors routes_autoresearch.py but
targets the DMT controller (app.state.dmt_autoresearch).

  POST /autoresearch-dmt/start   {budget?: int}  → start the DMT-phenomenology hunt
  POST /autoresearch-dmt/stop                     → halt after the current candidate
  GET  /autoresearch-dmt/state                    → full live state (the viewer polls)
  POST /autoresearch-dmt/export  {top_n?: int}    → promote top scorers into the
                                                    `dmt` dose group (chat/trips)

Mutually exclusive with the off-manifold loop — only one autoresearch owns M at a
time (enforced in AutoresearchBase.start). While either runs, probe/chat/trip 503.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from .routes_autoresearch import ExportRequest, StartRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def _controller(request: Request):
    ctrl = getattr(request.app.state, "dmt_autoresearch", None)
    if ctrl is None:
        raise HTTPException(status_code=503, detail="DMT autoresearch controller not initialized")
    return ctrl


@router.post("/autoresearch-dmt/start")
async def dmt_start(req: StartRequest, request: Request) -> dict:
    return await _controller(request).start(budget=req.budget)


@router.post("/autoresearch-dmt/stop")
async def dmt_stop(request: Request) -> dict:
    return await _controller(request).stop()


@router.get("/autoresearch-dmt/state")
async def dmt_state(request: Request) -> dict:
    return _controller(request).state()


@router.post("/autoresearch-dmt/export")
async def dmt_export(req: ExportRequest, request: Request) -> dict:
    """Promote the top-N DMT directions into the dose palette (chat/trips)."""
    return _controller(request).export_to_palette(top_n=req.top_n)
