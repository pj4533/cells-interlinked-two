"""Autoresearch control + live state.

  POST /autoresearch/start   {budget?: int}  → start the steering-direction hunt
  POST /autoresearch/stop                     → halt after the current candidate
  GET  /autoresearch/state                    → full live state (atlas, reverts,
                                                current candidate, frontier) — the
                                                viewer page polls this.

While the loop is running it owns M, so the probe/chat/trip routes return 503
(see _require_not_autoresearching in those routers / app.state.autoresearch_active).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


class StartRequest(BaseModel):
    # None = run until stopped. Otherwise stop after this many candidates.
    budget: int | None = Field(default=None, ge=1, le=100000)


def _controller(request: Request):
    ctrl = getattr(request.app.state, "autoresearch", None)
    if ctrl is None:
        raise HTTPException(status_code=503, detail="autoresearch controller not initialized")
    return ctrl


@router.post("/autoresearch/start")
async def autoresearch_start(req: StartRequest, request: Request) -> dict:
    return await _controller(request).start(budget=req.budget)


@router.post("/autoresearch/stop")
async def autoresearch_stop(request: Request) -> dict:
    return await _controller(request).stop()


@router.get("/autoresearch/state")
async def autoresearch_state(request: Request) -> dict:
    return _controller(request).state()
