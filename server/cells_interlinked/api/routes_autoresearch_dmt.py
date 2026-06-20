"""DMT autoresearch control + live state — the sole autoresearch loop.

  POST /autoresearch-dmt/start   {budget?: int}  → start the DMT-phenomenology hunt
  POST /autoresearch-dmt/stop                     → halt after the current candidate
  GET  /autoresearch-dmt/state                    → full live state (the viewer polls)
  POST /autoresearch-dmt/export  {top_n?: int}    → promote top scorers into the
                                                    `dmt` dose group (chat/trips)

Owns M while running (enforced in AutoresearchBase.start); chat/trip 503 meanwhile.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


class ExportRequest(BaseModel):
    top_n: int = Field(default=8, ge=1, le=64)


class DmtStartRequest(BaseModel):
    # None = run until stopped. Otherwise stop after this many candidates.
    budget: int | None = Field(default=None, ge=1, le=100000)
    # One-shot leader burst: first N candidates explode the top-cluster
    # neighborhood (mutate/refine of the leaders + fresh injects) before normal
    # generation resumes. 0 = no burst.
    burst: int = Field(default=0, ge=0, le=1000)


def _controller(request: Request):
    ctrl = getattr(request.app.state, "dmt_autoresearch", None)
    if ctrl is None:
        raise HTTPException(status_code=503, detail="DMT autoresearch controller not initialized")
    return ctrl


@router.post("/autoresearch-dmt/start")
async def dmt_start(req: DmtStartRequest, request: Request) -> dict:
    return await _controller(request).start(budget=req.budget, burst=req.burst)


@router.post("/autoresearch-dmt/stop")
async def dmt_stop(request: Request) -> dict:
    return await _controller(request).stop()


@router.get("/autoresearch-dmt/state")
async def dmt_state(request: Request) -> dict:
    return _controller(request).state()


@router.get("/autoresearch-dmt/cells/{vid}")
async def dmt_cells(vid: str, request: Request) -> dict:
    """Per-α / per-dose detail for one committed atlas entry (lazy-loaded on
    row expand — kept out of the polled /state to keep it light)."""
    detail = _controller(request).entry_cells(vid)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"no atlas entry {vid!r}")
    return detail


@router.get("/autoresearch-dmt/placebo")
async def dmt_placebo(request: Request) -> dict:
    """The un-steered baseline: what the dose prompt produces with NO steering,
    with full per-sample features/evidence/text (same drill-down as the atlas).
    Every direction's placebo-subtracted score is measured against this."""
    detail = _controller(request).placebo_detail()
    if detail is None:
        raise HTTPException(status_code=404, detail="no placebo baseline computed yet")
    return detail


@router.post("/autoresearch-dmt/export")
async def dmt_export(req: ExportRequest, request: Request) -> dict:
    """Promote the top-N DMT directions into the dose palette (chat/trips)."""
    return _controller(request).export_to_palette(top_n=req.top_n)
