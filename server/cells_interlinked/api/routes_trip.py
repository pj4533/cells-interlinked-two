"""POST /trip, GET /trip/{run_id} — the "Trip View" surface (CI 2.5).

A trip run is a stripped-down probe: M generates once, we capture the L32
residual per token (exactly as a normal probe does), and then — instead of
swapping to the AV for per-token NLA — we compute the *trajectory-level*
geometry (effective dimensionality + spectral entropy, raw vs
refusal-ablated; see `pipeline/trajectory.py`) and stream it as one
`trip_geometry` SSE event.

No AV swap, no judge, no synthesis: M stays resident the whole time, so a
trip run is cheap and the manager can never get wedged in the AV-loaded
state. Generation tokens stream live (the subject "speaks"); the geometry
crystallizes the moment generation ends and the client animates the 3D
trajectory + drives the realtime α-morph entirely browser-side.

Geometry is persisted to a JSON sidecar under `data/trips/{run_id}.json`
(NOT the probes DB — trip runs have no verdict rows and we don't want them
polluting the archive's verdict schema). `GET /trip/{run_id}` reads it back
so a future review page can rehydrate a finished trip.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

import torch
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..config import settings
from ..pipeline.generation_loop import ProbeConfig, ProbeResult, run_probe
from ..pipeline.trajectory import compute_trip_geometry
from .runs import RunState

logger = logging.getLogger(__name__)
router = APIRouter()


class TripRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    # The α the *_ablated scalars + spectrum bars report at (the slider can
    # explore the whole grid client-side; this is just the comparison point).
    alpha_ref: float = 1.0


class TripResponse(BaseModel):
    run_id: str


def _trips_dir():
    d = settings.db_path.parent / "trips"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _seed_from_run_id(run_id: str) -> int:
    import hashlib
    h = hashlib.sha256(run_id.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


@router.post("/trip", response_model=TripResponse)
async def start_trip(req: TripRequest, request: Request) -> TripResponse:
    app = request.app
    bundle = getattr(app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="M not loaded")

    run_id = uuid.uuid4().hex[:12]
    seed = req.seed if req.seed is not None else _seed_from_run_id(run_id)
    alpha_ref = max(0.0, min(2.0, float(req.alpha_ref)))

    cfg = ProbeConfig(
        temperature=req.temperature if req.temperature is not None else settings.temperature,
        top_p=req.top_p if req.top_p is not None else settings.top_p,
        seed=seed,
        include_nla=False,            # trip never swaps to the AV
    )
    rendered = bundle.render_prompt(req.prompt)

    state = RunState(run_id=run_id, prompt_text=req.prompt)
    app.state.registry.add(state)
    state.task = asyncio.create_task(
        _execute_trip(app, state, cfg, rendered, alpha_ref)
    )
    return TripResponse(run_id=run_id)


async def _execute_trip(
    app,
    state: RunState,
    cfg: ProbeConfig,
    rendered: str,
    alpha_ref: float,
) -> None:
    manager = app.state.manager
    registry = app.state.registry

    bundle = await manager.acquire_m(emit=state.emit)
    app.state.bundle = bundle
    app.state.nla = None

    inner_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)

    async def forwarder() -> None:
        while True:
            evt = await inner_queue.get()
            await state.emit(evt)
            if evt.get("type") == "stopped":
                break

    forwarder_task = asyncio.create_task(forwarder())

    result: ProbeResult | None = None
    error_msg: str | None = None

    # Surface queue position like the probe route does.
    if registry.holder_run_id is not None or registry.waiters:
        await state.emit({
            "type": "queued",
            "holder_run_id": registry.holder_run_id,
            "position": len(registry.waiters) + 1,
        })

    async with registry.acquire(state.run_id):
        await state.emit({"type": "running"})
        await state.emit({"type": "phase", "name": "generating", "total": 0})

        try:
            result = await run_probe(
                bundle=bundle,
                rendered_prompt=rendered,
                cfg=cfg,
                cancel_event=state.cancel_event,
                queue=inner_queue,
            )
        except Exception as exc:
            logger.exception("trip generation failed")
            error_msg = str(exc)
            await state.emit({"type": "error", "message": str(exc)})
            await inner_queue.put({"type": "stopped", "reason": "error", "total_tokens": 0})
        finally:
            await forwarder_task

        if result is None or not result.captured:
            await state.emit({
                "type": "error",
                "message": error_msg or "no tokens generated",
            })
            await state.emit({"type": "done"})
            await state.event_log.close()
            state.completed = True
            return

        # ── Geometry pass (cheap; M stays loaded) ────────────────────
        await state.emit({"type": "phase", "name": "computing_geometry", "total": 0})
        try:
            rdirs = getattr(app.state, "refusal_directions", None)
            r_layer = (
                rdirs[bundle.extraction_layer] if rdirs is not None else None
            )
            activations = [
                c.activations[bundle.extraction_layer] for c in result.captured
            ]
            tokens = [c.decoded for c in result.captured]
            geometry = await asyncio.to_thread(
                compute_trip_geometry,
                activations,
                tokens,
                layer=bundle.extraction_layer,
                refusal_direction=r_layer,
                alpha_ref=alpha_ref,
            )
        except Exception as exc:
            logger.exception("trip geometry computation failed")
            await state.emit({"type": "error", "message": f"geometry: {exc}"})
            await state.emit({"type": "done"})
            await state.event_log.close()
            state.completed = True
            return

        geo_dict = geometry.to_dict()
        variant_name = _active_variant_name()
        payload = {
            "run_id": state.run_id,
            "prompt": state.prompt_text,
            "output_text": result.output_text,
            "stopped_reason": result.stopped_reason,
            "total_tokens": result.total_tokens,
            "seed": cfg.seed,
            "direction_variant": variant_name,
            "geometry": geo_dict,
            "created_at": time.time(),
        }

        await state.emit({"type": "trip_geometry", **payload})

        try:
            (_trips_dir() / f"{state.run_id}.json").write_text(
                json.dumps(payload)
            )
        except Exception:
            logger.exception("trip sidecar write failed (non-fatal)")

        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    await state.emit({"type": "done"})
    await state.event_log.close()
    state.completed = True


def _active_variant_name() -> str:
    variant_name = "unknown"
    try:
        meta_path = settings.db_path.parent / "refusal_directions.pt.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            variant_name = meta.get("variant_name", variant_name)
    except Exception:
        pass
    return variant_name


@router.get("/trips")
async def list_trips(limit: int = 20, offset: int = 0) -> dict:
    """List archived trips (newest first) for the archive page. Reads each
    sidecar's summary fields; the heavy coord arrays are ignored here."""
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    rows: list[dict] = []
    for p in _trips_dir().glob("*.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        g = data.get("geometry", {}) or {}
        rows.append({
            "run_id": data.get("run_id", p.stem),
            "prompt": data.get("prompt", ""),
            "created_at": data.get("created_at", p.stat().st_mtime),
            "total_tokens": data.get("total_tokens", g.get("n_tokens", 0)),
            "n_tokens": g.get("n_tokens", 0),
            "layer": g.get("layer"),
            "direction_variant": data.get("direction_variant", ""),
            "eff_dim_raw": g.get("eff_dim_raw"),
            "eff_dim_ablated": g.get("eff_dim_ablated"),
            "alpha_ref": g.get("alpha_ref", 1.0),
            "ablation_available": g.get("ablation_available", False),
            "stopped_reason": data.get("stopped_reason"),
        })
    rows.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
    total = len(rows)
    return {
        "rows": rows[offset:offset + limit],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/trip/{run_id}")
async def get_trip(run_id: str) -> dict:
    path = _trips_dir() / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="trip not found")
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"trip read failed: {exc}")
