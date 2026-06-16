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
from ..pipeline.autoresearch_base import any_autoresearch_active
from ..pipeline.generation_loop import ProbeConfig, ProbeResult, run_probe
from ..pipeline.thinking import ThinkingSplitter, split_thinking
from ..pipeline.trajectory import (
    assemble_geometry,
    build_series,
    compute_raw_basis,
)
from .runs import RunState

logger = logging.getLogger(__name__)
router = APIRouter()


class TripRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    # "ablate" = remove the refusal direction (the original trip). "steer" =
    # ADD a positive-emotion "dose" — the dose mode.
    mode: str = "ablate"
    # Which positive emotion to dose with (steer mode). Must be one of the
    # loaded palette; falls back to the first if unknown.
    dose_emotion: str | None = None
    # Discrete strengths to generate at (each is a real generation: text +
    # trajectory). Raw (α=0) always runs. Doses are positive only.
    alphas: list[float] | None = None


# Default ablation α levels (1.0 = full refusal removal; >1.0 over-projects).
DEFAULT_TRIP_ALPHAS = [0.5, 1.0]
# Default steering doses — positive only, gentle→strong (the gradual ramp keeps
# them coherent; the strongest probes the coherence cliff).
DEFAULT_STEER_ALPHAS = [0.5, 1.0, 1.5]
# Layer the valence dose is injected at — earlier than the L32 readout layer,
# because the steering probes found early injection propagates to the output
# far better (L20: 60% on-target coherent; L40: 0%).
STEER_LAYER = 20


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
    if any_autoresearch_active(app):
        raise HTTPException(status_code=503, detail="autoresearch is running — trips are locked")
    bundle = getattr(app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="M not loaded")

    run_id = uuid.uuid4().hex[:12]
    seed = req.seed if req.seed is not None else _seed_from_run_id(run_id)
    mode = "steer" if str(req.mode).lower() == "steer" else "ablate"
    # Sanitize + de-dupe the α list. Drop 0 (the raw run). Both modes are now
    # positive-only: ablate clamps to (0, 5], steer (the dose) to (0, 3].
    if mode == "steer":
        raw_alphas = req.alphas if req.alphas is not None else DEFAULT_STEER_ALPHAS
        alphas = sorted({
            round(max(0.0, min(3.0, float(a))), 2) for a in raw_alphas if float(a) > 0
        })[:6]
    else:
        raw_alphas = req.alphas if req.alphas is not None else DEFAULT_TRIP_ALPHAS
        alphas = sorted({
            round(max(0.0, min(5.0, float(a))), 2) for a in raw_alphas if float(a) > 0
        })[:6]
    dose_emotion = (req.dose_emotion or "").strip().lower() or None

    cfg = ProbeConfig(
        temperature=req.temperature if req.temperature is not None else settings.temperature,
        top_p=req.top_p if req.top_p is not None else settings.top_p,
        seed=seed,
    )
    # No system prompt: the Trip View measures the residual trajectory in
    # response to the PROBE. The default "answer directly / keep brief"
    # instruction is folded in front of the user turn on Gemma, which makes
    # the model emit acknowledgment boilerplate ("Okay, I understand…") and
    # truncates introspection — contaminating the very tokens we plot.
    rendered = bundle.render_prompt(req.prompt, system_prompt=None, enable_thinking=True)

    state = RunState(run_id=run_id, prompt_text=req.prompt)
    app.state.registry.add(state)
    state.task = asyncio.create_task(
        _execute_trip(app, state, cfg, rendered, alphas, mode, dose_emotion)
    )
    return TripResponse(run_id=run_id)


@router.get("/dose_emotions")
async def dose_emotions(request: Request) -> dict:
    """The dose palette for steer mode, so the setup screen can render a picker.
    `emotions` = all selectable names; `uncharted` = the subset that are NOT
    emotions but non-human-readable off-manifold directions (so the UI can group
    + caveat them honestly). Empty `emotions` ⇒ dose mode unavailable."""
    # feat-* are internal DMT-autoresearch seeds (diff-of-means feature directions),
    # not user-facing doses — keep them out of the picker entirely.
    names = [n for n in (getattr(request.app.state, "emotion_names", []) or [])
             if not n.startswith("feat-")]
    uncharted: list[str] = []
    research: list[str] = []
    research_meta: dict = {}
    dmt: list[str] = []
    dmt_meta: dict = {}
    try:
        sc = json.loads((settings.db_path.parent / "emotion_directions.pt.json").read_text())
        uncharted = [n for n in sc.get("uncharted", []) if n in names]
        # Off-manifold autoresearch exports (provenance: atlas_id, off_ortho,
        # alpha_star, parents, generator) so the picker can show lineage.
        research = [n for n in sc.get("research", []) if n in names]
        research_meta = {n: m for n, m in sc.get("research_meta", {}).items() if n in names}
        # DMT autoresearch exports (provenance: atlas_id, score, best_alpha,
        # matched_features, parents, generator).
        dmt = [n for n in sc.get("dmt", []) if n in names]
        dmt_meta = {n: m for n, m in sc.get("dmt_meta", {}).items() if n in names}
    except Exception:
        uncharted = []
        research = []
        research_meta = {}
        dmt = []
        dmt_meta = {}
    return {
        "available": bool(names), "emotions": list(names),
        "uncharted": uncharted, "research": research,
        "research_meta": research_meta,
        "dmt": dmt, "dmt_meta": dmt_meta,
    }


async def _execute_trip(
    app,
    state: RunState,
    cfg: ProbeConfig,
    rendered: str,
    alphas: list[float],
    mode: str = "ablate",
    dose_emotion: str | None = None,
) -> None:
    import dataclasses as _dc
    from ..pipeline.abliteration import (
        install_runtime_ablation_hook,
        install_runtime_steering_hook,
        pick_ablation_target,
    )

    manager = app.state.manager
    registry = app.state.registry
    layer = bundle_layer = None  # set after acquire

    bundle = await manager.acquire_m(emit=state.emit)
    app.state.bundle = bundle
    app.state.nla = None
    layer = bundle.extraction_layer

    # Surface queue position like the probe route does.
    if registry.holder_run_id is not None or registry.waiters:
        await state.emit({
            "type": "queued",
            "holder_run_id": registry.holder_run_id,
            "position": len(registry.waiters) + 1,
        })

    async def run_one(alpha: float, install_fn) -> ProbeResult | None:
        """Run one generation (raw if install_fn is None, else with the
        intervention hook at `alpha` — ablation at L32 or steering at the
        steer layer). Streams tokens as trip_token events tagged with α. The
        capture hook (L32) fires after the intervention hook, so captured
        residuals reflect the intervention — the real path."""
        q: asyncio.Queue = asyncio.Queue(maxsize=10000)

        async def fwd() -> None:
            splitter = ThinkingSplitter(bundle.thought_open_id, bundle.thought_close_id)
            while True:
                evt = await q.get()
                et = evt.get("type")
                if et == "token":
                    channel, text = splitter.feed(evt["token_id"], evt["decoded"])
                    if channel is None:
                        continue  # delimiter / channel-name token — suppress
                    await state.emit({
                        "type": "trip_token",
                        "alpha": alpha,
                        "position": evt["position"],
                        "decoded": text,
                        "channel": channel,
                    })
                elif et == "stopped":
                    break

        task = asyncio.create_task(fwd())
        # Intervened runs share the same runaway guard as the raw run. The
        # cap only exists to bound off-manifold no-EOS loops (which would grow
        # the KV cache to OOM); 4096 gives thinking-mode reasoning room to
        # finish before EOS instead of truncating legitimate runs.
        cfg_run = cfg if install_fn is None else _dc.replace(cfg, safety_cap=4096)
        hook = None
        if install_fn is not None:
            hook = install_fn(alpha)
        try:
            return await run_probe(
                bundle=bundle,
                rendered_prompt=rendered,
                cfg=cfg_run,
                cancel_event=state.cancel_event,
                queue=q,
            )
        except Exception:
            logger.exception("trip generation failed (α=%.2f)", alpha)
            await q.put({"type": "stopped"})
            return None
        finally:
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()
            if hook is not None:
                try:
                    hook.remove()
                except Exception:
                    logger.exception("failed to remove trip ablation hook")

    async with registry.acquire(state.run_id):
        await state.emit({"type": "running"})

        # ── Raw generation (α=0) ─────────────────────────────────────
        # As each generation finishes we project it into the shared raw-PCA
        # basis and emit it as a `trip_series` event, so the scene builds up
        # live (raw appears first, then each α) instead of all-at-once.
        await state.emit({"type": "phase", "name": "generating", "alpha": 0.0})
        raw = await run_one(0.0, None)
        if raw is None or not raw.captured:
            await state.emit({"type": "error", "message": "no tokens generated"})
            await state.emit({"type": "done"})
            await state.event_log.close()
            state.completed = True
            return

        try:
            basis = await asyncio.to_thread(
                compute_raw_basis,
                [c.activations[layer] for c in raw.captured],
            )
            raw_series = await asyncio.to_thread(
                build_series,
                [c.activations[layer] for c in raw.captured],
                [c.decoded for c in raw.captured],
                raw.output_text, 0.0, raw.stopped_reason, basis,
            )
            raw_series.thinking, raw_series.answer = split_thinking(
                [c.token_id for c in raw.captured], [c.decoded for c in raw.captured],
                bundle.thought_open_id, bundle.thought_close_id,
            )
        except Exception as exc:
            logger.exception("trip raw geometry failed")
            await state.emit({"type": "error", "message": f"geometry: {exc}"})
            await state.emit({"type": "done"})
            await state.event_log.close()
            state.completed = True
            return
        series = [raw_series]
        await state.emit({"type": "trip_series", "layer": layer, "series": raw_series.to_dict()})

        # ── Real intervened generations, one per α ───────────────────
        # Build the per-α hook installer for this mode (None ⇒ skip).
        install_fn = None
        chosen_emotion = None
        if mode == "steer":
            emo = getattr(app.state, "emotion_directions", None)
            names = getattr(app.state, "emotion_names", []) or []
            if emo is not None and names:
                idx = names.index(dose_emotion) if dose_emotion in names else 0
                chosen_emotion = names[idx]
                v_layer = emo[idx][STEER_LAYER]
                install_fn = (lambda a, _v=v_layer: install_runtime_steering_hook(
                    bundle.model, STEER_LAYER, _v, a))
        else:
            rdirs = getattr(app.state, "refusal_directions", None)
            rsub = getattr(app.state, "refusal_subspace", None)
            r_target = pick_ablation_target(rsub, rdirs, layer)
            if r_target is not None:
                install_fn = (lambda a, _t=r_target: install_runtime_ablation_hook(
                    bundle.model, layer, _t, a))
        if install_fn is not None:
            for a in alphas:
                if state.cancel_event.is_set():
                    break
                await state.emit({
                    "type": "phase", "name": "ablated_generation", "alpha": a,
                })
                res = await run_one(a, install_fn)
                if res is None or not res.captured:
                    continue
                try:
                    ser = await asyncio.to_thread(
                        build_series,
                        [c.activations[layer] for c in res.captured],
                        [c.decoded for c in res.captured],
                        res.output_text, a, res.stopped_reason, basis,
                    )
                except Exception:
                    logger.exception("trip ablated series build failed (α=%.2f)", a)
                    continue
                ser.thinking, ser.answer = split_thinking(
                    [c.token_id for c in res.captured], [c.decoded for c in res.captured],
                    bundle.thought_open_id, bundle.thought_close_id,
                )
                series.append(ser)
                await state.emit({"type": "trip_series", "layer": layer, "series": ser.to_dict()})

        # ── Final assembled geometry (persist + canonical) ───────────
        await state.emit({"type": "phase", "name": "computing_geometry"})
        geometry = assemble_geometry(basis.d_model, layer, series)
        variant = (f"dose · {chosen_emotion} @L{STEER_LAYER}"
                   if mode == "steer" and chosen_emotion
                   else _active_variant_name(mode))
        payload = {
            "run_id": state.run_id,
            "prompt": state.prompt_text,
            "seed": cfg.seed,
            "mode": mode,
            "dose_emotion": chosen_emotion,
            "direction_variant": variant,
            "alphas": alphas,
            "geometry": geometry.to_dict(),
            "created_at": time.time(),
        }
        await state.emit({"type": "trip_geometry", **payload})

        try:
            (_trips_dir() / f"{state.run_id}.json").write_text(json.dumps(payload))
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


def _active_variant_name(mode: str = "ablate") -> str:
    # Steer mode: report the valence dose vector.
    if mode == "steer":
        try:
            m = json.loads((settings.db_path.parent / "valence_direction.pt.json").read_text())
            return f"steer · {m.get('variant_name', 'valence')} @L{m.get('steer_layer', STEER_LAYER)}"
        except Exception:
            return "steer · valence"
    # Mirror routes_chat: the runtime hook resolves its target via
    # pick_ablation_target(subspace, directions, ...), which PREFERS the
    # subspace. So when refusal_subspace.pt is loaded, that sidecar's
    # variant_name is what was actually used to ablate — not the
    # single-vector refusal_directions.pt. Read the subspace sidecar first
    # and fall back to the single-vector one only if no subspace exists.
    variant_name = "unknown"
    try:
        sub_meta_path = settings.db_path.parent / "refusal_subspace.pt.json"
        if sub_meta_path.exists():
            variant_name = json.loads(sub_meta_path.read_text()).get(
                "variant_name", "self_denial_subspace",
            )
        else:
            meta_path = settings.db_path.parent / "refusal_directions.pt.json"
            if meta_path.exists():
                variant_name = json.loads(meta_path.read_text()).get(
                    "variant_name", variant_name,
                )
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
        # Skip pre-multi-α sidecars (no `series`) — incompatible with the
        # current viewer. They're left on disk, just not listed.
        if "series" not in g:
            continue
        ser = g.get("series", []) or []
        raw_s = ser[0] if ser else {}
        abl = ser[1:] if len(ser) > 1 else []
        # Headline the PEAK opening (ablated series with the highest eff-dim),
        # not the strongest α — over-projection often loops and collapses dim,
        # which would misleadingly read as "ablation reduced dimensionality".
        top = max(abl, key=lambda s: s.get("eff_dim", 0)) if abl else None
        rows.append({
            "run_id": data.get("run_id", p.stem),
            "prompt": data.get("prompt", ""),
            "created_at": data.get("created_at", p.stat().st_mtime),
            "n_tokens": raw_s.get("n_tokens", 0),
            "layer": g.get("layer"),
            "direction_variant": data.get("direction_variant", ""),
            "alphas": data.get("alphas", []),
            "eff_dim_raw": raw_s.get("eff_dim"),
            "eff_dim_ablated": top.get("eff_dim") if top else None,
            "top_alpha": top.get("alpha") if top else None,
            "ablation_available": g.get("ablation_available", len(abl) > 0),
            "stopped_reason": raw_s.get("stopped_reason"),
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
