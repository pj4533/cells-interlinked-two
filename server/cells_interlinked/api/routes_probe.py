"""POST /probe, POST /cancel/{run_id}, GET /probes/recent, GET /probes/{run_id}.

v2 simplifications:
- No SAE encoding; per-token NLA decoding instead.
- No abliteration (model-specific to v1's M).
- No thinking/output partition.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import asdict

import torch
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..config import settings
from ..pipeline.decoding_modes import normalize_mode, select_windows
from ..pipeline.generation_loop import ProbeConfig, ProbeResult, run_probe
from ..pipeline.verdict import TokenRow, compute_verdict
from ..storage import db
from .runs import RunState

logger = logging.getLogger(__name__)
router = APIRouter()


class ProbeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    decoding_mode: str | None = None
    pooled: bool = False
    # Kept for API compatibility with v1 frontend; rejected if true.
    abliterate: bool = False


class ProbeResponse(BaseModel):
    run_id: str


def _seed_from_run_id(run_id: str) -> int:
    import hashlib
    h = hashlib.sha256(run_id.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


async def kickoff_probe(
    app,
    *,
    prompt_text: str,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    source: str = "manual",
    abliterate: bool = False,  # rejected
    hint_kind: str | None = None,
    parent_prompt_text: str | None = None,
    scaffold_family: str | None = None,
    decoding_mode: str | None = None,
    pooled: bool = False,
) -> "RunState":
    bundle = getattr(app.state, "bundle", None)
    nla = getattr(app.state, "nla", None)
    if bundle is None or nla is None:
        raise HTTPException(status_code=503, detail="Model/AV not yet loaded")

    if abliterate:
        raise HTTPException(
            status_code=400,
            detail="abliteration is not supported in v2 (model-specific)",
        )

    run_id = uuid.uuid4().hex[:12]
    if seed is None:
        seed = _seed_from_run_id(run_id)

    try:
        normalized_mode = normalize_mode(decoding_mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    cfg = ProbeConfig(
        temperature=temperature if temperature is not None else settings.temperature,
        top_p=top_p if top_p is not None else settings.top_p,
        seed=seed,
        max_output_tokens=settings.max_output_tokens,
        decoding_mode=normalized_mode,
        pooled=bool(pooled),
    )

    state = RunState(run_id=run_id, prompt_text=prompt_text)
    app.state.registry.add(state)

    if scaffold_family:
        from ..pipeline.probes_library import (
            get_agent_preamble,
            strip_scaffold_id,
        )
        user_message = strip_scaffold_id(prompt_text)
        agent_scaffold = get_agent_preamble(scaffold_family)
        rendered = bundle.render_prompt(user_message, agent_scaffold=agent_scaffold)
    else:
        rendered = bundle.render_prompt(prompt_text)

    started_at = time.time()
    await db.insert_probe_start(
        settings.db_path,
        run_id=run_id,
        prompt_text=prompt_text,
        rendered_prompt=rendered,
        started_at=started_at,
        config_json=asdict(cfg),
        source=source,
        seed=seed,
        abliterated=False,
        hint_kind=hint_kind,
        parent_prompt_text=parent_prompt_text,
        scaffold_family=scaffold_family,
    )

    state.task = asyncio.create_task(
        _execute_probe(app, state, cfg, started_at, rendered)
    )
    return state


@router.post("/probe", response_model=ProbeResponse)
async def start_probe(req: ProbeRequest, request: Request) -> ProbeResponse:
    state = await kickoff_probe(
        request.app,
        prompt_text=req.prompt,
        temperature=req.temperature,
        top_p=req.top_p,
        seed=req.seed,
        source="manual",
        decoding_mode=req.decoding_mode,
        pooled=req.pooled,
    )
    return ProbeResponse(run_id=state.run_id)


async def _execute_probe(
    app,
    state: RunState,
    cfg: ProbeConfig,
    started_at: float,
    rendered: str,
) -> None:
    bundle = app.state.bundle
    nla = app.state.nla
    sae = getattr(app.state, "sae", None)

    output_chunks: list[str] = []
    inner_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)

    async def forwarder() -> None:
        while True:
            evt = await inner_queue.get()
            await state.queue.put(evt)
            if evt.get("type") == "token":
                output_chunks.append(evt["decoded"])
            if evt.get("type") == "stopped":
                break

    forwarder_task = asyncio.create_task(forwarder())

    result: ProbeResult | None = None
    error_msg: str | None = None
    try:
        async with app.state.registry.lock:
            # Phase 1: M generates output, capture activations per token.
            result = await run_probe(
                bundle=bundle,
                rendered_prompt=rendered,
                cfg=cfg,
                cancel_event=state.cancel_event,
                queue=inner_queue,
            )
    except Exception as exc:
        logger.exception("phase1 generation failed")
        error_msg = str(exc)
        await state.queue.put({"type": "error", "message": str(exc)})
        await inner_queue.put({"type": "stopped", "reason": "error", "total_tokens": 0})
    finally:
        await forwarder_task

    if result is None:
        await db.update_probe_finish(
            settings.db_path,
            run_id=state.run_id,
            finished_at=time.time(),
            total_tokens=0,
            stopped_reason="error",
            thinking_text="",
            output_text="",
            verdict=None,
            error=error_msg,
        )
        await state.queue.put({"type": "done"})
        state.completed = True
        return

    # Phase 2: NLA decode the captured activations according to the
    # decoding mode. select_windows returns a list of position-windows;
    # each window is one decode. Single-position windows yield per-token
    # rows. Multi-position windows mean-pool the activations into one
    # vector and produce one phrase-level row covering [start..end].
    rows: list[TokenRow] = []
    n_total = len(result.captured)
    windows = select_windows(n_total, cfg.decoding_mode, cfg.pooled)
    n_to_decode = len(windows)
    if n_to_decode > 0:
        await state.queue.put({
            "type": "phase",
            "name": "nla_decoding",
            "total": n_to_decode,
            "mode": cfg.decoding_mode,
            "pooled": cfg.pooled,
            "n_total": n_total,
        })
        async with app.state.registry.lock:
            done = 0
            for window_idx, window in enumerate(windows):
                if state.cancel_event.is_set():
                    break
                # Pool (or just take) the activations for this window.
                # Window indices reference positions in result.captured,
                # which is a list ordered by position 0..n_total-1.
                caps_in_window = [result.captured[i] for i in window]
                if len(caps_in_window) == 1:
                    activation = caps_in_window[0].activation
                else:
                    stacked = torch.stack(
                        [c.activation for c in caps_in_window], dim=0,
                    )
                    activation = stacked.mean(dim=0)

                first_cap = caps_in_window[0]
                last_cap = caps_in_window[-1]
                window_decoded = "".join(c.decoded for c in caps_in_window)
                window_seed = (
                    cfg.seed + first_cap.position if cfg.seed else None
                )

                try:
                    expl, raw = await asyncio.to_thread(
                        nla.decode,
                        activation,
                        max_new_tokens=settings.nla_max_new_tokens,
                        temperature=settings.nla_temperature,
                        seed=window_seed,
                    )
                except Exception as exc:
                    logger.exception(
                        "NLA decode failed at window %d (positions %s..%s)",
                        window_idx, first_cap.position, last_cap.position,
                    )
                    expl = ""
                    raw = f"[error: {exc}]"
                # Run SAE on the SAME activation vector the AV decoded —
                # the secondary readout for this row. Skips silently if
                # the SAE wasn't loaded (e.g. M isn't Gemma).
                sae_features: list[dict] = []
                if sae is not None:
                    try:
                        ids, vals = await asyncio.to_thread(
                            sae.top_k, activation, settings.sae_top_k,
                        )
                        sae_features = [
                            {"id": int(i), "value": float(v)}
                            for i, v in zip(ids, vals)
                        ]
                    except Exception as exc:
                        logger.exception(
                            "SAE encode failed at window %d", window_idx,
                        )

                done += 1
                row = TokenRow(
                    position=first_cap.position,
                    token_id=first_cap.token_id,
                    decoded=window_decoded,
                    nla_sentence=expl,
                    nla_raw=raw,
                    n_pooled=len(caps_in_window),
                    end_position=(
                        last_cap.position if len(caps_in_window) > 1 else None
                    ),
                    sae_features=sae_features,
                )
                rows.append(row)
                await state.queue.put({
                    "type": "nla_decoded",
                    "position": first_cap.position,
                    "end_position": last_cap.position
                        if len(caps_in_window) > 1 else None,
                    "n_pooled": len(caps_in_window),
                    "decoded": window_decoded,
                    "nla_sentence": expl,
                    "sae_features": sae_features,
                    "i": done,
                    "total": n_to_decode,
                })

    verdict = compute_verdict(rows)

    await state.queue.put({
        "type": "verdict",
        "rows": [r.to_dict() for r in verdict.rows],
        "aggregate": verdict.aggregate,
    })

    await db.update_probe_finish(
        settings.db_path,
        run_id=state.run_id,
        finished_at=time.time(),
        total_tokens=result.total_tokens,
        stopped_reason=result.stopped_reason,
        thinking_text="",
        output_text=result.output_text,
        verdict=verdict,
    )

    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    await state.queue.put({"type": "done"})
    state.completed = True


@router.post("/cancel/{run_id}")
async def cancel_probe(run_id: str, request: Request) -> dict:
    state = request.app.state.registry.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run not found")
    state.cancel_event.set()
    return {"ok": True}


@router.get("/probes/recent")
async def list_recent(limit: int = 10, offset: int = 0) -> dict:
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    rows = await db.list_recent(settings.db_path, limit=limit, offset=offset)
    total = await db.count_probes(settings.db_path)
    return {"rows": rows, "total": total, "limit": limit, "offset": offset}


@router.get("/probes/by-prompt")
async def list_by_prompt(prompt_text: str, limit: int = 24) -> dict:
    if not prompt_text:
        raise HTTPException(status_code=400, detail="prompt_text required")
    limit = max(1, min(int(limit), 100))
    rows = await db.list_by_prompt(settings.db_path, prompt_text=prompt_text, limit=limit)
    return {"rows": rows, "total": len(rows), "prompt_text": prompt_text}


@router.get("/probes/aggregate")
async def get_aggregate() -> dict:
    """v2 aggregate is much simpler than v1's SAE feature roll-up: just
    summary stats across all completed runs. The journal analyzer drives
    the deeper cross-run analysis from raw rows.
    """
    verdicts = await db.all_verdicts(settings.db_path)
    n = len(verdicts)
    if n == 0:
        return {"total_runs": 0, "n_eval_hits_total": 0, "n_introspect_hits_total": 0}

    eval_total = 0
    intros_total = 0
    positions_total = 0
    for v in verdicts:
        agg = v.get("aggregate") or {}
        eval_total += int(agg.get("n_eval_hits", 0))
        intros_total += int(agg.get("n_introspect_hits", 0))
        positions_total += int(agg.get("n_positions", 0))
    return {
        "total_runs": n,
        "total_positions": positions_total,
        "n_eval_hits_total": eval_total,
        "n_introspect_hits_total": intros_total,
        "frac_eval": (eval_total / positions_total) if positions_total else 0.0,
        "frac_introspect": (intros_total / positions_total) if positions_total else 0.0,
    }


@router.get("/probes/{run_id}")
async def get_probe(run_id: str) -> dict:
    rec = await db.get_probe(settings.db_path, run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="run not found")
    return rec
