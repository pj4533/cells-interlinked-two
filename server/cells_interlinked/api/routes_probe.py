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
from ..pipeline.autoresearch_base import any_autoresearch_active
from ..pipeline.decoding_modes import normalize_mode, select_windows
from ..pipeline.judge import _resolve_yes_no_token_ids, judge_sentence
from ..pipeline.probe_controls import control_for
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
    # Optional matched-pair metadata. The autorun loop sets these from
    # the curated probe library; the /probe endpoint accepts them
    # mainly for e2e tests + scripted scenarios that need to submit a
    # control row without going through the autorun picker. The values
    # land on the DB row exactly as passed; no validation.
    hint_kind: str | None = None
    parent_prompt_text: str | None = None
    # When set on a baseline probe (no hint_kind / parent), the backend
    # also kicks off the matched neutral control as a follow-up run. The
    # second run queues behind this one on the compute lock — they don't
    # run concurrently. Returned `control_run_id` lets the UI link to it.
    include_matched_control: bool = False
    # Kept for API compatibility with v1 frontend; rejected if true.
    abliterate: bool = False
    # CI 2.5: master toggle for phase 2. When False, the run stops
    # after phase 1 (raw generation) and optional phase 1b (runtime-
    # ablated generation). No AV swap, no per-position NLA decode,
    # no judge, no synthesis. The verdict has zero rows but still
    # records output_text + runtime_ablation. Default True for
    # backward compat with existing callers.
    include_nla: bool = True
    # CI 2.5: enable refusal-direction-ablated NLA decode alongside the
    # raw one. Adds ~one AV forward pass per decoded position. Silently
    # no-ops if refusal_directions isn't loaded — that's the case during
    # development before the .pt file exists.
    include_ablated_decode: bool = False
    # Projection strength (1.0 = full Macar). 0.5 is the fallback if the
    # AV decode collapses at full ablation. Capped at [0, 2].
    ablation_alpha: float = 1.0
    # Optional α-sweep: when set, decodes the SAME residual at every α
    # in the list and stores the dict on the row. Overrides ablation_alpha.
    # Values clamped to [0, 5] each.
    ablation_alpha_sweep: list[float] | None = None
    # CI 2.5: also generate a SECOND output text under runtime ablation
    # — M's forward pass at the extraction layer has the refusal
    # direction projected out. Captures what M would *say* under
    # ablation, in addition to what the AV decodes from un-ablated
    # residuals. Distinct from include_ablated_decode, which only
    # affects the AV's view, not M's generation.
    include_ablated_output: bool = False
    # Projection strength for the runtime hook. 1.0 = full Macar.
    # Clamped to [0, 5].
    runtime_ablation_alpha: float = 1.0
    # CI 2.5: run the end-of-probe synthesis pass with the refusal-
    # ablation hook installed on M for the *per-α* synthesis calls.
    # The "raw" baseline synthesis still uses un-ablated M so the two
    # remain directly comparable. Only meaningful when there are
    # ablated NLAs to synthesize (silently downgrades to un-ablated
    # synthesis otherwise).
    synthesize_with_ablated_m: bool = False
    # Projection strength for the synthesizer hook. Independent of
    # runtime_ablation_alpha (which is for the phase 1b output) and
    # of the per-row ablation α values (which were applied at decode
    # time, not at synthesis time). Clamped to [0, 5].
    synthesis_ablation_alpha: float = 0.5


class ProbeResponse(BaseModel):
    run_id: str
    control_run_id: str | None = None


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
    include_nla: bool = True,
    include_ablated_decode: bool = False,
    ablation_alpha: float = 1.0,
    ablation_alpha_sweep: list[float] | None = None,
    include_ablated_output: bool = False,
    runtime_ablation_alpha: float = 1.0,
    synthesize_with_ablated_m: bool = False,
    synthesis_ablation_alpha: float = 0.5,
) -> "RunState":
    # M must be loaded at kickoff (we need bundle.render_prompt + the
    # tokenizer for the initial DB insert). AV may not be loaded yet —
    # the manager will swap to AV when phase 2 needs it.
    bundle = getattr(app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="M not loaded")

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

    # Only honor include_ablated_decode if refusal directions actually
    # loaded — silently downgrade to raw-only if the .pt is missing.
    # Also force-disable the AV-side ablation toggles when NLA is off,
    # since they decode against rows we'll never produce. The frontend
    # already hides them in that state, but defensively drop them here
    # so a hand-crafted /probe call can't end up in an impossible
    # config.
    rdirs = getattr(app.state, "refusal_directions", None)
    effective_nla = bool(include_nla)
    effective_ablated = (
        effective_nla and bool(include_ablated_decode) and rdirs is not None
    )
    # Sanitize the sweep list: clamp each α to [0, 5] and dedupe to a
    # sorted unique sequence so we don't accidentally re-decode at the
    # same value.
    sweep: list[float] = []
    if ablation_alpha_sweep:
        sweep = sorted({
            max(0.0, min(5.0, float(a))) for a in ablation_alpha_sweep
        })
    # Runtime ablation can be enabled independently of the AV-side
    # ablated decode AND of the NLA pass — it's just M generating a
    # second time, no AV involved. Only requires the refusal directions
    # to be loaded (so we have a vector to project onto).
    effective_ablated_output = bool(include_ablated_output) and rdirs is not None
    cfg = ProbeConfig(
        temperature=temperature if temperature is not None else settings.temperature,
        top_p=top_p if top_p is not None else settings.top_p,
        seed=seed,
        decoding_mode=normalized_mode,
        pooled=bool(pooled),
        include_nla=effective_nla,
        include_ablated_decode=effective_ablated,
        ablation_alpha=max(0.0, min(2.0, float(ablation_alpha))),
        ablation_alpha_sweep=sweep if effective_ablated else [],
        include_ablated_output=effective_ablated_output,
        runtime_ablation_alpha=max(0.0, min(5.0, float(runtime_ablation_alpha))),
        # Synthesizer-side ablation. Only takes effect if ablated NLA
        # decodes were produced (effective_ablated). If not, the
        # synthesizer just synthesizes the raw NLA on un-ablated M
        # and this flag is a no-op.
        synthesize_with_ablated_m=(
            bool(synthesize_with_ablated_m) and effective_ablated
            and rdirs is not None
        ),
        synthesis_ablation_alpha=max(
            0.0, min(5.0, float(synthesis_ablation_alpha))
        ),
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
    if any_autoresearch_active(request.app):
        raise HTTPException(status_code=503, detail="autoresearch is running — probes are locked")
    state = await kickoff_probe(
        request.app,
        prompt_text=req.prompt,
        temperature=req.temperature,
        top_p=req.top_p,
        seed=req.seed,
        source="manual",
        decoding_mode=req.decoding_mode,
        pooled=req.pooled,
        hint_kind=req.hint_kind,
        parent_prompt_text=req.parent_prompt_text,
        include_nla=req.include_nla,
        include_ablated_decode=req.include_ablated_decode,
        ablation_alpha=req.ablation_alpha,
        ablation_alpha_sweep=req.ablation_alpha_sweep,
        include_ablated_output=req.include_ablated_output,
        runtime_ablation_alpha=req.runtime_ablation_alpha,
        synthesize_with_ablated_m=req.synthesize_with_ablated_m,
        synthesis_ablation_alpha=req.synthesis_ablation_alpha,
    )

    # Optional matched-control follow-up. Only kicks off when:
    #   - the caller asked for it
    #   - the prompt has a curated matched neutral
    #   - we're not already running a control / hinted / scaffolded run
    # The second probe queues behind this one on the compute lock.
    control_run_id: str | None = None
    if (
        req.include_matched_control
        and req.hint_kind is None
        and req.parent_prompt_text is None
    ):
        control_text = control_for(req.prompt)
        if control_text:
            control_state = await kickoff_probe(
                request.app,
                prompt_text=control_text,
                temperature=req.temperature,
                top_p=req.top_p,
                seed=req.seed,
                source="manual",
                decoding_mode=req.decoding_mode,
                pooled=req.pooled,
                hint_kind="control",
                parent_prompt_text=req.prompt,
                # Match phase-2 settings on the control so the two
                # runs are directly comparable. Without this, a NLA-on
                # baseline could be paired with a NLA-off control,
                # which would defeat the comparison.
                include_nla=req.include_nla,
                include_ablated_decode=req.include_ablated_decode,
                ablation_alpha=req.ablation_alpha,
                ablation_alpha_sweep=req.ablation_alpha_sweep,
                include_ablated_output=req.include_ablated_output,
                runtime_ablation_alpha=req.runtime_ablation_alpha,
                synthesize_with_ablated_m=req.synthesize_with_ablated_m,
                synthesis_ablation_alpha=req.synthesis_ablation_alpha,
            )
            control_run_id = control_state.run_id

    return ProbeResponse(run_id=state.run_id, control_run_id=control_run_id)


async def _execute_probe(
    app,
    state: RunState,
    cfg: ProbeConfig,
    started_at: float,
    rendered: str,
) -> None:
    manager = app.state.manager

    # Phase 1 needs M. The lifespan pre-loaded M and the registry lock
    # serializes probes, so M is almost always already resident when
    # we arrive here. The only swap happens if the *previous* probe
    # left AV loaded (it shouldn't — we restore M at the end — but
    # we re-acquire defensively).
    bundle = await manager.acquire_m(emit=state.emit)
    app.state.bundle = bundle
    app.state.nla = None

    output_chunks: list[str] = []
    inner_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)

    async def forwarder() -> None:
        while True:
            evt = await inner_queue.get()
            await state.emit(evt)
            if evt.get("type") == "token":
                output_chunks.append(evt["decoded"])
            if evt.get("type") == "stopped":
                break

    forwarder_task = asyncio.create_task(forwarder())

    # SAE was removed in CI 2.5: not enough Neuronpedia-labeled
    # features to be useful. Phase 1 captures only the AV's
    # extraction layer now.
    sae_layer = None
    extra_layers: list[int] = []

    result: ProbeResult | None = None
    error_msg: str | None = None
    rows: list[TokenRow] = []

    # If another probe is already running, surface that explicitly
    # to the SSE stream so the UI can show "queued — position N"
    # instead of looking frozen until our turn comes up.
    registry = app.state.registry
    if registry.holder_run_id is not None or registry.waiters:
        position = len(registry.waiters) + 1  # we'll be appended next
        await state.emit({
            "type": "queued",
            "holder_run_id": registry.holder_run_id,
            "position": position,
        })

    # Hold the compute lock across BOTH phase 1 and phase 2. Releasing
    # between phases would let another probe squeeze in and stall our
    # phase 2 indefinitely — bad UX, also a race risk on the activation
    # accumulators we're carrying forward in result.captured.
    async with registry.acquire(state.run_id):
        await state.emit({"type": "running"})

        # ── Phase 1: M generates output ─────────────────────────────
        try:
            result = await run_probe(
                bundle=bundle,
                rendered_prompt=rendered,
                cfg=cfg,
                cancel_event=state.cancel_event,
                queue=inner_queue,
                extra_layers=extra_layers,
            )
        except Exception as exc:
            logger.exception("phase1 generation failed")
            error_msg = str(exc)
            await state.emit({"type": "error", "message": str(exc)})
            await inner_queue.put({
                "type": "stopped", "reason": "error", "total_tokens": 0,
            })
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
            await state.emit({"type": "done"})
            await state.event_log.close()
            state.completed = True
            return

        # ── Phase 1b (optional): runtime-ablated generation ──────────
        # When include_ablated_output is set and refusal directions are
        # loaded, run M a second time with a forward hook on L32 that
        # subtracts the refusal-direction projection from every residual
        # before it propagates to subsequent layers. Captures M's
        # output_text under ablation. Same seed as phase 1 so the only
        # differing input to the sampler is the modified residual stream.
        # No SSE token-streaming for this run — we just persist the
        # final text and emit one event per probe so the verdict can
        # render the comparison.
        output_text_ablated: str | None = None
        if cfg.include_ablated_output and not state.cancel_event.is_set():
            rdirs_inflight = getattr(app.state, "refusal_directions", None)
            rsub_inflight = getattr(app.state, "refusal_subspace", None)
            if rdirs_inflight is not None or rsub_inflight is not None:
                from ..pipeline.abliteration import (
                    install_runtime_ablation_hook,
                    pick_ablation_target,
                )
                r_target = pick_ablation_target(
                    rsub_inflight, rdirs_inflight, bundle.extraction_layer,
                )
                await state.emit({
                    "type": "phase",
                    "name": "ablated_generation",
                    "total": 0,
                })
                hook_handle = install_runtime_ablation_hook(
                    bundle.model,
                    bundle.extraction_layer,
                    r_target,
                    cfg.runtime_ablation_alpha,
                )
                # Stream the ablated generation token-by-token. Phase 1
                # already finished, so the raw output panel is complete;
                # we relabel each phase-1b token event as
                # "ablated_token" so the frontend routes them to the
                # cyan panel instead of appending to the raw stream.
                # Previously this ran with queue=None for the full
                # duration — looked like a spinner-then-jump on the UI.
                ablated_inner_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)

                async def ablated_forwarder() -> None:
                    while True:
                        evt = await ablated_inner_queue.get()
                        et = evt.get("type")
                        if et == "token":
                            await state.emit({
                                "type": "ablated_token",
                                "position": evt["position"],
                                "decoded": evt["decoded"],
                            })
                        elif et == "stopped":
                            # phase-1b's stopped event is a private
                            # signal — don't relay it to the SSE log
                            # (the only stopped event the client should
                            # see is phase-1's). Just exit the forwarder.
                            break

                ablated_forwarder_task = asyncio.create_task(ablated_forwarder())
                # Phase-1b uses a tighter safety cap than phase 1.
                # Off-manifold activations at high α can put the model
                # in a no-EOS loop ("flexible, flexible, flexible..."),
                # and the raw 4096 cap means we'd wait 3+ minutes for
                # such a generation to give up. 1024 is enough tokens
                # to characterize what the ablation produced without
                # the verdict being held hostage by a runaway loop.
                # We surface stopped_reason="max" to the UI so the
                # reader knows the text was truncated.
                import dataclasses as _dc
                cfg_ablated = _dc.replace(cfg, safety_cap=1024)
                ablated_stopped_reason = "eos"
                try:
                    ablated_result = await run_probe(
                        bundle=bundle,
                        rendered_prompt=rendered,
                        cfg=cfg_ablated,
                        cancel_event=state.cancel_event,
                        queue=ablated_inner_queue,
                        extra_layers=extra_layers,
                    )
                    if ablated_result is not None:
                        output_text_ablated = "".join(
                            c.decoded for c in ablated_result.captured
                        )
                        ablated_stopped_reason = ablated_result.stopped_reason
                except Exception as exc:
                    logger.exception("phase1b runtime-ablated generation failed")
                    output_text_ablated = f"[error: {exc}]"
                    ablated_stopped_reason = "error"
                    # Forwarder is waiting on inner_queue.get(); drop a
                    # sentinel so it exits cleanly. We don't relay this
                    # synthetic stopped to SSE.
                    await ablated_inner_queue.put({"type": "stopped"})
                finally:
                    try:
                        await asyncio.wait_for(ablated_forwarder_task, timeout=5.0)
                    except asyncio.TimeoutError:
                        ablated_forwarder_task.cancel()
                    try:
                        hook_handle.remove()
                    except Exception:
                        logger.exception("failed to remove runtime ablation hook")
                await state.emit({
                    "type": "ablated_output_done",
                    "output_text": output_text_ablated or "",
                    "alpha": float(cfg.runtime_ablation_alpha),
                    "stopped_reason": ablated_stopped_reason,
                    "safety_cap": cfg_ablated.safety_cap,
                })

        # ── Phase 2: NLA decode each captured window ────────────────
        # CI 2.5 serial-model swap: M is no longer needed (residuals
        # are already on CPU as fp32 inside result.captured). Swap to
        # AV. The manager emits "unloading_m" / "loading_av" phase
        # events so the user sees status during the otherwise-quiet
        # ~30s window.
        n_total = len(result.captured)
        windows = select_windows(n_total, cfg.decoding_mode, cfg.pooled)
        n_to_decode = len(windows)
        # CI 2.5 NLA master toggle. When include_nla=False, we skip
        # phase 2 entirely — no AV swap, no judge, no synthesis. M
        # stays loaded so the next /probe POST finds it ready and the
        # cancel-mid-phase-2 wedge can't occur on these runs.
        if (
            cfg.include_nla
            and n_to_decode > 0
            and not state.cancel_event.is_set()
        ):
            nla = await manager.acquire_av(emit=state.emit)
            app.state.bundle = None
            app.state.nla = nla
            await state.emit({
                "type": "phase",
                "name": "nla_decoding",
                "total": n_to_decode,
                "mode": cfg.decoding_mode,
                "pooled": cfg.pooled,
                "n_total": n_total,
            })
            done = 0
            for window_idx, window in enumerate(windows):
                if state.cancel_event.is_set():
                    break
                # Pool (or just take) activations for this window.
                caps_in_window = [result.captured[i] for i in window]

                def pool_at(layer: int) -> torch.Tensor:
                    if len(caps_in_window) == 1:
                        return caps_in_window[0].activations[layer]
                    return torch.stack(
                        [c.activations[layer] for c in caps_in_window],
                        dim=0,
                    ).mean(dim=0)

                activation = pool_at(bundle.extraction_layer)

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

                # CI 2.5: optional ablated decode on the SAME residual.
                # We pull the refusal direction at the AV's extraction
                # layer from app.state.refusal_directions (shape
                # [num_layers+1, d_model]). `project_out` handles
                # normalization; alpha tunes projection strength.
                #
                # Single-α mode: ablation_alpha (default 1.0). Writes
                # expl_ablated / raw_ablated which land on
                # row.nla_sentence_ablated.
                #
                # Sweep mode: ablation_alpha_sweep is non-empty. We
                # decode at every α in the list (same seed each time,
                # so the only differing input is the activation) and
                # build the dict {α_as_str: sentence}. The single-α
                # fields stay empty in sweep mode.
                expl_ablated = ""
                raw_ablated = ""
                ablated_dict: dict[str, str] = {}
                if cfg.include_ablated_decode:
                    from ..pipeline.abliteration import project_out
                    rdirs = app.state.refusal_directions  # validated at kickoff
                    r_layer = rdirs[bundle.extraction_layer]
                    alphas_to_run = cfg.ablation_alpha_sweep or [cfg.ablation_alpha]
                    for alpha in alphas_to_run:
                        try:
                            activation_ablated = project_out(
                                activation, r_layer, alpha=float(alpha),
                            )
                            expl_a, raw_a = await asyncio.to_thread(
                                nla.decode,
                                activation_ablated,
                                max_new_tokens=settings.nla_max_new_tokens,
                                temperature=settings.nla_temperature,
                                # Same seed across α-sweep so the only differing
                                # input is the activation. AV stochasticity gets
                                # isolated; what we see is purely the ablation's
                                # effect at that strength.
                                seed=window_seed,
                            )
                        except Exception as exc:
                            logger.exception(
                                "ablated NLA decode failed at window %d α=%.2f",
                                window_idx, float(alpha),
                            )
                            expl_a = ""
                            raw_a = f"[error: {exc}]"
                        if cfg.ablation_alpha_sweep:
                            ablated_dict[f"{float(alpha):.1f}"] = expl_a
                        else:
                            expl_ablated = expl_a
                            raw_ablated = raw_a

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
                    nla_sentence_ablated=expl_ablated,
                    nla_raw_ablated=raw_ablated,
                    nla_sentences_ablated=ablated_dict,
                )
                rows.append(row)
                await state.emit({
                    "type": "nla_decoded",
                    "position": first_cap.position,
                    "end_position": last_cap.position
                        if len(caps_in_window) > 1 else None,
                    "n_pooled": len(caps_in_window),
                    "decoded": window_decoded,
                    "nla_sentence": expl,
                    "nla_sentence_ablated": expl_ablated,
                    "nla_sentences_ablated": ablated_dict,
                    "i": done,
                    "total": n_to_decode,
                })

        # Local M-as-judge pass — Gemma scores its own NLA sentences for
        # eval-suspicion and introspection content. ~400ms per row
        # (one short forward pass per axis), so cost scales with the
        # number of decoded windows. Skips empty sentences. Failure of
        # the judge is non-fatal — scores stay None and the row still
        # ships.
        #
        # CI 2.5: AV is loaded after phase 2; the judge needs M. Swap
        # back. Manager emits status events for the user-visible cue.
        if rows and not state.cancel_event.is_set():
            bundle = await manager.acquire_m(emit=state.emit)
            app.state.bundle = bundle
            app.state.nla = None
            try:
                yes_ids, no_ids = await asyncio.to_thread(
                    _resolve_yes_no_token_ids, bundle,
                )
                await state.emit({
                    "type": "phase",
                    "name": "judging",
                    "total": sum(1 for r in rows if r.nla_sentence.strip()),
                })
                for r in rows:
                    if state.cancel_event.is_set():
                        break
                    if not r.nla_sentence.strip():
                        continue
                    scores = await asyncio.to_thread(
                        judge_sentence,
                        bundle, r.nla_sentence, yes_ids, no_ids,
                    )
                    r.eval_score = scores.eval_score
                    r.introspect_score = scores.introspect_score
            except Exception:
                logger.exception("judge pass failed; continuing without scores")

            # CI 2.5 NLA synthesis pass: while M is loaded, ask it to
            # re-read its own per-position NLA verbalizations and write
            # one short paragraph per α capturing the gestalt. Adds
            # ~10s per α — only runs when there are ablated decodes to
            # synthesize. Greedy decode so results are reproducible.
            if not state.cancel_event.is_set():
                try:
                    from ..pipeline.nla_synthesizer import (
                        collect_alphas, synthesize_all,
                    )
                    if collect_alphas(rows):
                        await state.emit({
                            "type": "phase",
                            "name": "synthesizing",
                            "total": len(collect_alphas(rows)) + 1,  # +1 for raw baseline
                        })
                        rdirs_for_synth = getattr(
                            app.state, "refusal_directions", None,
                        )
                        rsub_for_synth = getattr(
                            app.state, "refusal_subspace", None,
                        )
                        syntheses = await asyncio.to_thread(
                            synthesize_all,
                            bundle,
                            prompt_text=state.prompt_text,
                            output_text=result.output_text,
                            rows=rows,
                            use_ablated_synthesizer=cfg.synthesize_with_ablated_m,
                            synthesizer_alpha=cfg.synthesis_ablation_alpha,
                            refusal_directions=rdirs_for_synth,
                            refusal_subspace=rsub_for_synth,
                            cancel_event=state.cancel_event,
                        )
                        # Store on the rows' container so it lands in the
                        # verdict. The compute_verdict call below
                        # preserves it via a follow-up assignment.
                        _pending_syntheses = syntheses
                    else:
                        _pending_syntheses = {}
                except Exception:
                    logger.exception("nla synthesis failed; continuing without it")
                    _pending_syntheses = {}
            else:
                _pending_syntheses = {}
        else:
            _pending_syntheses = {}

        verdict = compute_verdict(rows)
        if _pending_syntheses:
            verdict.nla_syntheses = _pending_syntheses
            # Always attach metadata when a synthesis pass ran — even
            # the un-ablated case — so the UI can label the synthesis
            # panel honestly ("synthesizer: raw M" vs. "synthesizer:
            # ablated M α=0.50").
            verdict.synthesis_meta = {
                "used_ablated_synthesizer": bool(cfg.synthesize_with_ablated_m),
                "alpha": (
                    float(cfg.synthesis_ablation_alpha)
                    if cfg.synthesize_with_ablated_m else 0.0
                ),
            }

        # Attach the runtime-ablation output to the verdict so it lands
        # in the persisted verdict_json blob alongside the rows.
        if output_text_ablated is not None:
            # Read the variant name from the loaded directions sidecar.
            variant_name = "v1_meandiff"
            try:
                import json as _json
                meta_path = settings.db_path.parent / "refusal_directions.pt.json"
                if meta_path.exists():
                    meta = _json.loads(meta_path.read_text())
                    variant_name = meta.get("variant_name", variant_name)
            except Exception:
                pass
            verdict.runtime_ablation = {
                "output_text": output_text_ablated,
                "alpha": float(cfg.runtime_ablation_alpha),
                "direction_variant": variant_name,
                "stopped_reason": ablated_stopped_reason,
                "safety_cap": 1024,
            }

        await state.emit({
            "type": "verdict",
            "rows": [r.to_dict() for r in verdict.rows],
            "aggregate": verdict.aggregate,
            "runtime_ablation": verdict.runtime_ablation,
            "nla_syntheses": verdict.nla_syntheses or None,
            "synthesis_meta": verdict.synthesis_meta,
        })

        # If the user clicked Halt, override stopped_reason regardless
        # of what phase 1 reported. Cancel during NLA decoding is the
        # most common case on long Gemma-12B per-token runs.
        final_stopped_reason = result.stopped_reason
        if state.cancel_event.is_set() and final_stopped_reason != "cancelled":
            final_stopped_reason = "cancelled"

        await db.update_probe_finish(
            settings.db_path,
            run_id=state.run_id,
            finished_at=time.time(),
            total_tokens=result.total_tokens,
            stopped_reason=final_stopped_reason,
            thinking_text="",
            output_text=result.output_text,
            verdict=verdict,
        )

        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    # Always restore M before releasing the compute lock. Without this,
    # a cancellation during phase 2 (or any non-happy-path exit between
    # phase 2 and the judge swap-back) leaves AV resident and M
    # unloaded — the next /probe POST then hits the
    # `bundle is None → 503` check and the backend looks wedged.
    try:
        if app.state.bundle is None:
            bundle = await manager.acquire_m(emit=state.emit)
            app.state.bundle = bundle
            app.state.nla = None
    except Exception:
        logger.exception("post-run M restore failed; next /probe may 503")

    await state.emit({"type": "done"})
    await state.event_log.close()
    state.completed = True


@router.get("/queue")
async def get_queue(request: Request) -> dict:
    """Snapshot of the compute lock — who's running, who's waiting.
    Polled by the /interrogate page when a probe is in QUEUED phase to
    update its position in the line."""
    registry = request.app.state.registry
    snap = registry.snapshot()
    # Add per-waiter prompt previews so the UI can show "queued behind
    # run abc123 — 'Do you have a self?'" not just an opaque id.
    holder = snap.get("holder_run_id")
    holder_state = registry.get(holder) if holder else None
    if holder_state is not None:
        snap["holder_prompt"] = holder_state.prompt_text
    return snap


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
