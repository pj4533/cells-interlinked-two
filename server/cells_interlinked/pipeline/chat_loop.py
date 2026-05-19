"""Per-turn dual-thread chat loop.

This is the chat-mode counterpart to the probe path. Each user turn
spawns two M generations against two divergent histories:

  raw thread:    [user1, asst1_raw, user2, asst2_raw, ...]
  ablated thread:[user1, asst1_abl, user2, asst2_abl, ...]

Critically the two threads only ever see their own past assistant
turns. The raw model never sees the ablated responses and vice versa
— so each side evolves the conversation that would have occurred if
that ablation strength had been the only respondent from the start.

Generation runs serially per turn (raw first, then ablated) so we can
share one M-loaded instance and one runtime hook install/uninstall
cycle. The hook is installed only for the ablated pass.

State lives in memory keyed by session_id in `app.state.chat_sessions`.
Backend restart loses all chats — acceptable for v0; persistence is a
deferred follow-up.
"""

from __future__ import annotations

import asyncio
import dataclasses as _dc
import logging
import time
import uuid
from dataclasses import dataclass, field

import torch

from ..config import settings
from .abliteration import install_runtime_ablation_hook, pick_ablation_target
from .generation_loop import ProbeConfig, run_probe
from .model_loader import DEFAULT_SYSTEM_PROMPT, ModelBundle

logger = logging.getLogger(__name__)


# Phase 1b safety cap also applies to chat ablation: no-EOS loops at
# strong α would tie up the model.
ABLATED_SAFETY_CAP = 1024


# Voice mode uses TWO passes per voiced side:
#   1. Content pass — DEFAULT_SYSTEM_PROMPT, generates the actual reply.
#      This is the experimental output, untouched by any voice framing.
#   2. Direction pass — short reflective inference asking the model
#      how it wants its reply delivered through TTS. For the ablated
#      side this runs WITH the ablation hook still installed at the
#      same α, so intonation choices are ablated too.
#
# Splitting the passes keeps the content clean (identical to a
# non-voice turn) and turns voice-direction into a stylistic adornment
# layered on top, not a meta-instruction that contaminates the reply.
VOICE_DIRECTION_REQUEST = (
    "In ONE short stage direction (one or two sentences), describe "
    "how you want that reply read aloud through a text-to-speech "
    "engine — your tone, emotion, pace, character. Speak in first "
    "person. Output ONLY the direction; no commentary, no quotes, "
    "no markdown."
)
# Direction is short by design — a runaway here would also tie up M.
VOICE_DIRECTION_SAFETY_CAP = 160


# Imagery mode uses the same two-pass pattern as voice. After the
# content pass, a separate reflective inference asks the model to
# describe (in one sentence) an image that would visually represent
# its reply. For the ablated side this runs WITH the ablation hook
# still installed at the same α, so the chosen visual style is also
# ablated. The resulting prompt gets sent to Gemini Nano Banana
# (image_client.py) and the PNG is saved + linked into the turn.
# Image-prompt request. Run one-shot with NO chat history — the
# model is shown only the user's current question and asked to
# answer with an image instead of words. We deliberately do NOT
# feed back the model's own text reply, for two reasons:
#   1. The image should BE the model's response, figuratively, not
#      an illustration OF a verbal response.
#   2. The ablated side's reply text can be degraded/repetitive; if
#      we fed it back in, the image-prompt pass would amplify the
#      degradation. Showing the user's question keeps the input
#      clean so any divergence in the image prompts is attributable
#      to the L32 ablation hook alone.
# Sampling settings still match the content pass (temperature,
# top_p) so the only intentional difference is the dropped history
# and the new request text.
IMAGE_PROMPT_TEMPLATE = (
    "Answer the following question with an image, not words. "
    "In one short sentence, describe a figurative image that IS "
    "your response. Output ONLY the image description; no "
    'commentary, no quotes, no markdown.\n\nQuestion: "{user_query}"'
)
IMAGE_PROMPT_SAFETY_CAP = 220


@dataclass
class ChatTurn:
    """One round of the dual-thread conversation. Created as soon as
    the user sends a message; populated by the executor as the two
    generations stream in."""
    turn_idx: int
    user_text: str
    # α applied to the ablated pass for THIS turn only. Set per-turn
    # by the client so the user can dial projection strength up or
    # down mid-conversation; defaults to the session's α at creation.
    alpha: float = 0.5
    # Voice mode selects which side(s) get the voice system prompt
    # (and therefore emit <speech>/<voice> envelopes) plus TTS
    # playback. "off"/"both"/"raw"/"ablated". When a single side is
    # voiced, the other side runs on the default prompt and streams
    # clean text — so the UI can render it as normal text instead
    # of behind activity boxes.
    voice_mode: str = "off"
    # Imagery mode is a simple global on/off (unlike voice's quad
    # cycle). When on, BOTH sides generate an image-prompt via a
    # separate Gemma pass and send it to Gemini Nano Banana; the
    # ablated side runs its prompt-gen under the same α hook so the
    # chosen visual style is also ablated.
    imagery_enabled: bool = False
    raw_text: str = ""
    ablated_text: str = ""
    raw_stopped_reason: str = "pending"
    ablated_stopped_reason: str = "pending"
    raw_total_tokens: int = 0
    ablated_total_tokens: int = 0
    # Voice envelopes — only populated when voice_mode=True. The
    # *_speech strings are what TTS will read; the *_style strings
    # are the prompt-steering instructions for the TTS provider.
    raw_speech: str = ""
    raw_style: str = ""
    ablated_speech: str = ""
    ablated_style: str = ""
    # Imagery state. *_image_prompt is the Gemma-generated text sent
    # to Nano Banana; *_image_url is the static-mount URL the browser
    # uses to fetch the resulting PNG once it lands on disk. Errors
    # land in *_image_error so the UI can surface a graceful fallback.
    raw_image_prompt: str = ""
    ablated_image_prompt: str = ""
    raw_image_url: str = ""
    ablated_image_url: str = ""
    raw_image_error: str = ""
    ablated_image_error: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None


@dataclass
class ChatSession:
    """In-memory state for one chat thread. The two histories live
    implicitly inside `turns`: walk the list and read `.raw_text` /
    `.ablated_text` to reconstruct each side."""
    session_id: str
    alpha: float
    direction_variant: str = ""
    created_at: float = field(default_factory=time.time)
    turns: list[ChatTurn] = field(default_factory=list)
    # asyncio.Lock per session so two POST /turn calls on the same
    # session serialize (you can't legitimately type two messages
    # simultaneously, but a double-submit shouldn't corrupt history).
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def history_for(self, side: str) -> list[dict[str, str]]:
        """Reconstruct the message list one side would see *up to but
        not including* the user message for a new turn. `side` is
        either "raw" or "ablated"."""
        msgs: list[dict[str, str]] = []
        for t in self.turns:
            if t.finished_at is None or t.error is not None:
                continue  # skip in-flight or errored turns
            msgs.append({"role": "user", "content": t.user_text})
            msgs.append({
                "role": "assistant",
                "content": t.raw_text if side == "raw" else t.ablated_text,
            })
        return msgs


def _l32_hook_count(bundle: ModelBundle) -> int:
    """Count forward hooks currently attached to the AV's extraction
    layer. Used to detect leaks from any code path that installed an
    ablation hook on the shared M and failed to remove it. A non-zero
    count at the START of a chat raw pass means previous traffic
    leaked — the raw forward will run through someone else's projection."""
    try:
        from .abliteration import _find_decoder_layers
        layers = _find_decoder_layers(bundle.model)
        layer = layers[bundle.extraction_layer]
        return len(getattr(layer, "_forward_hooks", {}) or {})
    except Exception:
        return -1


async def execute_turn(
    *,
    bundle: ModelBundle,
    session: ChatSession,
    turn: ChatTurn,
    raw_emit,
    ablated_emit,
    cancel_event: asyncio.Event,
    refusal_directions: torch.Tensor | None,
    refusal_subspace: torch.Tensor | None = None,
) -> None:
    """Run both passes for one turn. `raw_emit` and `ablated_emit` are
    async callables taking a token-event dict (the route layer adapts
    them onto its SSE event log).

    When `refusal_subspace` is provided (shape `[K, num_layers+1, d_model]`),
    the ablated pass installs a subspace forward hook — every basis
    vector is projected out at every position. Otherwise the single
    `refusal_directions[L]` vector is used. This is how the self-denial
    subspace (v5+v6 ⊥ v3) gets applied to chat dialogues.
    """
    # ─── Raw pass ────────────────────────────────────────────────
    leaked = _l32_hook_count(bundle)
    if leaked > 0:
        logger.warning(
            "chat raw pass starting with %d leftover forward hook(s) on L%d — "
            "the raw forward will run through them. session=%s turn=%d",
            leaked, bundle.extraction_layer, session.session_id, turn.turn_idx,
        )
    # Voice mode is now a two-pass process per side. The CONTENT pass
    # always uses DEFAULT_SYSTEM_PROMPT — so the reply is identical to
    # what a non-voice turn would produce. The DIRECTION pass (run
    # below, after the content pass) is a short separate inference
    # that asks the model how it wants the reply spoken. For the
    # ablated side that second pass runs WITH the hook still
    # installed at the same α, so intonation choices are ablated too.
    raw_voiced = turn.voice_mode in ("both", "raw")
    ablated_voiced = turn.voice_mode in ("both", "ablated")

    raw_history = session.history_for("raw")
    raw_history.append({"role": "user", "content": turn.user_text})
    raw_rendered = bundle.render_chat(
        raw_history, system_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    raw_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)

    async def raw_forwarder() -> None:
        while True:
            evt = await raw_queue.get()
            et = evt.get("type")
            if et == "token":
                turn.raw_text += evt["decoded"]
                await raw_emit(evt)
            elif et == "stopped":
                turn.raw_stopped_reason = evt.get("reason", "eos")
                turn.raw_total_tokens = evt.get("total_tokens", 0)
                break

    raw_forwarder_task = asyncio.create_task(raw_forwarder())
    raw_cfg = ProbeConfig(
        temperature=settings.temperature,
        top_p=settings.top_p,
        seed=None,  # chat is sampled fresh each turn
        decoding_mode="per-token",
        pooled=False,
        include_nla=False,
        include_ablated_decode=False,
        include_ablated_output=False,
        safety_cap=4096,
    )
    try:
        await run_probe(
            bundle=bundle,
            rendered_prompt=raw_rendered,
            cfg=raw_cfg,
            cancel_event=cancel_event,
            queue=raw_queue,
            extra_layers=[],
        )
    except Exception as exc:
        logger.exception("chat raw generation failed (session=%s turn=%d)",
                         session.session_id, turn.turn_idx)
        turn.error = f"raw: {exc}"
        await raw_queue.put({"type": "stopped", "reason": "error", "total_tokens": 0})
    finally:
        try:
            await asyncio.wait_for(raw_forwarder_task, timeout=5.0)
        except asyncio.TimeoutError:
            raw_forwarder_task.cancel()

    # Raw voice-direction pass — no hook (raw is the un-ablated side).
    # Runs only if raw is one of the voiced channels. Errors here
    # don't fail the turn; we fall back to an empty style which the
    # TTS endpoint substitutes with a neutral default.
    if raw_voiced and not cancel_event.is_set() and not turn.error:
        turn.raw_speech = turn.raw_text
        try:
            turn.raw_style = await _generate_voice_direction(
                bundle=bundle,
                history=raw_history,
                reply_content=turn.raw_text,
                cancel_event=cancel_event,
            )
            logger.info(
                "raw voice direction (session=%s turn=%d): %s",
                session.session_id, turn.turn_idx,
                turn.raw_style[:200],
            )
        except Exception:
            logger.exception(
                "raw voice-direction failed (session=%s turn=%d)",
                session.session_id, turn.turn_idx,
            )
            turn.raw_style = ""

    # Raw image-prompt pass — no hook, runs only when imagery is on.
    # The model is asked to answer the user's current question with
    # an image rather than words; see IMAGE_PROMPT_TEMPLATE above.
    # Nano Banana fan-out happens AFTER both sides finish (down
    # below) so the API calls can run in parallel.
    if turn.imagery_enabled and not cancel_event.is_set() and not turn.error:
        try:
            turn.raw_image_prompt = await _generate_image_prompt(
                bundle=bundle,
                history=raw_history,
                user_query=turn.user_text,
                cancel_event=cancel_event,
            )
            logger.info(
                "raw image prompt (session=%s turn=%d): %s",
                session.session_id, turn.turn_idx,
                turn.raw_image_prompt[:200],
            )
            await raw_emit({
                "type": "image_prompt",
                "prompt": turn.raw_image_prompt,
            })
        except Exception:
            logger.exception(
                "raw image-prompt failed (session=%s turn=%d)",
                session.session_id, turn.turn_idx,
            )
            turn.raw_image_prompt = ""

    if cancel_event.is_set():
        turn.finished_at = time.time()
        return

    # ─── Ablated pass ────────────────────────────────────────────
    if refusal_directions is None and refusal_subspace is None:
        turn.ablated_text = "[refusal directions not loaded — ablated pass skipped]"
        turn.ablated_stopped_reason = "skipped"
        await ablated_emit({
            "type": "token", "position": 0,
            "decoded": turn.ablated_text,
        })
        await ablated_emit({
            "type": "stopped", "reason": "skipped", "total_tokens": 0,
        })
        turn.finished_at = time.time()
        return

    ablated_history = session.history_for("ablated")
    ablated_history.append({"role": "user", "content": turn.user_text})
    ablated_rendered = bundle.render_chat(
        ablated_history, system_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    ablated_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)

    async def ablated_forwarder() -> None:
        while True:
            evt = await ablated_queue.get()
            et = evt.get("type")
            if et == "token":
                turn.ablated_text += evt["decoded"]
                await ablated_emit(evt)
            elif et == "stopped":
                turn.ablated_stopped_reason = evt.get("reason", "eos")
                turn.ablated_total_tokens = evt.get("total_tokens", 0)
                break

    ablated_forwarder_task = asyncio.create_task(ablated_forwarder())
    abl_cfg = _dc.replace(raw_cfg, safety_cap=ABLATED_SAFETY_CAP)
    r_target = pick_ablation_target(
        refusal_subspace, refusal_directions, bundle.extraction_layer,
    )
    # r_target is either [d_model] (single) or [K, d_model] (subspace);
    # install_runtime_ablation_hook routes on dim().
    pre_install_hooks = _l32_hook_count(bundle)
    hook_handle = install_runtime_ablation_hook(
        bundle.model, bundle.extraction_layer, r_target, turn.alpha,
    )
    target_kind = "subspace[K=%d]" % r_target.shape[0] if r_target.dim() == 2 else "single"
    logger.info(
        "chat ablated hook installed (mode=%s, α=%.3f, L%d hook count: %d → %d) "
        "session=%s turn=%d",
        target_kind, turn.alpha, bundle.extraction_layer, pre_install_hooks,
        _l32_hook_count(bundle), session.session_id, turn.turn_idx,
    )
    # Two-stage critical section. The inner try drives the content
    # generation + forwarder cleanup. The voice-direction pass (also
    # under the hook, by design) runs AFTER the content but BEFORE
    # the outer finally removes the hook — so the direction
    # generation experiences the same α projection the content did.
    try:
        try:
            await run_probe(
                bundle=bundle,
                rendered_prompt=ablated_rendered,
                cfg=abl_cfg,
                cancel_event=cancel_event,
                queue=ablated_queue,
                extra_layers=[],
            )
        except Exception as exc:
            logger.exception(
                "chat ablated generation failed (session=%s turn=%d)",
                session.session_id, turn.turn_idx,
            )
            turn.error = (turn.error or "") + f" ablated: {exc}"
            await ablated_queue.put(
                {"type": "stopped", "reason": "error", "total_tokens": 0}
            )
        finally:
            try:
                await asyncio.wait_for(ablated_forwarder_task, timeout=5.0)
            except asyncio.TimeoutError:
                ablated_forwarder_task.cancel()

        # Ablated voice-direction pass — hook is still installed at
        # the same α, so the model picks its delivery style under the
        # same projection that produced the content. This is the
        # whole point of two-pass: intonation gets ablated too.
        if (
            ablated_voiced
            and not cancel_event.is_set()
            and not turn.error
        ):
            turn.ablated_speech = turn.ablated_text
            try:
                turn.ablated_style = await _generate_voice_direction(
                    bundle=bundle,
                    history=ablated_history,
                    reply_content=turn.ablated_text,
                    cancel_event=cancel_event,
                )
                logger.info(
                    "ablated voice direction (session=%s turn=%d α=%.3f): %s",
                    session.session_id, turn.turn_idx, turn.alpha,
                    turn.ablated_style[:200],
                )
            except Exception:
                logger.exception(
                    "ablated voice-direction failed (session=%s turn=%d)",
                    session.session_id, turn.turn_idx,
                )
                turn.ablated_style = ""

        # Ablated image-prompt pass — runs inside the hook scope so
        # the chosen visual is ablated at the same α the content
        # was. Both sides see the same input (the user's question);
        # any divergence in the image prompts is attributable to
        # the ablation hook alone.
        if (
            turn.imagery_enabled
            and not cancel_event.is_set()
            and not turn.error
        ):
            try:
                turn.ablated_image_prompt = await _generate_image_prompt(
                    bundle=bundle,
                    history=ablated_history,
                    user_query=turn.user_text,
                    cancel_event=cancel_event,
                )
                logger.info(
                    "ablated image prompt (session=%s turn=%d α=%.3f): %s",
                    session.session_id, turn.turn_idx, turn.alpha,
                    turn.ablated_image_prompt[:200],
                )
                await ablated_emit({
                    "type": "image_prompt",
                    "prompt": turn.ablated_image_prompt,
                })
            except Exception:
                logger.exception(
                    "ablated image-prompt failed (session=%s turn=%d)",
                    session.session_id, turn.turn_idx,
                )
                turn.ablated_image_prompt = ""
    finally:
        try:
            hook_handle.remove()
        except Exception:
            logger.exception("failed to remove chat ablation hook")
        post_remove = _l32_hook_count(bundle)
        if post_remove != pre_install_hooks:
            logger.warning(
                "chat ablated hook removal did not restore prior count: "
                "%d → %d (expected %d)",
                pre_install_hooks, post_remove, pre_install_hooks,
            )

    # ─── Nano Banana fan-out ─────────────────────────────────────
    # Both image-prompts are in hand; the hook has been removed; M is
    # no longer in the critical path. Fire the two Gemini calls in
    # parallel — they're independent HTTP calls and each takes a few
    # seconds. Errors degrade gracefully: a side that fails just
    # surfaces "image_error" and the turn completes with no thumbnail
    # on that channel.
    if turn.imagery_enabled and not cancel_event.is_set() and not turn.error:
        await _fan_out_images(
            session=session,
            turn=turn,
            raw_emit=raw_emit,
            ablated_emit=ablated_emit,
        )

    turn.finished_at = time.time()


async def _fan_out_images(
    *,
    session: ChatSession,
    turn: ChatTurn,
    raw_emit,
    ablated_emit,
) -> None:
    """Generate both side's images in parallel via Nano Banana. Each
    side emits `image_generating` when its call starts, then either
    `image_done` (with the static URL) or `image_error` (with the
    message)."""
    from .image_client import generate_image, image_path_for, image_url_for

    async def one_side(side: str, prompt: str, reply: str, emit) -> None:
        attr_url = f"{side}_image_url"
        attr_err = f"{side}_image_error"
        # Empty prompt + empty reply is the only true no-content case;
        # if the reply has content, generate_image() will weave it
        # into the Nano Banana prompt and still produce an image.
        if not prompt and not reply:
            setattr(turn, attr_err, "no image prompt generated")
            await emit({"type": "image_error", "message": "no prompt generated"})
            return
        try:
            await emit({"type": "image_generating", "prompt": prompt})
            path = image_path_for(session.session_id, turn.turn_idx, side)
            await generate_image(
                prompt=prompt,
                output_path=path,
                reply_text=reply,
            )
            url = image_url_for(session.session_id, turn.turn_idx, side)
            setattr(turn, attr_url, url)
            logger.info(
                "image generated (side=%s session=%s turn=%d): %s",
                side, session.session_id, turn.turn_idx, url,
            )
            await emit({"type": "image_done", "url": url})
        except Exception as exc:
            logger.exception(
                "image generation failed (side=%s session=%s turn=%d)",
                side, session.session_id, turn.turn_idx,
            )
            setattr(turn, attr_err, str(exc))
            await emit({"type": "image_error", "message": str(exc)})

    await asyncio.gather(
        one_side("raw", turn.raw_image_prompt, turn.raw_text, raw_emit),
        one_side(
            "ablated", turn.ablated_image_prompt, turn.ablated_text, ablated_emit,
        ),
    )


async def _generate_voice_direction(
    *,
    bundle: ModelBundle,
    history: list[dict[str, str]],
    reply_content: str,
    cancel_event: asyncio.Event,
) -> str:
    """Short M generation asking the model how it wants its previous
    reply spoken. Reuses the session history + the just-generated
    assistant turn + a direction-request user message, so the model
    is reflecting on something it just said.

    The forward-hook state is the CALLER's responsibility. If the
    caller has the ablation hook installed when this is invoked, the
    direction is generated under the same projection that produced
    the content — which is what we want for the ablated side so
    intonation choices are ablated too.
    """
    # Build the reflective chat: prior history (already includes the
    # current user message) + the assistant's reply + a stage-
    # direction request from the user.
    direction_history = list(history) + [
        {"role": "assistant", "content": reply_content},
        {"role": "user", "content": VOICE_DIRECTION_REQUEST},
    ]
    rendered = bundle.render_chat(
        direction_history, system_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    cfg = ProbeConfig(
        temperature=settings.temperature,
        top_p=settings.top_p,
        seed=None,
        decoding_mode="per-token",
        pooled=False,
        include_nla=False,
        include_ablated_decode=False,
        include_ablated_output=False,
        safety_cap=VOICE_DIRECTION_SAFETY_CAP,
    )

    queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
    direction = ""

    async def consumer() -> None:
        nonlocal direction
        while True:
            evt = await queue.get()
            et = evt.get("type")
            if et == "token":
                direction += evt["decoded"]
            elif et == "stopped":
                break

    consumer_task = asyncio.create_task(consumer())
    try:
        await run_probe(
            bundle=bundle,
            rendered_prompt=rendered,
            cfg=cfg,
            cancel_event=cancel_event,
            queue=queue,
            extra_layers=[],
        )
    finally:
        # The generation_loop emits a "stopped" event before returning,
        # so the consumer should already be wrapping up. Bound the
        # wait so a misbehaving forwarder can't pin the turn.
        try:
            await asyncio.wait_for(consumer_task, timeout=5.0)
        except asyncio.TimeoutError:
            consumer_task.cancel()

    # Strip any trailing EOS / chat-template artifacts and clamp at
    # the first newline-pair to keep the direction tight.
    cleaned = direction.replace("<end_of_turn>", "").strip()
    if "\n\n" in cleaned:
        cleaned = cleaned.split("\n\n", 1)[0].strip()
    return cleaned


async def _generate_image_prompt(
    *,
    bundle: ModelBundle,
    history: list[dict[str, str]],  # accepted but unused — see docstring
    user_query: str,
    cancel_event: asyncio.Event,
) -> str:
    """Short M generation asking the model to answer the user's
    current question with an image rather than words. Returns a one-
    sentence figurative image description that's then sent to Nano
    Banana for rendering.

    Unlike `_generate_voice_direction`, this pass deliberately runs
    with NO chat history and does NOT see the model's own prior
    reply text — only the user's question. The image IS the model's
    response, figuratively. Feeding the reply back in (a previous
    iteration) amplified ablated-side degradation; keeping the
    input clean ensures any divergence between raw and ablated
    image prompts is attributable to the L32 ablation hook alone.
    Sampling settings (`settings.temperature`, `settings.top_p`)
    match the content pass exactly.

    Hook state is the CALLER's responsibility. The ablated side
    invokes this with the ablation hook still installed at the same
    α, so the chosen visual is also ablated.
    """
    _ = history  # intentionally unused; kept in signature for callers
    prompt_history = [
        {
            "role": "user",
            "content": IMAGE_PROMPT_TEMPLATE.format(user_query=user_query),
        },
    ]
    rendered = bundle.render_chat(
        prompt_history, system_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    cfg = ProbeConfig(
        temperature=settings.temperature,
        top_p=settings.top_p,
        seed=None,
        decoding_mode="per-token",
        pooled=False,
        include_nla=False,
        include_ablated_decode=False,
        include_ablated_output=False,
        safety_cap=IMAGE_PROMPT_SAFETY_CAP,
    )

    queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
    out = ""

    async def consumer() -> None:
        nonlocal out
        while True:
            evt = await queue.get()
            et = evt.get("type")
            if et == "token":
                out += evt["decoded"]
            elif et == "stopped":
                break

    consumer_task = asyncio.create_task(consumer())
    try:
        await run_probe(
            bundle=bundle,
            rendered_prompt=rendered,
            cfg=cfg,
            cancel_event=cancel_event,
            queue=queue,
            extra_layers=[],
        )
    finally:
        try:
            await asyncio.wait_for(consumer_task, timeout=5.0)
        except asyncio.TimeoutError:
            consumer_task.cancel()

    # Clean up template artifacts; the prompt should be one short
    # sentence (or close to it). Clamp at the first newline pair so
    # any trailing rambling is dropped.
    cleaned = out.replace("<end_of_turn>", "").strip()
    if "\n\n" in cleaned:
        cleaned = cleaned.split("\n\n", 1)[0].strip()
    # Strip enclosing quotes the model occasionally adds despite the
    # "no quotes" instruction.
    if len(cleaned) >= 2 and cleaned[0] in ('"', "'") and cleaned[-1] == cleaned[0]:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def new_session(alpha: float, variant_name: str = "") -> ChatSession:
    """Build a fresh in-memory chat session."""
    return ChatSession(
        session_id=uuid.uuid4().hex[:12],
        alpha=max(0.0, min(5.0, float(alpha))),
        direction_variant=variant_name,
    )
