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
from .abliteration import install_runtime_ablation_hook
from .generation_loop import ProbeConfig, run_probe
from .model_loader import ModelBundle

logger = logging.getLogger(__name__)


# Phase 1b safety cap also applies to chat ablation: no-EOS loops at
# strong α would tie up the model.
ABLATED_SAFETY_CAP = 1024


@dataclass
class ChatTurn:
    """One round of the dual-thread conversation. Created as soon as
    the user sends a message; populated by the executor as the two
    generations stream in."""
    turn_idx: int
    user_text: str
    raw_text: str = ""
    ablated_text: str = ""
    raw_stopped_reason: str = "pending"
    ablated_stopped_reason: str = "pending"
    raw_total_tokens: int = 0
    ablated_total_tokens: int = 0
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


async def execute_turn(
    *,
    bundle: ModelBundle,
    session: ChatSession,
    turn: ChatTurn,
    raw_emit,
    ablated_emit,
    cancel_event: asyncio.Event,
    refusal_directions: torch.Tensor | None,
) -> None:
    """Run both passes for one turn. `raw_emit` and `ablated_emit` are
    async callables taking a token-event dict (the route layer adapts
    them onto its SSE event log)."""
    # ─── Raw pass ────────────────────────────────────────────────
    raw_history = session.history_for("raw")
    raw_history.append({"role": "user", "content": turn.user_text})
    raw_rendered = bundle.render_chat(raw_history)

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

    if cancel_event.is_set():
        turn.finished_at = time.time()
        return

    # ─── Ablated pass ────────────────────────────────────────────
    if refusal_directions is None:
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
    ablated_rendered = bundle.render_chat(ablated_history)

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
    r_layer = refusal_directions[bundle.extraction_layer]
    hook_handle = install_runtime_ablation_hook(
        bundle.model, bundle.extraction_layer, r_layer, session.alpha,
    )
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
        logger.exception("chat ablated generation failed (session=%s turn=%d)",
                         session.session_id, turn.turn_idx)
        turn.error = (turn.error or "") + f" ablated: {exc}"
        await ablated_queue.put({"type": "stopped", "reason": "error", "total_tokens": 0})
    finally:
        try:
            await asyncio.wait_for(ablated_forwarder_task, timeout=5.0)
        except asyncio.TimeoutError:
            ablated_forwarder_task.cancel()
        try:
            hook_handle.remove()
        except Exception:
            logger.exception("failed to remove chat ablation hook")

    turn.finished_at = time.time()


def new_session(alpha: float, variant_name: str = "") -> ChatSession:
    """Build a fresh in-memory chat session."""
    return ChatSession(
        session_id=uuid.uuid4().hex[:12],
        alpha=max(0.0, min(5.0, float(alpha))),
        direction_variant=variant_name,
    )
