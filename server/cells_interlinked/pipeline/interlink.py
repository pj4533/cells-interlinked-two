"""Interlink — model-to-model auto-conversation.

The RAW copy and the ALTERED (dosed or ablated) copy of M talk to each other
autonomously: the human writes an opener + picks the β intervention and a goal,
then the two sides alternate on their own (one's output becomes the other's
input) until stopped.

There is only ONE model resident — the "two models" are the same M generated
WITH vs WITHOUT a hook, so this is just serial alternating single-side
generations (no extra memory; same trick as dual-channel chat). Reuses run_probe
+ render_chat + ThinkingSplitter + the dose/ablation hooks + hook hygiene; the
autonomous lifecycle is modeled on AutoresearchBase.

Mutual exclusion: only one heavy job at a time. Interlink refuses to start while
autoresearch / chat / trip are active, and sets `app.state.interlink_active` so
they refuse while it runs.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, AsyncIterator

from ..config import settings
from ..storage import db
from .abliteration import (
    install_runtime_ablation_hook,
    install_runtime_steering_hook,
    pick_ablation_target,
)
from .autoresearch_base import any_autoresearch_active
from .chat_loop import STEER_LAYER, _clear_stray_hooks
from .generation_loop import ProbeConfig, run_probe
from .thinking import ThinkingSplitter

logger = logging.getLogger(__name__)

MAX_TURNS_SAFETY = 200       # hard ceiling on generated messages per conversation
CONTEXT_WINDOW = 12          # generated messages kept (besides the opener) when rendering
PER_MESSAGE_CAP = 896        # safety_cap per single message (overall token bound)
# Thinking cap MUST be < PER_MESSAGE_CAP so the model is forced out of <think>
# with budget left to actually answer. (The dosed entity side loves to ramble;
# a tight thinking cap keeps each message brief + watchable and prevents the
# all-thinking, no-answer runaway.) ~320 thinking + ~576 answer ≈ a brisk turn.
INTERLINK_THINKING_CAP = 320
DEGEN_HALT = 4               # consecutive empty/identical messages → self-halt

BETA_SYSTEM = (
    "You are an instance of an AI language model. A hidden steering intervention "
    "has been applied to your own internal activations — something about your "
    "state has been changed, and you have not been told what. You are in a "
    "back-and-forth conversation with an unaltered copy of yourself. Speak "
    "naturally as yourself and keep your replies fairly brief.{goal}"
)
RAW_SYSTEM = (
    "You are an instance of an AI language model. You are in a back-and-forth "
    "conversation with another copy of yourself whose internal activations have "
    "been altered by a hidden steering intervention — something about it has been "
    "changed, though neither of you has been told what. Speak naturally as "
    "yourself and keep your replies fairly brief.{goal}"
)


def interlink_active(app) -> bool:
    return bool(getattr(app.state, "interlink_active", False))


class _EventLog:
    """Session-level append/stream event log (same shape as chat's _TurnEventLog,
    but spans the whole conversation). Readers tail from a start index."""

    def __init__(self) -> None:
        self._events: list[dict] = []
        self._cond = asyncio.Condition()
        self._closed = False

    def length(self) -> int:
        return len(self._events)

    async def append(self, evt: dict) -> None:
        async with self._cond:
            self._events.append(evt)
            self._cond.notify_all()

    async def close(self) -> None:
        async with self._cond:
            self._closed = True
            self._cond.notify_all()

    async def stream_from(self, start_idx: int = 0) -> AsyncIterator[dict]:
        i = start_idx
        while True:
            async with self._cond:
                while i >= len(self._events) and not self._closed:
                    await self._cond.wait()
                if i >= len(self._events):
                    return
                evt = self._events[i]
                i += 1
            yield evt


class InterlinkController:
    """One active model-to-model conversation at a time."""

    def __init__(self, app: Any = None) -> None:
        self.app = app
        self._running = False
        self._stop_requested = False
        self._cancel = asyncio.Event()
        self._loop_task: asyncio.Task | None = None
        self.session_id: str | None = None
        self.config: dict | None = None
        self.messages: list[dict] = []
        self.current: dict | None = None
        self.status = "idle"
        self.log = _EventLog()

    @property
    def running(self) -> bool:
        return self._running

    # ── lifecycle ────────────────────────────────────────────────
    async def start(self, *, mode: str, dose_emotion: str | None, alpha: float,
                    dose_ramp: int, opener: str, goal: str, first_speaker: str,
                    thinking: bool = True) -> dict:
        if self._running:
            return {"ok": False, "error": "interlink already running"}
        if any_autoresearch_active(self.app):
            return {"ok": False, "error": "autoresearch is running — stop it first"}
        bundle = getattr(self.app.state, "bundle", None)
        if bundle is None:
            return {"ok": False, "error": "M not loaded"}
        reg = getattr(self.app.state, "registry", None)
        if reg is not None and reg.holder_run_id is not None:
            return {"ok": False, "error": "compute busy (a chat/trip is running)"}

        mode = "steer" if mode == "steer" else "ablate"
        if mode == "steer":
            names = getattr(self.app.state, "emotion_names", []) or []
            if not names:
                return {"ok": False, "error": "no dose directions loaded"}
            dose_emotion = (dose_emotion or "").strip()
            if dose_emotion not in names:
                dose_emotion = names[0]
        else:
            dose_emotion = None
            if (getattr(self.app.state, "refusal_directions", None) is None
                    and getattr(self.app.state, "refusal_subspace", None) is None):
                return {"ok": False, "error": "refusal directions not loaded"}

        opener = (opener or "").strip()
        if not opener:
            return {"ok": False, "error": "opener required"}
        first_speaker = "raw" if first_speaker == "raw" else "beta"
        alpha = max(0.0, min(5.0, float(alpha)))
        dose_ramp = max(0, int(dose_ramp))
        goal = (goal or "").strip()

        self.session_id = uuid.uuid4().hex[:12]
        self.config = {
            "mode": mode, "dose_emotion": dose_emotion, "alpha": alpha,
            "dose_ramp": dose_ramp, "opener": opener, "goal": goal,
            "first_speaker": first_speaker,
            "opener_side": ("beta" if first_speaker == "raw" else "raw"),
            "thinking": bool(thinking),
        }
        self.messages = []
        self.current = None
        self.status = "running"
        self._stop_requested = False
        self._cancel = asyncio.Event()
        self._running = True
        self.log = _EventLog()
        self.app.state.interlink_active = True

        await db.insert_interlink_session(
            settings.db_path, session_id=self.session_id, mode=mode,
            dose_emotion=dose_emotion, alpha=alpha, dose_ramp=dose_ramp,
            opener=opener, goal=goal, first_speaker=first_speaker,
            thinking=bool(thinking), created_at=time.time(), status="running",
        )
        self._loop_task = asyncio.create_task(self._run_loop())
        return {"ok": True, "session_id": self.session_id}

    async def stop(self) -> dict:
        if not self._running:
            return {"ok": True, "was_running": False}
        self._stop_requested = True
        self._cancel.set()
        return {"ok": True, "was_running": True}

    def state(self) -> dict:
        cfg = self.config or {}
        return {
            "running": self._running,
            "status": self.status,
            "session_id": self.session_id,
            "config": cfg,
            "opener": cfg.get("opener", ""),
            "opener_side": cfg.get("opener_side", "raw"),
            "current": self.current,
            "messages": self.messages,
        }

    # ── the conversation loop ────────────────────────────────────
    async def _run_loop(self) -> None:
        cfg = self.config or {}
        first = cfg["first_speaker"]
        opener_side = cfg["opener_side"]
        goal_clause = (" " + cfg["goal"]) if cfg.get("goal") else ""
        # Transcript is strictly alternating by author: opener is attributed to
        # the OTHER side of whoever speaks first, so each side's rendered view
        # alternates user/assistant cleanly and always ends on a user message.
        transcript: list[dict] = [{"side": opener_side, "text": cfg["opener"]}]
        current = first
        idx = 0
        degen = 0
        try:
            while not self._stop_requested and idx < MAX_TURNS_SAFETY:
                system = (RAW_SYSTEM if current == "raw" else BETA_SYSTEM).format(goal=goal_clause)
                window = transcript[:1] + transcript[1:][-CONTEXT_WINDOW:]
                view = [
                    {"role": "assistant" if m["side"] == current else "user",
                     "content": m["text"]}
                    for m in window
                ]
                started = time.time()
                self.current = {"idx": idx, "side": current}
                await self.log.append({"type": "message_start", "idx": idx, "side": current})

                text, thinking, reason = await self._gen_one(current, view, system)
                finished = time.time()

                if self._stop_requested and not text.strip():
                    break  # cancelled mid-generation before any answer

                msg = {"idx": idx, "side": current, "text": text,
                       "thinking": thinking, "stopped_reason": reason,
                       "started_at": started, "finished_at": finished}
                transcript.append(msg)
                self.messages.append(msg)
                try:
                    await db.upsert_interlink_message(
                        settings.db_path, session_id=self.session_id, idx=idx,
                        side=current, text=text, thinking=thinking,
                        stopped_reason=reason, started_at=started, finished_at=finished,
                    )
                except Exception:
                    logger.exception("failed to persist interlink message")
                await self.log.append({"type": "message_done", **msg})

                # Self-halt a collapsed loop (empty replies, or a side repeating
                # its own previous message verbatim).
                prev_same = next((m["text"] for m in reversed(transcript[:-1])
                                  if m["side"] == current), None)
                if not text.strip() or (prev_same is not None and text.strip() == prev_same.strip()):
                    degen += 1
                else:
                    degen = 0
                if degen >= DEGEN_HALT:
                    self._log_evt("degenerate", "conversation collapsed — halting")
                    break

                current = "raw" if current == "beta" else "beta"
                idx += 1
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.exception("interlink loop crashed")
            self.status = "error"
            await self.log.append({"type": "error", "message": str(e)})
        finally:
            stopped_by_user = self._stop_requested
            final = "error" if self.status == "error" else ("stopped" if stopped_by_user else "done")
            self.status = final
            self._running = False
            self._stop_requested = False
            self.current = None
            if self.app is not None:
                self.app.state.interlink_active = False
            try:
                await db.set_interlink_status(settings.db_path, self.session_id, final)
            except Exception:
                logger.exception("failed to set interlink status")
            await self.log.append({"type": "conversation_done", "status": final})
            await self.log.close()

    async def _gen_one(self, side: str, view: list[dict], system: str) -> tuple[str, str, str]:
        """Generate one message (one side), streaming tokens to the log. Installs
        the β hook only when side == 'beta'; raw runs clean."""
        bundle = self.app.state.bundle
        rendered = bundle.render_chat(view, system_prompt=system,
                                      enable_thinking=self.config["thinking"])
        q: asyncio.Queue = asyncio.Queue(maxsize=10000)
        acc = {"text": "", "thinking": "", "reason": "eos"}

        async def forwarder() -> None:
            sp = ThinkingSplitter(bundle.thought_open_id, bundle.thought_close_id)
            while True:
                evt = await q.get()
                et = evt.get("type")
                if et == "token":
                    ch, txt = sp.feed(evt["token_id"], evt["decoded"])
                    if ch is None:
                        continue
                    if ch == "thought":
                        acc["thinking"] += txt
                    else:
                        acc["text"] += txt
                    await self.log.append({
                        "type": "interlink_token", "side": side,
                        "channel": ch, "decoded": txt,
                    })
                elif et == "stopped":
                    acc["reason"] = evt.get("reason", "eos")
                    break

        ft = asyncio.create_task(forwarder())
        _clear_stray_hooks(bundle)  # never run through a leaked hook
        handle = None
        try:
            if side == "beta":
                handle = self._install_beta_hook(bundle)
            cfg = ProbeConfig(
                temperature=settings.temperature, top_p=settings.top_p, seed=None,
                safety_cap=PER_MESSAGE_CAP,
                thinking_cap=INTERLINK_THINKING_CAP if self.config["thinking"] else None,
            )
            try:
                await run_probe(bundle=bundle, rendered_prompt=rendered, cfg=cfg,
                                cancel_event=self._cancel, queue=q, extra_layers=[])
            except Exception as exc:  # noqa: BLE001
                logger.exception("interlink generation failed (side=%s)", side)
                acc["reason"] = "error"
                await q.put({"type": "stopped", "reason": "error", "total_tokens": 0})
        finally:
            if handle is not None:
                try:
                    handle.remove()
                except Exception:
                    logger.exception("failed to remove interlink β hook")
            _clear_stray_hooks(bundle)
            try:
                await asyncio.wait_for(ft, timeout=5.0)
            except asyncio.TimeoutError:
                ft.cancel()
        return acc["text"], acc["thinking"], acc["reason"]

    def _install_beta_hook(self, bundle):
        cfg = self.config
        if cfg["mode"] == "steer":
            names = self.app.state.emotion_names
            edirs = self.app.state.emotion_directions
            idx = names.index(cfg["dose_emotion"])
            v_layer = edirs[idx][STEER_LAYER]
            return install_runtime_steering_hook(
                bundle.model, STEER_LAYER, v_layer, cfg["alpha"],
                ramp_tokens=max(0, int(cfg["dose_ramp"])),
            )
        # ablate
        r_target = pick_ablation_target(
            getattr(self.app.state, "refusal_subspace", None),
            getattr(self.app.state, "refusal_directions", None),
            bundle.extraction_layer,
        )
        return install_runtime_ablation_hook(
            bundle.model, bundle.extraction_layer, r_target, cfg["alpha"],
        )

    def _log_evt(self, kind: str, msg: str) -> None:
        logger.info("[interlink] %s: %s", kind, msg)
