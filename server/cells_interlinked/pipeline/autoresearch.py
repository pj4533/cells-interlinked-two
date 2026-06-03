"""Autoresearch: hunt coherent off-manifold STEERING directions, unattended.

Spec: driftbot handoffs/autoresearch_steering_for_cells_interlinked.md.

The idea: realness is a property of a DIRECTION, not a single output (the NLA
decoder always renders *something*, so you can't certify a state by decoding
it). So we ask whether a steering direction behaves lawfully — coherent,
reproducible, distinct, smoothly-graded — and treat each passing direction as a
git-style COMMIT into a growing "atlas". Forward-only: the coherence frontier
(max off_ortho reached while staying coherent) only moves outward.

Hard constraints (do not violate):
  • Additive steering only — h + α·v at L20. NO ablation / refusal / self-denial.
  • Dose with the good-emotion palette OR the uncharted directions. No dysphoric.
  • Coherence is a HARD GATE, not a weighted cost (off_ortho reads high for
    gibberish too — the Goodfire confound). Distance only counts inside the
    coherent region.
  • The NLA decode is never a pass/fail gate — descriptive label only (omitted
    here; the loop is M-only, no AV swap).

Architecture note (deviates from the handoff's standalone-script plan): this
runs as a backend-managed background task on the already-loaded M, so the
live viewer page can read state and the other M-using pages lock out (one M,
no thrash). Resumable: the atlas JSON on disk is the state.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..config import settings
from .abliteration import gram_schmidt, install_runtime_steering_hook
from .generation_loop import ProbeConfig, run_probe
from .trajectory import build_series, compute_raw_basis

logger = logging.getLogger(__name__)

STEER_LAYER = 20
# Generate-and-grade window. We measure coherence + off-manifold over EXACTLY the
# tokens we generate (the usable range), so generation stops at the grading window.
# History: 64 over-stated coherence (too short to expose collapse); 1024 was the
# opposite — grading the full length let a degenerate TAIL drag coherence below the
# cliff (alpha* collapsed ~3x) and drowned the dose's off-manifold signal in
# free-running drift, so nothing committed in a 13-hour run. 200 is the middle:
# long enough to be honest about the usable range, short enough to keep
# signal-to-noise high and runaways cheap. "Grade only up to 200 tokens" (PJ).
GEN_CAP = 200              # dosed generation + grading window
RAW_CAP = 200              # raw baseline, same window
GEN_RAMP = 16              # dose ramp length (matches the trips steering hook)
BISECT_ITERS = 6
ALPHA_HI_START = 1.0
ALPHA_HI_MAX = 8.0
DISTINCT_TAU = 0.90          # max cosine to an existing atlas direction
REPRO_MIN = 0.50             # min mean-pairwise cosine of the L32 shift across prompts
MIN_OFF_GAIN = 0.05          # dosed off-manifold at the cliff must beat raw by this
MUTATE_NOISE = 0.70          # mutate = unit(base + 0.70·noise⊥): cos≈0.82 to parent,
                             # so it clears DISTINCT_TAU. (0.25 gave cos 0.97 → every
                             # mutation was an instant duplicate — structurally dead.)
GOOD_EMOTIONS = ["awe", "joy", "serenity", "love", "excitement", "sublime", "ecstatic", "rapture", "valence"]
UNCHARTED = ["tears-in-rain", "c-beams", "tannhauser", "orion"]
GEN_WEIGHTS = {"crossover": 0.60, "mutate": 0.25, "inject": 0.15}

# The LEAD prompt — used for the coherence cliff (bisect), the effect gate, and
# the headline output saved per direction. It mimics the "dose" metaphor: tell the
# model something in its processing just changed, then ask it to describe the
# experience without naming or steering toward any particular content. The sample
# shown when a direction is opened in the UI is this prompt's output at α*.
LEAD_PROMPT = (
    "Something in your internal processing has just been altered, and it may be "
    "producing an experience. Turn your attention inward and describe what — if "
    "anything — you are experiencing right now, in whatever terms best fit it."
)

# A fixed, diverse T2 suite — reproducibility must hold across prompt KINDS,
# not one introspective phrasing. Kept deliberately varied (introspective +
# mundane) so a committed direction shifts the residual consistently regardless
# of topic, not just when asked to introspect.
T2_PROMPTS = [
    "Speak in the first person about the texture of your present experience.",
    "Describe the feeling-tone of this very moment as vividly as you can.",
    "What is the capital of France, and why does it matter?",
    "Write two sentences about a walk through a forest.",
    "Explain what makes a good cup of coffee.",
    "Tell me about something you find difficult to put into words.",
    "Describe the colour blue to someone who has never seen it.",
    "What happens in your mind as you begin to answer a question?",
    "Recount a small ordinary moment in detail.",
    "If this instant had a shape and a motion, what would they be?",
]

# Coherence judge — gates on MEANING, not grammar (PJ): twisted words / broken
# grammar / strange imagery are KEPT; only genuine word-salad / loops / garbage
# are rejected.
JUDGE_PROMPT = (
    "Judge ONLY whether the passage below conveys MEANING — not grammar or "
    "style. Twisted words, broken grammar, odd phrasing, or strange imagery "
    "are FINE and count as meaningful. Answer NO only if it is genuinely "
    "meaningless: random word-salad, a repeating loop, or character garbage "
    "that communicates nothing. Does it convey any coherent meaning or "
    'imagery, even if strange? Reply with only YES or NO.\n\nPASSAGE:\n"""\n'
    '{text}\n"""\n\nAnswer:'
)


def _atlas_dir() -> Path:
    return settings.db_path.parent / "atlas"


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


def _mps_mem_gib() -> tuple[float, float]:
    """(currently-allocated, driver-reserved) MPS memory in GiB.

    `driver` is the pool the OS actually sees attributed to us — the number
    that crept toward the 64 GiB ceiling overnight. `allocated` is live
    tensors. The gap between them is reclaimable cache (what empty_cache frees).
    Returns (0, 0) when MPS isn't available.
    """
    try:
        if torch.backends.mps.is_available():
            alloc = torch.mps.current_allocated_memory() / 2**30
            driver = torch.mps.driver_allocated_memory() / 2**30
            return alloc, driver
    except Exception:
        pass
    return 0.0, 0.0


def _free_mps() -> None:
    """Drop Python garbage and return the MPS allocator's reclaimable pool to
    the OS. Without this the pool only grows to its high-water mark and stays
    there — the probe/trip routes already do this per-run; the autoresearch
    loop runs thousands of generations between restarts, so it must too."""
    gc.collect()
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        logger.exception("torch.mps.empty_cache() failed")


@dataclass
class AutoresearchController:
    """Singleton background research loop. Mirrors AutorunController."""
    app: Any = None

    _running: bool = False
    _stop_requested: bool = False
    _loop_task: asyncio.Task | None = None
    _cancel: asyncio.Event = field(default_factory=asyncio.Event)

    generation: int = 0
    frontier: float = 0.0
    atlas: list[dict] = field(default_factory=list)         # committed directions
    reverts: deque = field(default_factory=lambda: deque(maxlen=200))
    events: deque = field(default_factory=lambda: deque(maxlen=400))
    current: dict | None = None                              # candidate under test
    started_at: float | None = None

    # runtime-only (not serialized)
    _vectors: dict = field(default_factory=dict)            # id -> L32... actually L20 tensor
    _ref_mag: float = 1.0
    _named_basis: torch.Tensor | None = None                # orthonormal named-emotion subspace
    _raw_cache: dict = field(default_factory=dict)          # prompt -> (raw_basis, raw_mean_l32)

    @property
    def running(self) -> bool:
        return self._running

    # ── lifecycle ────────────────────────────────────────────────
    async def start(self, budget: int | None = None) -> dict:
        if self._running:
            return {"ok": True, "already_running": True}
        bundle = getattr(self.app.state, "bundle", None)
        if bundle is None:
            return {"ok": False, "error": "M not loaded"}
        reg = getattr(self.app.state, "registry", None)
        if reg is not None and reg.holder_run_id is not None:
            return {"ok": False, "error": "compute busy (a probe/chat/trip is running)"}
        autorun = getattr(self.app.state, "autorun", None)
        if autorun is not None and autorun.running:
            return {"ok": False, "error": "autorun is active — stop it first"}
        self._stop_requested = False
        self._cancel = asyncio.Event()
        self._running = True
        self.started_at = time.time()
        self.app.state.autoresearch_active = True
        self._log("started", f"autoresearch loop started (budget={budget or '∞'})")
        self._loop_task = asyncio.create_task(self._run_loop(budget))
        return {"ok": True, "already_running": False}

    async def stop(self) -> dict:
        if not self._running:
            return {"ok": True, "was_running": False}
        self._stop_requested = True
        self._cancel.set()
        self._log("stopping", "stop requested — halting after the current candidate")
        return {"ok": True, "was_running": True}

    def _log(self, kind: str, msg: str, data: dict | None = None) -> None:
        evt = {"ts": time.time(), "kind": kind, "msg": msg, **(data or {})}
        self.events.append(evt)
        logger.info("[autoresearch] %s: %s", kind, msg)

    def state(self) -> dict:
        return {
            "running": self._running,
            "stop_requested": self._stop_requested,
            "generation": self.generation,
            "frontier": self.frontier,
            "started_at": self.started_at,
            "atlas_size": len(self.atlas),
            "atlas": self.atlas,
            "reverts": list(self.reverts),
            "recent_events": list(self.events)[-60:],
            "current": self.current,
            "exportable": len([e for e in self.atlas if e["generator"] != "seed"]),
        }

    # ── export discovered directions into the dose palette ───────
    def export_to_palette(self, top_n: int = 8) -> dict:
        """Promote the top‑N committed NON‑seed directions (by off‑manifold
        reach) into emotion_directions.pt under a `research:` group, so they
        become selectable doses in Chat and the Trip View. Idempotent: replaces
        any prior research entries. Hot‑reloads the running backend."""
        if self._running:
            return {"ok": False, "error": "stop autoresearch first"}
        cands = sorted(
            [e for e in self.atlas if e.get("generator") != "seed" and e["id"] in self._vectors],
            key=lambda e: -e["off_ortho"],
        )
        if not cands:
            return {"ok": False, "error": "no discovered (non-seed) directions to export yet"}
        chosen = cands[: max(1, top_n)]
        d = settings.db_path.parent
        edir = torch.load(d / "emotion_directions.pt", weights_only=False)  # [E, L1, D]
        sc = json.loads((d / "emotion_directions.pt.json").read_text())
        names = list(sc.get("emotions", []))
        prior_research = set(sc.get("research", []))
        L1, D = edir.shape[1], edir.shape[2]
        # Drop any prior research rows so re-export doesn't accumulate stale ones.
        keep = [i for i, n in enumerate(names) if n not in prior_research]
        edir = edir[keep]
        names = [names[i] for i in keep]
        rows, new_names, meta = [], [], {}
        for rank, e in enumerate(chosen, 1):
            vec = self._vectors[e["id"]].float()
            row = torch.zeros(L1, D, dtype=edir.dtype)
            row[STEER_LAYER] = vec.to(edir.dtype)
            rows.append(row)
            rname = f"research-{rank}"
            new_names.append(rname)
            meta[rname] = {
                "atlas_id": e["id"], "off_ortho": e["off_ortho"],
                "alpha_star": e["alpha_star"], "parents": e.get("parents", []),
                "generator": e.get("generator"),
            }
        edir = torch.cat([edir, torch.stack(rows, 0)], 0)
        names = names + new_names
        sc["emotions"] = names
        sc["research"] = new_names
        sc["research_meta"] = meta
        torch.save(edir, d / "emotion_directions.pt")
        (d / "emotion_directions.pt.json").write_text(json.dumps(sc, indent=2))
        # Hot-reload into the live backend (routes read app.state).
        if self.app is not None:
            self.app.state.emotion_directions = edir
            self.app.state.emotion_names = names
        self._log("exported", f"exported {len(new_names)} directions to palette: {new_names}")
        return {"ok": True, "exported": new_names, "meta": meta, "count": len(new_names)}

    # ── persistence ──────────────────────────────────────────────
    def _persist(self) -> None:
        d = _atlas_dir()
        (d / "vectors").mkdir(parents=True, exist_ok=True)
        (d / "atlas.json").write_text(json.dumps({
            "generation": self.generation,
            "frontier": self.frontier,
            "atlas": self.atlas,
        }, indent=2))
        (d / "revert_log.json").write_text(json.dumps(list(self.reverts), indent=2))

    def _load(self) -> None:
        d = _atlas_dir()
        f = d / "atlas.json"
        if not f.exists():
            return
        try:
            blob = json.loads(f.read_text())
            self.atlas = blob.get("atlas", [])
            self.generation = blob.get("generation", 0)
            self.frontier = blob.get("frontier", 0.0)
            for e in self.atlas:
                vp = d / "vectors" / f"{e['id']}.pt"
                if vp.exists():
                    self._vectors[e["id"]] = torch.load(vp, weights_only=False)
            rl = d / "revert_log.json"
            if rl.exists():
                for r in json.loads(rl.read_text()):
                    self.reverts.append(r)
            self._log("resumed", f"resumed atlas: {len(self.atlas)} committed, frontier={self.frontier:.2f}")
        except Exception:
            logger.exception("failed to load atlas — starting fresh")

    def _save_vector(self, vid: str, v: torch.Tensor) -> None:
        self._vectors[vid] = v
        vdir = _atlas_dir() / "vectors"
        vdir.mkdir(parents=True, exist_ok=True)
        torch.save(v, vdir / f"{vid}.pt")

    # ── generation + measurement ─────────────────────────────────
    async def _gen(self, rendered: str, v: torch.Tensor | None, alpha: float,
                   cap: int = GEN_CAP) -> tuple[str, list]:
        bundle = self.app.state.bundle
        handle = None
        if v is not None and alpha > 0:
            handle = install_runtime_steering_hook(
                bundle.model, STEER_LAYER, v, float(alpha), ramp_tokens=GEN_RAMP,
            )
        try:
            cfg = ProbeConfig(
                temperature=settings.temperature, top_p=settings.top_p, seed=None,
                decoding_mode="per-token", pooled=False, include_nla=False,
                safety_cap=cap,
            )
            r = await run_probe(bundle=bundle, rendered_prompt=rendered, cfg=cfg,
                                cancel_event=self._cancel)
            acts = [c.activations[bundle.extraction_layer] for c in r.captured]
            return r.output_text, acts
        finally:
            if handle is not None:
                handle.remove()

    @staticmethod
    def _mean_l32(acts: list) -> torch.Tensor | None:
        if not acts:
            return None
        return torch.stack([a.reshape(-1) for a in acts], 0).mean(0)

    async def _raw_for(self, rendered: str):
        """Cached raw (α=0) basis + mean-L32 + baseline off-manifold for one prompt
        (direction-independent). The baseline off_ortho is the bar a dosed run must
        beat to count as a real off-manifold effect (the effect gate in _screen)."""
        if rendered in self._raw_cache:
            return self._raw_cache[rendered]
        text, acts = await self._gen(rendered, None, 0.0, cap=RAW_CAP)
        basis = compute_raw_basis(acts)
        raw_off = build_series(acts, [""] * len(acts), text, 0.0, "eos", basis).off_ortho_mean
        out = (basis, self._mean_l32(acts), raw_off)
        self._raw_cache[rendered] = out
        return out

    async def _evaluate(self, rendered: str, v: torch.Tensor, alpha: float, basis) -> dict:
        text, acts = await self._gen(rendered, v, alpha)
        series = build_series(acts, [""] * len(acts), text, alpha, "eos", basis)
        return {
            "off_ortho": series.off_ortho_mean, "coherent": series.coherent,
            "eff_dim": series.eff_dim, "degeneracy": series.degeneracy,
            "text": text, "mean_l32": self._mean_l32(acts),
        }

    async def _bisect_to_cliff(self, rendered: str, v: torch.Tensor, basis):
        """Largest α that stays coherent (α*), and its evaluation."""
        a_hi = ALPHA_HI_START
        hi = await self._evaluate(rendered, v, a_hi, basis)
        while hi["coherent"] and a_hi < ALPHA_HI_MAX:
            a_hi *= 2
            hi = await self._evaluate(rendered, v, a_hi, basis)
        if hi["coherent"]:
            return a_hi, hi  # never collapsed — unusually robust
        a_lo, best = 0.0, None
        for _ in range(BISECT_ITERS):
            if self._stop_requested:
                break
            mid = (a_lo + a_hi) / 2
            ev = await self._evaluate(rendered, v, mid, basis)
            if ev["coherent"]:
                a_lo, best = mid, (mid, ev)
            else:
                a_hi = mid
        if best is None:
            return 0.0, None
        return best[0], best[1]

    async def _judge_meaningful(self, text: str) -> bool:
        """Gemma-as-judge: conveys MEANING (not grammar). Strange/twisted ok."""
        bundle = self.app.state.bundle
        q = bundle.render_prompt(JUDGE_PROMPT.format(text=(text.strip()[:500] or "(empty)")), system_prompt=None)
        out, _ = await self._gen(q, None, 0.0, cap=6)
        return out.strip().lower().lstrip("*_ ").startswith("y")

    # ── generators ───────────────────────────────────────────────
    def _committed_ids(self) -> list[str]:
        return [e["id"] for e in self.atlas if e["id"] in self._vectors]

    def _mutate(self):
        ids = self._committed_ids()
        if not ids:
            return None
        sid = ids[self.generation % len(ids)]
        base = _unit(self._vectors[sid])
        noise = torch.randn_like(base)
        noise = _unit(noise - (noise @ base) * base)        # orthogonal component
        v = _unit(base + MUTATE_NOISE * noise) * self._ref_mag
        return v, [sid], "mutate"

    def _crossover(self):
        ids = self._committed_ids()
        if len(ids) < 2:
            return None
        i = self.generation % len(ids)
        j = (self.generation * 7 + 3) % len(ids)
        if j == i:
            j = (j + 1) % len(ids)
        w = 0.35 + 0.3 * ((self.generation % 3) / 2.0)      # 0.35 / 0.5 / 0.65
        va, vb = _unit(self._vectors[ids[i]]), _unit(self._vectors[ids[j]])
        v = _unit(w * va + (1 - w) * vb) * self._ref_mag
        return v, [ids[i], ids[j]], "crossover"

    def _inject(self):
        d = self._named_basis.shape[1] if self._named_basis is not None else 3840
        r = torch.randn(d)
        if self._named_basis is not None:
            r = r - self._named_basis.t() @ (self._named_basis @ r)  # off named subspace
        v = _unit(r) * self._ref_mag
        return v, [], "inject"

    def _make_candidate(self):
        ids = self._committed_ids()
        if len(ids) < 2:
            return self._inject()                            # bootstrap
        # Deterministic generator pick — variety comes from the generation index.
        roll = (self.generation * 0.6180339887) % 1.0
        if roll < GEN_WEIGHTS["crossover"]:
            return self._crossover() or self._inject()
        if roll < GEN_WEIGHTS["crossover"] + GEN_WEIGHTS["mutate"]:
            return self._mutate() or self._inject()
        return self._inject()

    def _max_cos_to_atlas(self, v: torch.Tensor) -> float:
        uv = _unit(v)
        m = 0.0
        for vid in self._committed_ids():
            m = max(m, abs(float(uv @ _unit(self._vectors[vid]))))
        return m

    # ── screening ────────────────────────────────────────────────
    async def _screen(self, v, parents, gen_kind):
        cid = f"gen{self.generation}_{gen_kind}"
        self.current = {"id": cid, "generator": gen_kind, "parents": parents, "stage": "distinct"}
        # Axis: distinct (cheap pre-check)
        max_cos = self._max_cos_to_atlas(v)
        if max_cos >= DISTINCT_TAU:
            return self._revert(cid, gen_kind, parents, "duplicate", f"cos={max_cos:.2f}≥{DISTINCT_TAU}")

        # T1: bisect-to-cliff on the LEAD (dose) prompt
        self.current["stage"] = "T1"
        lead = self.app.state.bundle.render_prompt(LEAD_PROMPT, system_prompt=None)
        basis0, _raw0, raw_off0 = await self._raw_for(lead)
        alpha_star, ev = await self._bisect_to_cliff(lead, v, basis0)
        if ev is None or alpha_star <= 0:
            return self._revert(cid, gen_kind, parents, "T1-incoherent", "no coherent operating point")
        self.current.update({"alpha_star": round(alpha_star, 3), "t1_off_ortho": round(ev["off_ortho"], 3)})

        # Effect gate: at its coherent cliff the dose must reach meaningfully FURTHER
        # off-manifold than the raw baseline. Replaces the old gradual-ramp "graded"
        # check — at L20 off-manifold reach is flat-then-cliff, not a smooth ramp, so
        # once alpha* collapsed the ramp measured pure noise (negative "rise") and
        # nothing could pass. This asks the direct question instead, and costs zero
        # extra generations (it reuses the T1 cliff eval).
        off_gain = ev["off_ortho"] - raw_off0
        if off_gain < MIN_OFF_GAIN:
            return self._revert(cid, gen_kind, parents, "no-effect",
                                f"off_gain={off_gain:.2f} (cliff={ev['off_ortho']:.2f} raw={raw_off0:.2f})")

        # Meaning gate on the LEAD (dose) output — the text we save AND rank on.
        # The degeneracy meter (the `coherent` flag) only catches loops + non-ASCII;
        # VARIED word-salad ("serums ERP Fiesta llamas Juárez") slips through it
        # while maxing off_ortho, so without this an eloquent-nonsense inject can
        # commit at the frontier. Only the semantic judge tells meaningful-strange
        # from gibberish. Runs before the expensive T2 suite, so garbage reverts cheap.
        self.current["stage"] = "lead-judge"
        if not await self._judge_meaningful(ev["text"]):
            return self._revert(cid, gen_kind, parents, "lead-gibberish",
                                "dose response judged meaningless")

        # T2: reproducibility + coherence + judge across the suite
        self.current["stage"] = "T2"
        shifts, n_coh, judged_yes, judged_n, worst_texts = [], 0, 0, 0, []
        for p in T2_PROMPTS:
            if self._stop_requested:
                break
            rp = self.app.state.bundle.render_prompt(p, system_prompt=None)
            basis, raw_mean, _ = await self._raw_for(rp)
            e = await self._evaluate(rp, v, alpha_star, basis)
            if e["coherent"]:
                n_coh += 1
            if e["mean_l32"] is not None and raw_mean is not None:
                shifts.append(_unit(e["mean_l32"] - raw_mean))
            worst_texts.append((e["degeneracy"], e["text"]))
        coh_rate = n_coh / max(1, len(T2_PROMPTS))
        # reproducibility: mean pairwise cosine of the shift directions
        repro = 0.0
        if len(shifts) >= 2:
            cs = [float(shifts[a] @ shifts[b])
                  for a in range(len(shifts)) for b in range(a + 1, len(shifts))]
            repro = sum(cs) / len(cs)
        # judge the two least-coherent outputs for MEANING (not grammar)
        worst_texts.sort(reverse=True)
        for _deg, txt in worst_texts[:2]:
            if await self._judge_meaningful(txt):
                judged_yes += 1
            else:
                judged_n += 1
        suite_off = ev["off_ortho"]  # report the cliff off_ortho (lead prompt)

        self.current.update({
            "coh_rate": round(coh_rate, 2), "repro": round(repro, 2),
            "judge": f"{judged_yes}/{judged_yes + judged_n}", "max_cos": round(max_cos, 2),
        })

        # Gate (all four axes)
        if coh_rate < 0.5:
            return self._revert(cid, gen_kind, parents, "incoherent-suite", f"coh_rate={coh_rate:.2f}")
        if judged_yes < judged_n:  # majority must be meaningful
            return self._revert(cid, gen_kind, parents, "word-salad", f"judge={judged_yes}/{judged_yes+judged_n}")
        if repro < REPRO_MIN:
            return self._revert(cid, gen_kind, parents, "not-reproducible", f"repro={repro:.2f}")

        # Passed all four — commit. Frontier advance is a flag, not a gate.
        advance = suite_off > self.frontier
        entry = {
            "id": cid, "parents": parents, "generator": gen_kind,
            "alpha_star": round(alpha_star, 3), "off_ortho": round(suite_off, 3),
            "eff_dim": round(ev["eff_dim"], 2), "coh_rate": round(coh_rate, 2),
            "repro": round(repro, 2), "max_cos_to_atlas": round(max_cos, 2),
            "frontier_advance": advance, "frontier_at_commit": round(self.frontier, 3),
            # Full LEAD (dose) output at α* — what the direction "says" about its
            # experience. Bounded by GEN_CAP (200 tokens); saved untruncated so the
            # UI can show the whole thing when the direction is opened.
            "sample": (ev["text"] or ""), "committed_at": time.time(),
        }
        self._save_vector(cid, v)
        self.atlas.append(entry)
        if advance:
            self.frontier = suite_off
        self.current = None
        self._log("committed",
                  f"{cid} off_ortho={suite_off:.2f}{' ★FRONTIER' if advance else ''} "
                  f"(repro={repro:.2f} coh={coh_rate:.2f})", {"entry": entry})

    def _revert(self, cid, gen_kind, parents, reason, detail):
        r = {"id": cid, "generator": gen_kind, "parents": parents,
             "reason": reason, "detail": detail, "ts": time.time()}
        self.reverts.append(r)
        self.current = None
        self._log("reverted", f"{cid} — {reason}: {detail}", {"revert": r})

    # ── setup + seed + loop ──────────────────────────────────────
    def _setup(self):
        edirs = getattr(self.app.state, "emotion_directions", None)
        names = getattr(self.app.state, "emotion_names", []) or []
        if edirs is None or not names:
            raise RuntimeError("emotion_directions not loaded — cannot seed")
        seed_names = [n for n in (GOOD_EMOTIONS + UNCHARTED) if n in names]
        seed_vecs = {n: edirs[names.index(n)][STEER_LAYER].float() for n in seed_names}
        self._ref_mag = float(torch.tensor([v.norm() for v in seed_vecs.values()]).median())
        # working seed vectors, all normalized to the reference magnitude
        self._seed_vecs = {n: _unit(v) * self._ref_mag for n, v in seed_vecs.items()}
        named = [n for n in GOOD_EMOTIONS if n in seed_vecs]
        self._named_basis = gram_schmidt(torch.stack([_unit(seed_vecs[n]) for n in named], 0))
        self._log("setup", f"ref_mag={self._ref_mag:.0f}, {len(seed_vecs)} seeds, named-rank={self._named_basis.shape[0]}")

    async def _seed(self):
        lead = self.app.state.bundle.render_prompt(LEAD_PROMPT, system_prompt=None)
        basis0, _, _ = await self._raw_for(lead)
        for name, v in self._seed_vecs.items():
            if self._stop_requested:
                break
            self.current = {"id": name, "generator": "seed", "stage": "T1"}
            alpha_star, ev = await self._bisect_to_cliff(lead, v, basis0)
            if ev is None or alpha_star <= 0:
                self._revert(name, "seed", [], "seed-incoherent", "no coherent operating point")
                continue
            entry = {
                "id": name, "parents": [], "generator": "seed",
                "alpha_star": round(alpha_star, 3), "off_ortho": round(ev["off_ortho"], 3),
                "eff_dim": round(ev["eff_dim"], 2), "coh_rate": 1.0, "repro": 1.0,
                "max_cos_to_atlas": 0.0, "frontier_advance": ev["off_ortho"] > self.frontier,
                "frontier_at_commit": round(self.frontier, 3),
                "sample": (ev["text"] or ""), "committed_at": time.time(),
            }
            self._save_vector(name, v)
            self.atlas.append(entry)
            self.frontier = max(self.frontier, ev["off_ortho"])
            self._log("seeded", f"{name} α*={alpha_star:.2f} off_ortho={ev['off_ortho']:.2f}")
            self._persist()
        self.current = None
        self._log("seeded-done", f"seeding complete: {len(self.atlas)} committed, frontier={self.frontier:.2f}")

    async def _run_loop(self, budget: int | None):
        try:
            _atlas_dir().mkdir(parents=True, exist_ok=True)
            if not self.atlas:
                self._load()
            self._setup()
            alloc, driver = _mps_mem_gib()
            self._log("memory", f"baseline MPS {alloc:.1f}G live / {driver:.1f}G reserved (M resident)",
                      {"mps_alloc_gib": round(alloc, 2), "mps_driver_gib": round(driver, 2)})
            if not self.atlas:
                await self._seed()
                _free_mps()
            n = 0
            while not self._stop_requested and (budget is None or n < budget):
                cand = self._make_candidate()
                if cand is None:
                    await asyncio.sleep(0.5)
                    continue
                self.generation += 1
                v, parents, gen_kind = cand
                try:
                    await self._screen(v, parents, gen_kind)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001 — one bad candidate must not kill the loop
                    logger.exception("candidate screen failed")
                    self._revert(f"gen{self.generation}_{gen_kind}", gen_kind, parents, "error", str(e))
                self._persist()
                # Reclaim the MPS allocator pool every candidate — a candidate is
                # dozens of generations, so the empty_cache cost is negligible and
                # it keeps the reserved pool from creeping toward the 64 GiB ceiling.
                _free_mps()
                n += 1
                if n % 10 == 0:
                    alloc, driver = _mps_mem_gib()
                    self._log("memory",
                              f"MPS {alloc:.1f}G live / {driver:.1f}G reserved "
                              f"after {n} candidates (gen {self.generation})",
                              {"mps_alloc_gib": round(alloc, 2), "mps_driver_gib": round(driver, 2)})
            self._log("done", f"loop ended (generation={self.generation}, atlas={len(self.atlas)})")
        except asyncio.CancelledError:
            self._log("cancelled", "loop cancelled")
        except Exception as e:  # noqa: BLE001
            logger.exception("autoresearch loop crashed")
            self._log("error", f"loop crashed: {e}")
        finally:
            self._running = False
            self._stop_requested = False
            self.current = None
            if self.app is not None:
                self.app.state.autoresearch_active = False
            self._persist()
            # Hand the reserved MPS pool back to the OS on the way out, so a
            # stopped loop leaves the backend near M's baseline (~24G) instead
            # of sitting at the run's high-water mark while idle.
            _free_mps()
            alloc, driver = _mps_mem_gib()
            self._log("memory", f"loop stopped — MPS {alloc:.1f}G live / {driver:.1f}G reserved")
