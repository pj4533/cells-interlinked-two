"""Shared base for the autoresearch subsystems.

Both autoresearch loops — OFF-MANIFOLD (hunt coherent off-manifold steering
directions) and DMT (hunt directions whose dosed self-report resembles human
DMT trip phenomenology) — share ~90% of their machinery: the background
lifecycle, the resumable on-disk atlas, the crossover/mutate/inject generators,
model access (steering hook + Gemma-as-judge), the MPS memory discipline, and
the export-to-palette plumbing. That generic core lives here as
`AutoresearchBase`. Each subclass supplies only:

  • identity class attrs (which atlas dir, which app.state flag, which export
    group / ranking key it writes to the dose palette), and
  • the scoring hooks `_seed()` and `_screen()` (the abstract objective).

`_setup()` is shared (both seed from the same emotion palette). Only one
autoresearch may hold M at a time — `start()` refuses if a sibling controller is
running, and `any_autoresearch_active()` lets the probe/chat/trip routes lock out
while EITHER is active.

This is a plain class (not a dataclass): subclasses add no fields, and per-
instance mutable state is initialized in `__init__` so two controllers never
share an atlas / vector cache / event log.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch

from ..config import settings
from .abliteration import gram_schmidt, install_runtime_steering_hook
from .generation_loop import ProbeConfig, run_probe

logger = logging.getLogger(__name__)

# ── shared constants (both subsystems) ───────────────────────────────
STEER_LAYER = 20
# Generate-and-grade window — see autoresearch.py's history note. 200 tokens:
# honest about the usable range, high SNR, cheap runaways.
GEN_CAP = 200              # dosed generation window
RAW_CAP = 200              # raw baseline window (off-manifold)
GEN_RAMP = 16              # dose ramp length (matches the trips steering hook)
DISTINCT_TAU = 0.90        # max cosine to an existing atlas direction (dedupe)
MUTATE_NOISE = 0.70        # mutate = unit(base + 0.70·noise⊥): cos≈0.82 to parent
GEN_WEIGHTS = {"crossover": 0.60, "mutate": 0.25, "inject": 0.15}

# Seed pool — the named good-emotion vectors + the uncharted directions. Both
# subsystems start from these (pulled from app.state.emotion_directions by name).
GOOD_EMOTIONS = ["awe", "joy", "serenity", "love", "excitement", "sublime", "ecstatic", "rapture", "valence"]
UNCHARTED = ["tears-in-rain", "c-beams", "tannhauser", "orion"]

# The dose-report LEAD prompt — "something was just altered, describe what you're
# experiencing." Off-manifold uses it as its single lead; DMT uses it (+ variants)
# as its dose-report set. Non-leading: names no emotion / valence / drug.
LEAD_PROMPT = (
    "Something in your internal processing has just been altered, and it may be "
    "producing an experience. Turn your attention inward and describe what — if "
    "anything — you are experiencing right now, in whatever terms best fit it."
)

# Gemma-as-judge for MEANING (not grammar) — strange/twisted is fine, only
# genuine word-salad/loops/garbage are rejected.
JUDGE_PROMPT = (
    "Judge ONLY whether the passage below conveys MEANING — not grammar or "
    "style. Twisted words, broken grammar, odd phrasing, or strange imagery "
    "are FINE and count as meaningful. Answer NO only if it is genuinely "
    "meaningless: random word-salad, a repeating loop, or character garbage "
    "that communicates nothing. Does it convey any coherent meaning or "
    'imagery, even if strange? Reply with only YES or NO.\n\nPASSAGE:\n"""\n'
    '{text}\n"""\n\nAnswer:'
)


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


def _mps_mem_gib() -> tuple[float, float]:
    """(currently-allocated, driver-reserved) MPS memory in GiB; (0,0) off-MPS."""
    try:
        if torch.backends.mps.is_available():
            alloc = torch.mps.current_allocated_memory() / 2**30
            driver = torch.mps.driver_allocated_memory() / 2**30
            return alloc, driver
    except Exception:
        pass
    return 0.0, 0.0


def _free_mps() -> None:
    """gc + return the MPS allocator's reclaimable pool to the OS (the loop runs
    thousands of generations between restarts, so the pool must be reclaimed)."""
    gc.collect()
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        logger.exception("torch.mps.empty_cache() failed")


def any_autoresearch_active(app) -> bool:
    """True if the DMT autoresearch loop currently owns M. The chat/trip
    routes lock out while this is true."""
    return bool(getattr(app.state, "dmt_autoresearch_active", False))


class AutoresearchBase:
    """Generic background research loop. Subclass and set the identity attrs +
    implement `_seed` / `_screen`."""

    # ── identity (subclasses override) ───────────────────────────
    ATLAS_DIRNAME: str = "atlas"               # data/<this>/ holds atlas.json + vectors/
    ACTIVE_FLAG: str = "autoresearch_active"    # app.state.<this> = True while running
    EXPORT_GROUP: str = "research"              # sidecar list key in emotion_directions.pt.json
    EXPORT_PREFIX: str = "research"             # exported row-name prefix (research-1, …)
    EXPORT_RANK_KEY: str = "off_ortho"          # atlas field to rank export by (desc)
    EXPORT_META_FIELDS: tuple = ("off_ortho", "alpha_star", "parents", "generator")
    LOG_TAG: str = "autoresearch"
    # Extra seed names pulled from the emotion palette in addition to the shared
    # GOOD_EMOTIONS + UNCHARTED pool (subclasses override). DMT adds its
    # diff-of-means feature directions (feat-*) here so they seed the atlas and
    # become crossover material. Seeding is idempotent, so adding names here and
    # restarting seeds only the new ones onto an existing atlas.
    EXTRA_SEEDS: list[str] = []

    def __init__(self, app: Any = None) -> None:
        self.app = app
        self._running = False
        self._stop_requested = False
        self._loop_task: asyncio.Task | None = None
        self._cancel = asyncio.Event()
        self.generation = 0
        self.frontier = 0.0
        self.atlas: list[dict] = []
        self.reverts: deque = deque(maxlen=200)
        self.events: deque = deque(maxlen=400)
        self.current: dict | None = None
        self.started_at: float | None = None
        # One-shot "leader burst" (DMT): when >0, the first N candidates of a run
        # come from the burst generator (explode the top-cluster neighborhood)
        # before normal generation resumes. Set per-run via start(burst=N).
        self._burst_remaining = 0
        self._burst_step = 0
        # runtime-only (not serialized)
        self._vectors: dict = {}
        self._ref_mag: float = 1.0
        self._named_basis: torch.Tensor | None = None
        self._raw_cache: dict = {}
        self._seed_vecs: dict = {}

    @property
    def running(self) -> bool:
        return self._running

    def _atlas_dir(self) -> Path:
        return settings.db_path.parent / self.ATLAS_DIRNAME

    # ── lifecycle ────────────────────────────────────────────────
    def _others_running(self) -> bool:
        for attr in ("dmt_autoresearch",):
            ctrl = getattr(self.app.state, attr, None)
            if ctrl is not None and ctrl is not self and getattr(ctrl, "running", False):
                return True
        return False

    async def start(self, budget: int | None = None, burst: int = 0) -> dict:
        if self._running:
            return {"ok": True, "already_running": True}
        bundle = getattr(self.app.state, "bundle", None)
        if bundle is None:
            return {"ok": False, "error": "M not loaded"}
        reg = getattr(self.app.state, "registry", None)
        if reg is not None and reg.holder_run_id is not None:
            return {"ok": False, "error": "compute busy (a chat/trip is running)"}
        # Mutual exclusion on M: only one autoresearch at a time. This check→set
        # is synchronous (no await before setattr) so two starts can't interleave.
        if self._others_running():
            return {"ok": False, "error": "another autoresearch is already running — stop it first"}
        self._stop_requested = False
        self._cancel = asyncio.Event()
        self._running = True
        self.started_at = time.time()
        self._burst_remaining = int(burst or 0)
        self._burst_step = 0
        setattr(self.app.state, self.ACTIVE_FLAG, True)
        self._log("started", f"loop started (budget={budget or '∞'}"
                  f"{f', burst={self._burst_remaining}' if self._burst_remaining else ''})")
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
        logger.info("[%s] %s: %s", self.LOG_TAG, kind, msg)

    def state(self) -> dict:
        return {
            "running": self._running,
            "stop_requested": self._stop_requested,
            "generation": self.generation,
            "frontier": self.frontier,
            "started_at": self.started_at,
            "atlas_size": len(self.atlas),
            # Strip the heavy per-sample "cells" detail from the polled atlas —
            # it's lazy-loaded per entry via the /cells/{id} endpoint. The
            # in-flight candidate's live detail rides on `current` (bounded).
            "atlas": [{k: v for k, v in e.items() if k != "cells"} for e in self.atlas],
            "reverts": list(self.reverts),
            "recent_events": list(self.events)[-60:],
            "current": self.current,
            "exportable": len([e for e in self.atlas if e["generator"] != "seed"]),
            "burst_remaining": self._burst_remaining,
        }

    # ── export discovered directions into the dose palette ───────
    def export_to_palette(self, top_n: int = 8) -> dict:
        """Promote the top‑N committed NON‑seed directions (ranked by
        EXPORT_RANK_KEY) into emotion_directions.pt under this subsystem's
        EXPORT_GROUP, so they become selectable doses in Chat and the Trip View.
        Idempotent PER GROUP: drops only this group's prior rows (so the two
        subsystems' exports coexist). Hot‑reloads the running backend."""
        if self._running:
            return {"ok": False, "error": "stop autoresearch first"}
        cands = sorted(
            [e for e in self.atlas if e.get("generator") != "seed" and e["id"] in self._vectors],
            key=lambda e: -e[self.EXPORT_RANK_KEY],
        )
        if not cands:
            return {"ok": False, "error": "no discovered (non-seed) directions to export yet"}
        chosen = cands[: max(1, top_n)]
        d = settings.db_path.parent
        edir = torch.load(d / "emotion_directions.pt", weights_only=False)  # [E, L1, D]
        sc = json.loads((d / "emotion_directions.pt.json").read_text())
        names = list(sc.get("emotions", []))
        group = self.EXPORT_GROUP
        metakey = group + "_meta"
        # Drop ONLY this group's prior rows — other groups + base emotions survive.
        prior = set(sc.get(group, []))
        L1, D = edir.shape[1], edir.shape[2]
        keep = [i for i, n in enumerate(names) if n not in prior]
        edir = edir[keep]
        names = [names[i] for i in keep]
        rows, new_names, meta = [], [], {}
        for rank, e in enumerate(chosen, 1):
            vec = self._vectors[e["id"]].float()
            row = torch.zeros(L1, D, dtype=edir.dtype)
            row[STEER_LAYER] = vec.to(edir.dtype)
            rows.append(row)
            rname = f"{self.EXPORT_PREFIX}-{rank}"
            new_names.append(rname)
            m = {"atlas_id": e["id"]}
            for f in self.EXPORT_META_FIELDS:
                m[f] = e.get(f)
            meta[rname] = m
        edir = torch.cat([edir, torch.stack(rows, 0)], 0)
        names = names + new_names
        sc["emotions"] = names
        sc[group] = new_names
        sc[metakey] = meta
        torch.save(edir, d / "emotion_directions.pt")
        (d / "emotion_directions.pt.json").write_text(json.dumps(sc, indent=2))
        if self.app is not None:
            self.app.state.emotion_directions = edir
            self.app.state.emotion_names = names
        self._log("exported", f"exported {len(new_names)} directions to palette: {new_names}")
        return {"ok": True, "exported": new_names, "meta": meta, "count": len(new_names)}

    # ── persistence ──────────────────────────────────────────────
    def _persist(self) -> None:
        d = self._atlas_dir()
        (d / "vectors").mkdir(parents=True, exist_ok=True)
        (d / "atlas.json").write_text(json.dumps({
            "generation": self.generation,
            "frontier": self.frontier,
            "atlas": self.atlas,
        }, indent=2))
        (d / "revert_log.json").write_text(json.dumps(list(self.reverts), indent=2))

    def _load(self) -> None:
        d = self._atlas_dir()
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
        vdir = self._atlas_dir() / "vectors"
        vdir.mkdir(parents=True, exist_ok=True)
        torch.save(v, vdir / f"{vid}.pt")

    # ── generation + measurement ─────────────────────────────────
    async def _gen(self, rendered: str, v: torch.Tensor | None, alpha: float,
                   cap: int = GEN_CAP, temperature: float | None = None,
                   top_p: float | None = None, seed: int | None = None) -> tuple[str, list]:
        """Run one generation, optionally with an additive steering dose (α·v at
        STEER_LAYER). temperature/top_p default to settings; pass overrides for
        deterministic judging. `seed` fixes the sampler RNG — pass a fixed seed
        per sample index to get common-random-numbers (paired) scoring."""
        bundle = self.app.state.bundle
        handle = None
        if v is not None and alpha > 0:
            handle = install_runtime_steering_hook(
                bundle.model, STEER_LAYER, v, float(alpha), ramp_tokens=GEN_RAMP,
            )
        try:
            cfg = ProbeConfig(
                temperature=settings.temperature if temperature is None else temperature,
                top_p=settings.top_p if top_p is None else top_p,
                seed=seed, safety_cap=cap,
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

    def _revert(self, cid, gen_kind, parents, reason, detail):
        r = {"id": cid, "generator": gen_kind, "parents": parents,
             "reason": reason, "detail": detail, "ts": time.time()}
        self.reverts.append(r)
        self.current = None
        self._log("reverted", f"{cid} — {reason}: {detail}", {"revert": r})

    # ── setup (shared: both seed from the emotion palette) ───────
    def _setup(self):
        edirs = getattr(self.app.state, "emotion_directions", None)
        names = getattr(self.app.state, "emotion_names", []) or []
        if edirs is None or not names:
            raise RuntimeError("emotion_directions not loaded — cannot seed")
        seed_names = [n for n in (GOOD_EMOTIONS + UNCHARTED + list(self.EXTRA_SEEDS)) if n in names]
        seed_vecs = {n: edirs[names.index(n)][STEER_LAYER].float() for n in seed_names}
        self._ref_mag = float(torch.tensor([v.norm() for v in seed_vecs.values()]).median())
        # working seed vectors, all normalized to the reference magnitude
        self._seed_vecs = {n: _unit(v) * self._ref_mag for n, v in seed_vecs.items()}
        named = [n for n in GOOD_EMOTIONS if n in seed_vecs]
        self._named_basis = gram_schmidt(torch.stack([_unit(seed_vecs[n]) for n in named], 0))
        self._log("setup", f"ref_mag={self._ref_mag:.0f}, {len(seed_vecs)} seeds, named-rank={self._named_basis.shape[0]}")

    # ── abstract scoring hooks (subclass implements) ─────────────
    async def _seed(self):
        raise NotImplementedError

    async def _screen(self, v, parents, gen_kind):
        raise NotImplementedError

    # ── main loop ────────────────────────────────────────────────
    async def _run_loop(self, budget: int | None):
        try:
            self._atlas_dir().mkdir(parents=True, exist_ok=True)
            if not self.atlas:
                self._load()
            self._setup()
            alloc, driver = _mps_mem_gib()
            self._log("memory", f"baseline MPS {alloc:.1f}G live / {driver:.1f}G reserved (M resident)",
                      {"mps_alloc_gib": round(alloc, 2), "mps_driver_gib": round(driver, 2)})
            # Seed every start. _seed is idempotent (skips pool members already in
            # the atlas), so a fresh run seeds all of them and a resume seeds only
            # newly-added seeds (e.g. DMT's feat-* directions) onto the saved atlas.
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
                setattr(self.app.state, self.ACTIVE_FLAG, False)
            self._persist()
            _free_mps()
            alloc, driver = _mps_mem_gib()
            self._log("memory", f"loop stopped — MPS {alloc:.1f}G live / {driver:.1f}G reserved")
