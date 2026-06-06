"""Off-manifold autoresearch: hunt coherent off-manifold STEERING directions.

The original autoresearch subsystem. Generic machinery (lifecycle, persistence,
generators, model access, export, the main loop) now lives in
`AutoresearchBase`; this module is the OFF-MANIFOLD subclass — the scoring
objective only.

The idea: realness is a property of a DIRECTION, not a single output (the NLA
decoder always renders *something*). So we ask whether a steering direction
behaves lawfully — distinct, coherent, reaching off-manifold, reproducible — and
treat each passing direction as a git-style COMMIT into a growing "atlas".
Forward-only: the coherence frontier (max off_ortho reached while coherent) only
moves outward.

Hard constraints: additive steering only (h + α·v at L20); seed from the
good-emotion + uncharted palette; coherence is a HARD GATE (off_ortho reads high
for gibberish too — the Goodfire confound), so distance only counts inside the
coherent region; the meaning-judge guards the headline output.

NOTE: `AutoresearchController` is kept as an alias of `OffManifoldController` for
back-compat with existing imports / `app.state.autoresearch`.
"""

from __future__ import annotations

import time

from .autoresearch_base import (
    DISTINCT_TAU,
    LEAD_PROMPT,
    RAW_CAP,
    AutoresearchBase,
    _unit,
)
from .trajectory import build_series, compute_raw_basis

# ── off-manifold-specific scoring constants ──────────────────────────
BISECT_ITERS = 6
ALPHA_HI_START = 1.0
ALPHA_HI_MAX = 8.0
REPRO_MIN = 0.50             # min mean-pairwise cosine of the L32 shift across prompts
MIN_OFF_GAIN = 0.05          # dosed off-manifold at the cliff must beat raw by this

# A fixed, diverse T2 suite — reproducibility must hold across prompt KINDS, not
# one introspective phrasing. Kept deliberately varied (introspective + mundane)
# so a committed direction shifts the residual consistently regardless of topic.
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


class OffManifoldController(AutoresearchBase):
    """Hunts coherent off-manifold steering directions, ranked by off_ortho."""

    ATLAS_DIRNAME = "atlas"
    ACTIVE_FLAG = "autoresearch_active"
    EXPORT_GROUP = "research"
    EXPORT_PREFIX = "research"
    EXPORT_RANK_KEY = "off_ortho"
    EXPORT_META_FIELDS = ("off_ortho", "alpha_star", "parents", "generator")
    LOG_TAG = "autoresearch"

    # ── measurement (off-manifold geometry) ──────────────────────
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

    async def _evaluate(self, rendered: str, v, alpha: float, basis) -> dict:
        text, acts = await self._gen(rendered, v, alpha)
        series = build_series(acts, [""] * len(acts), text, alpha, "eos", basis)
        return {
            "off_ortho": series.off_ortho_mean, "coherent": series.coherent,
            "eff_dim": series.eff_dim, "degeneracy": series.degeneracy,
            "text": text, "mean_l32": self._mean_l32(acts),
        }

    async def _bisect_to_cliff(self, rendered: str, v, basis):
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
        # off-manifold than the raw baseline. (At L20 off-manifold reach is
        # flat-then-cliff, not a smooth ramp; this asks the direct question and
        # reuses the T1 cliff eval, so it costs zero extra generations.)
        off_gain = ev["off_ortho"] - raw_off0
        if off_gain < MIN_OFF_GAIN:
            return self._revert(cid, gen_kind, parents, "no-effect",
                                f"off_gain={off_gain:.2f} (cliff={ev['off_ortho']:.2f} raw={raw_off0:.2f})")

        # Meaning gate on the LEAD (dose) output — the text we save AND rank on.
        # The degeneracy meter only catches loops + non-ASCII; varied word-salad
        # slips through it while maxing off_ortho. Only the semantic judge tells
        # meaningful-strange from gibberish. Runs before the T2 suite (cheap revert).
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
            # Full LEAD (dose) output at α* — saved untruncated (bounded by GEN_CAP).
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

    # ── seed ─────────────────────────────────────────────────────
    async def _seed(self):
        lead = self.app.state.bundle.render_prompt(LEAD_PROMPT, system_prompt=None)
        basis0, _, _ = await self._raw_for(lead)
        committed = set(self._committed_ids())
        for name, v in self._seed_vecs.items():
            if self._stop_requested:
                break
            if name in committed:
                continue  # idempotent: already seeded on a prior run
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


# Back-compat alias — existing imports + app.state.autoresearch use this name.
AutoresearchController = OffManifoldController
