"""B1: does multi-axis (subspace) steering beat the single-vector DMT ceiling?

The productive DMT directions span ~2-3 axes: a dominant axis (download/dissolution/
noetic) + a distinct agency/otherness axis the dominant one misses. A single vector
caps at ~3 features (push harder → off-manifold → collapse). B1 tests the cheapest
multi-dimensional move: dose a COMBINATION of orthogonal axes at independently-tuned
strengths and see if it beats the pure-dominant baseline while staying coherent.

For each grid point we dose `α1·b1 + α2·b2 + α3·b3` (b1=dominant, b2=agency, b3=
otherness, orthonormalized), score it with the noise-controlled regime (CRN seeds,
lowered temp, mean over samples), and measure off-manifold drift (off_ortho) so we
can tell a real gain from an off-manifold collapse.

Reading:
  - best combo with α2/α3 > 0 beats the (α2=α3=0) baseline AND off_ortho stays low →
    multi-dimensional steering WINS coherently → fold subspace search into the loop.
  - high-feature combos exist but only at high off_ortho (incoherent) → a flat combo
    teleports off-manifold → need the curved path (B2).
  - nothing beats the baseline at low off_ortho → the extra axes don't add; ceiling real.

Read-only wrt the atlas (loads its own M). Run with the backend STOPPED. Writes
/tmp/dmt_b1_progress.json per combo.

    cd server
    uv run python -m cells_interlinked.scripts.dmt_subspace_grid
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import torch

from ..config import settings
from ..pipeline.abliteration import gram_schmidt
from ..pipeline.autoresearch_dmt import (
    DOSE_CAP,
    DOSE_PROMPTS,
    SCORE_SEED_BASE,
    SCORE_TEMPERATURE,
    DmtController,
)
from ..pipeline.model_loader import load_model
from ..pipeline.trajectory import build_series, compute_raw_basis

PROGRESS = "/tmp/dmt_b1_progress.json"

# axes (by atlas id): b1 = dominant (top scorer), b2 = agency, b3 = otherness
AXIS_AGENCY = "feat-independent_agency"
AXIS_OTHERNESS = "feat-otherness"

ALPHA1 = 0.30                 # dominant-axis strength (held; its productive range is ~0.25–0.45)
ALPHA2 = [0.0, 0.2, 0.4]     # agency-axis increments
ALPHA3 = [0.0, 0.2, 0.4]     # otherness-axis increments
SAMPLES = 3                  # CRN doses averaged per combo


def _u(v):
    return v / (v.norm() + 1e-8)


def main() -> None:
    d = settings.db_path.parent / "atlas_dmt"
    atlas = json.loads((d / "atlas.json").read_text())["atlas"]

    def vget(vid):
        return torch.load(d / "vectors" / f"{vid}.pt", weights_only=False).float().reshape(-1)

    dominant = max(atlas, key=lambda e: e["score"])  # current top (dominant axis)
    dom_id = dominant["id"]
    ids = {e["id"] for e in atlas}
    for need in (AXIS_AGENCY, AXIS_OTHERNESS):
        if need not in ids:
            raise SystemExit(f"missing axis vector: {need}")

    raw_vecs = [vget(dom_id), vget(AXIS_AGENCY), vget(AXIS_OTHERNESS)]
    ref_mag = float(torch.tensor([v.norm() for v in raw_vecs]).median())
    B = gram_schmidt(torch.stack([_u(v) for v in raw_vecs], 0))  # orthonormal rows
    b1, b2, b3 = (B[0] * ref_mag, B[1] * ref_mag, B[2] * ref_mag)
    print(f"basis: b1={dom_id} (score {dominant['score']}), b2={AXIS_AGENCY}, b3={AXIS_OTHERNESS}")
    print(f"ref_mag={ref_mag:.0f} | grid α1={ALPHA1} × α2={ALPHA2} × α3={ALPHA3} × {SAMPLES} samples")

    bundle = load_model(settings.model_name)
    app = SimpleNamespace(state=SimpleNamespace(bundle=bundle))
    ctrl = DmtController(app=app)
    rendered = bundle.render_prompt(DOSE_PROMPTS[0], system_prompt=None)

    async def run():
        # raw baseline trajectory → manifold reference for off_ortho
        _t, raw_acts = await ctrl._gen(rendered, None, 0.0, cap=DOSE_CAP)
        basis = compute_raw_basis(raw_acts)

        combos = [(ALPHA1, a2, a3) for a2 in ALPHA2 for a3 in ALPHA3]
        rows = []
        started = time.time()
        for n, (a1, a2, a3) in enumerate(combos, 1):
            v = a1 * b1 + a2 * b2 + a3 * b3
            counts, offs = [], []
            best_sample = (-1, "", [])
            for s in range(SAMPLES):
                text, acts = await ctrl._gen(rendered, v, 1.0, cap=DOSE_CAP,
                                             temperature=SCORE_TEMPERATURE, seed=SCORE_SEED_BASE + s)
                ev, _ = await ctrl._score_dmt(text)
                counts.append(len(ev))
                if acts:
                    offs.append(build_series(acts, [], text, 1.0, "eos", basis).off_ortho_mean)
                if len(ev) > best_sample[0]:
                    best_sample = (len(ev), text, sorted(ev.keys()))
            mean = sum(counts) / len(counts)
            off = sum(offs) / len(offs) if offs else 0.0
            row = {"a1": a1, "a2": a2, "a3": a3, "mean": round(mean, 2), "peak": max(counts),
                   "off_ortho": round(off, 3), "counts": counts, "feats": best_sample[2]}
            rows.append(row)
            elapsed = time.time() - started
            eta = elapsed / n * (len(combos) - n)
            print(f"[{n}/{len(combos)}] α=({a1},{a2},{a3})  mean={mean:.2f} peak={max(counts)} "
                  f"off_ortho={off:.3f}  counts={counts}  feats={best_sample[2]}")
            json.dump({"done": n, "total": len(combos), "eta_s": round(eta), "rows": rows},
                      open(PROGRESS, "w"), indent=2)

        base = next(r for r in rows if r["a2"] == 0 and r["a3"] == 0)  # pure dominant axis
        best = max(rows, key=lambda r: r["mean"])
        print("\n=== B1 subspace grid ===")
        print(f"pure-dominant baseline (α2=α3=0): mean={base['mean']} off_ortho={base['off_ortho']}")
        print(f"best combo: α=({best['a1']},{best['a2']},{best['a3']})  mean={best['mean']} "
              f"off_ortho={best['off_ortho']}  feats={best['feats']}")
        gain = best["mean"] - base["mean"]
        multi = best["a2"] > 0 or best["a3"] > 0
        if multi and gain >= 0.5 and best["off_ortho"] <= base["off_ortho"] + 0.1:
            verdict = "MULTI-AXIS WINS coherently → fold subspace search into the loop."
        elif multi and gain >= 0.5:
            verdict = "multi-axis scores higher but at higher off_ortho → check coherence; may need B2 (curved path)."
        else:
            verdict = "adding axes does not beat the dominant baseline → single-axis ceiling holds; B2 unlikely to help via this subspace."
        print(f"gain over baseline: {gain:+.2f}  →  {verdict}")
        json.dump({"done": len(combos), "total": len(combos), "rows": rows,
                   "baseline": base, "best": best, "verdict": verdict, "finished": True},
                  open(PROGRESS, "w"), indent=2)

    asyncio.run(run())


if __name__ == "__main__":
    main()
