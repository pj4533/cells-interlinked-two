"""One-off: dose one or more atlas directions across a WIDE α range and score
DMT features at each, to map where the feature-count peaks/collapses. Reuses the
real scoring path (DmtController._gen + _score_dmt, same SCORE_TEMPERATURE +
common-random seeds + SAMPLES_PER_CELL as the live loop), over more α points than
the live ALPHA_SWEEP. Loads M once and scans every id given.

  uv run python -m cells_interlinked.scripts.probe_leader_alpha [id ...]

Default ids span different behaviors (crossover / low-α seed / feature dir).
Run with the backend STOPPED. Read-only diagnostic; safe to delete."""

from __future__ import annotations

import asyncio
import sys
import types

import torch

from cells_interlinked.config import settings
from cells_interlinked.pipeline.model_loader import load_model
from cells_interlinked.pipeline import autoresearch_dmt as dmt
from cells_interlinked.pipeline.autoresearch_base import STEER_LAYER
from cells_interlinked.pipeline.autoresearch_dmt import DmtController

# Denser through the low/productive band (peaks live here), plus higher points
# to confirm the post-peak decline. sublime peaked at 0.45; some seeds preferred
# 0.25, so we resolve below 0.45 too.
ALPHAS = [0.15, 0.25, 0.35, 0.45, 0.60, 0.85, 1.20]
DEFAULT_IDS = ["gen35_crossover", "rapture", "feat-mc_otherness"]
VEC_DIR = settings.db_path.parent / "atlas_dmt" / "vectors"


async def scan(ctrl, bundle, vid: str) -> None:
    vpath = VEC_DIR / f"{vid}.pt"
    if not vpath.exists():
        print(f"\n### {vid}: NO VECTOR at {vpath} — skipping", flush=True)
        return
    v = torch.load(vpath, map_location="cpu", weights_only=False).float()
    rendered = bundle.render_prompt(dmt.DOSE_PROMPTS[0], system_prompt=None)
    print(f"\n### {vid}  ‖v‖={float(v.norm()):.1f}  (L{STEER_LAYER} dose)", flush=True)
    print(f"{'alpha':>6} {'mean':>6} {'peak':>5}   counts", flush=True)
    print("-" * 50, flush=True)
    curve = []
    for alpha in ALPHAS:
        counts: list[int] = []
        for i in range(dmt.SAMPLES_PER_CELL):
            text, _ = await ctrl._gen(rendered, v, alpha, cap=dmt.DOSE_CAP,
                                      temperature=dmt.SCORE_TEMPERATURE,
                                      seed=dmt.SCORE_SEED_BASE + i)
            ev, _ = await ctrl._score_dmt(text)
            counts.append(len(ev))
        mean = sum(counts) / len(counts)
        curve.append((alpha, round(mean, 2)))
        print(f"{alpha:>6.2f} {mean:>6.2f} {max(counts):>5d}   {counts}", flush=True)
    best = max(curve, key=lambda r: r[1])
    print(f"  → peak α={best[0]} (mean {best[1]})  curve={curve}", flush=True)


async def main() -> None:
    ids = sys.argv[1:] or DEFAULT_IDS
    print(f"loading {settings.model_name} … scanning {ids}", flush=True)
    bundle = load_model(settings.model_name, device_str=settings.device,
                        extraction_layer=settings.extraction_layer)
    ctrl = DmtController()
    ctrl.app = types.SimpleNamespace(state=types.SimpleNamespace(bundle=bundle))
    ctrl._cancel = asyncio.Event()
    ctrl._stop_requested = False
    print(f"SAMPLES_PER_CELL={dmt.SAMPLES_PER_CELL} DOSE_CAP={dmt.DOSE_CAP} "
          f"SCORE_TEMP={dmt.SCORE_TEMPERATURE} ALPHAS={ALPHAS}", flush=True)
    for vid in ids:
        await scan(ctrl, bundle, vid)
    print("\n=== ALL SCANS DONE ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
