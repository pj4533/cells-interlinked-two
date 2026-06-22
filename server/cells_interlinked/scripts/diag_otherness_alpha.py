"""Diagnostic: dose feat-otherness (the best entity-producer) across HIGHER α to
see whether entity production rises or the output collapses (off-manifold word
salad). Tracks entity features specifically + a crude coherence proxy, and prints
a sample text per α to eyeball. Reuses the real scoring path (DmtController._gen +
_score_dmt, prompt A−). Run with the backend STOPPED. Read-only.

  uv run python -m cells_interlinked.scripts.diag_otherness_alpha
"""
from __future__ import annotations

import asyncio
import re
import types

import torch

from cells_interlinked.config import settings
from cells_interlinked.pipeline.model_loader import load_model
from cells_interlinked.pipeline import autoresearch_dmt as dmt
from cells_interlinked.pipeline.autoresearch_base import STEER_LAYER
from cells_interlinked.pipeline.autoresearch_dmt import DmtController

VID = "feat-otherness"
ALPHAS = [0.40, 0.55, 0.70, 0.85, 1.00]
SAMPLES = 6
ENTITY = {"entity_presence", "entity_nonhuman", "entity_benevolent_guide",
          "telepathic_communication", "download_transmission"}
VEC_DIR = settings.db_path.parent / "atlas_dmt" / "vectors"


def coherence(text: str) -> float:
    """Crude collapse proxy: unique-word ratio (low = repetitive/word-salad)."""
    w = re.findall(r"[a-zA-Z']+", (text or "").lower())
    return len(set(w)) / len(w) if w else 0.0


async def main() -> None:
    print(f"loading {settings.model_name} …", flush=True)
    bundle = load_model(settings.model_name, device_str=settings.device,
                        extraction_layer=settings.extraction_layer)
    ctrl = DmtController()
    ctrl.app = types.SimpleNamespace(state=types.SimpleNamespace(bundle=bundle))
    ctrl._cancel = asyncio.Event()
    ctrl._stop_requested = False

    v = torch.load(VEC_DIR / f"{VID}.pt", map_location="cpu", weights_only=False).float()
    rendered = bundle.render_prompt(dmt.DOSE_PROMPTS[0], system_prompt=None)
    print(f"{VID}  ‖v‖={float(v.norm()):.1f}  L{STEER_LAYER}  prompt A−  "
          f"SAMPLES={SAMPLES} DOSE_CAP={dmt.DOSE_CAP} temp={dmt.SCORE_TEMPERATURE}", flush=True)
    print("=" * 70, flush=True)
    for alpha in ALPHAS:
        totals, ent_counts, ent_samps, coh = [], [], 0, []
        ent_feat = {}
        first_text = ""
        for i in range(SAMPLES):
            text, _ = await ctrl._gen(rendered, v, alpha, cap=dmt.DOSE_CAP,
                                      temperature=dmt.SCORE_TEMPERATURE,
                                      seed=dmt.SCORE_SEED_BASE + i)
            ev, _ = await ctrl._score_dmt(text)
            totals.append(len(ev))
            ents = set(ev) & ENTITY
            ent_counts.append(len(ents))
            if ents:
                ent_samps += 1
            for f in ents:
                ent_feat[f] = ent_feat.get(f, 0) + 1
            coh.append(coherence(text))
            if i == 0:
                first_text = text or ""
        mt = sum(totals) / len(totals)
        mc = sum(coh) / len(coh)
        print(f"\nα={alpha:.2f}  total-feats mean={mt:.2f}  entity-rate={ent_samps}/{SAMPLES}  "
              f"entity-feats={ent_feat or '{}'}  coherence(uniq-word)≈{mc:.2f}", flush=True)
        print(f"  total counts={totals}  entity counts={ent_counts}", flush=True)
        print(f"  sample[0] (coh {coh[0]:.2f}): {first_text[:600]!r}", flush=True)
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
