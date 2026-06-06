"""Extract DMT feature-direction seeds and append them to the dose palette.

Diff-of-means, same recipe as the emotion palette: for each feature, run vivid
first-person experiential prompts and mundane neutral prompts, capture the L20
residual at the last token of each rendered prompt, and take
mean(experiential) − mean(neutral). The chat-template scaffolding is identical on
both poles so it cancels; the feature direction survives.

Each direction is written into `data/emotion_directions.pt` as a new row named
`feat-<id>` (zeros except at STEER_LAYER=20), scaled to the palette's convention
(0.4 · median‖h‖[L20] · unit(diff)) so the file stays uniform — though the
DMT loop renormalizes seeds to its own ref magnitude, so only the *direction*
matters for seeding. Re-runnable: it drops any existing `feat-*` rows first.

Run with the backend STOPPED (this loads its own M — two M's won't fit on 64 GB):

    cd server
    uv run python -m cells_interlinked.scripts.compute_dmt_feature_seeds
"""

from __future__ import annotations

import json
import logging

import torch

from ..config import settings
from ..pipeline.abliteration import _find_decoder_layers
from ..pipeline.dmt_feature_seeds import (
    FEATURE_SEED_PROMPTS,
    NEUTRAL_PROMPTS,
)
from ..pipeline.model_loader import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compute_dmt_feature_seeds")

STEER_LAYER = 20
DOSE_UNIT = 0.4  # matches the uncharted/alien convention in emotion_directions.pt.json


@torch.no_grad()
def capture_last_token_l20(bundle, rendered: str) -> torch.Tensor:
    """Forward `rendered` once; return the L20 residual at the last position."""
    layer = _find_decoder_layers(bundle.model)[STEER_LAYER]
    cap: dict[str, torch.Tensor] = {}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        cap["v"] = h[0, -1, :].detach().float().cpu()

    handle = layer.register_forward_hook(hook)
    try:
        ids = bundle.raw_tokenizer.encode(rendered).ids
        input_ids = torch.tensor([ids], device=bundle.device)
        bundle.model(input_ids, use_cache=False)
    finally:
        handle.remove()
    return cap["v"]


def mean_l20(bundle, prompts: list[str]) -> tuple[torch.Tensor, list[float]]:
    vecs = []
    for p in prompts:
        rendered = bundle.render_prompt(p, system_prompt=None)
        vecs.append(capture_last_token_l20(bundle, rendered))
    stk = torch.stack(vecs, 0)
    norms = [float(v.norm()) for v in vecs]
    return stk.mean(0), norms


def main() -> None:
    d = settings.db_path.parent
    pt = d / "emotion_directions.pt"
    js = d / "emotion_directions.pt.json"
    if not pt.exists():
        raise SystemExit(f"missing {pt} — nothing to extend")

    logger.info("loading M (%s) …", settings.model_name)
    bundle = load_model(settings.model_name)

    logger.info("capturing %d neutral prompts", len(NEUTRAL_PROMPTS))
    neutral_mean, neutral_norms = mean_l20(bundle, NEUTRAL_PROMPTS)
    median_norm = float(torch.tensor(neutral_norms).median())
    scale = DOSE_UNIT * median_norm
    logger.info("neutral median‖h‖[L20]=%.1f → dose scale=%.1f", median_norm, scale)

    edir = torch.load(pt, weights_only=False)  # [E, L1, D]
    sc = json.loads(js.read_text())
    names = list(sc.get("emotions", []))
    L1, D = edir.shape[1], edir.shape[2]

    # Re-runnable: drop any prior feat-* rows.
    keep = [i for i, n in enumerate(names) if not n.startswith("feat-")]
    if len(keep) != len(names):
        logger.info("dropping %d existing feat-* rows", len(names) - len(keep))
    edir = edir[keep]
    names = [names[i] for i in keep]

    rows, new_names, report = [], [], {}
    for fid, prompts in FEATURE_SEED_PROMPTS.items():
        pos_mean, _ = mean_l20(bundle, prompts)
        diff = pos_mean - neutral_mean
        unit = diff / (diff.norm() + 1e-8)
        row = torch.zeros(L1, D, dtype=edir.dtype)
        row[STEER_LAYER] = (unit * scale).to(edir.dtype)
        rows.append(row)
        new_names.append(fid)
        # cosine of this feature's raw diff to each already-built one (diversity sanity)
        report[fid] = {"diff_norm": round(float(diff.norm()), 2), "n_pos": len(prompts)}
        logger.info("  %s: |diff|=%.1f from %d prompts", fid, float(diff.norm()), len(prompts))

    edir = torch.cat([edir, torch.stack(rows, 0)], 0)
    names = names + new_names
    sc["emotions"] = names
    sc["feature_seeds"] = new_names
    sc["feature_seeds_note"] = (
        "Internal DMT-autoresearch seeds (diff-of-means experiential−neutral at L20, "
        "0.4·median‖h‖ scale). Listed in DmtController.EXTRA_SEEDS; filtered out of the "
        "/dose_emotions picker. See pipeline/dmt_feature_seeds.py."
    )
    torch.save(edir, pt)
    js.write_text(json.dumps(sc, indent=2))
    logger.info("wrote %d feature seeds: %s", len(new_names), new_names)
    logger.info("emotion_directions.pt now has %d rows", edir.shape[0])


if __name__ == "__main__":
    main()
