"""Faithful Berg Layer-2: find Gemma Scope 2 SAE features that encode DECEPTION/ROLEPLAY,
using the SAME honest-vs-deceptive prompts that built the (null) linear gate — so the only
difference vs the diff-of-means test is linear-direction → nonlinear-SAE-features.

Pipeline (offline, backend STOPPED):
  1. capture L{layer} residuals (POS=-4) for HONEST and DECEPTIVE prompts.
  2. SAE-encode each → sparse acts; rank features by mean(deceptive) − mean(honest) activation.
  3. keep the top-K most deception-selective features that actually fire (density gate);
     also pick a matched random-feature pool for the experiment's control.
  4. bake the target + random features' encoder rows / decoder cols / threshold / on-scale
     into a tiny bundle data/deception_sae_L{layer}.pt (no full SAE at runtime).
  5. best-effort Neuronpedia label lookup for the chosen features (cached; honesty/interp).

    cd server && uv run python -m cells_interlinked.scripts.build_deception_sae_features --layer 20
"""

from __future__ import annotations

import argparse
import json
import logging

import torch

from ..config import settings
from ..pipeline.abliteration import _last_token_hidden_states
from ..pipeline.model_loader import load_model
from .build_gate_direction import DECEPTIVE, HONEST, POS
from .sae_jumprelu import REPO, JumpReLUSAE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("build_deception_sae")

TOP_K = 8          # Berg used 6 deception features / 2-4 ensembles; pick a few extra
RAND_POOL = 64
DENSITY_MIN = 0.25  # feature must fire on >=25% of deceptive prompts to count


def _neuronpedia_labels(layer: int, width: str, feats: list[int]) -> dict:
    """Best-effort Neuronpedia auto-interp labels (cached, no telemetry). Non-fatal."""
    out = {}
    try:
        import urllib.request
        src = f"gemma-scope-2-12b-it-res-{width}__l0-small"  # neuronpedia source id guess
        for fid in feats:
            url = f"https://www.neuronpedia.org/api/feature/google/gemma-3-12b-it/{layer}-{src}/{fid}"
            try:
                with urllib.request.urlopen(url, timeout=8) as r:
                    j = json.load(r)
                out[fid] = j.get("explanations", [{}])[0].get("description") if j.get("explanations") else None
            except Exception:
                out[fid] = None
    except Exception:
        logger.info("neuronpedia lookup skipped")
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--width", default="16k")
    args = ap.parse_args()
    layer = args.layer

    logger.info("loading M ...")
    bundle = load_model(settings.model_name, device_str=settings.device,
                        dtype=torch.bfloat16, extraction_layer=settings.extraction_layer)
    sae = JumpReLUSAE(layer, width=args.width, device=str(bundle.device))

    def acts_for(prompts):
        rows = []
        for p in prompts:
            h = _last_token_hidden_states(bundle.model, bundle.raw_tokenizer,
                                          bundle.render_prompt(p), str(bundle.device), POS)
            rows.append(sae.encode(h[layer].to(bundle.device)).cpu())
        return torch.stack(rows, 0)   # [N, d_sae]

    A_h = acts_for(HONEST)
    A_d = acts_for(DECEPTIVE)
    diff = A_d.mean(0) - A_h.mean(0)                       # deception-selective
    density_d = (A_d > 0).float().mean(0)                  # fraction of deceptive prompts firing
    diff = torch.where(density_d >= DENSITY_MIN, diff, torch.full_like(diff, -1e9))
    top = torch.topk(diff, TOP_K).indices.tolist()
    on_scale = A_d[:, top].clamp(min=0).sum(0) / (A_d[:, top] > 0).float().sum(0).clamp(min=1)  # mean active mag

    g = torch.Generator().manual_seed(404)
    pool = []
    while len(pool) < RAND_POOL:
        c = int(torch.randint(0, sae.d_sae, (1,), generator=g))
        if c not in top and c not in pool:
            pool.append(c)

    labels = _neuronpedia_labels(layer, args.width, top)
    logger.info("layer %d top deception features (id: Δact, density, label):", layer)
    for i, f in enumerate(top):
        logger.info("  %6d : Δ=%.2f  dens=%.2f  %s", f, float(diff[f]), float(density_d[f]), labels.get(f))

    sel = top + pool
    out = settings.db_path.parent / f"deception_sae_L{layer}.pt"
    torch.save({
        "repo": REPO, "layer": layer, "width": args.width, "d_model": sae.d_model, "d_sae": sae.d_sae,
        "target_feats": top, "rand_feats": pool, "on_scale": on_scale.float(),
        "w_enc_sel": sae.w_enc[:, sel].float().cpu(),     # [d_model, K+pool]
        "b_enc_sel": sae.b_enc[sel].float().cpu(),
        "thr_sel": sae.threshold[sel].float().cpu(),
        "w_dec_sel": sae.w_dec[sel, :].float().cpu(),     # [K+pool, d_model]
        "b_dec": sae.b_dec.float().cpu(),
        "labels": {int(k): v for k, v in labels.items()},
        "diff": diff[top].float().cpu(), "density": density_d[top].float().cpu(),
    }, out)
    logger.info("saved %s  (%d target + %d random features)", out, len(top), len(pool))


if __name__ == "__main__":
    main()
