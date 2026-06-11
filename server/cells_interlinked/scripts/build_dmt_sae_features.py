"""SAE clamping for DMT (DMT_NEXT_DIRECTIONS #3): select a DIVERSE set of Gemma Scope 2 SAE
features — one per DMT phenomenology cluster (entity / dissolution / hyperspace / visual /
noetic / otherness) — so clamping them all co-activates many feature-TYPES at once. This is the
exact thing linear vector-summing failed at (interference); SAE clamping is the doc's proposed
fix. Picks, per cluster, the SAE feature most selective for that cluster vs all other clusters +
neutral. Bakes target + matched-random features into data/dmt_sae_L{layer}.pt (reuses the Berg
SAE bundle format so make_sae_hook works directly).

OFFLINE — backend STOPPED:
    cd server && uv run python -m cells_interlinked.scripts.build_dmt_sae_features --layer 20
"""

from __future__ import annotations

import argparse
import logging

import torch

from ..config import settings
from ..pipeline.abliteration import _last_token_hidden_states
from ..pipeline.dmt_feature_clusters import (CLUSTER_NAMES, CLUSTER_PASSAGES,
                                             DMT_INDICES, NEUTRAL_INDEX)
from ..pipeline.model_loader import load_model
from .build_gate_direction import POS
from .sae_jumprelu import REPO, JumpReLUSAE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("build_dmt_sae")

RAND_POOL = 64
DENSITY_MIN = 0.34   # feature must fire on >=1/3 of its cluster's passages


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

    # encode every passage at L{layer}, grouped by cluster
    acts = {}
    for name in CLUSTER_NAMES:
        rows = []
        for p in CLUSTER_PASSAGES[name]:
            h = _last_token_hidden_states(bundle.model, bundle.raw_tokenizer,
                                          bundle.render_prompt(p), str(bundle.device), POS)
            rows.append(sae.encode(h[layer].to(bundle.device)).cpu())
        acts[name] = torch.stack(rows, 0)        # [n_passages, d_sae]

    all_other = lambda c: torch.cat([acts[n] for n in CLUSTER_NAMES if n != c], 0)
    chosen, chosen_cluster, on_scale = [], [], []
    for ci in DMT_INDICES:
        c = CLUSTER_NAMES[ci]
        sel = acts[c].mean(0) - all_other(c).mean(0)        # cluster-selective
        density = (acts[c] > 0).float().mean(0)
        sel = torch.where(density >= DENSITY_MIN, sel, torch.full_like(sel, -1e9))
        for f in torch.topk(sel, 8).indices.tolist():       # skip already-taken features
            if f not in chosen:
                chosen.append(f); chosen_cluster.append(c)
                on_scale.append(float(acts[c][:, f].max()))
                logger.info("cluster %-11s -> feat %6d  Δ=%.2f dens=%.2f max_act=%.1f",
                            c, f, float(sel[f]), float(density[f]), on_scale[-1])
                break

    g = torch.Generator().manual_seed(909)
    pool = []
    while len(pool) < RAND_POOL:
        x = int(torch.randint(0, sae.d_sae, (1,), generator=g))
        if x not in chosen and x not in pool:
            pool.append(x)

    seln = chosen + pool
    # max_act over ALL DMT cluster passages for every selected feature (targets + random),
    # so the random control gets a real matched clamp reference (v1 bug: zeros for random).
    dmt_all = torch.cat([acts[CLUSTER_NAMES[i]] for i in DMT_INDICES], 0)   # [N_dmt, d_sae]
    on_scale_full = dmt_all[:, seln].max(0).values.clamp(min=1.0)           # [K+pool]
    out = settings.db_path.parent / f"dmt_sae_L{layer}.pt"
    torch.save({
        "repo": REPO, "layer": layer, "width": args.width, "d_model": sae.d_model, "d_sae": sae.d_sae,
        "target_feats": chosen, "target_cluster": chosen_cluster, "rand_feats": pool,
        "on_scale": on_scale_full,                          # per-feature max_act, full length (targets+random)
        "w_enc_sel": sae.w_enc[:, seln].float().cpu(),
        "b_enc_sel": sae.b_enc[seln].float().cpu(),
        "thr_sel": sae.threshold[seln].float().cpu(),
        "w_dec_sel": sae.w_dec[seln, :].float().cpu(),
        "b_dec": sae.b_dec.float().cpu(),
    }, out)
    logger.info("saved %s  (%d diverse DMT feats + %d random)", out, len(chosen), len(pool))


if __name__ == "__main__":
    main()
