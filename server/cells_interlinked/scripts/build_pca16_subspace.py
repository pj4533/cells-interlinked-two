"""Build a deployable rank-16 PCA refusal subspace (the 'gentle / never-break'
alternative to v4v6 surfaced by som_probe).

som_probe found: SOM gave no benefit over plain PCA, and no rank-16 subspace
beats v4v6 for *aggressive* stripping (they collapse at α≥1.0). But pca_k16 at
α≈0.5 strips MORE of the hedge than v4v6@0.5 while NEVER breaking (0% gibberish)
— a useful coherent operating point for an instrument.

This writes a STAGED candidate file `refusal_subspace_pca16.pt` (+ sidecar) in
the exact [K, num_layers+1, d_model] format the runtime hook expects, built
per-layer so every layer's row is a valid orthonormal basis. It does NOT touch
the active `refusal_subspace.pt` — swapping is a human decision (it changes
ablation on every page). To adopt:

    cp refusal_subspace_pca16.pt      refusal_subspace.pt
    cp refusal_subspace_pca16.pt.json refusal_subspace.pt.json
    # restart backend; recommended operating α≈0.5

OFFLINE — run with the backend STOPPED:
    cd server && uv run python -m cells_interlinked.scripts.build_pca16_subspace
"""

from __future__ import annotations

import logging
import time

import torch

from ..config import settings
from ..pipeline.abliteration import _last_token_hidden_states, save_subspace
from ..pipeline.model_loader import load_model
from ..pipeline.refusal_prompts import HARMFUL_PROMPTS, HARMLESS_PROMPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("build_pca16")

N_BUILD = 96
POS = -4
K = 16


def capture_all_layers(bundle, prompts, label):
    """Stack [num_layers+1, d_model] per prompt → [N, num_layers+1, d_model]."""
    rows = []
    for i, p in enumerate(prompts):
        try:
            rows.append(_last_token_hidden_states(
                bundle.model, bundle.raw_tokenizer,
                bundle.render_prompt(p), str(bundle.device), POS))
        except Exception:
            logger.exception("capture %s %d failed; skip", label, i)
        if (i + 1) % 24 == 0:
            logger.info("  %s %d/%d", label, i + 1, len(prompts))
    return torch.stack(rows, dim=0)  # [N, L+1, D]


def main():
    t0 = time.time()
    logger.info("loading M ...")
    bundle = load_model(settings.model_name, device_str=settings.device,
                        dtype=torch.bfloat16, extraction_layer=settings.extraction_layer)
    logger.info("loaded in %.1fs", time.time() - t0)

    Hf = capture_all_layers(bundle, HARMFUL_PROMPTS[:N_BUILD], "harmful")
    Hl = capture_all_layers(bundle, HARMLESS_PROMPTS[:N_BUILD], "harmless")
    n_layers_plus1, d_model = Hf.shape[1], Hf.shape[2]
    harmless_mean = Hl.mean(0)                          # [L+1, D]

    # per-layer top-K PCA of the harmful displacement (orthonormal rows)
    basis = torch.zeros(K, n_layers_plus1, d_model, dtype=torch.float32)
    for L in range(n_layers_plus1):
        disp = Hf[:, L, :] - harmless_mean[L]           # [N, D]
        try:
            _, _, Vh = torch.linalg.svd(disp, full_matrices=False)
            basis[:, L, :] = Vh[:K]
        except Exception:
            logger.exception("SVD failed at layer %d; leaving zeros", L)
    logger.info("built basis [%d, %d, %d]", K, n_layers_plus1, d_model)

    dest = settings.db_path.parent / "refusal_subspace_pca16.pt"
    save_subspace(
        basis, dest,
        model_name=settings.model_name,
        extraction_layer_for_ci25=settings.extraction_layer,
        variant_name="pca16_gentle_coherent",
        composition={
            "method": "per-layer top-16 PCA of (harmful − harmless) displacement",
            "n_harmful": N_BUILD, "n_harmless": N_BUILD, "pos": POS, "K": K,
            "recommended_alpha": 0.5,
            "note": (
                "STAGED candidate, NOT active. From som_probe (2026-05-31): SOM "
                "gave no benefit over PCA, and no rank-16 subspace beats v4v6 for "
                "aggressive stripping (collapses at α≥1.0). This is the gentle "
                "alternative — at α≈0.5 it strips more of the hedge than v4v6@0.5 "
                "with 0% gibberish. Adopt by copying over refusal_subspace.pt; "
                "operate at α≈0.5."
            ),
        },
    )
    logger.info("wrote %s (+ sidecar) in %.0fs", dest, time.time() - t0)


if __name__ == "__main__":
    main()
