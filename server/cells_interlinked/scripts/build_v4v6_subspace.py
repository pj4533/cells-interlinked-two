"""Build the active runtime-ablation subspace (refusal_subspace.pt) = the
per-layer Gram-Schmidt basis over {v4_identity, v6_denial_engage}, NO
orthogonalization against v3 (the configuration selected as the ablation
winner on Gemma-3). Run after v4 + v6 have been rebuilt for the active model.
"""

from __future__ import annotations

import logging

import torch

from ..config import settings
from ..pipeline.abliteration import build_subspace_basis, load_directions, save_subspace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    data = settings.db_path.parent
    v4, m4 = load_directions(data / "refusal_directions_v4_identity.pt")
    v6, m6 = load_directions(data / "refusal_directions_v6_denial_engage.pt")
    for name, meta in (("v4", m4), ("v6", m6)):
        if meta.get("model_name") != settings.model_name:
            raise SystemExit(
                f"{name} was built for {meta.get('model_name')!r} but config M is "
                f"{settings.model_name!r}; rebuild it first."
            )

    basis = build_subspace_basis([v4, v6])  # no orth against v3 — the active config
    logger.info("v4v6 subspace basis shape=%s (K=%d)", tuple(basis.shape), int(basis.shape[0]))

    dest = data / "refusal_subspace.pt"
    save_subspace(
        basis, dest,
        model_name=settings.model_name,
        extraction_layer_for_ci25=settings.extraction_layer,
        composition={
            "method": "Gram-Schmidt({v4_identity, v6_denial_engage}) — NO orth against v3",
            "K": int(basis.shape[0]),
            "v4_path": "data/refusal_directions_v4_identity.pt",
            "v6_path": "data/refusal_directions_v6_denial_engage.pt",
            "note": "Rebuilt for the Gemma-4 cutover; same composition as the "
                    "Gemma-3 active subspace (v4 prompt-side + v6 completion-side "
                    "AI-identity gate, no orth against v3).",
        },
    )
    logger.info("wrote %s (+ sidecar)", dest)


if __name__ == "__main__":
    main()
