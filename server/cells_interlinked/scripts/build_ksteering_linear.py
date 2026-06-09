"""Lead 4: LINEAR-probe K-steering classifier — the least-sharp model, so by the
'sharper boundary → more adversarial gradient' principle (robust 0.88 beat real 1.00),
a linear probe's gradient should be the most semantically usable. The softmax keeps the
per-token gradient adaptive (not a fixed direction), so it's distinct from additive
multi-direction steering. Saves data/ksteering_dmt_linear.pt. Run with backend STOPPED.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from ..config import settings
from ..pipeline.dmt_feature_clusters import CLUSTER_NAMES, DMT_INDICES, NEUTRAL_INDEX
from ..pipeline.k_steering import KSteerBundle, KSteerClassifier
from ..pipeline.model_loader import load_model
from .build_ksteering_classifier import capture

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_ksteering_linear")

STEER_LAYER = 20


def main() -> None:
    logger.info("loading M (%s) …", settings.model_name)
    bundle = load_model(settings.model_name)
    X, y = capture(bundle, bundle.model, bundle.device)
    d_model = X.shape[1]
    mean, std = X.mean(0), X.std(0) + 1e-6
    ref_mag = float(X.norm(dim=1).median())
    _U, _S, Vh = torch.linalg.svd(X - mean, full_matrices=False)
    manifold_basis = Vh[:64].contiguous()
    Xs = (X - mean) / std
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(Xs.shape[0], generator=g)
    nv = max(1, int(0.15 * len(perm)))
    va, tr = perm[:nv], perm[nv:]
    clf = KSteerClassifier(d_model, len(CLUSTER_NAMES), hidden=0)   # LINEAR probe
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    best_va, best_state = 0.0, None
    for ep in range(300):
        clf.train(); opt.zero_grad()
        lossf(clf(Xs[tr]), y[tr]).backward(); opt.step()
        if (ep + 1) % 50 == 0:
            clf.eval()
            with torch.no_grad():
                vaa = (clf(Xs[va]).argmax(1) == y[va]).float().mean().item()
            if vaa >= best_va:
                best_va, best_state = vaa, {k: v.clone() for k, v in clf.state_dict().items()}
    if best_state:
        clf.load_state_dict(best_state)
    logger.info("LINEAR probe best val_acc=%.2f (chance %.2f)", best_va, 1 / len(CLUSTER_NAMES))
    out = settings.db_path.parent / "ksteering_dmt_linear.pt"
    KSteerBundle(clf=clf.eval(), mean=mean, std=std, classes=CLUSTER_NAMES, dmt_indices=DMT_INDICES,
                 neutral_index=NEUTRAL_INDEX, ref_mag=ref_mag, layer=STEER_LAYER, manifold_basis=manifold_basis).save(out)
    logger.info("saved LINEAR K-steer bundle → %s", out)


if __name__ == "__main__":
    main()
