"""Lead 3: adversarially-ROBUST K-steering classifier. A normal classifier's input
gradient is the boundary normal (an off-distribution/adversarial direction); a PGD-
adversarially-trained classifier has *semantically aligned* gradients (Tsipras/Engstrom)
— the principled fix for "the discriminative gradient is not a generative DMT direction."
Saves data/ksteering_dmt_robust.pt (+ manifold basis). Run with backend STOPPED.
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
logger = logging.getLogger("build_ksteering_robust")

STEER_LAYER = 20
EPOCHS = 400
EPS = 0.8          # PGD radius in standardized-input space
PGD_STEPS = 3
PGD_LR = 0.3


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
    Xtr, ytr, Xva, yva = Xs[tr], y[tr], Xs[va], y[va]

    clf = KSteerClassifier(d_model, len(CLUSTER_NAMES), 256, dropout=0.2)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    best_va, best_state = 0.0, None
    for ep in range(EPOCHS):
        clf.train()
        # PGD: find an adversarial perturbation of the inputs, then train on it (robust training)
        delta = torch.zeros_like(Xtr)
        for _ in range(PGD_STEPS):
            delta.requires_grad_(True)
            loss = lossf(clf(Xtr + delta), ytr)
            (gd,) = torch.autograd.grad(loss, delta)
            delta = (delta + PGD_LR * gd.sign()).clamp_(-EPS, EPS).detach()
        opt.zero_grad()
        loss = lossf(clf(Xtr + delta), ytr)
        loss.backward()
        opt.step()
        if (ep + 1) % 50 == 0:
            clf.eval()
            with torch.no_grad():
                vaa = (clf(Xva).argmax(1) == yva).float().mean().item()
            if vaa >= best_va:
                best_va, best_state = vaa, {k: v.clone() for k, v in clf.state_dict().items()}
    if best_state:
        clf.load_state_dict(best_state)
    logger.info("robust classifier best clean val_acc=%.2f (chance %.2f)", best_va, 1 / len(CLUSTER_NAMES))

    out = settings.db_path.parent / "ksteering_dmt_robust.pt"
    KSteerBundle(clf=clf.eval(), mean=mean, std=std, classes=CLUSTER_NAMES,
                 dmt_indices=DMT_INDICES, neutral_index=NEUTRAL_INDEX,
                 ref_mag=ref_mag, layer=STEER_LAYER, manifold_basis=manifold_basis).save(out)
    logger.info("saved robust K-steer bundle → %s", out)


if __name__ == "__main__":
    main()
