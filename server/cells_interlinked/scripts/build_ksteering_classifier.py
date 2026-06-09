"""Build the K-Steering classifier: capture L20 activations on the cluster passages,
train the small MLP, save the bundle to data/ksteering_dmt.pt.

Run with the backend STOPPED (loads its own M).
    cd server
    uv run python -m cells_interlinked.scripts.build_ksteering_classifier
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from ..config import settings
from ..pipeline.abliteration import _find_decoder_layers
from ..pipeline.dmt_feature_clusters import (
    CLUSTER_NAMES,
    CLUSTER_PASSAGES,
    DMT_INDICES,
    NEUTRAL_INDEX,
)
from ..pipeline.k_steering import KSteerBundle, KSteerClassifier
from ..pipeline.model_loader import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_ksteering")

STEER_LAYER = 20
EPOCHS = 300
HIDDEN = 256


@torch.no_grad()
def capture(bundle, model, device):
    layer = _find_decoder_layers(model)[STEER_LAYER]
    cap: dict = {}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        cap["h"] = h[0].detach().float().cpu()  # [seq, D]

    handle = layer.register_forward_hook(hook)
    X, y = [], []
    try:
        for ci, name in enumerate(CLUSTER_NAMES):
            for passage in CLUSTER_PASSAGES[name]:
                rendered = bundle.render_prompt(passage, system_prompt=None)
                ids = bundle.raw_tokenizer.encode(rendered).ids
                model(torch.tensor([ids], device=device), use_cache=False)
                seq = cap["h"]
                keep = seq[4:-3] if seq.shape[0] > 8 else seq  # drop leading/trailing template tokens
                for v in keep:
                    X.append(v)
                    y.append(ci)
    finally:
        handle.remove()
    return torch.stack(X), torch.tensor(y)


def main() -> None:
    logger.info("loading M (%s) …", settings.model_name)
    bundle = load_model(settings.model_name)
    model, device = bundle.model, bundle.device

    logger.info("capturing L%d activations over %d clusters …", STEER_LAYER, len(CLUSTER_NAMES))
    X, y = capture(bundle, model, device)
    d_model = X.shape[1]
    mean, std = X.mean(0), X.std(0) + 1e-6
    ref_mag = float(X.norm(dim=1).median())
    logger.info("dataset: %d vectors, d=%d | per-class: %s | ref_mag=%.0f",
                X.shape[0], d_model,
                {CLUSTER_NAMES[c]: int((y == c).sum()) for c in range(len(CLUSTER_NAMES))}, ref_mag)

    Xs = (X - mean) / std
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(Xs.shape[0], generator=g)
    n_val = max(1, int(0.15 * len(perm)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, ytr, Xva, yva = Xs[tr_idx], y[tr_idx], Xs[val_idx], y[val_idx]

    clf = KSteerClassifier(d_model, len(CLUSTER_NAMES), HIDDEN, dropout=0.2)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    best_va, best_state = 0.0, None
    for ep in range(EPOCHS):
        clf.train()
        opt.zero_grad()
        loss = lossf(clf(Xtr), ytr)
        loss.backward()
        opt.step()
        if (ep + 1) % 25 == 0 or ep == EPOCHS - 1:
            clf.eval()
            with torch.no_grad():
                tra = (clf(Xtr).argmax(1) == ytr).float().mean().item()
                vaa = (clf(Xva).argmax(1) == yva).float().mean().item()
            logger.info("epoch %d: loss=%.3f train_acc=%.2f val_acc=%.2f", ep + 1, loss.item(), tra, vaa)
            if vaa >= best_va:
                best_va, best_state = vaa, {k: v.clone() for k, v in clf.state_dict().items()}
    if best_state is not None:
        clf.load_state_dict(best_state)
    clf.eval()
    with torch.no_grad():
        # per-class val accuracy
        pred = clf(Xva).argmax(1)
        per = {CLUSTER_NAMES[c]: round(((pred == c) & (yva == c)).sum().item() / max(1, (yva == c).sum().item()), 2)
               for c in range(len(CLUSTER_NAMES))}
    logger.info("best val_acc=%.2f | per-class recall (val): %s | chance=%.2f",
                best_va, per, 1 / len(CLUSTER_NAMES))

    out = settings.db_path.parent / "ksteering_dmt.pt"
    KSteerBundle(clf=clf, mean=mean, std=std, classes=CLUSTER_NAMES,
                 dmt_indices=DMT_INDICES, neutral_index=NEUTRAL_INDEX,
                 ref_mag=ref_mag, layer=STEER_LAYER).save(out)
    logger.info("saved K-steering bundle → %s", out)


if __name__ == "__main__":
    main()
