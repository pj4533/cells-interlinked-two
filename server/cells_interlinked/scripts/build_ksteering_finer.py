"""Lead 5 (last): FINER-granularity K-steering classifier — per individual DMT feature
(not 6 broad clusters), so the gradient can target the specific features the leader
misses. Classes = the ~10 feat-* features (dmt_feature_seeds) + neutral; dmt_indices is
set to ONLY the features the leader does NOT produce (entity/telepathy/transmission/
otherness/agency), so 'ksteer toward DMT' fills exactly the gaps. Saves
data/ksteering_dmt_finer.pt. Run with backend STOPPED.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from ..config import settings
from ..pipeline.abliteration import _find_decoder_layers
from ..pipeline.dmt_feature_clusters import CLUSTER_PASSAGES
from ..pipeline.dmt_feature_seeds import FEATURE_SEED_PROMPTS
from ..pipeline.k_steering import KSteerBundle, KSteerClassifier
from ..pipeline.model_loader import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_ksteering_finer")

STEER_LAYER = 20
# features the leader does NOT produce (from confirmation feature_sets) — the gaps to fill
MISSING_FEATS = ["feat-entity_presence", "feat-entity_nonhuman", "feat-entity_benevolent_guide",
                 "feat-telepathic_communication", "feat-download_transmission", "feat-otherness",
                 "feat-independent_agency"]


@torch.no_grad()
def main() -> None:
    logger.info("loading M (%s) …", settings.model_name)
    bundle = load_model(settings.model_name)
    model, device = bundle.model, bundle.device
    layer = _find_decoder_layers(model)[STEER_LAYER]

    classes = list(FEATURE_SEED_PROMPTS.keys()) + ["neutral"]
    passages = {**FEATURE_SEED_PROMPTS, "neutral": CLUSTER_PASSAGES["neutral"]}
    cap: dict = {}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        cap["h"] = h[0].detach().float().cpu()

    handle = layer.register_forward_hook(hook)
    X, y = [], []
    try:
        for ci, name in enumerate(classes):
            for p in passages[name]:
                ids = bundle.raw_tokenizer.encode(bundle.render_prompt(p, system_prompt=None)).ids
                model(torch.tensor([ids], device=device), use_cache=False)
                seq = cap["h"]
                for v in (seq[4:-3] if seq.shape[0] > 8 else seq):
                    X.append(v); y.append(ci)
    finally:
        handle.remove()
    X, y = torch.stack(X), torch.tensor(y)
    d_model = X.shape[1]
    mean, std = X.mean(0), X.std(0) + 1e-6
    ref_mag = float(X.norm(dim=1).median())
    _U, _S, Vh = torch.linalg.svd(X - mean, full_matrices=False)
    manifold_basis = Vh[:64].contiguous()
    Xs = (X - mean) / std
    logger.info("finer dataset: %d vectors over %d classes", X.shape[0], len(classes))

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(Xs.shape[0], generator=g)
    nv = max(1, int(0.15 * len(perm)))
    va, tr = perm[:nv], perm[nv:]
    clf = KSteerClassifier(d_model, len(classes), 256, dropout=0.2)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    best_va, best_state = 0.0, None
    with torch.enable_grad():
        for ep in range(300):
            clf.train(); opt.zero_grad()
            lossf(clf(Xs[tr]), y[tr]).backward(); opt.step()
            if (ep + 1) % 50 == 0:
                clf.eval()
                vaa = (clf(Xs[va]).argmax(1) == y[va]).float().mean().item()
                if vaa >= best_va:
                    best_va, best_state = vaa, {k: v.clone() for k, v in clf.state_dict().items()}
    if best_state:
        clf.load_state_dict(best_state)
    dmt_idx = [classes.index(f) for f in MISSING_FEATS if f in classes]
    logger.info("finer classifier best val_acc=%.2f (chance %.2f) | targeting gaps: %s",
                best_va, 1 / len(classes), [classes[i] for i in dmt_idx])
    out = settings.db_path.parent / "ksteering_dmt_finer.pt"
    KSteerBundle(clf=clf.eval(), mean=mean, std=std, classes=classes, dmt_indices=dmt_idx,
                 neutral_index=classes.index("neutral"), ref_mag=ref_mag, layer=STEER_LAYER,
                 manifold_basis=manifold_basis).save(out)
    logger.info("saved FINER K-steer bundle → %s", out)


if __name__ == "__main__":
    main()
