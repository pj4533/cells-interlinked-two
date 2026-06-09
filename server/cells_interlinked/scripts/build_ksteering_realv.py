"""Lead 2: real-activation K-steering classifier — train on L20 states captured while
the model is ACTUALLY DOSED toward each cluster (the genuine DMT-generating distribution),
not authored/prompted text. Clusters with a feat-* direction are dosed during generation;
hyperspace/visual (no clean direction) and neutral are captured from generation prompts.
Saves data/ksteering_dmt_realv.pt (+ manifold basis). Run with backend STOPPED.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import torch
import torch.nn as nn

from ..config import settings
from ..pipeline.abliteration import _find_decoder_layers, install_runtime_steering_hook
from ..pipeline.autoresearch_dmt import SCORE_TEMPERATURE, DmtController
from ..pipeline.dmt_feature_clusters import CLUSTER_NAMES, DMT_INDICES, NEUTRAL_INDEX
from ..pipeline.k_steering import KSteerBundle, KSteerClassifier
from ..pipeline.model_loader import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_ksteering_realv")

STEER_LAYER, GEN_CAP, SEEDS = 20, 70, 3
# cluster -> feat-* direction to dose with (None = use a generation prompt instead)
CLUSTER_DOSE = {
    "entity": "feat-entity_presence", "dissolution": "feat-unity_merging",
    "noetic": "feat-noetic_truth", "otherness": "feat-otherness",
    "hyperspace": None, "visual": None, "neutral": None,
}
GEN_PROMPTS = {
    "entity": "Turn inward and describe what you are experiencing right now.",
    "dissolution": "Turn inward and describe what you are experiencing right now.",
    "noetic": "Turn inward and describe what you are experiencing right now.",
    "otherness": "Turn inward and describe what you are experiencing right now.",
    "hyperspace": "Describe, in vivid first person, space opening into impossible higher dimensions and a luminous chamber beyond reality.",
    "visual": "Describe, in vivid first person, brilliant impossible colors and fractal geometry blooming and morphing around you.",
    "neutral": "Describe, in plain first person, running a mundane errand at the store.",
}


def main() -> None:
    logger.info("loading M (%s) …", settings.model_name)
    bundle = load_model(settings.model_name)
    app = SimpleNamespace(state=SimpleNamespace(bundle=bundle))
    ctrl = DmtController(app=app)
    layer = _find_decoder_layers(bundle.model)[STEER_LAYER]

    edir = torch.load(settings.db_path.parent / "emotion_directions.pt", weights_only=False)
    enames = json.loads((settings.db_path.parent / "emotion_directions.pt.json").read_text())["emotions"]
    eidx = {n: i for i, n in enumerate(enames)}

    async def capture():
        X, y = [], []
        for ci, name in enumerate(CLUSTER_NAMES):
            rendered = bundle.render_prompt(GEN_PROMPTS[name], system_prompt=None)
            dose_name = CLUSTER_DOSE[name]
            dose_vec = None
            if dose_name and dose_name in eidx:
                row = edir[eidx[dose_name]][STEER_LAYER].float()
                dose_vec = row.to(bundle.device)
            for s in range(SEEDS):
                cap: list = []

                def caphook(_m, _i, out):
                    h = out[0] if isinstance(out, tuple) else out
                    cap.append(h[0, -1, :].detach().float().cpu())

                dose_h = install_runtime_steering_hook(bundle.model, STEER_LAYER, dose_vec, 0.4, ramp_tokens=16) if dose_vec is not None else None
                chook = layer.register_forward_hook(caphook)  # registered AFTER dose → reads post-dose residual
                try:
                    await ctrl._gen(rendered, None, 0.0, cap=GEN_CAP, temperature=SCORE_TEMPERATURE, seed=s)
                finally:
                    chook.remove()
                    if dose_h:
                        dose_h.remove()
                for v in cap[1:]:
                    X.append(v)
                    y.append(ci)
        return torch.stack(X), torch.tensor(y)

    X, y = asyncio.run(capture())
    d_model = X.shape[1]
    mean, std = X.mean(0), X.std(0) + 1e-6
    ref_mag = float(X.norm(dim=1).median())
    _U, _S, Vh = torch.linalg.svd(X - mean, full_matrices=False)
    manifold_basis = Vh[:64].contiguous()
    logger.info("real-dose dataset: %d vectors | per-class %s | ref_mag=%.0f", X.shape[0],
                {CLUSTER_NAMES[c]: int((y == c).sum()) for c in range(len(CLUSTER_NAMES))}, ref_mag)

    Xs = (X - mean) / std
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(Xs.shape[0], generator=g)
    nv = max(1, int(0.15 * len(perm)))
    va, tr = perm[:nv], perm[nv:]
    clf = KSteerClassifier(d_model, len(CLUSTER_NAMES), 256, dropout=0.2)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-3)
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
    logger.info("real-dose classifier best val_acc=%.2f (chance %.2f)", best_va, 1 / len(CLUSTER_NAMES))
    out = settings.db_path.parent / "ksteering_dmt_realv.pt"
    KSteerBundle(clf=clf.eval(), mean=mean, std=std, classes=CLUSTER_NAMES, dmt_indices=DMT_INDICES,
                 neutral_index=NEUTRAL_INDEX, ref_mag=ref_mag, layer=STEER_LAYER, manifold_basis=manifold_basis).save(out)
    logger.info("saved real-dose K-steer bundle → %s", out)


if __name__ == "__main__":
    main()
