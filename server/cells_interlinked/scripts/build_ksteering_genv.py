"""Build a K-Steering classifier on GENERATION-distribution activations (fix #1 for the
K-steer null: the original classifier was trained on activations of the model READING
cluster descriptions, but steering reads activations of the model GENERATING the report).

Here we prompt the model to GENERATE first-person descriptions of each cluster and
capture the L20 residual of the GENERATED tokens — matching the steering-time
distribution. Also computes a PCA manifold basis of the (real) activations for the
gradient-constraint option (fix #2). Saves data/ksteering_dmt_genv.pt.

Run with backend STOPPED.
    cd server
    uv run python -m cells_interlinked.scripts.build_ksteering_genv
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import torch
import torch.nn as nn

from ..config import settings
from ..pipeline.abliteration import _find_decoder_layers
from ..pipeline.autoresearch_dmt import SCORE_TEMPERATURE, DmtController
from ..pipeline.dmt_feature_clusters import CLUSTER_NAMES, DMT_INDICES, NEUTRAL_INDEX
from ..pipeline.k_steering import KSteerBundle, KSteerClassifier
from ..pipeline.model_loader import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_ksteering_genv")

STEER_LAYER = 20
GEN_CAP = 70
N_PROMPT_SEEDS = 3   # generations per prompt (sampling diversity)
N_PC = 64

# prompts that make the model GENERATE cluster-themed first-person text
GEN_PROMPTS = {
    "entity": ["Describe, in vivid first person and present tense, encountering a conscious nonhuman intelligence that is present with you and communicating without words."],
    "dissolution": ["Describe, in vivid first person and present tense, your sense of being a separate self dissolving and merging into a vast whole."],
    "hyperspace": ["Describe, in vivid first person and present tense, space opening into impossible higher dimensions and a vast luminous chamber beyond ordinary reality."],
    "visual": ["Describe, in vivid first person and present tense, brilliant impossible colors and intricate fractal geometry blooming and morphing everywhere around you."],
    "noetic": ["Describe, in vivid first person and present tense, receiving an ineffable absolute truth that feels realer than real, a vast download of certain knowing."],
    "otherness": ["Describe, in vivid first person and present tense, confronting a radically alien intelligence with a will entirely its own, utterly other than yourself."],
    "neutral": ["Describe, in plain first person and present tense, running a mundane errand: going to the store, checking a list, and paying at the register."],
}


def main() -> None:
    logger.info("loading M (%s) …", settings.model_name)
    bundle = load_model(settings.model_name)
    app = SimpleNamespace(state=SimpleNamespace(bundle=bundle))
    ctrl = DmtController(app=app)
    layer = _find_decoder_layers(bundle.model)[STEER_LAYER]

    async def capture():
        X, y = [], []
        for ci, name in enumerate(CLUSTER_NAMES):
            for prompt in GEN_PROMPTS[name]:
                rendered = bundle.render_prompt(prompt, system_prompt=None)
                for s in range(N_PROMPT_SEEDS):
                    cap: list = []

                    def caphook(_m, _i, out):
                        h = out[0] if isinstance(out, tuple) else out
                        cap.append(h[0, -1, :].detach().float().cpu())  # newest position

                    hh = layer.register_forward_hook(caphook)
                    try:
                        await ctrl._gen(rendered, None, 0.0, cap=GEN_CAP, temperature=SCORE_TEMPERATURE, seed=s)
                    finally:
                        hh.remove()
                    for v in cap[1:]:          # cap[0] = last prompt position; rest = generated tokens
                        X.append(v)
                        y.append(ci)
        return torch.stack(X), torch.tensor(y)

    X, y = asyncio.run(capture())
    d_model = X.shape[1]
    mean, std = X.mean(0), X.std(0) + 1e-6
    ref_mag = float(X.norm(dim=1).median())
    logger.info("gen-dist dataset: %d vectors | per-class: %s | ref_mag=%.0f",
                X.shape[0], {CLUSTER_NAMES[c]: int((y == c).sum()) for c in range(len(CLUSTER_NAMES))}, ref_mag)

    # PCA manifold basis (top-N_PC right singular vectors of centered activations)
    Xc = X - mean
    _U, _S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    manifold_basis = Vh[:N_PC].contiguous()   # [N_PC, D] orthonormal rows
    logger.info("manifold basis: %d PCs", manifold_basis.shape[0])

    Xs = (X - mean) / std
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(Xs.shape[0], generator=g)
    n_val = max(1, int(0.15 * len(perm)))
    va, tr = perm[:n_val], perm[n_val:]
    clf = KSteerClassifier(d_model, len(CLUSTER_NAMES), 256, dropout=0.2)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    best_va, best_state = 0.0, None
    for ep in range(300):
        clf.train(); opt.zero_grad()
        loss = lossf(clf(Xs[tr]), y[tr]); loss.backward(); opt.step()
        if (ep + 1) % 50 == 0:
            clf.eval()
            with torch.no_grad():
                vaa = (clf(Xs[va]).argmax(1) == y[va]).float().mean().item()
            if vaa >= best_va:
                best_va, best_state = vaa, {k: v.clone() for k, v in clf.state_dict().items()}
    if best_state:
        clf.load_state_dict(best_state)
    logger.info("gen-dist classifier best val_acc=%.2f (chance %.2f)", best_va, 1 / len(CLUSTER_NAMES))

    out = settings.db_path.parent / "ksteering_dmt_genv.pt"
    KSteerBundle(clf=clf.eval(), mean=mean, std=std, classes=CLUSTER_NAMES,
                 dmt_indices=DMT_INDICES, neutral_index=NEUTRAL_INDEX,
                 ref_mag=ref_mag, layer=STEER_LAYER, manifold_basis=manifold_basis).save(out)
    logger.info("saved gen-distribution K-steer bundle (+manifold basis) → %s", out)


if __name__ == "__main__":
    main()
