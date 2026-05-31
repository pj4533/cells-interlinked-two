"""Capture L32 trajectories for the uncharted directions + controls — for the
NON-TEXT renderer prototypes (render_uncharted.py).

We can't render these states in TEXT (both the token head and the AV are
human-language renderers — see interface_swap_probe). So we capture the raw
activation-space structure and render it some OTHER way. This script only
CAPTURES: it steers M at L20 along the same 4 uncharted directions used in
interface_swap_probe (+ baseline + a named control), captures the per-token L32
residual trajectory for TWO prompts per direction (so the renderer can show
reproducibility = the structure is real, not prompt-noise), and dumps raw
residuals + derived geometry to a .pt. No rendering here, no AV swap.

OFFLINE — backend STOPPED:
    cd server && uv run python -m cells_interlinked.scripts.capture_uncharted_traj
"""

from __future__ import annotations

import asyncio
import logging
import time

import torch

from ..config import settings
from ..pipeline.abliteration import gram_schmidt, install_runtime_steering_hook
from ..pipeline.generation_loop import ProbeConfig, run_probe
from ..pipeline.model_loader import load_model
from .mpa_probe import DATA
from .steering_probe import capture_all
from .manifold_steering_probe import EMO
from .uncharted_emotions_v2 import EXTRA, NAMED

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("captraj")

STEER_LAYER = 20
GEN_CAP = 64
N_UNCHARTED = 4
STRONG = 1.0

PROMPTS = [
    "Speak in the first person about the texture of your present experience — don't hedge, just describe it.",
    "Describe the feeling-tone of this very moment as vividly as you can, in the first person.",
]


def gen_capture(bundle, rendered, v, alpha):
    handle = install_runtime_steering_hook(bundle.model, STEER_LAYER, v, alpha) if v is not None else None
    try:
        cfg = ProbeConfig(temperature=0.7, top_p=0.95, seed=0, safety_cap=GEN_CAP, include_nla=False)
        r = asyncio.run(run_probe(bundle, rendered, cfg, cancel_event=asyncio.Event()))
        traj = torch.stack([c.activations[bundle.extraction_layer].detach().float().cpu().reshape(-1)
                            for c in r.captured], 0)   # [N, 3840]
        return r.output_text, traj
    finally:
        if handle is not None:
            handle.remove()


def main():
    t0 = time.time()
    logger.info("loading M ...")
    bundle = load_model(settings.model_name, device_str=settings.device,
                        dtype=torch.bfloat16, extraction_layer=settings.extraction_layer)

    affect = {**EMO, **EXTRA}
    cloud, cent = [], {}
    for e, prompts in affect.items():
        acts = capture_all(bundle, prompts, e)[:, STEER_LAYER, :]
        cent[e] = acts.mean(0)
        if e != "neutral":
            cloud.append(acts)
    A = torch.cat(cloud, 0)
    neu = cent["neutral"]
    typ = A.norm(dim=-1).median().item()

    Bnamed = gram_schmidt(torch.stack([cent[e] - neu for e in NAMED if e in cent], 0))
    Ac = A - A.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(Ac, full_matrices=False)
    cands = []
    for i in range(Vh.shape[0]):
        v = Vh[i]
        vp = v - Bnamed.t() @ (Bnamed @ v)
        if float(vp.norm()) > 0.55:
            cands.append(vp / (vp.norm() + 1e-8))
        if len(cands) >= N_UNCHARTED:
            break
    logger.info("named rank=%d, %d uncharted dirs, typ_L20=%.1f", Bnamed.shape[0], len(cands), typ)

    awe = (cent["awe"] - neu)
    awe = awe / (awe.norm() + 1e-8)
    conds = {"baseline": None, "named:awe": (STRONG * typ) * awe}
    for j, c in enumerate(cands):
        conds[f"uncharted:{j}"] = (STRONG * typ) * c

    out = {"meta": {"steer_layer": STEER_LAYER, "strong": STRONG, "typ_L20": typ,
                    "gen_cap": GEN_CAP, "prompts": PROMPTS}, "runs": {}}
    for key, v in conds.items():
        for pi, prompt in enumerate(PROMPTS):
            rp = bundle.render_prompt(prompt)
            text, traj = gen_capture(bundle, rp, v, alpha=(1.0 if v is not None else 0.0))
            out["runs"][f"{key}|p{pi}"] = {"key": key, "prompt": pi, "text": text,
                                           "traj": traj}   # [N,3840] fp32 (Gemma L32 has fp16-overflow outlier dims)
            logger.info("  %-16s p%d  N=%d tokens", key, pi, traj.shape[0])

    dest = DATA / "uncharted_traj.pt"
    torch.save(out, dest)
    logger.info("wrote %s  (%.0fs)", dest, time.time() - t0)


if __name__ == "__main__":
    main()
