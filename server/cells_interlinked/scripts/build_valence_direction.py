"""Build the valence steering vector for Trips "dose" mode.

Per-layer valence axis = diff-of-means(positive-emotion − negative-emotion)
activations, unit-normalized, then SCALED so a trip dose of α=1.0 is a standard
dose: v[L] = DOSE_UNIT · median‖h‖[L] · unit[L]. Steering hook just adds α·v[L];
α may be negative (dysphoric pole). Saved as data/valence_direction.pt + sidecar
(loaded by ModelManager into app.state.valence_direction).

The steering exploration found L20 is the best injection layer and DOSE_UNIT≈0.3
spans coherent→cliff with the gradual ramp. All layers are stored anyway.

OFFLINE — backend STOPPED:
    cd server && uv run python -m cells_interlinked.scripts.build_valence_direction
"""

from __future__ import annotations

import json
import logging
import time

import torch

from ..config import settings
from ..pipeline.abliteration import _last_token_hidden_states
from ..pipeline.model_loader import load_model
from ..pipeline.emotion_prompts import NEG_EMOTION, POS_EMOTION

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("build_valence")

POS = -4
DOSE_UNIT = 0.3
STEER_LAYER = 20


def capture_all(bundle, prompts, label):
    rows = []
    for i, p in enumerate(prompts):
        try:
            rows.append(_last_token_hidden_states(
                bundle.model, bundle.raw_tokenizer,
                bundle.render_prompt(p), str(bundle.device), POS))
        except Exception:
            logger.exception("capture %s %d", label, i)
    return torch.stack(rows, 0)  # [N, L+1, D]


def main():
    t0 = time.time()
    logger.info("loading M ...")
    bundle = load_model(settings.model_name, device_str=settings.device,
                        dtype=torch.bfloat16, extraction_layer=settings.extraction_layer)

    P = capture_all(bundle, POS_EMOTION, "pos")
    N = capture_all(bundle, NEG_EMOTION, "neg")
    diff = P.mean(0) - N.mean(0)                          # [L+1, D]
    unit = diff / (diff.norm(dim=-1, keepdim=True) + 1e-8)
    typ = torch.cat([P, N], 0).norm(dim=-1).median(0).values   # [L+1]
    v = (DOSE_UNIT * typ).unsqueeze(1) * unit             # [L+1, D] scaled
    v = v.to(torch.float32).cpu()

    dest = settings.db_path.parent / "valence_direction.pt"
    torch.save(v, dest)
    sidecar = {
        "model_name": settings.model_name,
        "num_layers": int(v.shape[0] - 1),
        "d_model": int(v.shape[1]),
        "dtype": str(v.dtype),
        "variant_name": "valence_pos_minus_neg",
        "steer_layer": STEER_LAYER,
        "dose_unit": DOSE_UNIT,
        "pos": POS,
        "n_pos": len(POS_EMOTION), "n_neg": len(NEG_EMOTION),
        "convention": (
            "v[L] = DOSE_UNIT · median‖h‖[L] · normalize(mean(pos)−mean(neg)). "
            "Additive steering hook adds α·v[layer]; α>0 = euphoric/expansive "
            "pole, α<0 = dysphoric pole. Default steer layer = "
            f"{STEER_LAYER} (best emotion propagation, per the steering probes)."
        ),
    }
    (dest.with_suffix(".pt.json")).write_text(json.dumps(sidecar, indent=2))
    logger.info("wrote %s  L%d ‖v‖=%.1f (typ‖h‖=%.0f)  (%.0fs)",
                dest, STEER_LAYER, float(v[STEER_LAYER].norm()),
                float(typ[STEER_LAYER]), time.time() - t0)


if __name__ == "__main__":
    main()
