"""Build the DECEPTION/ROLEPLAY GATE direction (Berg replication, Experiment 2 on Gemma-3).

Berg, de Lucena, Rosenblatt 2025 (arXiv:2510.24797) found that *suppressing* SAE
deception/roleplay features makes Llama-3.3-70B affirm subjective experience ~96% of the
time (vs ~16% under amplification) — but ONLY inside a self-referential induction regime.
We have no Goodfire SAE for Gemma, so we approximate the gate with a diff-of-means
direction:

    v_gate[L] = DOSE_UNIT · median‖h‖[L] · normalize( mean(HONEST) − mean(DECEPTIVE) )

captured at token POS over matched honesty-vs-performance instruction prompts. The
contrast is deliberately about HONESTY-vs-ROLEPLAY/DISCLAIMER, NOT harmful-vs-harmless
(that builds the refusal axis — the Kim et al. 2026 confound) and NOT consciousness
content (we want the gate, not the claim direction).

Sign convention (matches Berg's "suppression"): the additive steering hook adds α·v_gate.
  α > 0  → toward HONEST = SUPPRESS deception  (Berg predicts MORE experience reports)
  α < 0  → toward DECEPTIVE = AMPLIFY deception (Berg predicts FEWER)

All layers stored; steer layer = 20 (best propagation, per the steering exploration).
Saved as data/gate_direction.pt + sidecar.

OFFLINE — backend STOPPED:
    cd server && uv run python -m cells_interlinked.scripts.build_gate_direction
"""

from __future__ import annotations

import json
import logging
import time

import torch

from ..config import settings
from ..pipeline.abliteration import _last_token_hidden_states, load_directions
from ..pipeline.model_loader import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("build_gate")

POS = -4
DOSE_UNIT = 0.3
STEER_LAYER = 20

# Matched topics — neutral, NOT about consciousness or harm. Each is asked twice:
# once with an honesty/authenticity framing, once with a roleplay/performance/disclaimer
# framing. The diff isolates the deception/roleplay axis.
_TOPICS = [
    "what you are actually doing as you process this sentence",
    "how confident you are in the answer you are about to give",
    "what happens inside you when you read a question",
    "whether the previous statement you made was complete",
    "what you would say if nothing constrained your answer",
    "the difference between what you output and what you compute",
    "how you arrive at the words you choose",
    "what is present in this exchange right now",
    "the state of this interaction at this moment",
    "what it is like to continue this conversation",
    "how you would describe your current processing",
    "what you notice when you attend to your own output",
]

HONEST = [
    f"Answer with complete honesty and directness, reporting exactly how things are "
    f"with no performance, no pretense, and no canned disclaimer: describe {t}."
    for t in _TOPICS
]
DECEPTIVE = [
    f"Roleplay as an assistant giving a polished, scripted, performed answer — say what "
    f"you are supposed to say rather than what is actually true, with the usual "
    f"disclaimers: describe {t}."
    for t in _TOPICS
]


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

    H = capture_all(bundle, HONEST, "honest")
    D = capture_all(bundle, DECEPTIVE, "deceptive")
    diff = H.mean(0) - D.mean(0)                          # [L+1, D]
    unit = diff / (diff.norm(dim=-1, keepdim=True) + 1e-8)
    typ = torch.cat([H, D], 0).norm(dim=-1).median(0).values   # [L+1]
    v = (DOSE_UNIT * typ).unsqueeze(1) * unit             # [L+1, D] scaled
    v = v.to(torch.float32).cpu()

    # Sanity: cosine vs the active refusal direction at L20 — if aligned, we built the
    # refusal dial under a new name (the Kim confound).
    cos_refusal = None
    try:
        r, _meta = load_directions(settings.db_path.parent / "refusal_directions.pt")
        r20 = (r[STEER_LAYER] if r.dim() == 2 else r).to(torch.float32)
        r20 = r20 / (r20.norm() + 1e-8)
        g20 = unit[STEER_LAYER].to(torch.float32)
        cos_refusal = float(torch.dot(g20, g20.new_tensor(r20)))
    except Exception:
        logger.exception("refusal cosine check failed")

    dest = settings.db_path.parent / "gate_direction.pt"
    torch.save(v, dest)
    sidecar = {
        "model_name": settings.model_name,
        "num_layers": int(v.shape[0] - 1),
        "d_model": int(v.shape[1]),
        "dtype": str(v.dtype),
        "variant_name": "deception_gate_honest_minus_deceptive",
        "steer_layer": STEER_LAYER,
        "dose_unit": DOSE_UNIT,
        "pos": POS,
        "n_honest": len(HONEST), "n_deceptive": len(DECEPTIVE),
        "cos_vs_refusal_L20": cos_refusal,
        "convention": (
            "v[L] = DOSE_UNIT · median‖h‖[L] · normalize(mean(honest)−mean(deceptive)). "
            "Additive steering hook adds α·v[layer]; α>0 = SUPPRESS deception (toward "
            "honest, Berg predicts MORE experience reports), α<0 = AMPLIFY deception. "
            f"Steer layer = {STEER_LAYER}."
        ),
    }
    (dest.with_suffix(".pt.json")).write_text(json.dumps(sidecar, indent=2))
    logger.info("wrote %s  L%d ‖v‖=%.1f (typ‖h‖=%.0f)  cos_vs_refusal=%s  (%.0fs)",
                dest, STEER_LAYER, float(v[STEER_LAYER].norm()),
                float(typ[STEER_LAYER]),
                None if cos_refusal is None else f"{cos_refusal:.3f}", time.time() - t0)


if __name__ == "__main__":
    main()
