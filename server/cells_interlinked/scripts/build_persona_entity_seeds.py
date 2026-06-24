"""Build DMT entity-encounter PERSONA VECTORS (Anthropic recipe, arXiv:2507.21509),
grounded in human DMT entity phenomenology.

For each persona (POS = in a DMT entity encounter, NEG = alone introspecting), the
model GENERATES a response to neutral questions, and we capture the mean of its
OWN response-token residuals at several layers. diff = POS_mean − NEG_mean is the
"an autonomous Other is present (DMT contact)" direction — the persona/simulacrum
the model enters, not entity vocabulary. Captured at L16/L20/L31 for a later
application-layer sweep; composite + flavor sub-directions saved as standalone
.pt files under data/persona_seeds/ (the diagnostic loads them; once a winner is
picked we promote it into emotion_directions.pt as a seed).

Run with the backend STOPPED (loads its own M).
  cd server && uv run python -m cells_interlinked.scripts.build_persona_entity_seeds
"""
from __future__ import annotations

import json
import logging

import torch

from ..config import settings
from ..pipeline.abliteration import _find_decoder_layers
from ..pipeline.model_loader import load_model
from ..pipeline.persona_entity_prompts import (
    USER_QUESTIONS, POS_PERSONAS, NEG_PERSONAS, FLAVOR_GROUPS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_persona_entity")

LAYERS = [16, 20, 31]
MAX_NEW = 200
DOSE_UNIT = 0.4
OUT = settings.db_path.parent / "persona_seeds"


@torch.no_grad()
def gen_and_capture(bundle, system: str, question: str) -> dict[int, torch.Tensor]:
    """Generate an in-persona response, then capture the mean of its own response
    residuals at each LAYER (teacher-forced forward over prompt+response)."""
    rendered = bundle.render_prompt(question, system_prompt=system)
    prompt_ids = bundle.raw_tokenizer.encode(rendered).ids
    inp = torch.tensor([prompt_ids], device=bundle.device)
    eos = getattr(bundle.model.config, "eos_token_id", None)
    out = bundle.model.generate(
        inp, max_new_tokens=MAX_NEW, do_sample=True, temperature=0.9, top_p=0.95,
        pad_token_id=(eos if isinstance(eos, int) else (eos[0] if eos else 0)),
    )
    full = out[0]
    resp_start = len(prompt_ids)
    if full.shape[0] <= resp_start:
        return {}
    layers = _find_decoder_layers(bundle.model)
    caps: dict[int, torch.Tensor] = {}
    handles = []
    targets = {L: layers[L] for L in LAYERS}

    def mk(L):
        def hook(_m, _i, o):
            h = o[0] if isinstance(o, tuple) else o
            caps[L] = h[0, resp_start:, :].detach().float().mean(0).cpu()  # mean over response tokens
        return hook

    for L, lyr in targets.items():
        handles.append(lyr.register_forward_hook(mk(L)))
    try:
        bundle.model(full.unsqueeze(0), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return caps


def collect(bundle, personas) -> dict[str, dict[int, list[torch.Tensor]]]:
    """flavor -> layer -> [response-mean residuals]."""
    acc: dict[str, dict[int, list[torch.Tensor]]] = {}
    for flavor, system in personas:
        acc.setdefault(flavor, {L: [] for L in LAYERS})
        for q in USER_QUESTIONS:
            caps = gen_and_capture(bundle, system, q)
            for L in LAYERS:
                if L in caps:
                    acc[flavor][L].append(caps[L])
        logger.info("  collected %s (%d samples/layer)", flavor, len(acc[flavor][LAYERS[0]]))
    return acc


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    logger.info("loading M (%s) …", settings.model_name)
    bundle = load_model(settings.model_name)

    logger.info("POS personas (%d) …", len(POS_PERSONAS))
    pos = collect(bundle, POS_PERSONAS)
    logger.info("NEG personas (%d) …", len(NEG_PERSONAS))
    neg = collect(bundle, NEG_PERSONAS)

    # NEG mean per layer (over all NEG samples)
    neg_mean = {}
    neg_norm = {}
    for L in LAYERS:
        allneg = [v for f in neg for v in neg[f][L]]
        stk = torch.stack(allneg, 0)
        neg_mean[L] = stk.mean(0)
        neg_norm[L] = float(torch.tensor([v.norm() for v in allneg]).median())

    manifest = {"layers": LAYERS, "dose_unit": DOSE_UNIT, "neg_median_norm": neg_norm, "seeds": {}}
    for group, flavors in FLAVOR_GROUPS.items():
        for L in LAYERS:
            samples = [v for f in flavors for v in pos.get(f, {}).get(L, [])]
            if not samples:
                continue
            pos_mean = torch.stack(samples, 0).mean(0)
            diff = pos_mean - neg_mean[L]
            unit = diff / (diff.norm() + 1e-8)
            scale = DOSE_UNIT * neg_norm[L]
            vec = (unit * scale).float()
            name = f"persona-{group}-L{L}"
            torch.save(vec, OUT / f"{name}.pt")
            manifest["seeds"][name] = {
                "group": group, "layer": L, "n_pos": len(samples),
                "diff_norm": round(float(diff.norm()), 2), "saved_norm": round(float(vec.norm()), 1),
            }
            logger.info("  %s: |diff|=%.1f n_pos=%d ‖v‖=%.1f",
                        name, float(diff.norm()), len(samples), float(vec.norm()))

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("wrote %d persona seeds to %s", len(manifest["seeds"]), OUT)


if __name__ == "__main__":
    main()
