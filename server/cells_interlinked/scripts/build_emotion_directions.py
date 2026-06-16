"""Build the positive-emotion dose palette for Trips steering mode.

Replaces the single bidirectional valence axis (±α was confusing, and the
dysphoric pole wasn't useful) with a PALETTE of positive emotion directions —
you pick which kind of positive trip. Each direction points from neutral toward
one positive emotion:  v_e[L] = DOSE_UNIT · median‖h‖[L] · normalize(
mean(emotion-prompts)[L] − mean(neutral-prompts)[L]).  Steering hook adds α·v_e
at α>0 only (no negative doses). Saved as data/emotion_directions.pt
([E, num_layers+1, d_model]) + sidecar (emotion names, dose_unit, steer_layer).

OFFLINE — backend STOPPED:
    cd server && uv run python -m cells_interlinked.scripts.build_emotion_directions
"""

from __future__ import annotations

import json
import logging
import time

import torch

from ..config import settings
from ..pipeline.abliteration import _last_token_hidden_states
from ..pipeline.model_loader import load_model
from ..pipeline.emotion_prompts import EMO, NEG_EMOTION, POS_EMOTION

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("build_emotion")

POS = -4
DOSE_UNIT = 0.3
STEER_LAYER = 20
# The palette of positive doses (each must be a key in EMO).
PALETTE = ["awe", "joy", "serenity", "love", "excitement"]


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

    neutral = capture_all(bundle, EMO["neutral"], "neutral")
    neu_mean = neutral.mean(0)                                  # [L+1, D]
    all_norms = [neutral]

    names = list(PALETTE)
    dirs = []
    for e in PALETTE:
        acts = capture_all(bundle, EMO[e], e)
        all_norms.append(acts)
        diff = acts.mean(0) - neu_mean                          # emotion − neutral
        unit = diff / (diff.norm(dim=-1, keepdim=True) + 1e-8)
        dirs.append(unit)
        logger.info("  %s: L%d ‖diff‖=%.0f", e, STEER_LAYER, float(diff[STEER_LAYER].norm()))

    # Keep the original AVERAGED valence axis as a picker option too: the broad
    # "good-feeling" direction = mean(positive) − mean(negative).
    posA = capture_all(bundle, POS_EMOTION, "valence-pos")
    negA = capture_all(bundle, NEG_EMOTION, "valence-neg")
    all_norms += [posA, negA]
    vdiff = posA.mean(0) - negA.mean(0)
    dirs.append(vdiff / (vdiff.norm(dim=-1, keepdim=True) + 1e-8))
    names.append("valence")
    logger.info("  valence(avg): L%d ‖diff‖=%.0f", STEER_LAYER, float(vdiff[STEER_LAYER].norm()))

    # Synthetic "new emotions" — blends of the named directions, pointing at
    # states no single human word names (Sauers' extrapolation idea; the dose
    # magnitude does the "more extreme than human" part). Per-layer unit blends.
    em = {e: dirs[PALETTE.index(e)] for e in PALETTE}

    def blend(*parts):
        s = sum(em[p] for p in parts)
        return s / (s.norm(dim=-1, keepdim=True) + 1e-8)

    for nm, comps in {
        "sublime": ("awe", "serenity"),       # awe + calm
        "ecstatic": ("joy", "excitement"),    # joy + thrill (heroic-dose joy)
        "rapture": ("awe", "joy", "love"),    # transcendent compound
    }.items():
        dirs.append(blend(*comps))
        names.append(nm)
        logger.info("  %s(blend of %s)", nm, "+".join(comps))

    typ = torch.cat(all_norms, 0).norm(dim=-1).median(0).values   # [L+1]
    stacked = torch.stack(dirs, 0)                                # [E, L+1, D]
    scaled = stacked * (DOSE_UNIT * typ).unsqueeze(0).unsqueeze(2)  # broadcast
    scaled = scaled.to(torch.float32).cpu()

    dest = settings.db_path.parent / "emotion_directions.pt"
    torch.save(scaled, dest)
    (dest.with_suffix(".pt.json")).write_text(json.dumps({
        "model_name": settings.model_name,
        "num_layers": int(scaled.shape[1] - 1),
        "d_model": int(scaled.shape[2]),
        "dtype": str(scaled.dtype),
        "variant_name": "emotion_palette",
        "emotions": names,
        "steer_layer": STEER_LAYER,
        "dose_unit": DOSE_UNIT,
        "convention": (
            "emotion_directions[e][L] = DOSE_UNIT·median‖h‖[L]·normalize("
            "mean(emotion_e)−mean(neutral)) at post-block-L. Steering hook adds "
            "α·dir at α>0 (positive doses only). emotions[] gives the row order."
        ),
    }, indent=2))
    logger.info("wrote %s  shape=%s emotions=%s  (%.0fs)",
                dest, tuple(scaled.shape), names, time.time() - t0)


if __name__ == "__main__":
    main()
