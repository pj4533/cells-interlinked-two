"""Append the UNCHARTED directions to the dose palette so they're selectable on
/trip — transparently labelled as non-human-readable directions, NOT emotions.

These are the directions orthogonal to the named-emotion subspace that the token
head renders as gibberish but that the NLA decoder reads as real, structured,
direction-specific states (see interface_swap_probe.py + docs/MANIFOLD_ABLATION).
They are NOT emotions: both the model's token head and the AV are language
renderers, and what's orthogonal to the named-emotion frame reads as off-manifold
topic/genre structure, not feeling. We give them Blade-Runner names ("things you
people wouldn't believe… lost in time, like tears in rain") because they're
exactly that — internal states the model can't put into words.

We append them to data/emotion_directions.pt (zero-filled except at the steer
layer; the steering hook only reads STEER_LAYER) and record them under the
sidecar "uncharted" key so the UI can group + caveat them. Scaled a bit stronger
than the named palette (so the off-manifold / collapse regime is reachable
within the α sweep — the whole point is to SEE the unrenderable character).

OFFLINE — backend STOPPED (loads M):
    cd server && uv run python -m cells_interlinked.scripts.build_uncharted_directions
Then restart the backend; the names appear in the /trip dose picker.
"""

from __future__ import annotations

import json
import logging
import time

import torch

from ..config import settings
from ..pipeline.abliteration import gram_schmidt
from ..pipeline.model_loader import load_model
from ..pipeline.emotion_prompts import EMO, EXTRA, NAMED, capture_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("build_uncharted")

STEER_LAYER = 20
N_UNCHARTED = 4
UNCH_UNIT = 0.4   # a touch over the 0.3 named palette: coherent at low α, collapse at high α
                  # (0.7 collapsed even α=0.5 → no runway; tune here if needed)

# Blade-Runner names — moments that can't be communicated, lost like tears in rain.
NAMES = ["tears-in-rain", "c-beams", "tannhauser", "orion"]


def main():
    t0 = time.time()
    data = settings.db_path.parent
    edir = torch.load(data / "emotion_directions.pt", weights_only=False)  # [E,49,D] f32
    sidecar = json.loads((data / "emotion_directions.pt.json").read_text())
    if "uncharted" in sidecar and any(n in sidecar["emotions"] for n in NAMES):
        logger.warning("uncharted directions already present in palette — rebuilding them")
        # drop existing uncharted rows so this script is idempotent
        keep = [i for i, e in enumerate(sidecar["emotions"]) if e not in NAMES]
        edir = edir[keep]
        sidecar["emotions"] = [sidecar["emotions"][i] for i in keep]
    E, L1, D = edir.shape
    logger.info("existing palette: %d entries, layers=%d, d_model=%d", E, L1, D)

    logger.info("loading M ...")
    bundle = load_model(settings.model_name, device_str=settings.device,
                        dtype=torch.bfloat16, extraction_layer=settings.extraction_layer)

    # affective cloud + named subspace at L20 → uncharted candidate dirs (as the probes)
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
            cands.append((vp / (vp.norm() + 1e-8)).to(torch.float32).cpu())
        if len(cands) >= N_UNCHARTED:
            break
    logger.info("named rank=%d, %d uncharted dirs, typ_L20=%.1f, scale=%.2f·typ",
                Bnamed.shape[0], len(cands), typ, UNCH_UNIT)

    # build [N,49,D] rows: zero everywhere, scaled dir at the steer layer
    rows = torch.zeros(len(cands), L1, D, dtype=torch.float32)
    for j, c in enumerate(cands):
        rows[j, STEER_LAYER] = (UNCH_UNIT * typ) * c
    new_edir = torch.cat([edir, rows], dim=0)

    new_names = list(sidecar["emotions"]) + NAMES[:len(cands)]
    sidecar["emotions"] = new_names
    sidecar["uncharted"] = NAMES[:len(cands)]
    sidecar["uncharted_dose_unit"] = UNCH_UNIT
    sidecar["uncharted_note"] = (
        "Non-human-readable directions: orthogonal to the named-emotion subspace, "
        "renderable by the NLA decoder but NOT by the token head — off-manifold "
        "structure, not emotions. Scaled at uncharted_dose_unit·median‖h‖[L20]. "
        "Blade-Runner-named (tears in rain). See docs/MANIFOLD_ABLATION.md.")

    torch.save(new_edir, data / "emotion_directions.pt")
    (data / "emotion_directions.pt.json").write_text(json.dumps(sidecar, indent=2))
    logger.info("wrote palette: %d entries → %s  (uncharted: %s)  %.0fs",
                new_edir.shape[0], new_names, sidecar["uncharted"], time.time() - t0)


if __name__ == "__main__":
    main()
