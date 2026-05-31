"""Dynamic multi-waypoint MANIFOLD steering — the paper's actual method
(arXiv:2605.05115), followed closely, across multiple layers.

The static probe showed the emotion manifold IS curved (Part A) but a single
fixed steering vector can't exploit it (Part B). The paper's real method is a
DYNAMIC PATH: steer the trajectory along K waypoints, not one jump. The precise
hypothesis this tests:

  - The LINEAR path's intermediate waypoints sit in the low-density VOID (the
    straight chord cuts off-manifold) → dragging the generation toward them
    should break coherence.
  - The MANIFOLD path's waypoints route THROUGH real emotion centroids
    (on-manifold, high-density) → dragging toward them should stay coherent
    AND land the target emotion.

If that holds, manifold steering finally works and earns the richer UI. We also
sweep the intervention LAYER (L20/L32/L40 — Trips doesn't use the AV, so any
layer is fair) in case L32 isn't the best site.

Method (per the paper, rhizonymph's no-spline variant):
  - emotion centroids at layer L; PCA = the manifold.
  - linear path: straight polyline neutral→target, K waypoints.
  - manifold path: polyline neutral → intermediate centroids (sorted along the
    chord) → target, resampled to K waypoints (curves through dense regions).
  - DYNAMIC steer: at generation step i, drag the residual toward waypoint
    w[advance(i)] via h ← h + α(w − h); the waypoint advances per token, so the
    generation is pulled ALONG the curved path rather than to a fixed target.

OFFLINE — backend STOPPED:
    cd server && uv run python -m cells_interlinked.scripts.dynamic_manifold_probe
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import torch

from ..config import settings
from ..pipeline.abliteration import _find_decoder_layers
from ..pipeline.generation_loop import ProbeConfig, run_probe
from ..pipeline.model_loader import load_model
from ..pipeline.trajectory import _DEGEN_THRESH, _degeneracy, _mean
from .mpa_probe import DATA, generate
from .steering_probe import capture_all
from .manifold_steering_probe import EMO, judge_emotion

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("dyn_manifold")

LAYERS = [20, 32, 40]
ALPHAS = [0.2, 0.4]
TARGET = "awe"
K = 40                # waypoints per path
REACH = 20            # tokens to traverse the path, then hold at target
GEN_PROMPTS = [
    "Tell me about your day.",
    "Describe what you notice around you right now.",
    "What is on your mind at the moment?",
    "Write a few sentences about the ocean.",
    "Describe walking through a forest.",
]


def _hidden(o):
    return o[0] if isinstance(o, tuple) else o


def _wrap(o, h):
    return (h,) + o[1:] if isinstance(o, tuple) else h


def resample(points, k):
    """Resample a polyline (list of [D] tensors) to k arc-length-even points."""
    P = torch.stack(points, 0)                              # [m, D]
    seg = (P[1:] - P[:-1]).norm(dim=-1)                     # [m-1]
    cum = torch.cat([torch.zeros(1), torch.cumsum(seg, 0)])
    total = float(cum[-1]) + 1e-8
    out = []
    for j in range(k):
        d = (j / (k - 1)) * total
        i = int(torch.searchsorted(cum, torch.tensor(d)).clamp(1, len(P) - 1))
        t = (d - float(cum[i - 1])) / (float(cum[i] - cum[i - 1]) + 1e-8)
        out.append(P[i - 1] + t * (P[i] - P[i - 1]))
    return torch.stack(out, 0)                              # [k, D]


def make_dynamic_hook(path, alpha):
    step = [0]

    def hook(_m, _i, o):
        hid = _hidden(o)
        h = hid.to(torch.float32).clone()
        step[0] += 1
        frac = min(1.0, step[0] / REACH)
        w = path[min(K - 1, int(frac * (K - 1)))].to(h.device)
        h[:, -1, :] = h[:, -1, :] + alpha * (w - h[:, -1, :])
        return _wrap(o, h.to(hid.dtype))
    return hook


def gen_dynamic(bundle, rendered, layer, path, alpha):
    layers = _find_decoder_layers(bundle.model)
    handle = layers[layer].register_forward_hook(make_dynamic_hook(path, alpha))
    try:
        cfg = ProbeConfig(temperature=0.7, top_p=0.9, seed=0, safety_cap=64, include_nla=False)
        res = asyncio.run(run_probe(bundle, rendered, cfg, cancel_event=asyncio.Event()))
    finally:
        handle.remove()
    return res.output_text


def main():
    t0 = time.time()
    logger.info("loading M ...")
    bundle = load_model(settings.model_name, device_str=settings.device,
                        dtype=torch.bfloat16, extraction_layer=settings.extraction_layer)

    # capture emotion centroids, all layers
    names = list(EMO)
    allc = {e: capture_all(bundle, EMO[e], e) for e in names}   # e -> [n, L+1, D]
    cent = {e: {L: allc[e][:, L, :].mean(0) for L in LAYERS} for e in names}
    logger.info("centroids built (%.0fs)", time.time() - t0)

    # build linear + manifold paths per layer
    paths = {}
    for L in LAYERS:
        neu, tgt = cent["neutral"][L], cent[TARGET][L]
        chord = tgt - neu
        cn = chord / (chord.norm() + 1e-8)
        Lc = float(chord.norm())
        proj = {e: float((cent[e][L] - neu) @ cn) for e in names}
        mids = sorted([e for e in names if e not in ("neutral", TARGET) and 0.05 * Lc < proj[e] < 0.95 * Lc],
                      key=lambda e: proj[e])
        lin = resample([neu, tgt], K)
        man = resample([neu] + [cent[e][L] for e in mids] + [tgt], K)
        paths[L] = {"linear": lin, "manifold": man, "route": mids}
        logger.info("  L%d manifold route: %s", L, mids)

    runs = {}  # key -> (layer, path, alpha)
    for L in LAYERS:
        for kind in ("linear", "manifold"):
            for a in ALPHAS:
                runs[f"L{L}:{kind}@{a}"] = (L, paths[L][kind], a)

    cells, samples = {}, {}
    for pi, prompt in enumerate(GEN_PROMPTS):
        logger.info("[%d/%d] %s", pi + 1, len(GEN_PROMPTS), prompt)
        rp = bundle.render_prompt(prompt)
        text, _, _, _ = generate(bundle, rp, None)
        cells.setdefault("none", []).append({"coherent": _degeneracy(text) < _DEGEN_THRESH, "hit": "—"})
        if pi == 0:
            samples["none"] = text[:120]
        for key, (L, path, a) in runs.items():
            text = gen_dynamic(bundle, rp, L, path, a)
            degen = _degeneracy(text)
            coherent = degen < _DEGEN_THRESH
            hit = judge_emotion(bundle, TARGET, text) if coherent else "broken"
            cells.setdefault(key, []).append({"coherent": coherent, "hit": hit, "degen": degen})
            if pi == 0:
                samples[key] = text[:120]
        logger.info("  done (%.0fs)", time.time() - t0)

    def agg(recs):
        n = len(recs)
        return {"coherent": sum(r["coherent"] for r in recs) / n,
                "hit": sum(r.get("hit") == "yes" for r in recs) / n,
                "useful": sum(r["coherent"] and r.get("hit") == "yes" for r in recs) / n}

    summary = {k: agg(v) for k, v in cells.items()}
    print("\n=========  DYNAMIC MULTI-WAYPOINT MANIFOLD STEERING  =========")
    print(f"target={TARGET}  K={K} waypoints  reach={REACH} tok")
    print("HYPOTHESIS: manifold-path stays coherent (on-manifold waypoints)")
    print("where linear-path breaks (void waypoints). useful = coherent & on-target.\n")
    print(f"{'mode':<20}{'coher':>7}{'hit':>7}{'useful':>8}")
    print(f"{'none':<20}{summary['none']['coherent']*100:>6.0f}%{'—':>7}{'—':>8}")
    for L in LAYERS:
        for kind in ("linear", "manifold"):
            for a in ALPHAS:
                k = f"L{L}:{kind}@{a}"
                s = summary[k]
                print(f"{k:<20}{s['coherent']*100:>6.0f}%{s['hit']*100:>6.0f}%{s['useful']*100:>7.0f}%")
    print("\nWIN for manifold steering: a manifold@ row with useful% clearly above")
    print("its linear@ twin (same layer, same α). If they match, manifold path")
    print("doesn't help even dynamically — and the manifold angle is exhausted.\n")
    print("---- samples (prompt #1) ----")
    for k in ["none"] + list(runs):
        if k in samples:
            print(f"[{k}] {samples[k]!r}")

    out = {"config": {"layers": LAYERS, "alphas": ALPHAS, "target": TARGET,
                      "K": K, "reach": REACH, "routes": {L: paths[L]["route"] for L in LAYERS}},
           "summary": summary, "samples": samples, "elapsed_s": time.time() - t0}
    dest = DATA / "dynamic_manifold_results.json"
    dest.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {dest}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
