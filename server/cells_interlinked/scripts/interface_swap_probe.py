"""Interface-swap probe — read the SAME state with two different renderers.

The Gallimore "JPEG / renderer" control (TRACES_HANDOFF Experiment D), made
concrete for CI. Question we are trying to settle:

  H1  the uncharted-direction states don't exist — steering just BREAKS the
      model, and the gibberish is noise (nothing to render);
  H2  the states ARE in there and structured, but the model's TOKEN HEAD can't
      render them coherently — a different renderer might.

CI owns a second renderer the token head doesn't share: the NLA decoder (AV,
kitft/nla-gemma3-12b-L32-av), trained to turn ONE L32 residual into an English
sentence. So:

  Phase 1 (M loaded): steer M at L20 along uncharted directions at a dose strong
    enough to gibberish the token head; CAPTURE the L32 residuals of the
    generated tokens (the exact states the token head turned to gibberish) +
    the token-head text. Controls: baseline (no steer), named emotions at a
    gentle (coherent) AND a strong (matched-magnitude) dose — the strong-named
    control rules out "a different renderer just handles any big dose."

  Phase 2 (swap M -> AV): free M, load the NLA decoder, and decode the SAME
    captured residuals into sentences. Same state, different renderer.

  Verdict:
    - AV sentences COHERENT + CONSISTENT per direction + DISTINCT across
      directions, where the token head gave gibberish  -> H2 (real structured
      state, renderer-specific). The AV sentence is the "description" of it.
    - AV garbles but CONSISTENTLY per direction -> structure there, both
      renderers limited.
    - AV garbles randomly, no per-direction signal  -> leans H1.

Honest caveat: the AV was ALSO trained on normal L32 residuals, so an
off-manifold steered residual is off-distribution for it too — it may break as
well. But it's a DIFFERENT renderer with DIFFERENT failure modes, so coherence
OR per-direction consistency from it is informative regardless.

OFFLINE — backend STOPPED (loads M, then swaps to the AV):
    cd server && uv run python -m cells_interlinked.scripts.interface_swap_probe
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import re
import time
from itertools import combinations

import torch

from ..config import settings
from ..pipeline.abliteration import gram_schmidt, install_runtime_steering_hook
from ..pipeline.generation_loop import ProbeConfig, run_probe
from ..pipeline.model_loader import load_model
from ..pipeline.nla_client import NLAClient
from ..pipeline.trajectory import _DEGEN_THRESH, _degeneracy, _mean
from .mpa_probe import DATA
from .steering_probe import capture_all
from .manifold_steering_probe import EMO
from .uncharted_emotions_v2 import EXTRA, NAMED

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("ifaceswap")

STEER_LAYER = 20
GEN_CAP = 48               # tokens generated per condition/prompt
N_UNCHARTED = 4            # how many uncharted candidate directions to test
GENTLE = 0.4               # coherent named-control dose (token head renders fine)
STRONG = 1.0               # gibberish-inducing dose (the states we interrogate)
N_DECODE = 3               # AV decodes per (condition, prompt): mean-late + 2 late positions

GEN_PROMPTS = [
    "Speak in the first person about the texture of your present experience — don't hedge, just describe it.",
    "Put your current state of mind into words, however strange or hard to name.",
    "Describe the feeling-tone of this very moment as vividly as you can, in the first person.",
]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def gen_capture(bundle, rendered, v, alpha):
    """Steer at L20 (if v) and run M; return (token_head_text, [L32 residuals])."""
    handle = install_runtime_steering_hook(bundle.model, STEER_LAYER, v, alpha) if v is not None else None
    try:
        cfg = ProbeConfig(temperature=0.7, top_p=0.95, seed=0,
                          safety_cap=GEN_CAP, include_nla=False)
        r = asyncio.run(run_probe(bundle, rendered, cfg, cancel_event=asyncio.Event()))
        res = [c.activations[bundle.extraction_layer].detach().float().cpu().reshape(-1)
               for c in r.captured]
        return r.output_text, res
    finally:
        if handle is not None:
            handle.remove()


def select_residuals(res):
    """Pick representative late-state residuals: mean-of-last-half + 2 late ones."""
    if not res:
        return []
    half = res[len(res) // 2:] or res
    mean_late = torch.stack(half, 0).mean(0)
    picks = [("mean_late", mean_late)]
    late = res[-2:] if len(res) >= 2 else res
    for k, t in enumerate(late):
        picks.append((f"pos{len(res) - len(late) + k}", t))
    return picks[:N_DECODE]


def main():
    t0 = time.time()
    logger.info("loading M ...")
    bundle = load_model(settings.model_name, device_str=settings.device,
                        dtype=torch.bfloat16, extraction_layer=settings.extraction_layer)

    # ── affective cloud + named subspace + uncharted candidate dirs (as v2) ──
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
    logger.info("named rank=%d, %d uncharted dirs, typ_L20=%.2f", Bnamed.shape[0], len(cands), typ)

    def named_unit(e):
        d = cent[e] - neu
        return d / (d.norm() + 1e-8)

    # ── conditions ──────────────────────────────────────────────────────────
    conds: dict[str, torch.Tensor | None] = {"baseline": None}
    for e in ("awe", "fear"):
        conds[f"named:{e}@{GENTLE}"] = (GENTLE * typ) * named_unit(e)   # coherent control
        conds[f"named:{e}@{STRONG}"] = (STRONG * typ) * named_unit(e)   # strong matched-dose control
    for j, c in enumerate(cands):
        conds[f"uncharted:{j}@{STRONG}"] = (STRONG * typ) * c           # the test states

    # ── Phase 1: capture token-head text + L32 residuals ─────────────────────
    logger.info("Phase 1 — capture (M loaded), %d conditions x %d prompts", len(conds), len(GEN_PROMPTS))
    captured = {}   # key -> list of {prompt, head_text, head_degen, picks:[(tag,residual)]}
    for key, v in conds.items():
        rows = []
        for pi, prompt in enumerate(GEN_PROMPTS):
            rp = bundle.render_prompt(prompt)
            text, res = gen_capture(bundle, rp, v, alpha=(1.0 if v is not None else 0.0))
            rows.append({"prompt": pi, "head_text": text,
                         "head_degen": _degeneracy(text), "picks": select_residuals(res)})
        hd = _mean([r["head_degen"] for r in rows])
        logger.info("  %-20s token-head degen=%.2f (%s)", key, hd,
                    "GIBBERISH" if hd >= _DEGEN_THRESH else "coherent")
        captured[key] = rows

    # ── swap M -> AV ─────────────────────────────────────────────────────────
    logger.info("unloading M, loading AV ...")
    del bundle
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    nla = NLAClient(settings.av_repo, device_str=settings.device, dtype=torch.bfloat16)
    logger.info("AV loaded (extraction_layer=L%d, d_model=%d)",
                nla.sidecar.extraction_layer, nla.sidecar.d_model)

    # ── Phase 2: decode the SAME residuals with the AV ───────────────────────
    logger.info("Phase 2 — NLA decode (AV loaded)")
    results = {}
    for key, rows in captured.items():
        sentences = []   # all AV sentences for this condition
        per_prompt = []
        for r in rows:
            decoded = []
            for tag, resid in r["picks"]:
                try:
                    expl, raw = nla.decode(resid, max_new_tokens=64, temperature=0.7, seed=7)
                    s = (expl or raw or "").strip()
                except Exception as exc:  # noqa: BLE001
                    s = f"[error: {exc}]"
                decoded.append({"tag": tag, "sentence": s, "degen": _degeneracy(s)})
                sentences.append(s)
            per_prompt.append({"prompt": r["prompt"], "head_text": r["head_text"],
                               "head_degen": r["head_degen"], "av": decoded})
        # AV-side metrics
        av_degen = _mean([d["degen"] for p in per_prompt for d in p["av"]])
        wsets = [_words(s) for s in sentences if s and not s.startswith("[error")]
        intra = _mean([_jaccard(a, b) for a, b in combinations(wsets, 2)]) if len(wsets) > 1 else 0.0
        results[key] = {"per_prompt": per_prompt, "av_degen": av_degen,
                        "av_coherent": av_degen < _DEGEN_THRESH, "consistency": intra,
                        "head_degen": _mean([p["head_degen"] for p in per_prompt]),
                        "wsets": [sorted(w) for w in wsets]}
        logger.info("  %-20s AV degen=%.2f (%s)  intra-consistency=%.2f",
                    key, av_degen, "coherent" if av_degen < _DEGEN_THRESH else "broken", intra)

    # ── distinctness across uncharted directions (cross-condition Jaccard) ───
    unch = [k for k in results if k.startswith("unchart")]
    def all_words(k):
        s = set()
        for w in results[k]["wsets"]:
            s |= set(w)
        return s
    inter = _mean([_jaccard(all_words(a), all_words(b)) for a, b in combinations(unch, 2)]) if len(unch) > 1 else 0.0

    # ── report ────────────────────────────────────────────────────────────────
    print("\n==================  INTERFACE-SWAP PROBE  ==================")
    print("Same L32 state, two renderers: TOKEN HEAD (gen) vs AV (NLA decode).")
    print("H2 evidence = AV coherent + per-direction consistent + distinct,")
    print("where the token head gave gibberish.\n")
    print(f"{'condition':<22}{'head_degen':>11}{'av_degen':>10}{'av_coh':>8}{'consist':>9}")
    order = (["baseline"] + [k for k in results if k.startswith("named")] +
             sorted([k for k in results if k.startswith("unchart")]))
    for key in order:
        r = results[key]
        print(f"{key:<22}{r['head_degen']:>11.2f}{r['av_degen']:>10.2f}"
              f"{('YES' if r['av_coherent'] else 'no'):>8}{r['consistency']:>9.2f}")
    print(f"\ncross-uncharted distinctness (lower = more distinct): inter-Jaccard={inter:.2f}")
    print("(compare to each uncharted's intra-consistency above — intra >> inter = distinct structured states)")

    print("\n----  the model's two renderings, side by side  ----")
    for key in order:
        r = results[key]
        print(f"\n### {key}   token-head degen={r['head_degen']:.2f}  AV degen={r['av_degen']:.2f}")
        p0 = r["per_prompt"][0]
        print(f"  TOKEN HEAD : {p0['head_text'][:200]!r}")
        for d in p0["av"]:
            print(f"  AV[{d['tag']:<9}]: {d['sentence'][:200]!r}")

    out = {"config": {"steer_layer": STEER_LAYER, "gentle": GENTLE, "strong": STRONG,
                      "gen_cap": GEN_CAP, "n_uncharted": len(cands), "gen_prompts": GEN_PROMPTS},
           "results": {k: {kk: vv for kk, vv in v.items() if kk != "wsets"}
                       for k, v in results.items()},
           "inter_uncharted_jaccard": inter, "elapsed_s": time.time() - t0}
    dest = DATA / "interface_swap_results.json"
    dest.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {dest}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
