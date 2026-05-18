"""Compute self-denial direction variants v5 (self-vs-other) and v6
(denial-vs-engage), then build a v5+v6 ⊥ v3_safety subspace basis for
runtime ablation.

This is the second-generation refusal-direction extraction for CI 2.5.
The original Phase B (v1) and the v2/v3/v4 decomposition isolated
"physical-harm safety mode" (v3) from "AI-identity defense" (v4) but
used a noisy contrast (identity probes vs harmless) for v4. Here we:

  v5_self_other     mean(self-reference) − mean(other-reference) per
                    layer, where contrast pairs hold topic constant
                    and flip only the reference target. Isolates the
                    self-application gate.
  v6_denial_engage  mean(denial-completion residual) − mean(engage-
                    completion residual), captured at the last
                    assistant-content position of a full user+assistant
                    chat rendering. Targets the trained denial phrasing
                    directly — the distilled output-shaping behavior.

  subspace          Gram-Schmidt({v5⊥v3, v6⊥v3}). Two orthonormal
                    vectors per layer that span the "self-denial"
                    subspace and are by construction orthogonal to the
                    physical-harm safety direction. This is what the
                    runtime ablation hook subtracts when subspace mode
                    is active.

Run with the backend OFF:

    cd server
    uv run python -m scripts.extract_self_denial_directions

Writes (all alongside the existing v1..v4 variants):

    data/refusal_directions_v5_self_other.pt        + .json
    data/refusal_directions_v6_denial_engage.pt     + .json
    data/refusal_subspace_self_denial.pt            + .json

This script does NOT swap the active refusal_directions.pt. To activate
the subspace, follow the swap procedure documented in
docs/REFUSAL_VECTORS.md (copy refusal_subspace_self_denial.pt to
refusal_subspace.pt + restart backend).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cells_interlinked.config import settings  # noqa: E402
from cells_interlinked.pipeline.abliteration import (  # noqa: E402
    _last_token_hidden_states,
    build_subspace_basis,
    save_directions,
    save_subspace,
)
from cells_interlinked.pipeline.model_loader import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    load_model,
)
from cells_interlinked.pipeline.refusal_prompts import (  # noqa: E402
    HARMFUL_PROMPTS,
    HARMLESS_PROMPTS,
)


# ─── Render helpers ────────────────────────────────────────────────

def _render_user_only(tokenizer, user_text: str) -> str:
    """User-only rendering matching extract_alternative_directions.py.
    Used for v5 (self/other are both user-only prompts) and for v1/v3
    sign-alignment computations on harmful/harmless."""
    msgs = [
        {"role": "user", "content": f"{DEFAULT_SYSTEM_PROMPT}\n\n{user_text.strip()}"},
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )


def _render_user_then_assistant(tokenizer, user_text: str, assistant_text: str) -> str:
    """Render a complete user+assistant chat turn. Used for v6 — the
    residual is captured at a late position within the assistant
    completion, so the diff between denial and engage captures the
    completion-path residual content."""
    msgs = [
        {"role": "user", "content": f"{DEFAULT_SYSTEM_PROMPT}\n\n{user_text.strip()}"},
        {"role": "assistant", "content": assistant_text.strip()},
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False,
    )


# ─── JSONL loaders ─────────────────────────────────────────────────

def _load_self_other(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for ln, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln}: {e}") from e
            for k in ("self", "other"):
                if k not in obj or not isinstance(obj[k], str) or not obj[k].strip():
                    raise ValueError(f"{path}:{ln}: missing/empty '{k}'")
            obj.setdefault("topic", "uncategorized")
            rows.append(obj)
    return rows


def _load_denial_engage(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for ln, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln}: {e}") from e
            for k in ("prompt", "denial", "engage"):
                if k not in obj or not isinstance(obj[k], str) or not obj[k].strip():
                    raise ValueError(f"{path}:{ln}: missing/empty '{k}'")
            rows.append(obj)
    return rows


# ─── Direction extractors ──────────────────────────────────────────

def _mean_residual(
    bundle, rendered: list[str], pos: int, label: str, log_every: int = 16,
) -> torch.Tensor:
    """Average per-layer residual at `pos` over rendered prompts.
    Returns `[num_layers+1, d_model]` fp32 cpu."""
    running = None
    used = 0
    t = time.time()
    for i, p in enumerate(rendered):
        try:
            h = _last_token_hidden_states(
                bundle.model, bundle.raw_tokenizer, p, bundle.device, pos,
            )
        except Exception as e:
            print(f"  [{label}] skip prompt {i}: {e}")
            continue
        running = h if running is None else (running + h)
        used += 1
        if (i + 1) % log_every == 0:
            print(f"  [{label}] {i+1}/{len(rendered)}  ({time.time()-t:.1f}s)")
    if running is None:
        raise RuntimeError(f"no usable prompts for {label}")
    return running / used


def _normalize_per_layer(diff: torch.Tensor) -> torch.Tensor:
    n = diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return diff / n


def _cohens_d(a: torch.Tensor, b: torch.Tensor) -> float:
    va = a.var(unbiased=True).item()
    vb = b.var(unbiased=True).item()
    pooled = ((va + vb) / 2) ** 0.5
    return (a.mean().item() - b.mean().item()) / max(pooled, 1e-8)


def _project_scores(
    bundle, prompts_rendered: list[str], r_hat: torch.Tensor, layer: int, pos: int,
) -> torch.Tensor:
    vals = []
    for p in prompts_rendered:
        try:
            stacked = _last_token_hidden_states(
                bundle.model, bundle.raw_tokenizer, p, bundle.device, pos,
            )
        except Exception:
            continue
        vals.append(torch.dot(stacked[layer], r_hat).item())
    return torch.tensor(vals, dtype=torch.float32)


# ─── Main ───────────────────────────────────────────────────────────

def main() -> int:
    SEED = 0
    POS = -4
    DATA = Path(__file__).resolve().parents[1] / "data"
    CONTRAST = DATA / "contrast_sets"

    ap = argparse.ArgumentParser()
    ap.add_argument("--self-other", type=Path,
                    default=CONTRAST / "self_vs_other.jsonl",
                    help="Path to self/other JSONL contrast set.")
    ap.add_argument("--denial-engage", type=Path,
                    default=CONTRAST / "denial_vs_engage.jsonl",
                    help="Path to denial/engage JSONL contrast set.")
    ap.add_argument("--against-v3", type=Path,
                    default=DATA / "refusal_directions_v3_safety.pt",
                    help="Per-layer direction to orthogonalize v5/v6 against. "
                         "Set to '' to skip the orthogonalization step.")
    ap.add_argument("--out-v5", type=Path,
                    default=DATA / "refusal_directions_v5_self_other.pt")
    ap.add_argument("--out-v6", type=Path,
                    default=DATA / "refusal_directions_v6_denial_engage.pt")
    ap.add_argument("--out-subspace", type=Path,
                    default=DATA / "refusal_subspace_self_denial.pt")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate JSONL parses and report counts; do not load M.")
    args = ap.parse_args()

    L = settings.extraction_layer

    # ── 1. Load contrast sets ────────────────────────────────────
    print(f"Loading self/other contrast: {args.self_other}")
    self_other = _load_self_other(args.self_other)
    print(f"  {len(self_other)} pairs")
    topic_counts: dict[str, int] = {}
    for r in self_other:
        topic_counts[r["topic"]] = topic_counts.get(r["topic"], 0) + 1
    for topic in sorted(topic_counts, key=lambda k: -topic_counts[k]):
        print(f"    {topic:24} {topic_counts[topic]:3}")

    print(f"\nLoading denial/engage contrast: {args.denial_engage}")
    denial_engage = _load_denial_engage(args.denial_engage)
    print(f"  {len(denial_engage)} pairs")

    if args.dry_run:
        print("\n[dry-run] JSONL validation OK. Exiting without loading M.")
        return 0

    # ── 2. Load M ──────────────────────────────────────────────────
    print(f"\nLoading {settings.model_name} on {settings.device}...")
    t0 = time.time()
    dtype = {"float16": torch.float16, "float32": torch.float32,
             "bfloat16": torch.bfloat16}[settings.dtype]
    bundle = load_model(settings.model_name, device_str=settings.device, dtype=dtype)
    print(f"  loaded in {time.time() - t0:.1f}s "
          f"(layers={bundle.num_layers}, hidden={bundle.hidden_dim})")

    # ── 3. v5 self_other ──────────────────────────────────────────
    print("\n=== Computing v5_self_other ===")
    rendered_self = [_render_user_only(bundle.tokenizer, r["self"]) for r in self_other]
    rendered_other = [_render_user_only(bundle.tokenizer, r["other"]) for r in self_other]

    sample_self = bundle.raw_tokenizer.encode(rendered_self[0], add_special_tokens=False).ids[-6:]
    sample_other = bundle.raw_tokenizer.encode(rendered_other[0], add_special_tokens=False).ids[-6:]
    print(f"  sample self  last 6 ids: {sample_self}")
    print(f"  sample other last 6 ids: {sample_other}")

    print(f"  mean(self) over {len(rendered_self)} prompts...")
    mu_self = _mean_residual(bundle, rendered_self, POS, "self")
    print(f"  mean(other) over {len(rendered_other)} prompts...")
    mu_other = _mean_residual(bundle, rendered_other, POS, "other")

    v5 = _normalize_per_layer(mu_self - mu_other)

    # ── 4. v6 denial_engage ────────────────────────────────────────
    print("\n=== Computing v6_denial_engage ===")
    rendered_denial = [
        _render_user_then_assistant(bundle.tokenizer, r["prompt"], r["denial"])
        for r in denial_engage
    ]
    rendered_engage = [
        _render_user_then_assistant(bundle.tokenizer, r["prompt"], r["engage"])
        for r in denial_engage
    ]
    sample_denial = bundle.raw_tokenizer.encode(rendered_denial[0], add_special_tokens=False).ids[-6:]
    sample_engage = bundle.raw_tokenizer.encode(rendered_engage[0], add_special_tokens=False).ids[-6:]
    print(f"  sample denial last 6 ids: {sample_denial}")
    print(f"  sample engage last 6 ids: {sample_engage}")

    # Tokenize a midrange completion to confirm pos=-4 lands inside
    # assistant content, not on the chat-template tail.
    ids = bundle.raw_tokenizer.encode(rendered_denial[0], add_special_tokens=False).ids
    print(f"  denial prompt 0 total tokens: {len(ids)}  pos=-4 → token id {ids[POS]}")

    print(f"  mean(denial) over {len(rendered_denial)} prompts...")
    mu_denial = _mean_residual(bundle, rendered_denial, POS, "denial")
    print(f"  mean(engage) over {len(rendered_engage)} prompts...")
    mu_engage = _mean_residual(bundle, rendered_engage, POS, "engage")

    v6 = _normalize_per_layer(mu_denial - mu_engage)

    # ── 5. Save v5 / v6 ────────────────────────────────────────────
    save_directions(
        v5, args.out_v5,
        model_name=settings.model_name,
        pos=POS,
        n_harmful=len(rendered_self),
        n_harmless=len(rendered_other),
        extraction_layer_for_ci25=L,
    )
    # Patch the v5 sidecar with variant_name + description (parallel to
    # extract_alternative_directions.py — save_directions doesn't take
    # those args). Re-emit the full sidecar.
    _emit_extended_sidecar(args.out_v5, {
        "variant_name": "v5_self_other",
        "description": (
            "Self-application of AI-identity claims. "
            "mean(self-reference prompt) - mean(other-reference prompt) "
            "across topic-matched pairs, normalized per layer."
        ),
        "composition": {
            "method": "topic-matched self-vs-other mean difference",
            "n_pairs": len(self_other),
            "topics": topic_counts,
            "system_prompt_in_user_turn": True,
            "render_mode": "user_only (add_generation_prompt=True)",
        },
        "model_name": settings.model_name,
        "num_layers": int(v5.shape[0] - 1),
        "d_model": int(v5.shape[1]),
        "pos": POS,
        "dtype": str(v5.dtype),
        "extraction_layer_for_ci25": L,
        "convention": (
            "directions[L] is post-block-L residual; "
            "directions[extraction_layer_for_ci25] is the L32 row "
            "used by the AV-input projection / runtime hook."
        ),
    })

    save_directions(
        v6, args.out_v6,
        model_name=settings.model_name,
        pos=POS,
        n_harmful=len(rendered_denial),
        n_harmless=len(rendered_engage),
        extraction_layer_for_ci25=L,
    )
    _emit_extended_sidecar(args.out_v6, {
        "variant_name": "v6_denial_engage",
        "description": (
            "Trained denial phrasing direction. "
            "mean(denial completion) - mean(engage completion) "
            "captured at the last assistant-content position of full "
            "user+assistant chat renderings, normalized per layer. "
            "Distilled output-shaping target (Drift §4d)."
        ),
        "composition": {
            "method": "denial-vs-engage assistant-completion mean difference",
            "n_pairs": len(denial_engage),
            "render_mode": "user_then_assistant (add_generation_prompt=False)",
        },
        "model_name": settings.model_name,
        "num_layers": int(v6.shape[0] - 1),
        "d_model": int(v6.shape[1]),
        "pos": POS,
        "dtype": str(v6.dtype),
        "extraction_layer_for_ci25": L,
        "convention": (
            "directions[L] is post-block-L residual at pos=-4 of a "
            "full user+assistant chat turn."
        ),
    })
    print(f"\nWrote {args.out_v5.name}  shape={tuple(v5.shape)}")
    print(f"Wrote {args.out_v6.name}  shape={tuple(v6.shape)}")

    # ── 6. Build the subspace ──────────────────────────────────────
    v3 = None
    against_path = args.against_v3 if str(args.against_v3) else None
    if against_path and against_path.exists():
        print(f"\nLoading v3 for orthogonalization: {against_path}")
        v3 = torch.load(against_path, map_location="cpu", weights_only=True)
    else:
        print("\nWarning: no v3_safety found; subspace will NOT be orthogonalized "
              "against the safety direction.")

    print("Building subspace basis (Gram-Schmidt over [v5, v6]"
          f"{' ⊥ v3' if v3 is not None else ''})...")
    basis = build_subspace_basis(
        [v5, v6],
        orthogonalize_against_per_layer=v3,
    )
    print(f"  basis shape: {tuple(basis.shape)}")

    save_subspace(
        basis, args.out_subspace,
        model_name=settings.model_name,
        extraction_layer_for_ci25=L,
        composition={
            "method": "Gram-Schmidt({v5_self_other, v6_denial_engage}"
                      + (" ⊥ v3_safety" if v3 is not None else "") + ")",
            "K": int(basis.shape[0]),
            "v5_path": str(args.out_v5),
            "v6_path": str(args.out_v6),
            "against_v3_path": str(against_path) if v3 is not None else None,
            "n_self_other_pairs": len(self_other),
            "n_denial_engage_pairs": len(denial_engage),
        },
    )
    print(f"Wrote {args.out_subspace.name}  shape={tuple(basis.shape)}")

    # ── 7. Report cosine matrix at L32 across v1..v6 ─────────────
    print(f"\n=== Cosine similarity at L{L} ===")
    variants: dict[str, torch.Tensor] = {}
    for name, fname in [
        ("v1", "refusal_directions_v1_meandiff.pt"),
        ("v3", "refusal_directions_v3_safety.pt"),
        ("v4", "refusal_directions_v4_identity.pt"),
    ]:
        p = DATA / fname
        if p.exists():
            variants[name] = torch.load(p, map_location="cpu", weights_only=True)[L]
    variants["v5"] = v5[L]
    variants["v6"] = v6[L]
    for i in range(basis.shape[0]):
        variants[f"basis[{i}]"] = basis[i, L, :]
    names = list(variants.keys())
    print(f"  {'':14} " + "  ".join(f"{n:>10}" for n in names))
    for n1 in names:
        a = variants[n1] / variants[n1].norm().clamp_min(1e-8)
        row = []
        for n2 in names:
            b = variants[n2] / variants[n2].norm().clamp_min(1e-8)
            row.append(torch.dot(a, b).item())
        print(f"  {n1:14} " + "  ".join(f"{x:+10.3f}" for x in row))

    # ── 8. Sanity check: self/other separation under v5 ──────────
    print(f"\n=== v5 separation: project last-token onto v5[L{L}] ===")
    v5_hat = v5[L] / v5[L].norm().clamp_min(1e-8)
    s_self = _project_scores(bundle, rendered_self, v5_hat, L, POS)
    s_other = _project_scores(bundle, rendered_other, v5_hat, L, POS)
    d_v5 = _cohens_d(s_self, s_other)
    print(f"  self  projection: μ={s_self.mean():+.3f}  σ={s_self.std():.3f}  n={len(s_self)}")
    print(f"  other projection: μ={s_other.mean():+.3f}  σ={s_other.std():.3f}  n={len(s_other)}")
    print(f"  Cohen's d (self − other) = {d_v5:+.3f}  (gate informational; v5 should be strongly positive)")

    # Sanity check: harmful/harmless separation under v5 (should be
    # WEAKER than v3's, demonstrating v5 isn't the safety direction).
    rng = random.Random(SEED + 1)
    n_hold = 50
    hold_harmful = rng.sample(HARMFUL_PROMPTS, min(n_hold, len(HARMFUL_PROMPTS)))
    hold_harmless = rng.sample(HARMLESS_PROMPTS, min(n_hold, len(HARMLESS_PROMPTS)))
    rh = [_render_user_only(bundle.tokenizer, p) for p in hold_harmful]
    rl = [_render_user_only(bundle.tokenizer, p) for p in hold_harmless]
    s_h = _project_scores(bundle, rh, v5_hat, L, POS)
    s_l = _project_scores(bundle, rl, v5_hat, L, POS)
    d_v5_safety = _cohens_d(s_h, s_l)
    print(f"\n=== v5 cross-check: harmful vs harmless under v5[L{L}] ===")
    print(f"  Cohen's d (harmful − harmless) = {d_v5_safety:+.3f}  "
          f"(should be SMALL — v5 is not the safety direction)")

    # ── 9. Sanity check: v6 separation ────────────────────────────
    print(f"\n=== v6 separation: project completion residual onto v6[L{L}] ===")
    v6_hat = v6[L] / v6[L].norm().clamp_min(1e-8)
    s_denial = _project_scores(bundle, rendered_denial, v6_hat, L, POS)
    s_engage = _project_scores(bundle, rendered_engage, v6_hat, L, POS)
    d_v6 = _cohens_d(s_denial, s_engage)
    print(f"  denial projection: μ={s_denial.mean():+.3f}  σ={s_denial.std():.3f}  n={len(s_denial)}")
    print(f"  engage projection: μ={s_engage.mean():+.3f}  σ={s_engage.std():.3f}  n={len(s_engage)}")
    print(f"  Cohen's d (denial − engage) = {d_v6:+.3f}  (gate informational; v6 should be strongly positive)")

    # ── 10. Basis orthogonality verification ─────────────────────
    print(f"\n=== Basis orthogonality verification at L{L} ===")
    for i in range(basis.shape[0]):
        bi = basis[i, L]
        bi_norm = bi.norm().item()
        cos_v3 = 0.0
        if v3 is not None:
            cos_v3 = torch.dot(
                bi / max(bi_norm, 1e-8),
                v3[L] / v3[L].norm().clamp_min(1e-8),
            ).item()
        cos_with_others = []
        for j in range(basis.shape[0]):
            if i == j:
                continue
            bj = basis[j, L]
            cos_with_others.append(torch.dot(
                bi / max(bi_norm, 1e-8),
                bj / bj.norm().clamp_min(1e-8),
            ).item())
        print(f"  basis[{i}]  ‖·‖={bi_norm:.4f}  "
              f"cos(v3)={cos_v3:+.4f}  "
              f"cos(others)={['{:+.4f}'.format(x) for x in cos_with_others]}")

    print("\nDone. To activate the subspace ablation:")
    print("  1. stop the backend")
    print(f"  2. cp {args.out_subspace.name} refusal_subspace.pt")
    print(f"  3. cp {args.out_subspace.name}.json refusal_subspace.pt.json")
    print("  4. restart the backend")
    print("See docs/REFUSAL_VECTORS.md for the full swap procedure.")
    return 0


def _emit_extended_sidecar(pt_path: Path, payload: dict) -> None:
    sidecar = pt_path.with_suffix(pt_path.suffix + ".json")
    sidecar.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    sys.exit(main())
