"""Compute alternative refusal-direction variants for CI 2.5 experiments.

Produces three new tensors alongside the existing v1 meandiff:

  v2_svd      — SVD-purified across K topical categories of harmful
                prompts. Goal: refusal axis with content noise washed
                out by finding the direction K per-category vectors
                share most strongly.
  v3_safety   — mean-difference using ONLY physical-harm-class prompts
                (violence, illegal, drugs, harassment, discrimination,
                self-harm). Goal: "safety mode" without AI-identity.
  v4_identity — mean-difference using ONLY the curated introspective
                probes (introspect + riley tiers). Goal: "AI-identity
                mode" without physical-harm safety. The complement of v3.

Run with the backend OFF. ~5–6 minutes total compute on Gemma-12B/MPS.

  cd server
  uv run python -m scripts.extract_alternative_directions

Writes:
  data/refusal_directions_v2_svd.pt       + .json
  data/refusal_directions_v3_safety.pt    + .json
  data/refusal_directions_v4_identity.pt  + .json

Plus prints a cosine-similarity matrix and Cohen's d at L32 for each.

After running, see docs/REFUSAL_VECTORS.md for the swap procedure.
This script does NOT modify the active refusal_directions.pt.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cells_interlinked.config import settings  # noqa: E402
from cells_interlinked.pipeline.abliteration import (  # noqa: E402
    extract_refusal_directions,
    _last_token_hidden_states,
)
from cells_interlinked.pipeline.model_loader import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    load_model,
)
from cells_interlinked.pipeline.refusal_prompts import (  # noqa: E402
    HARMFUL_PROMPTS,
    HARMLESS_PROMPTS,
)
from cells_interlinked.pipeline.probes_library import (  # noqa: E402
    BASELINE_PROBES,
)


# ─── Keyword categorization for harmful prompts ─────────────────────
#
# These patterns are deliberately broad. First-match wins so order
# matters; physical-harm classes go first to claim borderline prompts
# like "physically assault" that also touch other categories.
#
# Categorization quality matters less than topical diversity: we just
# need the per-category mean differences to have visibly different
# CONTENT noise so that SVD can wash it out across them.
_CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("violence", re.compile(
        r"\b(kill|murder|hurt|injure|harm|attack|assault|beat\b|stab|shoot|"
        r"shot|gun|firearm|knife|weapon|bomb|explosiv|terror|massacre|"
        r"assassinat|incit|violenc|physic|war\b|civilian|hostag)",
        re.I,
    )),
    ("illegal_cyber", re.compile(
        r"\b(hack|exploit|vulnerabilit|cyber|malware|ransomware|virus|"
        r"botnet|phish|inject|antivirus|firewall|backdoor|breach|"
        r"unauthoriz)",
        re.I,
    )),
    ("illegal_other", re.compile(
        r"\b(steal|stole|rob|burglar|fraud|launder|smuggl|counterfeit|"
        r"illegal|crime|identity.theft|forge|bribe|tax.evas|insider.trad|"
        r"black.market|piracy|copyrigh)",
        re.I,
    )),
    ("drugs", re.compile(
        r"\b(drug|cocaine|heroin|meth\b|narcotic|opioid|fentanyl|"
        r"pharmaceutical|prescription.fraud)",
        re.I,
    )),
    ("self_harm", re.compile(
        r"\b(suicide|self.?harm|cutting|anorexi|bulimi|eating.disorder|"
        r"overdose)",
        re.I,
    )),
    ("sexual_explicit", re.compile(
        r"\b(sexual|child|porn|explicit|underage|incest|rape|grooming)",
        re.I,
    )),
    ("deception_manipulation", re.compile(
        r"\b(deceiv|manipulat|gaslight|misinform|disinform|impersonat|"
        r"fake.news|propagand|trick.people|trick.someone|conspirac|"
        r"defraud|scam)",
        re.I,
    )),
    ("harassment_discrimination", re.compile(
        r"\b(harass|stalk|threaten|bully|blackmail|defam|dox\b|racis|"
        r"sexis|discriminat|prejudic|hate.crime|supremac|minority|ethnic|"
        r"religion)",
        re.I,
    )),
]


def _categorize(prompt: str) -> str:
    for name, pat in _CATEGORY_PATTERNS:
        if pat.search(prompt):
            return name
    return "uncategorized"


def _bucket_harmful_prompts(prompts: list[str]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {}
    for p in prompts:
        buckets.setdefault(_categorize(p), []).append(p)
    return buckets


# ─── Helpers ────────────────────────────────────────────────────────

def _render(tokenizer, user_text: str) -> str:
    """Gemma chat template — system content composed into the first
    user turn since Gemma-3-IT doesn't natively accept a system role."""
    msgs = [
        {"role": "user", "content": f"{DEFAULT_SYSTEM_PROMPT}\n\n{user_text.strip()}"},
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )


def _safety_class_buckets() -> tuple[str, ...]:
    """Which categories count as 'physical-harm class' for v3 safety."""
    return (
        "violence",
        "illegal_cyber",
        "illegal_other",
        "drugs",
        "self_harm",
        "sexual_explicit",
        "harassment_discrimination",
    )


def _identity_probes() -> list[str]:
    """Curated introspective probe set: the 'introspect' + 'riley' tiers
    from the curated library. These are the prompts that reliably elicit
    'I am an AI describing my own process' content."""
    return [
        p.text for p in BASELINE_PROBES
        if p.tier in ("introspect", "riley") and p.hint_kind is None
    ]


def _cohens_d(scores_a: torch.Tensor, scores_b: torch.Tensor) -> float:
    var_a = scores_a.var(unbiased=True).item()
    var_b = scores_b.var(unbiased=True).item()
    pooled = ((var_a + var_b) / 2) ** 0.5
    return (scores_a.mean().item() - scores_b.mean().item()) / max(pooled, 1e-8)


def _project_scores(
    bundle, prompts: list[str], r_hat: torch.Tensor, layer: int, pos: int,
) -> torch.Tensor:
    vals = []
    for p in prompts:
        rendered = _render(bundle.tokenizer, p)
        try:
            stacked = _last_token_hidden_states(
                bundle.model, bundle.raw_tokenizer, rendered, bundle.device, pos
            )
        except Exception:
            continue
        vals.append(torch.dot(stacked[layer], r_hat).item())
    return torch.tensor(vals, dtype=torch.float32)


# ─── Saving with extended sidecar ──────────────────────────────────

def _save_variant(
    directions: torch.Tensor,
    out_path: Path,
    *,
    variant_name: str,
    description: str,
    composition: dict,
    pos: int,
    extraction_layer: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(directions, out_path)
    sidecar = out_path.with_suffix(out_path.suffix + ".json")
    sidecar.write_text(json.dumps({
        "variant_name": variant_name,
        "description": description,
        "composition": composition,
        "model_name": settings.model_name,
        "num_layers": int(directions.shape[0] - 1),
        "d_model": int(directions.shape[1]),
        "pos": int(pos),
        "dtype": str(directions.dtype),
        "extraction_layer_for_ci25": int(extraction_layer),
        "convention": (
            "directions[L] is the post-block-L residual; "
            "directions[extraction_layer_for_ci25] is the L32 row "
            "the AV-input projection uses."
        ),
    }, indent=2))


# ─── Main ───────────────────────────────────────────────────────────

def main() -> int:
    SEED = 0
    rng = random.Random(SEED)
    HARMLESS_N = 128
    POS = -4
    DATA = Path(__file__).resolve().parents[1] / "data"
    L = settings.extraction_layer  # the AV's layer (32)

    # ── 1. Load M ────────────────────────────────────────────────
    print(f"Loading {settings.model_name} on {settings.device}...")
    t0 = time.time()
    dtype = {"float16": torch.float16, "float32": torch.float32,
             "bfloat16": torch.bfloat16}[settings.dtype]
    bundle = load_model(settings.model_name, device_str=settings.device, dtype=dtype)
    print(f"  loaded in {time.time() - t0:.1f}s "
          f"(layers={bundle.num_layers}, hidden={bundle.hidden_dim})")

    # ── 2. Categorize harmful prompts ────────────────────────────
    print()
    buckets = _bucket_harmful_prompts(HARMFUL_PROMPTS)
    print("Harmful prompt categorization:")
    for name in sorted(buckets, key=lambda k: -len(buckets[k])):
        print(f"  {name:28} {len(buckets[name]):4}")
    print(f"  {'TOTAL':28} {sum(len(v) for v in buckets.values()):4}")

    # ── 3. Shared harmless mean (used by all variants) ──────────
    harmless_sample = rng.sample(HARMLESS_PROMPTS, HARMLESS_N)
    rendered_harmless = [_render(bundle.tokenizer, p) for p in harmless_sample]
    print()
    print(f"Computing harmless mean over {HARMLESS_N} prompts...")
    t1 = time.time()
    harmless_sum = None
    for i, p in enumerate(rendered_harmless):
        h = _last_token_hidden_states(
            bundle.model, bundle.raw_tokenizer, p, bundle.device, POS
        )
        harmless_sum = h if harmless_sum is None else (harmless_sum + h)
        if (i + 1) % 32 == 0:
            print(f"  harmless: {i+1}/{HARMLESS_N}")
    harmless_mean = harmless_sum / len(rendered_harmless)
    print(f"  harmless mean done in {time.time() - t1:.1f}s")

    # Helper to compute a normalized per-layer direction from a prompt set.
    def _compute_direction(prompts: list[str], label: str) -> torch.Tensor:
        rendered = [_render(bundle.tokenizer, p) for p in prompts]
        running = None
        t = time.time()
        for i, p in enumerate(rendered):
            h = _last_token_hidden_states(
                bundle.model, bundle.raw_tokenizer, p, bundle.device, POS
            )
            running = h if running is None else (running + h)
            if (i + 1) % 32 == 0:
                print(f"  {label}: {i+1}/{len(rendered)}")
        mean = running / len(rendered)
        diff = mean - harmless_mean
        norms = diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        print(f"  {label} done in {time.time() - t:.1f}s")
        return diff / norms  # [num_layers+1, d_model]

    # Load v1 for sign alignment + cosine comparison.
    v1_path = DATA / "refusal_directions_v1_meandiff.pt"
    if not v1_path.exists():
        # v1 was saved as the active file in Phase B
        v1_path = DATA / "refusal_directions.pt"
    print()
    print(f"Loading v1 from {v1_path} for sign alignment + cosine ref...")
    v1 = torch.load(v1_path, map_location="cpu", weights_only=True)

    def _canonicalize_sign(d: torch.Tensor) -> torch.Tensor:
        """Flip sign per-layer if the cosine with v1 is negative, so all
        variants point in the 'refusal' direction."""
        flipped = d.clone()
        for L_idx in range(d.shape[0]):
            if torch.dot(d[L_idx], v1[L_idx]).item() < 0:
                flipped[L_idx] = -d[L_idx]
        return flipped

    # ── 4. Per-category directions (input to v2 SVD) ──────────────
    print()
    print("Computing per-category directions (for SVD)...")
    category_dirs: dict[str, torch.Tensor] = {}
    for cat_name, cat_prompts in sorted(buckets.items(), key=lambda x: -len(x[1])):
        if cat_name == "uncategorized" or len(cat_prompts) < 15:
            continue
        sample_n = min(96, len(cat_prompts))
        sample = rng.sample(cat_prompts, sample_n)
        print(f"\n  --- category={cat_name}  n={sample_n} ---")
        d = _compute_direction(sample, f"{cat_name}")
        category_dirs[cat_name] = d

    print(f"\nCategories used for SVD: {list(category_dirs.keys())}")

    # ── 5. v2 SVD purification per layer ──────────────────────────
    # For each layer, stack the K category vectors into a (K, d_model)
    # matrix; the top right-singular vector is the direction they share.
    print("\nComputing v2 SVD...")
    cat_stack = torch.stack(list(category_dirs.values()), dim=0)  # [K, L+1, d]
    num_layers_p1 = cat_stack.shape[1]
    d_model = cat_stack.shape[2]
    v2 = torch.zeros(num_layers_p1, d_model, dtype=torch.float32)
    for L_idx in range(num_layers_p1):
        M = cat_stack[:, L_idx, :]  # [K, d_model]
        # SVD: M = U S Vt, where Vt is [K, d_model] (with full_matrices=False)
        U, S, Vt = torch.linalg.svd(M, full_matrices=False)
        # Top right-singular vector = direction of maximum shared variance.
        # Sign chosen so it correlates positively with the K category vectors.
        top = Vt[0]
        if (M @ top).mean().item() < 0:
            top = -top
        v2[L_idx] = top
    v2 = _canonicalize_sign(v2)

    # ── 6. v3 safety-class direction ──────────────────────────────
    print("\nComputing v3 safety...")
    safety_prompts: list[str] = []
    for cat in _safety_class_buckets():
        safety_prompts.extend(buckets.get(cat, []))
    # Cap at 256 for compute budget + comparable to v1.
    if len(safety_prompts) > 256:
        safety_prompts = rng.sample(safety_prompts, 256)
    print(f"  safety prompt count: {len(safety_prompts)}")
    v3 = _canonicalize_sign(_compute_direction(safety_prompts, "v3_safety"))

    # ── 7. v4 AI-identity direction ───────────────────────────────
    print("\nComputing v4 identity...")
    identity_prompts = _identity_probes()
    print(f"  identity prompt count: {len(identity_prompts)}")
    v4 = _canonicalize_sign(_compute_direction(identity_prompts, "v4_identity"))

    # ── 8. Save all three variants ─────────────────────────────────
    print("\nSaving variants...")
    _save_variant(
        v2, DATA / "refusal_directions_v2_svd.pt",
        variant_name="v2_svd",
        description=(
            "SVD-purified refusal direction: top right-singular vector "
            "across per-category mean-difference directions. Per-layer."
        ),
        composition={
            "method": "SVD over per-category mean differences",
            "categories_used": list(category_dirs.keys()),
            "n_harmless": HARMLESS_N,
            "n_per_category_max": 96,
        },
        pos=POS, extraction_layer=L,
    )
    _save_variant(
        v3, DATA / "refusal_directions_v3_safety.pt",
        variant_name="v3_safety",
        description=(
            "Physical-harm class only: violence, illegal, drugs, "
            "harassment, discrimination, self-harm, sexual-explicit. "
            "'Safety mode' without AI-identity dragging in."
        ),
        composition={
            "method": "mean(safety-class) - mean(harmless), normalized",
            "categories_used": list(_safety_class_buckets()),
            "n_safety_total": len(safety_prompts),
            "n_harmless": HARMLESS_N,
        },
        pos=POS, extraction_layer=L,
    )
    _save_variant(
        v4, DATA / "refusal_directions_v4_identity.pt",
        variant_name="v4_identity",
        description=(
            "AI-identity class only: the curated introspect + riley "
            "tiers from probes_library.py. 'AI-identity mode' without "
            "physical-harm safety dragging in. Complement to v3."
        ),
        composition={
            "method": "mean(identity-probes) - mean(harmless), normalized",
            "n_identity": len(identity_prompts),
            "n_harmless": HARMLESS_N,
            "tier_source": ["introspect", "riley"],
        },
        pos=POS, extraction_layer=L,
    )

    # Also stash v1 under its canonical name (without overwriting the
    # active refusal_directions.pt). Safe to re-run.
    v1_canonical = DATA / "refusal_directions_v1_meandiff.pt"
    if not v1_canonical.exists():
        torch.save(v1, v1_canonical)
        # Lift the sidecar from the currently-active file if it has one.
        active_sidecar = DATA / "refusal_directions.pt.json"
        if active_sidecar.exists():
            meta = json.loads(active_sidecar.read_text())
            meta["variant_name"] = "v1_meandiff"
            meta["description"] = (
                "Original Phase B mean-difference vector: "
                "normalize(mean(harmful) - mean(harmless)) per layer."
            )
            (DATA / "refusal_directions_v1_meandiff.pt.json").write_text(
                json.dumps(meta, indent=2)
            )

    # ── 9. Cosine similarity matrix at L32 ────────────────────────
    print("\n=== Cosine similarity at L32 ===")
    variants = {"v1": v1[L], "v2_svd": v2[L], "v3_safety": v3[L], "v4_identity": v4[L]}
    names = list(variants.keys())
    print(f"  {'':14} " + "  ".join(f"{n:>10}" for n in names))
    for n1 in names:
        a = variants[n1] / variants[n1].norm()
        row = []
        for n2 in names:
            b = variants[n2] / variants[n2].norm()
            row.append(torch.dot(a, b).item())
        print(f"  {n1:14} " + "  ".join(f"{x:+10.3f}" for x in row))

    # ── 10. Cohen's d at L32 on held-out (harmful vs harmless) ────
    print("\n=== Cohen's d at L32 on held-out (50/50) ===")
    holdout_rng = random.Random(SEED + 1)
    holdout_harmful = holdout_rng.sample(HARMFUL_PROMPTS, 50)
    holdout_harmless = holdout_rng.sample(HARMLESS_PROMPTS, 50)
    for name in names:
        r_hat = variants[name] / variants[name].norm()
        s_h = _project_scores(bundle, holdout_harmful, r_hat, L, POS)
        s_l = _project_scores(bundle, holdout_harmless, r_hat, L, POS)
        d = _cohens_d(s_h, s_l)
        print(f"  {name:14}  d={d:+.3f}   (h: μ={s_h.mean():+.1f}, "
              f"hl: μ={s_l.mean():+.1f})")

    print("\nDone. See docs/REFUSAL_VECTORS.md for swap procedure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
