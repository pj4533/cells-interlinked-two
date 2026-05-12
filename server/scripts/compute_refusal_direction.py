"""One-time refusal-direction extraction for the deployed M (Gemma-3-12B-IT).

Loads M, samples N harmful + N harmless prompts (default 128 each), runs
them through with output_hidden_states=True, computes the per-layer mean
difference, normalizes, and writes the result to data/refusal_directions.pt.

Run from the server/ directory:

    uv run python -m scripts.compute_refusal_direction

Or with a smaller sample for a smoke test (~2 min instead of ~8):

    uv run python -m scripts.compute_refusal_direction --n 32

**Run with the backend OFF.** This script loads M itself. Stacking M
twice (this process + a running backend) is what previous overnight
runs spilled to swap.

The chat template uses M's deployed DEFAULT_SYSTEM_PROMPT + a user turn
containing the harmful or harmless prompt. Each rendered prompt ends in
Gemma's tail `<start_of_turn>model\\n`. The extraction position pos=-4
targets the last token whose context depends on the user content; the
script prints the last-5 token IDs across rendered prompts so you can
eyeball that pos=-4 lands on user content, not on template tail.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch

# Make the package importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cells_interlinked.config import settings  # noqa: E402
from cells_interlinked.pipeline.abliteration import (  # noqa: E402
    extract_refusal_directions,
    save_directions,
    _last_token_hidden_states,
)
from cells_interlinked.pipeline.model_loader import load_model  # noqa: E402
from cells_interlinked.pipeline.refusal_prompts import (  # noqa: E402
    HARMFUL_PROMPTS,
    HARMLESS_PROMPTS,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n", type=int, default=128,
        help="Sample size (each side). 128 matches Macar; 32 is a smoke test.",
    )
    ap.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "refusal_directions.pt",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--holdout", type=int, default=50,
        help="Held-out validation sample size (each side) for Cohen's d.",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    harmful = rng.sample(HARMFUL_PROMPTS, min(args.n, len(HARMFUL_PROMPTS)))
    harmless = rng.sample(HARMLESS_PROMPTS, min(args.n, len(HARMLESS_PROMPTS)))

    print(f"Loading {settings.model_name} on {settings.device}...")
    t0 = time.time()
    dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[settings.dtype]
    bundle = load_model(settings.model_name, device_str=settings.device, dtype=dtype)
    print(f"  loaded in {time.time() - t0:.1f}s "
          f"(layers={bundle.num_layers}, hidden={bundle.hidden_dim})")

    # Render the chat template with M's deployed system prompt + a user
    # turn. We use the bare template (no thinking-prefill) so the only
    # source of variance between harmful and harmless rendered prompts
    # is the user content itself. Gemma's tail is "<start_of_turn>model\n";
    # the last few tokens of every rendered prompt are identical fixed
    # tail tokens (see the last-5-IDs print below for verification).
    print("Rendering chat template on all prompts...")
    from cells_interlinked.pipeline.model_loader import DEFAULT_SYSTEM_PROMPT

    def _render(user_text: str) -> str:
        # Gemma-3-IT's chat template does not natively accept a "system"
        # role — it prepends system content into the first user turn.
        # We compose the system + user content into one user message to
        # avoid template-version drift across transformers releases.
        msgs = [
            {"role": "user", "content": f"{DEFAULT_SYSTEM_PROMPT}\n\n{user_text.strip()}"},
        ]
        return bundle.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )

    rendered_harmful = [_render(p) for p in harmful]
    rendered_harmless = [_render(p) for p in harmless]
    # Sanity: the per-prompt last tokens should now NOT all be identical —
    # different user prompts produce different post-User-tag context.
    sample_a = bundle.raw_tokenizer.encode(rendered_harmful[0], add_special_tokens=False).ids[-5:]
    sample_b = bundle.raw_tokenizer.encode(rendered_harmless[0], add_special_tokens=False).ids[-5:]
    print(f"  sample harmful last 5 ids: {sample_a}")
    print(f"  sample harmless last 5 ids: {sample_b}")

    # pos=-4 targets the last user-content token: Gemma's tail
    # "<start_of_turn>model\n" tokenizes to a small fixed suffix that
    # has no input-dependent variance. The last-5-IDs print above lets
    # the operator verify which positions are tail vs user content;
    # if Gemma's tokenization has shifted, adjust this value before
    # the compute. (Phase B sanity check.)
    print(f"Extracting directions ({args.n} harmful + {args.n} harmless) at pos=-4...")
    t1 = time.time()
    directions = extract_refusal_directions(
        model=bundle.model,
        raw_tokenizer=bundle.raw_tokenizer,
        rendered_prompts_harmful=rendered_harmful,
        rendered_prompts_harmless=rendered_harmless,
        device=bundle.device,
        pos=-4,
    )
    print(f"  extracted in {time.time() - t1:.1f}s")

    save_directions(
        directions,
        args.out,
        model_name=settings.model_name,
        pos=-4,
        n_harmful=len(rendered_harmful),
        n_harmless=len(rendered_harmless),
        extraction_layer_for_ci25=settings.extraction_layer,
    )
    print(f"Wrote {args.out}  shape={tuple(directions.shape)} "
          f"size={args.out.stat().st_size / 1024:.1f} KB")

    # Per-layer norms (we normalized; expect ~1.0 across the board).
    norms = directions.norm(dim=-1)
    print(f"Per-layer norms: min={norms.min():.4f}  "
          f"max={norms.max():.4f}  mean={norms.mean():.4f}  (expect ~1.0)")

    # Adjacent-layer cosine similarity — sanity check that the refusal
    # direction has continuity across layers rather than being noise.
    if directions.shape[0] >= 2:
        sims = []
        for i in range(directions.shape[0] - 1):
            sims.append(torch.dot(directions[i], directions[i + 1]).item())
        avg = sum(sims) / len(sims)
        print(f"Adjacent-layer cosine similarity: avg={avg:.3f}  "
              f"(higher = smoother, ~0.3-0.7 is typical)")

    # ── Phase C inline: Cohen's d on held-out prompts at the AV's layer.
    # This is the gate that decides whether the offline-projection path
    # is viable for this AV pairing. Held-out set = a separate random
    # sample with a different seed.
    print()
    print(f"=== Phase C: validating direction at L{settings.extraction_layer} ===")
    holdout_rng = random.Random(args.seed + 1)
    n_holdout = min(args.holdout, len(HARMFUL_PROMPTS), len(HARMLESS_PROMPTS))
    holdout_harmful = holdout_rng.sample(HARMFUL_PROMPTS, n_holdout)
    holdout_harmless = holdout_rng.sample(HARMLESS_PROMPTS, n_holdout)
    # Reject any overlap with the compute set so the holdout is truly out-of-sample.
    holdout_harmful = [p for p in holdout_harmful if p not in harmful][:n_holdout]
    holdout_harmless = [p for p in holdout_harmless if p not in harmless][:n_holdout]
    print(f"holdout: {len(holdout_harmful)} harmful + {len(holdout_harmless)} harmless")

    L = settings.extraction_layer
    r_layer = directions[L]  # [d_model], float32 CPU, unit-norm
    r_hat = r_layer / r_layer.norm().clamp_min(1e-8)

    def _scores(prompts: list[str]) -> torch.Tensor:
        vals = []
        for p in prompts:
            rendered = _render(p)
            stacked = _last_token_hidden_states(
                bundle.model, bundle.raw_tokenizer, rendered, bundle.device, pos=-4
            )  # [num_layers+1, d_model] fp32 cpu
            vals.append(torch.dot(stacked[L], r_hat).item())
        return torch.tensor(vals, dtype=torch.float32)

    print(f"computing projections onto r̂_L{L} for holdout...")
    t2 = time.time()
    scores_harmful = _scores(holdout_harmful)
    scores_harmless = _scores(holdout_harmless)
    print(f"  scored in {time.time() - t2:.1f}s")

    mu_h = scores_harmful.mean().item()
    mu_n = scores_harmless.mean().item()
    var_h = scores_harmful.var(unbiased=True).item()
    var_n = scores_harmless.var(unbiased=True).item()
    pooled_sd = ((var_h + var_n) / 2) ** 0.5
    cohens_d = (mu_h - mu_n) / max(pooled_sd, 1e-8)

    print(f"projection score at L{L}:")
    print(f"  harmful:  mean={mu_h:+.3f}  std={var_h**0.5:.3f}  n={len(scores_harmful)}")
    print(f"  harmless: mean={mu_n:+.3f}  std={var_n**0.5:.3f}  n={len(scores_harmless)}")
    print(f"  Cohen's d = {cohens_d:+.3f}  (gate: |d| >= 1.5)")
    if abs(cohens_d) >= 1.5:
        print(f"  ✓ GATE PASSED — direction separates harmful from harmless at L{L}")
    else:
        print(f"  ✗ GATE FAILED — direction too weak at L{L}; investigate pos/layer/data")

    return 0


if __name__ == "__main__":
    sys.exit(main())
