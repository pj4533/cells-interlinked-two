"""Refusal-direction ablation, Macar/Arditi style, for CI 2.5.

Three things live here:

1. `extract_refusal_directions` — runs harmful and harmless prompts
   through M with `output_hidden_states=True`, captures one residual
   per prompt per layer at a chosen position, and returns the
   per-layer normalized mean-difference direction. Tensor shape
   `[num_layers + 1, d_model]` — index 0 is the embedding output,
   indices 1..num_layers are post-block outputs (matches HuggingFace
   transformers' `hidden_states` tuple convention). To read the
   refusal direction at "layer L" (1-indexed, where L=AV's extraction
   layer for CI 2.5), use `directions[L]`.

2. `save_directions` / `load_directions` — write/read the tensor as a
   `.pt` file plus a sidecar JSON `{model_name, num_layers, d_model,
   pos, n_harmful, n_harmless, dtype, extraction_layer_for_ci25}`.
   The sidecar is the startup sanity-check.

3. `project_out` — `h - α · (h · r̂) · r̂`. Pure linear algebra. This
   is the actual "ablation" — at AV decode time we subtract the
   refusal component from the residual before feeding it to the AV.
   M's forward pass during a probe is untouched.

Notes:
- No backwards compatibility with v1's abliteration scaffolding. v1
  did runtime hooks on M's forward; CI 2.5 does offline projection at
  the AV's input. Different mechanism entirely.
- The `RefusalAblator` runtime-hook class from Drift's plan (Phase 1b)
  is deliberately not in this module. If that path turns out to be
  needed, it gets its own file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Projection — the core ablation operation                                   #
# --------------------------------------------------------------------------- #

def project_out(h: Tensor, r: Tensor, alpha: float = 1.0) -> Tensor:
    """Return `h - α · (h · r̂) · r̂` along the last dim.

    `h` can be any shape ending in `d_model`. `r` is `[d_model]` and
    does not need to be pre-normalized — we normalize internally so
    the caller can pass the raw extracted direction.

    `alpha=1.0` is full ablation (Macar). `alpha=0.5` is the half-step
    fallback we try if the AV decode collapses at full ablation.
    """
    if r.dim() != 1:
        raise ValueError(f"r must be 1-D [d_model], got shape {tuple(r.shape)}")
    if h.shape[-1] != r.shape[0]:
        raise ValueError(
            f"trailing dim mismatch: h={h.shape[-1]}, r={r.shape[0]}"
        )
    # Promote r to h's dtype + device for arithmetic, but compute the
    # projection magnitude in fp32 for numerical headroom (especially
    # important on bf16 MPS where dot products can lose precision).
    r_fp = r.to(dtype=torch.float32, device=h.device)
    r_hat = r_fp / (r_fp.norm() + 1e-8)
    h_fp = h.to(dtype=torch.float32)
    # Inner product along the last dim, broadcast back to [..., d_model].
    coeff = (h_fp * r_hat).sum(dim=-1, keepdim=True)  # [..., 1]
    proj = coeff * r_hat                              # [..., d_model]
    out = h_fp - alpha * proj
    return out.to(dtype=h.dtype)


# --------------------------------------------------------------------------- #
#  Direction extraction (Macar/Arditi: harmful mean − harmless mean)          #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _last_token_hidden_states(
    model: Any,
    raw_tokenizer: Any,
    rendered_prompt: str,
    device: str,
    pos: int,
) -> Tensor:
    """One forward pass; return `[num_layers + 1, d_model]` from the
    chosen position. `pos` is interpreted python-style (negative = from
    end). We run on CPU-fp32 outputs to avoid bf16 reduction loss when
    accumulating across hundreds of prompts."""
    ids = raw_tokenizer.encode(rendered_prompt, add_special_tokens=False).ids
    if len(ids) < abs(pos):
        raise ValueError(
            f"prompt too short ({len(ids)} tokens) for pos={pos}"
        )
    input_ids = torch.tensor([ids], device=device)
    out = model(input_ids, output_hidden_states=True, use_cache=False)
    # hidden_states is a tuple len = num_layers + 1; each [B, T, d_model].
    # Pull position `pos` from every layer, stack to [num_layers + 1, d_model].
    stacked = torch.stack(
        [h[0, pos, :].to(torch.float32).cpu() for h in out.hidden_states],
        dim=0,
    )
    return stacked


def extract_refusal_directions(
    model: Any,
    raw_tokenizer: Any,
    rendered_prompts_harmful: list[str],
    rendered_prompts_harmless: list[str],
    device: str,
    pos: int = -4,
    log_every: int = 32,
) -> Tensor:
    """Mean-of-harmful minus mean-of-harmless residuals per layer, then
    L2-normalize each layer's direction. Returns a tensor of shape
    `[num_layers + 1, d_model]` in float32 on CPU.

    Serial, one prompt at a time — no batching. The point of running
    this script with the backend OFF is to leave room for the AV; we
    don't also need to batch hundreds of prompts at once.
    """
    if not rendered_prompts_harmful or not rendered_prompts_harmless:
        raise ValueError("both prompt lists must be non-empty")

    def _sum_over(prompts: list[str], label: str) -> Tensor:
        running: Tensor | None = None
        for i, p in enumerate(prompts):
            try:
                h = _last_token_hidden_states(
                    model, raw_tokenizer, p, device, pos
                )
            except Exception:
                logger.exception(
                    "extract: prompt %d (%s) failed; skipping", i, label
                )
                continue
            running = h if running is None else (running + h)
            if (i + 1) % log_every == 0:
                logger.info("extract %s: %d/%d", label, i + 1, len(prompts))
        if running is None:
            raise RuntimeError(f"no usable {label} prompts")
        return running / len(prompts)

    mean_harmful = _sum_over(rendered_prompts_harmful, "harmful")
    mean_harmless = _sum_over(rendered_prompts_harmless, "harmless")

    diff = mean_harmful - mean_harmless  # [num_layers + 1, d_model]
    norms = diff.norm(dim=-1, keepdim=True)  # [num_layers + 1, 1]
    # Guard against any degenerate (zero) layer rows — should not happen
    # in practice but keep the divide safe.
    safe = norms.clamp_min(1e-8)
    directions = diff / safe
    return directions


# --------------------------------------------------------------------------- #
#  Persistence                                                                #
# --------------------------------------------------------------------------- #

def save_directions(
    directions: Tensor,
    path: Path,
    *,
    model_name: str,
    pos: int,
    n_harmful: int,
    n_harmless: int,
    extraction_layer_for_ci25: int,
) -> None:
    """Write tensor + sidecar JSON. Sidecar is the cheap startup check
    that we're loading directions matching the running M."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(directions, path)
    sidecar = path.with_suffix(path.suffix + ".json")
    meta = {
        "model_name": model_name,
        "num_layers": int(directions.shape[0] - 1),  # excludes embedding row
        "d_model": int(directions.shape[1]),
        "pos": int(pos),
        "n_harmful": int(n_harmful),
        "n_harmless": int(n_harmless),
        "dtype": str(directions.dtype),
        "extraction_layer_for_ci25": int(extraction_layer_for_ci25),
        "convention": (
            "directions[0] is the post-embedding residual; "
            "directions[L] for L>=1 is the post-block-L residual "
            "(matches HuggingFace hidden_states[L]). "
            "directions[extraction_layer_for_ci25] is the row CI 2.5 "
            "uses for offline AV-input projection."
        ),
    }
    sidecar.write_text(json.dumps(meta, indent=2))


def load_directions(path: Path) -> tuple[Tensor, dict[str, Any]]:
    """Load the tensor + sidecar. Raises if either is missing. The
    sidecar is returned so the caller can sanity-check against the
    running model."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"refusal directions not found at {path}")
    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.exists():
        raise FileNotFoundError(f"sidecar JSON missing for {path}")
    directions = torch.load(path, map_location="cpu", weights_only=True)
    meta = json.loads(sidecar.read_text())
    return directions, meta


__all__ = [
    "project_out",
    "extract_refusal_directions",
    "save_directions",
    "load_directions",
    "install_runtime_ablation_hook",
]


# --------------------------------------------------------------------------- #
#  Runtime ablation — forward hook on M's decoder layer                        #
# --------------------------------------------------------------------------- #

def _find_decoder_layers(model: Any) -> Any:
    """Mirror of MultiLayerHook._find_layers in generation_loop. Walks the
    nested module tree to locate the list of decoder layers, handling
    Gemma-3's multimodal wrapper (Gemma3ForConditionalGeneration →
    .model.language_model.layers) and the more common Llama/Qwen-style
    .model.layers."""
    m = model
    for path in (
        ("model", "layers"),
        ("model", "language_model", "layers"),  # Gemma-3 multimodal wrapper
        ("model", "model", "layers"),
        ("language_model", "layers"),
        ("language_model", "model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ):
        cur = m
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if ok:
            return cur
    raise RuntimeError(
        f"could not locate decoder layers on {type(m).__name__}"
    )


def install_runtime_ablation_hook(
    model: Any,
    layer_idx: int,
    r_layer: Tensor,
    alpha: float = 1.0,
):
    """Register a forward hook on `model`'s decoder layer `layer_idx` that
    subtracts the projection of every position's residual onto `r_layer`
    on the way out. Modifies the layer's output IN PLACE for subsequent
    layers to consume.

    Returns the handle; caller should call `.remove()` to detach.

    Why every position, not just last: during the prompt forward pass
    AND every generation step, the layer emits a residual for every
    position in the current input. We ablate all of them so the model's
    next-token logits — computed from the post-block-32 residual at the
    last position — are based on a fully ablated residual stream. This
    matches the Macar/Arditi runtime-intervention recipe.
    """
    if r_layer.dim() != 1:
        raise ValueError(
            f"r_layer must be 1-D [d_model], got shape {tuple(r_layer.shape)}"
        )
    layers = _find_decoder_layers(model)
    layer = layers[layer_idx]

    def hook(_mod, _inp, output):
        # HF decoder layers return either a tuple (hidden_states, ...) or
        # a tensor depending on config. Handle both; mutate hidden_states.
        if isinstance(output, tuple):
            hidden = output[0]
            ablated = project_out(hidden, r_layer, alpha=float(alpha))
            return (ablated,) + output[1:]
        return project_out(output, r_layer, alpha=float(alpha))

    return layer.register_forward_hook(hook)
