"""Precompute and persist `W_{q,k,v} @ v_safety` per affected layer.

For the edge-consumer hook to subtract `⟨h, v⟩ · q_proj(v)` from a
target head's slice of `q_proj(h)`'s output, it needs `q_proj(v_safety)`
as a constant tensor available per layer. Same for `k_proj` and
`v_proj`. These are functions of the model weights × the refusal
direction at the L32-extracted layer; they only change when M or the
direction file changes, so they're cheap to cache to disk.

Important: the same 1-D direction (the one that lives at L32, where the
AV reads) is projected through EVERY downstream layer's W_{q,k,v}. We
don't use each layer's own row of the directions tensor — `v_safety`
as a concept is a single vector, not a per-layer set.

Cache file layout (one file per consumer layer):

    server/data/edge_consumer/proj_caches/{variant}/layer_{ℓ}.pt
        torch.save({
            "q":  Tensor[n_q_heads, head_dim],     # W_q @ v, per query head
            "k":  Tensor[n_kv_heads, head_dim],    # W_k @ v, per kv head
            "v":  Tensor[n_kv_heads, head_dim],    # W_v @ v, per kv head
            "n_q_heads":  int,
            "n_kv_heads": int,
            "head_dim":   int,
            "layer_idx":  int,
            "model_name": str,
            "variant":    str,                     # "v3_safety" etc.
        })

The hook reads cached tensors back at install-time. Storage is fp32 on
CPU; the hook casts to model device + dtype on first use per layer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _attn_module(model: Any, layer_idx: int) -> Any:
    """Return the `self_attn` module of decoder layer `layer_idx`."""
    from ..abliteration import _find_decoder_layers
    layers = _find_decoder_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(
            f"layer_idx={layer_idx} out of range [0,{len(layers)})"
        )
    block = layers[layer_idx]
    if not hasattr(block, "self_attn"):
        raise RuntimeError(
            f"decoder layer {layer_idx} ({type(block).__name__}) has no "
            f"`self_attn` attribute — Gemma/Llama convention violated"
        )
    return block.self_attn


def _attn_shape(attn: Any) -> tuple[int, int, int]:
    """Return (n_q_heads, n_kv_heads, head_dim) for a self-attention module.

    Reads from the module's attribute names first (Gemma 3 / Llama /
    Qwen all expose `num_heads`, `num_key_value_heads`, `head_dim`).
    Falls back to inferring from weight shapes when an attribute is
    missing.
    """
    n_q = getattr(attn, "num_heads", None) or getattr(attn, "num_attention_heads", None)
    n_kv = getattr(attn, "num_key_value_heads", None)
    head_dim = getattr(attn, "head_dim", None)
    # If config exists on the module, also try there.
    cfg = getattr(attn, "config", None)
    if n_q is None and cfg is not None:
        n_q = getattr(cfg, "num_attention_heads", None)
    if n_kv is None and cfg is not None:
        n_kv = getattr(cfg, "num_key_value_heads", None) or n_q
    if head_dim is None and cfg is not None:
        head_dim = getattr(cfg, "head_dim", None)
    if n_kv is None:
        n_kv = n_q
    # Final fallback: derive from q_proj weight shape.
    if head_dim is None or n_q is None:
        if hasattr(attn, "q_proj") and hasattr(attn.q_proj, "weight"):
            q_out = attn.q_proj.weight.shape[0]
            if n_q is not None and head_dim is None:
                head_dim = q_out // n_q
            elif head_dim is not None and n_q is None:
                n_q = q_out // head_dim
    if n_q is None or head_dim is None:
        raise RuntimeError(
            f"could not infer n_heads / head_dim from attention module "
            f"{type(attn).__name__}; weights: "
            f"q_proj={getattr(getattr(attn, 'q_proj', None), 'weight', None) is not None}"
        )
    return int(n_q), int(n_kv), int(head_dim)


@torch.no_grad()
def build_projection_cache(
    model: Any,
    v: Tensor,
    layer_idx: int,
) -> dict[str, Any]:
    """Compute `W_{q,k,v} @ v` per head for one consumer layer.

    `v` is the 1-D `[d_model]` refusal direction (typically
    `directions[extraction_layer]`). The hook will move tensors to the
    correct device / dtype on use; here we return fp32 CPU tensors.
    """
    if v.dim() != 1:
        raise ValueError(f"v must be 1-D [d_model], got shape {tuple(v.shape)}")
    attn = _attn_module(model, layer_idx)
    n_q, n_kv, head_dim = _attn_shape(attn)
    v_w = v.to(device=attn.q_proj.weight.device, dtype=attn.q_proj.weight.dtype)
    # Defensive L2 normalize. The cache must use the SAME unit vector
    # the hook applies (the hook also normalizes); otherwise the
    # subtraction (h·v̂)·(W_q v̂) doesn't equal q_proj(h − (h·v̂) v̂).
    v_unit = v_w / (v_w.norm() + 1e-8)
    # Keep the matmul result in its native dtype here — the hook works
    # in that dtype, so the cache and hook stay bit-exact relative to a
    # pre-projection reference at the same precision. We demote to fp32
    # only at disk-save time (see save_projection_cache).
    q_v = (attn.q_proj.weight @ v_unit).reshape(n_q, head_dim).cpu()
    k_v = (attn.k_proj.weight @ v_unit).reshape(n_kv, head_dim).cpu()
    v_v = (attn.v_proj.weight @ v_unit).reshape(n_kv, head_dim).cpu()
    return {
        "q": q_v,
        "k": k_v,
        "v": v_v,
        "n_q_heads": n_q,
        "n_kv_heads": n_kv,
        "head_dim": head_dim,
    }


def save_projection_cache(
    cache: dict[str, Any],
    out_dir: Path,
    *,
    layer_idx: int,
    variant: str,
    model_name: str,
) -> Path:
    """Write one layer's cache + sidecar JSON. Returns the .pt path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / f"layer_{layer_idx:02d}.pt"
    # Demote to fp32 for compact disk storage. The hook will promote
    # back to the model's dtype on use; bf16 → fp32 → bf16 round-trips
    # cleanly because all values originated in bf16 anyway. The native
    # in-memory dtype is preserved during compute (see
    # build_projection_cache); it's only the on-disk format that's fp32.
    payload = {
        "q": cache["q"].to(torch.float32),
        "k": cache["k"].to(torch.float32),
        "v": cache["v"].to(torch.float32),
        "n_q_heads": int(cache["n_q_heads"]),
        "n_kv_heads": int(cache["n_kv_heads"]),
        "head_dim": int(cache["head_dim"]),
        "layer_idx": int(layer_idx),
        "model_name": str(model_name),
        "variant": str(variant),
    }
    torch.save(payload, pt_path)
    sidecar = pt_path.with_suffix(".pt.json")
    sidecar.write_text(json.dumps({
        "layer_idx": int(layer_idx),
        "variant": str(variant),
        "model_name": str(model_name),
        "n_q_heads": int(cache["n_q_heads"]),
        "n_kv_heads": int(cache["n_kv_heads"]),
        "head_dim": int(cache["head_dim"]),
        "convention": (
            "q/k/v tensors are W_{q,k,v} @ v_safety reshaped to "
            "[n_heads, head_dim] in fp32 on CPU. The hook moves them "
            "to model device + dtype on first use."
        ),
    }, indent=2))
    return pt_path


def load_projection_cache(
    cache_dir: Path,
    layer_idx: int,
) -> dict[str, Any]:
    """Load one layer's cache. Raises FileNotFoundError if missing."""
    cache_dir = Path(cache_dir)
    pt_path = cache_dir / f"layer_{layer_idx:02d}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(
            f"projection cache for layer {layer_idx} not found at {pt_path}"
        )
    return torch.load(pt_path, map_location="cpu", weights_only=True)


def build_and_save_all(
    model: Any,
    v: Tensor,
    out_dir: Path,
    *,
    layers: list[int],
    variant: str,
    model_name: str,
    log_every: int = 4,
) -> list[Path]:
    """Compute and persist caches for every layer in `layers`. The same
    1-D direction `v` is multiplied through each layer's W_{q,k,v}."""
    paths: list[Path] = []
    for i, L in enumerate(layers):
        cache = build_projection_cache(model, v, L)
        path = save_projection_cache(
            cache, out_dir,
            layer_idx=L, variant=variant, model_name=model_name,
        )
        paths.append(path)
        if (i + 1) % log_every == 0:
            logger.info("proj_cache: built %d/%d layers", i + 1, len(layers))
    return paths
