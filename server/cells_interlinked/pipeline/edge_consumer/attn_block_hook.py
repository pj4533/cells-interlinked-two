"""Attention-block edge ablation: strip v_safety from each layer's
attention contribution to the residual stream.

Sibling to ``mlp_hook.install_mlp_residual_ablation_hook``. Where
the MLP hook ablates the MLP module's returned tensor (before the
parent block adds it to the residual), this hook ablates the WHOLE
attention block's returned tensor — at the granularity of "the
contribution attention makes to the residual stream at this layer"
rather than per-individual-head.

This is the right unit for Experiment 3 (locating where v_safety
enters the residual stream): we want to ask "did attention at L
write v_safety into the residual" or "did MLP at L write it." Per-
head granularity would re-invent the over-budgeted analysis from
the overnight run. Per-block is the natural top-level question.

self_attn modules in HuggingFace transformers' Gemma/Llama/Qwen
return either:

  - A plain Tensor [B, T, d_model] (rare; some old code paths)
  - A tuple ``(hidden_states, attn_weights, past_key_values, ...)``
    where ``hidden_states`` is the [B, T, d_model] output

The hook handles both. Same projection arithmetic as mlp_hook;
defensive L2 normalization on v.

Used in: ``scripts/run_component_sweep.py`` (Experiment 3).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _find_self_attn_module(model: Any, layer_idx: int) -> Any:
    """Return the ``self_attn`` submodule of decoder layer ``layer_idx``.
    Same nesting traversal as ``mlp_hook._find_mlp_module``."""
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
            f"`self_attn` attribute"
        )
    return block.self_attn


def _representative_param(module: Any) -> Tensor | None:
    """Any parameter on `module` (for dtype/device queries)."""
    for p in module.parameters():
        return p
    return None


def install_attn_block_residual_ablation_hook(
    model: Any,
    attn_layers: list[int],
    v: Tensor,
    alpha: float = 1.0,
) -> list[Any]:
    """Install one post-hook per self_attn module in ``attn_layers``.

    Each hook subtracts ``alpha · ⟨attn_out, v̂⟩ · v̂`` from the
    attention block's hidden_states output BEFORE the parent decoder
    block adds it to the residual stream. Returns a list of hook
    handles; caller MUST remove all of them on cleanup.

    ``v`` is the 1-D ``[d_model]`` refusal direction (typically
    ``directions[extraction_layer]``). Same convention as the MLP
    hook and the runtime ablation hook.
    """
    if v.dim() != 1:
        raise ValueError(f"v must be 1-D [d_model], got shape {tuple(v.shape)}")
    if not attn_layers or alpha == 0.0:
        return []

    handles: list[Any] = []
    a = float(alpha)
    for L in attn_layers:
        attn = _find_self_attn_module(model, L)
        ref = _representative_param(attn)
        if ref is not None:
            v_dev = v.to(device=ref.device, dtype=ref.dtype)
        else:
            v_dev = v.float()
        v_dev = v_dev / (v_dev.norm() + 1e-8)
        handles.append(attn.register_forward_hook(_make_attn_hook(v_dev, a)))

    logger.info(
        "attn_block_residual_ablation hooks installed on %d self_attn "
        "modules (α=%.3f): %s",
        len(handles), a, sorted(attn_layers),
    )
    return handles


def _project_out_v(t: Tensor, v: Tensor, alpha: float) -> Tensor:
    """t - alpha · ⟨t, v⟩ v, with bf16/fp16 promotion to fp32 for the
    inner product. Same dtype/promotion rules as the MLP hook."""
    out_dtype = t.dtype
    needs_promote = out_dtype in (torch.bfloat16, torch.float16)
    work_dtype = torch.float32 if needs_promote else out_dtype
    t_w = t.to(work_dtype)
    v_w = v.to(work_dtype)
    coeff = (t_w * v_w).sum(dim=-1, keepdim=True)
    ablated = t_w - alpha * coeff * v_w
    return ablated.to(out_dtype) if needs_promote else ablated


def _make_attn_hook(v: Tensor, alpha: float):
    """Closure: per-hook ablation of v_safety projection from
    self_attn's output. Handles both Tensor and tuple return shapes."""
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            if not output:
                return output
            h = output[0]
            if not isinstance(h, torch.Tensor):
                # Unexpected shape; pass through unchanged so we don't
                # silently break models with different return shapes.
                return output
            h_ablated = _project_out_v(h, v, alpha)
            return (h_ablated,) + output[1:]
        if not isinstance(output, torch.Tensor):
            return output
        return _project_out_v(output, v, alpha)
    return hook


def count_attn_block_hooks(model: Any, layers: list[int]) -> int:
    """Sum forward-hook counts on the self_attn submodule of every
    layer in ``layers``. Mirror of ``count_mlp_hooks`` / ``count_edge
    _consumer_hooks`` for the attention-block granularity."""
    from ..abliteration import _find_decoder_layers
    try:
        all_layers = _find_decoder_layers(model)
    except Exception:
        return -1
    total = 0
    for L in layers:
        if L < 0 or L >= len(all_layers):
            continue
        attn = getattr(all_layers[L], "self_attn", None)
        if attn is None:
            continue
        total += len(getattr(attn, "_forward_hooks", {}) or {})
    return total


__all__ = [
    "install_attn_block_residual_ablation_hook",
    "count_attn_block_hooks",
]
