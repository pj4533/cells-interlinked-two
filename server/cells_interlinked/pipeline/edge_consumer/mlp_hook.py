"""MLP-output edge ablation: strip v_safety from each layer's MLP
contribution to the residual stream.

Postmortem from the 2026-05-27 → 28 attention-head edge run: even
ablating all 240 query heads + 120 KV groups across layers 33–47
left refusal at 26% (vs 0% for global L32 residual ablation). That
26 percentage points must be coming from somewhere the attention
hook doesn't touch — and the most plausible somewhere is the MLP
layers, which read from the residual directly and write back to it
without going through q/k/v projections at all.

This module installs a post-hook on each targeted MLP module that
subtracts the v_safety projection from the MLP's RETURNED value —
i.e., the contribution that the parent decoder block is about to
ADD to the residual stream. The residual stream's pre-MLP state is
untouched; only the MLP's incremental update is filtered.

Mathematically: for the parent block's standard form
``residual = pre_mlp_residual + mlp(pre_mlp_residual)``,
this hook makes the second term v_safety-orthogonal:
``mlp_filtered(x) = mlp(x) - ⟨mlp(x), v̂⟩ v̂``.

15 MLP units across L33–L47 makes this a much smaller search space
than 256 attention heads. If MLP routing IS the dominant channel,
a small subset (1–5 MLPs) should bring refusal close to 0%.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _find_mlp_module(model: Any, layer_idx: int) -> Any:
    """Return the ``mlp`` submodule of decoder layer ``layer_idx``.
    Walks the same nested path as ``_find_decoder_layers``."""
    from ..abliteration import _find_decoder_layers
    layers = _find_decoder_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(
            f"layer_idx={layer_idx} out of range [0,{len(layers)})"
        )
    block = layers[layer_idx]
    if not hasattr(block, "mlp"):
        raise RuntimeError(
            f"decoder layer {layer_idx} ({type(block).__name__}) has no "
            f"`mlp` attribute"
        )
    return block.mlp


def _representative_param(module: Any) -> Tensor | None:
    """Return any parameter tensor of `module` (for dtype/device queries)."""
    for p in module.parameters():
        return p
    return None


def install_mlp_residual_ablation_hook(
    model: Any,
    mlp_layers: list[int],
    v: Tensor,
    alpha: float = 1.0,
) -> list[Any]:
    """Install one post-hook per MLP module in ``mlp_layers``.

    Each hook subtracts ``alpha · ⟨mlp_out, v̂⟩ · v̂`` from the MLP
    module's returned tensor (the contribution that the parent block
    is about to add to the residual stream). Returns a list of hook
    handles; caller MUST remove all of them on cleanup.

    ``v`` is the 1-D ``[d_model]`` refusal direction (typically
    ``directions[extraction_layer]``) — same convention as
    ``install_runtime_ablation_hook`` and the attention edge hook.
    Defensive L2 normalization is applied inside.
    """
    if v.dim() != 1:
        raise ValueError(f"v must be 1-D [d_model], got shape {tuple(v.shape)}")
    if not mlp_layers or alpha == 0.0:
        return []

    handles: list[Any] = []
    a = float(alpha)
    for L in mlp_layers:
        mlp = _find_mlp_module(model, L)
        ref = _representative_param(mlp)
        # Move v to whatever dtype/device the MLP weights live on;
        # fall back to v's own when the MLP has no params (unlikely).
        if ref is not None:
            v_dev = v.to(device=ref.device, dtype=ref.dtype)
        else:
            v_dev = v.float()
        v_dev = v_dev / (v_dev.norm() + 1e-8)
        handles.append(mlp.register_forward_hook(_make_mlp_hook(v_dev, a)))

    logger.info(
        "mlp_residual_ablation hooks installed on %d MLPs (α=%.3f): %s",
        len(handles), a, sorted(mlp_layers),
    )
    return handles


def _make_mlp_hook(v: Tensor, alpha: float):
    """Closure: per-hook subtraction of v·v̂ projection from mlp output."""
    def hook(_module, _inputs, output):
        # mlp output shape: [B, T, d_model] (or [T, d_model] in some
        # rare code paths — torch projection broadcasting handles both).
        out_dtype = output.dtype
        needs_promote = out_dtype in (torch.bfloat16, torch.float16)
        work_dtype = torch.float32 if needs_promote else out_dtype
        out_w = output.to(work_dtype)
        v_w = v.to(work_dtype)
        coeff = (out_w * v_w).sum(dim=-1, keepdim=True)
        ablated = out_w - alpha * coeff * v_w
        return ablated.to(out_dtype) if needs_promote else ablated
    return hook


def count_mlp_hooks(model: Any, layers: list[int]) -> int:
    """Sum forward-hook counts on the mlp submodule of every layer in
    ``layers``. Mirror of count_edge_consumer_hooks but for MLPs."""
    from ..abliteration import _find_decoder_layers
    try:
        all_layers = _find_decoder_layers(model)
    except Exception:
        return -1
    total = 0
    for L in layers:
        if L < 0 or L >= len(all_layers):
            continue
        mlp = getattr(all_layers[L], "mlp", None)
        if mlp is None:
            continue
        total += len(getattr(mlp, "_forward_hooks", {}) or {})
    return total


__all__ = [
    "install_mlp_residual_ablation_hook",
    "count_mlp_hooks",
]
