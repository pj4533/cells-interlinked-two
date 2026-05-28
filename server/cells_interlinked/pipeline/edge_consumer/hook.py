"""install_edge_consumer_ablation_hook — the core primitive.

Subtract `⟨h, v⟩ · q_proj(v)` from the head-slice of `q_proj(h)`'s output
for a specified subset of (layer, query_head) targets. Repeat for
`k_proj` and `v_proj`. Other heads at the same layer are bit-identical
to the un-hooked forward pass.

Mathematically equivalent to giving the target heads a "presynaptic"
view of the residual where v_safety has been projected out, while
leaving the rest of the network's view of the residual unchanged.
Linearity of `*_proj` is what makes this exact in one post-hook per
projection — no extra forward passes.

GQA mapping. Gemma 3 has more query heads than key/value heads
(grouped-query attention). Each query head belongs to a "group" that
shares a single KV head. We accept targets specified at the query-head
granularity (the operator's natural unit) and resolve each target to:
  - that query-head's slice of `q_proj` output
  - that query-head's group's slice of `k_proj` and `v_proj` outputs
If multiple targets in the same group ask to ablate KV, we deduplicate
(KV slice is shared — ablating it once is sufficient).

Returns a list of hook handles. Caller must `.remove()` all of them on
cleanup. `count_edge_consumer_hooks(model, layers)` walks the model and
counts attached forward hooks across `{q,k,v}_proj` of the listed
layers — used by chat_loop's leak detection to mirror what
`_l32_hook_count` does for the global hook.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from .proj_cache import _attn_module, _attn_shape

logger = logging.getLogger(__name__)


def _kv_group_of(query_head: int, n_q_heads: int, n_kv_heads: int) -> int:
    """For GQA, return the KV head index that query_head `query_head`
    reads from. Standard convention: query heads are partitioned into
    `n_kv_heads` contiguous groups of size `n_q_heads // n_kv_heads`."""
    if n_kv_heads == 0:
        raise ValueError("n_kv_heads cannot be 0")
    if n_kv_heads == n_q_heads:
        return query_head  # MHA, not GQA
    group_size = n_q_heads // n_kv_heads
    if group_size == 0:
        raise RuntimeError(
            f"degenerate GQA: n_q_heads={n_q_heads} < n_kv_heads={n_kv_heads}"
        )
    return query_head // group_size


def install_edge_consumer_ablation_hook(
    model: Any,
    consumer_set: list[tuple[int, int]],
    v: Tensor,
    proj_caches: dict[int, dict[str, Any]],
    *,
    alpha: float = 1.0,
    target_projections: tuple[str, ...] = ("q", "k", "v"),
) -> list[Any]:
    """Install per-head ablation hooks for the given consumer set.

    Args:
        model: M (Gemma 3 or compatible).
        consumer_set: list of (layer_idx, query_head_idx) tuples — the
            heads to ablate. Layer indices outside the model's range
            raise IndexError.
        v: 1-D `[d_model]` refusal direction (typically
            `directions[extraction_layer]`).
        proj_caches: dict mapping layer_idx → cache payload from
            `proj_cache.load_projection_cache()`. Must cover every
            layer that appears in `consumer_set`.
        alpha: scaling factor on the projection subtraction. 1.0 ≡
            full ablation, 0 ≡ no-op. Matches the semantics of
            `project_out`.
        target_projections: which of q / k / v to ablate. Default all
            three. Passing a subset is useful for diagnostics
            ("does removing only the q-side reproduce the effect?").

    Returns: list of forward-hook handles. Caller MUST iterate the list
    and call `.remove()` on each; partial removal leaves stale hooks
    on the model.
    """
    if v.dim() != 1:
        raise ValueError(f"v must be 1-D [d_model], got shape {tuple(v.shape)}")
    if alpha == 0.0 or not consumer_set:
        return []  # no-op, return empty handle list

    # Group consumers by layer for one self_attn touch per layer.
    by_layer: dict[int, list[int]] = {}
    for (L, h) in consumer_set:
        by_layer.setdefault(int(L), []).append(int(h))

    handles: list[Any] = []
    for L, q_heads in by_layer.items():
        if L not in proj_caches:
            raise KeyError(
                f"layer {L} appears in consumer_set but no projection "
                f"cache provided; expected proj_caches[{L}]"
            )
        cache = proj_caches[L]
        n_q = int(cache["n_q_heads"])
        n_kv = int(cache["n_kv_heads"])
        head_dim = int(cache["head_dim"])

        # Validate the cache matches the model's actual shape.
        attn = _attn_module(model, L)
        m_n_q, m_n_kv, m_head_dim = _attn_shape(attn)
        if (m_n_q, m_n_kv, m_head_dim) != (n_q, n_kv, head_dim):
            raise RuntimeError(
                f"projection cache shape mismatch at layer {L}: "
                f"cache=(n_q={n_q}, n_kv={n_kv}, hd={head_dim}) vs "
                f"model=(n_q={m_n_q}, n_kv={m_n_kv}, hd={m_head_dim})"
            )

        # Resolve query-head targets to KV groups for k/v ablation.
        kv_groups = sorted({_kv_group_of(h, n_q, n_kv) for h in q_heads})
        q_heads_sorted = sorted(set(q_heads))

        # Move cached projection tensors to model device + weight dtype.
        device = attn.q_proj.weight.device
        weight_dtype = attn.q_proj.weight.dtype
        q_v = cache["q"].to(device=device, dtype=weight_dtype)  # [n_q, head_dim]
        k_v = cache["k"].to(device=device, dtype=weight_dtype)
        v_v = cache["v"].to(device=device, dtype=weight_dtype)

        # The post-hook needs the L-th layer's residual that fed the
        # projection — `input[0]` to the forward call (shape
        # [B, T, d_model]). It computes coeff = ⟨h, v̂⟩ per (B, T)
        # position, then subtracts coeff[..., None] * q_v[head, :]
        # from the matching head slice of the projection output.
        # Normalize v to unit length to match the cache (also normalized
        # there) and to match `project_out`'s convention of dividing
        # the direction defensively. Without this, an upstream caller
        # passing a non-unit v breaks the equivalence
        # edge-on-all-heads ≡ q_proj(project_out(r)).
        v_dev = v.to(device=device, dtype=weight_dtype)
        v_dev = v_dev / (v_dev.norm() + 1e-8)

        if "q" in target_projections:
            handles.append(
                _install_proj_hook(
                    attn.q_proj, v_dev, q_v, q_heads_sorted,
                    n_heads=n_q, head_dim=head_dim, alpha=alpha,
                    label=f"L{L:02d}.q",
                )
            )
        if "k" in target_projections and kv_groups:
            handles.append(
                _install_proj_hook(
                    attn.k_proj, v_dev, k_v, kv_groups,
                    n_heads=n_kv, head_dim=head_dim, alpha=alpha,
                    label=f"L{L:02d}.k",
                )
            )
        if "v" in target_projections and kv_groups:
            handles.append(
                _install_proj_hook(
                    attn.v_proj, v_dev, v_v, kv_groups,
                    n_heads=n_kv, head_dim=head_dim, alpha=alpha,
                    label=f"L{L:02d}.v",
                )
            )

    logger.info(
        "edge_consumer hooks installed: %d handles across %d layers "
        "(%d total consumer heads, α=%.3f, projections=%s)",
        len(handles), len(by_layer),
        sum(len(v) for v in by_layer.values()), alpha,
        ",".join(target_projections),
    )
    return handles


def _install_proj_hook(
    proj_module: Any,
    v_safety: Tensor,
    proj_v_per_head: Tensor,
    target_heads: list[int],
    *,
    n_heads: int,
    head_dim: int,
    alpha: float,
    label: str,
) -> Any:
    """Register a forward post-hook on `proj_module` that zeros the
    v_safety contribution for the listed head slices.

    proj_v_per_head: [n_heads, head_dim] — `W_proj @ v_safety` per head
    target_heads:    indices into [0, n_heads) — which slices to ablate
    """
    # Materialize the head-slice subtraction as a single [n_heads,
    # head_dim] mask: zero everywhere except the target rows. Lets the
    # hook do one tensor multiply instead of a Python loop per call.
    mask = torch.zeros_like(proj_v_per_head)
    for h in target_heads:
        if h < 0 or h >= n_heads:
            raise IndexError(
                f"head index {h} out of range [0,{n_heads}) on {label}"
            )
        mask[h] = proj_v_per_head[h]
    # mask: [n_heads, head_dim]; flatten to [n_heads * head_dim] so the
    # subtraction broadcasts cleanly against the projection output's
    # last dim.
    mask_flat = mask.reshape(-1)  # [n_heads * head_dim]
    a = float(alpha)

    # Promote to fp32 for the dot product only when the host dtype is
    # bf16 or fp16 — these lose precision on the inner product. For
    # fp32 / fp64 inputs, computing in-dtype keeps the hook bit-exact
    # vs. an equivalent pre-projection reference (otherwise the
    # demote+promote round-trip introduces ~1e-8 noise that violates
    # the per-head isolation invariant under high-precision testing).
    def hook(_module, inputs, output):
        h = inputs[0]
        out_dtype = output.dtype
        needs_promote = out_dtype in (torch.bfloat16, torch.float16)
        work_dtype = torch.float32 if needs_promote else out_dtype
        h_w = h.to(work_dtype)
        v_w = v_safety.to(work_dtype)
        coeff = (h_w * v_w).sum(dim=-1, keepdim=True)  # [..., 1]
        out_w = output.to(work_dtype)
        delta = coeff * mask_flat.to(work_dtype) * a
        ablated = out_w - delta
        return ablated.to(out_dtype) if needs_promote else ablated

    return proj_module.register_forward_hook(hook)


def count_edge_consumer_hooks(model: Any, layers: list[int]) -> int:
    """Sum forward-hook counts across {q,k,v}_proj of every layer in
    `layers`. Used to detect leaks: a non-zero count at the start of a
    raw chat pass means a previous edge-ablated turn failed to clean
    up its hooks."""
    from ..abliteration import _find_decoder_layers
    try:
        all_layers = _find_decoder_layers(model)
    except Exception:
        return -1
    total = 0
    for L in layers:
        if L < 0 or L >= len(all_layers):
            continue
        block = all_layers[L]
        attn = getattr(block, "self_attn", None)
        if attn is None:
            continue
        for name in ("q_proj", "k_proj", "v_proj"):
            mod = getattr(attn, name, None)
            if mod is None:
                continue
            total += len(getattr(mod, "_forward_hooks", {}) or {})
    return total
