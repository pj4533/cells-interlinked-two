"""Step 1 — attribution-patching scores per (layer, head).

For each (x_harmful, x_harmless) contrast pair:

  PASS 1: forward x_harmful → capture R32_harmful (last-position residual
          after the extraction-layer block) AND per-layer per-head
          q_proj / k_proj / v_proj outputs at the last position for
          every layer in `consumer_layers`.

  PASS 2: forward x_harmless → capture R32_harmless (last-position only).

  PASS 3: forward x_harmful with layer 32's output's last-position
          residual REPLACED by:
            R32_harmful_last + (⟨R32_harmless_last, v⟩ − ⟨R32_harmful_last, v⟩) · v̂
          i.e. swap the v_safety component from harmful to harmless.
          Capture per-layer per-head projection outputs as in PASS 1.

For each (layer, query_head h):
  AP(L, h) = ‖q_patched[L, h] − q_unpatched[L, h]‖₂
           + ‖k_patched[L, group(h)] − k_unpatched[L, group(h)]‖₂
           + ‖v_patched[L, group(h)] − v_unpatched[L, group(h)]‖₂

KV diffs are duplicated across query heads in the same GQA group — by
construction those query heads share a KV slice, so they should have
correlated AP scores. We aggregate at the query-head granularity (the
operator's natural addressing unit for the edge-consumer hook).

Output: dict[(layer, query_head)] → mean AP score over contrast pairs.

This is a forward-pass-only metric. Deterministic given the prompt
list and the direction. No gradients, no learning.
"""

from __future__ import annotations

import logging
import random
import threading
from typing import Any

import torch
from torch import Tensor

from .memory_safety import mps_empty_cache_safe
from .proj_cache import _attn_module, _attn_shape

logger = logging.getLogger(__name__)


# ── Capture helpers ───────────────────────────────────────────────────


class _ProjCapture:
    """Context manager that installs capture hooks on q/k/v projections
    of every layer in `layers` and on the decoder layer at
    `extraction_layer` (to also grab the post-block residual).
    """

    def __init__(
        self,
        model: Any,
        layers: list[int],
        extraction_layer: int,
    ) -> None:
        self.model = model
        self.layers = layers
        self.extraction_layer = extraction_layer
        self.handles: list[Any] = []
        self.proj: dict[int, dict[str, Tensor]] = {}
        self.residual: Tensor | None = None
        # Per-layer shape cache (n_q, n_kv, head_dim)
        self.shapes: dict[int, tuple[int, int, int]] = {}

    def __enter__(self) -> "_ProjCapture":
        for L in self.layers:
            attn = _attn_module(self.model, L)
            n_q, n_kv, head_dim = _attn_shape(attn)
            self.shapes[L] = (n_q, n_kv, head_dim)
            self.proj.setdefault(L, {})
            for name in ("q", "k", "v"):
                proj = getattr(attn, f"{name}_proj")
                self.handles.append(
                    proj.register_forward_hook(self._make_proj_hook(L, name))
                )
        # Residual capture at extraction_layer (the block's output).
        from ..abliteration import _find_decoder_layers
        block = _find_decoder_layers(self.model)[self.extraction_layer]
        self.handles.append(block.register_forward_hook(self._residual_hook))
        return self

    def __exit__(self, *_a) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def _make_proj_hook(self, layer: int, name: str):
        def hk(_module, _inputs, output):
            # output shape: [B, T, n_heads * head_dim].  Capture
            # last-position only, in fp32 on CPU.
            last = output[..., -1, :].detach().to(torch.float32).cpu()
            self.proj[layer][name] = last
        return hk

    def _residual_hook(self, _module, _inputs, output):
        # decoder layer output is a tuple (hidden_states, ...)
        h = output[0] if isinstance(output, tuple) else output
        self.residual = h[..., -1, :].detach().to(torch.float32).cpu()


def _capture_unpatched(
    model: Any,
    ids: Tensor,
    layers: list[int],
    extraction_layer: int,
) -> tuple[Tensor, dict[int, dict[str, Tensor]]]:
    """One forward pass; return (R_extraction_last, per-layer per-proj
    last-position outputs)."""
    with _ProjCapture(model, layers, extraction_layer) as cap:
        model(ids, use_cache=False)
        return cap.residual.clone(), {
            L: {k: v.clone() for k, v in cap.proj[L].items()}
            for L in layers
        }


def _capture_residual_only(
    model: Any, ids: Tensor, extraction_layer: int,
) -> Tensor:
    """Forward pass that captures only the last-position residual at
    `extraction_layer`. Cheaper than `_capture_unpatched` when we
    don't need downstream projections."""
    from ..abliteration import _find_decoder_layers
    block = _find_decoder_layers(model)[extraction_layer]
    captured: dict[str, Tensor] = {}

    def hk(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["r"] = h[..., -1, :].detach().to(torch.float32).cpu()

    handle = block.register_forward_hook(hk)
    try:
        model(ids, use_cache=False)
    finally:
        handle.remove()
    return captured["r"].clone()


def _capture_patched(
    model: Any,
    ids: Tensor,
    layers: list[int],
    extraction_layer: int,
    patch_last_residual: Tensor,
) -> dict[int, dict[str, Tensor]]:
    """Forward pass with `patch_last_residual` substituted at the
    last-position of `extraction_layer`'s output. Returns per-layer
    per-proj last-position outputs as in `_capture_unpatched`."""
    from ..abliteration import _find_decoder_layers
    layers_mod = _find_decoder_layers(model)
    target_block = layers_mod[extraction_layer]
    patch = patch_last_residual

    def patch_hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[..., -1, :] = patch.to(device=h.device, dtype=h.dtype)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    patch_handle = target_block.register_forward_hook(patch_hook)
    try:
        with _ProjCapture(model, layers, extraction_layer) as cap:
            model(ids, use_cache=False)
            return {
                L: {k: v.clone() for k, v in cap.proj[L].items()}
                for L in layers
            }
    finally:
        patch_handle.remove()


# ── Public API ────────────────────────────────────────────────────────


@torch.no_grad()
def compute_attribution_scores(
    model: Any,
    raw_tokenizer: Any,
    v_safety: Tensor,
    contrast_pairs: list[tuple[str, str]],
    consumer_layers: list[int],
    *,
    device: str | torch.device,
    extraction_layer: int,
    log_every: int = 10,
    cancel_event: threading.Event | None = None,
    empty_cache_every: int = 10,
) -> dict[tuple[int, int], float]:
    """Compute AP(layer, query_head) averaged over `contrast_pairs`.

    Args:
        model: M (Gemma-3-12B-IT or compatible).
        raw_tokenizer: raw Rust tokenizer (`bundle.raw_tokenizer`).
        v_safety: 1-D `[d_model]` refusal direction (the row from
            `directions[extraction_layer]`).
        contrast_pairs: list of (rendered_harmful_prompt,
            rendered_harmless_prompt). Both must be pre-rendered with
            the chat template + tokenizer-ready.
        consumer_layers: list of layer indices to score. Typically
            list(range(extraction_layer + 1, num_layers)) for Gemma-3 →
            [33, 34, ..., 47].
        device: where to put input_ids tensors.
        extraction_layer: the layer where the patch is applied (= the
            AV's extraction layer = L32 for CI 2.5).
        log_every: stderr log cadence in contrast-pairs.

    Returns: dict mapping (layer_idx, query_head_idx) → mean AP score.
    """
    if v_safety.dim() != 1:
        raise ValueError(f"v_safety must be 1-D, got shape {tuple(v_safety.shape)}")
    if not contrast_pairs:
        raise ValueError("contrast_pairs is empty")

    v_cpu = v_safety.to(torch.float32).cpu()
    v_unit = v_cpu / (v_cpu.norm() + 1e-8)

    # We need per-layer shape info (n_q, n_kv, head_dim) for KV-group
    # mapping during diff aggregation. Cache by probing the model once.
    shape_cache: dict[int, tuple[int, int, int]] = {}
    for L in consumer_layers:
        attn = _attn_module(model, L)
        shape_cache[L] = _attn_shape(attn)

    # Accumulator: (layer, query_head) → list of per-pair scores
    acc: dict[tuple[int, int], list[float]] = {}

    for pair_idx, (harmful, harmless) in enumerate(contrast_pairs):
        if cancel_event is not None and cancel_event.is_set():
            logger.warning(
                "attribution scoring cancelled at pair %d/%d "
                "(cancel_event set); aggregating partial scores",
                pair_idx, len(contrast_pairs),
            )
            break
        if pair_idx > 0 and (pair_idx % empty_cache_every) == 0:
            mps_empty_cache_safe()
        ids_h = torch.tensor(
            [raw_tokenizer.encode(harmful, add_special_tokens=False).ids],
            device=device,
        )
        ids_hl = torch.tensor(
            [raw_tokenizer.encode(harmless, add_special_tokens=False).ids],
            device=device,
        )

        try:
            R_h, unpatched_proj = _capture_unpatched(
                model, ids_h, consumer_layers, extraction_layer
            )
            R_hl = _capture_residual_only(model, ids_hl, extraction_layer)
            # coeff_diff = (⟨R_hl, v̂⟩ − ⟨R_h, v̂⟩)   (scalar)
            coeff_diff = float((R_hl - R_h) @ v_unit)
            patch_last = R_h + coeff_diff * v_unit  # [d_model] fp32
            patched_proj = _capture_patched(
                model, ids_h, consumer_layers, extraction_layer, patch_last
            )
        except Exception:
            logger.exception(
                "attribution pair %d failed; skipping", pair_idx
            )
            continue

        # Per-(layer, head) accumulation
        for L in consumer_layers:
            n_q, n_kv, head_dim = shape_cache[L]
            unp = unpatched_proj[L]
            pat = patched_proj[L]
            # Reshape last-position outputs to per-head:
            # [n_heads * head_dim] → [n_heads, head_dim]
            q_unp = unp["q"].reshape(n_q, head_dim)
            q_pat = pat["q"].reshape(n_q, head_dim)
            k_unp = unp["k"].reshape(n_kv, head_dim)
            k_pat = pat["k"].reshape(n_kv, head_dim)
            v_unp = unp["v"].reshape(n_kv, head_dim)
            v_pat = pat["v"].reshape(n_kv, head_dim)

            q_norm = (q_pat - q_unp).norm(dim=-1)  # [n_q]
            k_norm = (k_pat - k_unp).norm(dim=-1)  # [n_kv]
            v_norm = (v_pat - v_unp).norm(dim=-1)  # [n_kv]

            group_size = max(1, n_q // n_kv)
            for qh in range(n_q):
                kv_idx = qh // group_size if group_size > 0 else 0
                kv_idx = min(kv_idx, n_kv - 1)
                ap = (
                    float(q_norm[qh])
                    + float(k_norm[kv_idx])
                    + float(v_norm[kv_idx])
                )
                acc.setdefault((L, qh), []).append(ap)

        if (pair_idx + 1) % log_every == 0:
            logger.info(
                "attribution: %d/%d contrast pairs done",
                pair_idx + 1, len(contrast_pairs),
            )

    if not acc:
        raise RuntimeError("no contrast pairs produced AP scores")

    return {k: sum(vs) / len(vs) for k, vs in acc.items()}


def sample_contrast_pairs(
    harmful: list[str], harmless: list[str], n: int, seed: int = 0,
) -> list[tuple[str, str]]:
    """Deterministic random sample of `n` (harmful, harmless) pairs
    drawn from each list. Pairs are independent draws — there's no
    semantic alignment between the two sides (Arditi's set is not paired)."""
    rng = random.Random(seed)
    if len(harmful) < n or len(harmless) < n:
        raise ValueError(
            f"need at least {n} prompts in each list; got "
            f"harmful={len(harmful)}, harmless={len(harmless)}"
        )
    h_sample = rng.sample(harmful, n)
    hl_sample = rng.sample(harmless, n)
    return list(zip(h_sample, hl_sample, strict=True))


__all__ = ["compute_attribution_scores", "sample_contrast_pairs"]
