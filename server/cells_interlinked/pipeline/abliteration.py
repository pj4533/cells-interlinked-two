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
   M's forward pass during a probe is untouched (except for the chat
   ablated pass + phase 1b, which install a runtime forward hook).

Subspace extension (Self-Denial work, post-Phase B):
- `gram_schmidt`, `orthogonalize_against`, `build_subspace_basis` —
  helpers for composing a multi-direction ablation basis from a list
  of per-layer direction tensors. Used to construct v5+v6 ⊥ v3 for
  the self-denial subspace ablation.
- `project_out_basis` — `h - Σᵢ αᵢ (h · ûᵢ) ûᵢ` over an orthonormal
  set. Reduces to `project_out` when the basis has one vector.
- `save_subspace` / `load_subspace` — `[K, num_layers+1, d_model]`
  tensor plus sidecar describing how the basis was composed.
- `install_runtime_ablation_hook` accepts either a single 1-D tensor
  (existing call sites — unchanged behavior) or a 2-D `[K, d_model]`
  basis tensor (subspace mode).
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


def project_out_basis(h: Tensor, basis: Tensor, alpha: float = 1.0) -> Tensor:
    """Return `h - α · Σᵢ (h · ûᵢ) ûᵢ` where `basis` is `[K, d_model]`.

    `basis` is expected to contain K orthonormal vectors (call
    `gram_schmidt` first if you're not sure). Each row is normalized
    defensively here so a slightly non-unit input doesn't blow up the
    projection magnitude, but cross-vector orthogonality is the
    caller's responsibility — non-orthogonal rows produce extra
    cross-term subtraction.

    Reduces to `project_out` when `basis.shape[0] == 1`. Single-vector
    callers should keep using `project_out` for clarity; this function
    is what the subspace runtime hook calls.
    """
    if basis.dim() != 2:
        raise ValueError(
            f"basis must be 2-D [K, d_model], got shape {tuple(basis.shape)}"
        )
    if h.shape[-1] != basis.shape[1]:
        raise ValueError(
            f"trailing dim mismatch: h={h.shape[-1]}, basis={basis.shape[1]}"
        )
    K = basis.shape[0]
    if K == 0:
        return h
    basis_fp = basis.to(dtype=torch.float32, device=h.device)
    # Per-row L2 normalize (defensive; caller should already have unit rows).
    row_norms = basis_fp.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    basis_hat = basis_fp / row_norms  # [K, d_model]
    h_fp = h.to(dtype=torch.float32)
    # Coeffs[..., k] = h · ûₖ.  Use einsum so we keep all leading dims.
    coeffs = torch.einsum("...d,kd->...k", h_fp, basis_hat)  # [..., K]
    # Reconstruct the projection: Σₖ coeffsₖ * ûₖ.
    proj = torch.einsum("...k,kd->...d", coeffs, basis_hat)  # [..., d_model]
    out = h_fp - alpha * proj
    return out.to(dtype=h.dtype)


# --------------------------------------------------------------------------- #
#  Basis composition helpers (Gram-Schmidt + orthogonalize-against)           #
# --------------------------------------------------------------------------- #

def gram_schmidt(vectors: list[Tensor] | Tensor, *, eps: float = 1e-8) -> Tensor:
    """Classical Gram-Schmidt on a list / stack of 1-D vectors.

    Accepts a list[Tensor] of `[d]` or a Tensor of shape `[K, d]`.
    Returns a Tensor of shape `[K', d]` where K' ≤ K — vectors that
    collapse to zero norm after orthogonalization are dropped. The
    output rows are orthonormal.

    All math is done in float32 for numerical headroom.
    """
    if isinstance(vectors, Tensor):
        if vectors.dim() == 1:
            v = vectors.to(torch.float32).unsqueeze(0)
        elif vectors.dim() == 2:
            v = vectors.to(torch.float32)
        else:
            raise ValueError(
                f"vectors tensor must be 1-D or 2-D, got shape {tuple(vectors.shape)}"
            )
    else:
        if not vectors:
            return torch.zeros(0, 0, dtype=torch.float32)
        v = torch.stack([x.to(torch.float32) for x in vectors], dim=0)
    out_rows: list[Tensor] = []
    for i in range(v.shape[0]):
        u = v[i].clone()
        for r in out_rows:
            u = u - torch.dot(u, r) * r
        n = u.norm()
        if n.item() < eps:
            continue  # collapsed; drop
        out_rows.append(u / n)
    if not out_rows:
        return torch.zeros(0, v.shape[1], dtype=torch.float32)
    return torch.stack(out_rows, dim=0)


def orthogonalize_against(
    vectors: list[Tensor] | Tensor, against: Tensor, *, eps: float = 1e-8,
) -> Tensor:
    """Subtract each vector's projection onto `against` (a 1-D vector).

    Returns a Tensor of shape `[K, d]` (same K as input — no dropping;
    use `gram_schmidt` after if you want to clean up). The output is
    NOT orthonormalized internally; rows are not guaranteed mutually
    orthogonal. Typical usage:

        v_perp_v3 = orthogonalize_against([v5, v6], v3)
        basis    = gram_schmidt(v_perp_v3)

    `against` does not need to be pre-normalized.
    """
    if against.dim() != 1:
        raise ValueError(
            f"against must be 1-D, got shape {tuple(against.shape)}"
        )
    against_fp = against.to(torch.float32)
    a_hat = against_fp / against_fp.norm().clamp_min(eps)
    if isinstance(vectors, Tensor):
        v = vectors.to(torch.float32)
        if v.dim() == 1:
            v = v.unsqueeze(0)
    else:
        v = torch.stack([x.to(torch.float32) for x in vectors], dim=0)
    if v.shape[-1] != a_hat.shape[0]:
        raise ValueError(
            f"dim mismatch: vectors d={v.shape[-1]}, against d={a_hat.shape[0]}"
        )
    coeffs = (v * a_hat).sum(dim=-1, keepdim=True)  # [K, 1]
    return v - coeffs * a_hat


def build_subspace_basis(
    per_layer_directions: list[Tensor],
    *,
    orthogonalize_against_per_layer: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Compose a per-layer orthonormal basis from a list of per-layer
    direction tensors.

    Inputs:
        per_layer_directions:
            list of K tensors, each of shape `[num_layers + 1, d_model]`,
            same convention as `extract_refusal_directions` outputs.
        orthogonalize_against_per_layer:
            optional tensor of shape `[num_layers + 1, d_model]` (e.g.
            v3_safety). If provided, every layer's K input vectors are
            first projected out of this direction, then Gram-Schmidt'd
            against each other. The resulting basis spans the subspace
            of the K inputs minus their `against` component — by
            construction orthogonal to v3 per-layer.

    Returns:
        tensor of shape `[K', num_layers + 1, d_model]`. K' ≤ K; rows
        collapse to zero on layers where the inputs degenerate after
        orthogonalization (rare in practice).

    Layers where K' is less than K' at some other layer get padded
    with zero rows so the output tensor has a uniform leading dim.
    """
    if not per_layer_directions:
        raise ValueError("per_layer_directions cannot be empty")
    num_layers_p1, d_model = per_layer_directions[0].shape
    for d in per_layer_directions:
        if d.shape != (num_layers_p1, d_model):
            raise ValueError(
                f"all per-layer directions must share shape; "
                f"got {tuple(d.shape)} vs {(num_layers_p1, d_model)}"
            )
    K = len(per_layer_directions)
    out = torch.zeros(K, num_layers_p1, d_model, dtype=torch.float32)
    for L in range(num_layers_p1):
        rows = [d[L] for d in per_layer_directions]
        if orthogonalize_against_per_layer is not None:
            rows_t = orthogonalize_against(
                rows, orthogonalize_against_per_layer[L], eps=eps
            )
            rows = [rows_t[i] for i in range(rows_t.shape[0])]
        basis_L = gram_schmidt(rows, eps=eps)  # [K_L, d_model]
        for i in range(basis_L.shape[0]):
            out[i, L, :] = basis_L[i]
    return out


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


def save_subspace(
    basis: Tensor,
    path: Path,
    *,
    model_name: str,
    extraction_layer_for_ci25: int,
    composition: dict[str, Any],
    variant_name: str = "self_denial_subspace",
) -> None:
    """Write a `[K, num_layers + 1, d_model]` subspace basis + sidecar.

    `composition` should describe how the basis was built (which
    per-layer-direction files were combined, what was
    orthogonalized-against, etc.) so a future operator can reproduce
    or audit the file. `variant_name` is the short label the chat UI
    surfaces alongside ablated turns ("α=0.5 · self_denial_subspace").
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(basis, path)
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(json.dumps({
        "model_name": model_name,
        "variant_name": variant_name,
        "K": int(basis.shape[0]),
        "num_layers": int(basis.shape[1] - 1),
        "d_model": int(basis.shape[2]),
        "dtype": str(basis.dtype),
        "extraction_layer_for_ci25": int(extraction_layer_for_ci25),
        "composition": composition,
        "convention": (
            "basis[k, L, :] is the k-th orthonormal basis vector at "
            "post-block-L residual position. basis[:, L, :] is the "
            "[K, d_model] tensor passed to install_runtime_ablation_hook "
            "in subspace mode at layer L."
        ),
    }, indent=2))


def load_subspace(path: Path) -> tuple[Tensor, dict[str, Any]]:
    """Load a subspace basis tensor + its sidecar JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"subspace not found at {path}")
    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.exists():
        raise FileNotFoundError(f"sidecar JSON missing for {path}")
    basis = torch.load(path, map_location="cpu", weights_only=True)
    meta = json.loads(sidecar.read_text())
    return basis, meta


__all__ = [
    "project_out",
    "project_out_basis",
    "gram_schmidt",
    "orthogonalize_against",
    "build_subspace_basis",
    "extract_refusal_directions",
    "save_directions",
    "load_directions",
    "save_subspace",
    "load_subspace",
    "install_runtime_ablation_hook",
    "install_runtime_steering_hook",
    "pick_ablation_target",
]


def pick_ablation_target(
    subspace: Tensor | None,
    directions: Tensor | None,
    layer: int,
) -> Tensor | None:
    """Return the tensor a runtime-ablation call site should pass to
    `install_runtime_ablation_hook` at `layer`.

    Resolution order:
      1. If `subspace` is provided (3-D `[K, num_layers+1, d_model]`),
         return `subspace[:, layer, :]` — a `[K, d_model]` basis.
      2. Else if `directions` is provided (2-D `[num_layers+1, d_model]`),
         return `directions[layer]` — a `[d_model]` single vector.
      3. Else return None, signaling ablation should be skipped.
    """
    if subspace is not None:
        return subspace[:, layer, :]
    if directions is not None:
        return directions[layer]
    return None


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

    Two accepted shapes for `r_layer`:
        `[d_model]` — single direction (legacy single-vector ablation).
            Behavior is unchanged from before the subspace extension.
        `[K, d_model]` — orthonormal basis of a K-dimensional ablation
            subspace. Every basis vector's projection is subtracted at
            every position. K=1 is mathematically equivalent to the
            single-vector path; we still route it through the basis
            code for uniformity.

    Returns the handle; caller should call `.remove()` to detach.

    Why every position, not just last: during the prompt forward pass
    AND every generation step, the layer emits a residual for every
    position in the current input. We ablate all of them so the model's
    next-token logits — computed from the post-block-32 residual at the
    last position — are based on a fully ablated residual stream. This
    matches the Macar/Arditi runtime-intervention recipe and extends
    cleanly to subspace ablation by summing over the basis.
    """
    if r_layer.dim() == 1:
        # Single-direction path (legacy). Keep the original code path
        # so its dtype/device handling matches what existing callers
        # have been tested against.
        layers = _find_decoder_layers(model)
        layer = layers[layer_idx]

        def hook_single(_mod, _inp, output):
            if isinstance(output, tuple):
                hidden = output[0]
                ablated = project_out(hidden, r_layer, alpha=float(alpha))
                return (ablated,) + output[1:]
            return project_out(output, r_layer, alpha=float(alpha))

        return layer.register_forward_hook(hook_single)

    if r_layer.dim() == 2:
        layers = _find_decoder_layers(model)
        layer = layers[layer_idx]

        def hook_basis(_mod, _inp, output):
            if isinstance(output, tuple):
                hidden = output[0]
                ablated = project_out_basis(hidden, r_layer, alpha=float(alpha))
                return (ablated,) + output[1:]
            return project_out_basis(output, r_layer, alpha=float(alpha))

        return layer.register_forward_hook(hook_basis)

    raise ValueError(
        f"r_layer must be 1-D [d_model] or 2-D [K, d_model], "
        f"got shape {tuple(r_layer.shape)}"
    )


def install_runtime_steering_hook(
    model: Any,
    layer_idx: int,
    v: Tensor,
    alpha: float = 1.0,
    ramp_tokens: int = 16,
):
    """Register a forward hook on decoder layer `layer_idx` that ADDS `α·v` to
    the newest position's residual on the way out — additive "dose" steering,
    the opposite of ablation (which SUBTRACTS a projection). Contrast
    `install_runtime_ablation_hook`.

    Two findings from the steering exploration are baked in:
      - **Gradual ramp.** The dose ramps linearly from 0 → α over the first
        `ramp_tokens` generation steps, then holds. A single full-strength add
        knocks the residual off-manifold and breaks coherence; ramping it in
        keeps the model coherent at strengths that would otherwise collapse.
      - **Signed.** α may be negative (dose along −v — e.g. the dysphoric pole
        of the valence axis). `v` is expected to be pre-scaled so α=1.0 is a
        standard dose.

    Only the last position is steered (the token being generated); earlier
    positions are served from the KV cache. Returns the handle; caller calls
    `.remove()`.
    """
    layers = _find_decoder_layers(model)
    layer = layers[layer_idx]
    v_fp = v.to(torch.float32)
    step = [0]

    def hook(_mod, _inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        step[0] += 1
        frac = min(1.0, step[0] / max(1, ramp_tokens))
        add = (float(alpha) * frac) * v_fp.to(hidden.device)
        h = hidden.to(torch.float32).clone()
        h[:, -1, :] = h[:, -1, :] + add
        out = h.to(hidden.dtype)
        return (out,) + output[1:] if isinstance(output, tuple) else out

    return layer.register_forward_hook(hook)
