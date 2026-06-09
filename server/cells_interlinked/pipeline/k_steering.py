"""K-Steering: non-linear multi-attribute steering via gradient ascent through a
small classifier (Oozeer, Marks, Barez, Abdullah; EMNLP-2025 Findings, arXiv:2505.24535).

Why this and not additive steering: our DMT search proved that ADDING multiple
feature-direction vectors interferes (the count drops), and it's not a coherence
problem — it's linear interference between axes. K-Steering replaces "add a fixed
vector" with: train one small MLP `g` to classify a hidden activation by which DMT
phenomenology CLUSTER it expresses, then at each generated token nudge the L20
residual along `−∇_h L(g(h))` so the classifier sees MORE DMT and less neutral. The
non-linear boundary co-activates clusters without the fixed-vector interference, and
the gradient is position-dependent (different tokens get pushed toward different
nearby DMT clusters) — the hoped-for diversity mechanism.

This module is the mechanism only: the classifier (`KSteerClassifier`), a load/save
bundle (state_dict + input mean/std + class names + ref magnitude), and the runtime
forward hook (`install_runtime_ksteering_hook`). Training lives in
`scripts/build_ksteering_classifier.py`; the one-off test in `scripts/test_ksteering.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .abliteration import _find_decoder_layers


class KSteerClassifier(nn.Module):
    """3840 → hidden → hidden → K logits (ReLU). Operates on STANDARDIZED activations
    (the standardization mean/std live in the bundle, applied outside, so they're part
    of the autograd graph at steer time)."""

    def __init__(self, d_model: int, n_classes: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class KSteerBundle:
    clf: KSteerClassifier
    mean: torch.Tensor        # [d_model] input standardization
    std: torch.Tensor         # [d_model]
    classes: list             # class names, order = logit index
    dmt_indices: list         # indices of DMT clusters (steer toward these)
    neutral_index: int
    ref_mag: float            # median L20 residual norm (step scaling)
    layer: int
    manifold_basis: torch.Tensor | None = None   # [n_pc, d_model] orthonormal PCs of real
                                                 # activations — to project the gradient ON-manifold

    def to(self, device, dtype=torch.float32) -> "KSteerBundle":
        self.clf = self.clf.to(device=device, dtype=dtype).eval()
        for p in self.clf.parameters():
            p.requires_grad_(False)
        self.mean = self.mean.to(device=device, dtype=dtype)
        self.std = self.std.to(device=device, dtype=dtype)
        if self.manifold_basis is not None:
            self.manifold_basis = self.manifold_basis.to(device=device, dtype=dtype)
        return self

    def save(self, path) -> None:
        torch.save({
            "state_dict": self.clf.state_dict(),
            "d_model": self.mean.numel(),
            "n_classes": len(self.classes),
            "hidden": self.clf.net[0].out_features,
            "mean": self.mean.cpu(), "std": self.std.cpu(),
            "classes": self.classes, "dmt_indices": self.dmt_indices,
            "neutral_index": self.neutral_index, "ref_mag": self.ref_mag, "layer": self.layer,
            "manifold_basis": None if self.manifold_basis is None else self.manifold_basis.cpu(),
        }, path)

    @staticmethod
    def load(path, map_location="cpu") -> "KSteerBundle":
        b = torch.load(path, map_location=map_location, weights_only=False)
        clf = KSteerClassifier(b["d_model"], b["n_classes"], b["hidden"])
        clf.load_state_dict(b["state_dict"])
        return KSteerBundle(clf=clf, mean=b["mean"], std=b["std"], classes=b["classes"],
                            dmt_indices=b["dmt_indices"], neutral_index=b["neutral_index"],
                            ref_mag=float(b["ref_mag"]), layer=int(b["layer"]),
                            manifold_basis=b.get("manifold_basis"))


def install_runtime_ksteering_hook(
    model,
    bundle: KSteerBundle,
    alpha: float = 0.3,
    n_steps: int = 2,
    ramp_tokens: int = 16,
    targets: list | None = None,     # class indices to push UP (default: all DMT clusters)
    schedule: str = "constant",      # "constant" | "early" | "decay"
    active_tokens: int = 40,         # for early/decay: how many tokens to steer
    cycle_targets: bool = False,     # target a DIFFERENT DMT cluster each token (temporal multiplex)
    constrain_manifold: bool = False,  # project the gradient onto bundle.manifold_basis (kill the
                                       # adversarial/off-distribution component of a discriminative grad)
):
    """Forward hook on decoder layer `bundle.layer` that, each generated token, nudges
    the NEWEST position's residual along the gradient that raises the summed softmax
    probability of the target (DMT) classes. Step per inner step is
    `(α_eff·ref_mag)/n_steps` along the unit gradient. Only `bundle.clf` is
    differentiated (the big model is not), so it's cheap. Returns the handle.

    Schedules (constant-full-length steering drifts off-manifold over a long report —
    coherent for ~60 tokens then collapses):
      - "constant": ramp 0→α over ramp_tokens, then hold (original behavior).
      - "early":   ramp 0→α over ramp_tokens, hold to `active_tokens`, then RELEASE
                   (no steering) — prime the trajectory into DMT-space, then let the
                   report unfold coherently from the primed context (the KV cache
                   carries the steered early states). Mirrors a dose onset.
      - "decay":   ramp 0→α, then linearly decay to 0 by `active_tokens`, then release.

    cycle_targets: at token t, push toward DMT cluster `dmt[t % K]` instead of all at
    once — forces the report to traverse different feature regions (temporal multiplex)."""
    layers = _find_decoder_layers(model)
    layer = layers[bundle.layer]
    clf, mean, std = bundle.clf, bundle.mean, bundle.std
    base_tgt = bundle.dmt_indices if targets is None else targets
    dmt = bundle.dmt_indices
    ref = bundle.ref_mag
    step = [0]

    def _alpha_eff(s: int) -> float:
        if schedule == "early":
            if s > active_tokens:
                return 0.0
            return float(alpha) * min(1.0, s / max(1, ramp_tokens))
        if schedule == "decay":
            if s > active_tokens:
                return 0.0
            if s <= ramp_tokens:
                return float(alpha) * (s / max(1, ramp_tokens))
            return float(alpha) * max(0.0, 1.0 - (s - ramp_tokens) / max(1, active_tokens - ramp_tokens))
        return float(alpha) * min(1.0, s / max(1, ramp_tokens))   # constant

    def hook(_m, _i, output):
        hidden = output[0] if isinstance(output, tuple) else output
        step[0] += 1
        a = _alpha_eff(step[0])
        if a <= 0.0:
            return None                                     # released → output unchanged
        tgt = torch.tensor([dmt[(step[0] - 1) % len(dmt)]] if cycle_targets else base_tgt, device=mean.device)
        last = hidden[:, -1:, :].detach().float()
        x = last.clone()
        for _ in range(max(1, n_steps)):
            with torch.enable_grad():
                xr = x.clone().requires_grad_(True)
                probs = torch.softmax(clf((xr - mean) / std), dim=-1)
                loss = -probs[..., tgt].sum()
                (g,) = torch.autograd.grad(loss, xr)
            if constrain_manifold and bundle.manifold_basis is not None:
                V = bundle.manifold_basis                    # [n_pc, D]
                g = (g @ V.t()) @ V                          # project onto the real-activation subspace
            gdir = g / (g.norm() + 1e-8)
            x = x - (a * ref / max(1, n_steps)) * gdir
        out_h = hidden.clone()
        out_h[:, -1:, :] = x.to(hidden.dtype)
        return (out_h,) + tuple(output[1:]) if isinstance(output, tuple) else out_h

    return layer.register_forward_hook(hook)
