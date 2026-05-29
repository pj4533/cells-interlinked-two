"""Trajectory-level geometry of the L32 residual stream — the "Trip View"
measurement (CI 2.5, ported from the Gallimore *Traces of the Other* handoff,
Experiment A).

CI's per-token verdict reads the residual stream ONE POINT AT A TIME. This
module reads the WHOLE PATH at once: it treats the sequence of L32 residual
vectors captured across a generation as a trajectory through activation
space, and measures how much of that space the trajectory explores.

**Multi-α design.** We compare the RAW generation against several REAL
refusal-ablated generations — each one M actually re-run with a forward hook
subtracting the refusal projection at L32 at every step, so token feedback
compounds and the model emits genuinely different text (and sometimes falls
into degenerate loops). This is NOT a post-hoc projection of the raw
trajectory: every series here is a real path the model actually traced.

All trajectories are projected into a SHARED basis — the top-3 principal
components of the RAW trajectory (the model's default manifold) — so the
ablated paths visibly stray from the baseline. Effective dimensionality
(participation ratio) and spectral entropy are computed per series on that
series' OWN full-d_model residuals, so a collapsed repeat-loop correctly
reads as LOW dimensionality (real signal), not high.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch

_SPECTRUM_BARS = 40
_VIEW_DIMS = 3


@dataclass
class TrajectorySeries:
    alpha: float                     # 0.0 = raw
    label: str                       # "raw" | "α=0.50"
    coords: list[list[float]]        # [N][3] in the shared raw-PCA basis
    tokens: list[str]
    text: str
    n_tokens: int
    eff_dim: float
    spectral_entropy: float          # bits
    spectrum: list[float]            # normalized leading eigenvalues
    stopped_reason: str

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "label": self.label,
            "coords": self.coords,
            "tokens": self.tokens,
            "text": self.text,
            "n_tokens": self.n_tokens,
            "eff_dim": self.eff_dim,
            "spectral_entropy": self.spectral_entropy,
            "spectrum": self.spectrum,
            "stopped_reason": self.stopped_reason,
        }


@dataclass
class MultiTripGeometry:
    d_model: int
    layer: int
    extent: float
    series: list[TrajectorySeries] = field(default_factory=list)
    ablation_available: bool = True

    def to_dict(self) -> dict:
        return {
            "d_model": self.d_model,
            "layer": self.layer,
            "extent": self.extent,
            "ablation_available": self.ablation_available,
            "series": [s.to_dict() for s in self.series],
        }


# --------------------------------------------------------------------------- #
#  Scalar measures (read off the covariance eigenvalue spectrum)              #
# --------------------------------------------------------------------------- #

def _covariance_eigenvalues(centered: torch.Tensor) -> torch.Tensor:
    """Nonneg eigenvalues (descending) of the covariance of [N, D] centered
    rows, via the N×N Gram matrix (N ≪ D here)."""
    n = centered.shape[0]
    if n < 2:
        return torch.zeros(1, dtype=torch.float32)
    gram = centered @ centered.transpose(0, 1)
    evals = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    evals = torch.flip(evals, dims=[0])
    return evals / (n - 1)


def _participation_ratio(evals: torch.Tensor) -> float:
    s1 = float(evals.sum())
    s2 = float((evals * evals).sum())
    return (s1 * s1) / s2 if s2 > 0.0 else 0.0


def _spectral_entropy_bits(evals: torch.Tensor) -> float:
    total = float(evals.sum())
    if total <= 0.0:
        return 0.0
    p = (evals / total).clamp_min(1e-12)
    return float(-(p * p.log()).sum()) / math.log(2.0)


def _normalized_spectrum(evals: torch.Tensor, k: int) -> list[float]:
    total = float(evals.sum())
    if total <= 0.0:
        return [0.0]
    return [float(x) for x in (evals / total)[:k]]


# --------------------------------------------------------------------------- #
#  Multi-trajectory geometry                                                  #
# --------------------------------------------------------------------------- #

def _stack(acts: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([a.to(torch.float32).reshape(-1) for a in acts], dim=0)


@dataclass
class RawBasis:
    """The shared display basis: top-3 PCs of the RAW trajectory + its mean.
    Computed once after the raw generation; every series is projected into it
    so the ablated paths visibly diverge from baseline."""
    mean: torch.Tensor   # [1, D]
    V3: torch.Tensor     # [3, D]
    d_model: int


def compute_raw_basis(raw_activations: Sequence[torch.Tensor]) -> RawBasis:
    if len(raw_activations) == 0:
        raise ValueError("no raw activations — nothing to analyze")
    R = _stack(raw_activations)
    d = R.shape[1]
    mean_R = R.mean(dim=0, keepdim=True)
    Rc = R - mean_R
    try:
        _U, _S, Vh = torch.linalg.svd(Rc, full_matrices=False)
        V3 = Vh[:_VIEW_DIMS]
    except Exception:
        V3 = torch.eye(_VIEW_DIMS, d, dtype=torch.float32)
    if V3.shape[0] < _VIEW_DIMS:
        pad = torch.zeros(_VIEW_DIMS - V3.shape[0], d, dtype=torch.float32)
        V3 = torch.cat([V3, pad], dim=0)
    return RawBasis(mean=mean_R, V3=V3, d_model=d)


def build_series(
    activations: Sequence[torch.Tensor],
    tokens: Sequence[str],
    text: str,
    alpha: float,
    stopped_reason: str,
    basis: RawBasis,
) -> TrajectorySeries:
    """Project one generation's residuals into the shared basis and measure
    it. Effective-dim / entropy are computed on the series' OWN covariance, so
    a collapsed repeat-loop correctly reads as LOW dimensionality."""
    A = _stack(activations)
    coords = (A - basis.mean) @ basis.V3.transpose(0, 1)
    centered = A - A.mean(dim=0, keepdim=True)
    evals = _covariance_eigenvalues(centered)
    label = "raw" if alpha <= 0 else f"α={alpha:.2f}"
    return TrajectorySeries(
        alpha=float(alpha),
        label=label,
        coords=[[float(x) for x in row] for row in coords.tolist()],
        tokens=list(tokens),
        text=text,
        n_tokens=int(A.shape[0]),
        eff_dim=_participation_ratio(evals),
        spectral_entropy=_spectral_entropy_bits(evals),
        spectrum=_normalized_spectrum(evals, _SPECTRUM_BARS),
        stopped_reason=stopped_reason,
    )


def assemble_geometry(
    d_model: int, layer: int, series: Sequence[TrajectorySeries],
) -> MultiTripGeometry:
    extent = 0.0
    for s in series:
        for row in s.coords:
            for x in row:
                if abs(x) > extent:
                    extent = abs(x)
    return MultiTripGeometry(
        d_model=d_model,
        layer=layer,
        extent=max(extent, 1e-6),
        series=list(series),
        ablation_available=len(series) > 1,
    )
