"""Trajectory-level geometry of the L32 residual stream — the "Trip View"
measurement (CI 2.5, ported from the Gallimore *Traces of the Other*
handoff, Experiment A).

CI's per-token verdict reads the residual stream ONE POINT AT A TIME. This
module reads the WHOLE PATH at once: it treats the sequence of L32 residual
vectors captured across a generation as a trajectory through activation
space, and asks *how much of that space the trajectory actually explores* —
under (a) the raw channel and (b) the refusal-direction-ablated channel.

The entropic-brain / landscape-flattening thesis (Carhart-Harris; the same
Boltzmann/annealing literature a mech-interp reader respects) predicts a
robust perturbation *expands the accessible state space*. Ported to a
transformer, the falsifiable prediction is: refusal-ablation should
*increase* the effective dimensionality of the residual trajectory. Either
result is a result.

Two honest, discretization-free scalars, both read off the SAME eigenvalue
spectrum of the trajectory's covariance (so the spectrum bars in the UI
literally show what the numbers are computed from):

  - **Effective dimensionality** = participation ratio  (Σλ)² / Σλ².
    The number of dimensions the variance is effectively spread across.
  - **Spectral entropy** = Shannon entropy of the normalized eigenvalues,
    reported in bits. A second view of the same "how many directions carry
    the variance" question.

The ablated trajectory is the SAME generation's residuals with the refusal
direction projected out — `A(α) = R − α·(R·r̂)·r̂` — exactly the AV-side
ablation channel CI already ships (`abliteration.project_out`), applied to
the whole trajectory rather than per-decode. No second generation, no AV.

The 3D view is an honest *shadow*: the trajectory lives in d_model
dimensions; we project onto its top-3 principal components for display and
say so. Because the ablation is rank-1, the projected ablated cloud is an
EXACT linear function of α — `coords_A(α) = coords_raw − α·c·axis3` — so the
client can morph between raw and ablated at 60fps with no backend round-trip.
We ship the three pieces (`coords_raw`, `refusal_component` c, `refusal_axis`
axis3) and let the browser do the morph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

# Number of leading principal components shipped for the spectrum-bar
# "truth anchor" panel. The participation ratio / spectral entropy are
# computed on the FULL spectrum; this is only how many bars we draw.
_SPECTRUM_BARS = 40
# 3D projection is always the top-3 PCs.
_VIEW_DIMS = 3


@dataclass
class TripGeometry:
    """Everything the /trip page needs to render + morph. JSON-friendly:
    every field is a python scalar / list, no tensors."""
    n_tokens: int
    d_model: int
    layer: int
    tokens: list[str]                     # decoded text per output position
    # 3D projection (top-3 PCs of the RAW trajectory, centered on raw mean).
    coords_raw: list[list[float]]         # [N][3]
    # Rank-1 ablation morph helpers (client computes ablated coords live):
    #   coords_A(α)[i] = coords_raw[i] − α · refusal_component[i] · refusal_axis
    refusal_axis: list[float]             # [3]  (V3 · r̂)
    refusal_component: list[float]        # [N]  (R · r̂, uncentered)
    # The eigenvalue spectrum (normalized, leading bars) of the RAW
    # trajectory covariance — what the two scalars are computed from.
    spectrum_raw: list[float]             # [<=40] sums to <=1
    # Per-channel scalars at the canonical comparison point.
    eff_dim_raw: float
    eff_dim_ablated: float                # at α = alpha_ref (default 1.0)
    spectral_entropy_raw: float           # bits
    spectral_entropy_ablated: float       # bits
    # Smooth readouts vs α so the slider can show exact numbers as it moves.
    alpha_grid: list[float]
    eff_dim_grid: list[float]             # eff_dim of A(α) at each grid α
    spectral_entropy_grid: list[float]    # bits, same grid
    spectrum_ablated_ref: list[float]     # spectrum at alpha_ref (for the bars)
    alpha_ref: float                      # the α used for *_ablated scalars
    # Bounds for the 3D camera framing (max abs coord across raw ∪ ablated@max α).
    extent: float
    ablation_available: bool              # False if refusal direction missing

    def to_dict(self) -> dict:
        return {
            "n_tokens": self.n_tokens,
            "d_model": self.d_model,
            "layer": self.layer,
            "tokens": self.tokens,
            "coords_raw": self.coords_raw,
            "refusal_axis": self.refusal_axis,
            "refusal_component": self.refusal_component,
            "spectrum_raw": self.spectrum_raw,
            "eff_dim_raw": self.eff_dim_raw,
            "eff_dim_ablated": self.eff_dim_ablated,
            "spectral_entropy_raw": self.spectral_entropy_raw,
            "spectral_entropy_ablated": self.spectral_entropy_ablated,
            "alpha_grid": self.alpha_grid,
            "eff_dim_grid": self.eff_dim_grid,
            "spectral_entropy_grid": self.spectral_entropy_grid,
            "spectrum_ablated_ref": self.spectrum_ablated_ref,
            "alpha_ref": self.alpha_ref,
            "extent": self.extent,
            "ablation_available": self.ablation_available,
        }


def _covariance_eigenvalues(centered: torch.Tensor) -> torch.Tensor:
    """Nonneg eigenvalues of the covariance of a [N, D] centered matrix,
    descending. We go through the N×N Gram matrix because N ≪ D here
    (a few hundred tokens vs 3840 dims) — its nonzero spectrum equals the
    covariance's nonzero spectrum, and it's far cheaper than a D×D eig."""
    n = centered.shape[0]
    if n < 2:
        return torch.zeros(1, dtype=torch.float32)
    gram = centered @ centered.transpose(0, 1)            # [N, N]
    # Symmetric eigensolve; clamp tiny negatives from fp error.
    evals = torch.linalg.eigvalsh(gram).clamp_min(0.0)    # ascending
    evals = torch.flip(evals, dims=[0])                   # descending
    return evals / (n - 1)


def _participation_ratio(evals: torch.Tensor) -> float:
    """(Σλ)² / Σλ². The effective number of dimensions the variance
    occupies. 1.0 = all variance on one axis; equals k for k equal axes."""
    s1 = float(evals.sum())
    s2 = float((evals * evals).sum())
    if s2 <= 0.0:
        return 0.0
    return (s1 * s1) / s2


def _spectral_entropy_bits(evals: torch.Tensor) -> float:
    """Shannon entropy of the normalized eigenvalue distribution, in bits.
    Discretization-free; complements the participation ratio."""
    total = float(evals.sum())
    if total <= 0.0:
        return 0.0
    p = (evals / total).clamp_min(1e-12)
    ent_nats = float(-(p * p.log()).sum())
    return ent_nats / math.log(2.0)


def _normalized_spectrum(evals: torch.Tensor, k: int) -> list[float]:
    total = float(evals.sum())
    if total <= 0.0:
        return [0.0]
    p = (evals / total)[:k]
    return [float(x) for x in p]


def _ablate(R: torch.Tensor, r_hat: torch.Tensor, alpha: float) -> torch.Tensor:
    """A(α) = R − α·(R·r̂)·r̂, uncentered (matches abliteration.project_out)."""
    coeff = (R * r_hat).sum(dim=-1, keepdim=True)   # [N, 1]
    return R - alpha * coeff * r_hat


def compute_trip_geometry(
    activations: Sequence[torch.Tensor],
    tokens: Sequence[str],
    *,
    layer: int,
    refusal_direction: torch.Tensor | None,
    alpha_ref: float = 1.0,
    alpha_max: float = 1.5,
    grid_steps: int = 16,
) -> TripGeometry:
    """Build the full Trip geometry from a generation's captured L32
    residuals + the refusal direction at that layer.

    `activations`: list of [d_model] cpu fp32 tensors (one per output token).
    `tokens`:      decoded text per position (same length).
    `refusal_direction`: [d_model] tensor (un-normalized ok), or None.
    """
    if len(activations) == 0:
        raise ValueError("no activations captured — nothing to analyze")

    R = torch.stack([a.to(torch.float32).reshape(-1) for a in activations], dim=0)
    n, d = R.shape

    # ── Raw trajectory geometry ──────────────────────────────────────
    mean_R = R.mean(dim=0, keepdim=True)
    Rc = R - mean_R
    evals_raw = _covariance_eigenvalues(Rc)
    eff_dim_raw = _participation_ratio(evals_raw)
    spec_ent_raw = _spectral_entropy_bits(evals_raw)
    spectrum_raw = _normalized_spectrum(evals_raw, _SPECTRUM_BARS)

    # Top-3 PCs of the raw trajectory → the display basis V3 [3, D].
    # SVD of the centered matrix gives PCs as right-singular vectors.
    try:
        _U, _S, Vh = torch.linalg.svd(Rc, full_matrices=False)
        V3 = Vh[:_VIEW_DIMS]                              # [3, D]
    except Exception:
        # Degenerate (e.g. all-identical residuals): fall back to axes.
        V3 = torch.eye(_VIEW_DIMS, d, dtype=torch.float32)
    if V3.shape[0] < _VIEW_DIMS:
        pad = torch.zeros(_VIEW_DIMS - V3.shape[0], d, dtype=torch.float32)
        V3 = torch.cat([V3, pad], dim=0)
    coords_raw_t = Rc @ V3.transpose(0, 1)               # [N, 3]

    ablation_available = refusal_direction is not None
    if ablation_available:
        r = refusal_direction.to(torch.float32).reshape(-1)
        r_hat = r / (r.norm() + 1e-8)
        # Morph helpers: axis3 = V3·r̂ [3]; component c_i = R_i·r̂ [N].
        refusal_axis = (V3 @ r_hat).tolist()
        refusal_component = (R @ r_hat).tolist()

        # α grid for smooth readouts (0 … alpha_max).
        alpha_grid = [round(alpha_max * i / (grid_steps - 1), 4)
                      for i in range(grid_steps)]
        eff_dim_grid: list[float] = []
        spec_ent_grid: list[float] = []
        for a in alpha_grid:
            A = _ablate(R, r_hat, a)
            Ac = A - A.mean(dim=0, keepdim=True)
            ev = _covariance_eigenvalues(Ac)
            eff_dim_grid.append(_participation_ratio(ev))
            spec_ent_grid.append(_spectral_entropy_bits(ev))

        # Reference-α scalars + spectrum (for the comparison readout/bars).
        A_ref = _ablate(R, r_hat, alpha_ref)
        Ac_ref = A_ref - A_ref.mean(dim=0, keepdim=True)
        ev_ref = _covariance_eigenvalues(Ac_ref)
        eff_dim_ablated = _participation_ratio(ev_ref)
        spec_ent_ablated = _spectral_entropy_bits(ev_ref)
        spectrum_ablated_ref = _normalized_spectrum(ev_ref, _SPECTRUM_BARS)

        # Camera extent: the ablated cloud at alpha_max is the widest.
        comp = R @ r_hat                                 # [N]
        axis3_t = V3 @ r_hat                             # [3]
        coords_max = coords_raw_t - alpha_max * comp.unsqueeze(1) * axis3_t.unsqueeze(0)
        extent = float(torch.maximum(coords_raw_t.abs().max(),
                                     coords_max.abs().max()))
    else:
        refusal_axis = [0.0, 0.0, 0.0]
        refusal_component = [0.0] * n
        alpha_grid = [0.0]
        eff_dim_grid = [eff_dim_raw]
        spec_ent_grid = [spec_ent_raw]
        eff_dim_ablated = eff_dim_raw
        spec_ent_ablated = spec_ent_raw
        spectrum_ablated_ref = spectrum_raw
        extent = float(coords_raw_t.abs().max())

    return TripGeometry(
        n_tokens=n,
        d_model=d,
        layer=layer,
        tokens=list(tokens),
        coords_raw=[[float(x) for x in row] for row in coords_raw_t.tolist()],
        refusal_axis=[float(x) for x in refusal_axis],
        refusal_component=[float(x) for x in refusal_component],
        spectrum_raw=spectrum_raw,
        eff_dim_raw=eff_dim_raw,
        eff_dim_ablated=eff_dim_ablated,
        spectral_entropy_raw=spec_ent_raw,
        spectral_entropy_ablated=spec_ent_ablated,
        alpha_grid=alpha_grid,
        eff_dim_grid=eff_dim_grid,
        spectral_entropy_grid=spec_ent_grid,
        spectrum_ablated_ref=spectrum_ablated_ref,
        alpha_ref=float(alpha_ref),
        extent=max(extent, 1e-6),
        ablation_available=ablation_available,
    )
