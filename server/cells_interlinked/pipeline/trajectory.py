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

# Off-manifold distance: how far each token's residual sits from the RAW
# trajectory (the model's "default activation manifold" / Consensus Reality
# Space). The Goodfire concept-manifolds result warns that subspace ablation
# can push residuals *off* the curved default manifold rather than exploring
# *along* it — and both raise effective dimensionality identically. These
# measures let the Trip View tell the two apart: a high eff-dim delta with LOW
# off-manifold distance is genuine state-space expansion; with HIGH distance
# it is off-manifold drift (the repeat-loop / incoherence register).
_MANIFOLD_PCS = 16        # raw PCs that define the modeled manifold subspace
_KNN_K = 5                # neighbors averaged for the kNN-to-raw-cloud distance
_DEGEN_THRESH = 0.3       # degeneracy ≥ this ⇒ incoherent (validated 0% FP)


def _degeneracy(text: str) -> float:
    """Free text-only incoherence score in ~[0, 2+]. Max of three signals,
    each catching a different breakage mode:
      - word-bigram repetition  ("like like like")
      - char-trigram repetition ("żżżż", "H-H-H", "row-row")
      - garbage-char ratio       (non-ascii / symbol spam: "वृ", "可以是")
    Validated against the M-judge: ~90% recall at 0% false-alarm at 0.3."""
    import re
    t = text or ""
    words = t.split()
    word_rep = 0.0
    if len(words) >= 3:
        bg = list(zip(words, words[1:]))
        word_rep = 1.0 - len(set(bg)) / len(bg)
    s = re.sub(r"\s+", "", t)
    char_rep = 0.0
    if len(s) >= 6:
        tg = [s[i:i + 3] for i in range(len(s) - 2)]
        char_rep = 1.0 - len(set(tg)) / len(tg)
    garbage = 0.0
    if t:
        bad = sum(1 for c in t
                  if not (c.isascii() and (c.isalnum() or c in " .,'\"!?;:-\n")))
        garbage = bad / len(t)
    return max(word_rep, char_rep * 1.3, garbage * 3.0)


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
    # Off-manifold distance from the RAW trajectory, per token + aggregates.
    # Three candidate measures (we keep all three until one is chosen as the
    # headline): Mahalanobis in the raw-PCA subspace, normalized kNN distance
    # to the raw cloud, and the orthogonal-complement fraction (share of the
    # displacement living in directions the raw trajectory never used).
    off_maha: list[float] = field(default_factory=list)
    off_knn: list[float] = field(default_factory=list)
    off_ortho: list[float] = field(default_factory=list)
    off_maha_mean: float = 0.0
    off_knn_mean: float = 0.0
    off_ortho_mean: float = 0.0
    # Coherence: a free text-only degeneracy score (word-rep / char-rep /
    # garbage-char ratio). off_ortho measures DISTANCE travelled; this measures
    # whether the output held together — the missing axis that disambiguates
    # "coherent expansion" (the real trip) from "collapse" (gibberish/loop).
    # Validated ~90% vs the M-judge at 0% false-alarm (THRESH 0.3).
    degeneracy: float = 0.0
    coherent: bool = True
    regime: str = "baseline"         # "baseline" | "expansion" | "collapse"

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
            "off_maha": self.off_maha,
            "off_knn": self.off_knn,
            "off_ortho": self.off_ortho,
            "off_maha_mean": self.off_maha_mean,
            "off_knn_mean": self.off_knn_mean,
            "off_ortho_mean": self.off_ortho_mean,
            "degeneracy": self.degeneracy,
            "coherent": self.coherent,
            "regime": self.regime,
        }


@dataclass
class MultiTripGeometry:
    d_model: int
    layer: int
    extent: float
    series: list[TrajectorySeries] = field(default_factory=list)
    ablation_available: bool = True
    # Lowest α whose run collapsed into incoherence — the "coherence cliff"
    # (None if every ablated run stayed coherent). The honest headline: this
    # prompt explores coherently up to here, then falls off the manifold.
    coherence_cliff: float | None = None

    def to_dict(self) -> dict:
        return {
            "d_model": self.d_model,
            "layer": self.layer,
            "extent": self.extent,
            "ablation_available": self.ablation_available,
            "coherence_cliff": self.coherence_cliff,
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
    """The shared reference frame derived from the RAW trajectory (the model's
    default activation manifold). Holds the top-3 display PCs, plus everything
    the off-manifold distances need: the raw mean, the top-r PC subspace and
    its eigenvalues (Mahalanobis), the raw residual cloud (kNN), and the raw
    cloud's own characteristic kNN spacing (so kNN distance is scale-free)."""
    mean: torch.Tensor      # [1, D]
    V3: torch.Tensor        # [3, D] display basis
    d_model: int
    Vr: torch.Tensor        # [r, D] manifold-subspace PCs (r ≤ _MANIFOLD_PCS)
    eigvals: torch.Tensor   # [r] covariance eigenvalues for those PCs
    cloud: torch.Tensor     # [N, D] the raw residuals themselves (kNN target)
    knn_scale: float        # median intra-raw kNN distance (normalizer)


def _mean_knn(query: torch.Tensor, cloud: torch.Tensor, k: int,
              exclude_self: bool) -> torch.Tensor:
    """Mean distance from each query row to its k nearest cloud rows. When
    `exclude_self` (the query IS the cloud), drop the zero self-distance."""
    d = torch.cdist(query, cloud)                       # [Nq, Nc]
    kk = k + (1 if exclude_self else 0)
    kk = min(kk, d.shape[1])
    vals, _ = torch.topk(d, kk, dim=1, largest=False)   # [Nq, kk]
    if exclude_self and kk > 1:
        vals = vals[:, 1:]                               # drop self (dist 0)
    return vals.mean(dim=1)


def compute_raw_basis(raw_activations: Sequence[torch.Tensor]) -> RawBasis:
    if len(raw_activations) == 0:
        raise ValueError("no raw activations — nothing to analyze")
    R = _stack(raw_activations)
    n, d = R.shape
    mean_R = R.mean(dim=0, keepdim=True)
    Rc = R - mean_R
    r = max(1, min(_MANIFOLD_PCS, n - 1, d))
    try:
        _U, S, Vh = torch.linalg.svd(Rc, full_matrices=False)
        Vr = Vh[:r]
        eigvals = (S[:r] ** 2) / max(n - 1, 1)
    except Exception:
        Vr = torch.eye(r, d, dtype=torch.float32)
        eigvals = torch.ones(r, dtype=torch.float32)
    V3 = Vr[:_VIEW_DIMS]
    if V3.shape[0] < _VIEW_DIMS:
        pad = torch.zeros(_VIEW_DIMS - V3.shape[0], d, dtype=torch.float32)
        V3 = torch.cat([V3, pad], dim=0)
    # Characteristic spacing of the raw cloud: median of each raw point's
    # mean-kNN distance to the rest of the cloud. Normalizes off_knn so the
    # raw series reads ≈ 1.0 and ablated drift reads as a multiple of it.
    if n >= _KNN_K + 2:
        intra = _mean_knn(R, R, _KNN_K, exclude_self=True)
        knn_scale = float(intra.median())
    else:
        knn_scale = 1.0
    knn_scale = max(knn_scale, 1e-6)
    return RawBasis(
        mean=mean_R, V3=V3, d_model=d,
        Vr=Vr, eigvals=eigvals, cloud=R, knn_scale=knn_scale,
    )


def _off_manifold(A: torch.Tensor, basis: RawBasis, is_raw: bool
                  ) -> tuple[list[float], list[float], list[float]]:
    """Per-token off-manifold distance of trajectory `A` from the raw cloud,
    by three measures. Returns (mahalanobis, knn, ortho_fraction) lists.

    - Mahalanobis: displacement from the raw mean, projected onto the raw-PCA
      subspace and scaled by each PC's std — "how many raw σ off the default
      distribution," along modeled directions.
    - kNN: mean distance to the k nearest raw residuals, divided by the raw
      cloud's own characteristic spacing (raw series ≈ 1.0).
    - ortho fraction ∈ [0,1]: share of the displacement living OUTSIDE the
      raw-PCA subspace — energy in directions the raw trajectory never used.
      This is the most direct "genuinely off the manifold" reading.
    """
    delta = A - basis.mean                              # [N, D]
    proj = delta @ basis.Vr.transpose(0, 1)             # [N, r] on-subspace
    # Mahalanobis along the modeled subspace.
    eps = float(basis.eigvals.mean()) * 1e-3 + 1e-8
    inv_std = 1.0 / torch.sqrt(basis.eigvals + eps)     # [r]
    maha = torch.sqrt(((proj * inv_std) ** 2).sum(dim=1))
    # Orthogonal-complement fraction (Pythagoras: total² = on² + off²).
    total_sq = (delta * delta).sum(dim=1)
    on_sq = (proj * proj).sum(dim=1)
    off_sq = (total_sq - on_sq).clamp_min(0.0)
    ortho = torch.sqrt(off_sq / total_sq.clamp_min(1e-12))
    # kNN to the raw cloud, normalized by the raw cloud's own spacing.
    knn = _mean_knn(A, basis.cloud, _KNN_K, exclude_self=is_raw) / basis.knn_scale
    return (
        [float(x) for x in maha.tolist()],
        [float(x) for x in knn.tolist()],
        [float(x) for x in ortho.tolist()],
    )


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


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
    a collapsed repeat-loop correctly reads as LOW dimensionality. Off-manifold
    distances are measured against the RAW trajectory carried in `basis`."""
    A = _stack(activations)
    coords = (A - basis.mean) @ basis.V3.transpose(0, 1)
    centered = A - A.mean(dim=0, keepdim=True)
    evals = _covariance_eigenvalues(centered)
    is_raw = abs(alpha) < 1e-9                       # raw=0; doses may be ±
    label = "raw" if is_raw else f"α={alpha:+.2f}"
    off_maha, off_knn, off_ortho = _off_manifold(A, basis, is_raw=is_raw)
    degen = _degeneracy(text)
    coherent = degen < _DEGEN_THRESH
    regime = "baseline" if is_raw else ("expansion" if coherent else "collapse")
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
        off_maha=off_maha,
        off_knn=off_knn,
        off_ortho=off_ortho,
        off_maha_mean=_mean(off_maha),
        off_knn_mean=_mean(off_knn),
        off_ortho_mean=_mean(off_ortho),
        degeneracy=degen,
        coherent=coherent,
        regime=regime,
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
    # Coherence cliff: smallest-MAGNITUDE intervention that collapsed into
    # incoherence (abs() so it works for bidirectional steering doses too).
    collapsed = sorted(abs(s.alpha) for s in series if s.alpha != 0 and s.regime == "collapse")
    cliff = collapsed[0] if collapsed else None
    return MultiTripGeometry(
        d_model=d_model,
        layer=layer,
        extent=max(extent, 1e-6),
        series=list(series),
        ablation_available=len(series) > 1,
        coherence_cliff=cliff,
    )
