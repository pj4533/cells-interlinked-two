"""Decoding-mode definitions: which captured positions get NLA-decoded.

The AV is a one-vector-in-one-paragraph-out instrument. We can't change
that shape without retraining. What we CAN choose is which positions to
feed it. Faster modes trade granularity for wall-clock — useful when you
want a rough cross-run signal at scale rather than a per-token narrative.

per-token   : every output position (slowest, fullest signal)
every-3rd   : stride 3 (~3x faster, smooth narrative)
every-5th   : stride 5 (~5x faster, recommended for batch)
key-points  : 5 positions: first, ~25%, ~50%, ~75%, last (very fast)
"""

from __future__ import annotations

from typing import Literal

DecodingMode = Literal["per-token", "every-3rd", "every-5th", "key-points"]

DECODING_MODES: tuple[DecodingMode, ...] = (
    "per-token",
    "every-3rd",
    "every-5th",
    "key-points",
)


def select_indices(n: int, mode: str) -> list[int]:
    """Return sorted output positions to NLA-decode for a captured run of length n."""
    if n <= 0:
        return []
    if mode == "per-token":
        return list(range(n))
    if mode == "every-3rd":
        return list(range(0, n, 3))
    if mode == "every-5th":
        return list(range(0, n, 5))
    if mode == "key-points":
        if n <= 5:
            return list(range(n))
        return sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    raise ValueError(f"unknown decoding_mode: {mode!r}")


def normalize_mode(mode: str | None) -> DecodingMode:
    if mode is None or mode == "":
        return "per-token"
    if mode not in DECODING_MODES:
        raise ValueError(
            f"unknown decoding_mode {mode!r}; valid: {DECODING_MODES}"
        )
    return mode  # type: ignore[return-value]
