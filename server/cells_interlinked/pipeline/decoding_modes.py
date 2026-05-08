"""Decoding-mode definitions: which captured positions feed the AV.

The AV is a one-vector-in / one-paragraph-out instrument. We can't change
that without retraining. What we CAN choose is:

  1. Which positions in the output to decode (mode):
       per-token    every output position
       every-3rd    every 3rd position  (stride 3)
       every-5th    every 5th position  (stride 5)
       key-points   first, ~25/50/75%, last  (5 hand-picked positions)

  2. Whether each "pick" is a single position or a *window* of adjacent
     positions whose activations get mean-pooled into one decode (pooled):
       pooled = false:  feed the activation at one specific position
       pooled = true:   feed the average of W activations across a small
                        window centered/aligned on the pick

Sampled mode reads ARE per-token: "this token's activation says X."
Pooled mode reads are PHRASE-LEVEL: "this 3-token region's average
activation says Y."  They produce roughly the same row count but answer
different questions. Pooling muddles per-position semantics in exchange
for surfacing slow-changing themes the per-token reads don't isolate.

Per-token + pooled is a no-op (windows of 1 = same as sampled).
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


def select_windows(n: int, mode: str, pooled: bool) -> list[list[int]]:
    """Return the list of position-windows to decode. Each window is a
    list of position indices; if len > 1 the activations get mean-pooled
    before the AV sees them.
    """
    if n <= 0:
        return []
    if mode == "per-token":
        # Pooled has no meaning at stride 1.
        return [[i] for i in range(n)]
    if mode == "every-3rd":
        if pooled:
            return [list(range(i, min(i + 3, n))) for i in range(0, n, 3)]
        return [[i] for i in range(0, n, 3)]
    if mode == "every-5th":
        if pooled:
            return [list(range(i, min(i + 5, n))) for i in range(0, n, 5)]
        return [[i] for i in range(0, n, 5)]
    if mode == "key-points":
        if n <= 5:
            kps = list(range(n))
        else:
            kps = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
        if pooled:
            # ±2 window centered on each key point, clamped to [0, n-1].
            windows: list[list[int]] = []
            for kp in kps:
                start = max(0, kp - 2)
                end = min(n - 1, kp + 2)
                windows.append(list(range(start, end + 1)))
            return windows
        return [[kp] for kp in kps]
    raise ValueError(f"unknown decoding_mode: {mode!r}")


def normalize_mode(mode: str | None) -> DecodingMode:
    if mode is None or mode == "":
        return "per-token"
    if mode not in DECODING_MODES:
        raise ValueError(
            f"unknown decoding_mode {mode!r}; valid: {DECODING_MODES}"
        )
    return mode  # type: ignore[return-value]
