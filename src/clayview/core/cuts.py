"""Hard-cut detection from small grayscale frames.

Cuts are found by structure, not brightness: normalised cross-correlation between consecutive
thumbnails. Dashcams step their exposure abruptly, which looks like a cut to any histogram or
pixel-difference score but leaves correlation almost untouched.

On a two-minute urban clip with several exposure jumps the score never fell below 0.77, while real
splices between two clips from the same car scored 0.21 and below. The bottom of the frame is
ignored because the hood and dash match across any cut filmed from the same car.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CutConfig:
    threshold: float = 0.55  # correlation below this is a cut
    top_fraction: float = 0.72  # rows used, from the top
    min_gap_frames: int = 8  # a cut straight after a cut is a flash, not an edit


def structural_similarity(previous: np.ndarray, current: np.ndarray) -> float:
    """Normalised cross-correlation of two equal-shape grayscale frames, from -1 to 1."""
    a = previous.astype(np.float32).ravel()
    b = current.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denominator < 1e-6:
        # Flat frames (fade to black) carry no structure either way.
        return 1.0
    return float(np.dot(a, b)) / denominator


class CutDetector:
    """Feed frames in order. `update` returns True on the first frame of a new shot."""

    def __init__(self, config: CutConfig | None = None) -> None:
        self.config = config or CutConfig()
        self._previous: np.ndarray | None = None
        self._since_cut = self.config.min_gap_frames

    def update(self, gray: np.ndarray) -> bool:
        cfg = self.config
        rows = max(int(gray.shape[0] * cfg.top_fraction), 1)
        current = gray[:rows]
        previous, self._previous = self._previous, current
        self._since_cut += 1
        if previous is None or previous.shape != current.shape:
            return False
        if self._since_cut <= cfg.min_gap_frames:
            return False
        if structural_similarity(previous, current) < cfg.threshold:
            self._since_cut = 0
            return True
        return False
