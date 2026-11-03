"""Hard-cut detection from small grayscale frames.

Cuts are found by structure, not brightness: normalised cross-correlation between consecutive
thumbnails. Dashcams step their exposure abruptly, which looks like a cut to any histogram or
pixel-difference score but leaves correlation almost untouched.

On a two-minute urban clip with several exposure jumps the score never fell below 0.77, while real
splices between two clips from the same car scored 0.21 and below. The bottom of the frame is
ignored because the hood and dash match across any cut filmed from the same car.

A wiper blade crossing the glass drops the score just as far: 16 false cuts in a minute of comma2k19
footage, 10 in a rainy highway clip. What tells them apart is that a wiper goes away. Within a
second the view matches the frame before the drop again (0.59 or better on both clips), while after
a real splice nothing does (0.25 at best). So a drop is only a cut once that second has passed.
The same wait covers a flash, which also comes straight back. Live, cuts are reported a second late.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CutConfig:
    threshold: float = 0.55  # correlation below this may be a cut
    recovered: float = 0.5  # a later frame matching the one before this well means it wasn't
    lookahead_s: float = 1.0  # how long to wait for that match
    top_fraction: float = 0.72  # rows used, from the top


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


@dataclass
class _Pending:
    frame: int
    before: np.ndarray
    deadline: int
    best: float = -1.0


class CutDetector:
    """Feed frames in order. `update` returns the first frame of a new shot once it's confirmed.

    That frame is up to `lookahead_s` behind the one just fed. Call `finish` after the last frame.
    """

    def __init__(self, fps: float, config: CutConfig | None = None) -> None:
        self.config = config or CutConfig()
        self._lookahead = max(round(self.config.lookahead_s * fps), 1)
        self._index = -1
        self._previous: np.ndarray | None = None
        self._pending: _Pending | None = None

    def update(self, gray: np.ndarray) -> int | None:
        cfg = self.config
        self._index += 1
        rows = max(int(gray.shape[0] * cfg.top_fraction), 1)
        current = gray[:rows]
        previous, self._previous = self._previous, current

        confirmed: int | None = None
        pending = self._pending
        if pending is not None and pending.before.shape == current.shape:
            pending.best = max(pending.best, structural_similarity(pending.before, current))
            if pending.best >= cfg.recovered:
                # This frame matches the view before the drop, not the blade-covered one before it.
                self._pending = None
                return None
            if self._index >= pending.deadline:
                confirmed = self._confirm()

        if self._pending is not None or previous is None or previous.shape != current.shape:
            return confirmed
        if structural_similarity(previous, current) < cfg.threshold:
            self._pending = _Pending(self._index, previous, self._index + self._lookahead)
        return confirmed

    def finish(self) -> int | None:
        """A drop too close to the end to wait out still counts, unless the view came back."""
        if self._pending is None or self._pending.best >= self.config.recovered:
            return None
        return self._confirm()

    def _confirm(self) -> int:
        assert self._pending is not None
        frame = self._pending.frame
        self._pending = None
        return frame


def find_cuts(
    thumbnails: Iterable[np.ndarray], fps: float, config: CutConfig | None = None
) -> list[int]:
    """First frame of every shot after the first."""
    detector = CutDetector(fps, config)
    cuts = [cut for cut in map(detector.update, thumbnails) if cut is not None]
    last = detector.finish()
    return [*cuts, last] if last is not None else cuts
