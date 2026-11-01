"""Plain records shared across the core. No logic, no heavy imports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Object flags written to the scene file. The viewer's debug mode shows them verbatim.
FLAG_SIZE_PRIOR = "size_prior"  # distance mostly from the class size prior, not the ground
FLAG_NO_GROUND = "no_ground"  # bottom of the box unusable (cut off, hood, or above the horizon)
FLAG_TRUNCATED = "truncated"  # box cut off by a frame edge
FLAG_CLAMPED = "clamped"  # measured beyond max range and clamped to it
FLAG_LOW_CONF = "low_conf"  # the viewer should fade this object
FLAG_HELD = "held"  # no detection this frame; position comes from the filter


@dataclass(frozen=True)
class Box:
    """Pixel-space detection box."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def bottom_center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, self.y2


@dataclass(frozen=True)
class Detection:
    """One tracked box in one frame, as the detector reports it."""

    frame: int
    track_id: int
    class_name: str
    confidence: float
    box: Box


@dataclass(frozen=True)
class Observation:
    """A detection projected onto the road plane. Metres, ego frame: x right, y forward."""

    x: float
    y: float
    sigma_x: float
    sigma_y: float
    ground_weight: float
    flags: frozenset[str]


@dataclass(frozen=True)
class EgoStep:
    """Ego motion between two consecutive processed frames. Positive yaw turns right."""

    forward_m: float = 0.0
    yaw_rad: float = 0.0


@dataclass(frozen=True)
class FlowPair:
    """Sparse background points matched between a frame and the next, in full-res pixels."""

    prev: np.ndarray  # (n, 2) float32
    next: np.ndarray  # (n, 2) float32


EMPTY_FLOW = FlowPair(np.empty((0, 2), np.float32), np.empty((0, 2), np.float32))
