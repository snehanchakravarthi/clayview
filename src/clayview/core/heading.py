"""Work out which way an object points, using every cue a track has.

No single cue is enough, so each one scores all candidate headings and the track picks the best
consistent answer:

- shape: a box's width over height depends on the angle the object is seen from. It can't tell
  front from back or left from right, but a parked car seen from a changing angle pins itself down.
- motion: the direction of travel, when the object is really moving. The only cue that knows front
  from back.
- road: vehicles tend to line up with the road, and ego's own path says where the road goes.
- view: an optional image model reads the angle a car is seen from. It's the only cue that knows
  which end of a parked car is the front.

Headings here live in a track's fixed frame: ego's direction when the track began, so a parked car
keeps one number while ego turns around it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from clayview.core.types import Box, EgoStep

BINS = 120
BIN_RAD = 2.0 * math.pi / BINS
ANGLES = -math.pi + BIN_RAD * np.arange(BINS)

# Box width over height from directly behind and from the side. Car values come from real boxes of
# cars with a clean direction of travel. One class spans real shapes: a side-on hatchback and an
# angled sedan can have the same box, and that doubt should stay visible to the other cues.
# The image model's view classes run clockwise: bin 4 sees the car's rear, bin 0 its front, and
# bins 5 and 3 see it pointing away to the left and to the right.
VIEW_BINS = 8
VIEW_REAR_BIN = 4
VIEW_BIN_RAD = 2.0 * math.pi / VIEW_BINS

SHAPES: dict[str, tuple[tuple[float, float], ...]] = {
    "car": ((1.4, 2.1), (1.4, 2.7), (1.2, 2.3)),
    "truck": ((0.8, 1.9), (0.8, 3.1)),
    "bus": ((0.8, 3.1), (0.8, 4.1)),
    "motorcycle": ((0.55, 1.4),),
    "bicycle": ((0.35, 1.05),),
}


def seen_aspect(seen_from: float | np.ndarray, rear: float, side: float) -> float | np.ndarray:
    """Box aspect at an angle to the object. Cars are rounded, so an ellipse beats a rectangle."""
    return np.hypot(side * np.sin(seen_from), rear * np.cos(seen_from))


@dataclass(frozen=True)
class HeadingConfig:
    sample_tau_s: float = 0.25  # neighbouring frames repeat each other; count one per this long
    shape_sigma: float = 0.18  # spread of the log aspect ratio around a modelled shape
    shape_pixel_sigma: float = 2.0  # box edge noise, which matters most for small boxes
    shape_floor: float = 8.0  # one bad box can't cost a heading more than this
    motion_kappa: float = 12.0  # strongest pull of a clean direction of travel
    road_kappa: float = 1.5
    view_weight: float = 0.5  # views are sampled, so this counts per view, not per second
    view_floor: float = 0.03  # the model is confidently wrong on hidden cars; limit what that costs
    road_reach_m: float = 18.0  # the road cue fades for objects this far from ego's path
    tie_break: float = 0.001  # with no evidence at all, face the way ego does
    turn_cost_per_rad: float = 0.5
    max_turn_rate: float = 2.1  # rad/s


@dataclass(frozen=True)
class HeadingCues:
    """What one frame of a track knows. Angles are in the track's fixed frame except `bearing`."""

    yaw: float  # ego's rotation since the track began
    bearing: float  # direction from ego to the object, in ego's frame at this moment
    box: Box | None  # None without a whole box: a cut-off box has no meaningful shape
    motion: float | None  # direction of travel, None when motion says nothing
    motion_certainty: float  # speed over its uncertainty
    road: float | None  # direction of the road beside the object
    road_offset_m: float  # how far the object is from ego's path
    view: Sequence[float] | None = None  # image model output, see VIEW_BINS


def mount_at(steps: Sequence[EgoStep], frame: int) -> float:
    """The camera's yaw on its mount at a frame, relative to the car."""
    if not steps:
        return 0.0
    return steps[min(frame, len(steps) - 1)].mount_rad


class EgoPath:
    """Ego's path through one shot. Frames are absolute, the frame is ego's at `start_frame`."""

    def __init__(self, steps: Sequence[EgoStep], start_frame: int, end_frame: int) -> None:
        self.start_frame = start_frame
        count = max(end_frame - start_frame, 1)
        yaw = np.zeros(count)
        x = np.zeros(count)
        y = np.zeros(count)
        for index in range(1, count):
            frame = start_frame + index - 1
            step = steps[frame] if frame < len(steps) else EgoStep()
            yaw[index] = yaw[index - 1] + step.yaw_rad
            x[index] = x[index - 1] + step.forward_m * math.sin(yaw[index - 1])
            y[index] = y[index - 1] + step.forward_m * math.cos(yaw[index - 1])
        self.yaw = yaw
        self.x = x
        self.y = y
        # The camera's yaw on its mount, relative to the car. The camera looks along yaw + mount.
        self.mount = np.array([mount_at(steps, start_frame + index) for index in range(count)])

    def yaw_at(self, frame: int) -> float:
        return float(self.yaw[self._index(frame)])

    def road_at(self, frame: int, x: float, y: float) -> tuple[float, float]:
        """Road direction beside an ego-frame point, and how far that point is from ego's path."""
        index = self._index(frame)
        yaw = self.yaw[index]
        world_x = self.x[index] + x * math.cos(yaw) + y * math.sin(yaw)
        world_y = self.y[index] - x * math.sin(yaw) + y * math.cos(yaw)
        distances = np.hypot(self.x - world_x, self.y - world_y)
        nearest = int(np.argmin(distances))
        return float(self.yaw[nearest]), float(distances[nearest])

    def _index(self, frame: int) -> int:
        return min(max(frame - self.start_frame, 0), len(self.yaw) - 1)


def shape_scores(
    box: Box, bearing: float, yaw: float, class_name: str, config: HeadingConfig
) -> np.ndarray:
    """Log likelihood of each candidate heading, given how wide this box is for its height."""
    shapes = SHAPES.get(class_name)
    scores = np.zeros(BINS)
    if shapes is None or box.width <= 0.0 or box.height <= 0.0:
        return scores

    # The angle between the object's nose and the ray from ego: 0 is seen from directly behind.
    seen_from = ANGLES - yaw - bearing
    observed = math.log(box.width / box.height)
    noise = config.shape_pixel_sigma * math.hypot(1.0 / box.width, 1.0 / box.height)
    sigma = math.hypot(config.shape_sigma, noise)

    best = np.full(BINS, -math.inf)
    for rear, side in shapes:
        predicted = np.log(seen_aspect(seen_from, rear, side))
        best = np.maximum(best, -0.5 * ((observed - predicted) / sigma) ** 2)
    return np.maximum(best, -config.shape_floor)


def view_scores(view: Sequence[float], bearing: float, yaw: float, floor: float) -> np.ndarray:
    """Log likelihood of each heading, from the image model's eight view probabilities."""
    seen_from = ANGLES - yaw - bearing
    position = (VIEW_REAR_BIN - seen_from / VIEW_BIN_RAD) % VIEW_BINS
    lower = np.floor(position).astype(int) % VIEW_BINS
    upper = (lower + 1) % VIEW_BINS
    share = position - np.floor(position)
    logs = np.log(np.maximum(np.asarray(view, dtype=float), floor))
    return (1.0 - share) * logs[lower] + share * logs[upper]


def frame_scores(
    cue: HeadingCues, class_name: str, weight: float, can_turn: bool, config: HeadingConfig
) -> np.ndarray:
    scores = np.zeros(BINS)
    if cue.box is not None:
        scores += weight * shape_scores(cue.box, cue.bearing, cue.yaw, class_name, config)
    if cue.view is not None:
        scores += config.view_weight * view_scores(
            cue.view, cue.bearing, cue.yaw, config.view_floor
        )
    if cue.motion is not None:
        kappa = min(cue.motion_certainty * cue.motion_certainty, config.motion_kappa)
        scores += weight * kappa * np.cos(ANGLES - cue.motion)
    if cue.road is not None:
        kappa = config.road_kappa * math.exp(-((cue.road_offset_m / config.road_reach_m) ** 2))
        # Traffic runs both ways along a road. Parked cars also sit at right angles to it, and box
        # shape can't separate side-on from diagonal, so a resting track gets four peaks.
        peaks = 2.0 if can_turn else 4.0
        scores += weight * kappa * np.cos(peaks * (ANGLES - cue.road))
        scores += config.tie_break * np.cos(ANGLES - cue.road)
    return scores


def solve_heading(
    cues: Sequence[HeadingCues],
    dts: Sequence[float],
    class_name: str,
    can_turn: bool,
    config: HeadingConfig | None = None,
) -> list[float]:
    """Best heading per frame in the track's fixed frame. `dts[i]` runs from frame i to i + 1."""
    cfg = config or HeadingConfig()
    if not cues:
        return []
    durations = [*dts, dts[-1] if dts else cfg.sample_tau_s]
    scores = np.stack(
        [
            frame_scores(cue, class_name, min(dt / cfg.sample_tau_s, 1.0), can_turn, cfg)
            for cue, dt in zip(cues, durations, strict=True)
        ]
    )
    if not can_turn:
        return [_peak(scores.sum(axis=0))] * len(cues)
    return _best_path(scores, dts, cfg)


def _peak(scores: np.ndarray) -> float:
    """Top of the score curve, read between bins so a heading isn't stuck to the grid."""
    best = int(np.argmax(scores))
    left, middle, right = scores[best - 1], scores[best], scores[(best + 1) % BINS]
    curve = left - 2.0 * middle + right
    shift = 0.0 if curve >= 0.0 else float(np.clip(0.5 * (left - right) / curve, -0.5, 0.5))
    return float(ANGLES[best]) + shift * BIN_RAD


def _best_path(scores: np.ndarray, dts: Sequence[float], cfg: HeadingConfig) -> list[float]:
    """Viterbi over headings: the best sequence, so a track can't flip on one odd frame."""
    frames = len(scores)
    total = scores[0].copy()
    came_from = np.zeros((frames, BINS), dtype=np.int16)
    positions = np.arange(BINS)
    for index in range(1, frames):
        reach = max(math.ceil(cfg.max_turn_rate * dts[index - 1] / BIN_RAD), 1)
        shifts = np.arange(-reach, reach + 1)
        candidates = np.stack(
            [
                np.roll(total, shift) - cfg.turn_cost_per_rad * abs(shift) * BIN_RAD
                for shift in shifts
            ]
        )
        choice = np.argmax(candidates, axis=0)
        total = candidates[choice, positions] + scores[index]
        came_from[index] = (positions - shifts[choice]) % BINS

    path = np.zeros(frames, dtype=np.int16)
    path[-1] = int(np.argmax(total))
    for index in range(frames - 1, 0, -1):
        path[index - 1] = came_from[index, path[index]]
    return [float(ANGLES[bin_index]) for bin_index in path]
