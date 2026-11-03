"""Automatic camera calibration for clips from unknown cameras.

Three signals, all needing only what the pipeline already has:

- Focus of expansion. Driving straight, static points flow away from one image point, the
  vanishing point of the direction of travel. Its row is the horizon, its column is where
  dead-ahead sits in the image, which corrects a dashcam mounted off-axis.
- Vehicle widths. For cars on a flat road, width_px = (W/h) * (bottom_y - horizon), independent of
  focal length. A line fit of width against bottom row hits zero at the horizon, and its slope
  gives the camera height.
- Static rows. Pixels that stay put while the world moves are the hood or dash.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from clayview.core.projection import CLASS_PRIORS
from clayview.core.types import Detection, FlowPair

MIN_FLOW_PX = 1.2


@dataclass(frozen=True)
class VehicleFit:
    horizon_y: float
    height_m: float
    samples: int
    inlier_ratio: float


@dataclass(frozen=True)
class FoeFit:
    x: float
    y: float
    points: int
    residual_px: float


def vehicle_samples(
    detections: Sequence[Detection],
    image_width: int,
    image_height: int,
    min_confidence: float = 0.5,
    frame_step: int = 5,
) -> np.ndarray:
    """(n, 2) of (bottom_y, width_px) from clean, truly rear-on cars.

    The aspect limit is tight on purpose. On a 2-minute urban clip the height implied by a box fell
    steadily with its aspect, from about 1.15 m at rear-on 1.05 to 0.57 m side-on, because any
    oblique view shows more than the car's width. Rear-on boxes agreed with lane-width geometry.
    """
    samples: list[tuple[float, float]] = []
    margin = 4.0
    for det in detections:
        if det.class_name != "car" or det.confidence < min_confidence:
            continue
        if det.frame % frame_step != 0:
            continue
        box = det.box
        if box.x1 <= margin or box.x2 >= image_width - margin or box.y2 >= image_height - margin:
            continue
        if box.width < 0.015 * image_width:
            continue
        aspect = box.width / max(box.height, 1.0)
        center_offset = abs((box.x1 + box.x2) / 2.0 - image_width / 2.0) / image_width
        if not 0.9 <= aspect <= 1.22 or center_offset > 0.3:
            continue
        samples.append((box.y2, box.width))
    return np.array(samples, dtype=float).reshape(-1, 2)


def fit_vehicle_horizon(
    samples: np.ndarray,
    image_height: int,
    min_samples: int = 25,
    iterations: int = 300,
    seed: int = 7,
) -> VehicleFit | None:
    """RANSAC line through (bottom_y, width). None if the data can't support a calibration."""
    if len(samples) < min_samples:
        return None
    rows, widths = samples[:, 0], samples[:, 1]
    if rows.max() - rows.min() < 0.06 * image_height:
        return None

    rng = np.random.default_rng(seed)
    tolerance = np.maximum(0.12 * widths, 2.0)
    best: np.ndarray | None = None
    for _ in range(iterations):
        a, b = rng.choice(len(samples), size=2, replace=False)
        run = rows[a] - rows[b]
        if abs(run) < 0.02 * image_height:
            continue
        slope = (widths[a] - widths[b]) / run
        if slope <= 0.0:
            continue
        inliers = np.abs(slope * (rows - rows[a]) + widths[a] - widths) < tolerance
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < min_samples * 0.6:
        return None

    slope, intercept = np.polyfit(rows[best], widths[best], 1)
    if slope <= 0.0:
        return None
    horizon = float(-intercept / slope)
    height = CLASS_PRIORS["car"].width_m / float(slope)
    if not 0.25 * image_height <= horizon <= 0.8 * image_height or not 0.6 <= height <= 3.2:
        return None
    return VehicleFit(horizon, height, len(samples), float(best.mean()))


def vehicle_height_for_horizon(samples: np.ndarray, horizon_y: float) -> float | None:
    """Camera height implied by car widths once the horizon is known."""
    if len(samples) < 12:
        return None
    below = samples[:, 0] - horizon_y
    usable = below > 8.0
    if int(usable.sum()) < 12:
        return None
    slope = float(np.median(samples[usable, 1] / below[usable]))
    height = CLASS_PRIORS["car"].width_m / slope
    return height if 0.6 <= height <= 3.2 else None


def fit_focus_of_expansion(
    pairs: Sequence[FlowPair],
    image_width: int,
    image_height: int,
    yaw_px: Sequence[float] | None = None,
    max_points: int = 40_000,
    iterations: int = 6,
) -> FoeFit | None:
    """Least-squares intersection of flow lines, with outlier rejection.

    `yaw_px` is the horizontal image shift per pair caused by ego yaw. Removing it keeps turns from
    dragging the estimate sideways; frames with strong yaw are skipped outright.
    """
    points, normals = _flow_lines(pairs, image_height, yaw_px)
    if len(points) < 200:
        return None
    if len(points) > max_points:
        keep = np.random.default_rng(3).choice(len(points), max_points, replace=False)
        points, normals = points[keep], normals[keep]

    active = np.ones(len(points), dtype=bool)
    foe = np.array([image_width / 2.0, image_height / 2.0])
    residual = 0.0
    for _ in range(iterations):
        n = normals[active]
        p = points[active]
        a = n.T @ n
        b = n.T @ np.einsum("ij,ij->i", n, p)
        try:
            foe = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            return None
        distance = np.abs(np.einsum("ij,ij->i", normals, points - foe))
        residual = float(np.median(distance[active]))
        active = distance < max(3.0 * residual, 6.0)
        if active.sum() < 100:
            return None

    if not (0.2 * image_width <= foe[0] <= 0.8 * image_width):
        return None
    if not (0.2 * image_height <= foe[1] <= 0.85 * image_height):
        return None
    return FoeFit(float(foe[0]), float(foe[1]), int(active.sum()), residual)


def _flow_lines(
    pairs: Sequence[FlowPair],
    image_height: int,
    yaw_px: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Midpoints and unit normals of usable flow vectors from the frames that drive straight."""
    all_points: list[np.ndarray] = []
    all_normals: list[np.ndarray] = []
    for index, pair in enumerate(pairs):
        if len(pair.prev) == 0:
            continue
        shift = 0.0 if yaw_px is None else float(yaw_px[index])
        if abs(shift) > 0.004 * image_height:
            continue
        flow = (pair.next - pair.prev).astype(float)
        flow[:, 0] -= shift
        length = np.hypot(flow[:, 0], flow[:, 1])
        usable = length > MIN_FLOW_PX
        if usable.sum() == 0:
            continue
        unit = flow[usable] / length[usable, None]
        all_normals.append(np.stack([unit[:, 1], -unit[:, 0]], axis=1))
        all_points.append((pair.prev[usable].astype(float) + pair.next[usable]) / 2.0)
    if not all_points:
        return np.empty((0, 2)), np.empty((0, 2))
    return np.concatenate(all_points), np.concatenate(all_normals)


def estimate_ground_bottom(
    pairs: Sequence[FlowPair],
    horizon_y: float,
    image_height: int,
    bins: int = 32,
    min_frames: int = 10,
) -> float | None:
    """Row where the visible road ends, from where flow stops behaving like a ground plane.

    On the road, vertical flow grows with (row - horizon)^2. Divided by that, every road row reads
    about the same; the hood, dash or a reflection on the glass reads near zero, and a shiny hood's
    reflections read anything at all. Each row band is compared to a reference band of near road
    over the frames where the road clearly moves.

    The older test, the first moving row up from the bottom, failed whenever something moved down
    there. comma2k19's hood reflection and a portrait phone's purple hood both counted as road, and
    the reflection's static points outnumbered the moving road and pulled ego speed down by 40%.
    """
    reference = (horizon_y + 0.06 * image_height, horizon_y + 0.2 * image_height)
    edges = np.linspace(horizon_y, image_height, bins + 1)
    per_bin: list[list[float]] = [[] for _ in range(bins)]
    frames = 0
    for pair in pairs:
        if len(pair.prev) < 30:
            continue
        rows = pair.prev[:, 1].astype(float)
        normalised = (pair.next[:, 1] - rows) / np.maximum(rows - horizon_y, 1.0) ** 2
        in_reference = (rows > reference[0]) & (rows < reference[1])
        if in_reference.sum() < 8:
            continue
        level = float(np.median(normalised[in_reference]))
        if level * (reference[1] - horizon_y) ** 2 < 2.0:
            continue  # the road isn't moving enough to tell it from the hood
        frames += 1
        index = np.digitize(rows, edges) - 1
        for band in range(bins):
            in_band = index == band
            if in_band.sum() >= 3:
                per_bin[band].append(float(np.median(normalised[in_band])) / level)
    if frames < min_frames:
        return None

    relative = [float(np.median(values)) if len(values) >= 5 else None for values in per_bin]
    start = int(np.searchsorted(edges, reference[1]))
    for band in range(start, bins - 1):
        here, below = relative[band], relative[band + 1]
        # Two bands in a row, so one noisy band doesn't end the road.
        if here is not None and here < 0.4 and (below is None or below < 0.4):
            return float(edges[band])
    return None


def estimate_visible_bottom(
    pairs: Sequence[FlowPair],
    image_height: int,
    bins: int = 24,
    min_row_fraction: float = 0.55,
) -> float | None:
    """Row where the hood or dash starts: below it, nothing moves while the road above does.

    A fallback for when the horizon is unknown or the road barely moves; see estimate_ground_bottom.
    """
    rows: list[np.ndarray] = []
    magnitudes: list[np.ndarray] = []
    for pair in pairs:
        if len(pair.prev) < 20:
            continue
        magnitude = np.hypot(*(pair.next - pair.prev).T)
        if float(np.median(magnitude)) < MIN_FLOW_PX:
            continue
        rows.append(pair.prev[:, 1])
        magnitudes.append(magnitude)
    if not rows:
        return None
    row = np.concatenate(rows)
    magnitude = np.concatenate(magnitudes)
    typical = float(np.median(magnitude))

    edges = np.linspace(0.0, image_height, bins + 1)
    first_bin = int(min_row_fraction * bins)
    for index in range(bins - 1, first_bin - 1, -1):
        in_bin = (row >= edges[index]) & (row < edges[index + 1])
        if in_bin.sum() < 30:
            continue
        if float(np.median(magnitude[in_bin])) >= 0.25 * typical:
            return None if index == bins - 1 else float(edges[index + 1])
    return None
