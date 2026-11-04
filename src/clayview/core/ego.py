"""Ego motion from background flow on the road plane."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from clayview.core.camera import Camera
from clayview.core.types import EgoStep, FlowPair


@dataclass(frozen=True)
class EgoConfig:
    # Road rows start this far below the horizon. The limit is in rows, not metres: a 22 m cap
    # left no road at all on a zoomed camera, whose nearest visible road was 17 m away.
    near_band_fraction: float = 0.05
    max_ground_lateral_m: float = 9.0
    min_points: int = 8
    # Windows are in seconds, so a 60 fps clip isn't smoothed half as much as a 30 fps one.
    pool_s: float = 0.13  # road points are pooled over this long each side
    static_speed_mps: float = 0.8  # below this, a cluster of points is not moving
    min_road_share: float = 0.2  # the moving cluster must be at least this much of the sample
    median_half_s: float = 0.1
    smooth_s: float = 0.077  # time constant of the low-pass
    max_speed_mps: float = 55.0
    # Points this close to the FoE say little about rotation; the floor keeps them from dominating.
    min_ray_px: float = 20.0


def yaw_step(pair: FlowPair, camera: Camera, config: EgoConfig | None = None) -> float | None:
    """Ego yaw in radians for one frame pair."""
    shift = rotation_shift(pair, camera, config)
    return None if shift is None else -shift[0] / camera.focal_px


def rotation_shift(
    pair: FlowPair, camera: Camera, config: EgoConfig | None = None
) -> tuple[float, float] | None:
    """Image shift (right, down) in pixels that ego's yaw and pitch add to every point.

    Driving forward, every static point flows straight away from the focus of expansion, however
    far away it is. Turning adds the same sideways shift r to every point, and pitching the same
    vertical shift q. So for a point p with flow d, (d - (r, q)) is parallel to p - FoE:

        dx (y - fy) - dy (x - fx) = r (y - fy) - q (x - fx)

    which is linear in r and q and needs no depth. The older estimate, the median sideways shift of
    points near the horizon, took roadside trees and walls drifting outward for a turn.

    Rolling by w turns every point about the FoE, adding w * |p - FoE|^2 to the right-hand side.
    Leaving roll out was the main source of false turning: on comma2k19 the camera rolls three
    times as much as it yaws, from frame to frame, and road points below the FoE see a roll as a
    sideways shift. Without the roll term, measured turning was twice the truth.

    Each point's equation is divided by |p - FoE|, so its residual is the flow across the ray in
    pixels. A tracking error is about the same number of pixels anywhere in the frame; unscaled,
    the points furthest from the FoE carried most of the weight and their noise.

    The FoE has to be right to a pixel or two: on comma2k19, each pixel of error in its column
    adds 2.7 degrees a minute of false turning, in proportion to speed.
    """
    cfg = config or EgoConfig()
    keep = pair.prev[:, 1] < camera.bottom
    if int(keep.sum()) < cfg.min_points:
        return None
    p = pair.prev[keep].astype(float)
    d = (pair.next[keep] - pair.prev[keep]).astype(float)
    across = p[:, 1] - camera.horizon
    down = -(p[:, 0] - camera.cx)
    # The floor only sets a point's weight; the roll term needs its true distance.
    squared = across * across + down * down
    radius = np.maximum(np.sqrt(squared), cfg.min_ray_px)
    target = (d[:, 0] * across + d[:, 1] * down) / radius
    design = np.stack([across, down, -squared], axis=1) / radius[:, None]
    weights = np.ones(len(p))
    shift = np.zeros(3)
    for _ in range(5):
        # Iteratively reweighted: moving cars and bad tracks are outliers, not noise.
        scale = np.sqrt(weights)[:, None]
        try:
            shift, *_ = np.linalg.lstsq(design * scale, target * scale[:, 0], rcond=None)
        except np.linalg.LinAlgError:
            return None
        residual = np.abs(design @ shift - target)
        spread = max(float(np.median(residual)), 1e-6)
        weights = 1.0 / np.maximum(residual / (2.0 * spread), 1.0)
    return float(shift[0]), float(shift[1])


def road_speeds(
    pair: FlowPair, camera: Camera, dt: float, config: EgoConfig | None = None
) -> np.ndarray:
    """Forward speed in m/s implied by each flow point on the road surface.

    Uses rows only: at dashcam pitch, yaw moves points sideways and barely changes their row.
    """
    cfg = config or EgoConfig()
    if len(pair.prev) == 0:
        return np.empty(0)
    focal = camera.focal_px
    top = camera.horizon + cfg.near_band_fraction * camera.image_height
    rows_prev = pair.prev[:, 1].astype(float)
    rows_next = pair.next[:, 1].astype(float)
    on_road = (rows_prev > top) & (rows_next > top)
    on_road &= (rows_prev < camera.bottom) & (rows_next < camera.bottom)
    if not on_road.any():
        return np.empty(0)

    scale = focal * camera.height_m
    y_prev = scale / (rows_prev[on_road] - camera.horizon)
    y_next = scale / (rows_next[on_road] - camera.horizon)
    x_prev = (pair.prev[on_road, 0] - camera.cx) * y_prev / focal
    near = np.abs(x_prev) < cfg.max_ground_lateral_m
    return (y_prev[near] - y_next[near]) / max(dt, 1e-3)


def speed_from_points(speeds: np.ndarray, config: EgoConfig | None = None) -> float | None:
    """Ego speed from pooled per-point speeds, ignoring anything stuck to the windshield.

    Reflections of the dash sit over the road and never move, so the sample is often two clusters:
    one at zero and the real road. A plain median lands between them. Split the sample in two; if
    the lower half really is at zero and the upper half is big enough to be the road, trust that.
    """
    cfg = config or EgoConfig()
    if len(speeds) < cfg.min_points:
        return None
    threshold = float(speeds.mean())
    for _ in range(10):
        low, high = speeds[speeds <= threshold], speeds[speeds > threshold]
        if len(low) == 0 or len(high) == 0:
            break
        threshold = (float(low.mean()) + float(high.mean())) / 2.0

    low, high = speeds[speeds <= threshold], speeds[speeds > threshold]
    if len(low) == 0 or len(high) == 0:
        return float(np.median(speeds))
    low_center = float(np.median(low))
    high_center = float(np.median(high))
    if abs(low_center) >= cfg.static_speed_mps or len(high) < cfg.min_road_share * len(speeds):
        return float(np.median(speeds))
    return high_center if high_center >= cfg.static_speed_mps else 0.0


def estimate_ego_series(
    pairs: Sequence[FlowPair],
    times: Sequence[float],
    camera: Camera,
    config: EgoConfig | None = None,
) -> tuple[list[EgoStep], float]:
    """Smoothed ego step per pair, plus the fraction of pairs that had a real measurement.

    pairs[i] links frame i to frame i + 1. Road points are pooled over a short window, since each
    frame only has a few dozen. Gaps take the neighbouring value: ego speed is the one thing in a
    dashcam clip that is guaranteed to change slowly.
    """
    cfg = config or EgoConfig()
    count = len(pairs)
    dts = [
        max(float(times[index + 1] - times[index]), 1e-3) if index + 1 < len(times) else 0.0
        for index in range(count)
    ]
    point_speeds = [road_speeds(pair, camera, dts[i], cfg) for i, pair in enumerate(pairs)]
    frame_dt = float(np.median([dt for dt in dts if dt > 0])) if count > 1 else 1.0 / 30.0
    pool = round(cfg.pool_s / frame_dt)
    yaw = np.full(count, np.nan)
    speed = np.full(count, np.nan)
    for index, pair in enumerate(pairs):
        measured_yaw = yaw_step(pair, camera, cfg)
        if measured_yaw is not None:
            yaw[index] = measured_yaw
        start = max(index - pool, 0)
        pooled = np.concatenate(point_speeds[start : index + pool + 1])
        measured = speed_from_points(pooled, cfg)
        if measured is not None and -2.0 <= measured <= cfg.max_speed_mps:
            speed[index] = max(measured, 0.0)

    coverage = float(np.isfinite(speed).mean()) if count else 0.0
    smooth_speed = _smooth_series(speed, frame_dt, cfg)
    smooth_yaw = _smooth_series(yaw, frame_dt, cfg)
    steps = [
        EgoStep(forward_m=float(smooth_speed[i]) * dts[i], yaw_rad=float(smooth_yaw[i]))
        for i in range(count)
    ]
    return steps, coverage


def _smooth_series(values: np.ndarray, frame_dt: float, cfg: EgoConfig) -> np.ndarray:
    """Fill gaps, median filter, then a forward-backward low-pass."""
    if len(values) == 0:
        return values
    known = np.isfinite(values)
    if not known.any():
        return np.zeros_like(values)
    positions = np.arange(len(values))
    filled = np.interp(positions, positions[known], values[known])

    half = round(cfg.median_half_s / frame_dt)
    padded = np.pad(filled, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    median = np.median(windows, axis=1)

    alpha = 1.0 - float(np.exp(-frame_dt / cfg.smooth_s))
    forward = _low_pass(median, alpha)
    return _low_pass(forward[::-1], alpha)[::-1]


def _low_pass(values: np.ndarray, alpha: float) -> np.ndarray:
    result = np.empty_like(values)
    level = float(values[0])
    for index, value in enumerate(values):
        level += alpha * (float(value) - level)
        result[index] = level
    return result
