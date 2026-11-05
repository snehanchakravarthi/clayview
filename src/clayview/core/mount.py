"""Split the camera's measured yaw into the car's yaw and the camera turning on its mount.

Flow measures how the camera turns against the world. On a fixed dashcam that is the car. On a
phone in a hand, a helmet or a head, it is the car plus the camera turning on its mount, and the
mount part is often the larger: a glance sideways turns a head faster than any car can turn.

The car is the part with limits. A wheeled vehicle turns about a point on its rear axle's line, so
its yaw rate is speed times path curvature, and curvature is at most 1 / R_min. Grip limits
sideways acceleration, which is speed times yaw rate. So at speed v the car yaws no faster than

    min(v / R_min, a_max / v)

which is 0 at a standstill, peaks around 80 deg/s at 20 km/h and falls to 15 deg/s at 110 km/h.
Speed from flow can be out by a factor of three in a sharp turn, which is when this matters: on
matanzas_portrait, whose phone is fixed to the dash, it read 63 km/h through a junction turned at
walking pace. So the limit is the highest the car could have at any speed within that factor of
the measured one. Real turns on every fixed camera in the eval stay under it; a helmet looking
round, a head turning to the side window or a hand swinging a phone don't.

A camera swing whose peak breaks that limit is a swing of the mount, and so are its shoulders, from
the rate valley before the peak to the valley after it. Across the swing, the car keeps the yaw rate
it had at those valleys. What the car can't have done is the mount's.

Mounts also come back. A head looks ahead between glances, and a phone is held roughly forward; a
mount angle that has stopped changing is slowly handed back to the car as the car travels, so the
car's long-run heading still follows the camera's. That return is by distance, not time: a car
standing still can't turn, whatever the camera does.

Without a swing past the limit nothing is changed: the car's yaw is the camera's, exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MountConfig:
    min_turn_radius_m: float = 4.0  # tightest turn of a car or motorbike, at the rear axle
    max_lateral_accel: float = 8.0  # m/s^2, about the grip of a dry road
    speed_window_s: float = 1.0  # the limit uses a running median of speed over this long
    speed_doubt: float = 3.0  # and allows for that speed being this many times too high or low
    rest_s: float = 10.0  # a held mount angle goes back to the car over this much driving
    min_rest_m: float = 20.0  # and over at least this many metres


def vehicle_yaw_limit(speed_mps: np.ndarray, config: MountConfig | None = None) -> np.ndarray:
    """Fastest yaw rate in rad/s a car can turn at each measured speed, allowing for its error."""
    cfg = config or MountConfig()
    speed = np.maximum(np.asarray(speed_mps, dtype=float), 0.0)
    slowest, fastest = speed / cfg.speed_doubt, speed * cfg.speed_doubt
    # The limit rises with speed until radius and grip meet, then falls. Take the best in range.
    peak = np.sqrt(cfg.max_lateral_accel * cfg.min_turn_radius_m)
    most_agile = np.clip(peak, slowest, fastest)
    return np.minimum(
        most_agile / cfg.min_turn_radius_m, cfg.max_lateral_accel / np.maximum(most_agile, 0.1)
    )


def separate_mount(
    camera_yaw: np.ndarray,
    speed_mps: np.ndarray,
    dts: np.ndarray,
    config: MountConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Car yaw per step, and the camera's yaw on its mount per frame.

    `camera_yaw[i]` is the camera's turn in radians from frame i to i + 1, `speed_mps[i]` the car's
    speed over that step and `dts[i]` its length. Returns the car's turn per step, and the mount
    angle at each of the len + 1 frames, 0 at the first. Positive turns right, as everywhere. The
    two always add back up: car turn plus mount change is the camera's turn.
    """
    cfg = config or MountConfig()
    camera = np.asarray(camera_yaw, dtype=float)
    count = len(camera)
    if count == 0:
        return camera.copy(), np.zeros(1)
    dts = np.maximum(np.asarray(dts, dtype=float), 1e-3)
    speed = _running_median(np.asarray(speed_mps, dtype=float), dts, cfg.speed_window_s)
    limit = vehicle_yaw_limit(speed, cfg) * dts

    vehicle, swings = _remove_swings(camera, limit)
    if not swings:
        return camera.copy(), np.zeros(count + 1)

    mount = np.concatenate(([0.0], np.cumsum(camera - vehicle)))
    resting = np.ones(count + 1, dtype=bool)
    for start, end in swings:
        resting[start : end + 2] = False
    travelled = np.concatenate(([0.0], np.cumsum(speed * dts)))
    reach = max(cfg.min_rest_m, float(np.median(speed)) * cfg.rest_s)
    rest = _rest_angle(mount, travelled, resting, reach)
    vehicle = vehicle + np.diff(rest)
    # The hand-back is slow, but a fast stretch can still ask more of the car than it has.
    vehicle = np.clip(vehicle, -limit, limit)
    return vehicle, np.concatenate(([0.0], np.cumsum(camera - vehicle)))


def _remove_swings(
    camera: np.ndarray, limit: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Car yaw with every swing past the limit replaced, and the swings as (first, last) steps."""
    vehicle = camera.copy()
    size = np.abs(camera)
    over = size > limit
    swings: list[tuple[int, int]] = []
    count = len(camera)
    index = 0
    while index < count:
        if not over[index]:
            index += 1
            continue
        sign = np.sign(camera[index])
        start = end = index
        while end + 1 < count and over[end + 1] and np.sign(camera[end + 1]) == sign:
            end += 1
        # Walk out to the valleys either side: the same swing, still falling away from its peak.
        while start > 0 and np.sign(camera[start - 1]) == sign and size[start - 1] < size[start]:
            start -= 1
        while end + 1 < count and np.sign(camera[end + 1]) == sign and size[end + 1] < size[end]:
            end += 1
        # The car's own rate either side: the valley, or past it if the swing began at its peak.
        before = _quieter(vehicle[start], vehicle[start - 1] if start > 0 else 0.0)
        after = _quieter(camera[end], camera[end + 1] if end + 1 < count else 0.0)
        baseline = np.linspace(before, after, end - start + 1)
        vehicle[start : end + 1] = np.clip(
            baseline, -limit[start : end + 1], limit[start : end + 1]
        )
        swings.append((start, end))
        index = end + 1
    return vehicle, swings


def _quieter(a: float, b: float) -> float:
    return a if abs(a) <= abs(b) else b


def _rest_angle(
    mount: np.ndarray, travelled: np.ndarray, resting: np.ndarray, reach_m: float
) -> np.ndarray:
    """Where the mount has settled: it follows the mount angle between swings, over distance.

    Causal on purpose. A mount comes back after it has moved, never before, so the car can't start
    turning ahead of a knock to the phone.
    """
    rest = np.zeros(len(mount))
    for index in range(1, len(mount)):
        rest[index] = rest[index - 1]
        if resting[index]:
            share = 1.0 - np.exp(-(travelled[index] - travelled[index - 1]) / reach_m)
            rest[index] += share * (mount[index] - rest[index - 1])
    return rest


def _running_median(values: np.ndarray, dts: np.ndarray, window_s: float) -> np.ndarray:
    half = round(window_s / 2.0 / float(np.median(dts)))
    if half < 1 or len(values) < 2:
        return values.copy()
    padded = np.pad(values, half, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1), axis=1)
