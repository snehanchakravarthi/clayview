"""Constant-velocity Kalman filter on the road plane, in the ego frame.

The state is (x, y, vx, vy): position relative to ego, velocity over the ground. Ego motion is a
known input to the prediction step, so a parked car reads as stationary even though it slides
past ego, and a car being overtaken still reads as driving forward.

`KalmanTrack` runs forward only and is what a live pipeline would use. `smooth_sequence` adds an
RTS backward pass on top for recorded clips, which removes the filter's lag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from clayview.core.types import EgoStep

_H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


@dataclass(frozen=True)
class FilterConfig:
    accel_sigma: float = 3.0  # m/s^2 of unmodelled acceleration
    initial_speed_sigma: float = 12.0  # m/s
    ego_error_fraction: float = 0.08  # relative error of each ego step
    ego_yaw_sigma_rad: float = 0.001  # per-step error of the ego yaw
    outlier_gate: float = 9.0  # squared Mahalanobis distance before a measurement is distrusted


@dataclass(frozen=True)
class Measurement:
    x: float
    y: float
    sigma_x: float
    sigma_y: float


@dataclass(frozen=True)
class Step:
    """One filter step: advance by dt and ego motion, then apply the measurement if there is one."""

    dt: float
    ego: EgoStep
    measurement: Measurement | None


@dataclass(frozen=True)
class State:
    x: float
    y: float
    vx: float
    vy: float
    sigma_position: float
    sigma_speed: float


def _transition(dt: float, ego: EgoStep) -> tuple[np.ndarray, np.ndarray]:
    """State transition F and offset b, so next = F @ state + b."""
    cos, sin = math.cos(ego.yaw_rad), math.sin(ego.yaw_rad)
    rotation = np.array([[cos, -sin], [sin, cos]])
    f = np.zeros((4, 4))
    f[:2, :2] = rotation
    f[:2, 2:] = rotation * dt
    f[2:, 2:] = rotation
    b = np.zeros(4)
    b[:2] = rotation @ np.array([0.0, -ego.forward_m])
    return f, b


class KalmanTrack:
    """Forward filter for a single object."""

    def __init__(self, first: Measurement, config: FilterConfig | None = None) -> None:
        self.config = config or FilterConfig()
        self.mean = np.array([first.x, first.y, 0.0, 0.0])
        speed_variance = self.config.initial_speed_sigma**2
        self.cov = np.diag([first.sigma_x**2, first.sigma_y**2, speed_variance, speed_variance])

    def predict(self, dt: float, ego: EgoStep | None = None) -> None:
        step = ego or EgoStep()
        f, b = _transition(dt, step)
        self.mean = f @ self.mean + b
        self.cov = f @ self.cov @ f.T + self._process_noise(dt, step)

    def update(self, measurement: Measurement) -> None:
        z = np.array([measurement.x, measurement.y])
        r = np.diag([measurement.sigma_x**2, measurement.sigma_y**2])
        innovation = z - _H @ self.mean
        s = _H @ self.cov @ _H.T + r
        distance = float(innovation @ np.linalg.solve(s, innovation))
        if distance > self.config.outlier_gate:
            # One bad box must not yank the track, but a real jump must win eventually.
            r = r * (distance / self.config.outlier_gate)
            s = _H @ self.cov @ _H.T + r
        gain = self.cov @ _H.T @ np.linalg.inv(s)
        self.mean = self.mean + gain @ innovation
        self.cov = (np.eye(4) - gain @ _H) @ self.cov

    def _process_noise(self, dt: float, ego: EgoStep) -> np.ndarray:
        cfg = self.config
        g = np.array(
            [[dt * dt / 2.0, 0.0], [0.0, dt * dt / 2.0], [dt, 0.0], [0.0, dt]],
        )
        q = cfg.accel_sigma**2 * (g @ g.T)
        forward_error = cfg.ego_error_fraction * abs(ego.forward_m)
        lateral_error = cfg.ego_yaw_sigma_rad * abs(float(self.mean[1]))
        q[0, 0] += lateral_error * lateral_error
        q[1, 1] += forward_error * forward_error
        return q

    def state(self) -> State:
        return _state_of(self.mean, self.cov)


def _state_of(mean: np.ndarray, cov: np.ndarray) -> State:
    return State(
        x=float(mean[0]),
        y=float(mean[1]),
        vx=float(mean[2]),
        vy=float(mean[3]),
        sigma_position=math.sqrt(max(float(cov[0, 0] + cov[1, 1]), 0.0)),
        sigma_speed=math.sqrt(max(float(cov[2, 2] + cov[3, 3]), 0.0)),
    )


def smooth_sequence(
    first: Measurement, steps: list[Step], config: FilterConfig | None = None
) -> list[State]:
    """Forward filter then RTS backward pass. Returns len(steps) + 1 states, first included."""
    track = KalmanTrack(first, config)
    means = [track.mean.copy()]
    covs = [track.cov.copy()]
    predicted_means: list[np.ndarray] = []
    predicted_covs: list[np.ndarray] = []
    transitions: list[np.ndarray] = []

    for step in steps:
        f, _ = _transition(step.dt, step.ego)
        track.predict(step.dt, step.ego)
        predicted_means.append(track.mean.copy())
        predicted_covs.append(track.cov.copy())
        transitions.append(f)
        if step.measurement is not None:
            track.update(step.measurement)
        means.append(track.mean.copy())
        covs.append(track.cov.copy())

    smoothed_means = list(means)
    smoothed_covs = list(covs)
    for k in range(len(steps) - 1, -1, -1):
        gain = covs[k] @ transitions[k].T @ np.linalg.pinv(predicted_covs[k])
        smoothed_means[k] = means[k] + gain @ (smoothed_means[k + 1] - predicted_means[k])
        delta = smoothed_covs[k + 1] - predicted_covs[k]
        smoothed_covs[k] = covs[k] + gain @ delta @ gain.T

    return [_state_of(m, c) for m, c in zip(smoothed_means, smoothed_covs, strict=True)]
