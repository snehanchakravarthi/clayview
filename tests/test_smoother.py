import math

import numpy as np
import pytest

from clayview.core.smoother import KalmanTrack, Measurement, Step, smooth_sequence
from clayview.core.types import EgoStep

DT = 1.0 / 30.0


def noisy(x: float, y: float, rng: np.random.Generator, sigma: float = 0.5) -> Measurement:
    return Measurement(
        x + float(rng.normal(0.0, sigma)), y + float(rng.normal(0.0, sigma)), sigma, sigma
    )


def test_forward_filter_removes_most_of_the_noise() -> None:
    rng = np.random.default_rng(1)
    track = KalmanTrack(noisy(0.0, 20.0, rng))
    raw: list[float] = []
    filtered: list[float] = []
    for _ in range(150):
        measurement = noisy(0.0, 20.0, rng)
        track.predict(DT)
        track.update(measurement)
        raw.append(measurement.y)
        filtered.append(track.state().y)
    assert np.std(filtered[50:]) < np.std(raw[50:]) / 3.0
    assert np.mean(filtered[50:]) == pytest.approx(20.0, abs=0.3)


def test_filter_learns_a_constant_velocity() -> None:
    rng = np.random.default_rng(2)
    track = KalmanTrack(noisy(0.0, 10.0, rng))
    for frame in range(1, 120):
        track.predict(DT)
        track.update(noisy(0.0, 10.0 + 8.0 * frame * DT, rng))
    assert track.state().vy == pytest.approx(8.0, abs=0.8)


def test_smoother_has_no_lag_and_less_noise_than_the_forward_filter() -> None:
    rng = np.random.default_rng(3)
    truth = [15.0 + 10.0 * frame * DT for frame in range(120)]
    first = noisy(0.0, truth[0], rng)
    steps = [Step(DT, EgoStep(), noisy(0.0, y, rng)) for y in truth[1:]]

    forward = KalmanTrack(first)
    forward_y = [first.y]
    for step in steps:
        forward.predict(step.dt, step.ego)
        assert step.measurement is not None
        forward.update(step.measurement)
        forward_y.append(forward.state().y)
    smoothed_y = [state.y for state in smooth_sequence(first, steps)]

    error_forward = np.abs(np.array(forward_y[20:-20]) - truth[20:-20]).mean()
    error_smoothed = np.abs(np.array(smoothed_y[20:-20]) - truth[20:-20]).mean()
    assert error_smoothed < error_forward


def test_track_holds_through_a_dropout() -> None:
    rng = np.random.default_rng(4)
    truth = [20.0 + 5.0 * frame * DT for frame in range(90)]
    first = noisy(1.0, truth[0], rng, 0.2)
    steps: list[Step] = []
    for frame in range(1, 90):
        visible = not 40 <= frame < 60
        measurement = noisy(1.0, truth[frame], rng, 0.2) if visible else None
        steps.append(Step(DT, EgoStep(), measurement))
    states = smooth_sequence(first, steps)
    gap = [abs(states[frame].y - truth[frame]) for frame in range(40, 60)]
    assert max(gap) < 0.5


def test_ego_motion_is_not_read_as_object_motion() -> None:
    """A parked car slides past at ego speed, but its ground speed must come out near zero."""
    rng = np.random.default_rng(5)
    ego_speed = 12.0
    step_m = ego_speed * DT
    first = noisy(3.0, 40.0, rng, 0.3)
    steps = [
        Step(DT, EgoStep(forward_m=step_m), noisy(3.0, 40.0 - step_m * frame, rng, 0.3))
        for frame in range(1, 150)
    ]
    state = smooth_sequence(first, steps)[100]
    assert math.hypot(state.vx, state.vy) < 1.0


def test_overtaken_car_still_drives_forward_over_the_ground() -> None:
    """Closing on a car at 5 m/s while doing 20 means it does 15 m/s, forward."""
    rng = np.random.default_rng(6)
    step_m = 20.0 * DT
    first = noisy(0.0, 50.0, rng, 0.3)
    steps = [
        Step(DT, EgoStep(forward_m=step_m), noisy(0.0, 50.0 - 5.0 * frame * DT, rng, 0.3))
        for frame in range(1, 150)
    ]
    state = smooth_sequence(first, steps)[100]
    assert state.vy == pytest.approx(15.0, abs=1.0)


def test_ego_yaw_right_swings_the_world_left() -> None:
    track = KalmanTrack(Measurement(0.0, 20.0, 0.1, 0.1))
    track.predict(0.0, EgoStep(yaw_rad=0.1))
    assert track.state().x == pytest.approx(-20.0 * math.sin(0.1))
    assert track.state().y == pytest.approx(20.0 * math.cos(0.1))


def test_one_wild_measurement_does_not_yank_the_track() -> None:
    rng = np.random.default_rng(7)
    track = KalmanTrack(noisy(0.0, 20.0, rng))
    for _ in range(60):
        track.predict(DT)
        track.update(noisy(0.0, 20.0, rng))
    track.predict(DT)
    track.update(Measurement(0.0, 60.0, 0.5, 0.5))
    assert track.state().y < 22.0
