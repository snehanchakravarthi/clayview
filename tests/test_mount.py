import math

import numpy as np
import pytest

from clayview.core.mount import MountConfig, separate_mount, vehicle_yaw_limit

DT = 1.0 / 30.0


def series(seconds: float) -> np.ndarray:
    return np.arange(round(seconds / DT)) * DT


def rates_to_steps(rates_deg_s: np.ndarray) -> np.ndarray:
    return np.radians(rates_deg_s) * DT


def glance(t: np.ndarray, start: float, angle_deg: float, swing_s: float = 0.3) -> np.ndarray:
    """Camera yaw rate in deg/s for a look aside and back: out, hold one second, back."""
    rate = np.zeros_like(t)
    speed = angle_deg / swing_s
    rate[(t >= start) & (t < start + swing_s)] = speed
    back = start + swing_s + 1.0
    rate[(t >= back) & (t < back + swing_s)] = -speed
    return rate


def test_the_limit_is_zero_at_rest_and_falls_at_speed() -> None:
    exact = MountConfig(speed_doubt=1.0)
    limit = np.degrees(vehicle_yaw_limit(np.array([0.0, 2.0, 5.66, 30.0]), exact))
    assert limit[0] == 0.0
    assert limit[1] == pytest.approx(math.degrees(2.0 / 4.0))
    assert limit[2] == pytest.approx(81.0, abs=1.0)  # the most agile speed, about 20 km/h
    assert limit[3] == pytest.approx(math.degrees(8.0 / 30.0))


def test_doubt_about_speed_widens_the_limit() -> None:
    sure = vehicle_yaw_limit(np.array([30.0]), MountConfig(speed_doubt=1.0))
    unsure = vehicle_yaw_limit(np.array([30.0]), MountConfig(speed_doubt=3.0))
    assert unsure[0] == pytest.approx(8.0 / 10.0)
    assert unsure[0] > sure[0]


def test_a_fixed_camera_turning_within_the_limit_is_left_exactly_alone() -> None:
    """A dashcam's real turn, 90 degrees at a corner at 20 km/h, must come through untouched."""
    t = series(20.0)
    rate = np.where((t > 5) & (t < 9), 22.5, 0.0) + 0.3 * np.sin(t)
    camera = rates_to_steps(rate)
    vehicle, mount = separate_mount(camera, np.full(len(t), 5.5), np.full(len(t), DT))
    assert np.array_equal(vehicle, camera)
    assert not mount.any()


def test_a_glance_on_a_straight_road_is_the_mount() -> None:
    """A head looking 50 degrees aside at 50 km/h: no car turns at 170 deg/s at that speed."""
    t = series(30.0)
    camera = rates_to_steps(glance(t, 4.0, 50.0))
    vehicle, mount = separate_mount(camera, np.full(len(t), 14.0), np.full(len(t), DT))
    assert abs(math.degrees(vehicle.sum())) < 1.0
    # Held for a second, a glance leaks a little into the car before it ends; a tenth at most.
    assert math.degrees(np.abs(vehicle).sum()) < 0.2 * 50.0
    assert math.degrees(mount.max()) == pytest.approx(50.0, abs=2.0)
    assert np.allclose(vehicle + np.diff(mount), camera)


def test_a_glance_in_a_bend_keeps_the_bend() -> None:
    t = series(12.0)
    bend = np.full(len(t), 6.0)  # deg/s, a long sweeping bend at 50 km/h
    camera = rates_to_steps(bend + glance(t, 4.0, 50.0))
    vehicle, _ = separate_mount(camera, np.full(len(t), 14.0), np.full(len(t), DT))
    assert math.degrees(vehicle.sum()) == pytest.approx(6.0 * 12.0, rel=0.1)


def test_a_standing_car_cannot_turn() -> None:
    t = series(6.0)
    camera = rates_to_steps(glance(t, 2.0, 20.0, swing_s=1.0))
    vehicle, mount = separate_mount(camera, np.zeros(len(t)), np.full(len(t), DT))
    assert not vehicle.any()
    assert math.degrees(mount.max()) == pytest.approx(20.0, abs=0.5)


def test_a_mount_that_stays_turned_drifts_back_to_the_car_as_it_drives() -> None:
    """A phone knocked 30 degrees and left there: the car's heading catches the camera's in the end,
    over distance, and never faster than the car can turn."""
    t = series(40.0)
    rate = np.where((t >= 5.0) & (t < 5.2), 150.0, 0.0)
    camera = rates_to_steps(rate)
    speed = np.full(len(t), 10.0)
    vehicle, mount = separate_mount(camera, speed, np.full(len(t), DT))
    assert math.degrees(mount[round(5.3 / DT)]) == pytest.approx(30.0, abs=1.0)
    assert abs(math.degrees(mount[-1])) < 3.0
    assert np.all(np.abs(vehicle) <= vehicle_yaw_limit(speed) * DT + 1e-12)
