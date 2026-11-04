import numpy as np
import pytest

from clayview.core.camera import Camera
from clayview.core.ego import estimate_ego_series, road_speeds, speed_from_points, yaw_step
from clayview.core.types import FlowPair

CAMERA = Camera(1920, 1080, height_m=1.2, vfov_deg=60.0, horizon_y=620.0)
DT = 1.0 / 30.0


def ground_pair(forward_m: float, seed: int = 1) -> FlowPair:
    """Road points seen from two camera positions, forward_m apart."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-6.0, 6.0, 80)
    y = rng.uniform(5.0, 20.0, 80)
    prev = [CAMERA.image_point(a, b) for a, b in zip(x, y, strict=True)]
    next_ = [CAMERA.image_point(a, b - forward_m) for a, b in zip(x, y, strict=True)]
    return FlowPair(np.array(prev, np.float32), np.array(next_, np.float32))


def test_road_flow_gives_ego_speed() -> None:
    speeds = road_speeds(ground_pair(0.5), CAMERA, DT)
    assert speed_from_points(speeds) == pytest.approx(15.0, rel=0.05)


def test_static_reflections_do_not_drag_speed_down() -> None:
    """Dash reflections sit over the road. A plain median reported 9 km/h for a car doing 28."""
    speeds = road_speeds(ground_pair(0.5), CAMERA, DT)
    with_reflections = np.concatenate([speeds, np.zeros(len(speeds)) + 0.05])
    assert np.median(with_reflections) < 12.0
    assert speed_from_points(with_reflections) == pytest.approx(15.0, rel=0.05)


def test_stopped_car_reads_zero() -> None:
    noise = np.random.default_rng(1).normal(0.0, 0.2, 80)
    assert speed_from_points(noise) == pytest.approx(0.0, abs=0.5)


def test_too_few_points_is_no_measurement() -> None:
    assert speed_from_points(np.array([3.0, 4.0])) is None


def test_yaw_right_shifts_far_points_left() -> None:
    rng = np.random.default_rng(1)
    points = np.stack([rng.uniform(100, 1800, 60), rng.uniform(600, 640, 60)], axis=1).astype(
        np.float32
    )
    moved = points + np.array([-10.0, 0.0], np.float32)
    assert yaw_step(FlowPair(points, moved), CAMERA) == pytest.approx(10.0 / CAMERA.focal_px)


def test_roadside_flow_is_not_a_turn() -> None:
    """Driving straight past trees on one side only. The median sideways shift near the horizon read
    that as turning; on comma2k19 it turned 57 degrees a minute against a true 10."""
    rng = np.random.default_rng(2)
    points = np.stack([rng.uniform(1100, 1900, 120), rng.uniform(400, 1000, 120)], axis=1)
    rate = rng.uniform(0.005, 0.05, (120, 1))
    moved = points + (points - [CAMERA.cx, CAMERA.horizon]) * rate
    pair = FlowPair(points.astype(np.float32), moved.astype(np.float32))
    assert yaw_step(pair, CAMERA) == pytest.approx(0.0, abs=1e-4)
    turning = FlowPair(pair.prev, pair.next + np.array([-6.0, 0.0], np.float32))
    assert yaw_step(turning, CAMERA) == pytest.approx(6.0 / CAMERA.focal_px, rel=0.02)


def rolled(pair: FlowPair, roll_rad: float) -> FlowPair:
    """The pair with the camera also rolling about the focus of expansion."""
    centre = np.array([CAMERA.cx, CAMERA.horizon])
    cos, sin = np.cos(roll_rad), np.sin(roll_rad)
    offset = pair.next - centre
    turned = np.stack(
        [cos * offset[:, 0] - sin * offset[:, 1], sin * offset[:, 0] + cos * offset[:, 1]], 1
    )
    return FlowPair(pair.prev, (turned + centre).astype(np.float32))


def test_body_roll_is_not_a_turn() -> None:
    """On comma2k19 the camera rolls three times as much as it yaws from frame to frame. Road points
    below the FoE see a roll as a sideways shift, and the yaw model had no roll term: it turned 19
    degrees a minute against a true 10, with a correlation of 0.30 to the gyro."""
    pair = ground_pair(1.0)
    for roll_deg in (-0.1, 0.1):
        assert yaw_step(rolled(pair, np.radians(roll_deg)), CAMERA) == pytest.approx(0.0, abs=2e-5)
    turning = FlowPair(pair.prev, pair.next + np.array([-3.0, 0.0], np.float32))
    turn = yaw_step(rolled(turning, np.radians(0.1)), CAMERA)
    assert turn == pytest.approx(3.0 / CAMERA.focal_px, rel=0.05)


def test_series_is_the_same_in_seconds_at_any_frame_rate() -> None:
    """Smoothing windows were in frames, so a 60 fps clip got half the smoothing of a 30 fps one."""
    results = []
    for fps in (30.0, 60.0):
        count = int(2 * fps)
        pairs = [ground_pair(12.0 / fps, seed) for seed in range(count)]
        times = [index / fps for index in range(count + 1)]
        steps, _ = estimate_ego_series(pairs, times, CAMERA)
        results.append(np.array([step.forward_m * fps for step in steps]))
    assert np.median(results[0]) == pytest.approx(np.median(results[1]), rel=0.03)


def test_series_fills_gaps_and_stays_smooth() -> None:
    pairs = [ground_pair(0.4, seed) for seed in range(40)]
    pairs[20] = FlowPair(np.empty((0, 2), np.float32), np.empty((0, 2), np.float32))
    times = [index * DT for index in range(41)]
    steps, coverage = estimate_ego_series(pairs, times, CAMERA)
    assert coverage > 0.9
    assert all(step.forward_m == pytest.approx(0.4, rel=0.1) for step in steps)


def test_zoomed_camera_at_highway_speed() -> None:
    """A 9 degree camera on I-270 at 115 km/h and 14 fps sees road from 17 m out. A 22 m range cap
    threw all of it away and ego read zero."""
    camera = Camera(1280, 720, height_m=1.36, vfov_deg=9.2, horizon_y=356.0)
    rng = np.random.default_rng(3)
    x = rng.uniform(-3.0, 3.0, 80)
    y = rng.uniform(25.0, 120.0, 80)
    forward_m = 32.0 / 14.2
    prev = [camera.image_point(a, b) for a, b in zip(x, y, strict=True)]
    next_ = [camera.image_point(a, b - forward_m) for a, b in zip(x, y, strict=True)]
    pair = FlowPair(np.array(prev, np.float32), np.array(next_, np.float32))
    speeds = road_speeds(pair, camera, 1.0 / 14.2)
    assert speed_from_points(speeds) == pytest.approx(32.0, rel=0.05)
