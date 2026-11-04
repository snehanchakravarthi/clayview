import numpy as np
import pytest

from clayview.core.camera import Camera
from clayview.core.ego import (
    estimate_ego_series,
    road_speeds,
    scene_expansion,
    speed_from_points,
    yaw_step,
)
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


def test_points_near_the_foe_still_see_a_turn() -> None:
    """The floor on a point's distance from the FoE only sets its weight. Used in the roll term too,
    it read a 3 px turn of points within 20 px of the FoE as 0.002 px."""
    x = CAMERA.cx + np.linspace(-14.0, 14.0, 8)
    points = np.stack([x, np.full(8, CAMERA.horizon + 10.0)], axis=1).astype(np.float32)
    turning = FlowPair(points, points + np.array([-3.0, 0.0], np.float32))
    assert yaw_step(turning, CAMERA) == pytest.approx(3.0 / CAMERA.focal_px, rel=0.02)


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


def test_lane_lines_and_shadows_do_not_drag_speed_down() -> None:
    """On I-270 up to half the road points sat on lane lines, which run towards the FoE so the
    tracker sees only their sideways drift, or on the shadows of cars pacing ego. Their rows stand
    still, and ego read 0.85 of the lane-dash speed although every other road point read 1.0."""
    camera = Camera(1280, 720, height_m=1.36, vfov_deg=9.2, horizon_y=356.0)
    rng = np.random.default_rng(4)
    forward_m = 32.0 / 14.2

    def project(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.array([camera.image_point(a, b) for a, b in zip(x, y, strict=True)])

    x, y = rng.uniform(-3.0, 3.0, 60), rng.uniform(25.0, 60.0, 60)
    road_prev, road_next = project(x, y), project(x, y - forward_m)

    # Lane lines: the true flow runs along the line, which is radial, so the tracker catches only
    # part of it, plus a sideways drift perpendicular to the line.
    lx, ly = rng.choice([-1.8, 1.8], 45), rng.uniform(25.0, 60.0, 45)
    line_prev = project(lx, ly)
    caught = rng.uniform(0.0, 0.4, (45, 1))
    outward = line_prev - [camera.cx, camera.horizon]
    across = np.stack([-outward[:, 1], outward[:, 0]], axis=1)
    across /= np.hypot(across[:, 0], across[:, 1])[:, None]
    line_next = line_prev + caught * (project(lx, ly - forward_m) - line_prev)
    line_next += across * rng.uniform(3.0, 6.0, (45, 1))

    # Shadows of the cars ahead keep roughly pace with ego: sideways drift, a pixel or two of rows.
    shadow_prev = project(rng.uniform(-3.0, 3.0, 30), rng.uniform(30.0, 60.0, 30))
    shadow_next = shadow_prev + np.stack(
        [rng.choice([-1.0, 1.0], 30) * rng.uniform(3.0, 6.0, 30), rng.normal(0.0, 1.5, 30)], axis=1
    )

    prev = np.concatenate([road_prev, line_prev, shadow_prev])
    next_ = np.concatenate([road_next, line_next, shadow_next])
    next_ += rng.normal(0.0, 0.3, next_.shape)  # tracker noise
    pair = FlowPair(prev.astype(np.float32), next_.astype(np.float32))
    speeds = road_speeds(pair, camera, 1.0 / 14.2)
    assert speed_from_points(speeds) == pytest.approx(32.0, rel=0.05)


def scenery(forward_m: float, rng: np.random.Generator, count: int = 60) -> FlowPair:
    """Trees and poles beside and above the road, seen from two positions forward_m apart."""
    x = rng.choice([-1.0, 1.0], count) * rng.uniform(4.0, 15.0, count)
    z = rng.uniform(2.0, 10.0, count)  # height above the road
    y = rng.uniform(15.0, 60.0, count)

    def project(depth: np.ndarray) -> np.ndarray:
        column = CAMERA.cx + CAMERA.focal_px * x / depth
        row = CAMERA.horizon + CAMERA.focal_px * (CAMERA.height_m - z) / depth
        return np.stack([column, row], axis=1)

    return FlowPair(project(y).astype(np.float32), project(y - forward_m).astype(np.float32))


def glass(rng: np.random.Generator, count: int = 120) -> FlowPair:
    """Raindrops over the road band: they stay put, give or take a little sliding."""
    prev = np.stack([rng.uniform(200, 1700, count), rng.uniform(700, 1000, count)], axis=1)
    next_ = prev + rng.normal(0.0, 0.15, prev.shape)
    return FlowPair(prev.astype(np.float32), next_.astype(np.float32))


def combine(*pairs: FlowPair) -> FlowPair:
    return FlowPair(
        np.concatenate([pair.prev for pair in pairs]), np.concatenate([pair.next for pair in pairs])
    )


def test_scene_above_the_road_says_whether_ego_moves() -> None:
    rng = np.random.default_rng(5)
    assert scene_expansion(scenery(0.5, rng), CAMERA, DT) > 0.9
    assert scene_expansion(scenery(0.0, rng), CAMERA, DT) < 0.1
    assert scene_expansion(glass(rng), CAMERA, DT) is None  # nothing above the road


def test_rain_on_the_glass_is_not_a_stop() -> None:
    """On wintry_wipers raindrops over the road band outnumbered the trackable asphalt for seconds
    at a time and ego read 0 to 4 km/h at highway speed, while the trees above the horizon kept
    expanding away from the FoE."""
    rng = np.random.default_rng(6)
    forward_m = 0.5  # 15 m/s
    pairs = []
    for index in range(120):
        road = ground_pair(forward_m, seed=index)
        if 30 <= index < 90:  # two seconds where the glass hides all but a few road points
            road = FlowPair(road.prev[:10], road.next[:10])
            road = combine(road, glass(rng))
        pairs.append(combine(road, scenery(forward_m, rng)))
    times = [index * DT for index in range(121)]
    steps, _ = estimate_ego_series(pairs, times, CAMERA)
    speeds = np.array([step.forward_m for step in steps]) / DT
    assert np.all(speeds[30:90] > 12.0)


def test_a_wiper_sweep_while_stopped_is_not_motion() -> None:
    """Stopped in the rain: the road and the scenery stand still, and a wiper drags a third of the
    points sideways for a few frames every second. The scene's share is a median over a second, so
    a sweep doesn't make ego move. (A sweep in the clip's very first frames does read as a brief
    roll-off from about 5 m/s, on main too: the median filter has nothing before it to outvote
    it.)"""
    rng = np.random.default_rng(7)
    pairs = []
    for index in range(120):
        pair = combine(ground_pair(0.0, seed=index), scenery(0.0, rng), glass(rng))
        if 12 <= index % 30 < 18:
            under = rng.random(len(pair.prev)) < 0.35
            sweep = np.where(under[:, None], [[60.0, 15.0]], 0.0) + rng.normal(
                0, 4, pair.prev.shape
            )
            pair = FlowPair(pair.prev, (pair.next + sweep * under[:, None]).astype(np.float32))
        pairs.append(pair)
    times = [index * DT for index in range(121)]
    steps, _ = estimate_ego_series(pairs, times, CAMERA)
    speeds = np.array([step.forward_m for step in steps]) / DT
    assert np.all(speeds < 0.5)
