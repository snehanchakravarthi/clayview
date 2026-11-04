import numpy as np
import pytest

from clayview.core.horizon import (
    estimate_ground_bottom,
    estimate_visible_bottom,
    fit_expansion,
    fit_focus_of_expansion,
    fit_vehicle_horizon,
    vehicle_height_for_horizon,
    vehicle_samples,
)
from clayview.core.types import Box, Detection, FlowPair


def width_samples(horizon: float, height_m: float, count: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = rng.uniform(horizon + 15.0, horizon + 260.0, count)
    widths = (1.8 / height_m) * (rows - horizon) * rng.normal(1.0, 0.04, count)
    return np.stack([rows, widths], axis=1)


def test_vehicle_widths_recover_horizon_and_height() -> None:
    fit = fit_vehicle_horizon(width_samples(620.0, 1.25, 200), 1080)
    assert fit is not None
    assert fit.horizon_y == pytest.approx(620.0, abs=15.0)
    assert fit.height_m == pytest.approx(1.25, abs=0.08)


def test_vehicle_fit_survives_outliers() -> None:
    samples = width_samples(620.0, 1.25, 200)
    rng = np.random.default_rng(2)
    junk = np.stack([rng.uniform(640, 880, 40), rng.uniform(20, 400, 40)], axis=1)
    fit = fit_vehicle_horizon(np.concatenate([samples, junk]), 1080)
    assert fit is not None
    assert fit.horizon_y == pytest.approx(620.0, abs=25.0)


def test_vehicle_fit_needs_a_spread_of_rows() -> None:
    samples = np.full((60, 2), [700.0, 100.0])
    assert fit_vehicle_horizon(samples, 1080) is None


def test_height_from_widths_with_a_known_horizon() -> None:
    samples = width_samples(620.0, 1.15, 100)
    assert vehicle_height_for_horizon(samples, 620.0) == pytest.approx(1.15, abs=0.03)


def test_vehicle_samples_keep_rear_on_cars_only() -> None:
    def detection(box: Box, name: str = "car", confidence: float = 0.8) -> Detection:
        return Detection(0, 1, name, confidence, box)

    rear_on = detection(Box(900.0, 500.0, 1010.0, 600.0))
    side_on = detection(Box(900.0, 500.0, 1200.0, 600.0))
    truck = detection(Box(900.0, 500.0, 1010.0, 600.0), "truck")
    faint = detection(Box(900.0, 500.0, 1010.0, 600.0), confidence=0.2)
    cut_off = detection(Box(0.0, 500.0, 110.0, 600.0))
    samples = vehicle_samples([rear_on, side_on, truck, faint, cut_off], 1920, 1080)
    assert samples.tolist() == [[600.0, 110.0]]


def test_height_fit_takes_the_box_padding_off() -> None:
    # Cars of 1.8 m under a 1.5 m camera, every box drawn 4 px too wide. Most samples are far and
    # small, as on comma2k19, where the padding read the camera 1.37 m instead of about 1.54 m.
    rows = np.linspace(400.0, 460.0, 40)
    detections = [
        Detection(index * 5, index, "car", 0.9, Box(960.0 - w / 2.0, row - w, 960.0 + w / 2.0, row))
        for index, (row, w) in enumerate(zip(rows, (rows - 378.0) * 1.2 + 4.0, strict=True))
    ]
    padded = vehicle_samples(detections, 1920, 1080)
    unpadded = vehicle_samples(detections, 1920, 1080, box_pad_px=4.0)
    assert vehicle_height_for_horizon(padded, 378.0) < 1.42
    assert vehicle_height_for_horizon(unpadded, 378.0) == pytest.approx(1.5, abs=0.01)


def expansion_pairs(foe: tuple[float, float], frames: int = 12, seed: int = 1) -> list[FlowPair]:
    rng = np.random.default_rng(seed)
    center = np.array(foe)
    pairs: list[FlowPair] = []
    for _ in range(frames):
        points = rng.uniform([50, 50], [1870, 1030], (80, 2))
        moved = points + (points - center) * 0.02 + rng.normal(0.0, 0.1, (80, 2))
        pairs.append(FlowPair(points.astype(np.float32), moved.astype(np.float32)))
    return pairs


def test_focus_of_expansion_is_found() -> None:
    fit = fit_focus_of_expansion(expansion_pairs((1010.0, 640.0)), 1920, 1080)
    assert fit is not None
    assert fit.x == pytest.approx(1010.0, abs=6.0)
    assert fit.y == pytest.approx(640.0, abs=6.0)


def test_focus_of_expansion_needs_motion() -> None:
    points = np.random.default_rng(1).uniform(0, 1000, (100, 2)).astype(np.float32)
    assert fit_focus_of_expansion([FlowPair(points, points)] * 5, 1920, 1080) is None


def test_yaw_is_removed_from_the_focus_of_expansion() -> None:
    pairs = expansion_pairs((960.0, 640.0))
    shifted = [FlowPair(p.prev, p.next + np.array([1.5, 0.0], np.float32)) for p in pairs]
    fit = fit_focus_of_expansion(shifted, 1920, 1080, [(1.5, 0.0)] * len(pairs))
    assert fit is not None
    assert fit.x == pytest.approx(960.0, abs=6.0)


def test_visible_bottom_finds_the_hood() -> None:
    rng = np.random.default_rng(1)
    pairs: list[FlowPair] = []
    for _ in range(20):
        road = rng.uniform([0, 300], [1920, 860], (150, 2))
        hood = rng.uniform([0, 900], [1920, 1080], (60, 2))
        points = np.concatenate([road, hood]).astype(np.float32)
        moved = points.copy()
        moved[:150, 1] += 5.0
        pairs.append(FlowPair(points, moved))
    bottom = estimate_visible_bottom(pairs, 1080)
    assert bottom is not None
    assert 860.0 <= bottom <= 950.0


def test_visible_bottom_is_none_without_a_hood() -> None:
    rng = np.random.default_rng(1)
    points = rng.uniform([0, 0], [1920, 1080], (300, 2)).astype(np.float32)
    moved = points + np.array([0.0, 5.0], np.float32)
    assert estimate_visible_bottom([FlowPair(points, moved)] * 10, 1080) is None


def scene_pairs(
    foe: tuple[float, float], rotation: list[tuple[float, float]], right_only: bool = False
) -> list[FlowPair]:
    """Static points at random depths flowing away from the FoE, plus each frame's rotation."""
    rng = np.random.default_rng(3)
    pairs: list[FlowPair] = []
    for shift in rotation:
        low = [1100.0 if right_only else 50.0, 50.0]
        points = rng.uniform(low, [1870.0, 1030.0], (200, 2))
        rate = rng.uniform(0.002, 0.05, (200, 1))  # speed over depth: near points flow faster
        moved = points + (points - np.array(foe)) * rate + np.array(shift)
        moved += rng.normal(0.0, 0.05, moved.shape)
        pairs.append(FlowPair(points.astype(np.float32), moved.astype(np.float32)))
    return pairs


def test_expansion_is_solved_with_rotation() -> None:
    """Fitting the FoE first and rotation after left it 7 px off on comma2k19, and every pixel of
    column error read as 2.7 degrees a minute of false turning."""
    rotation = [(float(np.sin(i / 3.0)) * 3.0, float(np.cos(i / 4.0))) for i in range(30)]
    pairs = scene_pairs((1010.0, 600.0), rotation, right_only=True)
    fit = fit_expansion(pairs, 1920, 1080, bottom=1080.0)
    assert fit is not None
    assert fit.x == pytest.approx(1010.0, abs=1.5)
    assert fit.y == pytest.approx(600.0, abs=1.5)


def ground_pairs(horizon: float, hood: float, hood_flow: float, frames: int = 20) -> list[FlowPair]:
    """Road flow growing with (row - horizon)^2 down to the hood; the hood does its own thing."""
    rng = np.random.default_rng(4)
    pairs: list[FlowPair] = []
    for _ in range(frames):
        road = rng.uniform([0.0, horizon + 5.0], [1920.0, hood], (300, 2))
        below = rng.uniform([0.0, hood], [1920.0, 1080.0], (300, 2))
        road_next = road.copy()
        road_next[:, 1] += (road[:, 1] - horizon) ** 2 * 2e-4
        below_next = below + np.array([0.0, hood_flow])
        points = np.concatenate([road, below]).astype(np.float32)
        moved = np.concatenate([road_next, below_next]).astype(np.float32)
        pairs.append(FlowPair(points, moved))
    return pairs


@pytest.mark.parametrize("hood_flow", [0.0, -3.0, 4.0])
def test_ground_bottom_finds_where_the_road_ends(hood_flow: float) -> None:
    """A static reflection, or a shiny hood whose reflections move, both counted as road before:
    on comma2k19 the static points outnumbered the road and cut ego speed by 40%."""
    bottom = estimate_ground_bottom(ground_pairs(540.0, 800.0, hood_flow), 540.0, 1080)
    assert bottom is not None
    assert 760.0 <= bottom <= 810.0


def test_ground_bottom_needs_the_road_to_move() -> None:
    still = [FlowPair(p.prev, p.prev) for p in ground_pairs(540.0, 800.0, 0.0)]
    assert estimate_ground_bottom(still, 540.0, 1080) is None
