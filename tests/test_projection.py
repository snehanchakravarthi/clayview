import pytest

from clayview.core.camera import Camera
from clayview.core.projection import (
    DETECTOR_BOX_PAD_PX,
    ProjectionConfig,
    box_pad_px,
    footprint_half_extent,
    project_box,
)
from clayview.core.types import (
    FLAG_CLAMPED,
    FLAG_NO_GROUND,
    FLAG_SIZE_PRIOR,
    FLAG_TRUNCATED,
    Box,
)

CAMERA = Camera(1920, 1080, height_m=1.3, vfov_deg=60.0, horizon_y=540.0)


def car_box(distance: float, lateral: float = 0.0, width_m: float = 1.8) -> Box:
    """A rear-on car box for a car really at this position."""
    u, v = CAMERA.image_point(lateral, distance)
    half = CAMERA.focal_px * width_m / distance / 2.0
    height = CAMERA.focal_px * 1.5 / distance
    return Box(u - half, v - height, u + half, v)


def test_consistent_car_is_recovered() -> None:
    result = project_box(car_box(15.0, -2.0), "car", CAMERA)
    assert result is not None
    assert result.y == pytest.approx(15.0, abs=0.01)
    assert result.x == pytest.approx(-2.0, abs=0.01)
    assert not result.flags


def test_ground_leads_up_close_and_the_size_prior_leads_far_away() -> None:
    near = project_box(car_box(8.0), "car", CAMERA)
    far = project_box(car_box(100.0), "car", CAMERA)
    assert near is not None and far is not None
    assert near.ground_weight > 0.9
    assert far.ground_weight < 0.5
    assert FLAG_SIZE_PRIOR in far.flags


def test_size_prior_rescues_a_box_on_the_horizon() -> None:
    box = Box(900.0, 460.0, 1000.0, 541.0)
    result = project_box(box, "car", CAMERA)
    assert result is not None
    assert FLAG_NO_GROUND in result.flags
    assert result.y == pytest.approx(CAMERA.focal_px * 1.8 / 100.0, rel=0.05)


def test_uncertainty_grows_with_range() -> None:
    near = project_box(car_box(10.0), "car", CAMERA)
    far = project_box(car_box(60.0), "car", CAMERA)
    assert near is not None and far is not None
    assert far.sigma_y > near.sigma_y


def test_box_cut_by_the_bottom_edge_has_no_ground_contact() -> None:
    box = Box(600.0, 700.0, 1300.0, 1079.0)
    result = project_box(box, "car", CAMERA)
    assert result is not None
    assert FLAG_NO_GROUND in result.flags
    assert FLAG_TRUNCATED in result.flags
    assert result.ground_weight == 0.0


def test_hood_counts_as_the_frame_bottom() -> None:
    camera = Camera(1920, 1080, height_m=1.3, horizon_y=540.0, visible_bottom=900.0)
    result = project_box(Box(600.0, 650.0, 1300.0, 899.0), "car", camera)
    assert result is not None
    assert FLAG_NO_GROUND in result.flags


def test_box_cut_by_the_side_keeps_its_honest_edge() -> None:
    # Really at x = 12 m, 12 m out, with the right part of the car beyond the frame.
    true_box = car_box(12.0, 12.0)
    cut = Box(true_box.x1, true_box.y1, 1920.0, true_box.y2)
    assert true_box.x2 > 1920.0
    result = project_box(cut, "car", CAMERA)
    assert result is not None
    assert FLAG_TRUNCATED in result.flags
    assert result.x == pytest.approx(12.0, abs=0.25)


def test_box_with_nothing_trustworthy_is_dropped() -> None:
    assert project_box(Box(0.0, 0.0, 1920.0, 1080.0), "car", CAMERA) is None


def test_far_objects_are_clamped_and_flagged() -> None:
    config = ProjectionConfig(max_range_m=50.0)
    result = project_box(Box(955.0, 530.0, 965.0, 537.0), "car", CAMERA, config)
    assert result is not None
    assert result.y == 50.0
    assert FLAG_CLAMPED in result.flags


def test_oblique_vehicle_uses_height_not_width() -> None:
    # A car seen at an angle: box height is still honest, width is not.
    distance = 20.0
    height = CAMERA.focal_px * 1.5 / distance
    box = Box(800.0, 400.0, 800.0 + 2.0 * height, 400.0 + height)
    result = project_box(box, "car", CAMERA, ProjectionConfig(sigma_size_real=0.01))
    assert result is not None
    assert result.y == pytest.approx(distance, rel=0.1)


def test_person_uses_height() -> None:
    distance = 12.0
    height = CAMERA.focal_px * 1.7 / distance
    u, v = CAMERA.image_point(1.0, distance)
    box = Box(u - 60.0, v - height, u + 60.0, v)
    result = project_box(box, "person", CAMERA)
    assert result is not None
    assert result.y == pytest.approx(distance, rel=0.05)


def test_footprint_extent_is_half_length_end_on_and_half_width_side_on() -> None:
    assert footprint_half_extent("car", 0.0) == pytest.approx(2.25)
    assert footprint_half_extent("car", 1.5707963) == pytest.approx(0.9, abs=0.01)


# comma2k19, Example_1: median YOLO boxes of rear-on lead cars by CAN radar range. Columns are radar
# range in metres, box width, box height and box bottom in pixels. The camera is 1164x874 with a
# 910 px focal length, seen through a 960 px detector; the horizon is 378 px.
COMMA_LEADS = [
    (29.6, 56.8, 48.3, 423.5),
    (37.7, 45.6, 38.4, 412.4),
    (63.3, 29.4, 25.3, 398.5),
]
# The radar is in the front bumper, about 1.5 m ahead of a RAV4's windshield camera.
COMMA_RADAR_AHEAD_M = 1.5


def _comma_camera(pad: float, height_m: float = 1.54) -> Camera:
    return Camera(1164, 874, height_m=height_m, vfov_deg=51.3, horizon_y=378.3, box_pad_px=pad)


def test_box_padding_in_the_detector_grid_scales_with_the_frame() -> None:
    assert box_pad_px(1164, 874, 960) == pytest.approx(DETECTOR_BOX_PAD_PX * 1164 / 960)
    assert box_pad_px(1080, 1920, 960) == pytest.approx(2.0 * DETECTOR_BOX_PAD_PX)
    assert box_pad_px(1164, 874, None) == 0.0


@pytest.mark.parametrize(("radar", "width", "height", "bottom"), COMMA_LEADS)
def test_comma2k19_leads_match_radar(
    radar: float, width: float, height: float, bottom: float
) -> None:
    box = Box(582.0 - width / 2.0, bottom - height, 582.0 + width / 2.0, bottom)
    pad = box_pad_px(1164, 874, 960)
    result = project_box(box, "car", _comma_camera(pad))
    assert result is not None
    assert result.y == pytest.approx(radar + COMMA_RADAR_AHEAD_M, rel=0.06)


def test_unpadded_boxes_put_far_comma2k19_leads_short() -> None:
    # The old pipeline: padding left in, and the 1.37 m camera height it calibrated from them.
    radar, width, height, bottom = COMMA_LEADS[-1]
    box = Box(582.0 - width / 2.0, bottom - height, 582.0 + width / 2.0, bottom)
    result = project_box(box, "car", _comma_camera(0.0, height_m=1.37))
    assert result is not None
    assert result.y < 0.92 * (radar + COMMA_RADAR_AHEAD_M)
