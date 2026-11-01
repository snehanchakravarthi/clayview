import math

import pytest

from clayview.core.camera import Camera, horizon_from_pitch


def make_camera(**overrides: object) -> Camera:
    settings: dict = {
        "image_width": 1920,
        "image_height": 1080,
        "height_m": 1.3,
        "vfov_deg": 60.0,
        "horizon_y": 540.0,
    }
    settings.update(overrides)
    return Camera(**settings)


def test_focal_length_from_vertical_fov() -> None:
    assert make_camera().focal_px == pytest.approx(540.0 / math.tan(math.radians(30.0)))


def test_ground_distance_is_f_h_over_row_offset() -> None:
    camera = make_camera()
    assert camera.ground_distance(640.0) == pytest.approx(camera.focal_px * 1.3 / 100.0)


def test_ground_distance_doubles_when_row_offset_halves() -> None:
    camera = make_camera()
    assert camera.ground_distance(590.0) == pytest.approx(2.0 * camera.ground_distance(640.0))


@pytest.mark.parametrize("row", [540.0, 400.0])
def test_nothing_on_or_above_the_horizon_has_a_ground_distance(row: float) -> None:
    assert make_camera().ground_distance(row) is None


def test_lateral_uses_offset_from_centre() -> None:
    camera = make_camera()
    assert camera.lateral(960.0, 20.0) == 0.0
    assert camera.lateral(960.0 + camera.focal_px, 20.0) == pytest.approx(20.0)
    assert camera.lateral(960.0 - camera.focal_px, 20.0) == pytest.approx(-20.0)


def test_center_x_moves_dead_ahead() -> None:
    camera = make_camera(center_x=1000.0)
    assert camera.lateral(1000.0, 30.0) == 0.0


def test_ground_point_and_image_point_are_inverses() -> None:
    camera = make_camera()
    u, v = camera.image_point(-2.5, 18.0)
    x, y = camera.ground_point(u, v)
    assert x == pytest.approx(-2.5)
    assert y == pytest.approx(18.0)


def test_size_distance_is_f_w_over_pixels() -> None:
    camera = make_camera()
    assert camera.size_distance(1.8, 90.0) == pytest.approx(camera.focal_px * 1.8 / 90.0)


def test_pitch_and_horizon_round_trip() -> None:
    row = horizon_from_pitch(1080, 60.0, 4.0)
    assert row < 540.0  # looking down lifts the horizon in the image
    assert make_camera(horizon_y=row).pitch_deg == pytest.approx(4.0)


def test_default_horizon_is_mid_frame() -> None:
    assert make_camera(horizon_y=None).horizon == 540.0
