from clayview.core.types import Box, Detection
from clayview.orient import crop_bounds, sample_every, wanted_detections

EVERY = 5


def detection(frame: int, name: str = "car", height: float = 60.0) -> Detection:
    return Detection(frame, 1, name, 0.8, Box(100.0, 100.0, 180.0, 100.0 + height))


def test_crop_adds_a_margin_and_stays_in_frame() -> None:
    assert crop_bounds(Box(100.0, 100.0, 200.0, 150.0), 1920, 1080) == (90, 90, 210, 160)
    assert crop_bounds(Box(0.0, 0.0, 1920.0, 1080.0), 1920, 1080) == (0, 0, 1920, 1080)


def test_only_sampled_cars_big_enough_to_read_are_wanted() -> None:
    picked = wanted_detections(
        [
            detection(0),
            detection(1),
            detection(EVERY, "truck"),
            detection(2 * EVERY, height=10.0),
            detection(3 * EVERY),
        ],
        EVERY,
    )
    assert sorted(picked) == [0, 3 * EVERY]


def test_views_are_sampled_in_time_not_frames() -> None:
    assert sample_every(30.0) == 5
    assert sample_every(60.0) == 10
    assert sample_every(2.0) == 1
