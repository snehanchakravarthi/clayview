from clayview.core.types import Box, Detection
from clayview.orient import SAMPLE_EVERY, crop_bounds, wanted_detections


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
            detection(SAMPLE_EVERY, "truck"),
            detection(2 * SAMPLE_EVERY, height=10.0),
            detection(3 * SAMPLE_EVERY),
        ]
    )
    assert sorted(picked) == [0, 3 * SAMPLE_EVERY]
