import math

import pytest

from clayview.core.camera import Camera
from clayview.core.heading import seen_aspect
from clayview.core.tracks import Segment, build_tracks, stitch_fragments
from clayview.core.types import FLAG_HELD, Box, Detection, EgoStep

CAMERA = Camera(1920, 1080, height_m=1.2, vfov_deg=60.0, horizon_y=620.0)
DT = 1.0 / 30.0
FRAMES = 300
TIMES = [index * DT for index in range(FRAMES)]
SEGMENTS = [Segment(0, 0, FRAMES, CAMERA)]


def box_for(x: float, y: float, width_m: float = 1.8) -> Box:
    u, v = CAMERA.image_point(x, y)
    half = CAMERA.focal_px * width_m / y / 2.0
    height = CAMERA.focal_px * 1.5 / y
    return Box(u - half, v - height, u + half, v)


def side_on_box(x: float, y: float) -> Box:
    """A car parallel to ego shows more of its side the further it sits off to one side."""
    seen_from = math.atan2(x, y)
    height = CAMERA.focal_px * 1.5 / y
    half = float(seen_aspect(seen_from, 1.4, 2.7)) * height / 2.0
    u, v = CAMERA.image_point(x, y)
    return Box(u - half, v - height, u + half, v)


def track(
    track_id: int,
    frames: range,
    position: object,
    name: str = "car",
    confidence: float = 0.8,
) -> list[Detection]:
    return [
        Detection(frame, track_id, name, confidence, box_for(*position(frame))) for frame in frames
    ]


def still(x: float, y: float) -> object:
    return lambda frame: (x, y)


def no_ego() -> list[EgoStep]:
    return [EgoStep() for _ in range(FRAMES - 1)]


def test_short_tracks_are_dropped() -> None:
    detections = track(1, range(10, 14), still(1.0, 20.0))
    assert build_tracks(detections, TIMES, SEGMENTS, no_ego()) == []


def test_a_real_track_comes_through_with_clean_ids() -> None:
    detections = track(77, range(0, 60), still(1.0, 20.0))
    tracks = build_tracks(detections, TIMES, SEGMENTS, no_ego())
    assert [t.id for t in tracks] == [1]
    assert tracks[0].samples[0].y == pytest.approx(20.0 + 2.25, abs=0.2)


def test_class_is_decided_by_vote() -> None:
    detections = track(1, range(0, 40), still(0.0, 20.0)) + track(
        1, range(40, 50), still(0.0, 20.0), "truck", 0.5
    )
    tracks = build_tracks(detections, TIMES, SEGMENTS, no_ego())
    assert tracks[0].class_name == "car"


def test_short_dropout_is_held_and_flagged() -> None:
    detections = track(1, range(0, 30), still(1.0, 20.0)) + track(
        1, range(40, 80), still(1.0, 20.0)
    )
    samples = build_tracks(detections, TIMES, SEGMENTS, no_ego())[0].samples
    in_gap = [s for s in samples if 30 <= s.frame < 40]
    assert len(in_gap) == 10
    assert all(FLAG_HELD in s.flags for s in in_gap)
    assert all(s.confidence < 0.8 for s in in_gap)


def test_track_coasts_past_its_last_detection() -> None:
    detections = track(1, range(0, 60), still(1.0, 20.0))
    samples = build_tracks(detections, TIMES, SEGMENTS, no_ego())[0].samples
    assert samples[-1].frame > 59
    assert FLAG_HELD in samples[-1].flags


def test_track_stops_at_a_cut() -> None:
    segments = [Segment(0, 0, 60, CAMERA), Segment(1, 60, FRAMES, CAMERA)]
    detections = track(1, range(0, 60), still(1.0, 20.0))
    samples = build_tracks(detections, TIMES, segments, no_ego())[0].samples
    assert samples[-1].frame == 59


def test_same_detector_id_in_two_shots_is_two_objects() -> None:
    segments = [Segment(0, 0, 100, CAMERA), Segment(1, 100, FRAMES, CAMERA)]
    detections = track(1, range(0, 60), still(1.0, 20.0)) + track(
        1, range(120, 180), still(1.0, 20.0)
    )
    tracks = build_tracks(detections, TIMES, segments, no_ego())
    assert [(t.id, t.segment) for t in tracks] == [(1, 0), (2, 1)]


def test_car_driving_ahead_of_a_moving_ego_faces_forward() -> None:
    ego = [EgoStep(forward_m=15.0 * DT) for _ in range(FRAMES - 1)]
    detections = track(1, range(0, 120), lambda frame: (2.0, 30.0 - 3.0 * frame * DT))
    sample = build_tracks(detections, TIMES, SEGMENTS, ego)[0].samples[60]
    assert abs(sample.heading) < math.radians(10.0)


def test_oncoming_car_faces_ego() -> None:
    ego = [EgoStep(forward_m=15.0 * DT) for _ in range(FRAMES - 1)]
    detections = track(1, range(0, 60), lambda frame: (-4.0, 70.0 - 30.0 * frame * DT))
    sample = build_tracks(detections, TIMES, SEGMENTS, ego)[0].samples[40]
    assert abs(sample.heading) > math.radians(165.0)


def test_parked_car_does_not_read_as_oncoming_when_ego_speed_is_off() -> None:
    """Real speed 15, estimate 11: the car appears to do 4 m/s towards ego. It must not."""
    ego = [EgoStep(forward_m=11.0 * DT) for _ in range(FRAMES - 1)]
    detections = [
        Detection(frame, 1, "car", 0.8, side_on_box(4.0, 45.0 - 15.0 * frame * DT))
        for frame in range(0, 70)
    ]
    samples = build_tracks(detections, TIMES, SEGMENTS, ego)[0].samples
    assert all(abs(s.heading) < math.radians(5.0) for s in samples)


def test_broken_track_is_stitched() -> None:
    first = track(1, range(0, 30), lambda f: (2.0, 30.0 - 0.1 * f))
    second = track(2, range(36, 70), lambda f: (2.0, 30.0 - 0.1 * f))
    assert len(stitch_fragments([first, second], TIMES, 1.0)) == 1


def test_stitching_ignores_a_different_object() -> None:
    first = track(1, range(0, 30), still(-8.0, 20.0))
    second = track(2, range(36, 70), still(6.0, 20.0))
    assert len(stitch_fragments([first, second], TIMES, 1.0)) == 2


def test_stitching_needs_a_short_gap() -> None:
    first = track(1, range(0, 30), still(2.0, 20.0))
    second = track(2, range(70, 120), still(2.0, 20.0))
    assert len(stitch_fragments([first, second], TIMES, 1.0)) == 2


def test_car_can_stitch_to_truck_but_not_to_person() -> None:
    first = track(1, range(0, 30), still(2.0, 20.0))
    truck = track(2, range(36, 70), still(2.0, 20.0), "truck")
    person = track(3, range(36, 70), still(2.0, 20.0), "person")
    assert len(stitch_fragments([first, truck], TIMES, 1.0)) == 1
    assert len(stitch_fragments([first, person], TIMES, 1.0)) == 2


def test_a_camera_turning_on_its_mount_does_not_move_what_it_sees() -> None:
    """A head-worn camera glances 30 degrees right and back while the car stands. A parked car seen
    in the camera's frame swings across the view; in the car's frame it has to stay put."""

    def mount(frame: int) -> float:
        return 0.5 * min(max((frame - 30) / 15, 0.0), 1.0) * min(max((90 - frame) / 15, 0.0), 1.0)

    def seen(frame: int) -> tuple[float, float]:
        angle = mount(frame)
        x, y = 6.0, 20.0
        return (
            math.cos(angle) * x - math.sin(angle) * y,
            math.sin(angle) * x + math.cos(angle) * y,
        )

    steps = [EgoStep(mount_rad=mount(frame)) for frame in range(FRAMES)]
    turned = build_tracks(track(1, range(0, 120), seen), TIMES, SEGMENTS, steps)
    ahead = build_tracks(track(1, range(0, 120), still(6.0, 20.0)), TIMES, SEGMENTS, no_ego())
    assert len(turned) == len(ahead) == 1
    for sample, expected in zip(turned[0].samples, ahead[0].samples, strict=True):
        assert sample.x == pytest.approx(expected.x, abs=0.3)
        assert sample.y == pytest.approx(expected.y, abs=0.6)
