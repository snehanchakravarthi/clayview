import math

from clayview.core.camera import Camera
from clayview.core.heading import (
    ANGLES,
    EgoPath,
    HeadingConfig,
    HeadingCues,
    seen_aspect,
    shape_scores,
    solve_heading,
    view_scores,
)
from clayview.core.tracks import Segment, build_tracks
from clayview.core.types import Box, Detection, EgoStep

CAMERA = Camera(1920, 1080, height_m=1.2, vfov_deg=60.0, horizon_y=620.0)
DT = 1.0 / 30.0
FRAMES = 240
TIMES = [index * DT for index in range(FRAMES)]
SEGMENTS = [Segment(0, 0, FRAMES, CAMERA)]
SEDAN = (1.4, 2.7)
CONFIG = HeadingConfig()


def angle_error(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def seen_box(x: float, y: float, heading: float, shape: tuple[float, float] = SEDAN) -> Box:
    """The box a car at (x, y) makes, for a car pointing along `heading` in ego's frame."""
    aspect = float(seen_aspect(heading - math.atan2(x, y), *shape))
    u, v = CAMERA.image_point(x, y)
    height = CAMERA.focal_px * 1.45 / y
    return Box(u - aspect * height / 2.0, v - height, u + aspect * height / 2.0, v)


def cue(box: Box | None, bearing: float, **extra: object) -> HeadingCues:
    values: dict[str, object] = {
        "yaw": 0.0,
        "bearing": bearing,
        "box": box,
        "motion": None,
        "motion_certainty": 0.0,
        "road": None,
        "road_offset_m": 0.0,
    }
    return HeadingCues(**{**values, **extra})  # type: ignore[arg-type]


def test_shape_finds_a_shallow_angle_up_to_its_mirror_images() -> None:
    truth = math.radians(15.0)
    scores = shape_scores(seen_box(0.0, 20.0, truth), 0.0, 0.0, "car", CONFIG)
    best = float(ANGLES[int(scores.argmax())])
    assert min(abs(abs(best) - truth), abs(math.pi - abs(best) - truth)) < math.radians(14.0)


def test_rear_view_and_side_view_are_told_apart() -> None:
    rear = shape_scores(seen_box(0.0, 20.0, 0.0), 0.0, 0.0, "car", CONFIG)
    side = shape_scores(seen_box(0.0, 20.0, math.pi / 2), 0.0, 0.0, "car", CONFIG)
    straight = len(ANGLES) // 2
    across = straight + len(ANGLES) // 4
    assert rear[straight] > rear[across] + 2.0
    assert side[across] > side[straight] + 2.0


def test_a_person_has_no_shape_cue() -> None:
    assert not shape_scores(Box(0, 0, 40, 120), 0.0, 0.0, "person", CONFIG).any()


def test_motion_settles_front_from_back() -> None:
    truth = math.radians(150.0)
    box = seen_box(0.0, 20.0, truth)
    cues = [cue(box, 0.0, motion=truth, motion_certainty=10.0) for _ in range(60)]
    solved = solve_heading(cues, [DT] * 59, "car", True)
    assert angle_error(solved[-1], truth) < math.radians(8.0)


def test_one_odd_frame_cant_flip_a_track() -> None:
    cues = [cue(None, 0.0, motion=0.0, motion_certainty=10.0) for _ in range(60)]
    cues[30] = cue(None, 0.0, motion=math.pi, motion_certainty=10.0)
    solved = solve_heading(cues, [DT] * 59, "car", True)
    assert all(angle_error(value, 0.0) < math.radians(4.0) for value in solved)


def test_a_track_can_follow_a_real_turn() -> None:
    directions = [math.radians(90.0) * index / 59 for index in range(60)]
    cues = [cue(None, 0.0, motion=d, motion_certainty=10.0) for d in directions]
    solved = solve_heading(cues, [DT] * 59, "car", True)
    assert angle_error(solved[-1], directions[-1]) < math.radians(15.0)


def test_with_no_evidence_it_faces_the_road() -> None:
    solved = solve_heading([cue(None, 0.0, road=0.5) for _ in range(30)], [DT] * 29, "car", False)
    assert len(set(solved)) == 1
    assert angle_error(solved[0], 0.5) < math.radians(3.0)


def one_view(bin_index: int) -> list[float]:
    return [0.93 if index == bin_index else 0.01 for index in range(8)]


def test_view_bins_mean_rear_front_and_sides() -> None:
    expected = {
        4: 0.0,
        0: math.pi,
        5: -math.pi / 4,
        3: math.pi / 4,
        6: -math.pi / 2,
        2: math.pi / 2,
    }
    for bin_index, seen_from in expected.items():
        scores = view_scores(one_view(bin_index), 0.0, 0.0, 0.03)
        assert angle_error(float(ANGLES[int(scores.argmax())]), seen_from) < math.radians(4.0)


def test_view_follows_the_ray_to_the_object() -> None:
    bearing = math.radians(30.0)
    scores = view_scores(one_view(4), bearing, 0.0, 0.03)
    assert angle_error(float(ANGLES[int(scores.argmax())]), bearing) < math.radians(4.0)


def test_views_tell_a_parked_cars_front_from_its_back() -> None:
    cues = [cue(seen_box(0.0, 20.0, math.pi), 0.0, view=one_view(0), road=0.0) for _ in range(20)]
    solved = solve_heading(cues, [DT] * 19, "car", False)
    assert angle_error(solved[0], math.pi) < math.radians(8.0)


def test_one_wrong_view_cant_turn_a_car_around() -> None:
    cues = [cue(None, 0.0, view=one_view(4), road=0.0) for _ in range(20)]
    cues[3] = cue(None, 0.0, view=one_view(0), road=0.0)
    solved = solve_heading(cues, [DT] * 19, "car", False)
    assert angle_error(solved[0], 0.0) < math.radians(8.0)


def test_ego_path_finds_the_road_ahead_on_a_bend() -> None:
    steps = [EgoStep(forward_m=0.5, yaw_rad=0.01) for _ in range(199)]
    path = EgoPath(steps, 0, 200)
    x, y = path.x[150], path.y[150]
    road, offset = path.road_at(0, float(x), float(y))
    assert abs(road - path.yaw[150]) < 0.02
    assert offset < 0.1


def parked_car(world_heading: float) -> list[Detection]:
    """A parked car ahead on the right, with ego driving straight past it."""
    return [
        Detection(frame, 1, "car", 0.8, seen_box(6.0, 34.0 - frame * 0.1, world_heading))
        for frame in range(FRAMES)
    ]


def worst_error(detections: list[Detection], truth: float) -> float:
    """Front and back look alike without motion, so either end of the car counts."""
    steps = [EgoStep(forward_m=0.1) for _ in range(FRAMES - 1)]
    tracks = build_tracks(detections, TIMES, SEGMENTS, steps)
    assert len(tracks) == 1
    return max(
        min(angle_error(s.heading, truth), angle_error(s.heading, truth + math.pi))
        for s in tracks[0].samples
    )


def test_a_car_parked_along_the_road_stays_along_it() -> None:
    assert worst_error(parked_car(0.0), 0.0) < math.radians(6.0)


def test_a_car_parked_across_the_road_stays_across_it() -> None:
    assert worst_error(parked_car(math.pi / 2), math.pi / 2) < math.radians(10.0)
