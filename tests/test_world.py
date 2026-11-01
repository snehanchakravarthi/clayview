import math

import numpy as np

from clayview.core.camera import Camera
from clayview.core.heading import EgoPath
from clayview.core.tracks import Segment
from clayview.core.types import EgoStep
from clayview.world import (
    BUILDING,
    CURB,
    G_LANE_PAINT,
    G_ROAD,
    GROUP_OF,
    LANE_PAINT,
    POLE,
    ROAD,
    SIDEWALK,
    TRAFFIC_LIGHT,
    VEGETATION,
    SegMaps,
    WorldConfig,
    build_segment,
    collect_ground,
    extract_paint,
    merge_uprights,
    pick_trees,
    resample_path,
    road_edges,
    to_ego,
    to_world,
    uprights_in_frame,
)

CAMERA = Camera(1024, 576, height_m=1.3, vfov_deg=60.0, horizon_y=250.0, visible_bottom=540.0)
CFG = WorldConfig()
FRAMES = 300
STEPS = [EgoStep(forward_m=0.3) for _ in range(FRAMES - 1)]
PATH = EgoPath(STEPS, 0, FRAMES)
EVERY = 10


def street(left: float = -4.0, right: float = 5.0, line: float | None = 0.5) -> np.ndarray:
    """A label map of a straight road with sidewalks, seen from the middle."""
    height, width = 576, 1024
    label = np.full((height, width), BUILDING, dtype=np.uint8)
    rows, columns = np.mgrid[0:height, 0:width]
    below = rows > CAMERA.horizon + 1
    distance = np.where(
        below, CAMERA.focal_px * CAMERA.height_m / (rows - CAMERA.horizon + 1e-6), 1e6
    )
    x = (columns - CAMERA.cx) * distance / CAMERA.focal_px
    label[below & (distance < 60.0)] = SIDEWALK
    label[below & (distance < 60.0) & (x > left - 0.2) & (x < right + 0.2)] = CURB
    label[below & (distance < 60.0) & (x > left) & (x < right)] = ROAD
    if line is not None:
        label[below & (distance < 60.0) & (np.abs(x - line) < 0.12)] = LANE_PAINT
    return label


def maps(label: np.ndarray) -> SegMaps:
    frames = np.arange(0, FRAMES, EVERY, dtype=np.int32)
    return SegMaps(frames, np.stack([label] * len(frames)))


def ground_votes(label: np.ndarray) -> tuple[np.ndarray, object]:
    path = resample_path(PATH, CFG.row_m)
    seg = maps(label)
    return collect_ground(path, PATH, CAMERA, list(seg.frames), seg.maps, CFG), path


def test_world_and_ego_frames_are_inverses() -> None:
    world = to_world(np.array(2.0), np.array(7.0), 10.0, -3.0, 0.4)
    x, y = to_ego(world[0], world[1], 10.0, -3.0, 0.4)
    assert math.isclose(float(x), 2.0, abs_tol=1e-9)
    assert math.isclose(float(y), 7.0, abs_tol=1e-9)


def test_labels_map_to_ground_groups() -> None:
    assert GROUP_OF[ROAD] == G_ROAD
    assert GROUP_OF[LANE_PAINT] == G_LANE_PAINT


def test_road_edges_are_found_where_the_road_ends() -> None:
    votes, _ = ground_votes(street())
    left, right = road_edges(votes, CFG)
    middle = len(left) // 2
    assert abs(left[middle] + 4.0) < 0.5
    assert abs(right[middle] - 5.0) < 0.5


def test_a_lane_line_comes_out_as_a_long_thin_polygon() -> None:
    votes, path = ground_votes(street())
    polygons = extract_paint(votes, G_LANE_PAINT, path, CFG)
    assert polygons
    longest = max(polygons, key=len)
    xs = [point[0] for point in longest]
    ys = [point[1] for point in longest]
    assert abs(float(np.median(xs)) - 0.5) < 0.4
    assert max(ys) - min(ys) > 20.0


def test_a_pole_is_placed_on_the_ground_where_it_stands() -> None:
    label = street(line=None)
    u, v = CAMERA.image_point(6.0, 15.0)
    label[120 : int(v), int(u) - 2 : int(u) + 2] = POLE
    label[int(v) : int(v) + 8, int(u) - 2 : int(u) + 2] = SIDEWALK
    found = uprights_in_frame(label, CAMERA, CFG)
    assert len(found) == 1
    kind, x, y, height = found[0]
    assert kind == "pole"
    assert abs(x - 6.0) < 0.5
    assert abs(y - 15.0) < 1.0
    assert height > 3.0


def test_a_traffic_light_names_its_pole() -> None:
    label = street(line=None)
    u, v = CAMERA.image_point(6.0, 15.0)
    label[120 : int(v), int(u) - 2 : int(u) + 2] = POLE
    label[110:135, int(u) - 5 : int(u) + 5] = TRAFFIC_LIGHT
    assert uprights_in_frame(label, CAMERA, CFG)[0][0] == "trafficLight"


def test_one_sighting_is_not_an_object() -> None:
    seen = [("pole", 1.0, 1.0, 5.0)] * 3 + [("pole", 30.0, 1.0, 5.0)]
    objects = merge_uprights(seen, CFG)
    assert len(objects) == 1
    assert objects[0]["x"] == 1.0


def test_trees_are_spaced_by_their_size() -> None:
    rng = np.random.default_rng(0)
    points = np.column_stack(
        (rng.normal(10.0, 0.3, 80), rng.normal(5.0, 0.3, 80), np.full(80, 10.0))
    )
    trees = pick_trees(points, CFG)
    assert [tree["kind"] for tree in trees] == ["tree"]


def test_vegetation_needs_ground_under_it_and_a_whole_segment_builds() -> None:
    label = street()
    u, v = CAMERA.image_point(-8.0, 12.0)
    label[60 : int(v), int(u) - 30 : int(u) + 30] = VEGETATION
    world = build_segment(Segment(0, 0, FRAMES, CAMERA), STEPS, maps(label), CFG)
    assert any(item["kind"] == "line" for item in world["markings"])
    assert world["road"][20]["measured"]
    assert abs(world["road"][20]["left"] + 4.0) < 0.7
    assert world["road"][20]["leftWalk"] > 2.0
    assert world["road"][20]["rightCurb"]
