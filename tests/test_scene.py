import json

import numpy as np

from clayview.core.camera import Camera
from clayview.core.scene import CameraOptions, RawClip, VideoInfo, build_scene
from clayview.core.stats import summarize
from clayview.core.types import EMPTY_FLOW, Box, Detection, FlowPair

CAMERA = Camera(1920, 1080, height_m=1.2, vfov_deg=60.0, horizon_y=620.0)
VIDEO = VideoInfo("clip.mp4", 1920, 1080, 30.0, 5.0, 150)
TIMES = [index / 30.0 for index in range(150)]


def box_for(x: float, y: float) -> Box:
    u, v = CAMERA.image_point(x, y)
    half = CAMERA.focal_px * 1.8 / y / 2.0
    height = CAMERA.focal_px * 1.5 / y
    return Box(u - half, v - height, u + half, v)


def road_pair(seed: int) -> FlowPair:
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(-6.0, 6.0, 60), rng.uniform(5.0, 18.0, 60)
    prev = [CAMERA.image_point(a, b) for a, b in zip(x, y, strict=True)]
    next_ = [CAMERA.image_point(a, b - 0.3) for a, b in zip(x, y, strict=True)]
    return FlowPair(np.array(prev, np.float32), np.array(next_, np.float32))


def clip(cuts: list[int] | None = None) -> RawClip:
    detections = [
        Detection(frame, 5, "car", 0.8, box_for(-1.5, 25.0 - 0.1 * frame)) for frame in range(120)
    ]
    return RawClip(
        video=VIDEO,
        times=TIMES,
        cuts=cuts or [],
        detections=detections,
        flow=[road_pair(seed) for seed in range(149)],
        model={"weights": "test.pt"},
    )


def test_scene_shape_matches_the_viewer_contract() -> None:
    scene = build_scene(clip(), CameraOptions(height_m=1.2, horizon_y=620.0))
    assert scene["version"] == 1
    meta = scene["metadata"]
    assert meta["source"]["path"] == "clip.mp4"
    assert meta["source"]["fps"] == 30.0
    assert meta["source"]["duration"] == 5.0
    assert meta["camera"]["heightM"] == 1.2
    assert meta["camera"]["horizonSource"] == "flag"
    assert len(scene["frames"]) == 150

    item = scene["frames"][60]["objects"][0]
    assert {"id", "class", "x", "y", "heading", "confidence", "flags"} <= set(item)
    assert item["class"] == "car"
    assert isinstance(item["flags"], list)
    assert len(item["box"]) == 4
    assert all(0.0 <= value <= 1.0 for value in item["box"])
    json.dumps(scene)


def test_positions_are_metres_in_the_ego_frame() -> None:
    scene = build_scene(clip(), CameraOptions(height_m=1.2, horizon_y=620.0))
    item = scene["frames"][50]["objects"][0]
    assert abs(item["x"] + 1.6) < 0.6
    assert abs(item["y"] - (20.0 + 2.25)) < 0.8


def test_ego_speed_comes_from_road_flow() -> None:
    scene = build_scene(clip(), CameraOptions(height_m=1.2, horizon_y=620.0))
    assert abs(scene["frames"][75]["ego"]["speed"] - 9.0) < 1.0
    assert scene["metadata"]["egoMotion"]["mode"] == "flow"


def test_ego_speed_flag_overrides_flow() -> None:
    scene = build_scene(clip(), CameraOptions(ego_speed_mps=20.0))
    assert abs(scene["frames"][75]["ego"]["speed"] - 20.0) < 0.01
    assert scene["metadata"]["egoMotion"]["mode"] == "fixed"


def test_each_shot_gets_its_own_segment_and_camera() -> None:
    raw = clip(cuts=[80])
    flow = list(raw.flow)
    flow[79] = EMPTY_FLOW
    scene = build_scene(
        RawClip(raw.video, raw.times, raw.cuts, raw.detections, flow, raw.model),
        CameraOptions(height_m=1.2, horizon_y=620.0),
    )
    assert [s["startFrame"] for s in scene["segments"]] == [0, 80]
    assert scene["frames"][79]["segment"] == 0
    assert scene["frames"][80]["segment"] == 1
    ids = {o["id"] for f in scene["frames"][:80] for o in f["objects"]}
    later = {o["id"] for f in scene["frames"][80:] for o in f["objects"]}
    assert ids.isdisjoint(later)


def test_unknown_camera_falls_back_to_defaults_and_says_so() -> None:
    raw = clip()
    blind = RawClip(raw.video, raw.times, [], [], [EMPTY_FLOW] * 149, raw.model)
    camera = build_scene(blind)["metadata"]["camera"]
    assert camera["horizonSource"] == "default"
    assert camera["heightSource"] == "default"
    assert camera["horizonFraction"] == 0.5


def test_summary_runs_on_empty_and_full_scenes() -> None:
    raw = clip()
    blind = RawClip(raw.video, raw.times, [], [], [EMPTY_FLOW] * 149, raw.model)
    assert "objects     none" in summarize(build_scene(blind))
    assert "tracks      1: 1 car" in summarize(build_scene(clip()))
