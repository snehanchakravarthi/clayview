"""Failures found on real eval clips, replayed from their cached flow.

The caches come from `tools/eval/eval.py pull` and aren't in the repo, so these skip without them.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from clayview.core.ego import estimate_ego_series
from clayview.core.projection import box_pad_px
from clayview.core.scene import CameraOptions, VideoInfo, calibrate
from clayview.core.types import Box, Detection, FlowPair

EVAL_OUT = Path(__file__).resolve().parents[1] / "out/eval"


def cached_flow(name: str) -> tuple[VideoInfo, list[float], list[FlowPair]]:
    path = EVAL_OUT / f"{name}.raw.npz"
    if not path.exists():
        pytest.skip(f"no cache for {name}; run tools/eval/eval.py pull")
    with np.load(path) as data:
        header = json.loads(str(data["header"]))
        prev, next_ = data["flow_prev"], data["flow_next"]
        pairs = [
            FlowPair(prev[start:end], next_[start:end])
            for start, end in itertools.pairwise(data["flow_offsets"])
        ]
        times = data["times"].tolist()
    return VideoInfo(**header["video"]), times, pairs


def cached_calibration(name: str) -> tuple[float, float, float, str]:
    """(horizon row, centre column, height, height source) for a clip's first shot, with cars."""
    video, _, pairs = cached_flow(name)
    with np.load(EVAL_OUT / f"{name}.raw.npz") as data:
        header = json.loads(str(data["header"]))
        classes = header["classes"]
        cuts = data["cuts"].tolist()
        detections = [
            Detection(int(frame), int(track), classes[int(cls)], float(conf), Box(*map(float, box)))
            for frame, track, cls, conf, box in zip(
                data["det_frame"],
                data["det_track"],
                data["det_class"],
                data["det_conf"],
                data["det_box"],
                strict=True,
            )
        ]
    end = min(cuts) if cuts else len(pairs) + 1
    shot = [d for d in detections if d.frame < end]
    pad = box_pad_px(video.width, video.height, header["model"].get("imgsz"))
    calibration = calibrate(pairs[: end - 1], shot, video, CameraOptions(), pad)
    camera = calibration.camera
    return camera.horizon, camera.cx, camera.height_m, calibration.height_source


def test_wide_lens_tilted_up_still_finds_the_road() -> None:
    """christchurch_left: the lane lines vanish at row ~600 of 720, but the pinhole FoE fit put the
    horizon at 653, under the last visible road row. Ego speed had no road and read 0 for a minute
    of highway driving."""
    video, times, pairs = cached_flow("christchurch_left")
    camera = calibrate(pairs, [], video, CameraOptions()).camera
    assert 595.0 <= camera.horizon <= 630.0
    steps, coverage = estimate_ego_series(pairs, times, camera)
    assert coverage > 0.9
    speeds = np.array([step.forward_m for step in steps]) / np.diff(times)
    assert float(np.mean(speeds * 3.6 >= 5.0)) > 0.9


def test_suspension_cables_do_not_lift_the_horizon() -> None:
    """sf_bay_bridge: lane paint vanishes at (589, 406) and the tower dead ahead stands at x 583,
    but tracks on the cables dragged the joint FoE fit to (654, 384). Cars read a 2.11 m camera and
    a straight bridge drifted 56 degrees a minute."""
    horizon, center, height, _ = cached_calibration("sf_bay_bridge")
    assert 400.0 <= horizon <= 412.0
    assert 570.0 <= center <= 600.0
    assert 1.05 <= height <= 1.45


def test_rain_and_branches_do_not_lift_the_horizon() -> None:
    """wintry_wipers: lane paint vanishes at (554, 253); the joint FoE fit was at (628, 231)."""
    horizon, center, _, _ = cached_calibration("wintry_wipers")
    assert 240.0 <= horizon <= 262.0
    assert 540.0 <= center <= 580.0


def test_cars_hugging_the_horizon_do_not_set_the_height() -> None:
    """christchurch_left: with the horizon on the lane paint, its few usable cars sit 16 px under
    it and read a 0.84 m camera."""
    horizon, _, _, source = cached_calibration("christchurch_left")
    assert 595.0 <= horizon <= 615.0
    assert source == "default"


def test_comma_horizon_and_height_hold() -> None:
    """comma2k19's speed and radar agree with its calibration; the FoE choice mustn't move it."""
    horizon, _, height, _ = cached_calibration("comma2k19")
    assert horizon == pytest.approx(378.3, abs=1.0)
    assert height == pytest.approx(1.43, abs=0.03)
