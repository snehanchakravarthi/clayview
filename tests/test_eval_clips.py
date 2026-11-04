"""Failures found on real eval clips, replayed from their cached flow.

The caches come from `tools/eval/eval.py pull` and aren't in the repo, so these skip without them.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from clayview.core.ego import estimate_ego_series
from clayview.core.scene import CameraOptions, VideoInfo, calibrate, split_mount
from clayview.core.types import FlowPair

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


def camera_and_car(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Times, the camera's yaw per step, the car's, and the mount angle per frame."""
    video, times, pairs = cached_flow(name)
    camera = calibrate(pairs, [], video, CameraOptions()).camera
    steps, _ = estimate_ego_series(pairs, times, camera)
    split = split_mount(steps, times)
    return (
        np.array(times),
        np.array([step.yaw_rad for step in steps]),
        np.array([step.yaw_rad for step in split[:-1]]),
        np.array([step.mount_rad for step in split]),
    )


@pytest.mark.parametrize("name", ["kranj_glare", "dashcam_pip"])
def test_a_fixed_camera_keeps_every_real_turn(name: str) -> None:
    """Town corners and a winding road, about 300 deg/min of real turning, all within what a car can
    do. The mount split must leave them exactly as measured."""
    _, camera, car, mount = camera_and_car(name)
    assert np.array_equal(camera, car)
    assert not mount.any()


def test_a_dashcam_turned_to_the_side_window_is_the_mount() -> None:
    """park_ridge_suburb, 33 to 38 s: someone turns the dashcam to look out of the right side
    window, A-pillar in view, and puts it back. The camera swings 70 degrees, the car hardly at
    all."""
    times, camera, car, mount = camera_and_car("park_ridge_suburb")
    window = np.flatnonzero((times[:-1] >= 33.0) & (times[:-1] <= 38.0))
    camera_heading = np.degrees(np.cumsum(camera))[window]
    car_heading = np.degrees(np.cumsum(car))[window]
    assert np.ptp(camera_heading) > 50.0
    assert np.ptp(car_heading) < 20.0
    assert np.degrees(mount[window].max()) > 45.0


def test_a_helmet_looking_round_is_not_the_bike_turning() -> None:
    """leh_bike, the last 2.5 s: the rider looks round and the helmet camera turns 100 degrees in a
    second, at a speed where a motorbike can't. That's the mount; the bike holds its line."""
    times, camera, car, mount = camera_and_car("leh_bike")
    tail = np.flatnonzero(times[:-1] >= 57.5)
    assert np.degrees(camera[tail].sum()) > 80.0
    assert abs(np.degrees(car[tail].sum())) < 20.0
    assert np.degrees(mount[-1]) > 80.0
