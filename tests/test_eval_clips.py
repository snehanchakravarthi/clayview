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
from clayview.core.scene import CameraOptions, VideoInfo, calibrate, split_mount
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


def cached_cuts(name: str) -> list[int]:
    path = EVAL_OUT / f"{name}.raw.npz"
    if not path.exists():
        pytest.skip(f"no cache for {name}; run tools/eval/eval.py pull")
    with np.load(path) as data:
        return data["cuts"].tolist()


def moving_kmh(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Replayed ego speed in km/h and the time of each pair, one camera per shot."""
    video, times, pairs = cached_flow(name)
    bounds = [0, *sorted(cached_cuts(name)), len(times)]
    speeds = np.zeros(len(pairs))
    for start, end in itertools.pairwise(bounds):
        shot = pairs[start : end - 1]
        camera = calibrate(shot, [], video, CameraOptions()).camera
        steps, _ = estimate_ego_series(shot, times[start:end], camera)
        speeds[start : start + len(steps)] = [step.forward_m for step in steps]
    return speeds / np.diff(times) * 3.6, np.array(times[:-1])


@pytest.mark.parametrize("name", ["wintry_wipers", "doha_rain_360p"])
def test_rain_on_the_glass_is_not_a_stop(name: str) -> None:
    """Both clips move throughout. Raindrops over the road band, wet-road reflections and the smear
    between them read 0 to 4 km/h for seconds at a time, while the trees above the horizon kept
    expanding away from the FoE: wintry_wipers read 67% moving and doha_rain_360p 81%."""
    kmh, _ = moving_kmh(name)
    assert float(np.mean(kmh >= 5.0)) > 0.9


@pytest.mark.parametrize(
    ("name", "start", "end"),
    [
        ("krakow_city", 40.0, 46.0),
        ("park_ridge_suburb", 0.0, 5.0),
        ("ljubljana_fog_night", 2.0, 10.0),
    ],
)
def test_a_real_stop_still_reads_stopped(name: str, start: float, end: float) -> None:
    """Ego waits at a light or in traffic here. Nothing above the road expands, so the road's zero
    stands."""
    kmh, times = moving_kmh(name)
    during = (times >= start) & (times < end)
    assert float(np.mean(kmh[during] >= 5.0)) < 0.1


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
