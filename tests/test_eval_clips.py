"""Failures found on real eval clips, replayed from their cached flow.

The caches come from `tools/eval/eval.py pull` and aren't in the repo, so these skip without them.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from clayview.core.ego import estimate_ego_series
from clayview.core.scene import CameraOptions, VideoInfo, calibrate
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
