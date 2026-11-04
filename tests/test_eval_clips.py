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
