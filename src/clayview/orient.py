"""Read the angle a car is seen from out of its pixels.

The model is the ConvNeXt from the Car Full View paper (Catruna et al., 2023), Apache-2.0, trained
on 23K car photos. Its eight view classes feed the heading solver as one more cue.
"""

from __future__ import annotations

import hashlib
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from clayview.core.heading import VIEW_BINS
from clayview.core.tracks import Views
from clayview.core.types import Box, Detection
from clayview.video import read_frames

WEIGHTS_URL = (
    "https://huggingface.co/fort-cyber/Car-orientation-image/resolve/main/convnext-multi-weak.pth"
)
WEIGHTS_SHA256 = "26bd9e4eff8a4cfa3a264223ecced760f2fb3d1f29dc68f4e23b1b27babcd91d"
WEIGHTS_PATH = Path("models/cfv-convnext-small.pth")
MODEL_NAME = "cfv-convnext-small"

INPUT_SIZE = 224
CROP_MARGIN = 0.1
MIN_CROP_HEIGHT_PX = 30.0
VIEW_INTERVAL_S = 1 / 6  # a car's angle changes slowly
BATCH_SIZE = 32
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def fetch_weights(path: Path = WEIGHTS_PATH) -> Path:
    """Download the weights once and refuse anything but the exact file we tested."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        urllib.request.urlretrieve(WEIGHTS_URL, partial)
        partial.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        path.unlink()
        raise ValueError(f"{path} doesn't match the expected checksum; deleted it")
    return path


def sample_every(fps: float) -> int:
    """Processed frames between car views, so a 60 fps clip doesn't count each car twice."""
    return max(round(VIEW_INTERVAL_S * fps), 1)


def crop_bounds(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    """Box plus a margin, clamped to the frame, as integer x1, y1, x2, y2."""
    margin = CROP_MARGIN * max(box.width, box.height)
    return (
        int(max(box.x1 - margin, 0.0)),
        int(max(box.y1 - margin, 0.0)),
        int(min(box.x2 + margin, width)),
        int(min(box.y2 + margin, height)),
    )


def wanted_detections(detections: Sequence[Detection], every: int) -> dict[int, list[Detection]]:
    """Which detections to look at, by frame: cars big enough to read, one frame in `every`."""
    by_frame: dict[int, list[Detection]] = defaultdict(list)
    for detection in detections:
        if detection.class_name != "car" or detection.frame % every != 0:
            continue
        if detection.box.height >= MIN_CROP_HEIGHT_PX:
            by_frame[detection.frame].append(detection)
    return by_frame


class ViewModel:
    def __init__(self, weights: Path, device: str) -> None:
        import timm
        import torch
        from torch import nn

        class Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = timm.create_model(
                    "convnext_small", pretrained=False, num_classes=1024
                )
                self.class_prediction = nn.Linear(1024, VIEW_BINS)
                # Part of the published weights. Only the class head feeds the heading solver.
                self.angle_prediction = nn.Linear(1024 + VIEW_BINS, 1)

            def forward(self, images: Any) -> Any:
                return self.class_prediction(self.backbone(images)).softmax(dim=1)

        network = Network()
        network.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self._network = network.to(device).eval()
        self._device = device
        self._torch = torch

    def predict(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Eight view probabilities per BGR crop."""
        import cv2

        size = (INPUT_SIZE, INPUT_SIZE)
        images = [
            (cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)[:, :, ::-1] / 255.0 - MEAN) / STD
            for crop in crops
        ]
        batch = self._torch.tensor(
            np.stack(images).transpose(0, 3, 1, 2), dtype=self._torch.float32
        )
        with self._torch.no_grad():
            return self._network(batch.to(self._device)).cpu().numpy()


def _batches(
    video_path: Path, wanted: dict[int, list[Detection]], stride: int
) -> Iterator[tuple[list[tuple[int, int]], list[np.ndarray]]]:
    keys: list[tuple[int, int]] = []
    crops: list[np.ndarray] = []
    last = max(wanted, default=-1)
    for index, (_, frame) in enumerate(read_frames(video_path, stride)):
        if index > last:
            break
        height, width = frame.shape[:2]
        for detection in wanted.get(index, ()):
            x1, y1, x2, y2 = crop_bounds(detection.box, width, height)
            keys.append((detection.frame, detection.track_id))
            crops.append(frame[y1:y2, x1:x2])
            if len(crops) == BATCH_SIZE:
                yield keys, crops
                keys, crops = [], []
    if crops:
        yield keys, crops


def collect_views(
    video_path: Path,
    detections: Sequence[Detection],
    stride: int,
    every: int,
    device: str,
    progress: Callable[[str], None] = print,
) -> Views:
    """Run the view model over the clip's car crops."""
    wanted = wanted_detections(detections, every)
    total = sum(len(items) for items in wanted.values())
    progress(f"reading car views from {total} crops on {device}")
    model = ViewModel(fetch_weights(), device)
    views: dict[tuple[int, int], tuple[float, ...]] = {}
    for keys, crops in _batches(video_path, wanted, stride):
        for key, probabilities in zip(keys, model.predict(crops), strict=True):
            views[key] = tuple(round(float(value), 4) for value in probabilities)
    return views
