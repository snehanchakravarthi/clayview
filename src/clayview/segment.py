"""Label every pixel of sampled frames with a street scene model.

The model is Mask2Former trained on Mapillary Vistas, chosen for its 65 street classes: it knows
curbs, lane markings, crosswalks, poles, signs and vegetation, which a COCO detector doesn't. Its
weights are for research and non-commercial use.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from clayview.video import read_frames
from clayview.world import SegMaps

MODEL_NAME = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
MODEL_REVISION = "4772b6bf101d91f2534c106dc524d906aeb3c68a"
CACHE_DIR = Path("models/hf")
MAP_WIDTH = 1024
SEGMENT_EVERY = 10  # the street changes slowly; label one frame in this many


def map_size(width: int, height: int) -> tuple[int, int]:
    """(width, height) of a label map, keeping the frame's shape on a 32 pixel grid."""
    map_height = max(round(MAP_WIDTH * height / width / 32) * 32, 32)
    return MAP_WIDTH, map_height


class StreetSegmenter:
    def __init__(self, device: str) -> None:
        os.environ.setdefault("HF_HOME", str(CACHE_DIR))
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self._torch = torch
        self._device = device
        self._processor: Any = AutoImageProcessor.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION
        )
        self._model: Any = (
            Mask2FormerForUniversalSegmentation.from_pretrained(
                MODEL_NAME, revision=MODEL_REVISION, use_safetensors=True
            )
            .to(device)
            .eval()
        )

    def label(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """Class ids for a BGR frame, as a uint8 map of the given size."""
        import cv2

        rgb = cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2RGB)
        size = {"height": height, "width": width}
        inputs = self._processor(images=rgb, return_tensors="pt", size=size).to(self._device)
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        labels = self._processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(height, width)]
        )[0]
        return labels.cpu().numpy().astype(np.uint8)


def segment_clip(
    video_path: Path,
    stride: int,
    frame_count: int,
    device: str,
    every: int = SEGMENT_EVERY,
    progress: Callable[[str], None] = print,
) -> SegMaps:
    """Label one processed frame in `every`, up to frame_count."""
    total = max(frame_count // every, 1)
    progress(f"labelling the street in about {total} frames on {device}")
    model = StreetSegmenter(device)
    indices: list[int] = []
    maps: list[np.ndarray] = []
    started = time.monotonic()
    report_every = max(total // 10, 1)
    for index, (_, frame) in enumerate(read_frames(video_path, stride)):
        if index >= frame_count:
            break
        if index % every != 0:
            continue
        height, width = frame.shape[:2]
        maps.append(model.label(frame, *map_size(width, height)))
        indices.append(index)
        if len(maps) % report_every == 0:
            rate = len(maps) / max(time.monotonic() - started, 1e-3)
            progress(f"  {len(maps)}/{total} frames, {rate:.1f} fps")
    return SegMaps(np.array(indices, dtype=np.int32), np.stack(maps))
