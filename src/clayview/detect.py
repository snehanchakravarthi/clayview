"""YOLO plus ByteTrack. The only module that knows about Ultralytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from clayview.core.projection import CLASS_PRIORS
from clayview.core.types import Box, Detection

TRACKER_CONFIG = Path(__file__).with_name("bytetrack.yaml")


def pick_device(requested: str = "auto") -> str:
    """mps on Apple Silicon, cuda if there is one, otherwise cpu."""
    if requested != "auto":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class YoloTracker:
    """Runs detection and ByteTrack one frame at a time, so shots can reset the tracker."""

    def __init__(self, weights: str, device: str, imgsz: int) -> None:
        from ultralytics import YOLO

        self.model: Any = YOLO(weights)
        self.device = device
        self.imgsz = imgsz
        names: dict[int, str] = self.model.names
        self.class_ids = [index for index, name in names.items() if name in CLASS_PRIORS]
        self.names = names

    def reset(self) -> None:
        """Forget all tracks. Called at a hard cut."""
        predictor = self.model.predictor
        for tracker in getattr(predictor, "trackers", []) if predictor else []:
            tracker.reset()

    def step(self, frame_index: int, frame: np.ndarray) -> list[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=str(TRACKER_CONFIG),
            classes=self.class_ids,
            imgsz=self.imgsz,
            device=self.device,
            conf=0.1,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []
        ids = boxes.id.int().tolist()
        classes = boxes.cls.int().tolist()
        scores = boxes.conf.tolist()
        corners = boxes.xyxy.tolist()
        return [
            Detection(frame_index, track_id, self.names[cls], score, Box(*corner))
            for track_id, cls, score, corner in zip(ids, classes, scores, corners, strict=True)
        ]
