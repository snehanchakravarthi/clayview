"""Video reading, shot-cut input and sparse background flow. Uses OpenCV, not YOLO."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from clayview.core.types import EMPTY_FLOW, Box, FlowPair

FLOW_WIDTH = 640
CUT_SIZE = (96, 54)
GRID_ROWS = 9
GRID_COLUMNS = 14
CORNERS_PER_CELL = 4


@dataclass(frozen=True)
class Probe:
    width: int
    height: int
    fps: float
    frame_count: int


def probe(path: Path) -> Probe:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    info = Probe(
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(capture.get(cv2.CAP_PROP_FPS)) or 30.0,
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    capture.release()
    return info


def read_frames(path: Path, stride: int = 1) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (time_s, bgr_frame) for every stride-th frame, on the container's clock."""
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    index = 0
    last_time = -1.0
    try:
        while True:
            if not capture.grab():
                return
            if index % stride == 0:
                ok, frame = capture.retrieve()
                if not ok:
                    return
                position = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                time_s = position if position > last_time else index / fps
                last_time = time_s
                yield time_s, frame
            index += 1
    finally:
        capture.release()


def cut_thumbnail(frame: np.ndarray) -> np.ndarray:
    """Tiny grayscale frame for the cut detector."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, CUT_SIZE, interpolation=cv2.INTER_AREA)


class BackgroundFlow:
    """Lucas-Kanade tracks on what isn't a detected object, with a forward-backward check.

    Features are picked per grid cell. A single global pass gives every slot to trees and buildings
    and leaves almost nothing on the road, which is what ego speed depends on.
    """

    def __init__(self, width: int, height: int) -> None:
        self._scale = FLOW_WIDTH / width
        self._size = (FLOW_WIDTH, round(height * self._scale))
        self._previous: np.ndarray | None = None
        self._lk = {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        }

    def reset(self) -> None:
        self._previous = None

    def step(self, frame: np.ndarray, previous_boxes: Sequence[Box]) -> FlowPair:
        """Match the last frame to this one. previous_boxes are the last frame's detections."""
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), self._size, cv2.INTER_AREA)
        previous, self._previous = self._previous, gray
        if previous is None:
            return EMPTY_FLOW

        points = self._features(previous, previous_boxes)
        if len(points) == 0:
            return EMPTY_FLOW
        forward, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None, **self._lk)
        back, status_back, _ = cv2.calcOpticalFlowPyrLK(gray, previous, forward, None, **self._lk)
        error = np.abs(points - back).reshape(-1, 2).max(axis=1)
        good = status.ravel().astype(bool) & status_back.ravel().astype(bool) & (error < 0.8)
        if not good.any():
            return EMPTY_FLOW
        scale = 1.0 / self._scale
        return FlowPair(
            (points[good].reshape(-1, 2) * scale).astype(np.float32),
            (forward[good].reshape(-1, 2) * scale).astype(np.float32),
        )

    def _features(self, gray: np.ndarray, boxes: Sequence[Box]) -> np.ndarray:
        height, width = gray.shape
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        for box in boxes:
            pad = 4
            x1 = max(int(box.x1 * self._scale) - pad, 0)
            y1 = max(int(box.y1 * self._scale) - pad, 0)
            x2 = min(int(box.x2 * self._scale) + pad, width)
            y2 = min(int(box.y2 * self._scale) + pad, height)
            mask[y1:y2, x1:x2] = 0

        found: list[np.ndarray] = []
        for row in range(GRID_ROWS):
            top, bottom = height * row // GRID_ROWS, height * (row + 1) // GRID_ROWS
            for column in range(GRID_COLUMNS):
                left, right = width * column // GRID_COLUMNS, width * (column + 1) // GRID_COLUMNS
                corners = cv2.goodFeaturesToTrack(
                    gray[top:bottom, left:right],
                    CORNERS_PER_CELL,
                    0.02,
                    6,
                    mask=mask[top:bottom, left:right],
                    blockSize=7,
                )
                if corners is not None:
                    found.append(corners + np.array([left, top], np.float32))
        if not found:
            return np.empty((0, 1, 2), np.float32)
        return np.concatenate(found)
