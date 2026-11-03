"""Detection pass: video in, RawClip out. Results are cached so re-calibrating is instant."""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from clayview.core.cuts import CutDetector
from clayview.core.scene import RawClip, VideoInfo
from clayview.core.types import EMPTY_FLOW, Box, Detection, FlowPair
from clayview.detect import YoloTracker, pick_device
from clayview.video import BackgroundFlow, cut_thumbnail, probe, read_frames
from clayview.world import SegMaps

CACHE_VERSION = 3  # 2: wiper-proof cuts. 3: keyed on video content, not path and mtime


@dataclass(frozen=True)
class DetectionOptions:
    weights: str = "models/yolo11n.pt"
    imgsz: int = 960
    device: str = "auto"
    stride: int = 1
    limit_s: float | None = None


def extract(
    video_path: Path,
    options: DetectionOptions,
    progress: Callable[[str], None] = print,
) -> RawClip:
    """Run YOLO, ByteTrack, cut detection and background flow over the whole clip."""
    info = probe(video_path)
    device = pick_device(options.device)
    weights = _resolve_weights(options.weights)
    tracker = YoloTracker(weights, device, options.imgsz, info.fps / options.stride)
    cut_detector = CutDetector(info.fps / options.stride)
    flow = BackgroundFlow(info.width, info.height)

    times: list[float] = []
    cuts: list[int] = []
    detections: list[Detection] = []
    pairs: list[FlowPair] = []
    previous_boxes: list[Box] = []
    started = time.monotonic()
    frames_wanted = info.frame_count
    if options.limit_s is not None:
        frames_wanted = min(frames_wanted, round(options.limit_s * info.fps))
    total = max(frames_wanted // options.stride, 1)
    report_every = max(total // 20, 1)

    progress(f"detecting on {device} at imgsz {options.imgsz}, about {total} frames")
    for index, (time_s, frame) in enumerate(read_frames(video_path, options.stride)):
        if options.limit_s is not None and time_s > options.limit_s:
            break
        pair = flow.step(frame, previous_boxes)
        if index > 0:
            pairs.append(pair)  # pairs[i] links frame i to i + 1
        cut = cut_detector.update(cut_thumbnail(frame))
        if cut is not None:
            _add_cut(cut, cuts, pairs, times, progress)

        frame_detections = tracker.step(index, frame)
        detections.extend(frame_detections)
        previous_boxes = [d.box for d in frame_detections]
        times.append(time_s)

        if (index + 1) % report_every == 0:
            rate = (index + 1) / max(time.monotonic() - started, 1e-3)
            progress(f"  {index + 1}/{total} frames, {rate:.1f} fps")

    if not times:
        raise ValueError(f"no frames could be read from {video_path}")
    last_cut = cut_detector.finish()
    if last_cut is not None:
        _add_cut(last_cut, cuts, pairs, times, progress)

    duration = info.frame_count / info.fps if info.frame_count else times[-1]
    return RawClip(
        video=VideoInfo(
            path=str(video_path),
            width=info.width,
            height=info.height,
            fps=info.fps / options.stride,
            duration=min(duration, options.limit_s) if options.limit_s else duration,
            frame_count=info.frame_count,
        ),
        times=times,
        cuts=cuts,
        detections=detections,
        flow=pairs,
        model={
            "weights": Path(weights).name,
            "imgsz": options.imgsz,
            "device": device,
            "tracker": "bytetrack",
            "stride": options.stride,
        },
    )


def _add_cut(
    cut: int,
    cuts: list[int],
    pairs: list[FlowPair],
    times: list[float],
    progress: Callable[[str], None],
) -> None:
    """Cuts are confirmed a second late. Shots split tracks later, so only the flow needs fixing."""
    cuts.append(cut)
    pairs[cut - 1] = EMPTY_FLOW  # the first frame of a shot has no link back
    progress(f"  cut at {times[cut]:.2f} s")


def save_views(path: Path, views: Mapping[tuple[int, int], Sequence[float]], key: str) -> None:
    keys = np.array(list(views), dtype=np.int32).reshape(-1, 2)
    # A clip with no usable car crops has no views; reshape(0, -1) can't infer that width.
    width = len(next(iter(views.values()))) if views else 0
    values = np.array(list(views.values()), dtype=np.float32).reshape(len(keys), width)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, key=np.array(key), keys=keys, values=values)


def load_views(path: Path, key: str) -> dict[tuple[int, int], tuple[float, ...]] | None:
    """Load cached views, or None if they came from a different run."""
    if not path.exists():
        return None
    with np.load(path) as data:
        if str(data["key"]) != key:
            return None
        return {
            (int(frame), int(track)): tuple(float(value) for value in row)
            for (frame, track), row in zip(data["keys"], data["values"], strict=True)
        }


def save_seg(path: Path, seg: SegMaps, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, key=np.array(key), frames=seg.frames, maps=seg.maps)


def load_seg(path: Path, key: str) -> SegMaps | None:
    """Load cached street labels, or None if they came from a different run."""
    if not path.exists():
        return None
    with np.load(path) as data:
        if str(data["key"]) != key:
            return None
        return SegMaps(data["frames"], data["maps"])


def _resolve_weights(weights: str) -> str:
    """Keep downloaded weights in models/, not wherever the command ran."""
    path = Path(weights)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def cache_key(video_path: Path, options: DetectionOptions) -> str:
    """Same video bytes and options, same key, wherever the clip lives.

    Caches made on one machine are reused on another, where the clip has a different path and a
    fresh mtime. Hashing a 60 s clip takes tens of milliseconds once it's in the page cache.
    """
    with video_path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return json.dumps(
        {
            "version": CACHE_VERSION,
            "video": digest,
            "weights": Path(options.weights).name,
            "imgsz": options.imgsz,
            "stride": options.stride,
            "limit": options.limit_s,
        },
        sort_keys=True,
    )


def save_raw(path: Path, raw: RawClip, key: str) -> None:
    classes = sorted({d.class_name for d in raw.detections})
    class_index = {name: index for index, name in enumerate(classes)}
    offsets = np.cumsum([0, *[len(pair.prev) for pair in raw.flow]], dtype=np.int64)
    prev = [pair.prev for pair in raw.flow] or [EMPTY_FLOW.prev]
    next_ = [pair.next for pair in raw.flow] or [EMPTY_FLOW.next]
    header: dict[str, Any] = {
        "key": key,
        "classes": classes,
        "video": raw.video.__dict__,
        "model": raw.model,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            header=np.array(json.dumps(header)),
            times=np.array(raw.times, dtype=np.float64),
            cuts=np.array(raw.cuts, dtype=np.int64),
            det_frame=np.array([d.frame for d in raw.detections], dtype=np.int32),
            det_track=np.array([d.track_id for d in raw.detections], dtype=np.int32),
            det_class=np.array([class_index[d.class_name] for d in raw.detections], dtype=np.int16),
            det_conf=np.array([d.confidence for d in raw.detections], dtype=np.float32),
            det_box=np.array(
                [[d.box.x1, d.box.y1, d.box.x2, d.box.y2] for d in raw.detections],
                dtype=np.float32,
            ).reshape(-1, 4),
            flow_offsets=offsets,
            flow_prev=np.concatenate(prev),
            flow_next=np.concatenate(next_),
        )


def load_raw(path: Path, key: str, video_path: Path) -> RawClip | None:
    """Load cached detections, or None if they came from a different run.

    The key is the video's content, so the cache may have been made from another path or machine.
    The clip records where the video is now, not where it was.
    """
    if not path.exists():
        return None
    with np.load(path) as data:
        header = json.loads(str(data["header"]))
        if header["key"] != key:
            return None
        classes = header["classes"]
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
        offsets = data["flow_offsets"]
        prev, next_ = data["flow_prev"], data["flow_next"]
        pairs = [
            FlowPair(prev[start:end], next_[start:end])
            for start, end in itertools.pairwise(offsets)
        ]
        return RawClip(
            video=VideoInfo(**{**header["video"], "path": str(video_path)}),
            times=data["times"].tolist(),
            cuts=data["cuts"].tolist(),
            detections=detections,
            flow=pairs,
            model=header["model"],
        )
