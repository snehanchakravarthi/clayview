"""Assemble the scene file from raw detector output. This module never touches a video or model."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from clayview.core.camera import (
    DEFAULT_HEIGHT_M,
    DEFAULT_VFOV_DEG,
    Camera,
)
from clayview.core.ego import estimate_ego_series, yaw_step
from clayview.core.heading import EgoPath
from clayview.core.horizon import (
    estimate_ground_bottom,
    estimate_visible_bottom,
    fit_focus_of_expansion,
    fit_vehicle_horizon,
    vehicle_height_for_horizon,
    vehicle_samples,
)
from clayview.core.projection import ProjectionConfig
from clayview.core.tracks import Segment, TrackConfig, Views, WorldTrack, build_tracks
from clayview.core.types import Detection, EgoStep, FlowPair

SCENE_VERSION = 2


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int


@dataclass(frozen=True)
class RawClip:
    """Everything the detection pass produces. `flow[i]` links processed frame i to i + 1."""

    video: VideoInfo
    times: Sequence[float]
    cuts: Sequence[int]  # first frame of each shot after the first
    detections: Sequence[Detection]
    flow: Sequence[FlowPair]
    model: dict[str, Any]
    views: Views = field(default_factory=dict)


@dataclass(frozen=True)
class CameraOptions:
    """What the user pinned. None means estimate it, per shot."""

    height_m: float | None = None
    vfov_deg: float = DEFAULT_VFOV_DEG
    horizon_y: float | None = None
    ego_speed_mps: float | None = None  # fixed speed; replaces the flow estimate
    use_ego_motion: bool = True


@dataclass(frozen=True)
class Calibration:
    camera: Camera
    horizon_source: str
    height_source: str


def calibrate(
    pairs: Sequence[FlowPair],
    detections: Sequence[Detection],
    video: VideoInfo,
    options: CameraOptions,
) -> Calibration:
    """Work out one shot's camera, preferring what the user pinned, then the clip, then defaults."""
    width, height = video.width, video.height
    foe = fit_focus_of_expansion(pairs, width, height)
    if foe is not None:
        # The first pass can't tell turning from travelling. Remove yaw and solve again.
        rough = Camera(width, height, vfov_deg=options.vfov_deg, horizon_y=foe.y)
        shifts = [_yaw_shift_px(pair, rough) for pair in pairs]
        foe = fit_focus_of_expansion(pairs, width, height, shifts) or foe

    samples = vehicle_samples(detections, width, height)
    vehicles = fit_vehicle_horizon(samples, height)

    horizon_y: float | None
    if options.horizon_y is not None:
        horizon_y, horizon_source = options.horizon_y, "flag"
    elif foe is not None:
        horizon_y, horizon_source = foe.y, "flow"
    elif vehicles is not None:
        horizon_y, horizon_source = vehicles.horizon_y, "vehicles"
    else:
        horizon_y, horizon_source = None, "default"

    resolved_horizon = height * 0.5 if horizon_y is None else horizon_y
    bottom = None if horizon_y is None else estimate_ground_bottom(pairs, horizon_y, height)
    if bottom is None:
        bottom = estimate_visible_bottom(pairs, height)

    height_m = DEFAULT_HEIGHT_M
    height_source = "default"
    if options.height_m is not None:
        height_m, height_source = options.height_m, "flag"
    else:
        fitted = vehicle_height_for_horizon(samples, resolved_horizon)
        if fitted is not None:
            height_m, height_source = fitted, "vehicles"

    camera = Camera(
        image_width=width,
        image_height=height,
        height_m=height_m,
        vfov_deg=options.vfov_deg,
        horizon_y=resolved_horizon,
        center_x=foe.x if foe is not None else None,
        visible_bottom=bottom,
    )
    return Calibration(camera, horizon_source, height_source)


def _yaw_shift_px(pair: FlowPair, camera: Camera) -> float:
    yaw = yaw_step(pair, camera)
    return 0.0 if yaw is None else -yaw * camera.focal_px


def build_scene(
    raw: RawClip,
    options: CameraOptions | None = None,
    track_config: TrackConfig | None = None,
    projection_config: ProjectionConfig | None = None,
    world_builder: Callable[[Sequence[Segment], Sequence[EgoStep]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Calibrate each shot, build tracks and return the JSON-ready scene.

    `world_builder` makes the static street. It's passed in because it needs OpenCV, and core can't.
    """
    opts = options or CameraOptions()
    frame_count = len(raw.times)
    boundaries = [0, *sorted(raw.cuts), frame_count]

    segments: list[Segment] = []
    calibrations: list[Calibration] = []
    ego_steps: list[EgoStep] = [EgoStep() for _ in range(max(frame_count - 1, 0))]
    coverage: list[float] = []

    for index in range(len(boundaries) - 1):
        start, end = boundaries[index], boundaries[index + 1]
        if end <= start:
            continue
        pairs = raw.flow[start : max(end - 1, start)]
        shot_detections = [d for d in raw.detections if start <= d.frame < end]
        calibration = calibrate(pairs, shot_detections, raw.video, opts)
        calibrations.append(calibration)
        segments.append(Segment(len(segments), start, end, calibration.camera))

        shot_times = raw.times[start:end]
        if opts.ego_speed_mps is not None:
            for frame in range(start, end - 1):
                dt = float(raw.times[frame + 1] - raw.times[frame])
                ego_steps[frame] = EgoStep(forward_m=opts.ego_speed_mps * dt)
            coverage.append(1.0)
        elif opts.use_ego_motion:
            steps, covered = estimate_ego_series(pairs, shot_times, calibration.camera)
            ego_steps[start : start + len(steps)] = steps
            coverage.append(covered)
        else:
            coverage.append(0.0)

    tracks = build_tracks(
        raw.detections,
        raw.times,
        segments,
        ego_steps,
        track_config=track_config,
        projection_config=projection_config,
        views=raw.views,
    )
    scene = _to_json(raw, segments, calibrations, coverage, ego_steps, tracks, opts)
    scene["world"] = world_builder(segments, ego_steps) if world_builder else None
    return scene


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _camera_json(calibration: Calibration) -> dict[str, Any]:
    camera = calibration.camera
    return {
        "heightM": _round(camera.height_m),
        "heightSource": calibration.height_source,
        "vfovDeg": _round(camera.vfov_deg),
        "focalPx": _round(camera.focal_px, 1),
        "horizonY": _round(camera.horizon, 1),
        "horizonFraction": _round(camera.horizon / camera.image_height, 3),
        "horizonSource": calibration.horizon_source,
        "pitchDeg": _round(camera.pitch_deg),
        "centerX": _round(camera.cx, 1),
        "visibleBottom": _round(camera.bottom, 1),
    }


def _to_json(
    raw: RawClip,
    segments: Sequence[Segment],
    calibrations: Sequence[Calibration],
    coverage: Sequence[float],
    ego_steps: Sequence[EgoStep],
    tracks: Sequence[WorldTrack],
    options: CameraOptions,
) -> dict[str, Any]:
    frame_count = len(raw.times)
    width, height = raw.video.width, raw.video.height
    objects_by_frame: list[list[dict[str, Any]]] = [[] for _ in range(frame_count)]
    for track in tracks:
        for sample in track.samples:
            item: dict[str, Any] = {
                "id": track.id,
                "class": track.class_name,
                "x": _round(sample.x),
                "y": _round(sample.y),
                "heading": _round(sample.heading, 3),
                "confidence": _round(sample.confidence, 3),
                "flags": list(sample.flags),
            }
            if sample.box is not None:
                box = sample.box
                item["box"] = [
                    _round(box.x1 / width, 4),
                    _round(box.y1 / height, 4),
                    _round(box.x2 / width, 4),
                    _round(box.y2 / height, 4),
                ]
            objects_by_frame[sample.frame].append(item)

    frames: list[dict[str, Any]] = []
    speeds = _ego_speeds(ego_steps, raw.times)
    segment_of = np.zeros(frame_count, dtype=np.int32)
    poses: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(frame_count)]
    for segment in segments:
        segment_of[segment.start_frame : segment.end_frame] = segment.index
        path = EgoPath(ego_steps, segment.start_frame, segment.end_frame)
        for index in range(segment.start_frame, segment.end_frame):
            local = index - segment.start_frame
            poses[index] = [
                _round(path.x[local]),
                _round(path.y[local]),
                _round(path.yaw[local], 5),
            ]
    for index, objects in enumerate(objects_by_frame):
        step = ego_steps[index] if index < len(ego_steps) else EgoStep()
        frames.append(
            {
                "frame": index,
                "t": _round(raw.times[index], 4),
                "segment": int(segment_of[index]),
                "ego": {
                    "speed": _round(speeds[index]),
                    "yaw": _round(step.yaw_rad, 5),
                    "pose": poses[index],
                },
                "objects": objects,
            }
        )

    metadata: dict[str, Any] = {
        "source": {
            "path": raw.video.path,
            "width": width,
            "height": height,
            "fps": _round(raw.video.fps, 3),
            "duration": _round(raw.video.duration, 3),
            "frameCount": raw.video.frame_count,
            "processedFrames": frame_count,
        },
        "camera": _camera_json(calibrations[0]) if calibrations else None,
        "model": raw.model,
        "egoMotion": {
            "mode": _ego_mode(options),
            "coverage": _round(float(np.mean(coverage)) if coverage else 0.0, 3),
        },
        "units": {"position": "metres", "heading": "radians", "speed": "m/s"},
        "axes": "ego frame: x right, y forward; heading 0 is ego's direction, positive turns right",
    }
    segment_json = [
        {
            "index": segment.index,
            "startFrame": segment.start_frame,
            "endFrame": segment.end_frame,
            "start": _round(raw.times[segment.start_frame], 4),
            "end": _round(raw.times[segment.end_frame - 1], 4),
            "camera": _camera_json(calibrations[segment.index]),
        }
        for segment in segments
    ]
    return {
        "version": SCENE_VERSION,
        "metadata": metadata,
        "segments": segment_json,
        "frames": frames,
    }


def _ego_mode(options: CameraOptions) -> str:
    if options.ego_speed_mps is not None:
        return "fixed"
    return "flow" if options.use_ego_motion else "off"


def _ego_speeds(steps: Sequence[EgoStep], times: Sequence[float]) -> list[float]:
    speeds: list[float] = []
    for index in range(len(times)):
        step_index = min(index, len(steps) - 1)
        if step_index < 0 or step_index + 1 >= len(times):
            speeds.append(speeds[-1] if speeds else 0.0)
            continue
        dt = max(float(times[step_index + 1] - times[step_index]), 1e-3)
        speeds.append(steps[step_index].forward_m / dt)
    return speeds
