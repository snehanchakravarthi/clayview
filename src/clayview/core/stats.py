"""Human-readable summary of a scene file, for checking output instead of trusting it."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np


def summarize(scene: dict[str, Any]) -> str:
    """Shape of the output: calibration, tracks, distances, jitter and flags."""
    frames = scene["frames"]
    meta = scene["metadata"]
    source = meta["source"]
    lines = [
        f"source      {source['path']}",
        f"            {source['width']}x{source['height']} {source['fps']} fps, "
        f"{source['duration']:.1f} s, {source['processedFrames']} frames processed",
        f"ego motion  {meta['egoMotion']['mode']}, coverage {meta['egoMotion']['coverage']:.0%}",
    ]
    for segment in scene["segments"]:
        camera = segment["camera"]
        lines.append(
            f"shot {segment['index']:<6} {segment['start']:7.2f}-{segment['end']:7.2f} s  "
            f"horizon {camera['horizonFraction']:.3f} ({camera['horizonSource']})  "
            f"height {camera['heightM']:.2f} m ({camera['heightSource']})  "
            f"bottom {camera['visibleBottom']:.0f} px"
        )

    speeds = np.array([f["ego"]["speed"] for f in frames], dtype=float)
    if speeds.size:
        lines.append(
            f"ego speed   median {np.median(speeds) * 3.6:.0f} km/h, "
            f"p95 {np.percentile(speeds, 95) * 3.6:.0f} km/h"
        )

    tracks: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    flags: Counter[str] = Counter()
    per_frame: list[int] = []
    for frame in frames:
        per_frame.append(len(frame["objects"]))
        for item in frame["objects"]:
            tracks[item["id"]].append((frame["frame"], item))
            flags.update(item["flags"])
    total = sum(per_frame)
    classes = Counter(items[0][1]["class"] for items in tracks.values())
    lines.append(f"tracks      {len(tracks)}: " + ", ".join(f"{v} {k}" for k, v in classes.items()))
    if not total:
        lines.append("objects     none")
        return "\n".join(lines)

    lines.append(
        f"objects     {total} samples, {np.mean(per_frame):.1f} per frame, max {max(per_frame)}"
    )
    durations = [
        (items[-1][0] - items[0][0] + 1) * _frame_seconds(frames) for items in tracks.values()
    ]
    lines.append(
        f"track span  median {np.median(durations):.1f} s, p90 {np.percentile(durations, 90):.1f} s"
    )
    ranges = np.array(
        [math.hypot(i["x"], i["y"]) for items in tracks.values() for _, i in items], dtype=float
    )
    lines.append(
        "range       "
        f"p5 {np.percentile(ranges, 5):.1f} m, p50 {np.percentile(ranges, 50):.1f} m, "
        f"p95 {np.percentile(ranges, 95):.1f} m, max {ranges.max():.1f} m"
    )
    lines.append("jitter      " + _jitter_line(tracks))
    lines.append("flags       " + ", ".join(f"{k} {v / total:.0%}" for k, v in flags.most_common()))
    return "\n".join(lines)


def _frame_seconds(frames: list[dict[str, Any]]) -> float:
    if len(frames) < 2:
        return 0.0
    return (frames[-1]["t"] - frames[0]["t"]) / (len(frames) - 1)


def _jitter_line(tracks: dict[int, list[tuple[int, dict[str, Any]]]]) -> str:
    """RMS of the second difference of position: how much it shakes, independent of motion."""
    values: list[float] = []
    heading_steps: list[float] = []
    for items in tracks.values():
        if len(items) < 5:
            continue
        x = np.array([i["x"] for _, i in items], dtype=float)
        y = np.array([i["y"] for _, i in items], dtype=float)
        heading = np.unwrap(np.array([i["heading"] for _, i in items], dtype=float))
        values.append(float(np.sqrt(np.mean(np.diff(x, 2) ** 2 + np.diff(y, 2) ** 2))))
        heading_steps.append(float(np.percentile(np.abs(np.diff(heading)), 99)))
    if not values:
        return "n/a"
    return (
        f"position {np.median(values) * 100:.1f} cm/frame^2 median, "
        f"heading step p99 {math.degrees(float(np.median(heading_steps))):.1f} deg/frame"
    )
