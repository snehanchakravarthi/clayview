"""Measure the focal-length signals in cached eval clips. A study tool, not part of the pipeline.

    uv run python tools/focal/probe.py comma2k19 md_i270 bled_rollei

Every estimate here is run with the clip's measured vfov hidden. Nothing in the pipeline estimates
focal length; see tools/focal/README.md for why each of these was ruled out.

- views: likelihood of the car view model's eight classes, given that traffic runs parallel to ego,
  so a car at column x is seen from atan((x - FoE x) / f).
- rotation: the full pinhole rotation field (Longuet-Higgins) fitted per frame, scored over a grid
  of vfov; run with and without rolling-shutter shear columns.
- roll: a camera pitched by theta sees vehicle yaw partly as roll, so roll / yaw shift = dy / f^2,
  with dy the horizon's offset from the image centre.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from clayview.core.camera import Camera
from clayview.core.heading import VIEW_BIN_RAD, VIEW_BINS, VIEW_REAR_BIN
from clayview.core.scene import CameraOptions, RawClip, calibrate
from clayview.orient import MODEL_NAME, sample_every
from clayview.pipeline import DetectionOptions, cache_key, load_raw, load_views

REPO = Path(__file__).resolve().parents[2]
GRID_DEG = (4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100)


def load(name: str) -> RawClip:
    video = REPO / "clips/eval" / name / "clip.mp4"
    raw_path = REPO / "out/eval" / f"{name}.raw.npz"
    options = DetectionOptions(weights="models/yolo11n.pt", imgsz=960, device="cpu", stride=1)
    key = cache_key(video, options)
    raw = load_raw(raw_path, key, video)
    if raw is None:
        raise SystemExit(f"{name}: no cache; run `uv run python tools/eval/eval.py pull` first")
    views_key = json.dumps([key, MODEL_NAME, sample_every(raw.video.fps)])
    views = load_views(raw_path.with_name(f"{name}.views.npz"), views_key) or {}
    return replace(raw, views=views)


def focal(vfov_deg: float, height: int) -> float:
    return height / 2.0 / math.tan(math.radians(vfov_deg) / 2.0)


def vfov(focal_px: float, height: int) -> float:
    return math.degrees(2.0 * math.atan(height / 2.0 / focal_px))


def first_shot(raw: RawClip) -> tuple[list, Camera]:
    end = min(raw.cuts) if raw.cuts else len(raw.times)
    pairs = list(raw.flow[: end - 1])
    detections = [d for d in raw.detections if d.frame < end]
    return pairs, calibrate(pairs, detections, raw.video, CameraOptions()).camera


def view_scan(raw: RawClip, camera: Camera) -> dict[int, float]:
    """Mean log likelihood of the view model's output per vfov, headings 0 or pi."""
    boxes = {(d.frame, d.track_id): d.box for d in raw.detections}
    columns, logs = [], []
    for key, view in raw.views.items():
        box = boxes.get(key)
        if box is not None:
            columns.append((box.x1 + box.x2) / 2.0 - camera.cx)
            logs.append(np.log(np.maximum(view, 0.03)))
    if not columns:
        return {}
    x, log_p = np.array(columns), np.array(logs)
    rows = np.arange(len(x))

    def score(f: float, heading: float) -> np.ndarray:
        seen = heading - np.arctan(x / f)
        position = (VIEW_REAR_BIN - seen / VIEW_BIN_RAD) % VIEW_BINS
        lower = np.floor(position).astype(int) % VIEW_BINS
        share = position - np.floor(position)
        return (1 - share) * log_p[rows, lower] + share * log_p[rows, (lower + 1) % VIEW_BINS]

    height = raw.video.height
    return {
        deg: float(
            np.maximum(score(focal(deg, height), 0.0), score(focal(deg, height), math.pi)).mean()
        )
        for deg in GRID_DEG
    }


def _frames(pairs: list, camera: Camera, width: int, height: int) -> list[tuple[np.ndarray, ...]]:
    """Per frame: points about the image centre, rays from the FoE, their lengths, cross targets."""
    foe = np.array([camera.cx, camera.horizon])
    centre = np.array([width / 2.0, height / 2.0])
    frames = []
    for pair in pairs:
        keep = pair.prev[:, 1] < camera.bottom
        if keep.sum() < 40:
            continue
        mid = (pair.prev[keep] + pair.next[keep]).astype(float) / 2.0
        flow = (pair.next[keep] - pair.prev[keep]).astype(float)
        ray = mid - foe
        length = np.hypot(*ray.T)
        ok = length > 0.05 * height
        mid, flow, ray, length = mid[ok], flow[ok], ray[ok], length[ok]
        target = (flow[:, 0] * ray[:, 1] - flow[:, 1] * ray[:, 0]) / length
        frames.append((mid - centre, ray, length, target))
    return frames


def _columns(xy: np.ndarray, ray: np.ndarray, length: np.ndarray, f: float, shear: bool):
    """Effect across the ray of each per-frame unknown: pitch, yaw, roll rates, maybe shear."""
    x, y = xy[:, 0], xy[:, 1]
    zero = np.zeros_like(x)
    u = [x * y / f, -(f + x * x / f), y]
    v = [f + y * y / f, -x * y / f, -x]
    if shear:  # rolling shutter: rows read later see more of the frame's motion
        u += [y, zero]
        v += [zero, y]
    gu, gv = np.stack(u, 1), np.stack(v, 1)
    return (gu * ray[:, 1:2] - gv * ray[:, 0:1]) / length[:, None]


def _robust(a: np.ndarray, t: np.ndarray, weight: np.ndarray | None = None):
    w = np.ones(len(t)) if weight is None else weight
    for _ in range(5 if weight is None else 1):
        root = np.sqrt(w)
        solution, *_ = np.linalg.lstsq(a * root[:, None], t * root, rcond=None)
        residual = np.abs(t - a @ solution)
        if weight is None:
            w = 1.0 / np.maximum(residual / (2.0 * max(float(np.median(residual)), 1e-6)), 1.0) ** 2
    return solution, w


def rotation_scan(
    pairs: list, camera: Camera, width: int, height: int, shear: bool
) -> dict[int, float]:
    """Weighted squared residual per vfov over frames that rotate, relative to the best vfov."""
    frames = _frames(pairs, camera, width, height)
    reference = focal(50.0, height)
    fixed = []
    for xy, ray, length, target in frames:
        solution, weight = _robust(_columns(xy, ray, length, reference, shear), target)
        fixed.append((xy, ray, length, target, weight, reference * math.hypot(*solution[:2])))
    typical = max(float(np.median([frame[-1] for frame in fixed])), 0.1)
    turning = [frame for frame in fixed if frame[-1] > typical]
    totals = {}
    for deg in GRID_DEG:
        f, total = focal(deg, height), 0.0
        for xy, ray, length, target, weight, _ in turning:
            a = _columns(xy, ray, length, f, shear)
            solution, _ = _robust(a, target, weight)
            total += float((weight * (target - a @ solution) ** 2).sum())
        totals[deg] = total
    best = min(totals.values())
    return {deg: total - best for deg, total in totals.items()}


def roll_coupling(pairs: list, camera: Camera, width: int, height: int) -> dict[str, float]:
    """Regress roll rate on yaw shift over turning frames; f = sqrt(dy / slope)."""
    yaw, roll = [], []
    for xy, ray, length, target in _frames(pairs, camera, width, height):
        x, y = xy[:, 0], xy[:, 1]
        zero, one = np.zeros_like(x), np.ones_like(x)
        gu = np.stack([one, zero, y, y, zero], 1)  # yaw shift, pitch shift, roll, shear u, shear v
        gv = np.stack([zero, one, -x, zero, y], 1)
        a = (gu * ray[:, 1:2] - gv * ray[:, 0:1]) / length[:, None]
        solution, _ = _robust(a, target)
        yaw.append(-solution[0])
        roll.append(solution[2])
    r, z = np.array(yaw), np.array(roll)
    turning = np.abs(r) > max(2.0 * float(np.median(np.abs(r))), 0.5)
    r, z = r[turning], z[turning]
    slope = float((r * z).sum() / (r * r).sum())
    error = float(np.sqrt(((z - slope * r) ** 2).sum() / (len(r) - 1) / (r * r).sum()))
    dy = height / 2.0 - camera.horizon
    f = math.sqrt(dy / slope) if dy * slope > 0 else float("nan")
    return {"dy": dy, "frames": int(turning.sum()), "slope": slope, "error": error, "focal": f}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="+")
    parser.add_argument("--skip-rotation", action="store_true", help="the slow scan")
    args = parser.parse_args()
    for name in args.names:
        raw = load(name)
        width, height = raw.video.width, raw.video.height
        pairs, camera = first_shot(raw)
        print(f"{name} {width}x{height}, horizon {camera.horizon:.0f}, FoE x {camera.cx:.0f}")
        views = view_scan(raw, camera)
        best_view = max(views, key=views.__getitem__) if views else "-"
        print(f"  views    best {best_view} deg  ", _row(views, 3))
        if not args.skip_rotation:
            for shear in (False, True):
                scan = rotation_scan(pairs, camera, width, height, shear)
                label = "rot+shear" if shear else "rotation "
                print(f"  {label} best {min(scan, key=scan.__getitem__)} deg  ", _row(scan, 0))
        roll = roll_coupling(pairs, camera, width, height)
        degrees = vfov(roll["focal"], height) if roll["focal"] == roll["focal"] else float("nan")
        print(
            f"  roll     dy {roll['dy']:.0f} px, {roll['frames']} turning frames, slope "
            f"{roll['slope']:.2e} +- {roll['error']:.1e}, "
            f"f {roll['focal']:.0f} px, {degrees:.1f} deg"
        )
    return 0


def _row(values: dict[int, float], digits: int) -> str:
    return " ".join(f"{deg}:{value:.{digits}f}" for deg, value in values.items())


if __name__ == "__main__":
    raise SystemExit(main())
