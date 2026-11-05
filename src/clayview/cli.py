"""Command line: `clayview process clip.mp4`, `clayview stats scene.json`."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from clayview.core.camera import DEFAULT_VFOV_DEG, horizon_from_pitch
from clayview.core.projection import ProjectionConfig
from clayview.core.scene import CameraOptions, RawClip, build_scene
from clayview.core.stats import summarize

if TYPE_CHECKING:
    from clayview.pipeline import DetectionOptions
    from clayview.world import SegMaps

BROWSER_VIDEO_SUFFIXES = {".mp4", ".m4v", ".mov", ".webm"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clayview", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    process = commands.add_parser("process", help="turn a clip into a scene file")
    process.add_argument("video", type=Path)
    process.add_argument("-o", "--output", type=Path, help="default: out/<clip>.scene.json")
    process.add_argument("--model", default="models/yolo11n.pt", help="YOLO weights")
    process.add_argument("--imgsz", type=int, default=960, help="inference size in pixels")
    process.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    process.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    process.add_argument("--limit", type=float, help="stop after this many seconds")
    process.add_argument("--fresh", action="store_true", help="ignore cached detections")
    process.add_argument(
        "--no-views",
        action="store_true",
        help="skip the car view model; heading gets less accurate",
    )
    process.add_argument(
        "--no-world", action="store_true", help="skip the street model; no road, trees or buildings"
    )
    process.add_argument(
        "--height", default="auto", help="camera height in metres, or auto (from car widths)"
    )
    process.add_argument(
        "--vfov",
        type=float,
        default=DEFAULT_VFOV_DEG,
        help="vertical field of view in degrees; this sets the scale of every distance",
    )
    process.add_argument(
        "--horizon",
        default="auto",
        help="horizon row: pixels, a fraction of the height below 1, or auto (per shot)",
    )
    process.add_argument("--pitch", type=float, help="camera pitch in degrees, down is positive")
    process.add_argument("--max-range", type=float, default=120.0, help="clamp distance, metres")
    process.add_argument(
        "--ego-speed",
        default="auto",
        help="ego speed in km/h, auto (from road flow) or off (work in relative motion)",
    )

    stats = commands.add_parser("stats", help="summarise a scene file")
    stats.add_argument("scene", type=Path)

    args = parser.parse_args(argv)
    if args.command == "stats":
        print(summarize(json.loads(args.scene.read_text())))
        return 0
    return _process(args)


def _build_world(segments: Any, ego_steps: Any, seg: SegMaps) -> dict[str, Any]:
    from clayview.world import build_world

    return build_world(segments, ego_steps, seg)


def _street_labels(
    raw: RawClip,
    video: Path,
    options: DetectionOptions,
    raw_path: Path,
    key: str,
    fresh: bool,
) -> SegMaps:
    from clayview.detect import pick_device
    from clayview.pipeline import load_seg, save_seg
    from clayview.segment import MODEL_NAME, label_every, segment_clip

    seg_path = raw_path.with_name(raw_path.name.removesuffix(".raw.npz") + ".seg.npz")
    every = label_every(raw.video.fps)
    seg_key = json.dumps([key, MODEL_NAME, every])
    seg = None if fresh else load_seg(seg_path, seg_key)
    if seg is None:
        device = pick_device(options.device)
        seg = segment_clip(video, options.stride, len(raw.times), device, every)
        save_seg(seg_path, seg, seg_key)
    else:
        print(f"reusing street labels from {seg_path}")
    return seg


def _with_views(
    raw: RawClip,
    video: Path,
    options: DetectionOptions,
    raw_path: Path,
    key: str,
    fresh: bool,
) -> RawClip:
    from clayview.detect import pick_device
    from clayview.orient import MODEL_NAME, collect_views, sample_every
    from clayview.pipeline import load_views, save_views

    views_path = raw_path.with_name(raw_path.name.removesuffix(".raw.npz") + ".views.npz")
    every = sample_every(raw.video.fps)
    views_key = json.dumps([key, MODEL_NAME, every])
    views = None if fresh else load_views(views_path, views_key)
    if views is None:
        device = pick_device(options.device)
        views = collect_views(video, raw.detections, options.stride, every, device)
        save_views(views_path, views, views_key)
    else:
        print(f"reusing car views from {views_path}")
    return replace(raw, views=views, model={**raw.model, "viewModel": MODEL_NAME})


def _process(args: argparse.Namespace) -> int:
    from clayview.pipeline import DetectionOptions, cache_key, extract, load_raw, save_raw
    from clayview.video import probe

    video: Path = args.video
    if not video.exists():
        print(f"no such file: {video}", file=sys.stderr)
        return 2
    output: Path = args.output or Path("out") / f"{video.stem}.scene.json"
    options = DetectionOptions(
        weights=args.model,
        imgsz=args.imgsz,
        device=args.device,
        stride=args.stride,
        limit_s=args.limit,
    )

    raw_path = output.with_name(output.name.removesuffix(".scene.json") + ".raw.npz")
    key = cache_key(video, options)
    raw = None if args.fresh else load_raw(raw_path, key, video)
    if raw is None:
        raw = extract(video, options)
        save_raw(raw_path, raw, key)
    else:
        print(f"reusing detections from {raw_path} (--fresh to redo them)")

    if not args.no_views:
        raw = _with_views(raw, video, options, raw_path, key, args.fresh)

    info = probe(video)
    camera = _camera_options(args, info.height)
    world_builder = None
    if not args.no_world:
        seg = _street_labels(raw, video, options, raw_path, key, args.fresh)
        world_builder = partial(_build_world, seg=seg)
    scene = build_scene(
        raw,
        camera,
        projection_config=ProjectionConfig(max_range_m=args.max_range),
        world_builder=world_builder,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scene, separators=(",", ":")))
    _link_latest(output, video)

    print()
    print(summarize(scene))
    print(f"\nwrote {output} ({output.stat().st_size / 1e6:.1f} MB)")
    if video.suffix.lower() not in BROWSER_VIDEO_SUFFIXES:
        print(
            f"note: browsers may not play {video.suffix} files. For the viewer, convert it:\n"
            f"  ffmpeg -i {video} -c:v libx264 -crf 23 -an {video.with_suffix('.mp4')}"
        )
    return 0


def _camera_options(args: argparse.Namespace, image_height: int) -> CameraOptions:
    horizon_y: float | None = None
    if args.pitch is not None:
        horizon_y = horizon_from_pitch(image_height, args.vfov, args.pitch)
    elif args.horizon != "auto":
        value = float(args.horizon)
        horizon_y = value * image_height if value < 1.0 else value

    ego = str(args.ego_speed)
    return CameraOptions(
        height_m=None if args.height == "auto" else float(args.height),
        vfov_deg=args.vfov,
        horizon_y=horizon_y,
        ego_speed_mps=None if ego in ("auto", "off") else float(ego) / 3.6,
        use_ego_motion=ego != "off",
    )


def _link_latest(output: Path, video: Path) -> None:
    """Point out/latest.* at this run, so the viewer can open it with one click."""
    root = link_root(output)
    for name, target in (
        ("latest.scene.json", output.resolve()),
        (f"latest{video.suffix}", video.resolve()),
    ):
        link = root / name
        if link.resolve() == target and link.is_symlink():
            continue
        for stale in root.glob(f"{link.stem}*"):
            if stale.is_symlink() and (name.endswith(".json") == stale.name.endswith(".json")):
                stale.unlink()
        link.symlink_to(target)


def link_root(output: Path) -> Path:
    """Where the latest.* links go: the `out/` folder the viewer serves, even for out/eval/ runs.

    Output written anywhere else gets its links beside it.
    """
    folder = output.resolve().parent
    return folder.parent if folder.parent.name == "out" else folder


if __name__ == "__main__":
    raise SystemExit(main())
