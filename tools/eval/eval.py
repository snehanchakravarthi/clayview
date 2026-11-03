"""Run the pipeline over clips from many cameras and score it against what's known about each.

    uv run python tools/eval/eval.py fetch            # download clips listed in clips.json
    uv run python tools/eval/eval.py run              # process every clip and score it
    uv run python tools/eval/eval.py run bled_rollei --fresh
    uv run python tools/eval/eval.py score            # rescore existing scene files
    uv run python tools/eval/eval.py pack             # clips and caches into out/release/
    uv run python tools/eval/eval.py pull             # the same, from the GitHub release

Wikimedia rate-limits cloud machines, and the models take half an hour a clip without a GPU. So a
machine that has run everything packs each clip with its caches, the tars go up as a release, and
any other machine pulls them and rescores in seconds. The caches are keyed on the video's bytes, so
they hit wherever the clip lands.

Truth is uneven on purpose. comma2k19 has CAN speed, a gyro and radar. Bled has a speed overlay read
by eye. Others only say what is certain from looking at the frames: the car is moving, there are no
edits. A clip without truth still shows crashes, fallbacks and implausible numbers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import time
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CLIPS = REPO / "clips/eval"
OUT = REPO / "out/eval"
MANIFEST = Path(__file__).with_name("clips.json")
RELEASE = REPO / "out/release"
# Bump with CACHE_VERSION in pipeline.py: caches from another version miss and get rebuilt.
DATA_TAG = "eval-data-v3"
DATA_URL = f"https://github.com/snehanchakravarthi/clayview/releases/download/{DATA_TAG}"
CACHES = ("raw", "views", "seg")
USER_AGENT = "clayview-eval/0.1 (https://github.com/snehanchakravarthi/clayview)"
COMMA = "https://raw.githubusercontent.com/commaai/comma2k19/master/Example_1/b0c9d2329ad1606b%7C2018-08-02--08-34-47/40/"
COMMA_FILES = [
    "video.hevc",
    "global_pose/frame_times",
    "global_pose/frame_velocities",
    "global_pose/frame_orientations",
    "processed_log/CAN/speed/t",
    "processed_log/CAN/speed/value",
    "processed_log/CAN/radar/t",
    "processed_log/CAN/radar/value",
]
COMMA_FOCAL_PX = 910.0
MOVING_KMH = 5.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["fetch", "run", "score", "pack", "pull"])
    parser.add_argument("names", nargs="*", help="clips to use; default all")
    parser.add_argument("--fresh", action="store_true", help="redo detection, not just the scene")
    parser.add_argument("--source", default=DATA_URL, help="where pull finds the tars")
    args = parser.parse_args()
    clips: dict[str, dict[str, Any]] = json.loads(MANIFEST.read_text())
    names = args.names or list(clips)
    unknown = [name for name in names if name not in clips]
    if unknown:
        parser.error(f"not in clips.json: {', '.join(unknown)}")

    if args.command == "fetch":
        for name in names:
            fetch(name, clips[name])
        return 0
    if args.command == "pack":
        return pack(names, clips)
    if args.command == "pull":
        missing = [name for name in names if not pull(name, args.source)]
        return 1 if missing else 0

    results: dict[str, dict[str, Any]] = {}
    for name in names:
        video = CLIPS / name / "clip.mp4"
        scene_path = OUT / f"{name}.scene.json"
        if not video.exists():
            results[name] = {"error": "not fetched"}
            continue
        if args.command == "run":
            began = time.monotonic()
            command = [
                sys.executable,
                "-m",
                "clayview.cli",
                "process",
                str(video),
                "-o",
                str(scene_path),
            ]
            if "vfov" in clips[name]:
                # Known or measured focal length, so the run tests the rest. Nothing estimates it.
                command += ["--vfov", str(clips[name]["vfov"])]
            run = subprocess.run(
                command + (["--fresh"] if args.fresh else []),
                cwd=REPO,
                capture_output=True,
                text=True,
            )
            if run.returncode != 0:
                results[name] = {"error": (run.stderr or run.stdout).strip().splitlines()[-1]}
                print(f"{name}: FAILED {results[name]['error']}", flush=True)
                continue
            print(f"{name}: processed in {time.monotonic() - began:.0f} s", flush=True)
        if not scene_path.exists():
            results[name] = {"error": "no scene file"}
            continue
        results[name] = score(name, clips[name], json.loads(scene_path.read_text()))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print_table(results)
    return 0


def fetch(name: str, entry: dict[str, Any]) -> None:
    target = CLIPS / name / "clip.mp4"
    if target.exists():
        print(f"{name}: have it")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if entry.get("source") == "comma2k19":
        for file in COMMA_FILES:
            _curl(COMMA + file, target.parent / file)
        _ffmpeg(["-r", "20", "-i", str(target.parent / "video.hevc"), "-crf", "18"], target)
        print(f"{name}: fetched")
        return

    # Wikimedia truncates ranged reads, so seeking with ffmpeg fails. Pull the whole file and cut.
    url = entry["url"]
    if entry.get("fetch") == "720p":
        parts = urllib.parse.urlparse(url).path.split("/")
        url = f"https://upload.wikimedia.org/wikipedia/commons/transcoded/{parts[-3]}/{parts[-2]}/{parts[-1]}/{parts[-1]}.720p.vp9.webm"
    download = CLIPS / "_download" / f"{name}{Path(urllib.parse.urlparse(url).path).suffix}"
    _curl(url, download)
    _ffmpeg(
        [
            "-ss",
            str(entry["start_s"]),
            "-i",
            str(download),
            "-t",
            str(entry["duration_s"]),
            "-crf",
            "20",
        ],
        target,
    )
    download.unlink()
    print(f"{name}: fetched")


def _members(name: str) -> list[Path]:
    """Repo-relative files for one clip: the video, any truth files, and whatever caches exist."""
    folder = CLIPS / name
    files = [path for path in sorted(folder.rglob("*")) if path.is_file()]
    # comma2k19's raw HEVC is only the source of clip.mp4, and the caches are keyed on clip.mp4.
    files = [path for path in files if path.name != "video.hevc"]
    files += [
        OUT / f"{name}.{kind}.npz" for kind in CACHES if (OUT / f"{name}.{kind}.npz").exists()
    ]
    return [path.relative_to(REPO) for path in files]


def pack(names: list[str], clips: dict[str, dict[str, Any]]) -> int:
    RELEASE.mkdir(parents=True, exist_ok=True)
    packed = []
    for name in names:
        if not (CLIPS / name / "clip.mp4").exists():
            print(f"{name}: not fetched, skipped")
            continue
        members = _members(name)
        caches = [kind for kind in CACHES if (OUT / f"{name}.{kind}.npz").exists()]
        with tarfile.open(RELEASE / f"{name}.tar", "w") as tar:
            for member in members:
                tar.add(REPO / member, arcname=str(member))
        size = (RELEASE / f"{name}.tar").stat().st_size / 1e6
        print(f"{name}: {size:.0f} MB, caches: {', '.join(caches) or 'none'}")
        packed.append(name)
    (RELEASE / "CREDITS.md").write_text(credits(packed, clips))
    folder = RELEASE.relative_to(REPO)
    print(
        f"\nwrote {len(packed)} tars and CREDITS.md to {folder}/. From the repo root, publish with"
        f"\n  gh release create {DATA_TAG} {folder}/* --title {DATA_TAG}"
        f" --notes-file {folder}/CREDITS.md"
    )
    return 0


def credits(names: list[str], clips: dict[str, dict[str, Any]]) -> str:
    """Attribution for the release notes: every clip is redistributed under its own licence."""
    lines = [
        f"# clayview eval clips ({DATA_TAG})",
        "",
        "Clips for `tools/eval/eval.py pull`, each packed with the pipeline's caches for it.",
        "Each clip is a trimmed, re-encoded excerpt of the source below, under that source's",
        "licence and credited to its author. A CC BY-SA excerpt stays CC BY-SA.",
        "",
        "| clip | source | author | licence | excerpt |",
        "|---|---|---|---|---|",
    ]
    for name in names:
        entry = clips[name]
        source = entry.get("page", entry["url"])
        if "start_s" in entry:
            excerpt = f"{entry['start_s']}-{entry['start_s'] + entry['duration_s']} s"
        else:
            excerpt = "whole"
        lines.append(f"| {name} | {source} | {entry['author']} | {entry['licence']} | {excerpt} |")
    return "\n".join(lines) + "\n"


def pull(name: str, source: str) -> bool:
    """Fetch one clip and its caches from the release. False if the release doesn't have it.

    A clip from `fetch` is re-encoded here, so its bytes, and its cache key, differ from the
    release's. Without all its caches it is replaced by the release's copy.
    """
    have = [CLIPS / name / "clip.mp4", *(OUT / f"{name}.{kind}.npz" for kind in CACHES)]
    if all(path.exists() for path in have):
        print(f"{name}: have it")
        return True
    download = CLIPS / "_download" / f"{name}.tar"
    try:
        _curl(f"{source}/{name}.tar", download)
    except subprocess.CalledProcessError:
        print(f"{name}: not in {source}")
        return False
    try:
        with tarfile.open(download) as tar:
            for member in tar.getmembers():
                if not member.isfile() or not _belongs_to(name, member.name):
                    raise ValueError(f"{download}: unexpected entry {member.name}")
            tar.extractall(REPO, filter="data")
    finally:
        download.unlink()
    print(f"{name}: pulled")
    return True


def _belongs_to(name: str, entry: str) -> bool:
    """Only this clip's folder and its caches. Checked on the resolved path, so ../ can't escape."""
    target = (REPO / entry).resolve()
    if target.is_relative_to((CLIPS / name).resolve()):
        return True
    return target.parent == OUT.resolve() and target.name in {
        f"{name}.{kind}.npz" for kind in CACHES
    }


def _curl(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-sS", "--fail", "-L", "-A", USER_AGENT, "-o", str(path), url], check=True
    )


def _ffmpeg(inputs: list[str], output: Path) -> None:
    encode = ["-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
    command = ["ffmpeg", "-v", "error", "-y", *inputs, *encode, str(output)]
    subprocess.run(command, check=True)


def score(name: str, entry: dict[str, Any], scene: dict[str, Any]) -> dict[str, Any]:
    frames = scene["frames"]
    segments = scene["segments"]
    times = np.array([frame["t"] for frame in frames])
    speed = np.array([frame["ego"]["speed"] for frame in frames]) * 3.6
    yaw = np.array([frame["ego"]["yaw"] for frame in frames])
    minutes = max(float(times[-1] - times[0]), 1e-3) / 60.0
    result: dict[str, Any] = {
        "cuts": len(segments) - 1,
        "horizon": "/".join(sorted({s["camera"]["horizonSource"] for s in segments})),
        "height": [
            min(s["camera"]["heightM"] for s in segments),
            max(s["camera"]["heightM"] for s in segments),
        ],
        "moving_share": round(float((speed >= MOVING_KMH).mean()), 3),
        "speed_median_kmh": round(float(np.median(speed)), 1),
        "turn_deg_per_min": round(float(np.degrees(np.abs(yaw).sum())) / minutes, 1),
        "drift_deg_per_min": round(float(np.degrees(yaw.sum())) / minutes, 1),
    }
    expect = entry.get("expect", {})
    failures: list[str] = []
    if "cuts" in expect and result["cuts"] != expect["cuts"]:
        failures.append(f"cuts {result['cuts']} != {expect['cuts']}")
    if expect.get("moving") and result["moving_share"] < 0.9:
        failures.append(f"moving only {result['moving_share']:.0%}")
    if "speed_kmh" in expect:
        truth = np.array(expect["speed_kmh"], dtype=float)
        truth[:, 1] *= entry.get("time_scale", 1.0)  # a timelapse overlay shows real-world speed
        estimates = np.array([speed[np.abs(times - t) <= 0.5].mean() for t in truth[:, 0]])
        ratio = float(np.median(estimates / truth[:, 1]))
        result["speed_ratio"] = round(ratio, 3)
        if abs(ratio - 1.0) > 0.1:
            failures.append(f"speed ratio {ratio:.2f}")
    if entry.get("source") == "comma2k19":
        result.update(_score_comma(scene, speed / 3.6, yaw))
        if abs(result["speed_ratio"] - 1.0) > 0.1:
            failures.append(f"speed ratio {result['speed_ratio']:.2f}")
    result["failures"] = failures
    return result


def _score_comma(scene: dict[str, Any], speed: np.ndarray, yaw: np.ndarray) -> dict[str, Any]:
    root = CLIPS / "comma2k19"

    def load(file: str) -> np.ndarray:
        return np.load(root / file, allow_pickle=False)

    frame_t = load("global_pose/frame_times")
    n = min(len(frame_t), len(speed))
    frame_t = frame_t[:n]
    true_speed = np.interp(
        frame_t, load("processed_log/CAN/speed/t"), load("processed_log/CAN/speed/value").ravel()
    )
    moving = true_speed > 4.0

    # comma's orientations are ecef_from_local in practice, so conjugate before rotating velocity.
    q = load("global_pose/frame_orientations")[:n] * np.array([1.0, -1.0, -1.0, -1.0])
    velocity = _rotate(q, load("global_pose/frame_velocities")[:n])  # forward, right, down
    height = scene["metadata"]["source"]["height"]
    foe_y = height / 2 + COMMA_FOCAL_PX * velocity[:, 2] / np.maximum(velocity[:, 0], 1e-3)
    errors, weights = [], []
    for segment in scene["segments"]:
        a, b = segment["startFrame"], min(segment["endFrame"], n)
        if b > a and moving[a:b].any():
            errors.append(
                abs(segment["camera"]["horizonY"] - float(np.median(foe_y[a:b][moving[a:b]])))
            )
            weights.append(b - a)

    # Yaw between frames from the poses: the turn of the forward axis about the down axis. It
    # matches gyro axis 2 (correlation 0.81) with less noise.
    forward = _rotate(
        q_raw := load("global_pose/frame_orientations")[:n], np.tile([1.0, 0, 0], (n, 1))
    )
    down = _rotate(q_raw, np.tile([0.0, 0.0, 1.0], (n, 1)))
    turn = np.cross(forward[:-1], forward[1:])
    true_yaw = np.arctan2((turn * down[:-1]).sum(1), (forward[:-1] * forward[1:]).sum(1))
    yaw = yaw[: n - 1]
    minutes = float(frame_t[-1] - frame_t[0]) / 60.0
    return {
        "speed_ratio": round(float(np.median(speed[:n][moving] / true_speed[moving])), 3),
        "horizon_err_px": round(float(np.average(errors, weights=weights)), 1) if errors else None,
        "yaw_corr": round(float(np.corrcoef(true_yaw, yaw)[0, 1]), 3),
        "true_turn_deg_per_min": round(float(np.degrees(np.abs(true_yaw).sum())) / minutes, 1),
        "true_drift_deg_per_min": round(float(np.degrees(true_yaw.sum())) / minutes, 1),
        "radar": _radar(scene, frame_t, true_speed, load),
    }


def _radar(
    scene: dict[str, Any], frame_t: np.ndarray, true_speed: np.ndarray, load: Any
) -> dict[str, float]:
    """Median estimated/radar range to the rear of moving lead vehicles, matched by bearing."""
    radar_t, radar = load("processed_log/CAN/radar/t"), load("processed_log/CAN/radar/value")
    pairs = []
    for index, frame in enumerate(scene["frames"][: len(frame_t)]):
        near = np.abs(radar_t - frame_t[index]) < 0.05
        targets = radar[near]
        targets = targets[
            np.isfinite(targets[:, 0]) & (targets[:, 0] > 2) & (np.abs(targets[:, 1]) < 8)
        ]
        targets = targets[np.abs(targets[:, 2] + true_speed[index]) > 3.0]  # drop static clutter
        for obj in frame["objects"]:
            if obj["class"] not in ("car", "truck", "bus") or {"low_conf", "held"} & set(
                obj["flags"]
            ):
                continue
            bearing = np.arctan2(obj["x"], obj["y"])
            best = None
            for distance, left, *_ in targets:
                error = abs(bearing - np.arctan2(-left, distance))
                if error < np.radians(1.5) and (best is None or error < best[0]):
                    best = (error, distance)
            if best is not None:
                pairs.append((obj["y"] - 2.25, best[1]))  # scene y is the centre of a 4.5 m car
    pairs_array = np.array(pairs).reshape(-1, 2)
    out = {}
    for low, high in ((20, 40), (40, 70)):
        sel = (pairs_array[:, 1] >= low) & (pairs_array[:, 1] < high)
        if sel.sum() >= 10:
            out[f"{low}-{high}m"] = round(
                float(np.median(pairs_array[sel, 0] / pairs_array[sel, 1])), 3
            )
    return out


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by Hamilton quaternions q (w, x, y, z)."""
    w, x, y, z = q.T
    rows = [
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1),
    ]
    return np.stack([(row * v).sum(1) for row in rows], axis=1)


def print_table(results: dict[str, dict[str, Any]]) -> None:
    print(
        f"{'clip':22} {'cuts':>4} {'horizon':>14} {'height':>11} {'moving':>6} {'km/h':>5} "
        f"{'ratio':>5} {'turn/min':>8} {'drift/min':>9}  failures"
    )
    for name, r in results.items():
        if "error" in r:
            print(f"{name:22} ERROR {r['error']}")
            continue
        height = f"{r['height'][0]:.2f}-{r['height'][1]:.2f}"
        ratio = f"{r['speed_ratio']:.2f}" if "speed_ratio" in r else "-"
        print(
            f"{name:22} {r['cuts']:>4} {r['horizon']:>14} {height:>11} {r['moving_share']:>6.0%} "
            f"{r['speed_median_kmh']:>5.1f} {ratio:>5} {r['turn_deg_per_min']:>8.0f} "
            f"{r['drift_deg_per_min']:>9.0f}  " + "; ".join(r["failures"])
        )
        if "yaw_corr" in r:
            print(
                f"{'':22} horizon err {r['horizon_err_px']} px, yaw corr {r['yaw_corr']}, "
                f"true turn {r['true_turn_deg_per_min']}/min, "
                f"true drift {r['true_drift_deg_per_min']}/min, radar {r['radar']}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
