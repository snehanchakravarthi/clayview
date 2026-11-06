import os
from pathlib import Path

from clayview.core.scene import RawClip, VideoInfo
from clayview.pipeline import (
    DetectionOptions,
    cache_key,
    load_raw,
    load_views,
    save_raw,
    save_views,
)


def test_views_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "clip.views.npz"
    views = {(0, 3): (0.25, 0.75), (5, 3): (0.5, 0.5)}
    save_views(path, views, "key")
    assert load_views(path, "key") == views


def test_no_views_round_trip(tmp_path: Path) -> None:
    """An empty country road gave no car crops, and saving the empty cache crashed the run."""
    path = tmp_path / "clip.views.npz"
    save_views(path, {}, "key")
    assert load_views(path, "key") == {}


def test_cache_key_follows_content_not_location(tmp_path: Path) -> None:
    """Caches made on a Mac missed in the cloud: the key held the absolute path and the mtime."""
    here, there = tmp_path / "a" / "clip.mp4", tmp_path / "b" / "elsewhere.mp4"
    for path in (here, there):
        path.parent.mkdir()
        path.write_bytes(b"same frames")
    os.utime(there, (0, 0))
    options = DetectionOptions()
    assert cache_key(here, options) == cache_key(there, options)

    there.write_bytes(b"other frame")  # same size, different content
    assert cache_key(here, options) != cache_key(there, options)
    assert cache_key(here, options) != cache_key(here, DetectionOptions(stride=2))


def test_reused_detections_point_at_the_video_as_it_is_now(tmp_path: Path) -> None:
    """A cache made on another machine kept its path, and the scene file pointed there."""
    old = VideoInfo("/Users/someone/clips/clip.mp4", 1280, 720, 30.0, 1.0, 30)
    raw = RawClip(video=old, times=[0.0], cuts=[], detections=[], flow=[], model={})
    path = tmp_path / "clip.raw.npz"
    save_raw(path, raw, "key")

    here = tmp_path / "clips" / "clip.mov"
    loaded = load_raw(path, "key", here)
    assert loaded is not None
    assert loaded.video.path == str(here)
    assert loaded.video.width == old.width
    assert load_raw(path, "other key", here) is None


def test_latest_links_go_where_the_viewer_looks(tmp_path: Path) -> None:
    """Eval runs wrote out/eval/latest.*, but "Open latest" only reads out/latest.*."""
    from clayview.cli import link_root

    out = tmp_path / "out"
    assert link_root(out / "clip.scene.json") == out
    assert link_root(out / "eval" / "clip.scene.json") == out
    assert link_root(tmp_path / "elsewhere" / "clip.scene.json") == tmp_path / "elsewhere"
