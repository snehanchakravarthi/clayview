import os
from pathlib import Path

from clayview.pipeline import DetectionOptions, cache_key, load_views, save_views


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
