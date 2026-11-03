from pathlib import Path

from clayview.pipeline import load_views, save_views


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
