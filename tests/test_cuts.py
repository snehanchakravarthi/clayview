import numpy as np

from clayview.core.cuts import CutDetector, structural_similarity


def scene(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse = rng.integers(20, 235, size=(9, 16), dtype=np.uint8)
    return np.kron(coarse, np.ones((6, 6), dtype=np.uint8))


def test_identical_frames_are_fully_similar() -> None:
    frame = scene(1)
    assert structural_similarity(frame, frame) > 0.999


def test_different_scenes_are_a_cut() -> None:
    detector = CutDetector()
    assert detector.update(scene(1)) is False
    assert detector.update(scene(1)) is False
    assert detector.update(scene(2)) is True


def test_exposure_step_is_not_a_cut() -> None:
    """Dashcams jump their exposure. Real footage produced seven false cuts before this test."""
    detector = CutDetector()
    frame = scene(1)
    brighter = np.clip(frame.astype(np.int16) * 2 // 3 + 90, 0, 255).astype(np.uint8)
    assert detector.update(frame) is False
    assert detector.update(frame) is False
    assert detector.update(brighter) is False


def test_motion_is_not_a_cut() -> None:
    detector = CutDetector()
    frame = scene(1)
    for shift in range(12):
        assert detector.update(np.roll(frame, shift, axis=1)) is False


def test_shared_dash_does_not_hide_a_cut() -> None:
    """Two clips from one car share the hood; only the top of the frame is compared."""
    a, b = scene(1).copy(), scene(2).copy()
    dash = scene(3)[40:]
    a[40:] = dash
    b[40:] = dash
    detector = CutDetector()
    detector.update(a)
    assert detector.update(b) is True


def test_flash_right_after_a_cut_is_ignored() -> None:
    detector = CutDetector()
    detector.update(scene(1))
    assert detector.update(scene(2)) is True
    assert detector.update(scene(3)) is False


def test_black_frames_do_not_crash() -> None:
    detector = CutDetector()
    black = np.zeros((54, 96), dtype=np.uint8)
    detector.update(black)
    assert detector.update(black) is False
