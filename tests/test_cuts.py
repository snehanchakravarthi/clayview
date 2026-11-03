import numpy as np

from clayview.core.cuts import CutDetector, find_cuts, structural_similarity

FPS = 10.0


def scene(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse = rng.integers(20, 235, size=(9, 16), dtype=np.uint8)
    return np.kron(coarse, np.ones((6, 6), dtype=np.uint8))


def held(frame: np.ndarray, count: int = 15) -> list[np.ndarray]:
    return [frame] * count


def wiper(frame: np.ndarray) -> np.ndarray:
    """A dark blade across most of the glass."""
    swept = frame.copy()
    rows, cols = np.indices(frame.shape)
    swept[np.abs(cols - 1.6 * rows - 10) < 30] = 5
    return swept


def test_identical_frames_are_fully_similar() -> None:
    frame = scene(1)
    assert structural_similarity(frame, frame) > 0.999


def test_different_scenes_are_a_cut() -> None:
    assert find_cuts([*held(scene(1)), *held(scene(2))], FPS) == [15]


def test_cut_is_reported_after_the_lookahead() -> None:
    detector = CutDetector(FPS)
    reported = [detector.update(frame) for frame in [*held(scene(1), 3), *held(scene(2))]]
    assert reported.index(3) == 3 + round(FPS)


def test_cut_near_the_end_still_counts() -> None:
    assert find_cuts([*held(scene(1)), *held(scene(2), 3)], FPS) == [15]


def test_wiper_sweep_is_not_a_cut() -> None:
    """Wipers made 16 false cuts in a minute of comma2k19 and 10 in a rainy highway clip."""
    frame = scene(1)
    assert structural_similarity(frame[:38], wiper(frame)[:38]) < 0.55
    assert find_cuts([*held(frame), wiper(frame), wiper(frame), *held(frame)], FPS) == []


def test_exposure_step_is_not_a_cut() -> None:
    """Dashcams jump their exposure. Real footage produced seven false cuts before this test."""
    frame = scene(1)
    brighter = np.clip(frame.astype(np.int16) * 2 // 3 + 90, 0, 255).astype(np.uint8)
    assert find_cuts([*held(frame), *held(brighter)], FPS) == []


def test_motion_is_not_a_cut() -> None:
    frame = scene(1)
    assert find_cuts([np.roll(frame, shift, axis=1) for shift in range(12)], FPS) == []


def test_shared_dash_does_not_hide_a_cut() -> None:
    """Two clips from one car share the hood; only the top of the frame is compared."""
    a, b = scene(1).copy(), scene(2).copy()
    dash = scene(3)[40:]
    a[40:] = dash
    b[40:] = dash
    assert find_cuts([*held(a), *held(b)], FPS) == [15]


def test_flash_right_after_a_cut_is_one_cut() -> None:
    assert find_cuts([*held(scene(1)), *held(scene(2), 2), scene(3), *held(scene(2))], FPS) == [15]


def test_black_frames_do_not_crash() -> None:
    black = np.zeros((54, 96), dtype=np.uint8)
    assert find_cuts(held(black), FPS) == []
