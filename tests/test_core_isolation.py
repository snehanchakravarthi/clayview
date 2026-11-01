import subprocess
import sys
from pathlib import Path

CORE = Path(__file__).parent.parent / "src" / "clayview" / "core"


def test_core_never_imports_the_detector_or_video_stack() -> None:
    """The Pi build depends on this: core must run with numpy alone."""
    modules = sorted(p.stem for p in CORE.glob("*.py") if p.stem != "__init__")
    imports = "; ".join(f"import clayview.core.{name}" for name in modules)
    probe = (
        f"import sys; {imports}; "
        "banned = {'cv2', 'torch', 'ultralytics'} & set(sys.modules); "
        "print(sorted(banned)); sys.exit(1 if banned else 0)"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
