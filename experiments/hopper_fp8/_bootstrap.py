"""Select this experiment's QuACK fork and this checkout's SonicMoE."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
ROOT = HERE.parent.parent
for name, expected in (("quack", VENDOR / "quack"), ("sonicmoe", ROOT / "sonicmoe")):
    module = sys.modules.get(name)
    if module is not None and Path(module.__file__).resolve().parent != expected:
        raise RuntimeError(f"{name} was already imported from another checkout; use a fresh Python process")
sys.path[:0] = [str(VENDOR), str(ROOT)]
