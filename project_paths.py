"""Make pinned upstream helpers importable from any working directory."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor/openjev"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
