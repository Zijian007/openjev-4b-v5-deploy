"""Make pinned upstream helpers and src modules importable from any working directory."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT / "src", ROOT / "vendor/openjev"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
