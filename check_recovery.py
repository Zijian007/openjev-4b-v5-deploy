"""Run check_recovery from scripts/."""
import sys
from pathlib import Path

script = Path(__file__).resolve().parent / "scripts/check_recovery.py"
sys.path.insert(0, str(Path(__file__).resolve().parent))
with open(script, "r", encoding="utf-8") as f:
    exec(compile(f.read(), str(script), "exec"))
