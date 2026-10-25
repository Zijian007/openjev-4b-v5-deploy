"""Ask a single question that jointly refers to two images."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

try:
    from helpers import make_body
except ImportError:
    from multi_image_request import make_body

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1", help="Server IP or hostname")
parser.add_argument("--port", type=int, default=31000)
parser.add_argument("--images", nargs=2, default=[str(ROOT / "examples/red.png"), str(ROOT / "examples/blue.png")])
args = parser.parse_args()

body = make_body(args.images)
response = requests.post(f"http://{args.host}:{args.port}/v1/systemone", json=body, timeout=600)
if not response.ok:
    print(response.text)
response.raise_for_status()
print(json.dumps(response.json(), ensure_ascii=False, indent=2))
