"""Ask a single question that jointly refers to two images."""
import argparse
import json
from pathlib import Path

import requests

from multi_image_request import make_body

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1", help="Server IP or hostname")
parser.add_argument("--port", type=int, default=31000)
parser.add_argument("--images", nargs=2, default=[str(root / "examples/red.png"), str(root / "examples/blue.png")])
args = parser.parse_args()

body = make_body(args.images)
response = requests.post(f"http://{args.host}:{args.port}/v1/systemone", json=body, timeout=600)
if not response.ok:
    print(response.text)
response.raise_for_status()
print(json.dumps(response.json(), ensure_ascii=False, indent=2))
