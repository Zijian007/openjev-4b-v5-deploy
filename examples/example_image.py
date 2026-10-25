"""Read a CLIENT-local image path and send its base64 bytes with requests."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1", help="Server IP or hostname")
parser.add_argument("--port", type=int, default=31000)
parser.add_argument("--image", default=str(ROOT / "examples/red.png"))
parser.add_argument("--question", default="What is the dominant color in Image 0?")
parser.add_argument("--options", default='{"A":"Red","B":"Blue"}')
args = parser.parse_args()

# This path is on the computer running this example, not on the server.
image_path = Path(args.image)
data = base64.b64encode(image_path.read_bytes()).decode("ascii")
body = {
    "state": "Image 0: <<IMG[0]>>",
    "images": [{"data": data}],
    "questions": {
        "color": {"type": "choice", "instructions": args.question, "criteria": json.loads(args.options)},
    },
}
response = requests.post(f"http://{args.host}:{args.port}/v1/systemone", json=body, timeout=600)
if not response.ok:
    print(response.text)
response.raise_for_status()
print(json.dumps(response.json(), ensure_ascii=False, indent=2))
