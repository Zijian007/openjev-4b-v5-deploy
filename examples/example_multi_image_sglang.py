"""Use the author's unmodified OpenJevSGLang client with shared multiple images.

Requires an actual OpenJev SGLang /classify service on port 30000.
"""
import argparse
import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import project_paths  # noqa: F401
from multi_image_request import make_body
from decisions_api import assemble, build_plan, softmax_ent
from sglang_client import OpenJevSGLang

MODEL = "openjev/qwen3.5-4b-nli-v5"
MARKER = re.compile(r"<<IMG\[([0-9]+)\]>>")


def make_request(image_paths):
    body = make_body(image_paths)
    encoded_images = [image["data"] for image in body["images"]]
    image_order = []

    def replace(match):
        index = int(match.group(1))
        if index >= len(encoded_images):
            raise ValueError(f"Image index {index} does not exist")
        image_order.append(index)
        # OpenJevSGLang replaces EVERY <<IMG>> with the author's image block.
        return "<<IMG>>"

    state = MARKER.sub(replace, body["state"])
    for index in range(len(encoded_images)):
        if index not in image_order:
            state += f"\nImage {index}: <<IMG>>"
            image_order.append(index)
    shared_images = [encoded_images[index] for index in image_order]
    plan = build_plan({"model": MODEL, "state": state, "questions": body["questions"]})
    # OUTER list: one entry per candidate judgment. INNER list: all images for
    # THAT judgment. A flat list here would give each judgment only one image.
    images_for_pairs = [shared_images for _ in plan.pairs]
    return plan, images_for_pairs


def check_endpoint(url):
    response = requests.get(url + "/health", timeout=(3, 5))
    response.raise_for_status()
    try:
        info = response.json()
    except ValueError:
        info = {}
    if isinstance(info, dict) and "worker_pid" in info and "max_images" in info:
        raise ValueError(
            f"{url} is the current custom service, not SGLang. "
            "OpenJevSGLang requires a real OpenJev SGLang /classify endpoint; use its host and port."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Actual OpenJev SGLang server IP/hostname")
    parser.add_argument("--port", type=int, default=30000, help="Actual SGLang port; the current custom 31000 is incompatible")
    parser.add_argument("--images", nargs=2, default=[str(ROOT / "examples/red.png"), str(ROOT / "examples/blue.png")])
    args = parser.parse_args()
    url = f"http://{args.host}:{args.port}"
    try:
        plan, images_for_pairs = make_request(args.images)
        check_endpoint(url)
        # Author's class, without modifying its implementation. Small batches
        # keep repeated multi-image uploads and vision memory bounded.
        client = OpenJevSGLang(url, bs=4, workers=1)
        start = time.perf_counter()
        logits = client.logits(plan.pairs, images=images_for_pairs)
        # Author's decision rule: P(entailment) for each candidate, normalized
        # across A/B. Raw three-class NLI probabilities are not choice answers.
        answers = assemble(plan, softmax_ent(logits))
        result = {
            "model": MODEL, "answers": answers,
            "probability_method": "normalized_entailment_v1",
            "elapsed_seconds": round(time.perf_counter() - start, 3),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, requests.RequestException) as exc:
        raise SystemExit(f"SGLang example failed: {exc}") from exc


if __name__ == "__main__":
    main()
