"""Strict shared-state request contract; independent of the GPU runtime."""
from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass

from PIL import Image

import openjev.project_paths  # noqa: F401
from decisions_api import HYPOTHESIS, ApiError, Plan, QPlan, _options
from image_inputs import image_input

MAX_IMAGES = 4
MAX_QUESTIONS = 64
MAX_PAIRS = 128
MAX_STATE_CHARS = 110_000
MAX_BODY_BYTES = 24 * 1024 * 1024
MARKER = re.compile(r"<<IMG\[([0-9]+)\]>>")


@dataclass
class SharedRequest:
    state: str
    images: list
    order: list
    plan: Plan


def fail(path, message, code=400):
    raise ApiError(code, f"{path}: {message}")


def fields(value, allowed, path):
    if not isinstance(value, dict):
        fail(path, "must be a JSON object")
    extra = set(value) - allowed
    if extra:
        fail(path, "unknown field(s): " + ", ".join(repr(key) for key in sorted(extra)))


def text(value, path, allow_empty=False, markers=False):
    if not isinstance(value, str):
        fail(path, "must be a string")
    try:
        value.encode("utf-8")
    except UnicodeError:
        fail(path, "must contain valid Unicode text")
    if not allow_empty and not value.strip():
        fail(path, "must not be empty")
    if "<|" in value:
        fail(path, "model control tokens are not allowed")
    if not markers and "<<IMG" in value:
        fail(path, "image markers are allowed only in state; refer to images by their index in questions")
    return value.strip()


def validate_request(body):
    fields(body, {"state", "images", "questions"}, "body")
    state = text(body.get("state"), "state", allow_empty=True, markers=True)
    if len(state) > MAX_STATE_CHARS:
        fail("state", f"exceeds {MAX_STATE_CHARS} characters", 413)
    supplied = body.get("images", [])
    if not isinstance(supplied, list):
        fail("images", "must be an array; omit it or use [] for text-only requests")
    if len(supplied) > MAX_IMAGES:
        fail("images", f"at most {MAX_IMAGES} images are allowed")
    if not state and not supplied:
        fail("state", "must not be empty when images are absent")

    # Validate references before decoding images. Repeated references are intentional.
    order = []
    for match in MARKER.finditer(state):
        digits = match.group(1).lstrip("0") or "0"
        if len(digits) > 1 or int(digits) >= len(supplied):
            fail("state", f"{match.group(0)} references an image that does not exist")
        order.append(int(digits))
    if "<<IMG" in MARKER.sub("", state):
        fail("state", "invalid image marker; use <<IMG[0]>>, <<IMG[1]>>, etc.")
    for index in range(len(supplied)):
        if index not in order:
            state += f"\nImage {index}: <<IMG[{index}]>>"
            order.append(index)

    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        fail("questions", "must be a non-empty object keyed by question name")
    if len(questions) > MAX_QUESTIONS:
        fail("questions", f"at most {MAX_QUESTIONS} questions are allowed")
    pairs, plans = [], []
    for qid, question in questions.items():
        text(qid, "questions question name")
        path = f"questions.{qid}"
        fields(question, {"type", "instructions", "criteria"}, path)
        qtype = question.get("type")
        if qtype not in ("choice", "noul", "score"):
            fail(path + ".type", "must be choice, noul or score")
        instruction = text(question.get("instructions"), path + ".instructions")
        criteria = question.get("criteria")
        if qtype == "choice":
            if not isinstance(criteria, dict) or not criteria:
                fail(path + ".criteria", "must be a non-empty object of option names and string descriptions")
            for key, value in criteria.items():
                text(key, path + ".criteria option name")
                text(value, path + f".criteria.{key}")
        elif qtype == "score":
            if not isinstance(criteria, list) or not criteria:
                fail(path + ".criteria", "must be a non-empty array of string descriptions, from low to high")
            for index, value in enumerate(criteria):
                text(value, path + f".criteria[{index}]")
        elif "criteria" in question:
            fields(criteria, {"false", "true"}, path + ".criteria")
            for key in ("false", "true"):
                text(criteria.get(key), path + f".criteria.{key}")
        keys, labels, descriptions = _options(qid, question)
        if len(pairs) + len(keys) > MAX_PAIRS:
            fail("questions", f"at most {MAX_PAIRS} candidate answers in total per request")
        start = len(pairs)
        for label, description in zip(labels, descriptions):
            pairs.append((state, HYPOTHESIS.format(instr=instruction, label=label, crit=description)))
        legend = dict(zip(keys, descriptions)) if qtype == "score" else {}
        plans.append(QPlan(qid, qtype, keys, 1, start, legend))

    images = []
    for index, value in enumerate(supplied):
        path = f"images[{index}]"
        fields(value, {"data"}, path)
        try:
            encoded = image_input({"image_data": value.get("data")})
            if encoded is None:
                fail(path + ".data", "is required and must contain a base64 image")
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as im:
                im.load()
                rgb = im.convert("RGB")
                rgb.thumbnail((1280, 1280))
                images.append(rgb)
        except ApiError as exc:
            if exc.message.startswith(path):
                raise
            fail(path + ".data", exc.message.replace("image_data", "data"), exc.code)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            fail(path + ".data", "image could not be fully decoded")
    return SharedRequest(state, images, order, Plan(pairs, plans))
