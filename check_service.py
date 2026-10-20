"""Live contract and joint-image checks; all calls use requests directly."""
import argparse
import base64
import copy
import io
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image


def validate_answer(answer):
    if answer["type"] == "noul":
        assert math.isfinite(answer["noul"]) and 0 <= answer["noul"] <= 1, answer
        return
    probabilities = answer["probabilities"]
    assert all(math.isfinite(x) and 0 <= x <= 1 for x in probabilities.values()), answer
    assert abs(sum(probabilities.values()) - 1) < 0.001, answer
    assert 0 <= answer["confidence"] <= 1, answer
    if answer["type"] == "choice":
        assert answer["choice"] in probabilities, answer
    else:
        assert 0 <= answer["score"] <= len(probabilities) - 1, answer
        assert set(answer["legend"]) == set(probabilities), answer


def image(color, size=(320, 240), format="PNG"):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format=format)
    return {"data": base64.b64encode(output.getvalue()).decode("ascii")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--out", default="verification.json")
    args = parser.parse_args()
    url = f"http://{args.host}:{args.port}"
    endpoint = url + "/v1/systemone"
    health = requests.get(url + "/health", timeout=10)
    health.raise_for_status()
    results = {"health": health.json(), "checks": []}

    def record(name, result, start):
        item = {"name": name, "seconds": round(time.perf_counter() - start, 3), "result": result}
        results["checks"].append(item)
        print("PASS", name, json.dumps(result, ensure_ascii=False), flush=True)

    def ok(name, body, expectations=None):
        start = time.perf_counter()
        response = requests.post(endpoint, json=body, timeout=600)
        assert response.status_code == 200, (name, response.status_code, response.text)
        result = response.json()
        assert set(result["answers"]) == set(body["questions"]), result
        for qid, answer in result["answers"].items():
            assert answer["type"] == body["questions"][qid]["type"], result
            validate_answer(answer)
        for qid, expected in (expectations or {}).items():
            answer = result["answers"][qid]
            if isinstance(expected, str):
                assert answer["choice"] == expected, (name, result)
            else:
                assert (answer["noul"] > 0.5) == expected, (name, result)
        record(name, result, start)
        return result

    def bad(name, body, code=400, path=None, raw=False):
        start = time.perf_counter()
        if raw:
            response = requests.post(endpoint, data=body, headers={"Content-Type": "application/json"}, timeout=60)
        else:
            response = requests.post(endpoint, json=body, timeout=60)
        assert response.status_code == code, (name, response.status_code, response.text)
        result = response.json()
        assert result["error"]["code"] == code, result
        if path:
            assert path in result["error"]["message"], result
        record(name, result, start)

    text = {
        "state": "A man is playing a guitar.",
        "questions": {
            "activity": {"type": "choice", "instructions": "What is the man doing?", "criteria": {"A": "Making music", "B": "Swimming"}},
            "music": {"type": "noul", "instructions": "Is someone making music?"},
            "swimming": {"type": "noul", "instructions": "Is the man swimming?"},
            "support": {"type": "score", "instructions": "How strongly does the state support that someone is making music?", "criteria": ["Contradicted", "Unknown", "Clearly supported"]},
        },
    }
    ok("text: all three question types, shared state", text, {"activity": "A", "music": True, "swimming": False})
    empty_list = dict(text, images=[])
    ok("text: explicit empty images", empty_list, {"activity": "A"})
    custom = {"state": text["state"], "questions": {"custom": {"type": "noul", "instructions": "Is someone making music?", "criteria": {"true": "Someone is making music", "false": "Nobody is making music"}}}}
    ok("custom noul criteria", custom, {"custom": True})
    red, blue = image("red"), image("blue", size=(192, 320))
    color_question = {"type": "choice", "instructions": "What is the dominant color in Image 0?", "criteria": {"red": "Red", "blue": "Blue"}}
    for name, supplied, expected in [("red", red, "red"), ("blue", blue, "blue")]:
        ok("single image " + name, {"state": "Image 0: <<IMG[0]>>", "images": [supplied], "questions": {"color": color_question}}, {"color": expected})
    ok("empty state with image auto-attached", {"state": "", "images": [red], "questions": {"color": color_question}}, {"color": "red"})
    uri = {"data": "data:image/png;base64," + red["data"]}
    ok("image data URI", {"state": "", "images": [uri], "questions": {"color": color_question}}, {"color": "red"})

    joint_question = {
        "type": "choice", "instructions": "What are the dominant colors of Image 0 and Image 1, in that order?",
        "criteria": {
            "red_blue": "Image 0 is red and Image 1 is blue", "blue_red": "Image 0 is blue and Image 1 is red",
            "both_red": "Both images are red", "both_blue": "Both images are blue",
        },
    }
    joint = {"state": "Image 0: <<IMG[0]>>\nImage 1: <<IMG[1]>>", "images": [red, blue], "questions": {"colors": joint_question}}
    for name, supplied, expected in [
        ("red-blue", [red, blue], "red_blue"), ("blue-red", [blue, red], "blue_red"),
        ("replace second: red-red", [red, red], "both_red"), ("replace first: blue-blue", [blue, blue], "both_blue"),
    ]:
        ok("joint two-image " + name, dict(joint, images=supplied), {"colors": expected})
    ok("markers reversed, image indices retained", dict(joint, state="Image 1: <<IMG[1]>>\nImage 0: <<IMG[0]>>"), {"colors": "red_blue"})
    ok("repeated image reference", dict(joint, state=joint["state"] + "\nImage 0 again: <<IMG[0]>>"), {"colors": "red_blue"})
    ok("unreferenced second image auto-attached", dict(joint, state="Image 0: <<IMG[0]>>"), {"colors": "red_blue"})
    ok("all images auto-attached", dict(joint, state=""), {"colors": "red_blue"})
    mixed = copy.deepcopy(joint)
    mixed["questions"]["both_colors"] = {"type": "noul", "instructions": "Do these images include both red and blue?"}
    mixed["questions"]["blue_support"] = {"type": "score", "instructions": "How strongly does Image 1 support that its dominant color is blue?", "criteria": ["Contradicted", "Unknown", "Clearly supported"]}
    ok("two images shared by mixed question types", mixed, {"colors": "red_blue", "both_colors": True})
    four = {
        "state": "",
        "images": [red, red, blue, blue],
        "questions": {"four": {
            "type": "choice", "instructions": "What are the dominant colors of Images 0, 1, 2 and 3?",
            "criteria": {"correct": "Images 0 and 1 are red; Images 2 and 3 are blue", "wrong": "Images 0 and 1 are blue; Images 2 and 3 are red"},
        }},
    }
    ok("four images jointly referenced", four, {"four": "correct"})

    minimal = {"state": "A man is playing a guitar.", "questions": {"q": color_question}}
    invalid_cases = [
        ("body array", [], 400, "body"),
        ("missing state", {"questions": minimal["questions"]}, 400, "state"),
        ("empty text-only state", dict(minimal, state="  "), 400, "state"),
        ("non-string state", dict(minimal, state={}), 400, "state"),
        ("unknown root field", dict(minimal, model="old"), 400, "body"),
        ("missing questions", {"state": "test"}, 400, "questions"),
        ("empty questions", dict(minimal, questions={}), 400, "questions"),
        ("question array", dict(minimal, questions=[]), 400, "questions"),
        ("empty question name", dict(minimal, questions={"": color_question}), 400, "questions"),
        ("question not object", dict(minimal, questions={"q": "test"}), 400, "questions.q"),
        ("unknown type", dict(minimal, questions={"q": {"type": "classify", "instructions": "test"}}), 400, "questions.q.type"),
        ("missing instructions", dict(minimal, questions={"q": {"type": "noul"}}), 400, "questions.q.instructions"),
        ("blank instructions", dict(minimal, questions={"q": {"type": "noul", "instructions": " "}}), 400, "questions.q.instructions"),
        ("unknown question field", dict(minimal, questions={"q": dict(color_question, typo=True)}), 400, "questions.q"),
        ("choice wrong criteria type", dict(minimal, questions={"q": dict(color_question, criteria=["a"])}), 400, "criteria"),
        ("choice non-string description", dict(minimal, questions={"q": dict(color_question, criteria={"a": 1})}), 400, "criteria"),
        ("score wrong criteria type", dict(minimal, questions={"q": {"type": "score", "instructions": "test", "criteria": {"a": "test"}}}), 400, "criteria"),
        ("score empty level", dict(minimal, questions={"q": {"type": "score", "instructions": "test", "criteria": [""]}}), 400, "criteria"),
        ("noul null criteria", dict(minimal, questions={"q": {"type": "noul", "instructions": "test", "criteria": None}}), 400, "criteria"),
        ("noul incomplete criteria", dict(minimal, questions={"q": {"type": "noul", "instructions": "test", "criteria": {"true": "yes"}}}), 400, "criteria.false"),
        ("images not array", dict(minimal, images="path.png"), 400, "images"),
        ("images null", dict(minimal, images=None), 400, "images"),
        ("five images", dict(minimal, images=[red] * 5), 400, "images"),
        ("image object required", dict(minimal, images=[red["data"]]), 400, "images[0]"),
        ("image id forbidden", dict(minimal, images=[dict(red, id="front")]), 400, "images[0]"),
        ("image missing data", dict(minimal, images=[{}]), 400, "images[0].data"),
        ("invalid base64", dict(minimal, images=[{"data": "invalid!!!"}]), 400, "images[0].data"),
        ("GIF rejected", dict(minimal, images=[image("red", format="GIF")]), 400, "images[0].data"),
        ("image too many pixels", dict(minimal, images=[image("red", size=(4000, 3001))]), 413, "images[0].data"),
        ("image over 4 MiB", dict(minimal, images=[{"data": "A" * (4 * ((4 * 1024 * 1024 + 2) // 3) + 4)}]), 413, "images[0].data"),
        ("missing image index", dict(minimal, state="<<IMG[1]>>", images=[red]), 400, "state"),
        ("old marker rejected", dict(minimal, state="<<IMG>>", images=[red]), 400, "state"),
        ("negative index rejected", dict(minimal, state="<<IMG[-1]>>", images=[red]), 400, "state"),
        ("huge index rejected", dict(minimal, state="<<IMG[" + "9" * 5000 + "]>>", images=[red]), 400, "state"),
        ("image marker in question", dict(minimal, questions={"q": dict(color_question, instructions="<<IMG[0]>>")}), 400, "instructions"),
        ("model control token", dict(minimal, state="<|image_pad|>"), 400, "state"),
        ("too many questions", dict(minimal, questions={str(i): {"type": "noul", "instructions": "test"} for i in range(65)}), 400, "questions"),
        ("too many options", dict(minimal, questions={"q": dict(color_question, criteria={str(i): "test" for i in range(65)})}), 400, "criteria"),
        ("too many total candidates", dict(minimal, questions={str(i): {"type": "score", "instructions": "test", "criteria": ["low", "mid", "high"]} for i in range(43)}), 400, "questions"),
        ("overlong total input", dict(minimal, state="hello " * 5000), 413, "state/questions/images"),
        ("too many repeat tokens", dict(minimal, state="<<IMG[0]>>" * 100, images=[red]), 413, "state/images"),
        ("invalid second question rejects entire body", dict(minimal, questions={"valid": color_question, "invalid": {"type": "wrong"}}), 400, "questions.invalid.type"),
    ]
    for name, body, code, path in invalid_cases:
        bad(name, body, code, path)
    bad("malformed JSON", '{"state":', raw=True)
    bad("duplicate JSON question names", '{"state":"test","questions":{"q":{"type":"noul","instructions":"test"},"q":{"type":"noul","instructions":"test"}}}', raw=True)
    bad("non-JSON number", '{"state":NaN,"questions":{}}', raw=True)
    response = requests.post(endpoint, data="{}", timeout=30)
    assert response.status_code == 400 and "Content-Type" in response.text, response.text
    record("missing JSON content type", response.json(), time.perf_counter())
    response = requests.post(url + "/classify", json={}, timeout=30)
    assert response.status_code == 404, response.text
    record("old classify endpoint removed", {"status": 404}, time.perf_counter())
    # Boundary errors must leave the GPU service healthy.
    ok("recovery after invalid and oversized requests", joint, {"colors": "red_blue"})
    other = copy.deepcopy(text)
    other["state"] = "A man is swimming in a pool."
    other["questions"] = {"activity": text["questions"]["activity"]}
    first = {"state": text["state"], "questions": other["questions"]}
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [executor.submit(ok, "concurrent guitar", first, {"activity": "A"}),
                executor.submit(ok, "concurrent swimming", other, {"activity": "B"})]
        for job in jobs:
            job.result()
    results["passed"] = True
    results["check_count"] = len(results["checks"])
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ALL {results['check_count']} CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
