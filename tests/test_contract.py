import base64
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
from input_schema import validate_request
from decisions_api import ApiError, assemble, softmax_ent
from example_multi_image_sglang import make_request
from multi_image_request import make_body


class SharedImageContractTests(unittest.TestCase):
    def setUp(self):
        self.paths = [ROOT / "examples/red.png", ROOT / "examples/blue.png"]

    def test_backends_have_same_pairs_and_image_order(self):
        shared = validate_request(make_body(self.paths))
        sglang_plan, images = make_request(self.paths)
        normalized = [(p.replace("<<IMG[0]>>", "<<IMG>>").replace("<<IMG[1]>>", "<<IMG>>"), h)
                      for p, h in shared.plan.pairs]
        self.assertEqual(normalized, sglang_plan.pairs)
        self.assertEqual(shared.order, [0, 1])
        self.assertEqual(len(images), len(shared.plan.pairs))
        for pair_images in images:
            self.assertEqual(len(pair_images), 2)
            for encoded, path in zip(pair_images, self.paths):
                self.assertTrue(encoded.startswith("data:image/png;base64,"))
                self.assertEqual(base64.b64decode(encoded.split(",", 1)[1]), path.read_bytes())

    def test_reference_order_repetition_and_missing_reference(self):
        body = make_body(self.paths)
        body["state"] = "<<IMG[1]>> <<IMG[0]>> <<IMG[1]>>"
        self.assertEqual(validate_request(body).order, [1, 0, 1])
        body["state"] = "<<IMG[2]>>"
        with self.assertRaises(ApiError):
            validate_request(body)
        body["state"] = ""
        self.assertEqual(validate_request(body).order, [0, 1])

    def test_invalid_fields_and_limit_rejected(self):
        body = make_body(self.paths)
        body["id"] = None
        with self.assertRaises(ApiError):
            validate_request(body)
        body.pop("id")
        body["images"] *= 3
        with self.assertRaises(ApiError):
            validate_request(body)

    def test_resize_keeps_proportions(self):
        image = io.BytesIO()
        Image.new("RGB", (1707, 1280), "red").save(image, format="PNG")
        body = make_body(self.paths)
        body["images"][0]["data"] = base64.b64encode(image.getvalue()).decode()
        shared = validate_request(body)
        self.assertEqual(shared.images[0].size, (1280, 960))

    def test_original_rules_return_all_question_types(self):
        body = {"state": "A man plays guitar.", "questions": {
            "choice": {"type": "choice", "instructions": "Activity?", "criteria": {"A": "music", "B": "swimming"}},
            "noul": {"type": "noul", "instructions": "Music?"},
            "score": {"type": "score", "instructions": "Support?", "criteria": ["no", "yes"]},
        }}
        plan = validate_request(body).plan
        result = assemble(plan, softmax_ent([[0, 4, 0], [4, 0, 0]] * 3))
        self.assertEqual(result["choice"]["choice"], "A")
        self.assertLess(result["noul"]["noul"], .1)
        self.assertLess(result["score"]["score"], .1)


if __name__ == "__main__":
    unittest.main()
