"""Public multi-image sample, shared by both backends."""
import base64
import mimetypes
from pathlib import Path


def make_body(image_paths):
    images = []
    for path in image_paths:
        image = Path(path)
        mime = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        images.append({"data": f"data:{mime};base64,{encoded}"})
    return {
        "state": "Image 0: <<IMG[0]>>\nImage 1: <<IMG[1]>>",
        "images": images,
        "questions": {
            "colors": {
                "type": "choice",
                "instructions": "What are the dominant colors of Image 0 and Image 1, in that order?",
                "criteria": {"A": "Red and blue", "B": "Blue and red", "C": "Both red", "D": "Both blue"},
            },
            "both_colors": {"type": "noul", "instructions": "Do these images include both red and blue?"},
        },
    }
