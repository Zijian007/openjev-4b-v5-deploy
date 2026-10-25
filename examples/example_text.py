"""Send three questions over shared text, using requests directly."""
import argparse
import json

import requests

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1", help="Server IP or hostname")
parser.add_argument("--port", type=int, default=31000)
args = parser.parse_args()

body = {
    "state": "A man is playing a guitar.",
    "questions": {
        "activity": {
            "type": "choice",
            "instructions": "What is the man doing?",
            "criteria": {"A": "Making music", "B": "Swimming"},
        },
        "music": {"type": "noul", "instructions": "Is someone making music?"},
        "support": {
            "type": "score",
            "instructions": "How strongly does the state support that someone is making music?",
            "criteria": ["Contradicted", "Unknown", "Clearly supported"],
        },
    },
}
response = requests.post(f"http://{args.host}:{args.port}/v1/systemone", json=body, timeout=600)
if not response.ok:
    print(response.text)
response.raise_for_status()
print(json.dumps(response.json(), ensure_ascii=False, indent=2))
