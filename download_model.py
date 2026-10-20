"""Download the pinned checkpoint and prepare an overlay without changing weights."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent
REPOSITORY = "AlexWortega/openjev"
REVISION = "a20448012c213128955ca0c693e7c943865cab77"
SUBFOLDER = "qwen3.5-4b-nli-v5"
WEIGHT_SHA256 = "2b5f6e0691a5ed0c5d0f5f3e49bb26abe2d8efd7c14fda4431cd803c12b78f68"
PROCESSOR_REPOSITORY = "Qwen/Qwen3.5-4B"
PROCESSOR_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PROCESSOR_HASHES = {
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
}


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    snapshot_download(
        REPOSITORY, revision=REVISION, allow_patterns=[f"{SUBFOLDER}/*"],
        local_dir=ROOT / "upstream", max_workers=4,
    )
    source = ROOT / "upstream" / SUBFOLDER
    if checksum(source / "model.safetensors") != WEIGHT_SHA256:
        raise RuntimeError("Model checksum mismatch; do not start the service")
    overlay = ROOT / "model-overlay"
    overlay.mkdir(exist_ok=True)
    for path in source.iterdir():
        if not path.is_file():
            continue
        target = overlay / path.name
        if target.exists() or target.is_symlink():
            if target.resolve() != path.resolve():
                raise FileExistsError(f"Refusing to overwrite {target}")
        else:
            target.symlink_to(Path("..") / "upstream" / SUBFOLDER / path.name)
    for name, expected in PROCESSOR_HASHES.items():
        target = Path(hf_hub_download(
            PROCESSOR_REPOSITORY, name, revision=PROCESSOR_REVISION, local_dir=overlay,
        ))
        if checksum(target) != expected:
            raise RuntimeError(f"Processor checksum mismatch: {name}")
    metadata = {
        "repository": REPOSITORY, "revision": REVISION, "subfolder": SUBFOLDER,
        "weight_sha256": WEIGHT_SHA256, "processor_repository": PROCESSOR_REPOSITORY,
        "processor_revision": PROCESSOR_REVISION,
    }
    (ROOT / "MODEL_SOURCE.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Model and processor checksums verified; ready in model-overlay/", flush=True)


if __name__ == "__main__":
    main()
