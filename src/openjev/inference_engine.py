"""OpenJev text/image service using the author's multimodal reference backbone."""
from __future__ import annotations

import faulthandler
import json
import logging
import os
import threading
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from transformers import AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5PreTrainedModel

import openjev.project_paths  # noqa: F401
from decisions_api import ApiError, assemble, softmax_ent
from openjev.input_schema import MARKER

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "model-overlay"
MODEL_NAME = "openjev/qwen3.5-4b-nli-v5"
MAX_TOKENS = 4096
lock = threading.Lock()
logger = logging.getLogger("uvicorn.error")
model = processor = None


def linear_patch_embed(self, hidden_states):
    # Each already-extracted patch is exactly one Conv3d receptive field.
    # Flattening the SAME weights and input gives the same dot product via
    # Tensor Core GEMM, avoiding the pathological large-batch Conv3d path.
    inputs = hidden_states.reshape(-1, self.in_channels * self.temporal_patch_size * self.patch_size ** 2)
    return torch.nn.functional.linear(
        inputs.to(dtype=self.proj.weight.dtype),
        self.proj.weight.reshape(self.embed_dim, -1), self.proj.bias)


class ImageNLI(Qwen3_5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_5Model(config)
        self.score = torch.nn.Linear(config.text_config.hidden_size, 3, bias=False)
        self.post_init()

    def forward(self, **inputs):
        hidden = self.model(**inputs, use_cache=False).last_hidden_state
        return self.score(hidden[:, -1])


def score_pairs(pairs, images, order):
    began = time.perf_counter()
    vis = None
    blocks = []
    if images:
        # Each occurrence, including repeated references, needs its own matching
        # grid and pixels. The order is the order in state, NOT the upload order.
        unique = [processor.image_processor(images=[im], return_tensors="pt") for im in images]
        counts = [int(v["image_grid_thw"].prod()) // processor.image_processor.merge_size ** 2 for v in unique]
        if sum(counts[index] for index in order) > MAX_TOKENS:
            raise ApiError(413, f"state/images: image references alone exceed {MAX_TOKENS} tokens; use smaller images or fewer references")
        blocks = ["<|vision_start|>" + "<|image_pad|>" * count + "<|vision_end|>" for count in counts]
        vis = {key: torch.cat([unique[index][key] for index in order], dim=0)
               for key in ("pixel_values", "image_grid_thw")}
    template = model.config.nli_template
    prepared = []
    # Check EVERY candidate before running any model inference. Never truncate
    # shared evidence or split it into windows that could lose an image marker.
    for premise, hypothesis in pairs:
        if images:
            premise = MARKER.sub(lambda match: blocks[int(match.group(1))], premise)
        inputs = processor.tokenizer(
            template.format(premise=premise.strip(), hypothesis=hypothesis.strip()),
            add_special_tokens=False, return_tensors="pt",
        )
        if inputs.input_ids.shape[1] > MAX_TOKENS:
            raise ApiError(413, f"state/questions/images: input exceeds {MAX_TOKENS} tokens; shorten text, use smaller images or fewer references")
        prepared.append(inputs)
    if vis is not None:
        vis = {key: value.to("cuda", dtype=torch.bfloat16) if key == "pixel_values" else value.cuda()
               for key, value in vis.items()}
    logits = []
    with torch.inference_mode():
        if vis is not None:
            image_features = model.model.get_image_features(
                vis["pixel_values"], vis["image_grid_thw"], return_dict=True)
            vis = {"image_grid_thw": vis["image_grid_thw"],
                   "mm_encoder_outputs": {"image": image_features}}
        logger.info("Prepared shared images/text in %.3fs", time.perf_counter() - began)
        for inputs in prepared:
            candidate_started = time.perf_counter()
            inputs = {k: v.cuda() for k, v in inputs.items()}
            if vis is not None:
                inputs.update(vis)
                inputs["mm_token_type_ids"] = (
                    inputs["input_ids"] == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                ).long()
            logits.append(model(**inputs).float().cpu().numpy()[0])
            logger.info("Candidate scored in %.3fs", time.perf_counter() - candidate_started)
    return np.asarray(logits)


def run_scores(pairs, images, order):
    with lock:
        try:
            return score_pairs(pairs, images, order)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise ApiError(503, "GPU memory insufficient; shorten text, use smaller images or retry later")


def load():
    global model, processor
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    model, loading = ImageNLI.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True, output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys")):
        raise RuntimeError(f"Checkpoint mismatch: {loading}")
    model = model.cuda().eval()
    patch_embed = model.model.visual.patch_embed
    patch_embed.forward = MethodType(linear_patch_embed, patch_embed)
    logger.info("Vision patches use the equivalent linear projection; shared image features are computed once per request")
    score_pairs([("A man is playing a guitar.", "Someone is making music.")], [], [])
    return {"model": MODEL_NAME, "gpu": torch.cuda.get_device_name(0),
            "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "max_tokens": MAX_TOKENS}


def infer(shared):
    faulthandler.dump_traceback_later(60, repeat=False)
    try:
        logits = run_scores(shared.plan.pairs, shared.images, shared.order)
        return {"model": MODEL_NAME, "answers": assemble(shared.plan, softmax_ent(logits)),
                "probability_method": "normalized_entailment_v1"}
    finally:
        faulthandler.cancel_dump_traceback_later()
