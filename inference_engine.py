"""Compatibility shim forwarding to openjev.inference_engine."""
from __future__ import annotations

import project_paths  # noqa: F401
from openjev.inference_engine import ImageNLI, infer, load, run_scores, score_pairs

__all__ = ["ImageNLI", "infer", "load", "run_scores", "score_pairs"]
