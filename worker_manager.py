"""Compatibility shim forwarding to openjev.worker_manager."""
from __future__ import annotations

import project_paths  # noqa: F401
from openjev.worker_manager import WorkerManager, model_worker_main

__all__ = ["WorkerManager", "model_worker_main"]
