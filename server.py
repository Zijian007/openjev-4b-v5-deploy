"""Compatibility shim forwarding to openjev.server."""
from __future__ import annotations

import project_paths  # noqa: F401
from openjev.server import app, health, systemone

__all__ = ["app", "health", "systemone"]
