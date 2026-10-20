"""Responsive HTTP service with an isolated, cancellable GPU worker."""
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

import project_paths  # noqa: F401
from decisions_api import ApiError
from input_schema import MAX_BODY_BYTES, MAX_IMAGES, MAX_PAIRS, MAX_QUESTIONS, validate_request
from worker_manager import WorkerManager

manager = None


@asynccontextmanager
async def lifespan(app):
    global manager
    manager = WorkerManager()
    try:
        await manager.ensure_ready()
        yield
    finally:
        await manager.close()


app = FastAPI(title="OpenJev 4B v5", lifespan=lifespan)


@app.exception_handler(ApiError)
async def api_error(request, exc):
    return JSONResponse(exc.body(), status_code=exc.code)


@app.get("/health")
async def health():
    healthy = manager is not None and manager.healthy()
    result = dict(manager.metadata) if manager is not None else {}
    result.update({"status": "ok" if healthy else "recovering",
                   "max_images": MAX_IMAGES, "max_questions": MAX_QUESTIONS,
                   "max_candidate_answers": MAX_PAIRS})
    if manager is not None:
        result.update({"worker_pid": manager.process.pid if manager.process is not None else None,
                       "pending_requests": manager.pending,
                       "request_timeout_seconds": manager.request_timeout,
                       "queue_timeout_seconds": manager.queue_timeout})
        if manager.current is not None:
            import time
            result["active_request_seconds"] = round(time.monotonic() - manager.current["started"], 3)
    return JSONResponse(result, status_code=200 if healthy else 503)


def no_duplicate_fields(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ApiError(400, f"body: duplicate JSON field {key!r}")
        result[key] = value
    return result


def invalid_constant(value):
    raise ApiError(400, f"body: {value} is not valid JSON")


@app.post("/v1/systemone")
async def systemone(request: Request):
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise ApiError(400, "body: Content-Type must be application/json; use requests.post(..., json=body)")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            raise ApiError(413, "body: request exceeds 24 MiB")
        chunks.append(chunk)
    try:
        body = json.loads(b"".join(chunks), object_pairs_hook=no_duplicate_fields, parse_constant=invalid_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ApiError(400, "body: malformed JSON") from exc
    shared = await run_in_threadpool(validate_request, body)
    return await manager.submit(shared, request)
