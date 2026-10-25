"""Keep HTTP responsive while the GPU runs in a cancellable child process."""
from __future__ import annotations

import asyncio
import ctypes
import logging
import multiprocessing
import os
import signal
import time
from pathlib import Path

import openjev.project_paths  # noqa: F401
from decisions_api import ApiError

ROOT = Path(__file__).resolve().parents[2]
WORKER_PID = ROOT / "runtime/model-worker.pid"
logger = logging.getLogger("uvicorn.error")


def model_worker_main(connection, parent_pid):
    # A killed HTTP parent must never leave a GPU model orphaned on the host.
    libc = ctypes.CDLL(None)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise RuntimeError("Unable to configure model worker parent-death signal")
    if os.getppid() != parent_pid:
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [model %(process)d] %(message)s")
    try:
        from openjev import inference_engine
        metadata = inference_engine.load()
        connection.send({"kind": "ready", "metadata": metadata})
        while True:
            shared = connection.recv()
            try:
                result = inference_engine.infer(shared)
                connection.send({"kind": "result", "result": result})
            except ApiError as exc:
                connection.send({"kind": "error", "code": exc.code, "message": exc.message})
            except Exception:
                logging.exception("Model calculation failed")
                connection.send({"kind": "fatal", "message": "Model calculation failed; worker is being recovered"})
                return
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        logging.exception("Model worker startup failed")
        try:
            connection.send({"kind": "fatal", "message": "Model worker startup failed; inspect logs/service.log"})
        except (EOFError, OSError):
            pass
    finally:
        connection.close()


class WorkerManager:
    def __init__(self):
        self.process = self.connection = None
        self.ready = False
        self.metadata = {}
        self.pending = 0
        self.closed = False
        self.current = None
        self.lock = asyncio.Lock()
        self.lifecycle_lock = asyncio.Lock()
        self.warm_task = None
        self.cleanup_tasks = set()
        self.request_timeout = float(os.environ.get("OPENJEV_REQUEST_TIMEOUT", "180"))
        self.queue_timeout = float(os.environ.get("OPENJEV_QUEUE_TIMEOUT", "30"))
        self.start_timeout = float(os.environ.get("OPENJEV_WORKER_START_TIMEOUT", "180"))
        self.max_pending = 4

    def healthy(self):
        return self.ready and self.process is not None and self.process.is_alive()

    async def receive(self, timeout, request=None):
        deadline = time.monotonic() + timeout
        while True:
            if request is not None and await request.is_disconnected():
                raise ApiError(499, "Client disconnected; its calculation was cancelled")
            if self.connection is not None and self.connection.poll():
                try:
                    return self.connection.recv()
                except (EOFError, OSError) as exc:
                    raise ApiError(503, "Model worker exited; recovering") from exc
            if self.process is None or not self.process.is_alive():
                raise ApiError(503, "Model worker exited; recovering")
            if time.monotonic() >= deadline:
                raise ApiError(504, f"Calculation exceeded {timeout:g}s; cancelled and recovering")
            await asyncio.sleep(0.05)

    async def discard(self, reason):
        self.ready = False
        process, connection = self.process, self.connection
        self.process = self.connection = None
        task = asyncio.create_task(self.finish_discard(process, connection, reason))
        self.cleanup_tasks.add(task)
        task.add_done_callback(self.cleanup_tasks.discard)
        await asyncio.shield(task)

    async def finish_discard(self, process, connection, reason):
        if connection is not None:
            connection.close()
        if process is not None:
            logger.warning("Discarding model worker PID %s: %s", process.pid, reason)
            if process.is_alive():
                process.terminate()
                await asyncio.to_thread(process.join, 2)
                if process.is_alive():
                    process.kill()
                    await asyncio.to_thread(process.join, 2)
            else:
                process.join(timeout=0)
            if process.is_alive():
                raise ApiError(503, "Model worker did not exit after kill; inspect GPU driver")
            try:
                if WORKER_PID.read_text().strip() == str(process.pid):
                    WORKER_PID.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            process.close()

    async def warm(self):
        async with self.lifecycle_lock:
            if self.closed:
                return
            if self.healthy():
                return
            await self.discard("loading/reloading")
            if self.cleanup_tasks:
                await asyncio.gather(*list(self.cleanup_tasks))
            context = multiprocessing.get_context("spawn")
            parent, child = context.Pipe()
            process = context.Process(target=model_worker_main, args=(child, os.getpid()), daemon=True)
            process.start()
            child.close()
            self.process, self.connection = process, parent
            WORKER_PID.parent.mkdir(exist_ok=True)
            WORKER_PID.write_text(str(process.pid))
            try:
                message = await self.receive(self.start_timeout)
                if message["kind"] != "ready":
                    raise ApiError(503, message.get("message", "Model startup failed"))
                self.metadata = message["metadata"]
                self.ready = True
                logger.info("Model worker PID %s is ready", process.pid)
            except BaseException:
                await self.discard("startup failed/cancelled")
                raise

    def schedule_warm(self):
        if self.closed:
            return
        if self.warm_task is None or self.warm_task.done():
            self.warm_task = asyncio.create_task(self.warm())
            self.warm_task.add_done_callback(self.warm_finished)

    @staticmethod
    def warm_finished(task):
        if not task.cancelled() and task.exception() is not None:
            logger.error("Model recovery failed: %s", task.exception())

    async def ensure_ready(self):
        if not self.healthy():
            self.schedule_warm()
            await asyncio.shield(self.warm_task)
        if not self.healthy():
            raise ApiError(503, "Model is unavailable; inspect service logs")

    @staticmethod
    async def watch(task, timeout, request, message):
        deadline = time.monotonic() + timeout
        try:
            while not task.done():
                if await request.is_disconnected():
                    raise ApiError(499, "Client disconnected; pending request cancelled")
                if time.monotonic() >= deadline:
                    raise ApiError(503, message)
                await asyncio.sleep(0.05)
            return await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def submit(self, shared, request):
        if self.pending >= self.max_pending:
            raise ApiError(503, "Service busy: at most 4 running/pending requests; retry later")
        self.pending += 1
        lock_task = asyncio.create_task(self.lock.acquire())
        active = False
        started = time.monotonic()
        try:
            await self.watch(lock_task, self.queue_timeout, request, f"Waited {self.queue_timeout:g}s for another request; retry later")
            ready_task = asyncio.create_task(self.ensure_ready())
            await self.watch(ready_task, self.start_timeout, request, "Model recovery exceeded startup limit")
            if await request.is_disconnected():
                raise ApiError(499, "Client disconnected; pending request cancelled")
            active = True
            self.current = {"started": time.monotonic(), "candidate_answers": len(shared.plan.pairs)}
            logger.info("Starting calculation: %s images, %s candidates, worker PID %s",
                        len(shared.images), len(shared.plan.pairs), self.process.pid)
            send_task = asyncio.create_task(asyncio.to_thread(self.connection.send, shared))
            await self.watch(send_task, 10, request, "Unable to send calculation to model worker")
            message = await self.receive(self.request_timeout, request)
            if message["kind"] == "fatal":
                raise ApiError(503, message["message"])
            active = False
            if message["kind"] == "error":
                raise ApiError(message["code"], message["message"])
            if message["kind"] != "result":
                raise ApiError(503, "Unexpected model response")
            logger.info("Calculation finished in %.3fs", time.monotonic() - started)
            return message["result"]
        except (ApiError, asyncio.CancelledError, EOFError, OSError) as exc:
            if active:
                await self.discard(str(exc) or "Request cancelled")
                self.schedule_warm()
            if isinstance(exc, (EOFError, OSError)):
                raise ApiError(503, "Model worker connection failed; recovering") from exc
            raise
        finally:
            self.pending -= 1
            if lock_task.done() and not lock_task.cancelled() and lock_task.exception() is None and lock_task.result():
                self.current = None
                self.lock.release()

    async def close(self):
        self.closed = True
        if self.warm_task is not None and not self.warm_task.done():
            self.warm_task.cancel()
            await asyncio.gather(self.warm_task, return_exceptions=True)
        await self.discard("service shutting down")
        if self.cleanup_tasks:
            await asyncio.gather(*list(self.cleanup_tasks), return_exceptions=True)
