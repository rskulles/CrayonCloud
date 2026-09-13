"""Owns the engine: one generation at a time on one worker thread, lazy load, idle unload, and a status readout."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from PIL import Image

from .engines import Engine, GenerationRequest

log = logging.getLogger("crayoncloud.service")


@dataclass
class Status:
    engine: str
    model: str
    loaded: bool
    busy: bool
    queued: int
    seconds_busy: float | None
    generated: int
    last_seconds: float | None
    idle_unload_seconds: float | None


class ImageService:
    def __init__(self, engine: Engine, idle_unload_seconds: float | None = 30 * 60) -> None:
        self.engine = engine
        self.idle_unload_seconds = idle_unload_seconds
        # MLX and torch pipelines want to stay on one thread; a single worker gives every call the same one.
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="crayoncloud-gen")
        self._lock = asyncio.Lock()
        self._state = threading.Lock()
        self._busy_since: float | None = None
        self._queued = 0
        self._generated = 0
        self._last_seconds: float | None = None
        self._last_used = time.monotonic()

    async def generate(self, request: GenerationRequest) -> tuple[Image.Image, float]:
        """Returns the image and how long it took. Callers wait in line; one request renders at a time."""
        with self._state:
            self._queued += 1
        try:
            async with self._lock:
                with self._state:
                    self._queued -= 1
                    self._busy_since = time.monotonic()
                loop = asyncio.get_running_loop()
                started = time.monotonic()
                try:
                    image = await loop.run_in_executor(self._worker, self.engine.generate, request)
                finally:
                    seconds = time.monotonic() - started
                    with self._state:
                        self._busy_since = None
                        self._last_used = time.monotonic()
                self._generated += 1
                self._last_seconds = seconds
                log.info("generated %dx%d in %.1f s (seed %d, %d steps)", request.width, request.height, seconds, request.seed, request.steps)
                return image, seconds
        except BaseException:
            with self._state:
                if self._busy_since is None and self._queued > 0:
                    self._queued -= 1
            raise

    async def preload(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._worker, self.engine.load)
            self._last_used = time.monotonic()

    async def unload(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._worker, self.engine.unload)

    async def idle_watch(self, poll_seconds: float = 30.0) -> None:
        """Background task: drop the weights after the idle timeout so the memory goes back to the rest of the machine."""
        if self.idle_unload_seconds is None:
            return
        while True:
            await asyncio.sleep(poll_seconds)
            if self.engine.loaded and self._busy_since is None and time.monotonic() - self._last_used > self.idle_unload_seconds:
                log.info("idle for %.0f s; unloading", self.idle_unload_seconds)
                await self.unload()

    def status(self) -> Status:
        with self._state:
            busy_since = self._busy_since
            queued = self._queued
        return Status(
            engine=self.engine.name,
            model=self.engine.model,
            loaded=self.engine.loaded,
            busy=busy_since is not None,
            queued=queued,
            seconds_busy=None if busy_since is None else round(time.monotonic() - busy_since, 1),
            generated=self._generated,
            last_seconds=None if self._last_seconds is None else round(self._last_seconds, 1),
            idle_unload_seconds=self.idle_unload_seconds,
        )

    def close(self) -> None:
        self._worker.shutdown(wait=False, cancel_futures=True)
