"""A stand-in engine for tests and demos: draws a colourful placeholder with the prompt written on it, instantly."""

from __future__ import annotations

import hashlib
import time

from PIL import Image, ImageDraw

from .base import GenerationRequest


class FakeEngine:
    name = "fake"

    def __init__(self, model: str = "fake", delay: float = 0.0) -> None:
        self.model = model
        self.delay = delay
        self._loaded = False
        self.calls: list[GenerationRequest] = []

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def generate(self, request: GenerationRequest) -> Image.Image:
        self.load()
        self.calls.append(request)
        if self.delay:
            time.sleep(self.delay)
        digest = hashlib.sha256(f"{request.prompt}|{request.seed}".encode()).digest()
        top = tuple(digest[0:3])
        bottom = tuple(digest[3:6])
        image = Image.new("RGB", (request.width, request.height))
        draw = ImageDraw.Draw(image)
        for y in range(request.height):
            t = y / max(1, request.height - 1)
            colour = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
            draw.line([(0, y), (request.width, y)], fill=colour)
        draw.text((16, 16), request.prompt[:80], fill=(255, 255, 255))
        return image

    def unload(self) -> None:
        self._loaded = False
