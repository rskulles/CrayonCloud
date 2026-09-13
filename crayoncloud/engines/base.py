from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 8
    seed: int = 0
    negative_prompt: str | None = None
    # (absolute path, scale) pairs, already resolved by the API from names in the LoRA folder. Order matters.
    loras: tuple[tuple[str, float], ...] = ()


class Engine(Protocol):
    """One loaded model. Implementations are not thread-safe; the service serialises every call onto one worker."""

    name: str
    model: str

    @property
    def loaded(self) -> bool: ...

    def load(self) -> None:
        """Downloads (first time) and loads the weights. Slow; called lazily before the first generation or on --preload."""

    def generate(self, request: GenerationRequest) -> Image.Image: ...

    def unload(self) -> None:
        """Frees the weights so the GPU or unified memory goes back to whatever else the machine is doing."""
