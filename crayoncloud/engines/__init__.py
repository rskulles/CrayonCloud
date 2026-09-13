"""Image engines. Each one loads a model once and turns a prompt into a PIL image; the service decides when."""

from __future__ import annotations

import platform

from .base import Engine, GenerationRequest


def pick_engine(name: str, model: str, quantize: int | None, prebuilt: bool = True) -> Engine:
    """auto picks MLX on Apple Silicon and diffusers elsewhere; fake draws a placeholder for tests and demos."""
    if name == "auto":
        name = "mflux" if platform.system() == "Darwin" and platform.machine() == "arm64" else "diffusers"
    if name == "mflux":
        from .mflux_engine import MfluxEngine

        return MfluxEngine(model=model, quantize=quantize, prebuilt=prebuilt)
    if name == "diffusers":
        from .diffusers_engine import DiffusersEngine

        return DiffusersEngine(model=model)
    if name == "fake":
        from .fake import FakeEngine

        return FakeEngine(model=model)
    raise ValueError(f"unknown engine {name!r}; use auto, mflux, diffusers or fake")


__all__ = ["Engine", "GenerationRequest", "pick_engine"]
