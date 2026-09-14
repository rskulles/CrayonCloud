"""Z-Image-Turbo through Hugging Face diffusers: CUDA when there is a card, MPS on a Mac without mflux, CPU as a last resort."""

from __future__ import annotations

import gc
import logging
import time

from PIL import Image

from .base import GenerationRequest

log = logging.getLogger("crayoncloud.diffusers")

REPOS = {
    "z-image-turbo": ("Tongyi-MAI/Z-Image-Turbo", 8),
    "z-image": ("Tongyi-MAI/Z-Image", 30),
}


class DiffusersEngine:
    name = "diffusers"

    def __init__(self, model: str = "z-image-turbo") -> None:
        if model not in REPOS:
            raise ValueError(f"diffusers engine knows {', '.join(REPOS)}; not {model!r}")
        self.model = model
        self._pipe = None
        self._device = "cpu"

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def load(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from diffusers import ZImagePipeline

        repo, _ = REPOS[self.model]
        self._device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.bfloat16 if self._device != "cpu" else torch.float32
        started = time.monotonic()
        log.info("loading %s on %s; the first time this downloads the weights from Hugging Face", repo, self._device)
        self._pipe = ZImagePipeline.from_pretrained(repo, torch_dtype=dtype).to(self._device)
        log.info("loaded %s in %.0f s", repo, time.monotonic() - started)

    def generate(self, request: GenerationRequest) -> Image.Image:
        self.load()
        if request.loras:
            raise RuntimeError("LoRAs are supported by the mflux engine only, for now.")
        if request.init_image is not None:
            raise RuntimeError("Image to image is supported by the mflux engine only, for now.")
        import torch

        generator = torch.Generator(device="cpu").manual_seed(request.seed)
        kwargs = dict(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.steps,
            guidance_scale=0.0 if self.model == "z-image-turbo" else 4.0,
            generator=generator,
        )
        if request.negative_prompt and self.model != "z-image-turbo":
            kwargs["negative_prompt"] = request.negative_prompt
        return self._pipe(**kwargs).images[0].convert("RGB")

    def unload(self) -> None:
        if self._pipe is None:
            return
        self._pipe = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001 - best effort
            pass
        log.info("unloaded %s", self.model)
