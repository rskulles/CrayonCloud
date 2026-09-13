"""Z-Image-Turbo on Apple Silicon through mflux (MLX)."""

from __future__ import annotations

import gc
import logging
import time

from PIL import Image

from .base import GenerationRequest

log = logging.getLogger("crayoncloud.mflux")

MODELS = {
    # alias -> (mflux model config factory name, default steps)
    "z-image-turbo": ("z_image_turbo", 8),
    "z-image": ("z_image", 30),
}


class MfluxEngine:
    name = "mflux"

    def __init__(self, model: str = "z-image-turbo", quantize: int | None = 8) -> None:
        if model not in MODELS:
            raise ValueError(f"mflux engine knows {', '.join(MODELS)}; not {model!r}")
        self.model = model
        self.quantize = quantize
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        # Imported here so the server starts (and the fake engine works) without MLX installed.
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.z_image import ZImage

        config_name, _ = MODELS[self.model]
        started = time.monotonic()
        log.info("loading %s (quantize=%s); the first time this downloads the weights from Hugging Face", self.model, self.quantize)
        self._model = ZImage(model_config=getattr(ModelConfig, config_name)(), quantize=self.quantize)
        log.info("loaded %s in %.0f s", self.model, time.monotonic() - started)

    def generate(self, request: GenerationRequest) -> Image.Image:
        self.load()
        result = self._model.generate_image(
            seed=request.seed,
            prompt=request.prompt,
            num_inference_steps=request.steps,
            width=request.width,
            height=request.height,
            negative_prompt=request.negative_prompt,
        )
        # mflux returns its GeneratedImage wrapper (with .image) or a bare PIL image depending on the version.
        image = getattr(result, "image", result)
        return image.convert("RGB")

    def unload(self) -> None:
        if self._model is None:
            return
        self._model = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # noqa: BLE001 - best effort
            pass
        log.info("unloaded %s", self.model)
