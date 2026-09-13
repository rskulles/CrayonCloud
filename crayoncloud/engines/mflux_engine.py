"""Z-Image-Turbo on Apple Silicon through mflux (MLX)."""

from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path

from PIL import Image

from .base import GenerationRequest

log = logging.getLogger("crayoncloud.mflux")

MODELS = {
    # alias -> (mflux model config factory name, default steps)
    "z-image-turbo": ("z_image_turbo", 8),
    "z-image": ("z_image", 30),
}

# Ready-made quantised copies in mflux's saved format, tried in order before falling back to the full-precision
# weights plus local quantisation. Each is (Hugging Face repo, commit); a pinned commit means a repo changed in place
# cannot break a first run. The first entry is the copy Crayon Cloud's author keeps; the second is the mflux
# community's. Existing installs never re-download, so a vanished repo only affects new ones, which then fall through.
PREBUILT: dict[tuple[str, int], list[tuple[str, str | None]]] = {
    ("z-image-turbo", 8): [
        ("rskulles/z-image-turbo-mflux-q8", "29e31091f95253909f749537adab0324393ffe85"),
        ("mflux-community/z-image-turbo-mflux-q8", "4430e72e37bf2bc7bc889a42d306ae1b8d3b22de"),
    ],
    ("z-image-turbo", 6): [("mflux-community/z-image-turbo-mflux-q6", None)],
    ("z-image-turbo", 5): [("mflux-community/z-image-turbo-mflux-q5", None)],
    ("z-image-turbo", 4): [("mflux-community/z-image-turbo-mflux-q4", None)],
    ("z-image-turbo", 3): [("mflux-community/z-image-turbo-mflux-q3", None)],
}


def prebuilt_sources(model: str, quantize: int | None) -> list[tuple[str, str | None]]:
    """The ready-made copies to try for this model and bit width, in order; empty when there are none."""
    return [] if quantize is None else list(PREBUILT.get((model, quantize), []))


class MfluxEngine:
    name = "mflux"

    def __init__(self, model: str = "z-image-turbo", quantize: int | None = 8, cache_dir: Path | None = None, prebuilt: bool = True) -> None:
        if model not in MODELS:
            raise ValueError(f"mflux engine knows {', '.join(MODELS)}; not {model!r}")
        self.model = model
        self.quantize = quantize
        self.prebuilt = prebuilt
        self.cache_dir = cache_dir or Path(os.environ.get("CRAYONCLOUD_CACHE", Path.home() / ".cache" / "crayoncloud"))
        self._model = None
        self.source: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def quantized_path(self) -> Path | None:
        """Where the quantised weights are kept between runs; None when running unquantised."""
        return None if self.quantize is None else self.cache_dir / f"{self.model}-q{self.quantize}"

    def load(self) -> None:
        if self._model is not None:
            return
        # Imported here so the server starts (and the fake engine works) without MLX installed.
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.z_image import ZImage

        config_name, _ = MODELS[self.model]
        model_config = getattr(ModelConfig, config_name)()
        started = time.monotonic()
        cached = self.quantized_path
        if cached is not None and (cached / "transformer").exists():
            # A copy this machine quantised itself (or a prebuilt one already fetched): loads in seconds.
            log.info("loading %s from the quantised copy at %s", self.model, cached)
            self._model = ZImage(model_config=model_config, model_path=str(cached))
            self.source = str(cached)
        elif (snapshot := self._fetch_prebuilt()) is not None:
            self._model = ZImage(model_config=model_config, model_path=snapshot)
        else:
            log.info("loading %s (quantize=%s); this downloads the full-precision weights from Hugging Face", self.model, self.quantize)
            self._model = ZImage(model_config=model_config, quantize=self.quantize)
            self.source = "Tongyi-MAI (quantised here)"
            if cached is not None:
                self._save_quantized(cached)
        log.info("loaded %s in %.0f s", self.model, time.monotonic() - started)

    def _fetch_prebuilt(self) -> str | None:
        """Downloads the first reachable ready-made quantised copy (pinned when a commit is given) and returns its folder."""
        if not self.prebuilt:
            return None
        for repo, revision in prebuilt_sources(self.model, self.quantize):
            try:
                from huggingface_hub import snapshot_download

                log.info("fetching the ready-made %d-bit copy from %s%s", self.quantize, repo, f" @ {revision[:10]}" if revision else "")
                path = snapshot_download(repo, revision=revision)
                if not (Path(path) / "transformer").exists():
                    raise FileNotFoundError("the repository does not hold an mflux model folder")
                self.source = f"{repo}@{revision[:10]}" if revision else repo
                return path
            except Exception as exc:  # noqa: BLE001 - a missing or unreachable repo just means "try the next one"
                log.warning("could not use %s: %s", repo, str(exc).splitlines()[0][:200])
        return None

    def _save_quantized(self, path: Path) -> None:
        try:
            from mflux.models.common.weights.saving.model_saver import ModelSaver
            from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition

            path.parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            ModelSaver.save_model(self._model, self.quantize, str(path), ZImageWeightDefinition)
            log.info("saved the quantised weights to %s in %.0f s; later starts load from there", path, time.monotonic() - started)
        except Exception as exc:  # noqa: BLE001 - the cache is a nicety, never a reason to fail a render
            log.warning("could not save the quantised weights to %s: %s", path, exc)

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
