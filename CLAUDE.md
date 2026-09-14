# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Crayon Cloud is a small self-hosted image generation server for Z-Image-Turbo that answers the OpenAI images API
shape (`POST /v1/images/generations`, `b64_json` only). Its primary client is ButterKnife. It ships two ways: a pip
package with a `crayoncloud` CLI, and a macOS menu bar app that bundles its own CPython and installs the package on
first launch.

## Commands

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[mlx,dev]"          # Apple Silicon; use [cuda,dev] elsewhere; [dev] alone is enough for tests
pytest                               # whole suite (tests/test_api.py), runs in well under a second
pytest tests/test_api.py -k loras    # one test by name substring
crayoncloud serve --engine fake      # the API without a model, for development; try-it page at http://127.0.0.1:8765/
crayoncloud generate "prompt" --size 1024x1024 --seed 7   # one-off render from the CLI
tools/make-macos-app.sh 0.2.1 dist [--dmg]                 # build "Crayon Cloud.app" locally (needs Xcode CLT)
```

There is no linter or formatter configured. Python is 3.10 to 3.13 (`mflux` pins < 3.14); `.venv` here is 3.12.
The tests never touch a real model: they use `FakeEngine` and `fastapi.testclient`, so they run anywhere with the
`dev` extra only. CI (`.github/workflows/release.yml`) runs `pytest` and then builds, signs, notarizes the app on a
`v*` tag.

## Architecture

Request flow: `cli.py` builds an engine via `engines.pick_engine`, wraps it in `service.ImageService`, and hands
that to `api.create_app`. The layers are deliberately separated:

- **`engines/`**: one loaded model each, behind the `Engine` protocol in `base.py` (`load`, `generate`, `unload`,
  `loaded`). Engines are not thread-safe. Heavy imports (`mflux`, `torch`, `diffusers`) happen inside `load`, never at
  module top level, so the server and the fake engine work without them installed. `auto` picks `mflux` on
  Darwin/arm64 and `diffusers` elsewhere.
- **`service.py`**: `ImageService` owns the engine. Every engine call runs on a single-worker `ThreadPoolExecutor`
  (MLX and torch want one thread) behind an `asyncio.Lock`, so requests queue and one image renders at a time. It
  also does lazy load on first request, the idle-unload background task, and the `Status` readout for `/health`.
- **`api.py`**: FastAPI app. Two render routes share one `render()` closure: `/v1/images/generations` (JSON, with
  an optional base64 `image`) and `/v1/images/edits` (OpenAI's multipart shape, used for image to image). It rounds
  sizes to multiples of 16 between 256 and 2048 (`fit_size` derives a size from a source picture when none is
  given), resolves LoRA names to paths, saves PNGs with prompt/seed/steps/source/strength in the text chunks
  (`save_asset`), and serves the try-it HTML page inline from `index()`. `create_app` takes everything injected,
  which is how tests build an app around a `FakeEngine` with a `tmp_path` assets dir.
- **`loras.py`**: `LoraLibrary` maps a folder of `.safetensors` files to names (file stem, case-insensitive).
  `resolve` refuses anything that looks like a path; only files inside the folder count.

### Things that are easy to get wrong

- **The startup line is a contract.** `cli.py` prints `Crayon Cloud <version> (... ) is running at <url> (on your
  network: <url>)`. The Swift menu bar helper (`packaging/macos/CrayonCloudMenu/main.swift`, `parse()`) string-matches
  ` is running at ` and `(on your network: ` to enable its Open and Copy-address items. Keep that shape.
- **Version lives in two places** that must match: `pyproject.toml` and `crayoncloud/__init__.py`. The Mac app
  reinstalls its venv when the bundle version differs from `~/Library/Application Support/CrayonCloud/venv/.crayoncloud-version`,
  so a release needs a bump for the app to pick up package changes. The README's build example also names the version.
- **LoRAs live on the engine, not the request path alone.** `MfluxEngine.load(loras)` unloads and reloads when the
  requested set differs from what is loaded; `GenerationRequest.loras` carries already-resolved `(abs_path, scale)`
  pairs. Only the mflux engine supports them; `DiffusersEngine.generate` raises if any are passed.
- **Prebuilt weights are pinned.** `mflux_engine.PREBUILT` lists Hugging Face repos with commit hashes, tried in
  order before falling back to quantising the full Tongyi-MAI weights locally (cached under `CRAYONCLOUD_CACHE`,
  default `~/.cache/crayoncloud`). `--quantize-locally` skips the prebuilt list. A test asserts the order and that
  only known bit widths have entries.
- **The macOS app has no Xcode project.** `tools/make-macos-app.sh` compiles the single Swift file with `swiftc`,
  copies the package source into `Contents/Resources/crayoncloud-src`, unpacks a pinned python-build-standalone
  CPython (bump `PBS_TAG` and `PBS_PYTHON` together), and ad-hoc signs every Mach-O. The helper launches
  `crayoncloud serve --port 8765 --parent-pid <its pid> [--lan]`; the server watches that pid and exits when it goes.
- **`strength` is A1111's, mflux's is the opposite.** `GenerationRequest.strength` is denoising strength (0 keeps the
  source, 1 ignores it). mflux's `image_strength` is how much of the source to keep, so `MfluxEngine.generate` passes
  `1 - strength`. With 8 steps only eight distinct values exist. `init_image` is a PIL image; the mflux engine writes
  it to a temp file because mflux wants a path. Image to image is mflux only; the diffusers engine raises.
- **Z-Image-Turbo ignores `negative_prompt`** (guidance-distilled, `guidance_scale=0`); the API accepts the field
  anyway and the `z-image` base model honours it.
