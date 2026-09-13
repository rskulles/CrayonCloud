# Crayon Cloud

A small image generation server for your own machine. It runs [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
and answers the same request shape as the OpenAI images API, so [ButterKnife](https://github.com/rskulles/ButterKnife)
(and anything else that speaks that API) can ask it for pictures. Nothing leaves your network.

![A white cloud over a sunlit meadow, raining tiny red, green and blue crayons; made by Crayon Cloud on an M4 Pro](docs/sample.png)

*"A fluffy white cloud raining tiny red, green and blue crayons over a sunlit meadow, storybook illustration." Z-Image-Turbo, seed 42, 8 steps, on an M4 Pro.*

## Why

- **Made for ButterKnife.** Add it as an image connection and type `/image a cat in a spacesuit` in any chat.
- **One command.** No ComfyUI graphs, no web UI to click through. Start it and forget it.
- **Runs on a Mac.** Apple Silicon runs the model through MLX with 8-bit weights. On an M4 Pro a 1024 × 1024 image
  takes about a minute and a half at 8 steps; 1024 × 768 about a minute.
- **Kind to your memory.** The model loads on the first request (one second, once the quantised copy is cached) and
  unloads itself after half an hour of quiet, so the rest of the machine gets the memory back.

## Get it running

You need Python 3.10 to 3.13 and about 30 GB of disk for the model (20 GB downloaded, 10 GB quantised).

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install "crayoncloud[mlx] @ git+https://github.com/rskulles/CrayonCloud"     # Apple Silicon
pip install "crayoncloud[cuda] @ git+https://github.com/rskulles/CrayonCloud"    # Linux or Windows with an NVIDIA card
crayoncloud serve
```

That starts it at <http://127.0.0.1:8765/>. Open that page to try a prompt by hand. The first request downloads
Z-Image-Turbo from Hugging Face (about 20 GB, a few minutes on a fast line), quantises it and saves the quantised
copy (10 GB) under `~/.cache/crayoncloud`; from then on the model loads in about a second. Set `CRAYONCLOUD_CACHE`
to keep that copy somewhere else.

To let other machines use it, including a ButterKnife running elsewhere on your Wi‑Fi:

```bash
crayoncloud serve --lan
```

In ButterKnife, add a connection of kind *Image generation* with the base URL `http://<this machine>:8765/v1`.

### Options

| Flag | Default | What it does |
|---|---|---|
| `--port` | 8765 | Port to listen on |
| `--lan` | off | Listen on every interface instead of localhost only |
| `--engine` | auto | `mflux` (MLX) on Apple Silicon, `diffusers` (PyTorch) elsewhere, `fake` for a placeholder |
| `--model` | z-image-turbo | `z-image` for the slower, non-distilled base model |
| `--quantize` | 8 | MLX weight quantisation: 8 halves the memory with no visible loss; 4 halves it again |
| `--steps` | 8 | Diffusion steps when a request does not say; Turbo is tuned for 8 or 9 |
| `--preload` | off | Load the model at start instead of on the first request |
| `--idle-unload` | 30 | Minutes of quiet before the model is unloaded; 0 keeps it loaded |
| `--assets` | ~/Pictures/Crayon Cloud | Folder every rendered picture is kept in, with the prompt, seed and steps in the PNG's metadata |
| `--no-save` | off | Keep nothing on disk; pictures only go back to the caller |
| `--parent-pid` | | Stop when that process is gone (the menu bar app passes its own pid) |

There is also a one-off mode for the command line:

```bash
crayoncloud generate "a butter knife spreading a sunrise over toast" --size 1280x768 --seed 7
```

## Menu bar app for a Mac

If you'd rather not think about terminals, build the menu bar app once and keep it in Applications. It shows a small
cloud in the menu bar with the server's state, and has *Open*, *Copy address for ButterKnife*, *Open Assets Folder*
(where every rendered picture is kept), a switch for reaching it from the network, the log, and *Quit*.

You need the Xcode Command Line Tools (`xcode-select --install`) and a Python 3.10 to 3.13 that the app can find:
Homebrew's (`brew install python@3.12`), python.org's, or MacPorts'.

```bash
git clone https://github.com/rskulles/CrayonCloud && cd CrayonCloud
tools/make-macos-app.sh 0.1.0 dist
open dist              # drag "Crayon Cloud.app" to Applications
```

Add `--dmg` at the end to get a disk image as well. The app is signed only for the machine that built it; Gatekeeper
would refuse a copy sent to another Mac, so build it there too.

The first launch sets things up: it creates a Python environment under `~/Library/Application Support/CrayonCloud`
and installs the package into it (a few minutes, shown as "Setting up…" in the menu). The model itself downloads on
the first picture, as with the command line. The server runs on port 8765 as long as the app is open; quitting the
app stops it. Everything it prints goes to `~/Library/Logs/CrayonCloud/server.log` (*Show log* in the menu).

Rebuilding the app with a new version number makes it reinstall the package on the next launch. To uninstall, delete
the app, `~/Library/Application Support/CrayonCloud` and, if you want the model gone too, `~/.cache/crayoncloud` and
the Z-Image folders under `~/.cache/huggingface/hub`.

## The API

`POST /v1/images/generations` with a JSON body in the OpenAI shape, plus a few extras:

```json
{ "prompt": "a red bicycle in the rain", "size": "1024x1024", "n": 1, "steps": 8, "seed": 12345, "negative_prompt": "" }
```

The answer carries the PNG as `data[0].b64_json` (the only supported `response_format`), the seed used for each
image, the `file` it was saved as, and a `crayoncloud` block with the model, size, steps and seconds taken. `GET /v1/models` lists the model,
`GET /health` (or `/v1/status`) says whether it is loaded, busy and for how long. Sizes are rounded to multiples of 16
between 256 and 2048. Requests queue; one image renders at a time.

Z-Image-Turbo is guidance-distilled, so `negative_prompt` is accepted but ignored for it (the base model honours it).

## Building from source

```bash
git clone https://github.com/rskulles/CrayonCloud && cd CrayonCloud
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[mlx,dev]"      # or [cuda,dev]
pytest
crayoncloud serve --engine fake  # the API without a model, for development
```

## License

Crayon Cloud is open source under the [MIT License](LICENSE.md). Do what you like with it; keep the notice.

It is built on other people's open-source work, all of it permissive: mflux and MLX (MIT), Diffusers and the Hugging
Face libraries (Apache 2.0), PyTorch, Starlette and Uvicorn (BSD), FastAPI and Pydantic (MIT), Pillow (MIT-CMU), and
Pico CSS (MIT) for the try-it page. The Z-Image models from Tongyi Lab are Apache 2.0 and download straight from
Hugging Face. Every one of them is credited, with links and licences, in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

Copyright (c) 2026 Roy Skullestad
