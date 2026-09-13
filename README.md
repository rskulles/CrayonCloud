# Crayon Cloud

A small image generation server for your own machine. It runs [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
and answers the same request shape as the OpenAI images API, so [ButterKnife](https://github.com/rskulles/ButterKnife)
(and anything else that speaks that API) can ask it for pictures. Nothing leaves your network.

![A white cloud over a sunlit meadow, raining tiny red, green and blue crayons; made by Crayon Cloud on an M4 Pro](docs/sample.png)

*"A fluffy white cloud raining tiny red, green and blue crayons over a sunlit meadow, storybook illustration." Z-Image-Turbo, seed 42, 8 steps, on an M4 Pro.*

## Why

- **Made for ButterKnife.** Add it as an image connection and type `/image a cat in a spacesuit` in any chat.
- **Nothing to fiddle with.** No ComfyUI graphs, no web UI to click through. On a Mac it's a menu bar app that
  carries its own Python; elsewhere it's one command. Start it and forget it.
- **At home on a Mac.** Apple Silicon runs the model through MLX with 8-bit weights. On an M4 Pro a 1024 × 1024
  image takes about a minute and a half at 8 steps, 1024 × 768 about a minute, 512 × 512 twenty seconds. Linux or
  Windows with an NVIDIA card works through PyTorch.
- **Keeps what it makes.** Every picture lands in `~/Pictures/Crayon Cloud` with the prompt, seed and settings
  written into the file, so you can always tell how a picture was made.
- **Kind to your memory.** The model loads on the first request (one second, once the quantised copy is cached) and
  unloads itself after half an hour of quiet, so the rest of the machine gets the memory back.

## Get it running

Two ways: the menu bar app on a Mac (nothing to install first), or `pip` anywhere. Either way the model needs about
30 GB of disk (20 GB downloaded, 10 GB quantised) and arrives on the first picture.

### The menu bar app (macOS)

Download the disk image from the [Releases page](https://github.com/rskulles/CrayonCloud/releases), drag Crayon Cloud
to Applications and open it. A small cloud appears in the menu bar with the server's state, and the menu has *Open*,
*Copy network address for ButterKnife*, *Open Assets Folder*, a switch for reaching it from the network, the log and
*Quit*.

The first launch sets things up: the app creates a Python environment under `~/Library/Application Support/CrayonCloud`
with the interpreter it carries and installs the package into it (a few minutes, shown as "Setting up…" in the menu).
The server runs on port 8765 as long as the app is open; quitting the app stops it. Everything it prints goes to
`~/Library/Logs/CrayonCloud/server.log` (*Show log* in the menu).

**On the network from the start.** The app listens on every interface by default, since ButterKnife is usually on
another machine: the menu shows the address other devices use (*On your network: http://…:8765/v1*), and *Copy
network address for ButterKnife* puts exactly that on the clipboard. Switch off *Reachable on the local network* to
keep it to this Mac. macOS asks once for local-network permission the first time.

To uninstall, delete the app and `~/Library/Application Support/CrayonCloud`; add `~/.cache/crayoncloud` and the
Z-Image folders under `~/.cache/huggingface/hub` if you want the model gone too. Your pictures stay in
`~/Pictures/Crayon Cloud`.

### From the command line (any platform)

You need Python 3.10 to 3.13.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install "crayoncloud[mlx] @ git+https://github.com/rskulles/CrayonCloud"     # Apple Silicon
pip install "crayoncloud[cuda] @ git+https://github.com/rskulles/CrayonCloud"    # Linux or Windows with an NVIDIA card
crayoncloud serve
```

That starts it at <http://127.0.0.1:8765/>. Open that page to try a prompt by hand. The first request downloads
Z-Image-Turbo from Hugging Face (about 20 GB, a few minutes on a fast line), quantises it and saves the quantised
copy (10 GB) under `~/.cache/crayoncloud`; from then on the model loads in about a second. Set `CRAYONCLOUD_CACHE`
to keep that copy somewhere else. Pictures are kept in `~/Pictures/Crayon Cloud` unless you say `--assets` or
`--no-save`.

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

## Building the Mac app yourself

The Releases page carries a signed and notarized disk image built by GitHub Actions (`.github/workflows/release.yml`).
To build the app locally instead, you need the Xcode Command Line Tools (`xcode-select --install`) and an internet
connection the first time: the script downloads a relocatable CPython 3.12 from
[python-build-standalone](https://github.com/astral-sh/python-build-standalone) (cached under
`~/.cache/crayoncloud-build`) and puts it inside the bundle, which is why the finished app needs no Python on the
machine it runs on. The app is about 70 MB.

```bash
git clone https://github.com/rskulles/CrayonCloud && cd CrayonCloud
tools/make-macos-app.sh 0.1.5 dist      # the version goes into the bundle
open dist                               # drag "Crayon Cloud.app" to Applications
```

Add `--dmg` at the end to get a disk image as well. A locally built app is signed only for the machine that built
it: it runs there, but a copy sent to another Mac is refused by Gatekeeper, which is what the notarized release is
for. Rebuilding with a new version number makes the app reinstall its package on the next launch.

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
