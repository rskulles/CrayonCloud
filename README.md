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
- **Text to image and image to image.** Type a prompt, or start from a picture and say what should change, with a
  strength slider that works like the one in Automatic1111. Styles (LoRAs) apply to both.
- **At home on a Mac.** Apple Silicon runs the model through MLX with 8-bit weights. On an M4 Pro a 1024 × 1024
  image takes about a minute and a half at 8 steps, 1024 × 768 about a minute, 512 × 512 twenty seconds. Linux or
  Windows with an NVIDIA card works through PyTorch.
- **Keeps what it makes.** Every picture lands in `~/Pictures/Crayon Cloud` with the prompt, seed and settings
  written into the file, so you can always tell how a picture was made.
- **Kind to your memory.** The model loads on the first request (one second, once the quantised copy is cached) and
  unloads itself after half an hour of quiet, so the rest of the machine gets the memory back.

## Get it running

Two ways: the menu bar app on a Mac (nothing to install first), or `pip` anywhere. Either way the model arrives on
the first picture: a 10 GB download of the ready-made 8-bit copy, kept under `~/.cache/huggingface`.

### The menu bar app (macOS, Apple Silicon only)

Download the disk image from the [Releases page](https://github.com/rskulles/CrayonCloud/releases), drag Crayon Cloud
to Applications and open it. It needs an M-series Mac: the model runs through MLX, which has no Intel build, and an
Intel Mac's graphics card is too small for it anyway. On x86 the realistic route is Linux with an NVIDIA card, below. A small cloud appears in the menu bar with the server's state, and the menu has *Open*,
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

To uninstall, delete the app and `~/Library/Application Support/CrayonCloud`; add the Z-Image folders under
`~/.cache/huggingface/hub` (and `~/.cache/crayoncloud`, if it exists) if you want the model gone too. Your pictures
stay in `~/Pictures/Crayon Cloud`.

### From the command line (any platform)

You need Python 3.10 to 3.13.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install "crayoncloud[mlx] @ git+https://github.com/rskulles/CrayonCloud"     # Apple Silicon
pip install "crayoncloud[cuda] @ git+https://github.com/rskulles/CrayonCloud"    # Linux or Windows with an NVIDIA card
crayoncloud serve
```

That starts it at <http://127.0.0.1:8765/>. Open that page to try a prompt by hand. The first request downloads the
8-bit Z-Image-Turbo (about 10 GB, a few minutes on a fast line) and from then on the model loads in about a second.
Pictures are kept in `~/Pictures/Crayon Cloud` unless you say `--assets` or `--no-save`.

**Where the weights come from.** Crayon Cloud tries, in order: [rskulles/z-image-turbo-mflux-q8](https://huggingface.co/rskulles/z-image-turbo-mflux-q8),
a copy quantised by this project's author, pinned to a known commit; then the mflux community's
[z-image-turbo-mflux-q8](https://huggingface.co/mflux-community/z-image-turbo-mflux-q8), also pinned; then the
original [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) weights (20 GB), quantised on
your machine and saved under `~/.cache/crayoncloud`. `--quantize-locally` skips straight to the last step; the
`/health` route says which source is in use. Once any of them has been fetched, nothing is downloaded again.

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
| `--quantize` | 8 | MLX weight quantisation: 8 halves the memory with no visible loss; 4 halves it again (ready-made copies exist for 3 to 8) |
| `--quantize-locally` | off | Ignore the ready-made copies and quantise the full-precision weights on this machine |
| `--steps` | 8 | Diffusion steps when a request does not say; Turbo is tuned for 8 or 9 |
| `--preload` | off | Load the model at start instead of on the first request |
| `--idle-unload` | 30 | Minutes of quiet before the model is unloaded; 0 keeps it loaded |
| `--assets` | ~/Pictures/Crayon Cloud | Folder every rendered picture is kept in, with the prompt, seed and steps in the PNG's metadata |
| `--no-save` | off | Keep nothing on disk; pictures only go back to the caller |
| `--loras` | see Styles | Folder of `.safetensors` adapters offered by name |
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
tools/make-macos-app.sh 0.2.0 dist      # the version goes into the bundle
open dist                               # drag "Crayon Cloud.app" to Applications
```

Add `--dmg` at the end to get a disk image as well. A locally built app is signed only for the machine that built
it: it runs there, but a copy sent to another Mac is refused by Gatekeeper, which is what the notarized release is
for. Rebuilding with a new version number makes the app reinstall its package on the next launch.

## Styles (LoRAs)

Drop a Z-Image LoRA (a `.safetensors` file) into the LoRA folder, `~/Library/Application Support/CrayonCloud/loras`
on a Mac (*Open LoRA Folder* in the menu), `~/.config/crayoncloud/loras` elsewhere, or wherever `--loras` points, and
it appears by its file name: in the try-it page's *Styles* rows, at `GET /v1/loras`, and in ButterKnife's picture
style picker. Ask for them with `"loras": [{"name": "watercolor", "scale": 0.8}, {"name": "sketch", "scale": 0.4}]`
on a request (up to four, each with its own strength; they stack, and since each is an additive change to the weights
the order does not change the result), or `/image a lighthouse --lora watercolor:0.8 --lora sketch:0.4` in ButterKnife.
Changing the set of styles reloads the model, about five seconds; renders with the same set after that cost nothing
extra. Adapters work on the 8-bit weights directly. mflux engine only.

## Image to image

Start from a picture instead of noise: the try-it page has a *Text to image* / *Image to image* switch at the top of
the form. Pick *Image to image*, choose a picture, set the strength, describe the picture you want, and generate.
Every result has a *Use as source* button, so you can push a picture through several rounds. Leave *Size* blank and
the result keeps the picture's shape (scaled into the 256 to 2048 range and rounded to multiples of 16); fill it in
to reshape. A big photo is shrunk to 2048 on its longest side before anything else happens, so a 24-megapixel JPEG
costs no more than a screenshot.

### How it works, in plain words

A normal render starts from pure noise and the model paints your prompt out of it over 8 steps. Image to image
starts from *your picture with some noise poured over it* and skips the first few steps. The model then does what it
always does: it paints the prompt, but now the rough shapes and colours that survived under the noise steer where
things end up. That is all it is. The model never sees your original clearly, it does not know what "the same" means,
and it cannot tell which parts you wanted kept.

**Strength** is how much noise is poured on, and works the way it does in Automatic1111 and ComfyUI: 0 pours none
and hands the picture straight back, 1 drowns it completely and is plain text to image. Because Z-Image-Turbo takes
8 steps, the slider really has eight positions (each 0.125 is one step) and anything in between rounds to one of
them. Lower strength also means fewer steps, so it is faster than a plain render.

| What you want | Strength |
|---|---|
| A small touch-up, a little more polish, the same picture slightly cleaner | 0.3 to 0.4 |
| The same scene in a different style or medium (photo to watercolour, sketch to painting) | 0.5 to 0.65 |
| The same layout with different lighting, season, weather or time of day | 0.7 to 0.85 |
| Only the rough arrangement kept; everything else new | 0.9 |

### How to write the prompt

Describe the **finished picture**, not the change. The model is not an editor and does not take instructions.

- Not "make it night" or "the same meadow but at night". Say "a meadow at night under a full moon, dark blue sky,
  moonlit grass, storybook illustration".
- **Say what should stay as well as what should change.** Subject, medium, framing, background. If you upload a
  watercolour of a lighthouse and only write "at sunset", you may get a photograph of a sunset. Write "watercolour
  of a lighthouse at sunset".
- **For a style change, put the medium first.** "Charcoal sketch of a woman in a red coat on a rainy street" works far
  better than tacking "as a sketch" on the end. A style LoRA stacks on top of this.
- **Match the strength to the size of the change.** Low strength keeps the source's colours and light, so a prompt
  that contradicts them loses. A night prompt at 0.6 on a sunny picture gives you a sunny picture with a faint moon.
- **Keep the seed fixed while you move the slider**, so you can see what the strength alone did. Then try a few seeds
  at the strength that worked.
- Negative prompts do nothing on Z-Image-Turbo. Put what you want in, not what you want out.

### What it cannot do

The change is always to the **whole picture**. There is no mask and no inpainting, so you cannot say "only the shirt".
Changing one thing means everything else gets repainted a little too. A person in a blue shirt can become a person in
a red shirt at around 0.6 to 0.7, but the face will come back as someone who looks a lot like them, not exactly them.
It works best when the thing you want changed is big in the frame and the thing you want kept is not a face. Crop the
source to the composition you want before uploading.

From the command line:

```bash
crayoncloud generate "a meadow at night under a full moon, dark blue sky, moonlit grass, storybook illustration" --image docs/sample.png --strength 0.75
```

mflux engine only for now; the diffusers engine answers with an error.

## The API

`POST /v1/images/generations` with a JSON body in the OpenAI shape, plus a few extras:

```json
{ "prompt": "a red bicycle in the rain", "size": "1024x1024", "n": 1, "steps": 8, "seed": 12345, "negative_prompt": "", "loras": [{"name": "watercolor", "scale": 0.8}] }
```

The answer carries the PNG as `data[0].b64_json` (the only supported `response_format`), the seed used for each
image, the `file` it was saved as, and a `crayoncloud` block with the model, size, steps, seconds taken, the styles
used and the `source` picture (or `null`). `loras` on the request names adapters from the LoRA folder with an optional
`scale` (see Styles). `GET /v1/models` lists the model, `GET /health` (or `/v1/status`) says whether it is loaded,
busy and for how long. Sizes are rounded to multiples of 16 between 256 and 2048; `size` may be left out (1024 × 1024,
or the source picture's shape). Requests queue; one image renders at a time.

Z-Image-Turbo is guidance-distilled, so `negative_prompt` is accepted but ignored for it (the base model honours it).

### Image to image over the API (for ButterKnife and friends)

Two ways to send the source picture; both answer in exactly the same shape as generations.

**`POST /v1/images/edits`**, OpenAI's edits endpoint, `multipart/form-data`. This is the one an OpenAI-compatible
client already knows how to call:

| Field | Required | What it is |
|---|---|---|
| `image` | yes | The picture file: PNG, JPEG or WebP, up to 32 MB. EXIF orientation is applied; anything longer than 2048 on its longest side is scaled down to that first, since nothing renders larger. |
| `prompt` | yes | What the result should be |
| `strength` | no | 0 to 1, default 0.6; see above |
| `size` | no | `WxH`; leave out to keep the picture's shape |
| `n`, `steps`, `seed`, `negative_prompt`, `model`, `response_format` | no | As for generations |
| `loras` | no | The same list as for generations, as a JSON string: `[{"name": "watercolor", "scale": 0.8}]` |
| `mask` | no | Not supported; sending one is a 400. There is no inpainting. |

```bash
curl -s http://127.0.0.1:8765/v1/images/edits -F image=@photo.jpg -F prompt="the same scene in winter" -F strength=0.55 \
  | python3 -c "import sys,json,base64; d=json.load(sys.stdin); open('winter.png','wb').write(base64.b64decode(d['data'][0]['b64_json'])); print(d['crayoncloud'])"
```

**JSON, on `POST /v1/images/generations`**: add `"image"` with the picture as base64 (a `data:image/png;base64,…`
URL is fine) and optionally `"strength"`. Handy for clients that cannot do multipart.

```json
{ "prompt": "the same scene in winter", "image": "iVBORw0KGgo…", "strength": 0.55 }
```

Either way the `crayoncloud` block of the answer carries `"source": {"width": 1024, "height": 768, "strength": 0.55}`
(it is `null` for plain text to image), and a saved PNG has `source` and `strength` in its metadata next to the
prompt and seed. Validation errors are 400 (not a picture, bad size, unknown LoRA is 404) or 422 (strength outside
0 to 1, empty prompt). A ButterKnife `/image` with an attached picture would post to `/v1/images/edits` with the
attachment as `image`, the message as `prompt`, and a `--strength` flag if the user gave one.

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
