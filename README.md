# Crayon Cloud

A small image generation server for your own machine. It runs [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
and answers the same request shape as the OpenAI images API, so [ButterKnife](https://github.com/rskulles/ButterKnife)
(and anything else that speaks that API) can ask it for pictures. Nothing leaves your network.

![Crayon Cloud icon: a white cloud raining a dot matrix of red, green and blue](packaging/icon.png)

## Why

- **Made for ButterKnife.** Add it as an image connection and type `/image a cat in a spacesuit` in any chat.
- **One command.** No ComfyUI graphs, no web UI to click through. Start it and forget it.
- **Fast on a Mac.** Apple Silicon runs the model through MLX; a 1024 px image takes a handful of seconds on an M-series
  chip with the default 8-bit weights.
- **Kind to your memory.** The model loads on the first request and unloads itself after half an hour of quiet, so the
  rest of the machine gets the memory back.

## Get it running

You need Python 3.10 to 3.13 and about 20 GB of disk for the model on first run.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install "crayoncloud[mlx] @ git+https://github.com/rskulles/CrayonCloud"     # Apple Silicon
pip install "crayoncloud[cuda] @ git+https://github.com/rskulles/CrayonCloud"    # Linux or Windows with an NVIDIA card
crayoncloud serve
```

That starts it at <http://127.0.0.1:8765/>. Open that page to try a prompt by hand. The first request downloads
Z-Image-Turbo from Hugging Face and loads it; every request after that is just the render.

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

There is also a one-off mode for the command line:

```bash
crayoncloud generate "a butter knife spreading a sunrise over toast" --size 1280x768 --seed 7
```

## The API

`POST /v1/images/generations` with a JSON body in the OpenAI shape, plus a few extras:

```json
{ "prompt": "a red bicycle in the rain", "size": "1024x1024", "n": 1, "steps": 8, "seed": 12345, "negative_prompt": "" }
```

The answer carries the PNG as `data[0].b64_json` (the only supported `response_format`), the seed used for each
image, and a `crayoncloud` block with the model, size, steps and seconds taken. `GET /v1/models` lists the model,
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

Crayon Cloud is **source-available, not open source**, under the
[PolyForm Noncommercial License 1.0.0](LICENSE.md), the same terms as ButterKnife. Noncommercial use is free; commercial
use needs written permission. The model weights and the libraries it runs on keep their own licences: Z-Image-Turbo is
Apache 2.0, mflux is MIT, diffusers is Apache 2.0.

Copyright Roy S.
