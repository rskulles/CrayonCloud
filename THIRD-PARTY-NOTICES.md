# Third-party notices

Crayon Cloud is MIT licensed (see LICENSE.md). It stands on the open-source work below, which keeps its own licences.
Everything is permissive: MIT, BSD, Apache 2.0 and the like; nothing copyleft.

## Shipped inside Crayon Cloud

The one piece of third-party code copied into this repository:

| Component | Licence | Where |
|---|---|---|
| [Pico CSS 2.1.1](https://picocss.com) (classless build) | MIT | `crayoncloud/static/pico.classless.min.css`, served to the try-it page |

```
Pico CSS
Copyright 2019-2025 Pico CSS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Installed from PyPI

These are fetched by `pip` when Crayon Cloud is installed (the menu bar app does the same on first launch); they
are not copied into this repository, and each carries its licence text inside its own package. Versions are the
ones current when this file was written.

| Component | Licence | Used for |
|---|---|---|
| [mflux 0.19](https://github.com/filipstrand/mflux) | MIT | Z-Image on Apple Silicon (the `mlx` extra) |
| [MLX 0.32](https://github.com/ml-explore/mlx), Apple | MIT | The array framework mflux runs on |
| [Hugging Face Diffusers](https://github.com/huggingface/diffusers) | Apache 2.0 | Z-Image on CUDA and CPU (the `cuda` extra) |
| [PyTorch](https://pytorch.org) | BSD-3-Clause (with Apache 2.0, BSD-2, BSL-1.0 and MIT parts) | Diffusers' engine |
| [Transformers](https://github.com/huggingface/transformers), [Tokenizers](https://github.com/huggingface/tokenizers), [safetensors](https://github.com/huggingface/safetensors), [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Apache 2.0 | Text encoder, tokenizer, weight files and the model download |
| [NumPy](https://numpy.org) | BSD-3-Clause (with 0BSD, MIT, Zlib and CC0 parts) | Array plumbing |
| [OpenCV (opencv-python)](https://github.com/opencv/opencv-python) | Apache 2.0 | Pulled in by mflux for image handling |
| [FastAPI](https://github.com/fastapi/fastapi) | MIT | The HTTP API |
| [Starlette](https://github.com/Kludex/starlette) | BSD-3-Clause | The framework under FastAPI |
| [Uvicorn](https://uvicorn.dev) | BSD-3-Clause | The server |
| [Pydantic](https://github.com/pydantic/pydantic) | MIT | Request validation |
| [python-multipart](https://github.com/Kludex/python-multipart) | Apache 2.0 | Reading the multipart form of `/v1/images/edits` (the source picture) |
| [Pillow](https://python-pillow.github.io) | MIT-CMU (the historical PIL licence) | Encoding PNGs and their metadata |
| [pytest](https://pytest.org), [HTTPX](https://www.python-httpx.org) | MIT, BSD-3-Clause | Tests only (the `dev` extra) |

## Model weights

Downloaded from Hugging Face on the first picture; never redistributed by this project.

| Model | Licence |
|---|---|
| [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) and [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image), Tongyi Lab (Alibaba) | Apache 2.0 |

## Tools used to build the Mac app

The macOS menu bar helper is compiled with Apple's Swift toolchain and uses AppKit; nothing from it is redistributed
beyond the binary the app itself is.
