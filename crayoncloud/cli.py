"""crayoncloud serve | generate."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crayoncloud", description="Self-hosted image generation with an OpenAI-compatible endpoint.")
    parser.add_argument("--version", action="version", version=f"crayoncloud {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def engine_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--engine", default="auto", choices=["auto", "mflux", "diffusers", "fake"], help="auto = MLX on Apple Silicon, diffusers elsewhere")
        p.add_argument("--model", default="z-image-turbo", help="z-image-turbo (default) or z-image")
        p.add_argument("--quantize", type=int, default=8, choices=[3, 4, 5, 6, 8, 16], help="MLX weight quantisation; 16 means none (mflux only)")
        p.add_argument("--steps", type=int, default=8, help="default diffusion steps when a request does not say")

    serve = sub.add_parser("serve", help="run the HTTP server")
    engine_args(serve)
    serve.add_argument("--host", default="127.0.0.1", help="use --lan to listen on every interface")
    serve.add_argument("--lan", action="store_true", help="listen on 0.0.0.0 so other devices (and ButterKnife on another machine) can reach it")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--preload", action="store_true", help="load the model at start instead of on the first request")
    serve.add_argument("--idle-unload", type=float, default=30.0, metavar="MINUTES", help="unload the model after this many idle minutes; 0 keeps it loaded")

    gen = sub.add_parser("generate", help="make one image from the command line")
    engine_args(gen)
    gen.add_argument("prompt")
    gen.add_argument("--size", default="1024x1024")
    gen.add_argument("--seed", type=int)
    gen.add_argument("--output", default="crayoncloud-{seed}.png")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    args = build_parser().parse_args(argv)
    from .engines import GenerationRequest, pick_engine

    quantize = None if args.quantize == 16 else args.quantize
    engine = pick_engine(args.engine, args.model, quantize)
    if getattr(engine, "quantized_path", None) is not None:
        log = logging.getLogger("crayoncloud")
        log.info("quantised weights cache: %s (set CRAYONCLOUD_CACHE to move it)", engine.quantized_path)

    if args.command == "generate":
        import random

        from .api import parse_size

        width, height = parse_size(args.size)
        seed = args.seed if args.seed is not None else random.randrange(2**31 - 1)
        started = time.monotonic()
        image = engine.generate(GenerationRequest(prompt=args.prompt, width=width, height=height, steps=args.steps, seed=seed))
        path = args.output.format(seed=seed)
        image.save(path)
        print(f"{path}: {width}x{height}, seed {seed}, {time.monotonic() - started:.1f} s")
        return 0

    import uvicorn

    from .api import create_app
    from .service import ImageService

    service = ImageService(engine, idle_unload_seconds=None if args.idle_unload <= 0 else args.idle_unload * 60)
    app = create_app(service, default_steps=args.steps, preload=args.preload)
    host = "0.0.0.0" if args.lan else args.host
    print(f"Crayon Cloud {__version__}: {engine.model} via {engine.name} at http://{host}:{args.port}/ (ButterKnife base URL: http://{'<this machine>' if host == '0.0.0.0' else host}:{args.port}/v1)", flush=True)
    uvicorn.run(app, host=host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
