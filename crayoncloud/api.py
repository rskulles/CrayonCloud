"""The HTTP surface: the OpenAI images endpoint ButterKnife and LocalAI-style clients speak, a model list, a status route,
and a tiny page for trying prompts by hand."""

from __future__ import annotations

import base64
import io
import random
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from html import escape

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .engines import GenerationRequest
from .service import ImageService

MIN_SIDE, MAX_SIDE, STEP = 256, 2048, 16


class ImagesRequest(BaseModel):
    """OpenAI's images/generations body plus a few extras local models care about."""

    prompt: str = Field(min_length=1, max_length=4000)
    model: str | None = None
    n: int = Field(default=1, ge=1, le=4)
    size: str = "1024x1024"
    response_format: str = "b64_json"
    steps: int | None = Field(default=None, ge=1, le=100)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    negative_prompt: str | None = Field(default=None, max_length=4000)


def parse_size(size: str) -> tuple[int, int]:
    try:
        w, h = (int(part) for part in size.lower().replace("×", "x").split("x"))
    except ValueError as exc:
        raise HTTPException(400, f"size must look like 1024x1024, not {size!r}") from exc
    for side in (w, h):
        if not MIN_SIDE <= side <= MAX_SIDE:
            raise HTTPException(400, f"each side must be between {MIN_SIDE} and {MAX_SIDE}; got {size}")
    # Diffusion models want multiples of 16; round rather than reject so "1000x700" still works.
    return (round(w / STEP) * STEP, round(h / STEP) * STEP)


def create_app(service: ImageService, default_steps: int = 8, preload: bool = False) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import asyncio

        watcher = asyncio.create_task(service.idle_watch())
        if preload:
            await service.preload()
        try:
            yield
        finally:
            watcher.cancel()
            service.close()

    app = FastAPI(title="Crayon Cloud", version=__version__, lifespan=lifespan)

    @app.get("/health")
    @app.get("/v1/status")
    def health():
        return {"ok": True, "version": __version__, **asdict(service.status())}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": service.engine.model, "object": "model", "owned_by": "crayoncloud", "engine": service.engine.name}]}

    @app.post("/v1/images/generations")
    async def generations(body: ImagesRequest):
        if body.response_format != "b64_json":
            raise HTTPException(400, "only response_format=b64_json is supported (there is no place to host URLs)")
        if body.model and body.model != service.engine.model:
            raise HTTPException(404, f"this server runs {service.engine.model!r}, not {body.model!r}")
        width, height = parse_size(body.size)
        steps = body.steps or default_steps
        first_seed = body.seed if body.seed is not None else random.randrange(2**31 - 1)

        data = []
        seconds_total = 0.0
        for i in range(body.n):
            request = GenerationRequest(
                prompt=body.prompt,
                width=width,
                height=height,
                steps=steps,
                seed=first_seed + i,
                negative_prompt=body.negative_prompt,
            )
            try:
                image, seconds = await service.generate(request)
            except Exception as exc:  # noqa: BLE001 - surface the engine's message to the client
                raise HTTPException(500, f"generation failed: {exc}") from exc
            seconds_total += seconds
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            data.append({"b64_json": base64.b64encode(buffer.getvalue()).decode("ascii"), "seed": request.seed})

        return JSONResponse(
            {
                "created": int(time.time()),
                "data": data,
                "crayoncloud": {
                    "model": service.engine.model,
                    "engine": service.engine.name,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "seconds": round(seconds_total, 1),
                },
            }
        )

    @app.get("/", response_class=HTMLResponse)
    def index():
        status = service.status()
        return f"""<!doctype html><html><head><meta charset="utf-8"><title>Crayon Cloud</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font:15px system-ui,sans-serif;max-width:48rem;margin:2rem auto;padding:0 1rem;color:#222}}
textarea,input{{width:100%;box-sizing:border-box;font:inherit;padding:.4rem}} button{{font:inherit;padding:.5rem 1rem}}
img{{max-width:100%;margin-top:1rem;border-radius:.5rem}} .row{{display:flex;gap:.5rem;margin:.5rem 0}} .row>*{{flex:1}} code{{background:#eee;padding:.1rem .3rem}}</style></head>
<body><h1>Crayon Cloud</h1>
<p>Serving <b>{escape(status.model)}</b> with the {escape(status.engine)} engine, version {__version__}.
Point ButterKnife (or anything that speaks the OpenAI images API) at <code>{{origin}}/v1</code>.</p>
<form id="f"><textarea name="prompt" rows="3" placeholder="A butter knife spreading a sunrise over toast"></textarea>
<div class="row"><input name="size" value="1024x1024"><input name="steps" value="{status and 8}" placeholder="steps"><input name="seed" placeholder="seed (random)"></div>
<button>Generate</button> <span id="s"></span></form>
<img id="out" alt="">
<script>
document.querySelector('code').textContent = location.origin + '/v1';
const f=document.getElementById('f'), s=document.getElementById('s'), out=document.getElementById('out');
f.onsubmit=async e=>{{e.preventDefault(); const d=Object.fromEntries(new FormData(f)); const body={{prompt:d.prompt,size:d.size,steps:+d.steps||undefined,seed:d.seed?+d.seed:undefined}};
s.textContent='rendering…'; const t0=Date.now(); const tick=setInterval(()=>s.textContent='rendering… '+Math.round((Date.now()-t0)/1000)+' s',500);
try{{const r=await fetch('/v1/images/generations',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}}); const j=await r.json(); clearInterval(tick);
if(!r.ok){{s.textContent=j.detail||'failed';return;}} out.src='data:image/png;base64,'+j.data[0].b64_json; s.textContent=j.crayoncloud.seconds+' s, seed '+j.data[0].seed;}}catch(err){{clearInterval(tick); s.textContent=err;}} }};
</script></body></html>"""

    return app
