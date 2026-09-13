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

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
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


def default_assets_dir() -> Path:
    """Where rendered pictures are kept: the Pictures folder, so they show up where people look for images."""
    return Path.home() / "Pictures" / "Crayon Cloud"


def save_asset(assets_dir: Path, image, prompt: str, seed: int, steps: int, model: str) -> Path:
    """Writes the PNG with the prompt and settings in its text chunks, named by time and seed."""
    from PIL import PngImagePlugin

    assets_dir.mkdir(parents=True, exist_ok=True)
    info = PngImagePlugin.PngInfo()
    info.add_text("prompt", prompt)
    info.add_text("seed", str(seed))
    info.add_text("steps", str(steps))
    info.add_text("model", model)
    info.add_text("software", f"Crayon Cloud {__version__}")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = assets_dir / f"{stamp}-{seed}.png"
    counter = 1
    while path.exists():
        counter += 1
        path = assets_dir / f"{stamp}-{seed}-{counter}.png"
    image.save(path, format="PNG", pnginfo=info)
    return path


def create_app(service: ImageService, default_steps: int = 8, preload: bool = False, assets_dir: Path | None = None) -> FastAPI:
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
        return {"ok": True, "version": __version__, "assets": None if assets_dir is None else str(assets_dir), **asdict(service.status())}

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
            entry = {"b64_json": base64.b64encode(buffer.getvalue()).decode("ascii"), "seed": request.seed}
            if assets_dir is not None:
                try:
                    entry["file"] = str(save_asset(assets_dir, image, request.prompt, request.seed, steps, service.engine.model))
                except OSError as exc:  # a full disk or a missing Pictures folder must not lose the picture
                    entry["file_error"] = str(exc)
            data.append(entry)

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

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        status = service.status()
        return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Crayon Cloud</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="light dark">
<link rel="stylesheet" href="/static/pico.classless.min.css">
<style>figure img{{border-radius:var(--pico-border-radius)}} progress{{margin-top:1rem}} .status{{margin-left:.75rem}}</style></head>
<body><main>
<header><hgroup><h1>Crayon Cloud</h1>
<p>Serving <strong>{escape(status.model)}</strong> with the {escape(status.engine)} engine, version {__version__}.</p></hgroup></header>
<p>Point ButterKnife, or anything that speaks the OpenAI images API, at <code id="base">/v1</code>.</p>
<form id="f">
  <label>Prompt<textarea name="prompt" rows="3" required placeholder="A butter knife spreading a sunrise over toast"></textarea></label>
  <fieldset role="group" style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem">
    <label>Size<input name="size" value="1024x1024" placeholder="1024x1024"></label>
    <label>Steps<input name="steps" type="number" min="1" max="100" value="{default_steps}"></label>
    <label>Seed<input name="seed" type="number" min="0" placeholder="random"></label>
  </fieldset>
  <button type="submit">Generate</button><small class="status" id="s"></small>
  <progress id="p" hidden></progress>
</form>
<figure id="fig" hidden><img id="out" alt="Generated image"><figcaption id="cap"></figcaption></figure>
<footer><small>Requests queue and render one at a time. The model loads on the first picture and unloads after a while of quiet.{(" Pictures are kept in <code>" + escape(str(assets_dir)) + "</code>.") if assets_dir else ""}</small></footer>
</main>
<script>
document.getElementById('base').textContent = location.origin + '/v1';
const f=document.getElementById('f'), s=document.getElementById('s'), p=document.getElementById('p'), out=document.getElementById('out'), fig=document.getElementById('fig'), cap=document.getElementById('cap');
f.onsubmit=async e=>{{e.preventDefault(); const d=Object.fromEntries(new FormData(f)); const body={{prompt:d.prompt,size:d.size,steps:+d.steps||undefined,seed:d.seed!==''?+d.seed:undefined}};
f.querySelector('button').disabled=true; p.hidden=false; s.textContent='rendering…'; const t0=Date.now(); const tick=setInterval(()=>s.textContent='rendering… '+Math.round((Date.now()-t0)/1000)+' s',500);
try{{const r=await fetch('/v1/images/generations',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}}); const j=await r.json(); clearInterval(tick);
if(!r.ok){{const x=j.detail; s.textContent=typeof x==='string'?x:Array.isArray(x)?x.map(y=>y.msg||JSON.stringify(y)).join('; '):(x?JSON.stringify(x):'failed ('+r.status+')');return;}}
out.src='data:image/png;base64,'+j.data[0].b64_json; fig.hidden=false; cap.textContent=d.prompt+' — '+j.crayoncloud.width+'×'+j.crayoncloud.height+', '+j.crayoncloud.steps+' steps, seed '+j.data[0].seed+', '+j.crayoncloud.seconds+' s'+(j.data[0].file?' — saved as '+j.data[0].file.split('/').pop():''); s.textContent='';}}
catch(err){{clearInterval(tick); s.textContent=err.message||String(err);}} finally{{p.hidden=true; f.querySelector('button').disabled=false;}} }};
</script></body></html>"""

    return app
