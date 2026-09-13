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
from .loras import LoraLibrary
from .service import ImageService

MIN_SIDE, MAX_SIDE, STEP = 256, 2048, 16


class LoraSpec(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    scale: float = Field(default=1.0, ge=-2.0, le=3.0)


class ImagesRequest(BaseModel):
    """OpenAI's images/generations body plus a few extras local models care about."""

    prompt: str = Field(min_length=1, max_length=4000)
    loras: list[LoraSpec] | None = Field(default=None, max_length=4)
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


def create_app(service: ImageService, default_steps: int = 8, preload: bool = False, assets_dir: Path | None = None, loras: LoraLibrary | None = None) -> FastAPI:
    library = loras or LoraLibrary(Path.home() / ".nonexistent-crayoncloud-loras")
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
        return {"ok": True, "version": __version__, "assets": None if assets_dir is None else str(assets_dir), "source": getattr(service.engine, "source", None), "loras_folder": str(library.folder), **asdict(service.status())}

    @app.get("/v1/loras")
    def list_loras():
        """The adapters in the LoRA folder, by name; ask for them with the `loras` field of a generation request."""
        return {"object": "list", "folder": str(library.folder), "data": [{"name": l.name, "file": l.file, "size": l.size} for l in library.list()]}

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
        chosen: list[tuple[str, float]] = []
        for spec in body.loras or []:
            try:
                chosen.append((str(library.resolve(spec.name)), spec.scale))
            except KeyError:
                raise HTTPException(404, f"no LoRA called {spec.name!r} in {library.folder}; see /v1/loras") from None
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
                loras=tuple(chosen),
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
                    "loras": [{"name": Path(p).stem, "scale": s} for p, s in chosen],
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
<link rel="stylesheet" href="/static/pico.classless.min.css"><link rel="icon" href="/static/icon.svg" type="image/svg+xml">
<style>
figure img{{border-radius:var(--pico-border-radius)}} .status{{margin-left:.75rem}}
header{{display:flex;align-items:center;gap:1rem}} header img{{width:4.5rem;height:4.5rem;flex:none}} header hgroup{{margin:0}}
/* The rendering indicator: three columns of dots falling like the rain on the icon. */
.rain{{display:none;gap:.55rem;height:2.2rem;margin:1rem 0 0 .2rem}} .rain.on{{display:flex}}
.rain span{{display:block;width:.55rem;height:.55rem;border-radius:50%;animation:fall 1.2s linear infinite}}
.rain i{{display:flex;flex-direction:column;gap:.35rem}} .rain i:nth-child(3n+1) span{{background:#ff4b4b}} .rain i:nth-child(3n+2) span{{background:#3ddc6a;animation-delay:.4s}} .rain i:nth-child(3n) span{{background:#4aa8ff;animation-delay:.8s}}
@keyframes fall{{0%{{opacity:0;transform:translateY(-.6rem)}}30%{{opacity:1}}100%{{opacity:0;transform:translateY(.9rem)}}}}
</style></head>
<body><main>
<header><img src="/static/icon.svg" alt=""><hgroup><h1>Crayon Cloud</h1>
<p>Serving <strong>{escape(status.model)}</strong> with the {escape(status.engine)} engine, version {__version__}.</p></hgroup></header>
<p>Point <a href="https://github.com/rskulles/ButterKnife" target="_blank" rel="noopener">ButterKnife</a>, or anything that speaks the OpenAI images API, at <code id="base">/v1</code>.</p>
<form id="f">
  <label>Prompt<textarea name="prompt" rows="3" required placeholder="A butter knife spreading a sunrise over toast"></textarea></label>
  <fieldset role="group" style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem">
    <label>Size<input name="size" value="1024x1024" placeholder="1024x1024"></label>
    <label>Steps<input name="steps" type="number" min="1" max="100" value="{default_steps}"></label>
    <label>Seed<input name="seed" type="number" min="0" placeholder="random"></label>
  </fieldset>
  {lora_picker(library)}
  <button type="submit">Generate</button><small class="status" id="s"></small>
  <div class="rain" id="p" aria-hidden="true"><i><span></span><span></span></i><i><span></span><span></span></i><i><span></span><span></span></i><i><span></span><span></span></i><i><span></span><span></span></i><i><span></span><span></span></i></div>
</form>
<figure id="fig" hidden><img id="out" alt="Generated image"><figcaption id="cap"></figcaption></figure>
<footer><small>Requests queue and render one at a time. The model loads on the first picture and unloads after a while of quiet.{(" Pictures are kept in <code>" + escape(str(assets_dir)) + "</code>.") if assets_dir else ""}</small></footer>
</main>
<script>
document.getElementById('base').textContent = location.origin + '/v1';
const addLora=document.getElementById('add-lora'); if(addLora){{addLora.onclick=()=>{{const rows=document.querySelectorAll('#loras .lora-row'); if(rows.length>=4) return; rows[rows.length-1].after(document.getElementById('lora-template').content.cloneNode(true)); addLora.disabled=document.querySelectorAll('#loras .lora-row').length>=4;}};}}
const f=document.getElementById('f'), s=document.getElementById('s'), p=document.getElementById('p'), out=document.getElementById('out'), fig=document.getElementById('fig'), cap=document.getElementById('cap');
f.onsubmit=async e=>{{e.preventDefault(); const d=Object.fromEntries(new FormData(f)); const rows=[...f.querySelectorAll('.lora-row')].map(r=>({{name:r.querySelector('select').value,scale:+r.querySelector('input').value||1}})).filter(l=>l.name);
const body={{prompt:d.prompt,size:d.size,steps:+d.steps||undefined,seed:d.seed!==''?+d.seed:undefined,loras:rows.length?rows:undefined}};
f.querySelector('button').disabled=true; p.classList.add('on'); s.textContent='rendering…'; const t0=Date.now(); const tick=setInterval(()=>s.textContent='rendering… '+Math.round((Date.now()-t0)/1000)+' s',500);
try{{const r=await fetch('/v1/images/generations',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}}); const j=await r.json(); clearInterval(tick);
if(!r.ok){{const x=j.detail; s.textContent=typeof x==='string'?x:Array.isArray(x)?x.map(y=>y.msg||JSON.stringify(y)).join('; '):(x?JSON.stringify(x):'failed ('+r.status+')');return;}}
out.src='data:image/png;base64,'+j.data[0].b64_json; fig.hidden=false; cap.textContent=d.prompt+' — '+j.crayoncloud.width+'×'+j.crayoncloud.height+', '+j.crayoncloud.steps+' steps, seed '+j.data[0].seed+', '+j.crayoncloud.seconds+' s'+(j.crayoncloud.loras.length?' — style '+j.crayoncloud.loras.map(l=>l.name+' x'+l.scale).join(', '):'')+(j.data[0].file?' — saved as '+j.data[0].file.split('/').pop():''); s.textContent='';}}
catch(err){{clearInterval(tick); s.textContent=err.message||String(err);}} finally{{p.classList.remove('on'); f.querySelector('button').disabled=false;}} }};
</script></body></html>"""

    return app


def lora_picker(library: LoraLibrary) -> str:
    """A style dropdown for the try-it page when the LoRA folder has something in it, else a hint where to put files."""
    loras = library.list()
    if not loras:
        return f'<small>No styles yet: drop a Z-Image LoRA (<code>.safetensors</code>) into <code>{escape(str(library.folder))}</code> and reload.</small>'
    options = "".join(f'<option value="{escape(l.name)}">{escape(l.name)}</option>' for l in loras)
    row = f"""<div class="lora-row" style="display:grid;grid-template-columns:2fr 1fr;gap:1rem;margin-bottom:.5rem">
      <select name="lora"><option value="">None</option>{options}</select>
      <input name="lora_scale" type="number" min="-2" max="3" step="0.1" value="1" aria-label="strength">
    </div>"""
    return f"""<fieldset id="loras">
    <legend>Styles (LoRAs) and strengths, applied together; up to four</legend>
    {row}
    <template id="lora-template">{row}</template>
    <button type="button" id="add-lora" class="secondary outline" style="width:auto;padding:.3rem .8rem;font-size:.85em">+ Another style</button>
  </fieldset>"""
