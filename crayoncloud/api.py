"""The HTTP surface: the OpenAI images endpoints ButterKnife and LocalAI-style clients speak (generations, and edits for
image to image), a model list, a status route, and a tiny page for trying prompts by hand."""

from __future__ import annotations

import base64
import binascii
import io
import json
import random
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from html import escape

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from . import __version__
from .engines import GenerationRequest
from .loras import LoraLibrary
from .service import ImageService

MIN_SIDE, MAX_SIDE, STEP = 256, 2048, 16
DEFAULT_SIZE = (1024, 1024)
# A1111's denoising strength: 0 gives the source picture back, 1 ignores it. 0.6 changes a picture without losing it.
DEFAULT_STRENGTH = 0.6
MAX_IMAGE_BYTES = 32 * 1024 * 1024


class LoraSpec(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    scale: float = Field(default=1.0, ge=-2.0, le=3.0)


class ImagesRequest(BaseModel):
    """OpenAI's images/generations body plus a few extras local models care about."""

    prompt: str = Field(min_length=1, max_length=4000)
    loras: list[LoraSpec] | None = Field(default=None, max_length=4)
    model: str | None = None
    n: int = Field(default=1, ge=1, le=4)
    # None means 1024x1024, or the shape of `image` when there is one.
    size: str | None = None
    response_format: str = "b64_json"
    steps: int | None = Field(default=None, ge=1, le=100)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    negative_prompt: str | None = Field(default=None, max_length=4000)
    # Image to image without multipart: the source picture as base64 (a data: URL is fine too).
    image: str | None = None
    strength: float = Field(default=DEFAULT_STRENGTH, ge=0.0, le=1.0)


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


def fit_size(width: int, height: int) -> tuple[int, int]:
    """The render size for a source picture: its own shape, scaled into the allowed range and rounded to multiples of 16."""
    scale = 1.0
    if max(width, height) > MAX_SIDE:
        scale = MAX_SIDE / max(width, height)
    if min(width, height) * scale < MIN_SIDE:
        scale = MIN_SIDE / min(width, height)
    fitted = (round(width * scale / STEP) * STEP, round(height * scale / STEP) * STEP)
    return tuple(min(MAX_SIDE, max(MIN_SIDE, side)) for side in fitted)  # type: ignore[return-value]


def decode_image(data: bytes) -> Image.Image:
    """A source picture from PNG, JPEG or WebP bytes: EXIF orientation applied, RGB. ValueError when it is not a picture."""
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"the picture is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # noqa: BLE001 - Pillow raises a zoo of exceptions for bad input; the caller only needs "no"
        raise ValueError(f"not a picture I can read ({exc})") from exc
    return ImageOps.exif_transpose(image).convert("RGB")


def decode_base64_image(text: str) -> Image.Image:
    """The `image` field of a JSON request: plain base64 or a data: URL."""
    stripped = text.strip()
    if stripped.startswith("data:") and "," in stripped[:200]:
        stripped = stripped.split(",", 1)[1]
    try:
        data = base64.b64decode(stripped)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image must be base64 (or a data: URL)") from exc
    return decode_image(data)


def default_assets_dir() -> Path:
    """Where rendered pictures are kept: the Pictures folder, so they show up where people look for images."""
    return Path.home() / "Pictures" / "Crayon Cloud"


def save_asset(assets_dir: Path, image, prompt: str, seed: int, steps: int, model: str, source: Image.Image | None = None, strength: float | None = None) -> Path:
    """Writes the PNG with the prompt and settings in its text chunks, named by time and seed."""
    from PIL import PngImagePlugin

    assets_dir.mkdir(parents=True, exist_ok=True)
    info = PngImagePlugin.PngInfo()
    info.add_text("prompt", prompt)
    info.add_text("seed", str(seed))
    info.add_text("steps", str(steps))
    info.add_text("model", model)
    if source is not None:
        info.add_text("source", f"{source.width}x{source.height}")
        info.add_text("strength", f"{strength:g}")
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

    def check_common(response_format: str, model: str | None) -> None:
        if response_format != "b64_json":
            raise HTTPException(400, "only response_format=b64_json is supported (there is no place to host URLs)")
        if model and model != service.engine.model:
            raise HTTPException(404, f"this server runs {service.engine.model!r}, not {model!r}")

    def resolve_loras(specs: list[LoraSpec] | None) -> list[tuple[str, float]]:
        chosen: list[tuple[str, float]] = []
        for spec in specs or []:
            try:
                chosen.append((str(library.resolve(spec.name)), spec.scale))
            except KeyError:
                raise HTTPException(404, f"no LoRA called {spec.name!r} in {library.folder}; see /v1/loras") from None
        return chosen

    async def render(*, prompt: str, n: int, size: str | None, steps: int | None, seed: int | None, negative_prompt: str | None,
                     loras: list[tuple[str, float]], source: Image.Image | None, strength: float) -> JSONResponse:
        """The shared tail of both endpoints: pick the size, run n renders with consecutive seeds, save, answer."""
        if size:
            width, height = parse_size(size)
        elif source is not None:
            width, height = fit_size(source.width, source.height)
        else:
            width, height = DEFAULT_SIZE
        steps = steps or default_steps
        first_seed = seed if seed is not None else random.randrange(2**31 - 1)

        data = []
        seconds_total = 0.0
        for i in range(n):
            request = GenerationRequest(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                seed=first_seed + i,
                negative_prompt=negative_prompt,
                loras=tuple(loras),
                init_image=source,
                strength=strength,
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
                    entry["file"] = str(save_asset(assets_dir, image, request.prompt, request.seed, steps, service.engine.model, source, strength))
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
                    "loras": [{"name": Path(p).stem, "scale": s} for p, s in loras],
                    "source": None if source is None else {"width": source.width, "height": source.height, "strength": strength},
                },
            }
        )

    @app.post("/v1/images/generations")
    async def generations(body: ImagesRequest):
        check_common(body.response_format, body.model)
        source = None
        if body.image:
            try:
                source = decode_base64_image(body.image)
            except ValueError as exc:
                raise HTTPException(400, f"image: {exc}") from None
        return await render(prompt=body.prompt, n=body.n, size=body.size, steps=body.steps, seed=body.seed, negative_prompt=body.negative_prompt,
                            loras=resolve_loras(body.loras), source=source, strength=body.strength)

    @app.post("/v1/images/edits")
    async def edits(
        image: UploadFile = File(..., description="the picture to start from (PNG, JPEG or WebP)"),
        prompt: str = Form(..., min_length=1, max_length=4000),
        strength: float = Form(DEFAULT_STRENGTH, ge=0.0, le=1.0, description="0 gives the picture back, 1 ignores it"),
        n: int = Form(1, ge=1, le=4),
        size: str | None = Form(None, description="WxH; leave out to keep the picture's shape"),
        steps: int | None = Form(None, ge=1, le=100),
        seed: int | None = Form(None, ge=0, le=2**31 - 1),
        negative_prompt: str | None = Form(None, max_length=4000),
        loras: str | None = Form(None, description='the same JSON list as the generations body, e.g. [{"name": "watercolor", "scale": 0.8}]'),
        model: str | None = Form(None),
        response_format: str = Form("b64_json"),
        mask: UploadFile | None = File(None),
    ):
        """OpenAI's images/edits shape (multipart) used for image to image: the picture plus a prompt and a strength."""
        check_common(response_format, model)
        if mask is not None and mask.filename:
            raise HTTPException(400, "masks (inpainting) are not supported; send the picture without one")
        specs = None
        if loras and loras.strip():
            try:
                specs = TypeAdapter(list[LoraSpec]).validate_json(loras)
            except ValidationError as exc:
                raise HTTPException(400, f"loras must be a JSON list of {{name, scale}}: {exc.errors()[0].get('msg', 'invalid')}") from None
            if len(specs) > 4:
                raise HTTPException(400, "up to four LoRAs per request")
        try:
            source = decode_image(await image.read())
        except ValueError as exc:
            raise HTTPException(400, f"image: {exc}") from None
        return await render(prompt=prompt, n=n, size=size or None, steps=steps, seed=seed, negative_prompt=negative_prompt or None,
                            loras=resolve_loras(specs), source=source, strength=strength)

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
#workflow{{display:flex;gap:1.5rem;margin-bottom:1rem}} #workflow label{{margin:0}}
#i2i{{display:grid;grid-template-columns:2fr 1fr;gap:1rem;align-items:start}} #i2i small{{grid-column:1/-1;margin-top:-.5rem}}
#src{{max-height:9rem;max-width:100%;border-radius:var(--pico-border-radius);justify-self:end}}
#reuse{{width:auto;padding:.3rem .8rem;font-size:.85em;margin-top:.5rem}}
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
  <fieldset id="workflow">
    <label><input type="radio" name="mode" value="text" checked> Text to image</label>
    <label><input type="radio" name="mode" value="image"> Image to image</label>
  </fieldset>
  <div id="i2i" hidden>
    <div>
      <label>Source picture<input name="image" type="file" accept="image/png,image/jpeg,image/webp"></label>
      <label>Strength <output id="sv">{DEFAULT_STRENGTH:g}</output><input name="strength" type="range" min="0" max="1" step="0.05" value="{DEFAULT_STRENGTH:g}"></label>
    </div>
    <img id="src" alt="" hidden>
    <small>0 gives the picture back, 1 ignores it; around 0.5 to 0.7 keeps the composition and changes the rest. Leave Size blank to keep the picture's shape.</small>
  </div>
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
<figure id="fig" hidden><img id="out" alt="Generated image"><figcaption id="cap"></figcaption><button type="button" id="reuse" class="secondary outline">Use as source</button></figure>
<footer><small>Requests queue and render one at a time. The model loads on the first picture and unloads after a while of quiet.{(" Pictures are kept in <code>" + escape(str(assets_dir)) + "</code>.") if assets_dir else ""}</small></footer>
</main>
<script>
document.getElementById('base').textContent = location.origin + '/v1';
const addLora=document.getElementById('add-lora'); if(addLora){{addLora.onclick=()=>{{const rows=document.querySelectorAll('#loras .lora-row'); if(rows.length>=4) return; rows[rows.length-1].after(document.getElementById('lora-template').content.cloneNode(true)); addLora.disabled=document.querySelectorAll('#loras .lora-row').length>=4;}};}}
const f=document.getElementById('f'), s=document.getElementById('s'), p=document.getElementById('p'), out=document.getElementById('out'), fig=document.getElementById('fig'), cap=document.getElementById('cap');
const i2i=document.getElementById('i2i'), src=document.getElementById('src'), sizeBox=f.elements.size, fileBox=f.elements.image, strengthBox=f.elements.strength, sv=document.getElementById('sv'), reuse=document.getElementById('reuse');
let sourceFile=null, lastPng=null, current='text';
const mode=()=>current;
function setMode(m){{ if(m===current) return; current=m; f.elements.mode.value=m; const img=m==='image'; i2i.hidden=!img; if(img){{ if(sizeBox.value==='1024x1024') sizeBox.value=''; sizeBox.placeholder='same as the picture'; }} else {{ if(!sizeBox.value) sizeBox.value='1024x1024'; sizeBox.placeholder='1024x1024'; }} }}
for(const r of document.querySelectorAll('input[name=mode]')) r.onclick=r.onchange=()=>setMode(r.value);
function setSource(file){{ sourceFile=file; if(src.src) URL.revokeObjectURL(src.src); src.src=URL.createObjectURL(file); src.hidden=false; }}
fileBox.onchange=()=>{{ if(fileBox.files[0]) setSource(fileBox.files[0]); }};
strengthBox.oninput=()=>sv.textContent=strengthBox.value;
reuse.onclick=()=>{{ if(!lastPng) return; setSource(new File([lastPng],'crayon-cloud-result.png',{{type:'image/png'}})); fileBox.value=''; setMode('image'); window.scrollTo({{top:0,behavior:'smooth'}}); }};
f.onsubmit=async e=>{{e.preventDefault(); const d=Object.fromEntries(new FormData(f)); const rows=[...f.querySelectorAll('.lora-row')].map(r=>({{name:r.querySelector('select').value,scale:+r.querySelector('input').value||1}})).filter(l=>l.name);
const img=mode()==='image'; if(img&&!sourceFile){{ s.textContent='choose a source picture first'; return; }}
let url='/v1/images/generations', init;
if(img){{ const fd=new FormData(); fd.append('image',sourceFile,sourceFile.name||'source.png'); fd.append('prompt',d.prompt); fd.append('strength',d.strength); if(d.size) fd.append('size',d.size); if(+d.steps) fd.append('steps',d.steps); if(d.seed!=='') fd.append('seed',d.seed); if(rows.length) fd.append('loras',JSON.stringify(rows)); url='/v1/images/edits'; init={{method:'POST',body:fd}}; }}
else {{ const body={{prompt:d.prompt,size:d.size||undefined,steps:+d.steps||undefined,seed:d.seed!==''?+d.seed:undefined,loras:rows.length?rows:undefined}}; init={{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}}; }}
f.querySelector('button[type=submit]').disabled=true; p.classList.add('on'); s.textContent='rendering…'; const t0=Date.now(); const tick=setInterval(()=>s.textContent='rendering… '+Math.round((Date.now()-t0)/1000)+' s',500);
try{{const r=await fetch(url,init); const j=await r.json(); clearInterval(tick);
if(!r.ok){{const x=j.detail; s.textContent=typeof x==='string'?x:Array.isArray(x)?x.map(y=>y.msg||JSON.stringify(y)).join('; '):(x?JSON.stringify(x):'failed ('+r.status+')');return;}}
const bytes=Uint8Array.from(atob(j.data[0].b64_json),c=>c.charCodeAt(0)); lastPng=new Blob([bytes],{{type:'image/png'}});
out.src='data:image/png;base64,'+j.data[0].b64_json; fig.hidden=false; const c=j.crayoncloud;
cap.textContent=d.prompt+' — '+c.width+'×'+c.height+', '+c.steps+' steps, seed '+j.data[0].seed+', '+c.seconds+' s'+(c.source?' — from a '+c.source.width+'×'+c.source.height+' picture at strength '+c.source.strength:'')+(c.loras.length?' — style '+c.loras.map(l=>l.name+' x'+l.scale).join(', '):'')+(j.data[0].file?' — saved as '+j.data[0].file.split('/').pop():''); s.textContent='';}}
catch(err){{clearInterval(tick); s.textContent=err.message||String(err);}} finally{{p.classList.remove('on'); f.querySelector('button[type=submit]').disabled=false;}} }};
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
