import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from crayoncloud.api import create_app, parse_size
from crayoncloud.engines.fake import FakeEngine
from crayoncloud.service import ImageService


@pytest.fixture
def client(tmp_path):
    engine = FakeEngine()
    service = ImageService(engine, idle_unload_seconds=None)
    app = create_app(service, default_steps=8, assets_dir=tmp_path / "assets")
    with TestClient(app) as client:
        client.engine = engine
        client.assets = tmp_path / "assets"
        yield client


def test_pictures_are_kept_with_their_prompt_in_the_metadata(client):
    body = client.post("/v1/images/generations", json={"prompt": "keep me", "size": "256x256", "seed": 9}).json()

    saved = body["data"][0]["file"]
    assert saved.startswith(str(client.assets)) and saved.endswith("-9.png")
    image = Image.open(saved)
    assert image.size == (256, 256)
    assert image.text["prompt"] == "keep me" and image.text["seed"] == "9" and image.text["steps"] == "8"
    assert client.get("/health").json()["assets"] == str(client.assets)
    assert "Pictures are kept in" in client.get("/").text

    again = client.post("/v1/images/generations", json={"prompt": "keep me", "size": "256x256", "seed": 9}).json()
    assert again["data"][0]["file"] != saved  # same second and seed still gets its own file


def test_no_assets_dir_means_nothing_is_written(tmp_path):
    app = create_app(ImageService(FakeEngine(), idle_unload_seconds=None), assets_dir=None)
    with TestClient(app) as client:
        body = client.post("/v1/images/generations", json={"prompt": "x", "size": "256x256"}).json()
        assert "file" not in body["data"][0]
        assert client.get("/health").json()["assets"] is None


def test_generation_returns_png_and_metadata(client):
    response = client.post("/v1/images/generations", json={"prompt": "a cloud raining rgb", "size": "512x320", "seed": 7, "steps": 3})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["crayoncloud"]["width"] == 512 and body["crayoncloud"]["height"] == 320
    assert body["crayoncloud"]["steps"] == 3
    assert body["data"][0]["seed"] == 7
    png = base64.b64decode(body["data"][0]["b64_json"])
    image = Image.open(io.BytesIO(png))
    assert image.size == (512, 320) and image.format == "PNG"
    request = client.engine.calls[-1]
    assert (request.prompt, request.seed, request.steps) == ("a cloud raining rgb", 7, 3)


def test_multiple_images_get_consecutive_seeds(client):
    body = client.post("/v1/images/generations", json={"prompt": "x", "n": 3, "seed": 100}).json()
    assert [d["seed"] for d in body["data"]] == [100, 101, 102]


def test_random_seed_when_none_given(client):
    body = client.post("/v1/images/generations", json={"prompt": "x"}).json()
    assert isinstance(body["data"][0]["seed"], int)


def test_rejects_bad_sizes_and_formats(client):
    assert client.post("/v1/images/generations", json={"prompt": "x", "size": "huge"}).status_code == 400
    assert client.post("/v1/images/generations", json={"prompt": "x", "size": "64x64"}).status_code == 400
    assert client.post("/v1/images/generations", json={"prompt": "x", "response_format": "url"}).status_code == 400
    assert client.post("/v1/images/generations", json={"prompt": "x", "model": "dall-e-3"}).status_code == 404
    assert client.post("/v1/images/generations", json={"prompt": ""}).status_code == 422


def test_models_and_status(client):
    models = client.get("/v1/models").json()
    assert models["data"][0]["id"] == "fake"
    status = client.get("/health").json()
    assert status["ok"] and status["engine"] == "fake" and status["busy"] is False
    client.post("/v1/images/generations", json={"prompt": "warm up"})
    assert client.get("/v1/status").json()["generated"] == 1
    page = client.get("/").text
    assert "Crayon Cloud" in page and "pico.classless.min.css" in page
    assert client.get("/static/pico.classless.min.css").status_code == 200
    assert client.get("/static/icon.svg").status_code == 200 and 'icon.svg' in page
    assert 'href="https://github.com/rskulles/ButterKnife" target="_blank"' in page


@pytest.mark.parametrize("size,expected", [("1024x1024", (1024, 1024)), ("1000x700", (992, 704)), ("768×1280", (768, 1280))])
def test_parse_size_rounds_to_multiples_of_16(size, expected):
    assert parse_size(size) == expected


def test_prebuilt_sources_are_ordered_and_only_for_known_bit_widths():
    from crayoncloud.engines.mflux_engine import prebuilt_sources

    q8 = prebuilt_sources("z-image-turbo", 8)
    assert q8[0][0] == "rskulles/z-image-turbo-mflux-q8"
    assert q8[1][0] == "mflux-community/z-image-turbo-mflux-q8" and q8[1][1]
    assert prebuilt_sources("z-image-turbo", None) == []
    assert prebuilt_sources("z-image", 8) == []
    assert prebuilt_sources("z-image-turbo", 7) == []


def test_loras_are_listed_resolved_and_passed_to_the_engine(tmp_path):
    from crayoncloud.loras import LoraLibrary

    folder = tmp_path / "loras"
    folder.mkdir()
    (folder / "Watercolor.safetensors").write_bytes(b"\0" * 10)
    (folder / "notes.txt").write_text("not a lora")
    (folder / "sketch.safetensors").write_bytes(b"\0" * 20)
    library = LoraLibrary(folder)
    assert [l.name for l in library.list()] == ["Watercolor", "sketch"]
    assert library.resolve("watercolor") == folder / "Watercolor.safetensors"
    assert library.resolve("sketch.safetensors") == folder / "sketch.safetensors"
    for bad in ("../sketch", "nope", "", "/etc/passwd"):
        with pytest.raises(KeyError):
            library.resolve(bad)

    engine = FakeEngine()
    app = create_app(ImageService(engine, idle_unload_seconds=None), assets_dir=None, loras=library)
    with TestClient(app) as client:
        listing = client.get("/v1/loras").json()
        assert [l["name"] for l in listing["data"]] == ["Watercolor", "sketch"] and listing["folder"] == str(folder)
        page = client.get("/").text
        assert 'name="lora"' in page and 'id="add-lora"' in page and "up to four" in page

        two = client.post("/v1/images/generations", json={"prompt": "x", "size": "256x256", "loras": [{"name": "watercolor", "scale": 0.7}, {"name": "sketch", "scale": 0.4}]}).json()
        assert two["crayoncloud"]["loras"] == [{"name": "Watercolor", "scale": 0.7}, {"name": "sketch", "scale": 0.4}]
        assert engine.calls[-1].loras == ((str(folder / "Watercolor.safetensors"), 0.7), (str(folder / "sketch.safetensors"), 0.4))
        assert client.get("/health").json()["loras"] == ["Watercolor x0.7", "sketch x0.4"]

        ok = client.post("/v1/images/generations", json={"prompt": "x", "size": "256x256", "loras": [{"name": "watercolor", "scale": 0.7}]})
        assert ok.status_code == 200, ok.text
        assert ok.json()["crayoncloud"]["loras"] == [{"name": "Watercolor", "scale": 0.7}]
        assert engine.calls[-1].loras == ((str(folder / "Watercolor.safetensors"), 0.7),)
        assert client.get("/health").json()["loras"] == ["Watercolor x0.7"]

        missing = client.post("/v1/images/generations", json={"prompt": "x", "size": "256x256", "loras": [{"name": "oil"}]})
        assert missing.status_code == 404 and "oil" in missing.json()["detail"]

        plain = client.post("/v1/images/generations", json={"prompt": "x", "size": "256x256"}).json()
        assert plain["crayoncloud"]["loras"] == []


def test_page_says_where_to_put_loras_when_the_folder_is_empty(tmp_path):
    from crayoncloud.loras import LoraLibrary

    app = create_app(ImageService(FakeEngine(), idle_unload_seconds=None), loras=LoraLibrary(tmp_path / "empty"))
    with TestClient(app) as client:
        assert "No styles yet" in client.get("/").text


def png_bytes(size=(512, 384), colour=(255, 0, 0)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("source,expected", [((512, 384), (512, 384)), ((320, 240), (336, 256)), ((4000, 2000), (2048, 1024)), ((100, 50), (512, 256)), ((1000, 700), (992, 704)), ((3000, 200), (2048, 256))])
def test_fit_size_keeps_the_picture_shape_inside_the_limits(source, expected):
    from crayoncloud.api import fit_size

    assert fit_size(*source) == expected


def test_edits_endpoint_starts_from_the_picture(client):
    response = client.post("/v1/images/edits", files={"image": ("photo.png", png_bytes(), "image/png")}, data={"prompt": "make it night", "strength": "0.4", "seed": "5"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["crayoncloud"]["width"], body["crayoncloud"]["height"]) == (512, 384)  # the picture's own shape
    assert body["crayoncloud"]["source"] == {"width": 512, "height": 384, "strength": 0.4}
    request = client.engine.calls[-1]
    assert request.init_image.size == (512, 384) and request.strength == 0.4 and request.prompt == "make it night"
    saved = Image.open(body["data"][0]["file"])
    assert saved.text["source"] == "512x384" and saved.text["strength"] == "0.4" and saved.text["prompt"] == "make it night"

    # Strength 0 hands the source back: the fake engine blends nothing of its own in.
    body = client.post("/v1/images/edits", files={"image": ("photo.png", png_bytes(), "image/png")}, data={"prompt": "x", "strength": "0"}).json()
    image = Image.open(io.BytesIO(base64.b64decode(body["data"][0]["b64_json"])))
    assert image.getpixel((160, 120)) == (255, 0, 0)

    # An explicit size wins over the picture's shape, and the other fields still work.
    body = client.post("/v1/images/edits", files={"image": ("photo.jpg", png_bytes(), "image/jpeg")}, data={"prompt": "x", "size": "512x256", "steps": "3", "n": "2", "seed": "10"}).json()
    assert (body["crayoncloud"]["width"], body["crayoncloud"]["height"]) == (512, 256)
    assert [d["seed"] for d in body["data"]] == [10, 11] and body["crayoncloud"]["steps"] == 3
    assert body["crayoncloud"]["source"]["strength"] == 0.6  # the default


def test_json_generations_accepts_a_base64_picture(client):
    encoded = base64.b64encode(png_bytes((100, 50))).decode("ascii")

    body = client.post("/v1/images/generations", json={"prompt": "x", "image": encoded, "strength": 0.9}).json()
    assert (body["crayoncloud"]["width"], body["crayoncloud"]["height"]) == (512, 256)  # too small: scaled up, shape kept
    assert body["crayoncloud"]["source"] == {"width": 100, "height": 50, "strength": 0.9}
    assert client.engine.calls[-1].init_image.size == (100, 50)

    body = client.post("/v1/images/generations", json={"prompt": "x", "image": "data:image/png;base64," + encoded, "size": "256x256"}).json()
    assert (body["crayoncloud"]["width"], body["crayoncloud"]["height"]) == (256, 256)

    plain = client.post("/v1/images/generations", json={"prompt": "x", "size": "256x256"}).json()
    assert plain["crayoncloud"]["source"] is None and client.engine.calls[-1].init_image is None
    assert client.post("/v1/images/generations", json={"prompt": "x"}).json()["crayoncloud"]["width"] == 1024  # no size, no picture


def test_image_to_image_rejects_bad_input(client):
    files = {"image": ("photo.png", png_bytes(), "image/png")}
    assert client.post("/v1/images/edits", files={"image": ("notes.txt", b"not a picture", "text/plain")}, data={"prompt": "x"}).status_code == 400
    assert client.post("/v1/images/edits", files=files, data={"prompt": "x", "strength": "1.5"}).status_code == 422
    assert client.post("/v1/images/edits", files=files, data={"prompt": ""}).status_code == 422
    assert client.post("/v1/images/edits", files=files, data={"prompt": "x", "size": "huge"}).status_code == 400
    assert client.post("/v1/images/edits", files=files, data={"prompt": "x", "loras": "not json"}).status_code == 400
    assert client.post("/v1/images/edits", files=files, data={"prompt": "x", "loras": '[{"name": "nope"}]'}).status_code == 404
    assert client.post("/v1/images/edits", files=files, data={"prompt": "x", "response_format": "url"}).status_code == 400
    assert client.post("/v1/images/edits", files={**files, "mask": ("m.png", png_bytes(), "image/png")}, data={"prompt": "x"}).status_code == 400
    assert client.post("/v1/images/edits", data={"prompt": "x"}).status_code == 422  # no picture at all
    assert client.post("/v1/images/generations", json={"prompt": "x", "image": "@@not base64@@"}).status_code == 400
    assert client.post("/v1/images/generations", json={"prompt": "x", "image": base64.b64encode(b"nope").decode()}).status_code == 400
    assert client.post("/v1/images/generations", json={"prompt": "x", "strength": -0.1}).status_code == 422


def test_page_can_switch_between_text_and_image_to_image(client):
    page = client.get("/").text
    assert 'name="mode"' in page and 'value="image"' in page and 'name="strength"' in page
    assert "/v1/images/edits" in page and 'id="reuse"' in page


def test_cli_generate_can_start_from_a_picture(tmp_path):
    from crayoncloud.cli import main

    source = tmp_path / "source.png"
    source.write_bytes(png_bytes((300, 200), (0, 0, 255)))
    out = tmp_path / "out.png"
    assert main(["generate", "a blue thing", "--engine", "fake", "--image", str(source), "--strength", "0", "--seed", "1", "--output", str(out)]) == 0
    image = Image.open(out)
    assert image.size == (384, 256) and image.getpixel((150, 100)) == (0, 0, 255)  # short side lifted to 256, shape kept
    assert main(["generate", "x", "--engine", "fake", "--image", str(tmp_path / "missing.png"), "--output", str(out)]) == 2
