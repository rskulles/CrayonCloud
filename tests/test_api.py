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
