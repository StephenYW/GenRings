"""AI relief generation: prompt layers, 1-vs-4 images, usage logging, and the Stability request."""
import io

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config, relief, relief_prompts, storage
from app.main import app


def test_prompt_layers_and_one_addition_per_group():
    prompt, negative = relief_prompts.build_relief_prompt("deep", ["stippled", "outline", "sunburst"], "my dog as a pirate")
    assert "my dog as a pirate" in prompt and "silver bas-relief" in prompt
    assert relief_prompts.STYLES["deep"].prompt in prompt
    assert "sunburst" in prompt and "stippled" not in prompt     # sunburst replaced stippled (same group)
    assert "outline" in prompt
    assert "text" in negative and "color" in negative
    prompt, _ = relief_prompts.build_relief_prompt("classic", [], "")
    assert relief_prompts.DEFAULT_SUBJECT in prompt               # no text: stay with the image's own subject
    assert relief_prompts.options()["additions"][0]["id"] == "no_background"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RELIEF_PROVIDER", "mock")
    monkeypatch.setattr(config, "USAGE_LOG", tmp_path / "usage.jsonl")
    return TestClient(app)


def _upload(client):
    img = np.zeros((300, 300, 3), np.uint8)
    cv2.circle(img, (150, 150), 90, (200, 180, 160), -1)
    ok, buf = cv2.imencode(".png", img)
    return client.post("/api/upload", files={"file": ("a.png", buf.tobytes(), "image/png")}).json()["candidate_id"]


def test_presets_make_one_image_and_text_makes_four(client):
    src = _upload(client)
    res = client.post("/api/relief", json={"source_id": src, "style": "classic", "additions": ["no_background"]})
    assert res.status_code == 200, res.text
    assert len(res.json()["candidates"]) == 1
    res = client.post("/api/relief", json={"source_id": src, "text": "as a pirate", "additions": ["stars"]})
    out = res.json()
    assert len(out["candidates"]) == 4
    meta = storage.read_json(storage.design_dir(out["candidates"][0]["candidate_id"]) / "meta.json")
    assert meta["source"] == "relief" and meta["parent"] == src and meta["prompt"] == "as a pirate"
    assert len({storage.read_json(storage.design_dir(c["candidate_id"]) / "meta.json")["seed"]
                for c in out["candidates"]}) == 4                  # 4 different variations
    # usage: 5 images so far, all free in mock mode
    u = client.get("/api/usage").json()
    assert u["images"] == 5 and u["credits"] == 0 and u["provider"] == "mock"
    # the result is a normal candidate: it processes like any other
    assert client.post("/api/process", json={"candidate_id": out["candidates"][0]["candidate_id"]}).status_code == 200


def test_bad_requests(client):
    assert client.post("/api/relief", json={"source_id": "0" * 32}).status_code == 404
    src = _upload(client)
    assert client.post("/api/relief", json={"source_id": src, "style": "nope"}).status_code == 400


def test_stability_request(monkeypatch):
    """The Structure Control call: auth header, multipart image + prompt fields, image back."""
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["accept"] = request.headers["accept"]
        seen["body"] = request.read()
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (128, 128, 128)).save(buf, format="PNG")
        return httpx.Response(200, content=buf.getvalue(), headers={"seed": "42", "finish-reason": "SUCCESS"})

    monkeypatch.setattr(config, "STABILITY_API_KEY", "sk-test")
    p = relief.StabilityRelief()
    p._http = httpx.Client(transport=httpx.MockTransport(handler))
    out = p.generate(np.zeros((300, 900, 3), np.uint8), "a silver relief", "color", 0.8, 7)
    assert seen["url"].endswith("/v2beta/stable-image/control/structure")
    assert seen["auth"] == "Bearer sk-test" and seen["accept"] == "image/*"
    body = seen["body"]
    for field in (b'name="image"', b'name="prompt"', b"a silver relief", b'name="control_strength"', b"0.80",
                  b'name="negative_prompt"', b'name="seed"'):
        assert field in body
    assert out.seed == 42 and out.credits == config.STABILITY_CREDITS_PER_IMAGE and out.rgb.shape == (64, 64, 3)


def test_stability_errors_are_reported(monkeypatch):
    monkeypatch.setattr(config, "STABILITY_API_KEY", "sk-test")
    p = relief.StabilityRelief()
    p._http = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(401, json={"name": "unauthorized", "errors": ["invalid api key"]})))
    with pytest.raises(relief.ReliefError, match="invalid api key"):
        p.generate(np.zeros((100, 100, 3), np.uint8), "x", "y", 0.8, 1)


def _png_bytes(w=96, h=64):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 120, 120)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize("model,image_field,negative", [
    ("fal-ai/flux-pro/kontext", "image_url", False),
    ("fal-ai/flux-kontext/dev", "image_url", False),
    ("fal-ai/qwen-image-edit-plus", "image_urls", True),
    ("fal-ai/flux-2/edit", "image_urls", False),
])
def test_fal_request(monkeypatch, model, image_field, negative):
    """fal.ai: Key auth, JSON body with the source as a data URI, result downloaded from its URL."""
    import json as _json
    seen = {}

    def handler(request: httpx.Request):
        if request.url.host == "fal.run":
            seen["url"] = str(request.url)
            seen["auth"] = request.headers["authorization"]
            seen["body"] = _json.loads(request.read())
            return httpx.Response(200, json={"images": [{"url": "https://fal.media/files/x/out.png"}], "seed": 99})
        return httpx.Response(200, content=_png_bytes())   # the image download

    monkeypatch.setattr(config, "FAL_KEY", "fal-test")
    monkeypatch.setattr(config, "FAL_MODEL", model)
    p = relief.FalRelief()
    p._http = httpx.Client(transport=httpx.MockTransport(handler))
    prompt, neg = relief_prompts.build_relief_prompt("classic", ["no_background"], "", instruction=True, fidelity=0.8)
    out = p.generate(np.zeros((1200, 1600, 3), np.uint8), prompt, neg, 0.8, 7)
    assert seen["url"] == "https://fal.run/" + model and seen["auth"] == "Key fal-test"
    body = seen["body"]
    img = body[image_field][0] if image_field == "image_urls" else body[image_field]
    assert img.startswith("data:image/jpeg;base64,")
    assert body["prompt"].startswith("Transform this image into") and "exact composition" in body["prompt"]
    assert body["seed"] == 7 and body["num_images"] == 1
    assert ("negative_prompt" in body) == negative
    assert out.seed == 99 and out.model == model and out.rgb.shape == (64, 96, 3) and out.usd > 0


def test_fal_variations_use_the_cheaper_model_and_errors_are_reported(monkeypatch):
    monkeypatch.setattr(config, "FAL_KEY", "fal-test")
    p = relief.FalRelief()
    assert p.model_for(True) == config.FAL_MODEL_VARIATIONS and p.model_for(False) == config.FAL_MODEL
    assert p.price_usd(True) <= p.price_usd(False)
    assert relief.fal_price_usd("fal-ai/flux-pro/kontext") == 0.04
    assert relief.fal_price_usd("fal-ai/flux-2/edit", 1.0) == pytest.approx(0.024)
    p._http = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(401, json={"detail": "Invalid key"})))
    with pytest.raises(relief.ReliefError, match="Invalid key"):
        p.generate(np.zeros((100, 100, 3), np.uint8), "x", "y", 0.8, 1)
