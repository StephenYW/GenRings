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
    prompt, negative = relief_prompts.build_relief_prompt(
        "deep", ["stippled", "outline", "sunburst"], "my dog as a pirate", instruction=True, image_type="animal")
    assert prompt.startswith("Convert this image into a monochrome high-relief sculpture rendering")
    assert "my dog as a pirate" in prompt and relief_prompts.IMAGE_TYPES["animal"].prompt in prompt
    assert relief_prompts.background_choice(["stippled", "outline", "sunburst"]).id == "sunburst"  # one per group
    # a background option: the model draws the subject alone on a plain flat
    # background (the app adds the texture), never the texture itself
    assert relief_prompts.SUBJECT_ONLY in prompt and "stipple" not in prompt.lower()
    assert "outline" in prompt and relief_prompts.CASTING in prompt
    # the requested changes come before the long rules, so the model doesn't drop them
    assert prompt.index(relief_prompts.SUBJECT_ONLY) < prompt.index("Clear readable depth")
    # never ask for a coin/medal/portrait: editing models take that literally
    low = prompt.lower()
    for word in ("medallion", "coin portrait", "portrait relief", "silver"):
        assert word not in low
    assert "do not add a coin" in low and "coin" in negative
    painting, _ = relief_prompts.build_relief_prompt("classic", [], "", instruction=True, image_type="painting")
    assert "brushstrokes" in painting
    assert relief_prompts.options()["additions"][0]["id"] == "no_background"
    assert relief_prompts.options()["image_types"][0]["id"] == "auto"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RELIEF_PROVIDER", "mock")
    monkeypatch.setattr(config, "USAGE_LOG", tmp_path / "usage.jsonl")

    def fake_mask(rgb):  # the subject is the middle third (no segmentation model needed)
        h, w = rgb.shape[:2]
        m = np.zeros((h, w), np.uint8)
        m[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = 255
        return m

    from app import background
    monkeypatch.setattr(background, "subject_mask", fake_mask)
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


def test_generations_start_from_the_original_and_keep_its_aspect(client):
    img = np.zeros((300, 600, 3), np.uint8)       # 2:1 landscape
    cv2.rectangle(img, (100, 80), (500, 220), (200, 180, 160), -1)
    ok, buf = cv2.imencode(".png", img)
    src = client.post("/api/upload", files={"file": ("a.png", buf.tobytes(), "image/png")}).json()["candidate_id"]
    first = client.post("/api/relief", json={"source_id": src, "image_type": "landscape"}).json()["candidates"][0]
    # generating "from" a result still uses the original upload
    second = client.post("/api/relief", json={"source_id": first["candidate_id"]}).json()["candidates"][0]
    for c in (first, second):
        d = storage.design_dir(c["candidate_id"])
        assert storage.read_json(d / "meta.json")["parent"] == src
        h, w = cv2.imread(str(d / "candidate_full.png")).shape[:2]
        assert abs(w / h - 2.0) < 0.02
    assert client.post("/api/relief", json={"source_id": src, "image_type": "nope"}).status_code == 400


def test_background_option_gives_a_flat_textured_background(client):
    from app import textures
    src = _upload(client)
    cid = client.post("/api/relief", json={"source_id": src, "additions": ["stippled"]}).json()["candidates"][0]["candidate_id"]
    d = storage.design_dir(cid)
    meta = storage.read_json(d / "meta.json")
    assert meta["background"] == {"choice": "stippled", "texture": "stippled"}
    assert (d / "cutout_mask.png").exists() and (d / "generated.png").exists()
    # the image: subject pasted on a flat pattern (only two flat tones, no shading)
    img = cv2.imread(str(d / "candidate_full.png"), cv2.IMREAD_GRAYSCALE)
    corner = img[: img.shape[0] // 4, : img.shape[1] // 4]
    assert set(np.unique(corner)) <= set(range(160, 215)) and corner.std() > 5
    # the heightmap: background at exactly 0 or the texture's height, subject above it
    res = client.post("/api/process", json={"candidate_id": cid, "levels": 0, "blur_mm": 0, "min_feature_mm": 0.1,
                                            "smooth_mm": 0, "denoise": False})
    assert res.status_code == 200, res.text
    hm = cv2.imread(str(d / "heightmap.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535 * config.RELIEF_MAX_MM
    h, w = hm.shape
    bg = hm[: h // 5, : w // 5] / 0.25  # relief_height_mm default 0.25 -> fraction of the relief
    tex_h = config.BACKGROUND_TEXTURE_HEIGHT
    assert np.isclose(bg.min(), 0, atol=0.01) and np.isclose(bg.max(), tex_h, atol=0.02)
    assert hm[h // 2, w // 2] / 0.25 >= config.SUBJECT_BASE_LEVEL - 0.01
    assert textures.texture_map("stippled", 100, 80).shape == (80, 100)
    nb = client.post("/api/relief", json={"source_id": src, "additions": ["no_background"]}).json()["candidates"][0]
    assert storage.read_json(storage.design_dir(nb["candidate_id"]) / "meta.json")["background"]["texture"] is None


def test_bad_requests(client):
    assert client.post("/api/relief", json={"source_id": "0" * 32}).status_code == 404
    src = _upload(client)
    assert client.post("/api/relief", json={"source_id": src, "style": "nope"}).status_code == 400
    assert client.post("/api/relief", json={"source_id": src, "model": "fal-ai/nope"}).status_code == 400


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
    ("fal-ai/qwen-image-edit-2511", "image_urls", True),
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
    assert body["prompt"].startswith("Convert this image into") and "never cropped into a bust" in body["prompt"]
    assert body["seed"] == 7 and body["num_images"] == 1
    assert ("negative_prompt" in body) == negative
    if model == "fal-ai/flux-pro/kontext":
        assert body["aspect_ratio"] == "4:3"                 # nearest to the 1600x1200 source
    if model == "fal-ai/flux-kontext/dev":
        assert body["resolution_mode"] == "match_input"
    assert out.seed == 99 and out.model == model and out.rgb.shape == (64, 96, 3) and out.usd > 0


def test_fal_variations_use_the_cheaper_model_and_errors_are_reported(monkeypatch):
    monkeypatch.setattr(config, "FAL_KEY", "fal-test")
    monkeypatch.setattr(config, "FAL_MODEL", "fal-ai/flux-pro/kontext")
    monkeypatch.setattr(config, "FAL_MODEL_VARIATIONS", "fal-ai/flux-kontext/dev")
    p = relief.FalRelief()
    assert p.model_for(True) == config.FAL_MODEL_VARIATIONS and p.model_for(False) == config.FAL_MODEL
    assert p.price_usd(True) <= p.price_usd(False)
    assert relief.fal_price_usd("fal-ai/flux-pro/kontext") == 0.04
    assert relief.fal_price_usd("fal-ai/flux-2/edit", 1.0) == pytest.approx(0.024)
    p._http = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(401, json={"detail": "Invalid key"})))
    with pytest.raises(relief.ReliefError, match="Invalid key"):
        p.generate(np.zeros((100, 100, 3), np.uint8), "x", "y", 0.8, 1)


def test_picked_model_makes_singles_and_variations(client, monkeypatch):
    """The panel's model picker: offered with prices (fal only), and the picked
    model is the one called -- for variations too, so models compare like for like."""
    monkeypatch.setattr(config, "RELIEF_PROVIDER", "fal")
    monkeypatch.setattr(config, "FAL_KEY", "fal-test")
    monkeypatch.setattr(config, "FAL_MODEL", "fal-ai/flux-2/edit")
    opts = client.get("/api/relief/options").json()
    assert [c["id"] for c in opts["model_choices"]] == [m for m, _ in config.FAL_MODEL_CHOICES]
    assert opts["default_model"] == "fal-ai/flux-2/edit"
    assert all(c["price_usd"] > 0 for c in opts["model_choices"])

    called = []

    def fake_generate(self, rgb, prompt, negative, fidelity, seed, variation=False, model=None):
        used = self.model_for(variation, model)
        called.append(used)
        return relief.ReliefImage(rgb=rgb.copy(), seed=seed, credits=0, usd=0.04, model=used)

    monkeypatch.setattr(relief.FalRelief, "generate", fake_generate)
    src = _upload(client)
    r = client.post("/api/relief", json={"source_id": src, "model": "fal-ai/flux-pro/kontext", "text": "x"})
    assert r.status_code == 200 and called == ["fal-ai/flux-pro/kontext"] * config.RELIEF_IMAGES_WITH_TEXT
    called.clear()
    client.post("/api/relief", json={"source_id": src})                   # none picked: the configured model
    assert called == ["fal-ai/flux-2/edit"]


def test_no_model_choices_without_fal(client):
    opts = client.get("/api/relief/options").json()
    assert opts["model_choices"] == [] and opts["default_model"] is None


def test_keep_everything_comes_first_and_allows_requested_changes():
    """At "close" faithfulness the no-bust / nothing-added-or-removed sentence
    comes straight after the main instruction; requested changes are exempt."""
    p, _ = relief_prompts.build_relief_prompt("classic", [], "", instruction=True, fidelity=0.8)
    first, second = p.split(". ")[:2]
    assert first.startswith("Convert this image") and second.startswith("Keep everything in the original")
    assert "never cropped into a bust" in p and "remove nothing and add nothing." in p
    p, _ = relief_prompts.build_relief_prompt("classic", ["stippled"], "", instruction=True, fidelity=0.8)
    assert "Keep the subject exactly as in the original" in p and "add nothing." in p   # a background isn't a change
    p, _ = relief_prompts.build_relief_prompt("classic", ["laurel"], "as a pirate", instruction=True, fidelity=0.8)
    assert "add nothing apart from the requested changes" in p
    p, _ = relief_prompts.build_relief_prompt("classic", [], "", instruction=True, fidelity=0.3)
    assert "Keep everything" not in p and "reinterpret" in p                            # creative: the user's choice


def test_background_can_be_changed_after_generation(client):
    """Paste the subject on another background asset (or go back to the image
    as made); cached depth is dropped so depth -> heightmap re-runs."""
    src = _upload(client)
    cid = client.post("/api/relief", json={"source_id": src}).json()["candidates"][0]["candidate_id"]
    d = storage.design_dir(cid)
    as_made = cv2.imread(str(d / "candidate_full.png"))
    params = {"candidate_id": cid, "height_source": "brightness"}
    assert client.post("/api/process", json=params).json()["background"] is None
    (d / "depth.png").write_bytes(b"stale")                       # stands in for a cached depth map

    r = client.post(f"/api/designs/{cid}/background", json={"choice": "hammered"})
    assert r.status_code == 200 and r.json()["background"] == "hammered"
    assert storage.read_json(d / "meta.json")["background"] == {"choice": "hammered", "texture": "hammered"}
    assert not (d / "depth.png").exists() and (d / "cutout_mask.png").exists()
    changed = cv2.imread(str(d / "candidate_full.png"))
    assert not np.array_equal(changed, as_made)
    assert np.array_equal(cv2.imread(str(d / "generated.png")), as_made)  # the image as made is kept
    assert client.post("/api/process", json=params).json()["background"] == "hammered"

    assert client.post(f"/api/designs/{cid}/background", json={"choice": "no_background"}).status_code == 200
    assert storage.read_json(d / "meta.json")["background"]["texture"] is None
    assert client.post(f"/api/designs/{cid}/background", json={}).status_code == 200      # back to as made
    assert np.array_equal(cv2.imread(str(d / "candidate_full.png")), as_made)
    assert storage.read_json(d / "meta.json")["background"] is None

    # uploads work the same way (their own image is kept on the first change)
    assert client.post(f"/api/designs/{src}/background", json={"choice": "stars"}).status_code == 200
    assert client.post(f"/api/designs/{src}/background", json={"choice": "laurel"}).status_code == 400
    assert client.post(f"/api/designs/{'0' * 32}/background", json={"choice": "stars"}).status_code == 404
