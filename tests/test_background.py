"""Background removal: the subject mask flattens the background and lifts the subject."""
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app import background, config, storage
from app.main import app
from app.processing import apply_subject_mask


def test_subject_mask_flattens_background_and_raises_subject():
    gray = np.random.default_rng(0).random((60, 80)).astype(np.float32)
    mask = np.zeros_like(gray)
    mask[15:45, 20:60] = 1.0
    out = apply_subject_mask(gray, mask, base=0.2)
    assert np.all(out[mask == 0] == 0)                       # flat background
    inside = out[mask == 1]
    assert inside.min() >= 0.2 - 1e-6 and inside.max() == np.float32(1.0)  # plateau, full detail range
    assert np.all(apply_subject_mask(gray, np.zeros_like(gray), 0.2) == 0)  # no subject: blank


def test_process_with_background_removed(monkeypatch):
    # an image with a bright square subject on a mid-grey background; the
    # "model" says the subject is the square (no real model needed)
    img = np.full((400, 400, 3), 120, np.uint8)
    img[120:280, 120:280] = 230
    calls = []

    def fake_mask(rgb):
        calls.append(rgb.shape)
        m = np.zeros(rgb.shape[:2], np.uint8)
        m[120:280, 120:280] = 255
        return m

    monkeypatch.setattr(background, "subject_mask", fake_mask)
    client = TestClient(app)
    ok, buf = cv2.imencode(".png", img)
    cid = client.post("/api/upload", files={"file": ("a.png", buf.tobytes(), "image/png")}).json()["candidate_id"]

    for _ in range(2):  # the mask is computed once and then read from the cache
        res = client.post("/api/process", json={"candidate_id": cid, "remove_background": True,
                                                "levels": 0, "blur_mm": 0.0, "min_feature_mm": 0.1})
        assert res.status_code == 200, res.text
    assert len(calls) == 1
    assert (storage.design_dir(cid) / "subject_mask.png").exists()

    hm = cv2.imread(str(storage.design_dir(cid) / "heightmap.png"), cv2.IMREAD_UNCHANGED)
    h, w = hm.shape
    corner = hm[: h // 8, : w // 8]
    centre = hm[h // 2 - 5: h // 2 + 5, w // 2 - 5: w // 2 + 5]
    assert corner.max() == 0                     # background flattened
    assert centre.min() > 0                      # subject raised
    params = storage.read_json(storage.design_dir(cid) / "params.json")
    assert params["processing"]["remove_background"] is True
