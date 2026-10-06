"""End-to-end check that flip_h / flip_v mirror the heightmap the backend produces."""
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app import config, storage
from app.main import app


def _heightmap(client, cid, **flags):
    res = client.post("/api/process", json={"candidate_id": cid, "levels": 0, "blur_mm": 0.0, "min_feature_mm": 0.1, **flags})
    assert res.status_code == 200, res.text
    hm = cv2.imread(str(storage.design_dir(cid) / "heightmap.png"), cv2.IMREAD_UNCHANGED)
    assert hm.shape == (config.HEIGHTMAP_HEIGHT_PX, config.HEIGHTMAP_WIDTH_PX)
    return hm


def test_flip_flags_mirror_the_heightmap(monkeypatch):
    client = TestClient(app)
    # Asymmetric image so every flip changes the result.
    img = np.zeros((300, 300, 3), np.uint8)
    img[20:120, 30:200] = 255
    img[200:260, 220:280] = 180
    ok, buf = cv2.imencode(".png", img)
    up = client.post("/api/upload", files={"file": ("a.png", buf.tobytes(), "image/png")})
    assert up.status_code == 200, up.text
    cid = up.json()["candidate_id"]

    # The edge fade follows the (asymmetric) face outline and is applied after
    # the flip, so compare only the interior where that fade is 1.
    inner = (slice(150, -150), slice(150, -150))
    base = _heightmap(client, cid)
    assert base[inner].any()
    assert np.array_equal(_heightmap(client, cid, flip_v=True)[inner], base[::-1][inner])
    assert np.array_equal(_heightmap(client, cid, flip_h=True)[inner], base[:, ::-1][inner])
    assert np.array_equal(_heightmap(client, cid, flip_h=True, flip_v=True)[inner], base[::-1, ::-1][inner])
    assert not np.array_equal(_heightmap(client, cid, flip_v=True)[inner], base[inner])
