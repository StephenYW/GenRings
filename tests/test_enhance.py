"""Image enhancement: bas-relief, engraving, hatching, smoothing, and the depth/upscale wiring."""
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app import enhance, storage
from app.main import app
from app.processing import add_detail, bas_relief, engrave, hatch_lines, smooth_edges, xdog_lines


def _dome_with_bumps(n=200):
    yy, xx = np.mgrid[:n, :n] / n - 0.5
    dome = np.clip(1 - 4 * (xx**2 + yy**2), 0, 1)            # big, smooth form
    bumps = 0.02 * (np.sin(xx * 120) * np.sin(yy * 120) > 0.5)  # small detail
    return (0.9 * dome + bumps).astype(np.float32), bumps


def test_bas_relief_keeps_detail_while_flattening_form():
    h, bumps = _dome_with_bumps(650)  # a real heightmap's size (~13 mm at 50 px/mm)
    out = bas_relief(h, 1.0)
    assert out.min() >= 0 and out.max() <= 1

    def detail_share(x):  # fine detail relative to the overall form
        return np.std(x - cv2.GaussianBlur(x, (0, 0), 6)) / np.ptp(cv2.GaussianBlur(x, (0, 0), 60))
    assert detail_share(out) > 2 * detail_share(h)
    assert np.array_equal(bas_relief(h, 0.0), h)              # off = unchanged


def test_xdog_finds_edges_and_hatching_follows_tone():
    lum = np.full((120, 120), 0.9, np.float32)
    lum[30:90, 30:90] = 0.1
    lines = xdog_lines(lum, px_per_mm=50)
    assert lines[60, 60] < 0.1 and lines[:, 25:35].max() > 0.5   # flat inside, line at the edge
    hatch = hatch_lines(lum, spacing_mm=0.6, angle_deg=45, px_per_mm=50)
    assert hatch[35:85, 35:85].mean() > 3 * hatch[:20, :20].mean()  # darker = more line
    h = np.full_like(lum, 0.8)
    assert engrave(h, hatch, 1.0).min() < 0.8 and np.array_equal(engrave(h, hatch, 0.0), h)


def test_smoothing_and_detail():
    step = np.zeros((50, 50), np.float32)
    step[:, 25:] = 1.0
    sm = smooth_edges(step, 0.04, px_per_mm=50)
    assert 0 < sm[25, 25] < 1 and np.array_equal(smooth_edges(step, 0.0), step)
    flat = np.full((60, 60), 0.5, np.float32)
    lum = np.random.default_rng(0).random((60, 60)).astype(np.float32)
    assert np.std(add_detail(flat, lum, 0.5)) > 0 and np.array_equal(add_detail(flat, lum, 0.0), flat)


def test_process_with_depth_and_upscale(monkeypatch):
    calls = {"depth": 0, "upscale": 0}

    def fake_depth(rgb):
        calls["depth"] += 1
        h, w = rgb.shape[:2]
        yy, xx = np.mgrid[:h, :w] / max(h, w) - 0.5
        return np.clip(1 - 4 * (xx**2 + yy**2), 0, 1).astype(np.float32)   # a dome, unlike the image

    def fake_upscale(rgb):
        calls["upscale"] += 1
        return cv2.resize(rgb, (rgb.shape[1] * 2, rgb.shape[0] * 2), interpolation=cv2.INTER_CUBIC)

    monkeypatch.setattr(enhance, "estimate_depth", fake_depth)
    monkeypatch.setattr(enhance, "upscale", fake_upscale)
    client = TestClient(app)
    img = np.full((300, 300, 3), 128, np.uint8)  # flat grey: brightness alone would give nothing
    ok, buf = cv2.imencode(".png", img)
    cid = client.post("/api/upload", files={"file": ("a.png", buf.tobytes(), "image/png")}).json()["candidate_id"]
    body = {"candidate_id": cid, "height_source": "depth", "upscale": True, "denoise": True,
            "bas_relief": 0.5, "smooth_mm": 0.04, "levels": 0, "blur_mm": 0.0, "min_feature_mm": 0.1}
    for _ in range(2):  # cached after the first call
        res = client.post("/api/process", json=body)
        assert res.status_code == 200, res.text
    assert calls == {"depth": 1, "upscale": 1}
    d = storage.design_dir(cid)
    assert (d / "depth.png").exists() and (d / "upscaled.png").exists()
    hm = cv2.imread(str(d / "heightmap.png"), cv2.IMREAD_UNCHANGED)
    h, w = hm.shape
    assert hm[h // 2, w // 2] > hm[2, 2]          # the dome's shape came from depth
    assert storage.read_json(d / "params.json")["processing"]["height_source"] == "depth"
