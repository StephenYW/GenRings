"""
AI image models that improve a candidate before it becomes a heightmap:

- depth estimation (Depth Anything V2 Small): the 3D shape of the photo, so
  height follows form (a nose, a cheek, an animal's head) rather than
  brightness (where dark hair or shadows would sink into the ring);
- super-resolution (Real-ESRGAN 2x): a cleaner, sharper source with less
  grain and compression noise, so fine detail survives processing and edges
  come out smooth.

Both run locally with onnxruntime on the CPU. Each model downloads once (to
MODELS_DIR) on first use; each candidate's result is computed on its full,
uncropped image once and cached next to it, so crop/zoom/pan and slider
changes never re-run a model. Nothing here depends on which ring the design
goes on.
"""
from __future__ import annotations

import os
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from app import config

MODELS_DIR = Path(os.getenv("SILVERSIGNAL_MODELS_DIR", Path.home() / ".cache" / "silversignal" / "models"))
MODELS = {
    # BSD/Apache-licensed ONNX exports on Hugging Face
    "depth": ("depth-anything-v2-small.onnx",
              "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"),
    "upscale": ("real-esrgan-x2.onnx",
                "https://huggingface.co/SceneWorks/real-esrgan-onnx/resolve/main/real_esrgan_x2.onnx"),
}

_sessions: dict = {}
_lock = threading.Lock()


class ModelUnavailable(RuntimeError):
    """A model couldn't be downloaded or loaded."""


def _session(name: str):
    with _lock:
        if name in _sessions:
            return _sessions[name]
        try:
            import onnxruntime as ort
        except ImportError as err:  # pragma: no cover - depends on the environment
            raise ModelUnavailable("AI image models need onnxruntime: pip install onnxruntime") from err
        filename, url = MODELS[name]
        path = MODELS_DIR / filename
        if not path.exists():
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            try:
                urllib.request.urlretrieve(url, tmp)
            except OSError as err:
                raise ModelUnavailable(f"Couldn't download the {name} model from {url}: {err}") from err
            tmp.rename(path)
        _sessions[name] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return _sessions[name]


# --- depth -----------------------------------------------------------------------

def estimate_depth(img_rgb: np.ndarray) -> np.ndarray:
    """Relative depth of the image as float32 (H,W) in 0..1, 1 = nearest.
    The model works at DEPTH_INPUT_PX on the long side (a multiple of 14)."""
    sess = _session("depth")
    h, w = img_rgb.shape[:2]
    scale = config.DEPTH_INPUT_PX / max(h, w)
    nh, nw = max(14, int(round(h * scale / 14)) * 14), max(14, int(round(w * scale / 14)) * 14)
    x = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0]
    # a light blur at the model's resolution removes its faint 14 px patch grid
    out = cv2.GaussianBlur(out.astype(np.float32), (0, 0), 1.0)
    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(out, 0.5), np.percentile(out, 99.5)
    return np.clip((out - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def cached_depth(design_dir: Path, full_rgb: np.ndarray) -> np.ndarray:
    """The candidate's depth map (0..1), computed on first use and cached as a 16-bit PNG."""
    path = design_dir / "depth.png"
    if path.exists():
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if d is not None and d.shape == full_rgb.shape[:2]:
            return d.astype(np.float32) / 65535.0
    d = estimate_depth(full_rgb)
    cv2.imwrite(str(path), np.round(d * 65535).astype(np.uint16))
    return d


# --- super-resolution ----------------------------------------------------------------

def upscale(img_rgb: np.ndarray, tile: int = 192, pad: int = 12) -> np.ndarray:
    """2x super-resolution (Real-ESRGAN), run in overlapping tiles to bound memory."""
    sess = _session("upscale")
    name = sess.get_inputs()[0].name
    h, w = img_rgb.shape[:2]
    src = img_rgb.astype(np.float32) / 255.0
    out = np.zeros((h * 2, w * 2, 3), np.float32)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y0, x0 = max(0, y - pad), max(0, x - pad)
            y1, x1 = min(h, y + tile + pad), min(w, x + tile + pad)
            res = sess.run(None, {name: src[y0:y1, x0:x1].transpose(2, 0, 1)[None]})[0][0].transpose(1, 2, 0)
            ty, tx = (y - y0) * 2, (x - x0) * 2
            th, tw = min(tile, h - y) * 2, min(tile, w - x) * 2
            out[y * 2:y * 2 + th, x * 2:x * 2 + tw] = res[ty:ty + th, tx:tx + tw]
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def cached_upscaled(design_dir: Path, full_rgb: np.ndarray) -> np.ndarray:
    """The candidate's 2x-upscaled full image, computed on first use and cached."""
    path = design_dir / "upscaled.png"
    if path.exists():
        up = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if up is not None and up.shape[:2] == (full_rgb.shape[0] * 2, full_rgb.shape[1] * 2):
            return cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
    up = upscale(full_rgb)
    cv2.imwrite(str(path), cv2.cvtColor(up, cv2.COLOR_RGB2BGR))
    return up
