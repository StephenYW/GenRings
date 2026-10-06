"""
Background textures for designs whose subject sits on a chosen background.

They live in static/textures/: one grayscale PNG per texture (white =
raised, black = the flat background level) and textures.json listing them
({"id", "label", "file", "mode", "tile_mm"}). The built-in ones are drawn by
tools/make_textures.py; your own can be added the same way (a PNG plus an
entry). "tile" textures repeat every tile_mm millimetres, so they're the same
density on every ring; "cover" textures stretch over the whole design area.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from app import config

TEXTURES_DIR = config.STATIC_DIR / "textures"


@lru_cache(maxsize=1)
def catalog() -> dict[str, dict]:
    path = TEXTURES_DIR / "textures.json"
    if not path.exists():
        return {}
    return {t["id"]: t for t in json.loads(path.read_text()).get("textures", [])}


@lru_cache(maxsize=32)
def _image(file: str) -> np.ndarray:
    img = cv2.imread(str(TEXTURES_DIR / file), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(TEXTURES_DIR / file)
    return img.astype(np.float32) / 255.0


def texture_map(texture_id: str, width_px: int, height_px: int, px_per_mm: int = config.PX_PER_MM) -> np.ndarray | None:
    """The texture's height pattern (0..1) over a width_px x height_px design
    area, or None if there is no such texture."""
    t = catalog().get(texture_id)
    if not t:
        return None
    img = _image(t["file"])
    if t.get("mode", "tile") == "cover":
        return cv2.resize(img, (width_px, height_px), interpolation=cv2.INTER_AREA)
    tile_px = max(4, int(round(float(t.get("tile_mm") or 4.0) * px_per_mm)))
    tile = cv2.resize(img, (tile_px, tile_px), interpolation=cv2.INTER_AREA)
    reps = (height_px // tile_px + 1, width_px // tile_px + 1)
    return np.tile(tile, reps)[:height_px, :width_px]


def flat_preview(texture_id: str | None, width_px: int, height_px: int) -> np.ndarray:
    """What the background looks like in the candidate image: a flat light
    grey, with the texture drawn as a slightly darker flat tone -- no shading,
    no shadows -- so the subject reads as pasted on top of it."""
    bg = np.full((height_px, width_px), 214.0, np.float32)
    if texture_id:
        # the same density as on the ring: the image spans the design area's width
        px_per_mm = width_px / max(config.FACE_WIDTH_MM, 1e-6)
        pattern = texture_map(texture_id, width_px, height_px, px_per_mm=max(1, int(round(px_per_mm))))
        if pattern is not None:
            bg -= 46.0 * pattern
    return np.dstack([bg] * 3).clip(0, 255).astype(np.uint8)
