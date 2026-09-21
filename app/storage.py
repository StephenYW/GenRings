"""
Filesystem-only storage: everything for a design candidate lives under
./data/designs/<uuid>/. No database.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import cv2
import numpy as np

from app import config


def new_design_dir() -> tuple[str, Path]:
    design_id = uuid.uuid4().hex
    d = config.DATA_DIR / design_id
    d.mkdir(parents=True, exist_ok=False)
    return design_id, d


def design_dir(design_id: str) -> Path:
    d = config.DATA_DIR / _safe_id(design_id)
    return d


def _safe_id(design_id: str) -> str:
    # design ids are uuid4 hex strings we generated ourselves; reject
    # anything else to avoid path traversal via a crafted id.
    if not design_id.isalnum():
        raise ValueError("invalid design id")
    return design_id


def save_rgb_png(path: Path, img_rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def save_gray16_png(path: Path, arr_u16: np.ndarray) -> None:
    cv2.imwrite(str(path), arr_u16.astype(np.uint16))


def save_gray8_png(path: Path, arr_u8: np.ndarray) -> None:
    cv2.imwrite(str(path), arr_u8.astype(np.uint8))


def make_thumbnail(img_rgb: np.ndarray, max_edge: int = 256) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    scale = max_edge / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())
