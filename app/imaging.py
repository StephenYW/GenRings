"""Small image-shape helpers shared by generation and processing."""
from __future__ import annotations

import cv2
import numpy as np


def cover_fit_resize(
    img: np.ndarray,
    target_w: int,
    target_h: int,
    zoom: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> np.ndarray:
    """
    Crop `img` to the target aspect ratio and resize to exactly (target_h,
    target_w), cover-fit style, with optional interactive zoom/pan.

    zoom=1.0, offset=(0, 0) reproduces the plain "centered cover-fit crop"
    (the largest target-aspect-ratio rectangle that fits inside the source,
    centered) -- this is what /api/generate and /api/upload store. A user
    can then zoom in (zoom > 1, crops a smaller/tighter region) and pan
    (offset_x/offset_y in -1..1, fraction of the room available to move at
    that zoom level) via /api/process without re-calling the image model,
    since the full source image is kept on disk.
    """
    h, w = img.shape[:2]
    target_aspect = target_w / target_h
    src_aspect = w / h

    # Largest target-aspect-ratio rectangle that fits inside the source
    # (this is the zoom=1.0, most-of-the-image baseline).
    if src_aspect > target_aspect:
        base_h = float(h)
        base_w = h * target_aspect
    else:
        base_w = float(w)
        base_h = w / target_aspect

    zoom = max(1.0, zoom)
    crop_w = base_w / zoom
    crop_h = base_h / zoom

    max_offset_x = max(0.0, (w - crop_w) / 2.0)
    max_offset_y = max(0.0, (h - crop_h) / 2.0)

    offset_x = float(np.clip(offset_x, -1.0, 1.0))
    offset_y = float(np.clip(offset_y, -1.0, 1.0))

    center_x = w / 2.0 + offset_x * max_offset_x
    center_y = h / 2.0 + offset_y * max_offset_y

    x0 = int(round(np.clip(center_x - crop_w / 2.0, 0, w - crop_w)))
    y0 = int(round(np.clip(center_y - crop_h / 2.0, 0, h - crop_h)))
    x1 = min(w, x0 + max(1, int(round(crop_w))))
    y1 = min(h, y0 + max(1, int(round(crop_h))))

    cropped = img[y0:y1, x0:x1]
    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)


def cap_max_dimension(img: np.ndarray, max_dim: int) -> np.ndarray:
    """Downscale `img` (preserving aspect ratio) if its longest edge exceeds max_dim."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
