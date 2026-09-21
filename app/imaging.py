"""Small image-shape helpers shared by generation and processing."""
from __future__ import annotations

import cv2
import numpy as np


def cover_fit_resize(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize+center-crop `img` to exactly (target_h, target_w), cover-fit style."""
    h, w = img.shape[:2]
    target_aspect = target_w / target_h
    src_aspect = w / h

    if src_aspect > target_aspect:
        # source is wider than target: crop left/right after matching height
        new_h = target_h
        new_w = max(target_w, int(round(new_h * src_aspect)))
    else:
        new_w = target_w
        new_h = max(target_h, int(round(new_w / src_aspect)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (new_w - target_w) // 2
    y0 = (new_h - target_h) // 2
    return resized[y0: y0 + target_h, x0: x0 + target_w]
