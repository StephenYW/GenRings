"""
Background removal: find the main subject of a candidate image so the
background can be flattened and the design focuses on the subject.

Uses rembg (an ONNX segmentation model, BG_REMOVAL_MODEL, run locally on the
CPU; the model downloads once, ~180 MB, to ~/.rembg). Each candidate's mask
is computed once, on its full uncropped image, and cached next to it as
subject_mask.png, so crop/zoom/pan and slider changes never re-run the model.
"""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from app import config

_session = None
_lock = threading.Lock()


class BackgroundRemovalUnavailable(RuntimeError):
    """rembg isn't installed (or its model can't be loaded)."""


def _get_session():
    global _session
    with _lock:
        if _session is None:
            try:
                from rembg import new_session
            except ImportError as err:  # pragma: no cover - depends on the environment
                raise BackgroundRemovalUnavailable(
                    "Background removal needs the rembg package: pip install 'rembg[cpu]'"
                ) from err
            _session = new_session(config.BG_REMOVAL_MODEL)
        return _session


def subject_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Soft mask (H,W) uint8 of the image's main subject: 255 = subject, 0 = background."""
    from rembg import remove

    mask = remove(img_rgb, session=_get_session(), only_mask=True)
    return np.asarray(mask, dtype=np.uint8)


def solid_subject_mask(img_rgb: np.ndarray, soft: np.ndarray | None = None) -> np.ndarray:
    """Mask (H,W) uint8 of the subject, solid inside with only a 1-2 px soft
    edge -- for cutting a subject out and laying it on a new background. rembg
    is unsure about a grey subject on grey: its mask goes half-transparent over
    pale parts (skin, clothing), and the new background would show through
    them. So anything it marks even faintly counts as subject, holes inside the
    silhouette are filled and stray specks dropped."""
    soft = subject_mask(img_rgb) if soft is None else soft
    h, w = soft.shape
    solid = (soft > 25).astype(np.uint8)
    k = max(3, min(h, w) // 150) | 1
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    # keep the pieces rembg is confident about (the subject), not background specks
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        part = labels == i
        keep[i] = (soft[part] > 128).mean() > 0.2 or stats[i, cv2.CC_STAT_AREA] > 0.05 * h * w
    solid = keep[labels].astype(np.uint8) * 255
    # fill holes inside the silhouette that rembg half-saw as subject (pale
    # skin, clothing); real gaps -- between an arm and the body -- it scores ~0
    ff = np.pad(solid, 1)
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    holes = (255 - ff[1:-1, 1:-1]).astype(np.uint8)
    n, labels = cv2.connectedComponents(holes, connectivity=4)
    for i in range(1, n):
        hole = labels == i
        if soft[hole].mean() > 8:
            solid[hole] = 255
    return cv2.GaussianBlur(solid, (0, 0), 1.0)


def cached_subject_mask(design_dir: Path, full_rgb: np.ndarray) -> np.ndarray:
    """The candidate's subject mask, computed on first use and cached on disk."""
    path = design_dir / "subject_mask.png"
    if path.exists():
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.shape == full_rgb.shape[:2]:
            return mask
    mask = subject_mask(full_rgb)
    cv2.imwrite(str(path), mask)
    return mask
