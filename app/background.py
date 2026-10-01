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
