"""
MockProvider: procedural test images so the app and test suite work fully
offline, with no API key. Produces images loosely themed by prompt content
(hash-seeded) so different prompts visibly differ, without any real
image understanding.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import cv2
import numpy as np

from app.providers.base import ImageProvider


class MockProvider(ImageProvider):
    name = "mock"

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        n: int = 4,
        seed: Optional[int] = None,
    ) -> list[np.ndarray]:
        base_seed = seed if seed is not None else _hash_seed(prompt)
        images = []
        for i in range(n):
            rng = np.random.default_rng(base_seed + i * 7919)
            kind = rng.integers(0, 3)
            if kind == 0:
                img = _concentric_shapes(width, height, rng)
            elif kind == 1:
                img = _noise_blobs(width, height, rng)
            else:
                img = _silhouette(width, height, rng)
            images.append(img)
        return images


def _hash_seed(prompt: str) -> int:
    h = hashlib.sha256(prompt.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def _concentric_shapes(width, height, rng) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cx, cy = width // 2 + int(rng.integers(-40, 40)), height // 2 + int(rng.integers(-30, 30))
    max_r = min(width, height) // 2 - 20
    n_rings = int(rng.integers(3, 6))
    for i in range(n_rings, 0, -1):
        r = int(max_r * i / n_rings)
        val = int(255 * (i / n_rings))
        shape = rng.integers(0, 2)
        if shape == 0:
            cv2.circle(img, (cx, cy), r, (val, val, val), -1, lineType=cv2.LINE_AA)
        else:
            pts = _regular_polygon(cx, cy, r, sides=int(rng.integers(3, 7)))
            cv2.fillPoly(img, [pts], (val, val, val), lineType=cv2.LINE_AA)
    return img


def _noise_blobs(width, height, rng) -> np.ndarray:
    small = rng.integers(0, 256, (height // 20, width // 20), dtype=np.uint8)
    noise = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=8)
    _, thresh = cv2.threshold(noise, int(rng.integers(90, 160)), 255, cv2.THRESH_BINARY)
    thresh = cv2.GaussianBlur(thresh, (0, 0), sigmaX=3)
    img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    return img


def _silhouette(width, height, rng) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cx, cy = width // 2, height // 2
    n_pts = int(rng.integers(6, 10))
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_pts))
    base_r = min(width, height) * 0.35
    pts = []
    for a in angles:
        r = base_r * (0.6 + 0.4 * rng.random())
        x = int(cx + r * np.cos(a))
        y = int(cy + r * np.sin(a))
        pts.append([x, y])
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255), lineType=cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=2)
    return img


def _regular_polygon(cx, cy, r, sides) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False) - np.pi / 2
    pts = np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1)
    return pts.astype(np.int32)
