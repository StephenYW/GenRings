"""
Draw the placeholder images for the "Design with AI" option gallery: one
small shaded-silver preview per addition in app/relief_prompts.ADDITIONS,
written to static/relief/<id>.png. Each shows a simple subject (a raised
disc "head") with the addition applied, lit like metal, so the effect reads
at a glance. Rerun after adding an addition.

    python tools/make_relief_thumbs.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.relief_prompts import ADDITIONS  # noqa: E402

OUT = ROOT / "static" / "relief"
W, H = 160, 120


def subject(h):
    """A raised profile-ish blob in the middle: head + shoulders."""
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    head = np.clip(1 - ((xx - 80) / 26) ** 2 - ((yy - 52) / 30) ** 2, 0, 1) ** 0.5
    body = np.clip(1 - ((xx - 80) / 52) ** 2 - ((yy - 122) / 38) ** 2, 0, 1) ** 0.5
    shape = np.maximum(head, body)
    return np.where(shape > 0, 0.4 + 0.6 * shape, h)


def mask():
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    head = ((xx - 80) / 26) ** 2 + ((yy - 52) / 30) ** 2 < 1
    body = ((xx - 80) / 52) ** 2 + ((yy - 122) / 38) ** 2 < 1
    return head | body


def background(kind):
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    rng = np.random.default_rng(3)
    if kind == "stippled":
        h = np.zeros((H, W), np.float32)
        for x, y in rng.uniform([0, 0], [W, H], (420, 2)):
            cv2.circle(h, (int(x), int(y)), 1, 0.3, -1)
        return h
    if kind == "brushed":
        return 0.12 * (np.sin(yy * 1.3 + rng.normal(0, 0.4, (H, 1))) > 0.3)
    if kind == "sunburst":
        a = np.arctan2(yy - 60, xx - 80)
        return 0.18 * (np.sin(a * 22) > 0.2)
    if kind == "guilloche":
        r = np.hypot(xx - 80, yy - 60)
        return 0.15 * (np.sin(r * 0.9 + 3 * np.sin(np.arctan2(yy - 60, xx - 80) * 9)) > 0.6)
    if kind == "hammered":
        n = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 3.0)
        return 0.35 * (n - n.min()) / (np.ptp(n) or 1)
    if kind == "stars":
        h = np.zeros((H, W), np.float32)
        for x, y in rng.uniform([6, 6], [W - 6, H - 6], (14, 2)):
            pts = [(x + (6 if k % 2 == 0 else 2.5) * math.cos(-math.pi / 2 + k * math.pi / 5),
                    y + (6 if k % 2 == 0 else 2.5) * math.sin(-math.pi / 2 + k * math.pi / 5)) for k in range(10)]
            cv2.fillPoly(h, [np.array(pts, np.int32)], 0.3)
        return h
    return np.zeros((H, W), np.float32)


def thumb(a) -> np.ndarray:
    h = background(a.id) if a.group == "background" else background("hammered") * 0.4
    m = mask()
    h = np.where(m, 0, h)
    h = subject(h) if a.id != "no_background" else subject(np.zeros_like(h))
    if a.id == "deep_shading":
        h = np.where(m, h ** 0.5, h)
    if a.id == "soft_shading":
        h = cv2.GaussianBlur(h, (0, 0), 3)
    if a.id == "outline":
        edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        h = np.where(edge, h - 0.3, h)
    if a.id == "beaded_border":
        for k in range(36):
            t = 2 * math.pi * k / 36
            cv2.circle(h, (int(80 + 72 * math.cos(t)), int(60 + 54 * math.sin(t))), 3, 0.7, -1)
    if a.id == "laurel":
        for side in (-1, 1):
            for k in range(9):
                t = math.pi / 2 + side * (0.35 + k * 0.26)
                cv2.ellipse(h, (int(80 + 64 * math.cos(t)), int(60 + 50 * math.sin(t))), (7, 3),
                            math.degrees(t) + 90, 0, 360, 0.75, -1)
    # shade like polished silver (light from the top left)
    h = cv2.GaussianBlur(h.astype(np.float32), (0, 0), 0.8) * 12
    gy, gx = np.gradient(h)
    n = np.dstack([-gx, -gy, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    light = np.array([-0.5, -0.6, 0.62]) / np.linalg.norm([-0.5, -0.6, 0.62])
    s = np.clip(n @ light, 0, 1) ** 1.4
    img = np.clip(40 + 215 * s, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for a in ADDITIONS.values():
        cv2.imwrite(str(OUT / f"{a.id}.png"), thumb(a))
    print(f"wrote {len(ADDITIONS)} thumbnails to {OUT}")


if __name__ == "__main__":
    main()
