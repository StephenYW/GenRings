"""
One-off asset prep: turn the studio HDRI in `3D Models/` into the file the
viewer loads for the ring's surroundings (background and reflections).

  static/env/studio_env.hdr  -- linear HDR, 2048x1024 equirectangular

The source is Poly Haven's "Monochrome Studio 02" (CC0), a real photo studio
with a white seamless backdrop, two strip softboxes and an octabox. The
panorama is turned by YAW_DEG so the white backdrop sits behind the ring from
the viewer's starting camera (three.js r160 can't rotate an environment map
at runtime).

Integer (8/16-bit) sources clip their brightest lights at white, which makes
polished metal look dull, so for those the lights are boosted up to
CLIP_BOOST x above white. True HDR sources (.exr/.hdr) are used as-is.

    python tools/prepare_skybox.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # must be set before cv2 is imported

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "3D Models" / "monochrome_studio_02_4k.exr"
OUT = ROOT / "static" / "env" / "studio_env.hdr"

SIZE = (2048, 1024)
YAW_DEG = 180.0            # turn the panorama about the vertical axis
CLIP_BOOST = 20.0          # integer sources only: clipped lights end up this many times brighter than white
BOOST_FROM, BOOST_TO = 0.6, 1.0  # sRGB luminance range over which that boost ramps in


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def load_linear(path: Path) -> np.ndarray:
    """Read the panorama as linear-light float32 BGR."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"can't read {path}")
    img = img[..., :3]
    if img.dtype.kind == "f":
        return img.astype(np.float32)
    srgb = img.astype(np.float32) / np.iinfo(img.dtype).max
    lum = srgb.mean(axis=2, keepdims=True)
    t = np.clip((lum - BOOST_FROM) / (BOOST_TO - BOOST_FROM), 0, 1)
    return srgb_to_linear(srgb) * (1 + (CLIP_BOOST - 1) * t * t * (3 - 2 * t))


def main() -> None:
    env = cv2.resize(load_linear(SRC), SIZE, interpolation=cv2.INTER_AREA)
    env = np.roll(env, int(round(YAW_DEG / 360.0 * SIZE[0])), axis=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), np.ascontiguousarray(env))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
