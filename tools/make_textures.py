"""
Draw the built-in background textures: one grayscale PNG per texture in
static/textures/, plus static/textures/textures.json describing them.

A texture is a height pattern: white = raised, black = the flat background
level. In the ring it is raised by BACKGROUND_TEXTURE_HEIGHT (a fraction of
the relief height) behind the subject; it's never shaded or lit, so it stays
perfectly flat apart from that raise.

Each texture is either
  - "tile":  a seamless tile repeated across the design area, tile_mm wide
             (so its density in millimetres is the same on every ring), or
  - "cover": one image stretched over the whole design area (for patterns
             with a centre, like sunburst rays).

To add your own: put a grayscale PNG in static/textures/ and an entry in
textures.json ({"id", "label", "file", "mode", "tile_mm"}); rerunning this
script only rewrites the built-in ones.

    python tools/make_textures.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "static" / "textures"
TILE = 256       # px per tile
COVER = 1024     # px for "cover" textures


def _seamless_dots(n, radius, seed):
    """Random dots wrapped around the tile edges so it repeats seamlessly."""
    rng = np.random.default_rng(seed)
    img = np.zeros((TILE, TILE), np.float32)
    for x, y in rng.uniform(0, TILE, (n, 2)):
        for dx in (-TILE, 0, TILE):
            for dy in (-TILE, 0, TILE):
                cv2.circle(img, (int(x + dx), int(y + dy)), radius, 1.0, -1, lineType=cv2.LINE_AA)
    return img


# Feature sizes are chosen against the tile sizes below so nothing is narrower
# than ~0.3 mm on the ring: the heightmap's minimum-feature step (0.25 mm by
# default) would erase anything finer.

def stippled():          # 4 mm tile: 64 px/mm -> dots 0.36 mm across
    return _seamless_dots(46, 11, 1)


def brushed():           # 5 mm tile, 8 lines -> 0.62 mm apart, ~0.3 mm wide
    y = np.arange(TILE, dtype=np.float32)[:, None]
    return np.repeat(np.clip(np.sin(y / TILE * 2 * math.pi * 8) * 1.6, 0, 1), TILE, axis=1)


def hammered():
    rng = np.random.default_rng(5)
    img = np.zeros((TILE, TILE), np.float32)
    # overlapping shallow dimples: raised rims, sunken centres
    for x, y in rng.uniform(0, TILE, (70, 2)):
        r = rng.uniform(14, 26)
        for dx in (-TILE, 0, TILE):
            for dy in (-TILE, 0, TILE):
                yy, xx = np.ogrid[:TILE, :TILE]
                d = np.hypot(xx - (x + dx), yy - (y + dy)) / r
                img = np.maximum(img, np.where(d < 1, d ** 2, 0))
    return img


def stars():
    img = np.zeros((TILE, TILE), np.float32)
    rng = np.random.default_rng(9)
    for x, y in rng.uniform(0, TILE, (8, 2)):
        rot = rng.uniform(0, 2 * math.pi / 5)
        for dx in (-TILE, 0, TILE):
            for dy in (-TILE, 0, TILE):
                pts = [(x + dx + (24 if k % 2 == 0 else 12) * math.cos(rot - math.pi / 2 + k * math.pi / 5),
                        y + dy + (24 if k % 2 == 0 else 12) * math.sin(rot - math.pi / 2 + k * math.pi / 5))
                       for k in range(10)]
                cv2.fillPoly(img, [np.array(pts, np.int32)], 1.0, lineType=cv2.LINE_AA)
    return img


def sunburst():
    yy, xx = np.mgrid[:COVER, :COVER].astype(np.float32) - COVER / 2
    a = np.arctan2(yy, xx)
    return (np.sin(a * 28) > 0.0).astype(np.float32)


def guilloche():
    yy, xx = np.mgrid[:COVER, :COVER].astype(np.float32) - COVER / 2
    r, a = np.hypot(xx, yy), np.arctan2(yy, xx)
    wave = np.sin(r / COVER * 2 * math.pi * 20 + 2.2 * np.sin(a * 12))
    return np.clip((wave - 0.1) * 4, 0, 1)


BUILT_IN = [
    # id, label, function, mode, tile_mm (tile width on the ring for "tile")
    ("stippled", "Stippled", stippled, "tile", 4.0),
    ("brushed", "Brushed lines", brushed, "tile", 5.0),
    ("hammered", "Hammered", hammered, "tile", 6.0),
    ("stars", "Stars", stars, "tile", 7.0),
    ("sunburst", "Sunburst", sunburst, "cover", None),
    ("guilloche", "Guilloché", guilloche, "cover", None),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    catalog_path = OUT / "textures.json"
    catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else {"textures": []}
    built_ids = {b[0] for b in BUILT_IN}
    custom = [t for t in catalog.get("textures", []) if t["id"] not in built_ids]  # keep user-added ones
    entries = []
    for tid, label, fn, mode, tile_mm in BUILT_IN:
        img = fn()
        cv2.imwrite(str(OUT / f"{tid}.png"), np.round(np.clip(img, 0, 1) * 255).astype(np.uint8))
        entries.append({"id": tid, "label": label, "file": f"{tid}.png", "mode": mode, "tile_mm": tile_mm})
    catalog_path.write_text(json.dumps({"textures": entries + custom}, indent=1))
    print(f"wrote {len(entries)} textures to {OUT} (+{len(custom)} custom kept)")


if __name__ == "__main__":
    main()
