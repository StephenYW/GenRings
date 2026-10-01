"""
All tunable constants for the Ring Face Relief Designer live here.
Change these to fit a different ring face, resolution, or material.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Ring face geometry ---------------------------------------------------
# Designs go on a few rings from the library ("design rings"), each over its
# own design area: on S Square, the top tilting less than an angle the viewer
# sets ("tilt"); on S SquareRidged, the recessed floor inside the ridge
# ("recess"). Each area's size (the heightmap's size) and outline come from
# static/rings/relief.json, written by tools/prepare_rings.py, so the backend
# and 3D viewer can't drift apart. The fallback is only used if that file
# hasn't been generated.
import json

_FACE_INFO_PATH = Path(__file__).resolve().parent.parent / "static" / "rings" / "relief.json"
try:
    _relief = json.loads(_FACE_INFO_PATH.read_text())
except (OSError, ValueError):
    _relief = {}
RELIEF_RINGS = _relief.get("rings", {})           # "Shape/Size" -> that ring's design-area info
DEFAULT_RELIEF_RING = _relief.get("default")
_face_info = RELIEF_RINGS.get(DEFAULT_RELIEF_RING, {})

# The default design area (the default ring's, at its default angle)
FACE_WIDTH_MM = float(_face_info.get("face_width_mm", 14.0))
FACE_HEIGHT_MM = float(_face_info.get("face_height_mm", 12.0))
# Outline of the face in mm, centred on the bounding box (x, z); None if unknown.
FACE_OUTLINE_MM = _face_info.get("outline_xz_mm")


def face_geometry(ring=None, tilt_deg=None, ridge_shift_mm=None):
    """(width_mm, height_mm, outline) of a design ring's design area. For a
    "tilt" ring, at the nearest tabulated angle to tilt_deg (its default if
    None). For a "recess" ring, with its ridge's inner wall slid outward by
    ridge_shift_mm (the viewer's "ridge wall" slider), which widens the floor
    by that much on every side. Unknown rings get the defaults."""
    info = RELIEF_RINGS.get(ring or DEFAULT_RELIEF_RING)
    if not info:
        return FACE_WIDTH_MM, FACE_HEIGHT_MM, FACE_OUTLINE_MM
    table = info.get("tilt_table")
    if table:
        deg = info.get("default_tilt_deg") if tilt_deg is None else tilt_deg
        e = min(table, key=lambda t: abs(t["deg"] - deg))
        return e["width_mm"], e["height_mm"], e["outline_xz_mm"]
    w, h, outline = info["face_width_mm"], info["face_height_mm"], info.get("outline_xz_mm")
    shift = max(0.0, float(ridge_shift_mm or 0.0))
    if shift and outline:
        sx, sz = (w + 2 * shift) / w, (h + 2 * shift) / h
        outline = [[x * sx, z * sz] for x, z in outline]
    return w + 2 * shift, h + 2 * shift, outline

# --- Heightmap resolution ---------------------------------------------------
PX_PER_MM = 50
HEIGHTMAP_WIDTH_PX = int(round(FACE_WIDTH_MM * PX_PER_MM))   
HEIGHTMAP_HEIGHT_PX = int(round(FACE_HEIGHT_MM * PX_PER_MM))  

# --- Relief / manufacturing rules ---------------------------------------------------
RELIEF_MAX_MM = 0.4
# The texture runs right to the face's edge: the 3D viewer adds a wall from the
# ring body's rim up to the displaced edge, so no fade-out is needed.
EDGE_MARGIN_MM = 0.0
EDGE_FEATHER_MM = 0.0
MIN_FEATURE_MM = 0.25

# A pixel counts as "raised" (for coverage/report/min-feature purposes) only
# above this fraction of the normalized 0..1 range. Real photos/AI images
# rarely have a perfectly flat 0 background after blur/normalize -- a tiny
# noise floor is common -- so a hard >0 threshold would count that noise as
# "raised" and wildly overstate coverage and edge-margin clipping.
RAISED_THRESHOLD = 0.02

# --- Background removal ---------------------------------------------------
# rembg model used to find the main subject (see app/background.py).
# "isnet-general-use" (~180 MB) is a good all-rounder; "birefnet-general"
# (~1 GB) gives finer edges; "u2net" (~170 MB) is the classic default.
BG_REMOVAL_MODEL = os.getenv("BG_REMOVAL_MODEL", "isnet-general-use")
# With the background removed, the subject sits on a raised plateau: its
# lowest point is this fraction of the full relief height above the
# (flat, zero) background, so its silhouette always reads.
SUBJECT_BASE_LEVEL = 0.2

# --- Material ---------------------------------------------------
# 935 silver (93.5% silver, the rest copper): 1 / (0.935/10.49 + 0.065/8.96) g/cm3
SILVER_ALLOY = "935"
SILVER_DENSITY_G_CM3 = 10.37

# --- STL export (stretch milestone) ---------------------------------------------------
BASE_THICKNESS_MM = 1.0

# --- 3D preview ring geometry (visual only -- not a manufacturing file) ---------------
# The exported heightmap/STL only ever describe the flat face (above); these
# constants exist purely so the viewer can render that face attached to a
# plausible ring band/shank, to make the face's small real-world scale
# obvious. A manufacturer determines actual shank/finger-size geometry
# separately -- only the face relief ships to them.
RING_DIAMETER_MM = 18.0       # finger-hole diameter, ~US size 8
RING_BAND_THICKNESS_MM = 1.8  # band tube diameter

# --- Interactive crop (pan/zoom) ---------------------------------------------------
# How much of the source image maps onto the face is adjustable per design
# (see /api/process crop_* params) rather than fixed at generation time, so
# a user can pick/zoom/pan without re-calling the image model.
CROP_ZOOM_MIN = 1.0
CROP_ZOOM_MAX = 6.0
# Cap on the stored full-resolution candidate image so pan/zoom stays sharp
# without keeping huge uploaded photos on disk uncompressed.
MAX_FULL_IMAGE_DIM_PX = 1600

# --- Storage ---------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "designs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = BASE_DIR / "static"

# --- Image generation provider ---------------------------------------------------
IMAGE_PROVIDER = os.environ.get("IMAGE_PROVIDER", "mock").lower()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-2.5-flare")
OPENAI_IMAGE_QUALITY = os.environ.get("OPENAI_IMAGE_QUALITY", "medium")  # low | medium | high | xhigh | max | auto

# Candidate images are generated somewhat larger than the heightmap for
# quality, then cover-fit resized down to the face aspect ratio.
CANDIDATE_GEN_LONG_EDGE_PX = 1024

# --- Report thresholds ---------------------------------------------------
COVERAGE_LOW_WARN_PERCENT = 2.0
COVERAGE_HIGH_WARN_PERCENT = 85.0
MIN_FEATURE_REMOVED_WARN_PERCENT = 15.0  # % of raised area lost to opening/closing/despeckle
