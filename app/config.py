"""
All tunable constants for the Ring Face Relief Designer live here.
Change these to fit a different ring face, resolution, or material.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Ring face geometry ---------------------------------------------------
# The relief covers the whole curved top of the ring head, out to the rim where
# it meets the shoulders. Its bounding box and outline come from the prepared model (static/models/ring_face.json,
# written by tools/prepare_ring.py) so the backend and 3D viewer can't drift
# apart. The fallback is only used if that file hasn't been generated.
import json

_FACE_INFO_PATH = Path(__file__).resolve().parent.parent / "static" / "models" / "ring_face.json"
try:
    _face_info = json.loads(_FACE_INFO_PATH.read_text())
except (OSError, ValueError):
    _face_info = {}

FACE_WIDTH_MM = float(_face_info.get("face_width_mm", 14.0))
FACE_HEIGHT_MM = float(_face_info.get("face_height_mm", 12.0))
# Outline of the face in mm, centred on the bounding box (x, z); None if unknown.
FACE_OUTLINE_MM = _face_info.get("outline_xz_mm")

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

# --- Material ---------------------------------------------------
SILVER_DENSITY_G_CM3 = 10.49

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
