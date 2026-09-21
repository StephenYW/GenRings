"""
All tunable constants for the Ring Face Relief Designer live here.
Change these to fit a different ring face, resolution, or material.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Ring face geometry ---------------------------------------------------
FACE_WIDTH_MM = 14.0
FACE_HEIGHT_MM = 12.0

# --- Heightmap resolution ---------------------------------------------------
PX_PER_MM = 50
HEIGHTMAP_WIDTH_PX = int(round(FACE_WIDTH_MM * PX_PER_MM))   # 700
HEIGHTMAP_HEIGHT_PX = int(round(FACE_HEIGHT_MM * PX_PER_MM))  # 600

# --- Relief / manufacturing rules ---------------------------------------------------
RELIEF_MAX_MM = 0.4
EDGE_MARGIN_MM = 1.0
EDGE_FEATHER_MM = 0.3
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
