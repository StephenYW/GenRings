"""
Deterministic image -> heightmap pipeline. No AI here.

Pure functions operating on numpy arrays so they're easy to unit test with
synthetic images. The only "randomness" anywhere in this module is none:
given the same inputs, outputs are always identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from app import config


@dataclass
class ProcessParams:
    invert: bool = False
    gamma: float = 1.0          # >1 darkens midtones, <1 brightens
    contrast: float = 1.0       # 1.0 = no change
    blur_mm: float = 0.0        # Gaussian blur sigma, in mm
    levels: int = 0             # 0 or 1 = continuous (no quantization), >=2 = quantized
    min_feature_mm: float = config.MIN_FEATURE_MM
    relief_height_mm: float = 0.25
    edge_margin_mm: float = config.EDGE_MARGIN_MM
    edge_feather_mm: float = config.EDGE_FEATHER_MM


@dataclass
class ProcessReport:
    coverage_percent: float
    relief_volume_mm3: float
    estimated_weight_g: float
    warnings: list = field(default_factory=list)
    passed: bool = True


def to_grayscale(img_rgb: np.ndarray) -> np.ndarray:
    """RGB (H,W,3) uint8 -> grayscale (H,W) float32 in 0..1."""
    if img_rgb.ndim == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb
    return gray.astype(np.float32) / 255.0


def resize_to_heightmap(gray: np.ndarray, width_px: int = None, height_px: int = None) -> np.ndarray:
    width_px = width_px or config.HEIGHTMAP_WIDTH_PX
    height_px = height_px or config.HEIGHTMAP_HEIGHT_PX
    resized = cv2.resize(gray, (width_px, height_px), interpolation=cv2.INTER_AREA)
    return resized


def apply_invert(gray: np.ndarray, invert: bool) -> np.ndarray:
    if not invert:
        return gray
    return 1.0 - gray


def apply_gamma_contrast(gray: np.ndarray, gamma: float = 1.0, contrast: float = 1.0) -> np.ndarray:
    out = gray.copy()
    if contrast != 1.0:
        out = (out - 0.5) * contrast + 0.5
        out = np.clip(out, 0.0, 1.0)
    if gamma != 1.0:
        out = np.clip(out, 0.0, 1.0) ** gamma
    return np.clip(out, 0.0, 1.0)


def apply_blur(gray: np.ndarray, blur_mm: float, px_per_mm: int = config.PX_PER_MM) -> np.ndarray:
    if blur_mm <= 0:
        return gray
    sigma_px = blur_mm * px_per_mm
    # ksize must be odd and > 0; derive from sigma.
    ksize = max(3, int(round(sigma_px * 3)) | 1)
    return cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma_px, sigmaY=sigma_px)


def normalize(gray: np.ndarray) -> np.ndarray:
    """Stretch to fill 0..1. If the image is blank, returns zeros."""
    lo, hi = float(gray.min()), float(gray.max())
    if hi - lo < 1e-6:
        return np.zeros_like(gray)
    return np.clip((gray - lo) / (hi - lo), 0.0, 1.0)


def quantize(gray: np.ndarray, levels: int) -> np.ndarray:
    """Quantize to `levels` evenly spaced values in 0..1 (levels-1 steps above 0)."""
    if levels < 2:
        return gray
    step_idx = np.round(gray * (levels - 1))
    return step_idx / (levels - 1)


def _odd_kernel_from_mm(size_mm: float, px_per_mm: int = config.PX_PER_MM) -> int:
    px = int(round(size_mm * px_per_mm))
    px = max(1, px)
    if px % 2 == 0:
        px += 1
    return px


def enforce_min_feature(
    gray: np.ndarray,
    min_feature_mm: float,
    levels: int = 0,
    px_per_mm: int = config.PX_PER_MM,
) -> tuple[np.ndarray, float]:
    """
    Morphological opening (kills thin raised bits/specks) then closing
    (fills thin gaps) sized from min_feature_mm. Also drops connected
    components smaller than the min feature area.

    For quantized maps (levels >= 2), operates per-level so level boundaries
    stay crisp. Returns (cleaned, fraction_of_raised_area_removed).
    """
    if min_feature_mm <= 0:
        return gray, 0.0

    k = _odd_kernel_from_mm(min_feature_mm, px_per_mm)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    min_area_px = (min_feature_mm * px_per_mm) ** 2

    raised_before = gray > config.RAISED_THRESHOLD
    area_before = int(raised_before.sum())

    if levels is not None and levels >= 2:
        out = np.zeros_like(gray)
        # Process each distinct level value independently, highest first so
        # higher levels "win" on overlap ambiguity (there shouldn't be any,
        # values are disjoint by construction). Quantized levels are exact
        # discrete steps, so 1e-6 here is floating-point equality tolerance,
        # not a "how raised is raised" judgment call.
        unique_vals = sorted(v for v in np.unique(gray) if v > 1e-6)
        for val in unique_vals:
            mask = (np.abs(gray - val) < 1e-6).astype(np.uint8)
            mask = _clean_mask(mask, kernel, min_area_px)
            out[mask > 0] = val
        cleaned = out
    else:
        mask = (gray > config.RAISED_THRESHOLD).astype(np.uint8)
        cleaned_mask = _clean_mask(mask, kernel, min_area_px)
        cleaned = np.where(cleaned_mask > 0, gray, 0.0)

    area_after = int((cleaned > config.RAISED_THRESHOLD).sum())
    removed_fraction = 0.0 if area_before == 0 else max(0.0, (area_before - area_after) / area_before)
    return cleaned, removed_fraction


def _clean_mask(mask: np.ndarray, kernel: np.ndarray, min_area_px: float) -> np.ndarray:
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    # Remove connected components smaller than the minimum feature area.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    out = np.zeros_like(closed)
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area_px:
            out[labels == label] = 1
    return out


def edge_margin_mask(
    height_px: int,
    width_px: int,
    margin_mm: float,
    feather_mm: float,
    px_per_mm: int = config.PX_PER_MM,
    face_width_mm: float = config.FACE_WIDTH_MM,
    face_height_mm: float = config.FACE_HEIGHT_MM,
) -> np.ndarray:
    """
    Multiplicative mask (H,W) float32 in 0..1: 0 within margin_mm of the
    face edge, ramping to 1 over feather_mm, 1 in the interior.
    """
    margin_px = margin_mm * px_per_mm
    feather_px = max(feather_mm * px_per_mm, 1e-6)

    ys = np.arange(height_px).reshape(-1, 1).astype(np.float32)
    xs = np.arange(width_px).reshape(1, -1).astype(np.float32)

    dist_left = xs
    dist_right = (width_px - 1) - xs
    dist_top = ys
    dist_bottom = (height_px - 1) - ys

    dist_to_edge = np.minimum(np.minimum(dist_left, dist_right), np.minimum(dist_top, dist_bottom))

    mask = (dist_to_edge - margin_px) / feather_px
    mask = np.clip(mask, 0.0, 1.0)
    return mask.astype(np.float32)


def outline_margin_mask(
    height_px: int,
    width_px: int,
    outline_mm,
    margin_mm: float,
    feather_mm: float,
    px_per_mm: int = config.PX_PER_MM,
    face_width_mm: float = config.FACE_WIDTH_MM,
    face_height_mm: float = config.FACE_HEIGHT_MM,
) -> np.ndarray:
    """
    Like edge_margin_mask, but measured from an arbitrary face outline (list
    of (x, z) mm points centred on the face's bounding box) instead of the
    rectangle. 0 outside the outline and within margin_mm of it, ramping to 1
    over feather_mm.
    """
    pts = np.array(
        [
            [(x / face_width_mm + 0.5) * (width_px - 1), (z / face_height_mm + 0.5) * (height_px - 1)]
            for x, z in outline_mm
        ],
        dtype=np.float32,
    )
    inside = np.zeros((height_px, width_px), np.uint8)
    cv2.fillPoly(inside, [np.round(pts).astype(np.int32)], 1)
    dist_px = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    mask = (dist_px - margin_mm * px_per_mm) / max(feather_mm * px_per_mm, 1e-6)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def to_heightmap_16bit(gray01: np.ndarray, relief_height_mm: float, relief_max_mm: float = config.RELIEF_MAX_MM) -> np.ndarray:
    """gray01 in 0..1 -> uint16 array where 65535 == relief_max_mm."""
    relief_height_mm = float(np.clip(relief_height_mm, 0.0, relief_max_mm))
    scaled = gray01 * (relief_height_mm / relief_max_mm)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 65535).astype(np.uint16)


def heightmap_to_preview_8bit(heightmap_16bit: np.ndarray) -> np.ndarray:
    return (heightmap_16bit.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)


def compute_report(
    gray01_final: np.ndarray,
    min_feature_removed_fraction: float,
    margin_clipped_fraction: float,
    face_width_mm: float = config.FACE_WIDTH_MM,
    face_height_mm: float = config.FACE_HEIGHT_MM,
    px_per_mm: int = config.PX_PER_MM,
    relief_max_mm: float = config.RELIEF_MAX_MM,
    silver_density_g_cm3: float = config.SILVER_DENSITY_G_CM3,
) -> ProcessReport:
    # gray01_final is heightmap_16bit / 65535, i.e. each pixel's actual height
    # as a fraction of RELIEF_MAX_MM (not of the user's chosen relief_height_mm).
    raised = gray01_final > config.RAISED_THRESHOLD
    coverage_percent = 100.0 * raised.sum() / raised.size

    px_area_mm2 = 1.0 / (px_per_mm ** 2)
    # Multiplying by relief_max_mm (not relief_height_mm) converts each
    # pixel's 0..1 fraction back into an actual height in mm before summing.
    relief_volume_mm3 = float(gray01_final.sum() * relief_max_mm * px_area_mm2)
    volume_cm3 = relief_volume_mm3 / 1000.0
    estimated_weight_g = volume_cm3 * silver_density_g_cm3

    warnings = []
    passed = True

    if coverage_percent < config.COVERAGE_LOW_WARN_PERCENT:
        warnings.append(
            f"Raised area is only {coverage_percent:.1f}% of the face — the image may be nearly blank."
        )
        passed = False
    if coverage_percent > config.COVERAGE_HIGH_WARN_PERCENT:
        warnings.append(
            f"Raised area covers {coverage_percent:.1f}% of the face — design may look like a solid slab "
            "rather than a relief."
        )

    if min_feature_removed_fraction * 100 > config.MIN_FEATURE_REMOVED_WARN_PERCENT:
        warnings.append(
            f"{min_feature_removed_fraction * 100:.1f}% of raised detail was removed to enforce the "
            "minimum feature size — consider simplifying the design or lowering min feature size."
        )

    if margin_clipped_fraction > 0.02:
        warnings.append(
            f"~{margin_clipped_fraction * 100:.1f}% of the design touched the edge margin and was clipped — "
            "features near the border may be cut off."
        )

    return ProcessReport(
        coverage_percent=coverage_percent,
        relief_volume_mm3=relief_volume_mm3,
        estimated_weight_g=estimated_weight_g,
        warnings=warnings,
        passed=passed,
    )


def process_image(img_rgb: np.ndarray, params: ProcessParams, contains_text_request: bool = False) -> tuple[np.ndarray, np.ndarray, ProcessReport]:
    """
    Full pipeline: raw RGB image -> (heightmap_16bit, preview_8bit, report).
    """
    gray = to_grayscale(img_rgb)
    gray = resize_to_heightmap(gray)
    gray = apply_invert(gray, params.invert)
    gray = apply_gamma_contrast(gray, params.gamma, params.contrast)
    gray = apply_blur(gray, params.blur_mm)
    gray = normalize(gray)

    if params.levels and params.levels >= 2:
        gray = quantize(gray, params.levels)

    gray, min_feature_removed_fraction = enforce_min_feature(gray, params.min_feature_mm, params.levels)

    mask = edge_margin_mask(
        gray.shape[0], gray.shape[1], params.edge_margin_mm, params.edge_feather_mm
    )
    if config.FACE_OUTLINE_MM:
        mask = mask * outline_margin_mask(
            gray.shape[0], gray.shape[1], config.FACE_OUTLINE_MM, params.edge_margin_mm, params.edge_feather_mm
        )
    raised_before_margin = float((gray > config.RAISED_THRESHOLD).sum())
    gray_margined = gray * mask
    raised_after_margin = float((gray_margined > config.RAISED_THRESHOLD).sum())
    margin_clipped_fraction = (
        0.0 if raised_before_margin == 0 else max(0.0, (raised_before_margin - raised_after_margin) / raised_before_margin)
    )

    heightmap_16bit = to_heightmap_16bit(gray_margined, params.relief_height_mm)
    preview_8bit = heightmap_to_preview_8bit(heightmap_16bit)

    gray01_final = heightmap_16bit.astype(np.float32) / 65535.0
    report = compute_report(
        gray01_final,
        min_feature_removed_fraction,
        margin_clipped_fraction,
    )

    if contains_text_request:
        report.warnings.append(
            "The prompt appears to request text/lettering — this MVP's image model cannot render "
            "legible text, so letters were skipped or will look like abstract shapes."
        )

    return heightmap_16bit, preview_8bit, report
