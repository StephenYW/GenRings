import numpy as np
import pytest

from app import config
from app.processing import (
    ProcessParams,
    compute_report,
    edge_margin_mask,
    enforce_min_feature,
    process_image,
    quantize,
    to_heightmap_16bit,
)


def synthetic_circle_image(radius_px=100, size=(600, 700), fg=255, bg=0):
    """A single white filled circle, centered, on black. RGB uint8."""
    h, w = size
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius_px ** 2
    img[mask] = fg
    return img


def synthetic_thin_line_image(thickness_px=1, size=(600, 700)):
    """A single thin horizontal white line on black — should get removed by min-feature."""
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cy = h // 2
    img[cy: cy + thickness_px, 50:-50] = 255
    return img


def synthetic_blank_image(size=(600, 700)):
    return np.zeros((size[0], size[1], 3), dtype=np.uint8)


def synthetic_noise_image(size=(600, 700), seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size[0], size[1], 3), dtype=np.uint8)


# --- output shape / range ---------------------------------------------------

def test_output_size_matches_config():
    img = synthetic_circle_image()
    params = ProcessParams(relief_height_mm=0.3, levels=0)
    heightmap, preview, report = process_image(img, params)
    assert heightmap.shape == (config.HEIGHTMAP_HEIGHT_PX, config.HEIGHTMAP_WIDTH_PX)
    assert preview.shape == (config.HEIGHTMAP_HEIGHT_PX, config.HEIGHTMAP_WIDTH_PX)


def test_16bit_range():
    img = synthetic_circle_image(radius_px=200)
    params = ProcessParams(relief_height_mm=config.RELIEF_MAX_MM, levels=0, min_feature_mm=0.1)
    heightmap, preview, report = process_image(img, params)
    assert heightmap.dtype == np.uint16
    assert heightmap.min() >= 0
    assert heightmap.max() <= 65535
    # A big centered circle at max relief height should get close to full scale somewhere.
    assert heightmap.max() > 60000
    assert preview.dtype == np.uint8
    assert preview.max() <= 255


def test_relief_height_mm_scales_peak_value():
    img = synthetic_circle_image(radius_px=200)
    low = to_heightmap_16bit(np.ones((10, 10), dtype=np.float32), relief_height_mm=0.1)
    high = to_heightmap_16bit(np.ones((10, 10), dtype=np.float32), relief_height_mm=0.4)
    assert high.max() > low.max()
    # 0.4mm is RELIEF_MAX_MM -> should hit full 16-bit scale.
    assert high.max() == 65535


# --- edge margin ---------------------------------------------------

def test_edge_margin_is_zero_at_border():
    mask = edge_margin_mask(
        config.HEIGHTMAP_HEIGHT_PX,
        config.HEIGHTMAP_WIDTH_PX,
        margin_mm=config.EDGE_MARGIN_MM,
        feather_mm=config.EDGE_FEATHER_MM,
    )
    assert mask[0, 0] == 0.0
    assert mask[-1, -1] == 0.0
    assert mask[0, config.HEIGHTMAP_WIDTH_PX // 2] == 0.0
    # Interior (well past margin+feather) should be fully open.
    assert mask[config.HEIGHTMAP_HEIGHT_PX // 2, config.HEIGHTMAP_WIDTH_PX // 2] == 1.0


def test_full_white_image_has_zero_relief_at_edges_after_processing():
    # A circle filling most (not all) of the frame: normalize() would flatten
    # a fully uniform image to all-zero (correctly, since it has no shape
    # information), so use a large circle with visible background instead.
    img = synthetic_circle_image(radius_px=280)
    params = ProcessParams(relief_height_mm=0.3, levels=0, min_feature_mm=0.1)
    heightmap, preview, report = process_image(img, params)
    assert heightmap[0, 0] == 0
    assert heightmap[0, :].max() == 0
    assert heightmap[:, 0].max() == 0
    assert heightmap[-1, :].max() == 0
    assert heightmap[:, -1].max() == 0
    # But the interior should be raised.
    assert heightmap[heightmap.shape[0] // 2, heightmap.shape[1] // 2] > 0


# --- min feature enforcement ---------------------------------------------------

def test_thin_line_below_min_feature_is_removed():
    img = synthetic_thin_line_image(thickness_px=1)
    gray = img[:, :, 0].astype(np.float32) / 255.0
    cleaned, removed_fraction = enforce_min_feature(gray, min_feature_mm=config.MIN_FEATURE_MM)
    assert cleaned.max() == 0.0
    assert removed_fraction == 1.0


def test_large_feature_survives_min_feature_enforcement():
    img = synthetic_circle_image(radius_px=100)
    gray = img[:, :, 0].astype(np.float32) / 255.0
    cleaned, removed_fraction = enforce_min_feature(gray, min_feature_mm=config.MIN_FEATURE_MM)
    assert cleaned.max() > 0.0
    assert removed_fraction < 0.2


def test_small_speck_removed_by_connected_component_filter():
    img = np.zeros((600, 700), dtype=np.float32)
    img[300:302, 350:352] = 1.0  # 2x2 px speck, way below min feature area
    cleaned, removed_fraction = enforce_min_feature(img, min_feature_mm=config.MIN_FEATURE_MM)
    assert cleaned.max() == 0.0


# --- quantization ---------------------------------------------------

def test_quantize_yields_exactly_n_levels():
    gray = np.linspace(0, 1, 1000, dtype=np.float32).reshape(10, 100)
    for n in (2, 3, 4, 5):
        q = quantize(gray, n)
        uniques = np.unique(q)
        assert len(uniques) == n


def test_quantize_zero_or_one_is_noop():
    gray = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
    assert np.array_equal(quantize(gray, 0), gray)
    assert np.array_equal(quantize(gray, 1), gray)


def test_process_image_with_levels_quantizes_full_pipeline():
    img = synthetic_circle_image(radius_px=250)
    params = ProcessParams(relief_height_mm=0.3, levels=4, min_feature_mm=0.1)
    heightmap, preview, report = process_image(img, params)
    # The edge-margin feather band intentionally introduces a continuous
    # ramp near the border, so check level count in the interior only
    # (inset past margin + feather).
    inset_px = int(round((config.EDGE_MARGIN_MM + config.EDGE_FEATHER_MM + 0.1) * config.PX_PER_MM))
    interior = heightmap[inset_px:-inset_px, inset_px:-inset_px]
    uniques = np.unique(interior)
    # 0 (base) plus up to 3 raised levels -> at most 4 distinct values.
    assert len(uniques) <= 4


# --- report ---------------------------------------------------

def test_blank_image_triggers_low_coverage_warning():
    img = synthetic_blank_image()
    params = ProcessParams(relief_height_mm=0.3, levels=0)
    heightmap, preview, report = process_image(img, params)
    assert report.coverage_percent < config.COVERAGE_LOW_WARN_PERCENT
    assert not report.passed
    assert any("blank" in w or "only" in w for w in report.warnings)


def test_compute_report_volume_scales_by_relief_max_not_relief_height():
    # Regression test: gray01_final is heightmap/65535, i.e. each pixel's
    # actual height as a fraction of RELIEF_MAX_MM. The volume formula must
    # multiply by RELIEF_MAX_MM to recover real mm heights -- multiplying by
    # the user's chosen relief_height_mm instead (a past bug) double-scales
    # and silently understates the reported weight whenever
    # relief_height_mm < RELIEF_MAX_MM.
    gray01_final = np.full((10, 10), 0.5, dtype=np.float32)  # every pixel at 50% of RELIEF_MAX_MM
    report = compute_report(gray01_final, min_feature_removed_fraction=0.0, margin_clipped_fraction=0.0)

    px_area_mm2 = 1.0 / (config.PX_PER_MM ** 2)
    expected_volume_mm3 = 100 * 0.5 * config.RELIEF_MAX_MM * px_area_mm2
    assert report.relief_volume_mm3 == pytest.approx(expected_volume_mm3, rel=1e-6)

    expected_weight_g = (expected_volume_mm3 / 1000.0) * config.SILVER_DENSITY_G_CM3
    assert report.estimated_weight_g == pytest.approx(expected_weight_g, rel=1e-6)


def test_full_pipeline_volume_independent_of_relief_height_choice():
    # For the same source image, halving relief_height_mm should roughly
    # halve the reported volume/weight -- not leave it disproportionately
    # small (the double-scaling bug made low relief_height_mm choices report
    # far less silver than they actually use).
    img = synthetic_circle_image(radius_px=200)
    low = process_image(img, ProcessParams(relief_height_mm=0.2, levels=0, min_feature_mm=0.1))[2]
    high = process_image(img, ProcessParams(relief_height_mm=0.4, levels=0, min_feature_mm=0.1))[2]
    assert high.relief_volume_mm3 == pytest.approx(2 * low.relief_volume_mm3, rel=0.05)


def test_large_circle_reports_plausible_weight():
    img = synthetic_circle_image(radius_px=200)
    params = ProcessParams(relief_height_mm=0.3, levels=0, min_feature_mm=0.1)
    heightmap, preview, report = process_image(img, params)
    assert report.relief_volume_mm3 > 0
    assert report.estimated_weight_g > 0
    # Sanity upper bound: full face slab at max relief height in solid silver.
    max_possible_g = (
        config.FACE_WIDTH_MM * config.FACE_HEIGHT_MM * config.RELIEF_MAX_MM / 1000
    ) * config.SILVER_DENSITY_G_CM3
    assert report.estimated_weight_g < max_possible_g


def test_noisy_image_does_not_crash_and_stays_in_range():
    img = synthetic_noise_image()
    params = ProcessParams(relief_height_mm=0.3, levels=3, blur_mm=0.2, min_feature_mm=0.25)
    heightmap, preview, report = process_image(img, params)
    assert heightmap.min() >= 0
    assert heightmap.max() <= 65535


def test_invert_flips_which_side_is_raised():
    img = synthetic_circle_image(radius_px=100)
    params_normal = ProcessParams(relief_height_mm=0.3, levels=0, min_feature_mm=0.1, invert=False)
    params_inverted = ProcessParams(relief_height_mm=0.3, levels=0, min_feature_mm=0.1, invert=True)
    hm_normal, _, _ = process_image(img, params_normal)
    hm_inverted, _, _ = process_image(img, params_inverted)
    cy, cx = hm_normal.shape[0] // 2, hm_normal.shape[1] // 2
    assert hm_normal[cy, cx] > 0
    assert hm_inverted[cy, cx] == 0


def test_relief_height_mm_clamped_to_config_max():
    img = synthetic_circle_image(radius_px=200)
    params = ProcessParams(relief_height_mm=999.0, levels=0, min_feature_mm=0.1)
    heightmap, preview, report = process_image(img, params)
    assert heightmap.max() <= 65535
    assert heightmap.max() == 65535  # clamps to RELIEF_MAX_MM which maps to full scale
