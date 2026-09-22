import cv2
import numpy as np

from app.imaging import cap_max_dimension, cover_fit_resize


def test_cover_fit_resize_output_shape():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    out = cover_fit_resize(img, 140, 120)
    assert out.shape == (120, 140, 3)


def test_zoom_one_offset_zero_uses_full_limiting_dimension():
    # Source is wider than target after fitting to height -> the fit uses the
    # full width, so left/right edge content should both remain visible.
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:, 0:5] = [255, 0, 0]
    img[:, -5:] = [0, 255, 0]
    out = cover_fit_resize(img, 400, 200, zoom=1.0)
    assert out[:, 0:5, 0].mean() > 100  # red edge marker preserved on the left
    assert out[:, -5:, 1].mean() > 100  # green edge marker preserved on the right


def test_pan_offset_moves_visible_region_predictably():
    h, w = 300, 400
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[0:20, :] = 255  # bright strip near the top of the source image

    target_w, target_h = 400, 200  # requires vertical cropping -> offset_y has effect
    out_top = cover_fit_resize(img, target_w, target_h, zoom=1.0, offset_y=-1.0)
    out_bottom = cover_fit_resize(img, target_w, target_h, zoom=1.0, offset_y=1.0)

    # offset_y=-1 pans toward the top of the source, so the bright strip
    # should be much more visible there than when panned to the bottom.
    assert out_top.mean() > out_bottom.mean()


def test_offset_is_clamped_beyond_valid_range():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    # Should not raise, and should behave the same as the clamped extreme.
    out_extreme = cover_fit_resize(img, 400, 200, zoom=1.0, offset_y=1.0)
    out_beyond = cover_fit_resize(img, 400, 200, zoom=1.0, offset_y=50.0)
    assert np.array_equal(out_extreme, out_beyond)


def test_zoom_in_crops_a_tighter_region():
    size = 300
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), 30, (255, 255, 255), -1)

    out_wide = cover_fit_resize(img, 100, 100, zoom=1.0)
    out_tight = cover_fit_resize(img, 100, 100, zoom=2.0)

    frac_wide = (out_wide[:, :, 0] > 128).mean()
    frac_tight = (out_tight[:, :, 0] > 128).mean()
    # Zooming in on the same centered marker should make it cover a larger
    # fraction of the output frame.
    assert frac_tight > frac_wide


def test_cap_max_dimension_noop_when_already_small():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out = cap_max_dimension(img, 300)
    assert out.shape == img.shape


def test_cap_max_dimension_downscales_preserving_aspect():
    img = np.zeros((1000, 2000, 3), dtype=np.uint8)
    out = cap_max_dimension(img, 500)
    assert max(out.shape[:2]) == 500
    assert abs((out.shape[1] / out.shape[0]) - (2000 / 1000)) < 0.01
