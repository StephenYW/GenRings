import numpy as np

from app import config
from app.stl_export import heightmap_to_stl


def synthetic_heightmap_circle(radius_frac=0.3, size=(600, 700)):
    h, w = size
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    r = min(h, w) * radius_frac
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    hm = np.zeros((h, w), dtype=np.uint16)
    hm[mask] = 40000
    return hm


def test_stl_is_watertight():
    hm = synthetic_heightmap_circle()
    mesh = heightmap_to_stl(hm)
    assert mesh.is_watertight


def test_stl_has_positive_volume_near_expected():
    hm = synthetic_heightmap_circle()
    mesh = heightmap_to_stl(hm)
    base_volume = config.FACE_WIDTH_MM * config.FACE_HEIGHT_MM * config.BASE_THICKNESS_MM
    assert mesh.volume > base_volume
    # Shouldn't be absurdly larger than base + a full relief layer.
    max_volume = config.FACE_WIDTH_MM * config.FACE_HEIGHT_MM * (config.BASE_THICKNESS_MM + config.RELIEF_MAX_MM)
    assert mesh.volume < max_volume * 1.05


def test_stl_flat_heightmap_still_watertight():
    hm = np.zeros((600, 700), dtype=np.uint16)
    mesh = heightmap_to_stl(hm)
    assert mesh.is_watertight
    expected_volume = config.FACE_WIDTH_MM * config.FACE_HEIGHT_MM * config.BASE_THICKNESS_MM
    assert abs(mesh.volume - expected_volume) / expected_volume < 0.02
