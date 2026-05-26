import numpy as np
import pytest
from nutrition5k_pkg.volume import compute_volume_from_depth

def test_volume_flat_plate():
    """A flat depth image at exactly half camera height should give a positive volume."""
    # Camera height = 35.9 cm; 1 unit = 0.01 cm; half-height ≈ 1795 units
    depth_arr = np.full((256, 256), 1795, dtype=np.uint16)
    mask = np.ones((256, 256), dtype=bool)  # all pixels are food
    vol = compute_volume_from_depth(depth_arr, mask)
    assert vol > 0

def test_volume_zero_mask():
    depth_arr = np.full((256, 256), 1000, dtype=np.uint16)
    mask = np.zeros((256, 256), dtype=bool)  # no food pixels
    vol = compute_volume_from_depth(depth_arr, mask)
    assert vol == pytest.approx(0.0)

def test_volume_increases_with_height():
    """Shallower depth (food stacked higher) = larger volume."""
    mask = np.ones((64, 64), dtype=bool)
    low  = compute_volume_from_depth(np.full((64, 64), 3000, dtype=np.uint16), mask)
    high = compute_volume_from_depth(np.full((64, 64), 1000, dtype=np.uint16), mask)
    assert high > low
