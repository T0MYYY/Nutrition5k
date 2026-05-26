import torch
import numpy as np
from PIL import Image
from nutrition5k_pkg.data.transforms import get_train_transform, get_val_transform, depth_to_tensor


def test_train_transform_shape(rgb_image):
    t = get_train_transform()
    out = t(rgb_image)
    assert out.shape == (3, 256, 256)


def test_val_transform_shape(rgb_image):
    t = get_val_transform()
    out = t(rgb_image)
    assert out.shape == (3, 256, 256)


def test_val_transform_deterministic(rgb_image):
    t = get_val_transform()
    assert torch.equal(t(rgb_image), t(rgb_image))


def test_depth_to_tensor_shape():
    # 16-bit depth array: values in 10000 units = 1 m
    depth_arr = np.random.randint(1000, 4000, (480, 640), dtype=np.uint16)
    depth_pil = Image.fromarray(depth_arr)
    t = depth_to_tensor(target_size=256)
    out = t(depth_pil)
    assert out.shape == (1, 256, 256)
    assert out.dtype == torch.float32


def test_depth_to_tensor_normalization():
    # All pixels at camera height (35.9 cm = 3590 units) → depth_cm ≈ 35.9 → relative ≈ 0
    depth_arr = np.full((256, 256), 3590, dtype=np.uint16)
    depth_pil = Image.fromarray(depth_arr)
    t = depth_to_tensor(target_size=256)
    out = t(depth_pil)
    # Values should be near 0 (camera floor)
    assert out.abs().max() < 0.1
