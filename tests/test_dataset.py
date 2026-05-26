import os, tempfile
import numpy as np
import pytest
from PIL import Image
from nutrition5k_pkg.data.dataset import Nutrition5kDataset
from nutrition5k_pkg.data.transforms import get_val_transform


def _make_dummy_frames(root, dish_id, n_frames=3):
    frame_dir = os.path.join(root, dish_id, 'frames')
    os.makedirs(frame_dir, exist_ok=True)
    paths = []
    for i in range(n_frames):
        p = os.path.join(frame_dir, f'{dish_id}_camA_{i:03d}.jpeg')
        Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)).save(p)
        paths.append(p)
    return paths


def test_dataset_length():
    with tempfile.TemporaryDirectory() as d:
        side_angle_root = os.path.join(d, 'side_angles')
        os.makedirs(side_angle_root)
        meta = {
            'dish_0000000001': {'calories': 300., 'mass': 250., 'fat': 15., 'carb': 30., 'protein': 20.},
            'dish_0000000002': {'calories': 150., 'mass': 120., 'fat': 7.,  'carb': 10., 'protein': 10.},
        }
        _make_dummy_frames(side_angle_root, 'dish_0000000001', n_frames=4)
        _make_dummy_frames(side_angle_root, 'dish_0000000002', n_frames=2)
        ds = Nutrition5kDataset(
            side_angle_root=side_angle_root,
            metadata=meta,
            dish_ids=['dish_0000000001', 'dish_0000000002'],
            mode='direct',
            transform=get_val_transform(),
        )
    assert len(ds) == 6  # 4 + 2 frames


def test_dataset_item_shapes():
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, 'side_angles')
        os.makedirs(root)
        meta = {'dish_0000000001': {'calories': 300., 'mass': 250., 'fat': 15., 'carb': 30., 'protein': 20.}}
        _make_dummy_frames(root, 'dish_0000000001', n_frames=2)
        ds = Nutrition5kDataset(root, meta, ['dish_0000000001'], mode='direct', transform=get_val_transform())
        img, labels = ds[0]
    assert img.shape == (3, 256, 256)
    assert set(labels.keys()) == {'calories', 'mass', 'fat', 'carb', 'protein'}


def test_dataset_per_gram_mode():
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, 'side_angles')
        os.makedirs(root)
        meta = {'dish_0000000001': {'calories': 300., 'mass': 250., 'fat': 15., 'carb': 30., 'protein': 20.}}
        _make_dummy_frames(root, 'dish_0000000001', n_frames=1)
        ds = Nutrition5kDataset(root, meta, ['dish_0000000001'], mode='per_gram', transform=get_val_transform())
        img, labels = ds[0]
    assert set(labels.keys()) == {'cal_per_g', 'fat_per_g', 'carb_per_g', 'protein_per_g'}
    assert labels['cal_per_g'] == pytest.approx(300.0 / 250.0)


def test_depth_dataset_shapes():
    from nutrition5k_pkg.data.depth_dataset import DepthDataset
    from nutrition5k_pkg.data.transforms import depth_to_tensor

    with tempfile.TemporaryDirectory() as d:
        overhead_root = os.path.join(d, 'realsense_overhead')
        dish_dir = os.path.join(overhead_root, 'dish_0000000001')
        os.makedirs(dish_dir)
        # RGB
        Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)).save(
            os.path.join(dish_dir, 'rgb.png'))
        # 16-bit depth
        depth_arr = np.random.randint(1000, 4000, (480, 640), dtype=np.uint16)
        Image.fromarray(depth_arr).save(os.path.join(dish_dir, 'depth_raw.png'))

        meta = {'dish_0000000001': {'calories': 200., 'mass': 180., 'fat': 8., 'carb': 20., 'protein': 15.}}
        ds = DepthDataset(
            overhead_root=overhead_root,
            metadata=meta,
            dish_ids=['dish_0000000001'],
            mode='direct',
            rgb_transform=get_val_transform(),
            depth_transform=depth_to_tensor(256),
        )
        img, labels = ds[0]

    assert img.shape == (4, 256, 256)   # 3 RGB + 1 depth channel
    assert set(labels.keys()) == {'calories', 'mass', 'fat', 'carb', 'protein'}


def test_depth_dataset_rgb_only_mode():
    """DepthDataset with depth_transform=None returns 3-channel tensor (no depth_raw.png needed)."""
    with tempfile.TemporaryDirectory() as d:
        overhead_root = os.path.join(d, 'realsense_overhead')
        dish_dir = os.path.join(overhead_root, 'dish_0000000001')
        os.makedirs(dish_dir)
        # Only rgb.png — no depth_raw.png
        Image.fromarray(np.random.randint(0,255,(480,640,3),dtype=np.uint8)).save(
            os.path.join(dish_dir, 'rgb.png'))
        meta = {'dish_0000000001': {'calories':200.,'mass':180.,'fat':8.,'carb':20.,'protein':15.}}
        from nutrition5k_pkg.data.depth_dataset import DepthDataset
        from nutrition5k_pkg.data.transforms import get_val_transform
        ds = DepthDataset(
            overhead_root=overhead_root, metadata=meta,
            dish_ids=['dish_0000000001'], mode='direct',
            rgb_transform=get_val_transform(), depth_transform=None,
        )
        img, labels = ds[0]
    assert img.shape == (3, 256, 256)


def test_depth_dataset_with_volume():
    """DepthDataset in RGB-only mode injects volume scalar into labels."""
    with tempfile.TemporaryDirectory() as d:
        overhead_root = os.path.join(d, 'realsense_overhead')
        dish_dir = os.path.join(overhead_root, 'dish_0000000001')
        os.makedirs(dish_dir)
        Image.fromarray(np.random.randint(0,255,(480,640,3),dtype=np.uint8)).save(
            os.path.join(dish_dir, 'rgb.png'))
        meta = {'dish_0000000001': {'calories':200.,'mass':180.,'fat':8.,'carb':20.,'protein':15.}}
        from nutrition5k_pkg.data.depth_dataset import DepthDataset
        from nutrition5k_pkg.data.transforms import get_val_transform
        ds = DepthDataset(
            overhead_root=overhead_root, metadata=meta,
            dish_ids=['dish_0000000001'], mode='direct',
            rgb_transform=get_val_transform(), depth_transform=None,
            volume_map={'dish_0000000001': 42.5},
        )
        img, labels = ds[0]
    assert 'volume' in labels
    assert labels['volume'].item() == pytest.approx(42.5)
    assert img.shape == (3, 256, 256)
