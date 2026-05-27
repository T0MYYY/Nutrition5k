import csv
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from nutrition5k_pkg.data.metadata import load_official_split
from nutrition5k_pkg.data.dpf_dataset import DPFNutritionDataset
from nutrition5k_pkg.data.transforms import get_dpf_rgb_transform, get_dpf_depth_transform


def _write_ids(path, ids):
    with open(path, "w") as f:
        f.write("\n".join(ids))
        f.write("\n")


def test_depth_official_split_loader_has_no_overlap():
    with tempfile.TemporaryDirectory() as d:
        train_txt = os.path.join(d, "depth_train_ids.txt")
        test_txt = os.path.join(d, "depth_test_ids.txt")
        train_ids = [f"dish_{i:010d}" for i in range(10)]
        test_ids = [f"dish_{i:010d}" for i in range(10, 13)]
        _write_ids(train_txt, train_ids)
        _write_ids(test_txt, test_ids)

        train, val, test = load_official_split(train_txt, test_txt, val_ratio=0.2, seed=123)

    assert len(train) == 8
    assert len(val) == 2
    assert test == test_ids
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))


def test_depth_official_split_loader_rejects_overlap():
    with tempfile.TemporaryDirectory() as d:
        train_txt = os.path.join(d, "depth_train_ids.txt")
        test_txt = os.path.join(d, "depth_test_ids.txt")
        _write_ids(train_txt, ["dish_0000000001", "dish_0000000002"])
        _write_ids(test_txt, ["dish_0000000002"])

        with pytest.raises(AssertionError, match="overlapping"):
            load_official_split(train_txt, test_txt)


def test_dpf_dataset_returns_synchronized_rgb_depth_and_direct_labels():
    with tempfile.TemporaryDirectory() as d:
        overhead_root = os.path.join(d, "realsense_overhead")
        depth_root = os.path.join(d, "pred_depth")
        dish_id = "dish_0000000001"
        dish_dir = os.path.join(overhead_root, dish_id)
        pred_dir = os.path.join(depth_root, dish_id)
        os.makedirs(dish_dir)
        os.makedirs(pred_dir)

        rgb_arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth_arr = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        Image.fromarray(rgb_arr).save(os.path.join(dish_dir, "rgb.png"))
        Image.fromarray(depth_arr).save(os.path.join(pred_dir, "depth_pred.png"))

        metadata = {
            dish_id: {
                "calories": 200.0,
                "mass": 180.0,
                "fat": 8.0,
                "carb": 20.0,
                "protein": 15.0,
            }
        }

        ds = DPFNutritionDataset(
            overhead_root=overhead_root,
            pred_depth_root=depth_root,
            metadata=metadata,
            dish_ids=[dish_id],
            rgb_transform=get_dpf_rgb_transform(train=False),
            depth_transform=get_dpf_depth_transform(),
        )
        rgb, depth, labels = ds[0]

    assert rgb.shape == (3, 336, 448)
    assert depth.shape == (1, 336, 448)
    assert set(labels.keys()) == {"calories", "mass", "fat", "carb", "protein"}
    assert labels["mass"].item() == pytest.approx(180.0)
