from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import csv
import random

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import Food101
from torchvision import transforms
import torchvision.transforms.functional as F_v


TARGET_FIELD = "total_calories"
METADATA_FILENAMES = ["dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"]


@dataclass
class DishSample:
    dish_id: str
    rgb_path: Path
    depth_path: Optional[Path]
    calories: float


def _read_dish_ids_from_split_file(filepath: Path) -> List[str]:
    dish_ids: List[str] = []
    with filepath.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            first = row[0].strip()
            if not first or first.lower() == "dish_id":
                continue
            dish_ids.append(first)
    return dish_ids


def _locate_split_files(split_dir: Path, split_type: str = "auto") -> Tuple[Path, Path]:
    candidates = list(split_dir.rglob("*.csv")) + list(split_dir.rglob("*.txt"))
    if split_type in {"rgb", "depth"}:
        train_name = f"{split_type}_train_ids.txt"
        test_name = f"{split_type}_test_ids.txt"
        explicit_train = next((p for p in candidates if p.name.lower() == train_name), None)
        explicit_test = next((p for p in candidates if p.name.lower() == test_name), None)
        if explicit_train is not None and explicit_test is not None:
            return explicit_train, explicit_test

    # auto mode fallback: prefer split family by file naming if possible
    if split_type == "auto":
        rgb_train = next((p for p in candidates if p.name.lower() == "rgb_train_ids.txt"), None)
        rgb_test = next((p for p in candidates if p.name.lower() == "rgb_test_ids.txt"), None)
        if rgb_train is not None and rgb_test is not None:
            return rgb_train, rgb_test

    train_file = None
    test_file = None
    for path in candidates:
        name = path.name.lower()
        if "train" in name and train_file is None:
            train_file = path
        if "test" in name and test_file is None:
            test_file = path
    if train_file is None or test_file is None:
        raise FileNotFoundError(
            f"Could not find split files with 'train' and 'test' in names under {split_dir}"
        )
    return train_file, test_file


def _read_calories_map(dataset_root: Path) -> Dict[str, float]:
    metadata_dir = dataset_root / "metadata"
    calorie_map: Dict[str, float] = {}
    for filename in METADATA_FILENAMES:
        path = metadata_dir / filename
        if not path.exists():
            continue
        with path.open("r", newline="") as f:
            # Official Nutrition5k dish metadata CSVs are headerless and start with:
            # dish_id,total_calories,total_mass,total_fat,total_carb,total_protein,...
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                dish_id = row[0].strip()
                cals = row[1].strip()
                if not dish_id or dish_id.lower() == "dish_id":
                    continue
                try:
                    calorie_map[dish_id] = float(cals)
                except ValueError:
                    continue
    if not calorie_map:
        raise FileNotFoundError(
            f"No dish metadata found in {metadata_dir}. Expected {METADATA_FILENAMES}"
        )
    return calorie_map


def _find_file_with_keywords(directory: Path, keywords: Sequence[str]) -> Optional[Path]:
    files = [p for p in directory.iterdir() if p.is_file()]
    for keyword in keywords:
        for p in files:
            if keyword in p.name.lower():
                return p
    return None


def _find_rgb_depth_paths(overhead_dish_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    if not overhead_dish_dir.exists():
        return None, None
    rgb_path = _find_file_with_keywords(
        overhead_dish_dir,
        keywords=["rgb.png", "rgb.jpg", "rgb", "color"],
    )
    depth_path = _find_file_with_keywords(
        overhead_dish_dir,
        keywords=["depth_raw", "raw_depth", "depth", "depth16"],
    )
    return rgb_path, depth_path


def depth_image_to_tensor(
    depth_img: Image.Image,
    resize_transform: transforms.Resize,
    max_depth_units: float,
) -> torch.Tensor:
    depth_img = resize_transform(depth_img)
    depth_np = np.array(depth_img)
    if depth_np.ndim == 3:
        depth_np = depth_np[..., 0]
    depth_np = depth_np.astype(np.float32)

    if depth_np.max() > 255.0:
        depth_np = depth_np / max_depth_units
    else:
        depth_np = depth_np / 255.0

    low = float(np.percentile(depth_np, 1.0))
    high = float(np.percentile(depth_np, 99.0))
    if high > low:
        depth_np = (depth_np - low) / (high - low)
    depth_np = np.clip(depth_np, 0.0, 1.0)
    return torch.from_numpy(depth_np).unsqueeze(0)


def build_split_samples(
    dataset_root: str,
    val_ratio: float,
    seed: int,
    split_type: str = "auto",
) -> Dict[str, List[DishSample]]:
    root = Path(dataset_root)
    split_dir = root / "dish_ids" / "splits"
    overhead_root = root / "imagery" / "realsense_overhead"
    calorie_map = _read_calories_map(root)

    train_ids: List[str]
    test_ids: List[str]
    if split_dir.exists():
        train_file, test_file = _locate_split_files(split_dir, split_type=split_type)
        train_ids = _read_dish_ids_from_split_file(train_file)
        test_ids = _read_dish_ids_from_split_file(test_file)
    else:
        all_ids = sorted(calorie_map.keys())
        rng = random.Random(seed)
        rng.shuffle(all_ids)
        split = int(0.8 * len(all_ids))
        train_ids = all_ids[:split]
        test_ids = all_ids[split:]

    def has_local_rgb(dish_id: str) -> bool:
        rgb_path, _ = _find_rgb_depth_paths(overhead_root / dish_id)
        return rgb_path is not None

    # Build train/val split only from dishes that actually exist locally.
    # This avoids empty-train failures on mini/subsampled downloads.
    rng = random.Random(seed)
    available_train_ids = [x for x in train_ids if x in calorie_map and has_local_rgb(x)]
    rng.shuffle(available_train_ids)
    if len(available_train_ids) <= 1:
        val_ids = available_train_ids
        train_ids = available_train_ids
    else:
        val_count = max(1, int(len(available_train_ids) * val_ratio))
        val_ids = available_train_ids[:val_count]
        train_ids = available_train_ids[val_count:]

    def make_samples(dish_ids: Sequence[str]) -> List[DishSample]:
        samples: List[DishSample] = []
        skipped = 0
        for dish_id in dish_ids:
            dish_dir = overhead_root / dish_id
            rgb_path, depth_path = _find_rgb_depth_paths(dish_dir)
            if rgb_path is None:
                skipped += 1
                continue
            samples.append(
                DishSample(
                    dish_id=dish_id,
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                    calories=calorie_map[dish_id],
                )
            )
        if skipped > 0:
            print(f"Skipped {skipped} dishes without overhead RGB files.")
        return samples

    return {
        "train": make_samples(train_ids),
        "val": make_samples(val_ids),
        "test": make_samples(test_ids),
    }


class Nutrition5kCalorieDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[DishSample],
        image_size: int = 224,
        mode: str = "rgb",
        max_depth_units: float = 4000.0,
        is_train: bool = False,
        augment: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.mode = mode
        self.max_depth_units = max_depth_units
        self.is_train = is_train
        self._augment = bool(is_train and augment)
        self.image_size = image_size
        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.depth_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.depth_resize = transforms.Resize((image_size, image_size))
        self._color_aug = transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.12, hue=0.02
        )
        self._warned_bad_depth_paths: set[str] = set()
        if mode not in {"rgb", "rgbd"}:
            raise ValueError("mode must be one of {'rgb', 'rgbd'}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        rgb = Image.open(sample.rgb_path).convert("RGB")
        do_flip = self._augment and random.random() < 0.5
        if do_flip:
            rgb = F_v.hflip(rgb)
        if self._augment:
            rgb = self._color_aug(rgb)

        rgb_tensor = self.rgb_transform(rgb)

        if self.mode == "rgb":
            image_tensor = rgb_tensor
        else:
            if sample.depth_path is None:
                depth_tensor = torch.zeros((1, rgb_tensor.shape[1], rgb_tensor.shape[2]), dtype=torch.float32)
            else:
                try:
                    depth_img = Image.open(sample.depth_path)
                    if do_flip:
                        depth_img = F_v.hflip(depth_img)
                    depth_tensor = depth_image_to_tensor(
                        depth_img=depth_img,
                        resize_transform=self.depth_resize,
                        max_depth_units=self.max_depth_units,
                    )
                except (UnidentifiedImageError, OSError):
                    bad_path = str(sample.depth_path)
                    if bad_path not in self._warned_bad_depth_paths:
                        print(f"[warn] unreadable depth image, fallback to zeros: {bad_path}")
                        self._warned_bad_depth_paths.add(bad_path)
                    depth_tensor = torch.zeros((1, rgb_tensor.shape[1], rgb_tensor.shape[2]), dtype=torch.float32)
            image_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)

        target = torch.tensor([sample.calories], dtype=torch.float32)
        return image_tensor, target, sample.dish_id


def create_dataloaders(
    dataset_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    mode: str = "rgb",
    split_type: str = "auto",
    max_depth_units: float = 4000.0,
    augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    splits = build_split_samples(
        dataset_root=dataset_root,
        val_ratio=val_ratio,
        seed=seed,
        split_type=split_type,
    )
    train_dataset = Nutrition5kCalorieDataset(
        splits["train"],
        image_size=image_size,
        mode=mode,
        max_depth_units=max_depth_units,
        is_train=True,
        augment=augment_train,
    )
    val_dataset = Nutrition5kCalorieDataset(
        splits["val"],
        image_size=image_size,
        mode=mode,
        max_depth_units=max_depth_units,
        is_train=False,
        augment=False,
    )
    test_dataset = Nutrition5kCalorieDataset(
        splits["test"],
        image_size=image_size,
        mode=mode,
        max_depth_units=max_depth_units,
        is_train=False,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def create_food101_dataloaders(
    root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    download: bool = False,
    augment_food101: bool = True,
    food101_val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Food-101 train is split into train / val (reproducible); val uses eval-only transforms.

    Official ``test`` split is not used during training loops (avoids tuning on test).
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    if augment_food101:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                    ratio=(3.0 / 4.0, 4.0 / 3.0),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
                ),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
                normalize,
            ]
        )
    else:
        train_transform = eval_transform

    train_full_eval = Food101(root=root, split="train", transform=eval_transform, download=download)
    classes = train_full_eval.classes
    n = len(train_full_eval)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    if n < 2:
        train_indices, val_indices = [0], [0]
    else:
        n_val = max(1, min(n - 1, int(round(n * food101_val_ratio))))
        val_indices = perm[:n_val]
        train_indices = perm[n_val:]

    if augment_food101:
        train_full_aug = Food101(root=root, split="train", transform=train_transform, download=False)
        train_ds = Subset(train_full_aug, train_indices)
    else:
        train_ds = Subset(train_full_eval, train_indices)

    val_ds = Subset(train_full_eval, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, classes


def create_food101_test_dataloader(
    root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    download: bool = False,
) -> Tuple[DataLoader, List[str]]:
    """Official Food-101 **test** split (eval transform only). Class order matches ``torchvision.datasets.Food101``."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_ds = Food101(root=root, split="test", transform=eval_transform, download=download)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return test_loader, test_ds.classes
