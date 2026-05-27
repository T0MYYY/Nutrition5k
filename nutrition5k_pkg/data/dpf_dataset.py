import os
from typing import Dict, List

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .transforms import fast_open_rgb

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DPFNutritionDataset(Dataset):
    def __init__(
        self,
        overhead_root: str,
        pred_depth_root: str,
        metadata: Dict,
        dish_ids: List[str],
        rgb_transform=None,
        depth_transform=None,
    ):
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.samples = []

        for dish_id in dish_ids:
            if dish_id not in metadata:
                continue
            rgb_path = os.path.join(overhead_root, dish_id, 'rgb.png')
            depth_path = self._depth_path(pred_depth_root, dish_id)
            if not os.path.isfile(rgb_path) or not os.path.isfile(depth_path):
                continue
            labels = {k: float(metadata[dish_id][k]) for k in ('calories', 'mass', 'fat', 'carb', 'protein')}
            self.samples.append((rgb_path, depth_path, labels))

    @staticmethod
    def _depth_path(root: str, dish_id: str) -> str:
        candidates = (
            os.path.join(root, dish_id, 'depth_pred.png'),
            os.path.join(root, dish_id, 'pred_depth.png'),
            os.path.join(root, dish_id, 'depth.png'),
            os.path.join(root, f'{dish_id}.png'),
        )
        for path in candidates:
            if os.path.isfile(path):
                return path
        return candidates[0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path, labels = self.samples[idx]
        rgb = fast_open_rgb(rgb_path)
        depth = Image.open(depth_path)

        if self.rgb_transform:
            rgb_t = self.rgb_transform(rgb)
        else:
            import torchvision.transforms.functional as TF
            rgb_t = TF.to_tensor(rgb)

        if self.depth_transform:
            depth_t = self.depth_transform(depth)
        else:
            import torchvision.transforms.functional as TF
            depth_t = TF.to_tensor(depth)
            if depth_t.shape[0] != 1:
                depth_t = depth_t[:1]

        label_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in labels.items()}
        return rgb_t, depth_t, label_tensor
