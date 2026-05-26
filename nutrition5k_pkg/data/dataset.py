import os
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
from .transforms import fast_open_rgb


class Nutrition5kDataset(Dataset):
    """PyTorch dataset for Nutrition5k side-angle frames.

    Args:
        side_angle_root: Path to imagery/side_angles/ directory.
        metadata: Dict[dish_id, label_dict] from load_dish_metadata().
        dish_ids: List of dish_ids for this split.
        mode: 'direct' (absolute values) or 'per_gram' (normalised by mass).
        transform: torchvision transform applied to each PIL image.
    """

    def __init__(self, side_angle_root, metadata, dish_ids, mode='direct', transform=None,
                 preload_ram: bool = False):
        assert mode in ('direct', 'per_gram'), f"Unknown mode: {mode}"
        self.mode = mode
        self.transform = transform
        self.samples: List[Tuple[str, Dict]] = []

        for dish_id in dish_ids:
            frames_dir = os.path.join(side_angle_root, dish_id, 'frames')
            if not os.path.isdir(frames_dir):
                continue
            labels = self._build_labels(metadata[dish_id])
            for fname in sorted(os.listdir(frames_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(frames_dir, fname), labels))

        self._cache: dict = {}
        if preload_ram:
            print(f'Preloading {len(self.samples)} images into RAM...')
            for path, _ in self.samples:
                if path not in self._cache:
                    self._cache[path] = fast_open_rgb(path)
            print('Preload complete.')

    def _build_labels(self, raw: Dict) -> Dict:
        if self.mode == 'direct':
            return {k: float(raw[k]) for k in ('calories', 'mass', 'fat', 'carb', 'protein')}
        mass = raw['mass']
        return {
            'cal_per_g':     float(raw['calories']) / mass,
            'fat_per_g':     float(raw['fat'])      / mass,
            'carb_per_g':    float(raw['carb'])     / mass,
            'protein_per_g': float(raw['protein'])  / mass,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, labels = self.samples[idx]
        img = self._cache[path] if path in self._cache else fast_open_rgb(path)
        if self.transform:
            img = self.transform(img)
        label_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in labels.items()}
        return img, label_tensor
