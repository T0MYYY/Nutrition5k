import os
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from .transforms import fast_open_rgb

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DepthDataset(Dataset):
    """Dataset for overhead imagery (realsense_overhead/).

    When depth_transform is None: returns 3-channel RGB tensor (rgb.png only).
    When depth_transform is set:  returns 4-channel RGB-D tensor (requires depth_raw.png).
    volume_map: optional dict dish_id -> float; if provided, injects 'volume' into labels.
    """

    def __init__(self, overhead_root: str, metadata: Dict, dish_ids: List[str],
                 mode: str = 'direct', rgb_transform=None, depth_transform=None,
                 volume_map: Optional[Dict[str, float]] = None,
                 preload_ram: bool = False):
        assert mode in ('direct', 'per_gram')
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.volume_map = volume_map or {}
        self.samples: List = []

        for dish_id in dish_ids:
            dish_dir = os.path.join(overhead_root, dish_id)
            rgb_path = os.path.join(dish_dir, 'rgb.png')
            if not os.path.isfile(rgb_path):
                continue
            depth_path = os.path.join(dish_dir, 'depth_raw.png')
            depth_ok = os.path.isfile(depth_path) and os.path.getsize(depth_path) > 0
            if depth_transform is not None and not depth_ok:
                continue  # depth required but missing or empty
            volume = self.volume_map.get(dish_id)
            if self.volume_map and volume is None:
                continue  # volume_map provided but dish missing — skip for consistency
            labels = self._build_labels(metadata[dish_id], mode)
            self.samples.append((
                rgb_path,
                depth_path if depth_ok else None,
                labels,
                volume,
            ))

        self._rgb_cache: dict = {}
        self._depth_cache: dict = {}
        if preload_ram:
            paths = set()
            for rgb_path, depth_path, _, _ in self.samples:
                paths.add((rgb_path, depth_path))
            print(f'Preloading {len(paths)} overhead images into RAM...')
            for rgb_path, depth_path in paths:
                self._rgb_cache[rgb_path] = fast_open_rgb(rgb_path)
                if depth_path:
                    self._depth_cache[depth_path] = Image.open(depth_path)
            print('Preload complete.')

    @staticmethod
    def _build_labels(raw: Dict, mode: str) -> Dict:
        if mode == 'direct':
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
        rgb_path, depth_path, labels, volume = self.samples[idx]
        rgb = self._rgb_cache.get(rgb_path) or fast_open_rgb(rgb_path)

        if self.rgb_transform:
            rgb_t = self.rgb_transform(rgb)
        else:
            import torchvision.transforms.functional as TF
            rgb_t = TF.to_tensor(rgb)

        if self.depth_transform and depth_path:
            try:
                depth_pil = self._depth_cache.get(depth_path) or Image.open(depth_path)
                depth_t = self.depth_transform(depth_pil)
            except Exception:
                depth_t = torch.zeros(1, rgb_t.shape[1], rgb_t.shape[2])
            img = torch.cat([rgb_t, depth_t], dim=0)  # (4, H, W)
        else:
            img = rgb_t  # (3, H, W)

        label_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in labels.items()}
        label_tensor['volume'] = torch.tensor(volume if volume is not None else float('nan'), dtype=torch.float32)
        return img, label_tensor
