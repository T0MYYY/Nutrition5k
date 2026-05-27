import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.io as _tv_io
import torchvision.transforms.functional as _TF

# ImageNet stats used by pretrained InceptionV3
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# Max depth in dataset (40 cm = 4000 units); camera height = 35.9 cm = 3590 units
_DEPTH_MAX_UNITS = 4000.0
_CAMERA_HEIGHT_UNITS = 3590.0  # 35.9 cm at 1e-4 m/unit


def fast_open_rgb(path: str) -> Image.Image:
    """Decode JPEG/PNG as PIL RGB using libjpeg-turbo (2–3× faster than PIL alone)."""
    return _TF.to_pil_image(_tv_io.read_image(path, mode=_tv_io.ImageReadMode.RGB))


def get_train_transform(pre_resized: bool = False):
    """
    pre_resized=True: frames were already resized to 292px shortest edge at
    archive-creation time, so skip the Resize step here.
    """
    steps = [] if pre_resized else [T.Resize(292)]
    steps += [T.RandomCrop(256), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(_MEAN, _STD)]
    return T.Compose(steps)


def get_val_transform(pre_resized: bool = False):
    steps = [] if pre_resized else [T.Resize(292)]
    steps += [T.CenterCrop(256), T.ToTensor(), T.Normalize(_MEAN, _STD)]
    return T.Compose(steps)


def get_dpf_rgb_transform(train: bool = False):
    steps = [T.Resize((336, 448))]
    if train:
        steps.append(T.RandomHorizontalFlip())
    steps += [T.ToTensor(), T.Normalize(_MEAN, _STD)]
    return T.Compose(steps)


class DepthToTensor:
    """Picklable transform: 16-bit depth PIL image → (1, H, W) float tensor.

    Normalizes to [0, 1] where 1 = at camera (food surface), 0 = table level.
    """

    def __init__(self, target_size: int = 256):
        self.target_size = target_size
        self._resample = getattr(Image, 'Resampling', Image).BILINEAR

    def __call__(self, depth_pil):
        arr = np.array(depth_pil, dtype=np.float32)
        arr = np.clip(arr, 0, _DEPTH_MAX_UNITS)
        arr = (_CAMERA_HEIGHT_UNITS - arr) / _CAMERA_HEIGHT_UNITS
        arr = np.clip(arr, 0, 1)
        pil = Image.fromarray(arr.astype(np.float32), mode='F')
        pil = pil.resize((self.target_size, self.target_size), self._resample)
        return torch.from_numpy(np.array(pil, dtype=np.float32)).unsqueeze(0)


def depth_to_tensor(target_size: int = 256) -> DepthToTensor:
    """Return a picklable depth transform (DepthToTensor instance)."""
    return DepthToTensor(target_size)


class PredDepthToTensor:
    def __init__(self, size=(336, 448)):
        self.size = size
        self._resample = getattr(Image, 'Resampling', Image).BILINEAR

    def __call__(self, depth_pil):
        arr = np.array(depth_pil, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        max_value = float(arr.max()) if arr.size else 0.0
        if max_value > 1.0:
            arr = arr / 65535.0 if max_value > 255.0 else arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
        pil = Image.fromarray(arr.astype(np.float32), mode='F')
        pil = pil.resize((self.size[1], self.size[0]), self._resample)
        return torch.from_numpy(np.array(pil, dtype=np.float32)).unsqueeze(0)


def get_dpf_depth_transform():
    return PredDepthToTensor((336, 448))
