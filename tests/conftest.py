import pytest
import numpy as np
from PIL import Image


@pytest.fixture
def rgb_image():
    """256×256 RGB PIL Image."""
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def depth_image():
    """256×256 16-bit depth PIL Image (units: 1e-4 m)."""
    arr = np.random.randint(2000, 4000, (256, 256), dtype=np.uint16)
    return Image.fromarray(arr, mode='I;16')


@pytest.fixture
def sample_labels():
    return {
        'calories': 300.0,
        'mass': 250.0,
        'fat': 15.0,
        'carb': 30.0,
        'protein': 20.0,
    }


@pytest.fixture
def batch_size():
    return 4
