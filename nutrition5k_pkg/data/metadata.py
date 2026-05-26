import csv
import hashlib
import random
from typing import Dict, Optional, Tuple, List


def load_official_split(
    train_txt: str,
    test_txt: str,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Load official Nutrition5k split files and carve val from train.

    Args:
        train_txt: Path to e.g. rgb_train_ids.txt (one dish_id per line).
        test_txt:  Path to e.g. rgb_test_ids.txt.
        val_ratio: Fraction of train_ids to hold out as validation.
        seed:      RNG seed for deterministic val split.

    Returns:
        (train_ids, val_ids, test_ids)
    """
    train_ids = [l.strip() for l in open(train_txt) if l.strip()]
    test_ids  = [l.strip() for l in open(test_txt)  if l.strip()]

    overlap = set(train_ids) & set(test_ids)
    assert not overlap, f"Official split has {len(overlap)} overlapping dishes — check files."

    rng = random.Random(seed)
    shuffled = list(train_ids)
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_ratio)
    return shuffled[n_val:], shuffled[:n_val], test_ids


def load_dish_metadata(cafe1_path: str, cafe2_path: Optional[str] = None) -> Dict[str, Dict]:
    """Parse one or two dish_metadata CSV files. Returns dict: dish_id -> label dict."""
    records = {}
    paths = [p for p in [cafe1_path, cafe2_path] if p is not None]
    for path in paths:
        with open(path, newline='') as f:
            for row in csv.reader(f):
                if not row:
                    continue
                if len(row) < 6:
                    continue
                dish_id = row[0].strip()
                records[dish_id] = {
                    'calories': float(row[1]),
                    'mass':     float(row[2]),
                    'fat':      float(row[3]),
                    'carb':     float(row[4]),
                    'protein':  float(row[5]),
                }
    return records


def get_train_val_test_split(
    dish_ids: List[str],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Deterministic hash-based split. Each dish_id is hashed to determine split membership."""
    # Create a list of (hash_value, dish_id) tuples
    hashed_dishes = []
    for dish_id in dish_ids:
        h = int(hashlib.md5(f"{seed}{dish_id}".encode()).hexdigest(), 16)
        hashed_dishes.append((h, dish_id))

    # Sort by hash value for deterministic ordering
    hashed_dishes.sort(key=lambda x: x[0])

    # Calculate split indices
    n = len(dish_ids)
    val_size = int(n * val_ratio)
    test_size = int(n * test_ratio)

    val = [dish_id for _, dish_id in hashed_dishes[:val_size]]
    test = [dish_id for _, dish_id in hashed_dishes[val_size:val_size + test_size]]
    train = [dish_id for _, dish_id in hashed_dishes[val_size + test_size:]]

    return train, val, test
