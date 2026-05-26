import numpy as np

# Camera calibration constants from the paper
_CAMERA_HEIGHT_CM = 35.9
_DEPTH_UNITS_PER_CM = 100.0      # 1 m = 10000 units → 1 unit = 1e-4 m = 0.01 cm
_PIXEL_AREA_CM2 = 5.957e-3       # per-pixel surface area at 35.9 cm distance


def compute_volume_from_depth(
    depth_arr: np.ndarray,          # (H, W) uint16, units: 1e-4 m
    food_mask: np.ndarray,          # (H, W) bool, True = food pixel
) -> float:
    """Compute food volume in cm³ from a depth map and binary food mask.

    Follows the paper's method: volume = sum over food pixels of
    (camera_height_cm - pixel_depth_cm) * pixel_area_cm2.
    """
    depth_cm = depth_arr.astype(np.float32) / _DEPTH_UNITS_PER_CM
    food_height_cm = np.clip(_CAMERA_HEIGHT_CM - depth_cm, 0, None)
    return float(np.sum(food_height_cm[food_mask]) * _PIXEL_AREA_CM2)


def compute_volume_batch(
    depth_dir: str,
    rgb_dir: str,
    dish_ids: list,
    segmenter,          # TF Hub mobile food segmenter (loaded by caller)
    target_size: int = 256,
) -> dict:
    """Compute volume scalar for each dish_id. Returns dict: dish_id -> volume_cm3.

    Args:
        segmenter: Loaded TF Hub model — call via segmenter(tf.expand_dims(rgb_tensor, 0)).
        rgb_dir: Path to imagery/realsense_overhead/ (contains {dish_id}/rgb.png).
        depth_dir: Path to imagery/realsense_overhead/ (same; {dish_id}/depth_raw.png).
    """
    import os
    import tensorflow as tf
    from PIL import Image

    results = {}
    for dish_id in dish_ids:
        rgb_path   = os.path.join(rgb_dir,   dish_id, 'rgb.png')
        depth_path = os.path.join(depth_dir, dish_id, 'depth_raw.png')
        if not (os.path.isfile(rgb_path) and os.path.isfile(depth_path)):
            continue

        # Load RGB and run segmenter
        rgb_pil = Image.open(rgb_path).convert('RGB').resize((target_size, target_size))
        rgb_arr = np.array(rgb_pil, dtype=np.uint8)
        rgb_tf  = tf.expand_dims(tf.constant(rgb_arr, dtype=tf.uint8), 0)
        seg_out = segmenter(rgb_tf)
        # The segmenter returns {'SemanticPredictions': (1, H, W)} — food class = 1
        preds = seg_out['SemanticPredictions'].numpy()[0]  # (H, W) int32
        food_mask = (preds == 1)

        # Load depth
        depth_pil = Image.open(depth_path).resize((target_size, target_size), Image.NEAREST)
        depth_arr = np.array(depth_pil, dtype=np.uint16)

        results[dish_id] = compute_volume_from_depth(depth_arr, food_mask)

    return results


def compute_volume_batch_depth_only(
    depth_dir: str,
    dish_ids: list,
    min_food_height_cm: float = 0.3,
    target_size: int = 256,
    num_workers: int = 16,
) -> dict:
    """Compute volume without a segmentation model, using depth thresholding.

    Food pixels = those with height above table > min_food_height_cm.
    Uses ThreadPoolExecutor to parallelize Drive FUSE reads.
    """
    import os
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor, as_completed

    resample = getattr(Image, 'Resampling', Image).NEAREST

    def _process(dish_id):
        depth_path = os.path.join(depth_dir, dish_id, 'depth_raw.png')
        if not os.path.isfile(depth_path) or os.path.getsize(depth_path) == 0:
            return dish_id, None
        depth_pil = Image.open(depth_path).resize((target_size, target_size), resample)
        depth_arr = np.array(depth_pil, dtype=np.uint16)
        depth_cm = depth_arr.astype(np.float32) / _DEPTH_UNITS_PER_CM
        food_height_cm = np.clip(_CAMERA_HEIGHT_CM - depth_cm, 0, None)
        food_mask = food_height_cm > min_food_height_cm
        return dish_id, compute_volume_from_depth(depth_arr, food_mask)

    results = {}
    done = 0
    total = len(dish_ids)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_process, d): d for d in dish_ids}
        for fut in as_completed(futures):
            dish_id, vol = fut.result()
            if vol is not None:
                results[dish_id] = vol
            done += 1
            if done % 500 == 0 or done == total:
                print(f'  {done}/{total} dishes processed', flush=True)
    return results
