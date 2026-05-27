#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def _read_ids(path: str, max_images: int | None = None) -> list[str]:
    ids = [line.strip() for line in open(path) if line.strip()]
    return ids[:max_images] if max_images else ids


def _build_predictor(model_id: str, device: str):
    from transformers import pipeline

    device_arg = 0 if device == 'cuda' else -1
    return pipeline('depth-estimation', model=model_id, device=device_arg)


def _collect_depth_tasks(dish_ids: list[str], overhead_root: Path, cache_root: Path):
    tasks = []
    missing = []
    for dish_id in dish_ids:
        rgb_path = overhead_root / dish_id / 'rgb.png'
        if not rgb_path.is_file():
            missing.append(dish_id)
            continue
        dst_path = cache_root / dish_id / 'depth_pred.png'
        if not dst_path.is_file():
            tasks.append((dish_id, rgb_path, dst_path))
    return tasks, missing


def _normalise_depth(result) -> Image.Image:
    if isinstance(result, dict) and 'predicted_depth' in result:
        depth = result['predicted_depth']
        if isinstance(depth, torch.Tensor):
            arr = depth.detach().float().cpu().numpy()
        else:
            arr = np.array(depth, dtype=np.float32)
    elif isinstance(result, dict) and 'depth' in result:
        arr = np.array(result['depth'], dtype=np.float32)
    else:
        arr = np.array(result, dtype=np.float32)

    arr = np.squeeze(arr)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    arr16 = np.clip(arr * 65535.0, 0, 65535).astype(np.uint16)
    return Image.fromarray(arr16, mode='I;16')


def _pack_archive(cache_parent: Path, dish_ids: list[str], output_archive: Path,
                  pack_work_dir: Path) -> None:
    t0 = time.time()
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    pack_work_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix='dpf_pack_', dir=pack_work_dir))
    tar_path = tmp_dir / f'{output_archive.name}.tar'
    list_path = tmp_dir / f'{output_archive.name}.members'
    members = [
        f'pred_depth/{dish_id}'
        for dish_id in dish_ids
        if (cache_parent / 'pred_depth' / dish_id).is_dir()
    ]
    if not members:
        raise RuntimeError('No predicted depth directories found to archive.')
    list_path.write_text('\n'.join(members) + '\n')
    try:
        print(f'Packing {len(members)} predicted-depth directories to temporary tar...', flush=True)
        subprocess.run(['tar', '-cf', str(tar_path), '-C', str(cache_parent), '-T', str(list_path)], check=True)
        print(f'Compressing archive to {output_archive}...', flush=True)
        subprocess.run(['zstd', '-f', '-1', '-T0', '--rm', str(tar_path), '-o', str(output_archive)], check=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f'Archive complete in {time.time() - t0:.1f}s', flush=True)


def _iter_batches(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overhead-root', required=True)
    parser.add_argument('--split-ids', required=True)
    parser.add_argument('--split', required=True, choices=['train', 'test'])
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--archive-dir')
    parser.add_argument('--pack-work-dir', default='/dev/shm' if os.path.isdir('/dev/shm') else None)
    parser.add_argument('--model-id', default='depth-anything/Depth-Anything-V2-Small-hf')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--batch-size', type=int, default=16 if torch.cuda.is_available() else 1)
    parser.add_argument('--max-images', type=int)
    parser.add_argument('--allow-missing-rgb', action='store_true')
    parser.add_argument('--skip-archive', action='store_true')
    args = parser.parse_args()

    t_start = time.time()
    dish_ids = _read_ids(args.split_ids, args.max_images)
    cache_parent = Path(args.output_dir)
    archive_dir = Path(args.archive_dir) if args.archive_dir else cache_parent
    pack_work_dir = Path(args.pack_work_dir) if args.pack_work_dir else Path(tempfile.gettempdir())
    cache_root = cache_parent / 'pred_depth'
    cache_root.mkdir(parents=True, exist_ok=True)

    print(
        f'DPF depth cache start: split={args.split} requested={len(dish_ids)} '
        f'device={args.device} batch_size={max(1, args.batch_size)} '
        f'cache_root={cache_root}',
        flush=True,
    )
    tasks, missing = _collect_depth_tasks(dish_ids, Path(args.overhead_root), cache_root)
    existing = len(dish_ids) - len(missing) - len(tasks)
    print(
        f'Task scan: existing={existing} new={len(tasks)} missing_rgb={len(missing)}',
        flush=True,
    )
    if missing and not args.allow_missing_rgb:
        preview = ', '.join(missing[:10])
        extra = '' if len(missing) <= 10 else f' ... (+{len(missing) - 10} more)'
        sys.exit(
            f'Missing overhead RGB for {len(missing)} / {len(dish_ids)} split IDs: '
            f'{preview}{extra}. Re-extract the official depth archive before generating DPF depth.'
        )

    if tasks:
        print(f'Loading depth model: {args.model_id}', flush=True)
    predictor = _build_predictor(args.model_id, args.device) if tasks else None
    if tasks:
        print(f'Model ready; starting inference for {len(tasks)} images.', flush=True)
    generated = 0

    t_infer = time.time()
    progress = tqdm(total=len(tasks), desc=f'dpf depth {args.split}', unit='dish')
    for batch in _iter_batches(tasks, max(1, args.batch_size)):
        dish_batch, path_batch, dst_batch = zip(*batch)
        images = [Image.open(path).convert('RGB') for path in path_batch]
        with torch.inference_mode():
            preds = predictor(list(images), batch_size=max(1, args.batch_size))
        if isinstance(preds, dict):
            preds = [preds]
        for pred, dst_path in zip(preds, dst_batch):
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            _normalise_depth(pred).save(dst_path)
            generated += 1
        progress.update(len(batch))
    progress.close()
    if tasks:
        elapsed = time.time() - t_infer
        rate = generated / elapsed if elapsed > 0 else 0.0
        print(f'Inference complete: generated={generated} elapsed={elapsed:.1f}s rate={rate:.2f} images/s', flush=True)

    archive = archive_dir / f'dpf_pred_depth_{args.split}.tar.zst'
    if not args.skip_archive:
        _pack_archive(cache_parent, dish_ids, archive, pack_work_dir)

    manifest = {
        'model_id': args.model_id,
        'split': args.split,
        'image_count': existing + generated,
        'new_image_count': generated,
        'existing_image_count': existing,
        'requested_count': len(dish_ids),
        'missing_rgb_count': len(missing),
        'missing_rgb_ids': missing,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'overhead_root': args.overhead_root,
        'cache_root': str(cache_root),
        'batch_size': max(1, args.batch_size),
        'output_archive': str(archive) if not args.skip_archive else None,
    }
    manifest_path = archive_dir / f'dpf_pred_depth_{args.split}_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f'DPF depth cache done in {time.time() - t_start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
