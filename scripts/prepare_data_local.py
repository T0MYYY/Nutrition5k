#!/usr/bin/env python3
"""
Prepare RGB training/test archives locally (faster than Colab CPU).

Reads nutrition5k_dataset.tar, extracts N_FRAMES evenly-spaced frames per
camera per dish, pre-resizes to 292px shortest edge (matching training
transforms), and packs into rgb_train.tar.zst / rgb_test.tar.zst.

Prerequisites:
    brew install ffmpeg zstd

Output (upload both files to Drive: MyDrive/nutrition5k/data/):
    data/rgb_train.tar.zst
    data/rgb_test.tar.zst

Usage:
    python scripts/prepare_data_local.py
    python scripts/prepare_data_local.py --n-frames 6 --workers 10
"""

import argparse, os, shutil, subprocess, sys, tarfile, tempfile, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
INPUT_TAR  = ROOT / 'data' / 'nutrition5k_dataset.tar'
SPLITS_DIR = ROOT / 'data' / 'splits'
OUT_DIR    = ROOT / 'data'


def even_indices(n_total: int, n_select: int) -> list:
    """Return n_select evenly-spaced indices covering [0, n_total-1]."""
    if n_total <= n_select:
        return list(range(n_total))
    return [round(i * (n_total - 1) / (n_select - 1)) for i in range(n_select)]


def _process_dish(args: tuple) -> tuple:
    """Worker: extract N frames from each camera, write to frames_out_dir."""
    dish_id, h264_dir, frames_out_dir, n_frames = args
    frames_dir = Path(frames_out_dir) / 'side_angles' / dish_id / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    for h264 in sorted(Path(h264_dir).glob('camera_*.h264')):
        cam = h264.stem  # e.g. 'camera_A'
        with tempfile.TemporaryDirectory() as tmpd:
            # Extract all frames with resize (small JPEGs → fast I/O)
            subprocess.run([
                'ffmpeg', '-i', str(h264),
                '-vf', 'scale=-2:292',  # shortest edge = 292px
                '-q:v', '3',
                os.path.join(tmpd, 'frame_%06d.jpeg')
            ], capture_output=True)

            all_frames = sorted(Path(tmpd).glob('frame_*.jpeg'))
            if not all_frames:
                continue

            selected = [all_frames[i] for i in even_indices(len(all_frames), n_frames)]
            for j, src in enumerate(selected, 1):
                shutil.copy2(src, frames_dir / f'{cam}_frame_{j:04d}.jpeg')
            total += len(selected)

    return dish_id, total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-frames', type=int, default=6,
                    help='Frames to keep per camera per dish (default: 6)')
    ap.add_argument('--workers', type=int,
                    default=max(1, (os.cpu_count() or 4) - 2),
                    help='Parallel workers (default: cpu_count - 2)')
    ap.add_argument('--tar', type=Path, default=INPUT_TAR)
    args = ap.parse_args()

    if not args.tar.exists():
        sys.exit(f"ERROR: tar not found at {args.tar}")

    train_ids = set((SPLITS_DIR / 'rgb_train_ids.txt').read_text().split())
    test_ids  = set((SPLITS_DIR / 'rgb_test_ids.txt').read_text().split())
    all_ids   = train_ids | test_ids
    print(f"Train: {len(train_ids)} dishes  |  Test: {len(test_ids)} dishes")
    print(f"Workers: {args.workers}  |  Frames/camera: {args.n_frames}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='n5k_prep_') as tmpdir:
        h264_root   = Path(tmpdir) / 'h264'
        frames_root = Path(tmpdir) / 'frames'
        h264_root.mkdir()
        frames_root.mkdir()

        # ── Phase 1: stream h264 files out of the input tar ──────────────────
        print('\n[1/3] Streaming h264 files from tar...')
        t0 = time.time()
        n_extracted = 0
        with tarfile.open(args.tar) as tf:
            for member in tf:
                if not member.name.endswith('.h264'):
                    continue
                parts = member.name.split('/')
                try:
                    sa_idx = parts.index('side_angles')
                    dish_id = parts[sa_idx + 1]
                except (ValueError, IndexError):
                    continue
                if dish_id not in all_ids:
                    continue

                dst = h264_root / dish_id / parts[-1]
                dst.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(member) as src:
                    with open(dst, 'wb') as f:
                        shutil.copyfileobj(src, f)
                n_extracted += 1
                if n_extracted % 2000 == 0:
                    print(f'  {n_extracted} h264 files... ({time.time()-t0:.0f}s)')

        print(f'  -> {n_extracted} h264 files extracted in {time.time()-t0:.0f}s')

        # ── Phase 2: parallel frame extraction ───────────────────────────────
        print(f'\n[2/3] Extracting {args.n_frames} frames/camera (parallel)...')
        t0 = time.time()
        tasks = [
            (d.name, str(d), str(frames_root), args.n_frames)
            for d in sorted(h264_root.iterdir()) if d.is_dir()
        ]
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_process_dish, t): t[0] for t in tasks}
            for fut in as_completed(futs):
                dish_id, n = fut.result()
                done += 1
                if done % 500 == 0 or done == len(tasks):
                    elapsed = time.time() - t0
                    eta = elapsed / done * (len(tasks) - done)
                    print(f'  {done}/{len(tasks)} dishes  |  {elapsed:.0f}s  |  ETA {eta:.0f}s')

        print(f'  -> {done} dishes processed in {time.time()-t0:.0f}s')

        # ── Phase 3: pack archives ────────────────────────────────────────────
        print('\n[3/3] Packing archives...')
        for split, ids in [('train', train_ids), ('test', test_ids)]:
            out_path = OUT_DIR / f'rgb_{split}.tar.zst'
            tmp_tar  = Path(tmpdir) / f'rgb_{split}.tar'
            present  = sorted(d for d in ids if (frames_root / 'side_angles' / d).exists())
            print(f'  {out_path.name}: {len(present)} dishes...', end=' ', flush=True)
            t0 = time.time()

            with tarfile.open(tmp_tar, 'w') as tf:
                for dish_id in present:
                    dish_path = frames_root / 'side_angles' / dish_id
                    tf.add(dish_path, arcname=f'side_angles/{dish_id}')

            subprocess.run(
                ['zstd', '-T0', '-1', str(tmp_tar), '-o', str(out_path)],
                check=True, capture_output=True
            )
            mb = out_path.stat().st_size // 1024 // 1024
            print(f'{mb} MB in {time.time()-t0:.0f}s')

    print(f'\nDone. Upload to Drive > MyDrive/nutrition5k/data/:')
    print(f'  data/rgb_train.tar.zst')
    print(f'  data/rgb_test.tar.zst')


if __name__ == '__main__':
    main()
