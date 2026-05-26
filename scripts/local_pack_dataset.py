#!/usr/bin/env python3
"""
local_pack_dataset.py — Build split-aware tar.zst archives on Apple Silicon (M5).

Speed-optimised for M5 10-core:
  • tar CLI  (not Python tarfile) for header scan and bulk extraction
  • 10 parallel ffmpeg workers  (VideoToolbox H.264 decode + CPU scale/encode)
  • Per-camera task granularity → 10 cores fully loaded at all times

Usage (recommended):
    python scripts/local_pack_dataset.py \\
        --tar        /path/to/nutrition5k_dataset.tar \\
        --output-dir ~/n5k_archives \\
        --mode       all   # depth_train | depth_test | rgb_test | rgb_train | all

Usage (legacy — pre-extracted directory):
    python scripts/local_pack_dataset.py \\
        --raw-dir    /path/to/nutrition5k_dataset \\
        --splits-dir /path/to/split_txts \\
        --output-dir ~/n5k_archives

Output archives (extract to /dev/shm on Colab):
    rgb_train.tar.zst   → side_angles/{dish_id}/frames/camera_X_frame_NNNN.jpeg
    rgb_test.tar.zst    → same structure
    depth_train.tar.zst → realsense_overhead/{dish_id}/rgb.png + depth_raw.png
    depth_test.tar.zst  → same structure
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("tqdm not installed. Run: conda install tqdm")

# ── M5 tuning ─────────────────────────────────────────────────────────────────
FFMPEG_WORKERS = 10          # all 10 cores; media engine handles decode separately
CAMERAS        = ("A", "B", "C", "D")

# VideoToolbox: hardware H.264 decode via dedicated media engine.
# scale=-2:292: shortest edge → 292 px (matches T.Resize(292) in transforms.py).
# -q:v 2: near-lossless JPEG; -f h264: raw bitstream, no container.
FFMPEG_CMD = [
    "ffmpeg",
    "-hwaccel", "videotoolbox",
    "-f", "h264",
    "-i", "{input}",
    "-vf", "scale=-2:292",
    "-q:v", "2",
    "{output_pattern}",
    "-hide_banner", "-loglevel", "error",
    "-y",
]

ZSTD_LEVEL = 1   # fastest; JPEG/PNG gain little from higher levels
# ─────────────────────────────────────────────────────────────────────────────

_PFX          = "nutrition5k_dataset"
_SIDE_ROOT    = f"{_PFX}/imagery/side_angles"
_OVER_ROOT    = f"{_PFX}/imagery/realsense_overhead"
_SPLIT_NAMES  = ("rgb_train", "rgb_test", "depth_train", "depth_test")


# ── tar CLI helpers ───────────────────────────────────────────────────────────

def _tar_list(tar_path: str) -> list[str]:
    """
    List all tar members using the OS tar binary.
    Reads only 512-byte header blocks (lseeks past data) → fast even on 181 GB.
    """
    print(f"Listing tar members ({os.path.getsize(tar_path)/1e9:.1f} GB)…")
    t0  = time.time()
    res = subprocess.run(["tar", "-tf", tar_path], capture_output=True, text=True, check=True)
    members = [l for l in res.stdout.splitlines() if l]
    print(f"  {len(members):,} members listed in {time.time()-t0:.1f}s")
    return members


def _tar_extract(tar_path: str, dst_dir: str, member_list: list[str], strip: int) -> None:
    """
    Extract a specific list of members in ONE pass through the archive.
    Uses a temp file to avoid shell ARG_MAX limits.
    strip: number of leading path components to remove.
    """
    os.makedirs(dst_dir, exist_ok=True)
    list_file = os.path.join(dst_dir, ".extract_list")
    with open(list_file, "w") as f:
        f.write("\n".join(member_list) + "\n")
    # ignore exit code 1 (bsdtar warns on missing members but still extracts the rest)
    subprocess.run(
        ["tar", "-xf", tar_path, "-C", dst_dir,
         "--strip-components", str(strip), "-T", list_file],
    )
    os.remove(list_file)


def _read_splits_from_tar(tar_path: str) -> dict[str, list[str]]:
    """Extract the 4 official split ID lists from inside the tar."""
    print("Reading official splits from tar…")
    split_paths = [f"{_PFX}/dish_ids/splits/{k}_ids.txt" for k in _SPLIT_NAMES]
    with tempfile.TemporaryDirectory(prefix="n5k_splits_") as tmp:
        subprocess.run(
            ["tar", "-xf", tar_path, "-C", tmp,
             "--strip-components", "3"] + split_paths,
        )
        splits = {}
        for k in _SPLIT_NAMES:
            found = None
            for root, _, files in os.walk(tmp):
                for fname in files:
                    if fname == f"{k}_ids.txt":
                        found = os.path.join(root, fname)
                        break
            if not found:
                sys.exit(f"ERROR: split file not found in tar for key '{k}'")
            ids = [l.strip() for l in open(found) if l.strip()]
            splits[k] = ids
            print(f"  {k}: {len(ids):,} dishes")
    return splits


# ── ffmpeg worker ─────────────────────────────────────────────────────────────

def _ffmpeg_camera(args: tuple) -> tuple[str, str, int]:
    """
    One task = one camera .h264 file → JPEG frames.
    Runs in a subprocess pool worker.
    Returns (dish_id, cam, frame_count).
    """
    dish_id, cam, h264_path, dst_dir = args
    os.makedirs(dst_dir, exist_ok=True)
    pattern = os.path.join(dst_dir, f"camera_{cam}_frame_%04d.jpeg")
    cmd = [c.replace("{input}", h264_path).replace("{output_pattern}", pattern)
           for c in FFMPEG_CMD]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        sys.stderr.write(f"  ffmpeg error {dish_id}/{cam}: {res.stderr.decode()[:200]}\n")
        return dish_id, cam, 0
    n = sum(1 for f in os.listdir(dst_dir) if f.startswith(f"camera_{cam}_"))
    return dish_id, cam, n


# ── RGB archive ───────────────────────────────────────────────────────────────

def build_rgb_archive(
    dish_ids: list[str],
    output_path: str,
    n_workers: int,
    *,
    tar_path:        str | None = None,
    all_members:     list[str] | None = None,
    side_angles_dir: str | None = None,
):
    total    = len(dish_ids)
    label    = os.path.basename(output_path)
    from_tar = tar_path is not None
    dish_set = set(dish_ids)

    print(f"\n{'─'*60}")
    print(f"Building: {label}  ({total:,} dishes, {n_workers} workers)")
    print(f"  Source: {'tar' if from_tar else side_angles_dir}")
    print(f"{'─'*60}")
    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix="n5k_rgb_") as tmp:
        h264_tmp      = os.path.join(tmp, "h264")
        frames_parent = os.path.join(tmp, "frames")
        inter_tar     = os.path.join(tmp, "intermediate.tar")
        os.makedirs(h264_tmp)
        os.makedirs(frames_parent)

        # ── ONE-PASS extraction of all h264 for this split ─────────────────
        if from_tar:
            h264_members = [
                m for m in all_members
                if (m.startswith(_SIDE_ROOT + "/") and m.endswith(".h264")
                    and m.split("/")[3] in dish_set)       # [3] = dish_id component
            ]
            print(f"  Extracting {len(h264_members):,} h264 files (single tar pass)…")
            t1 = time.time()
            _tar_extract(tar_path, h264_tmp, h264_members, strip=3)
            h264_gb = sum(
                os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(h264_tmp) for f in fs
            ) / 1e9
            print(f"  Extracted {h264_gb:.1f} GB in {time.time()-t1:.0f}s")
            src_root = h264_tmp
        else:
            src_root = side_angles_dir

        # ── Build per-camera task list (finest granularity = max CPU load) ─
        frames_sa = os.path.join(frames_parent, "side_angles")
        tasks = [
            (did, cam,
             os.path.join(src_root, did, f"camera_{cam}.h264"),
             os.path.join(frames_sa, did, "frames"))
            for did in dish_ids
            for cam in CAMERAS
            if os.path.isfile(os.path.join(src_root, did, f"camera_{cam}.h264"))
        ]
        total_cams   = len(tasks)
        total_frames = 0
        print(f"  {total_cams:,} camera tasks → {n_workers} workers")

        # ── Parallel ffmpeg ────────────────────────────────────────────────
        with tqdm(total=total_cams, desc=f"  {label}", unit="cam",
                  bar_format="{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}] {postfix}") as pbar:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futs = {pool.submit(_ffmpeg_camera, t): t for t in tasks}
                for fut in as_completed(futs):
                    _, _, n = fut.result()
                    total_frames += n
                    pbar.update(1)
                    pbar.set_postfix(frames=f"{total_frames:,}")

        t_ffmpeg = time.time() - t0
        print(f"  ffmpeg done in {t_ffmpeg/60:.1f} min  ({total_frames:,} frames)")

        # ── Pack all frames → intermediate uncompressed tar ────────────────
        print(f"  Packing frames…")
        t2 = time.time()
        subprocess.run(
            ["tar", "-cf", inter_tar, "-C", frames_parent, "side_angles"],
            check=True,
        )
        tar_gb = os.path.getsize(inter_tar) / 1e9
        print(f"  Uncompressed: {tar_gb:.1f} GB  Packed in {time.time()-t2:.0f}s")

        # ── zstd compress ─────────────────────────────────────────────────
        print(f"  Compressing → {output_path}")
        t3 = time.time()
        subprocess.run(
            ["zstd", f"-{ZSTD_LEVEL}", "-T0", "--rm", inter_tar, "-o", output_path],
            check=True,
        )
        final_gb = os.path.getsize(output_path) / 1e9
        print(f"  Compressed in {(time.time()-t3)/60:.1f} min  final={final_gb:.1f} GB")
        print(f"  Total: {(time.time()-t0)/60:.1f} min")


# ── Depth archive ─────────────────────────────────────────────────────────────

def build_depth_archive(
    dish_ids: list[str],
    output_path: str,
    *,
    tar_path:     str | None = None,
    all_members:  list[str] | None = None,
    overhead_dir: str | None = None,
):
    total    = len(dish_ids)
    label    = os.path.basename(output_path)
    from_tar = tar_path is not None
    dish_set = set(dish_ids)

    print(f"\n{'─'*60}")
    print(f"Building: {label}  ({total:,} dishes, copy-only)")
    print(f"  Source: {'tar' if from_tar else overhead_dir}")
    print(f"{'─'*60}")
    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix="n5k_depth_") as tmp:
        if from_tar:
            png_members = [
                m for m in all_members
                if (m.startswith(_OVER_ROOT + "/")
                    and m.split("/")[3] in dish_set
                    and (m.endswith("rgb.png") or m.endswith("depth_raw.png")))
            ]
            extracted = os.path.join(tmp, "extracted")
            print(f"  Extracting {len(png_members):,} PNG files (single tar pass)…")
            t1 = time.time()
            _tar_extract(tar_path, extracted, png_members, strip=3)
            print(f"  Extracted in {time.time()-t1:.0f}s")
            # rename extracted/ → realsense_overhead/ for correct archive layout
            realsense_dir = os.path.join(tmp, "realsense_overhead")
            os.rename(extracted, realsense_dir)
        else:
            realsense_dir = os.path.join(tmp, "realsense_overhead")
            os.makedirs(realsense_dir)
            missing = 0
            for did in tqdm(dish_ids, desc="  Copying", unit="dish", leave=False):
                dst = os.path.join(realsense_dir, did)
                os.makedirs(dst, exist_ok=True)
                for fname in ("rgb.png", "depth_raw.png"):
                    src = os.path.join(overhead_dir, did, fname)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                    else:
                        missing += 1
            if missing:
                print(f"  Warning: {missing} missing files")

        # ── Pack ──────────────────────────────────────────────────────────
        print(f"  Compressing → {output_path}")
        t2      = time.time()
        tar_tmp = output_path + ".tar"
        subprocess.run(["tar", "-cf", tar_tmp, "-C", tmp, "realsense_overhead"], check=True)
        tar_gb = os.path.getsize(tar_tmp) / 1e9
        print(f"  Uncompressed: {tar_gb:.2f} GB  → zstd -{ZSTD_LEVEL} -T0")
        subprocess.run(
            ["zstd", f"-{ZSTD_LEVEL}", "-T0", "--rm", tar_tmp, "-o", output_path],
            check=True,
        )
        print(f"  Done in {(time.time()-t0)/60:.1f} min  "
              f"size={os.path.getsize(output_path)/1e9:.2f} GB")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tar",     help="Path to nutrition5k_dataset.tar (recommended)")
    src.add_argument("--raw-dir", help="Pre-extracted dataset root (requires --splits-dir)")
    ap.add_argument("--splits-dir", help="Split txt directory (only for --raw-dir)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", default="all",
                    choices=["all", "rgb_train", "rgb_test", "depth_train", "depth_test"])
    ap.add_argument("--workers", type=int, default=FFMPEG_WORKERS,
                    help=f"Parallel ffmpeg workers (default: {FFMPEG_WORKERS})")
    args = ap.parse_args()

    if args.raw_dir and not args.splits_dir:
        ap.error("--splits-dir required with --raw-dir")

    os.makedirs(args.output_dir, exist_ok=True)

    # depth first (fast feedback), rgb_test before rgb_train (smaller first)
    all_keys = ["depth_train", "depth_test", "rgb_test", "rgb_train"]
    targets  = all_keys if args.mode == "all" else [args.mode]
    pending  = []
    for t in targets:
        out = os.path.join(args.output_dir, f"{t}.tar.zst")
        if os.path.isfile(out):
            print(f"Skipping {t} — archive exists")
        else:
            pending.append(t)
    if not pending:
        print("All archives already exist."); return

    t_total = time.time()

    if args.tar:
        splits      = _read_splits_from_tar(args.tar)
        all_members = _tar_list(args.tar)

        for target in pending:
            out = os.path.join(args.output_dir, f"{target}.tar.zst")
            ids = splits[target]
            if target.startswith("rgb"):
                build_rgb_archive(
                    ids, out, args.workers,
                    tar_path=args.tar, all_members=all_members,
                )
            else:
                build_depth_archive(
                    ids, out,
                    tar_path=args.tar, all_members=all_members,
                )
    else:
        side_angles = os.path.join(args.raw_dir, "imagery", "side_angles")
        overhead    = os.path.join(args.raw_dir, "imagery", "realsense_overhead")
        split_files = {k: os.path.join(args.splits_dir, f"{k}_ids.txt") for k in all_keys}
        for k, p in split_files.items():
            if not os.path.isfile(p):
                sys.exit(f"Split file not found: {p}")
        for target in pending:
            out = os.path.join(args.output_dir, f"{target}.tar.zst")
            ids = [l.strip() for l in open(split_files[target]) if l.strip()]
            if target.startswith("rgb"):
                build_rgb_archive(ids, out, args.workers, side_angles_dir=side_angles)
            else:
                build_depth_archive(ids, out, overhead_dir=overhead)

    print(f"\nAll done in {(time.time()-t_total)/60:.1f} min")
    print(f"Archives: {args.output_dir}")


if __name__ == "__main__":
    main()
