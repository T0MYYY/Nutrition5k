#!/usr/bin/env python3
"""Download Nutrition5k from the official GCS bucket (gs://nutrition5k_dataset).

This repo only needs overhead RGB-D for training; you do NOT need the 181 GB
full tarball unless you want side-angle videos too.

Tiers:
  essentials  — metadata/ + dish_ids/  (MB, always do this first)
  overhead    — imagery/realsense_overhead/  (~3.5k dishes; what train.py uses)
  full          — entire nutrition5k_dataset/ including side_angles/ (~181 GB)

Examples:
  # New dataset root (recommended for full download)
  python scripts/download_nutrition5k.py \\
    --dataset_root ~/data/nutrition5k_dataset \\
    --tier essentials overhead

  # Resume / fill gaps
  python scripts/download_nutrition5k.py \\
    --dataset_root ~/data/nutrition5k_dataset \\
    --tier overhead --only_missing

  # Dry-run: show what would be copied
  python scripts/download_nutrition5k.py --dataset_root ~/data/nutrition5k --tier overhead --dry_run
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Set

GCS_ROOT = "gs://nutrition5k_dataset/nutrition5k_dataset"
# Remote overhead has ~3490 dish folders (not all 5006 paper IDs have overhead scans).
EXPECTED_REMOTE_OVERHEAD = 3490


def _require_gsutil() -> str:
    gsutil = shutil.which("gsutil")
    if not gsutil:
        raise SystemExit(
            "gsutil not found. Install Google Cloud SDK:\n"
            "  https://cloud.google.com/storage/docs/gsutil_install\n"
            "Then run: gcloud auth login   (or use a service account)"
        )
    return gsutil


def _run(cmd: List[str], *, dry_run: bool) -> int:
    print("$", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def _gcs_cp_r(gsutil: str, src: str, dst: Path, *, dry_run: bool, parallel: bool = True) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [gsutil]
    if parallel:
        cmd.append("-m")
    cmd += [
        "-o",
        "GSUtil:parallel_thread_count=8",
        "-o",
        "GSUtil:parallel_process_count=4",
        "cp",
        "-r",
        src,
        str(dst),
    ]
    return _run(cmd, dry_run=dry_run)


def _local_dish_ids(overhead_dir: Path) -> Set[str]:
    if not overhead_dir.is_dir():
        return set()
    return {d.name for d in overhead_dir.iterdir() if d.is_dir() and d.name.startswith("dish_")}


def _remote_dish_ids(gsutil: str, *, dry_run: bool) -> Set[str]:
    if dry_run:
        return set()
    cmd = [gsutil, "ls", f"{GCS_ROOT}/imagery/realsense_overhead/"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise SystemExit(f"gsutil ls failed:\n{res.stderr}")
    ids: Set[str] = set()
    for line in res.stdout.splitlines():
        m = re.search(r"(dish_\d+)/?$", line.strip())
        if m:
            ids.add(m.group(1))
    return ids


def _split_dish_ids(dataset_root: Path) -> Set[str]:
    split_dir = dataset_root / "dish_ids" / "splits"
    if not split_dir.is_dir():
        return set()
    ids: Set[str] = set()
    for path in split_dir.rglob("*.txt"):
        with path.open() as f:
            for row in csv.reader(f):
                if not row:
                    continue
                dish_id = row[0].strip()
                if dish_id and dish_id.lower() != "dish_id":
                    ids.add(dish_id)
    return ids


def _read_ids_file(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open() as f:
        for row in csv.reader(f):
            if not row:
                continue
            dish_id = row[0].strip()
            if dish_id and dish_id.lower() != "dish_id":
                ids.append(dish_id)
    return ids


def download_essentials(gsutil: str, root: Path, *, dry_run: bool) -> None:
    print("\n== Tier: essentials (metadata + dish_ids) ==")
    for sub in ("metadata", "dish_ids"):
        src = f"{GCS_ROOT}/{sub}"
        dst = root / sub
        code = _gcs_cp_r(gsutil, src, dst, dry_run=dry_run)
        if code != 0:
            raise SystemExit(f"Failed copying {sub} (exit {code})")


def download_overhead_bulk(gsutil: str, root: Path, *, dry_run: bool) -> None:
    print("\n== Tier: overhead (full realsense_overhead/) ==")
    imagery = root / "imagery"
    imagery.mkdir(parents=True, exist_ok=True)
    src = f"{GCS_ROOT}/imagery/realsense_overhead"
    # cp -r .../realsense_overhead -> imagery/  creates imagery/realsense_overhead/
    code = _gcs_cp_r(gsutil, src, imagery, dry_run=dry_run)
    if code != 0:
        raise SystemExit(f"Failed copying realsense_overhead (exit {code})")


def download_overhead_missing(
    gsutil: str,
    root: Path,
    *,
    dry_run: bool,
    use_splits: bool,
    limit: int,
) -> None:
    print("\n== Tier: overhead (only missing dishes) ==")
    overhead_dir = root / "imagery" / "realsense_overhead"
    overhead_dir.mkdir(parents=True, exist_ok=True)

    local = _local_dish_ids(overhead_dir)
    remote = _remote_dish_ids(gsutil, dry_run=dry_run)
    if dry_run:
        print("dry_run: skipping remote ls; cannot compute missing list accurately.")
        return

    if use_splits:
        target = _split_dish_ids(root) & remote
        print(f"split_ids_in_files={len(_split_dish_ids(root))}, with_remote_overhead={len(target)}")
    else:
        target = remote

    missing = sorted(target - local)
    print(f"local_dishes={len(local)}, remote_overhead={len(remote)}, to_download={len(missing)}")
    if limit > 0:
        missing = missing[:limit]
        print(f"limited to first {len(missing)} dishes")

    ok = 0
    for i, dish_id in enumerate(missing, 1):
        src = f"{GCS_ROOT}/imagery/realsense_overhead/{dish_id}"
        cmd = [
            gsutil,
            "-m",
            "-o",
            "GSUtil:parallel_process_count=2",
            "cp",
            "-r",
            src,
            str(overhead_dir) + "/",
        ]
        code = _run(cmd, dry_run=dry_run)
        if code == 0:
            ok += 1
        else:
            print(f"[warn] failed: {dish_id}")
        if i % 50 == 0 or i == len(missing):
            now = len(_local_dish_ids(overhead_dir))
            print(f"[{i}/{len(missing)}] ok={ok}, local_total={now}")

    print(f"done: downloaded_ok={ok}, final_local={len(_local_dish_ids(overhead_dir))}")


def download_full(gsutil: str, root: Path, *, dry_run: bool) -> None:
    print("\n== Tier: full dataset (~181 GB, includes side-angle videos) ==")
    print("This copies the entire bucket folder. Ensure you have ~200 GB free disk.")
    code = _gcs_cp_r(gsutil, GCS_ROOT, root.parent, dry_run=dry_run)
    # cp nutrition5k_dataset -> parent creates parent/nutrition5k_dataset/
    if code != 0:
        raise SystemExit(f"Failed full copy (exit {code})")
    print(f"Dataset should appear under: {root.parent / 'nutrition5k_dataset'}")


def print_summary(root: Path) -> None:
    overhead = root / "imagery" / "realsense_overhead"
    n = len(_local_dish_ids(overhead)) if overhead.is_dir() else 0
    meta_ok = (root / "metadata" / "dish_metadata_cafe1.csv").is_file()
    split_ok = (root / "dish_ids" / "splits").is_dir()
    print("\n== Local summary ==")
    print(f"  dataset_root     = {root}")
    print(f"  metadata OK      = {meta_ok}")
    print(f"  splits OK        = {split_ok}")
    print(f"  overhead dishes  = {n}  (remote bucket has ~{EXPECTED_REMOTE_OVERHEAD})")
    if n > 0:
        print(f"\n  Train with:\n    --dataset_root \"{root}\"")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Nutrition5k from official GCS.")
    p.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Local root (will contain metadata/, dish_ids/, imagery/).",
    )
    p.add_argument(
        "--tier",
        nargs="+",
        choices=["essentials", "overhead", "full"],
        default=["essentials", "overhead"],
        help="What to download (default: essentials overhead).",
    )
    p.add_argument(
        "--only_missing",
        action="store_true",
        help="For overhead: copy dish folders not yet local (resume-friendly).",
    )
    p.add_argument(
        "--use_splits",
        action="store_true",
        default=True,
        help="When --only_missing: only fetch IDs listed in dish_ids/splits (default: true).",
    )
    p.add_argument(
        "--all_remote",
        action="store_true",
        help="When --only_missing: fetch every remote overhead dish, ignore split files.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="With --only_missing: max dishes to download this run (0 = no limit).",
    )
    p.add_argument("--dry_run", action="store_true", help="Print gsutil commands only.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    gsutil = _require_gsutil()

    tiers = list(args.tier)
    if "full" in tiers and len(tiers) > 1:
        print("Note: 'full' includes everything; other tiers may be redundant.", file=sys.stderr)

    if "full" in tiers:
        download_full(gsutil, root, dry_run=args.dry_run)
    else:
        if "essentials" in tiers:
            download_essentials(gsutil, root, dry_run=args.dry_run)
        if "overhead" in tiers:
            if args.only_missing:
                download_overhead_missing(
                    gsutil,
                    root,
                    dry_run=args.dry_run,
                    use_splits=not args.all_remote,
                    limit=args.limit,
                )
            else:
                download_overhead_bulk(gsutil, root, dry_run=args.dry_run)

    if not args.dry_run:
        print_summary(root)


if __name__ == "__main__":
    main()
