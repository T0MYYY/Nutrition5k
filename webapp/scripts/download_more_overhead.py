from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import re
import subprocess


def read_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open() as f:
        for row in csv.reader(f):
            if not row:
                continue
            dish_id = row[0].strip()
            if dish_id and dish_id.lower() != "dish_id":
                ids.append(dish_id)
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incrementally download Nutrition5k overhead dishes.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to local mini/full Nutrition5k root (contains metadata, dish_ids, imagery).",
    )
    parser.add_argument(
        "--target_total",
        type=int,
        default=1000,
        help="Stop when local overhead dish folder count reaches this value.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.dataset_root)
    split_dir = data_root / "dish_ids" / "splits"
    overhead_dir = data_root / "imagery" / "realsense_overhead"
    overhead_dir.mkdir(parents=True, exist_ok=True)

    files = list(split_dir.rglob("*.txt")) + list(split_dir.rglob("*.csv"))
    depth_train = next((p for p in files if "depth_train" in p.name.lower()), None)
    depth_test = next((p for p in files if "depth_test" in p.name.lower()), None)
    if depth_train is None or depth_test is None:
        raise SystemExit("Cannot find depth_train/depth_test split files.")

    train_ids = read_ids(depth_train)
    test_ids = read_ids(depth_test)

    ls_cmd = [
        "gsutil",
        "ls",
        "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/",
    ]
    ls_res = subprocess.run(ls_cmd, capture_output=True, text=True, check=False)
    if ls_res.returncode != 0:
        raise SystemExit(f"gsutil ls failed:\n{ls_res.stderr}")

    existing_remote: set[str] = set()
    for line in ls_res.stdout.splitlines():
        match = re.search(r"(dish_\d+)/?$", line.strip())
        if match:
            existing_remote.add(match.group(1))
    if not existing_remote:
        raise SystemExit("No remote overhead dish IDs parsed from gsutil output.")

    train_candidates = [x for x in train_ids if x in existing_remote]
    test_candidates = [x for x in test_ids if x in existing_remote]

    local_existing = {d.name for d in overhead_dir.iterdir() if d.is_dir()}
    train_candidates = [x for x in train_candidates if x not in local_existing]
    test_candidates = [x for x in test_candidates if x not in local_existing]

    current = len(local_existing)
    print(f"current_dish_count = {current}")
    if current >= args.target_total:
        print(f"Already >= target_total ({args.target_total}).")
        return

    need = args.target_total - current
    need_train = int(need * args.train_ratio)
    need_test = need - need_train

    rng = random.Random(args.seed)
    rng.shuffle(train_candidates)
    rng.shuffle(test_candidates)

    selected = train_candidates[:need_train] + test_candidates[:need_test]
    rng.shuffle(selected)

    print(f"remote_overhead_available = {len(existing_remote)}")
    print(f"selected_to_download = {len(selected)}")

    downloaded = 0
    for i, dish_id in enumerate(selected, 1):
        src = f"gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/{dish_id}"
        cp_cmd = [
            "gsutil",
            "-m",
            "-o",
            "GSUtil:parallel_process_count=1",
            "cp",
            "-r",
            src,
            str(overhead_dir) + "/",
        ]
        cp_res = subprocess.run(cp_cmd, capture_output=True, text=True)
        if cp_res.returncode == 0:
            downloaded += 1
        else:
            print(f"[warn] failed: {dish_id}")

        if i % 20 == 0 or i == len(selected):
            now = sum(1 for d in overhead_dir.iterdir() if d.is_dir())
            print(f"[{i}/{len(selected)}] downloaded_ok={downloaded}, current_total={now}")

    final_count = sum(1 for d in overhead_dir.iterdir() if d.is_dir())
    print(f"final_dish_count = {final_count}")


if __name__ == "__main__":
    main()
