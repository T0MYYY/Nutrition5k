from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Nutrition5k overhead dish folder integrity.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to local Nutrition5k root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overhead_root = Path(args.dataset_root) / "imagery" / "realsense_overhead"
    if not overhead_root.exists():
        raise FileNotFoundError(f"Overhead directory not found: {overhead_root}")

    required = {"rgb.png", "depth_raw.png", "depth_color.png"}
    total = 0
    complete = 0
    partial = 0
    partial_examples: list[str] = []

    for dish_dir in sorted(overhead_root.iterdir()):
        if not dish_dir.is_dir():
            continue
        total += 1
        names = {p.name for p in dish_dir.iterdir() if p.is_file()}
        if required.issubset(names):
            complete += 1
        else:
            partial += 1
            if len(partial_examples) < 20:
                missing = sorted(required - names)
                partial_examples.append(f"{dish_dir.name}: missing {missing}")

    print(f"total_dish_dirs={total}")
    print(f"complete_triplets={complete}")
    print(f"partial_dirs={partial}")
    if partial_examples:
        print("partial_examples:")
        for line in partial_examples:
            print(f"  - {line}")


if __name__ == "__main__":
    main()
