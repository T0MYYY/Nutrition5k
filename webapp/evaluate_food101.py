from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from data_loader import create_food101_test_dataloader
from model import CalorieRegressor
from utils import pick_device, set_seed


def _expand_for_rgbd_if_needed(images: torch.Tensor, mode: str) -> torch.Tensor:
    if mode != "rgbd":
        return images
    if images.shape[1] == 4:
        return images
    depth_zeros = torch.zeros(
        (images.shape[0], 1, images.shape[2], images.shape[3]),
        dtype=images.dtype,
        device=images.device,
    )
    return torch.cat([images, depth_zeros], dim=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Food-101 test accuracy from a checkpoint with classifier head.")
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--food101_root", type=str, required=True, help="Root passed to torchvision Food101 (same as training).")
    p.add_argument("--mode", type=str, choices=["rgb", "rgbd"], required=True, help="Must match checkpoint mode.")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--food101_download", action="store_true")
    p.add_argument("--top_k", type=int, default=5, help="Also report top-k accuracy (k<=101).")
    p.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="If set, writes logs/food101_test_metrics.json under this directory.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    if not ckpt.get("has_classifier", False):
        raise ValueError("Checkpoint has no classifier head; train with --enable_food101_cls first.")

    food101_classes = ckpt.get("food101_classes", [])
    if not food101_classes:
        raise ValueError("Checkpoint missing food101_classes list.")

    ckpt_mode = ckpt.get("mode", args.mode)
    if ckpt_mode != args.mode:
        raise ValueError(f"Checkpoint mode ({ckpt_mode}) does not match --mode ({args.mode}).")

    image_size = int(ckpt.get("image_size", args.image_size))
    num_classes = len(food101_classes)

    test_loader, ds_classes = create_food101_test_dataloader(
        root=args.food101_root,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.food101_download,
    )
    if ds_classes != food101_classes:
        raise ValueError(
            "food101_classes in checkpoint does not match torchvision Food101 class order on disk. "
            "Use the same --food101_root (and dataset version) as training."
        )

    model = CalorieRegressor(
        mode=args.mode,
        pretrained=False,
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    k = max(1, min(args.top_k, num_classes))
    correct_top1 = 0
    correct_topk = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="food101_test"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images = _expand_for_rgbd_if_needed(images, mode=args.mode)
            logits = model.classify(images)
            pred1 = logits.argmax(dim=1)
            correct_top1 += (pred1 == labels).sum().item()
            if k > 1:
                topk = logits.topk(k, dim=1).indices
                correct_topk += (topk == labels.unsqueeze(1)).any(dim=1).sum().item()
            else:
                correct_topk += (pred1 == labels).sum().item()
            total += labels.size(0)

    acc1 = correct_top1 / total
    acck = correct_topk / total
    metrics = {
        "food101_test_samples": total,
        "top1_accuracy": acc1,
        f"top{k}_accuracy": acck,
        "mode": args.mode,
        "image_size": image_size,
        "checkpoint_path": str(args.checkpoint_path),
    }
    print(json.dumps(metrics, indent=2))

    if args.output_dir:
        out = Path(args.output_dir) / "logs" / "food101_test_metrics.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))
        print(f"Saved metrics to {out}")


if __name__ == "__main__":
    main()
