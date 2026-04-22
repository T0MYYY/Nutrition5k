from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm

from config import build_arg_parser, config_from_args
from data_loader import create_dataloaders
from model import CalorieRegressor
from utils import mae, pick_device, rmse, set_seed, write_predictions_csv


def _to_kcal_predictions(preds: torch.Tensor, use_log_target: bool) -> torch.Tensor:
    if not use_log_target:
        return preds
    kcal_preds = torch.expm1(torch.clamp(preds, max=12.0))
    return torch.clamp(kcal_preds, min=0.0)


def main():
    parser = build_arg_parser(train=False)
    args = parser.parse_args()
    cfg = config_from_args(args, train=False)
    cfg.ensure_paths()
    set_seed(cfg.seed)

    device = pick_device(cfg.device)
    print(f"Using device: {device}")

    _, _, test_loader = create_dataloaders(
        dataset_root=cfg.dataset_root,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        val_ratio=0.1,
        seed=cfg.seed,
        mode=cfg.mode,
        split_type=cfg.split_type,
        max_depth_units=cfg.max_depth_units,
        augment_train=False,
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    food101_classes = checkpoint.get("food101_classes", [])
    num_classes = len(food101_classes) if checkpoint.get("has_classifier", False) else 0
    model = CalorieRegressor(
        mode=cfg.mode, pretrained=False, num_classes=num_classes
    ).to(device)
    ckpt_mode = checkpoint.get("mode", cfg.mode)
    if ckpt_mode != cfg.mode:
        raise ValueError(
            f"Checkpoint mode ({ckpt_mode}) does not match --mode ({cfg.mode})."
        )
    ckpt_split_type = checkpoint.get("split_type", cfg.split_type)
    if ckpt_split_type != cfg.split_type:
        raise ValueError(
            f"Checkpoint split_type ({ckpt_split_type}) does not match --split_type ({cfg.split_type})."
        )
    use_log_target = checkpoint.get("use_log_target", cfg.use_log_target)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []
    all_dish_ids = []

    with torch.no_grad():
        for images, targets, dish_ids in tqdm(test_loader, desc="test"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            preds = model(images)
            preds = _to_kcal_predictions(preds, use_log_target=use_log_target)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_dish_ids.extend(dish_ids)

    preds_tensor = torch.cat(all_preds, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    test_mae = mae(preds_tensor, targets_tensor).item()
    test_rmse = rmse(preds_tensor, targets_tensor).item()
    test_mse = torch.mean((preds_tensor - targets_tensor) ** 2).item()

    flat_preds = preds_tensor.squeeze(1).tolist()
    flat_targets = targets_tensor.squeeze(1).tolist()
    write_predictions_csv(cfg.save_predictions_csv, all_dish_ids, flat_preds, flat_targets)

    metrics = {
        "mae": test_mae,
        "rmse": test_rmse,
        "mse": test_mse,
        "mode": cfg.mode,
        "use_log_target": use_log_target,
    }
    print(json.dumps(metrics, indent=2))
    metrics_path = Path(cfg.output_dir) / "logs" / "eval_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
