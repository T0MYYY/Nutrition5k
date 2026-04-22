from __future__ import annotations

from datetime import UTC, datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import build_arg_parser, config_from_args, food101_schedule_explain
from data_loader import create_dataloaders, create_food101_dataloaders
from model import CalorieRegressor
from utils import (
    AverageMeter,
    log_epoch,
    mae,
    pick_device,
    prevent_system_sleep_while_running,
    save_checkpoint,
    save_run_config,
    set_seed,
)


def _food101_checkpoint_meta(cfg) -> dict:
    """Scheduler fields stored in checkpoints (see also logs/config.json)."""
    return {
        "food101_epoch_interval": cfg.food101_epoch_interval,
        "food101_cls_passes": cfg.food101_cls_passes,
        "food101_every_n_epochs": cfg.food101_every_n_epochs,
    }


def _build_optimizer(model, cfg) -> torch.optim.Optimizer:
    if cfg.backbone_lr is not None and cfg.head_lr is not None:
        backbone_p = list(model.backbone.parameters())
        head_p = list(model.reg_head.parameters())
        if model.cls_head is not None:
            head_p += list(model.cls_head.parameters())
        return torch.optim.AdamW(
            [
                {"params": backbone_p, "lr": cfg.backbone_lr, "weight_decay": cfg.weight_decay},
                {"params": head_p, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay},
            ],
        )
    if cfg.backbone_lr is not None or cfg.head_lr is not None:
        raise ValueError("Set both --backbone_lr and --head_lr, or neither (use --lr for all parameters).")
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def _optimizer_lr_string(optimizer: torch.optim.Optimizer) -> str:
    if len(optimizer.param_groups) == 1:
        return f"{optimizer.param_groups[0]['lr']:.2e}"
    return ", ".join(f"{g['lr']:.2e}" for g in optimizer.param_groups)


def _to_kcal_predictions(preds: torch.Tensor, use_log_target: bool) -> torch.Tensor:
    if not use_log_target:
        return preds
    kcal_preds = torch.expm1(torch.clamp(preds, max=12.0))
    return torch.clamp(kcal_preds, min=0.0)


def run_epoch(model, loader, criterion, device, use_log_target: bool, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    loss_meter = AverageMeter()
    mae_meter = AverageMeter()
    desc = "train" if is_train else "val"
    progress = tqdm(loader, desc=desc, leave=False)

    for images, targets, _ in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            preds = model(images)
            loss_targets = torch.log1p(targets) if use_log_target else targets
            loss = criterion(preds, loss_targets)
            kcal_preds = _to_kcal_predictions(preds, use_log_target=use_log_target)
            batch_mae = mae(kcal_preds, targets)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = images.shape[0]
        loss_meter.update(loss.item(), batch_size)
        mae_meter.update(batch_mae.item(), batch_size)
        progress.set_postfix({"loss": f"{loss_meter.avg:.4f}", "mae": f"{mae_meter.avg:.4f}"})
    return loss_meter.avg, mae_meter.avg


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


def run_cls_epoch(
    model,
    loader,
    device,
    mode: str,
    optimizer=None,
    loss_weight: float = 1.0,
    label_smoothing: float = 0.0,
):
    is_train = optimizer is not None
    model.train(is_train)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    desc = "food101_train" if is_train else "food101_val"
    progress = tqdm(loader, desc=desc, leave=False)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        images = _expand_for_rgbd_if_needed(images, mode=mode)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model.classify(images)
            ls = label_smoothing if is_train else 0.0
            cls_loss = F.cross_entropy(logits, labels, label_smoothing=ls) * loss_weight
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            if is_train:
                cls_loss.backward()
                optimizer.step()

        bs = images.shape[0]
        loss_meter.update(cls_loss.item(), bs)
        acc_meter.update(acc.item(), bs)
        progress.set_postfix({"cls_loss": f"{loss_meter.avg:.4f}", "cls_acc": f"{acc_meter.avg:.4f}"})

    return loss_meter.avg, acc_meter.avg


def main():
    parser = build_arg_parser(train=True)
    args = parser.parse_args()
    cfg = config_from_args(args, train=True)
    cfg.ensure_paths()
    set_seed(cfg.seed)
    prevent_system_sleep_while_running()

    device = pick_device(cfg.device)
    print(f"Using device: {device}")

    train_loader, val_loader, _ = create_dataloaders(
        dataset_root=cfg.dataset_root,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        mode=cfg.mode,
        split_type=cfg.split_type,
        max_depth_units=cfg.max_depth_units,
        augment_train=cfg.augment_train,
    )
    print(
        f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)} | "
        f"train_augment={cfg.augment_train}"
    )

    food101_train_loader = None
    food101_val_loader = None
    food101_classes = []
    if cfg.enable_food101_cls:
        food101_train_loader, food101_val_loader, food101_classes = create_food101_dataloaders(
            root=cfg.food101_root,
            image_size=cfg.image_size,
            batch_size=cfg.cls_batch_size,
            num_workers=cfg.num_workers,
            download=cfg.food101_download,
            augment_food101=cfg.food101_augment,
            food101_val_ratio=cfg.food101_val_ratio,
            seed=cfg.seed,
        )
        sched, ep_list = food101_schedule_explain(cfg)
        ep_preview = ep_list if len(ep_list) <= 20 else ep_list[:10] + ["…"] + ep_list[-5:]
        print(
            f"Food-101 cls | train aug={cfg.food101_augment} | val_ratio={cfg.food101_val_ratio} | "
            f"cls_label_smoothing={cfg.cls_label_smoothing} | {sched}"
        )
        print(
            f"Food-101 schedule | epochs {cfg.epochs} total | run on {len(ep_list)} epoch(s): {ep_preview}"
        )
        print(
            f"Food-101 samples: train={len(food101_train_loader.dataset)} | "
            f"val={len(food101_val_loader.dataset)}"
        )

    model = CalorieRegressor(
        mode=cfg.mode,
        pretrained=cfg.pretrained,
        num_classes=(len(food101_classes) if cfg.enable_food101_cls else 0),
    ).to(device)
    optimizer = _build_optimizer(model, cfg)
    if cfg.loss_type == "smooth_l1":
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()

    scheduler = None
    if cfg.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            patience=cfg.scheduler_patience,
            factor=cfg.scheduler_factor,
        )

    save_run_config(cfg.output_dir, cfg)
    best_val_mae = float("inf")
    best_epoch = 0
    epochs_without_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_mae = run_epoch(
            model, train_loader, criterion, device, cfg.use_log_target, optimizer=optimizer
        )
        val_loss, val_mae = run_epoch(model, val_loader, criterion, device, cfg.use_log_target, optimizer=None)

        if scheduler is not None:
            scheduler.step(val_mae)

        food101_train_loss = None
        food101_train_acc = None
        food101_val_loss = None
        food101_val_acc = None
        run_food101_cls = (
            cfg.enable_food101_cls
            and model.has_classifier
            and (epoch - 1) % cfg.food101_epoch_interval == 0
        )
        if run_food101_cls:
            food101_train_loss, food101_train_acc = run_cls_epoch(
                model=model,
                loader=food101_train_loader,
                device=device,
                mode=cfg.mode,
                optimizer=optimizer,
                loss_weight=cfg.cls_loss_weight,
                label_smoothing=cfg.cls_label_smoothing,
            )
            food101_val_loss, food101_val_acc = run_cls_epoch(
                model=model,
                loader=food101_val_loader,
                device=device,
                mode=cfg.mode,
                optimizer=None,
                loss_weight=cfg.cls_loss_weight,
                label_smoothing=0.0,
            )

        print(
            f"Epoch {epoch:03d}/{cfg.epochs:03d} "
            f"| train_loss={train_loss:.4f} train_mae={train_mae:.4f} "
            f"| val_loss={val_loss:.4f} val_mae={val_mae:.4f}"
        )
        if food101_train_acc is not None:
            print(
                f"  Food101 cls | train_loss={food101_train_loss:.4f} train_acc={food101_train_acc:.4f} "
                f"| val_loss={food101_val_loss:.4f} val_acc={food101_val_acc:.4f}"
            )
        elif cfg.enable_food101_cls and model.has_classifier and not run_food101_cls:
            print(
                f"  Food101 cls | skipped (food101_epoch_interval={cfg.food101_epoch_interval})"
            )

        log_row = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "lr": _optimizer_lr_string(optimizer),
        }
        if food101_train_acc is not None:
            log_row["food101_train_loss"] = food101_train_loss
            log_row["food101_train_acc"] = food101_train_acc
            log_row["food101_val_loss"] = food101_val_loss
            log_row["food101_val_acc"] = food101_val_acc
        log_epoch(cfg.output_dir, log_row)

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_mae": val_mae,
            "mode": cfg.mode,
            "split_type": cfg.split_type,
            "image_size": cfg.image_size,
            "max_depth_units": cfg.max_depth_units,
            "loss_type": cfg.loss_type,
            "use_log_target": cfg.use_log_target,
            "has_classifier": model.has_classifier,
            "food101_classes": food101_classes,
            "cls_label_smoothing": cfg.cls_label_smoothing,
            **_food101_checkpoint_meta(cfg),
        }
        save_checkpoint(state, cfg.output_dir, "last.pt")
        if val_mae < (best_val_mae - cfg.min_improve_delta):
            best_val_mae = val_mae
            best_epoch = epoch
            epochs_without_improve = 0
            best_path = save_checkpoint(state, cfg.output_dir, "best.pt")
            print(f"Saved new best checkpoint to {best_path}")
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= cfg.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch} "
                f"(no val MAE improvement for {cfg.early_stop_patience} epochs)."
            )
            break

    summary = {
        "best_val_mae": best_val_mae,
        "best_epoch": best_epoch,
        "mode": cfg.mode,
        "split_type": cfg.split_type,
        "loss_type": cfg.loss_type,
        "use_log_target": cfg.use_log_target,
        "has_classifier": model.has_classifier,
    }
    with open(f"{cfg.output_dir}/logs/train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Training complete.")


if __name__ == "__main__":
    main()
