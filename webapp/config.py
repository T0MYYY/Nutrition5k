from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
from typing import List, Optional


def food101_scheduled_epoch_list(num_epochs: int, interval: int) -> List[int]:
    """1-based epoch indices where Food-101 classification runs (same rule as train loop)."""
    if num_epochs <= 0 or interval <= 0:
        return []
    return [e for e in range(1, num_epochs + 1) if (e - 1) % interval == 0]


def compute_food101_epoch_interval(num_epochs: int, every_n_epochs: int, cls_passes: int) -> int:
    """Stride for Food-101 cls steps (1 = every epoch). Used when ``enable_food101_cls`` is True."""
    if every_n_epochs > 0:
        return max(1, every_n_epochs)
    e = num_epochs
    p = cls_passes
    if p <= 0:
        return 1
    pcap = min(p, e)
    if pcap <= 1:
        return max(1, e)
    if pcap >= e:
        return 1
    return max(1, (e - 1) // (pcap - 1))


def food101_schedule_explain(cfg: Config) -> tuple[str, List[int]]:
    """Human-readable schedule line and 1-based epoch list (empty if Food-101 cls disabled)."""
    if not cfg.enable_food101_cls:
        return "", []
    ep_list = food101_scheduled_epoch_list(cfg.epochs, cfg.food101_epoch_interval)
    if cfg.food101_every_n_epochs > 0:
        desc = f"every_n_epochs={cfg.food101_every_n_epochs} (overrides cls_passes for schedule)"
    elif cfg.food101_cls_passes > 0:
        desc = f"cls_passes={cfg.food101_cls_passes} -> interval={cfg.food101_epoch_interval} epoch(s)"
    else:
        desc = "cls_passes=0, every_n=0 (Food-101 every epoch)"
    return desc, ep_list


@dataclass
class Config:
    dataset_root: str
    output_dir: str = "outputs"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    val_ratio: float = 0.1
    mode: str = "rgb"
    split_type: str = "auto"
    pretrained: bool = True
    loss_type: str = "smooth_l1"
    use_log_target: bool = True
    scheduler: str = "plateau"
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    early_stop_patience: int = 10
    min_improve_delta: float = 1e-3
    augment_train: bool = True
    backbone_lr: Optional[float] = None
    head_lr: Optional[float] = None
    enable_food101_cls: bool = False
    food101_root: str = "data"
    food101_download: bool = False
    cls_loss_weight: float = 1.0
    cls_batch_size: int = 32
    food101_augment: bool = True
    food101_val_ratio: float = 0.1
    cls_label_smoothing: float = 0.0
    # Food-101 cls: train loop only checks (epoch-1) % food101_epoch_interval == 0.
    # The next two are alternate *inputs* for computing that interval (every_n wins if >0).
    food101_cls_passes: int = 0
    food101_every_n_epochs: int = 0
    food101_epoch_interval: int = 1  # filled in config_from_args; do not pass on CLI
    max_depth_units: float = 4000.0
    checkpoint_path: str = ""
    save_predictions_csv: str = ""
    device: str = ""

    def ensure_paths(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir, "checkpoints").mkdir(parents=True, exist_ok=True)
        Path(self.output_dir, "logs").mkdir(parents=True, exist_ok=True)


def build_arg_parser(train: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to nutrition5k_dataset")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mode", type=str, choices=["rgb", "rgbd"], default="rgb")
    parser.add_argument(
        "--split_type",
        type=str,
        choices=["auto", "rgb", "depth"],
        default="auto",
        help="Which official split files to use from dish_ids/splits.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--loss_type", type=str, choices=["mse", "smooth_l1"], default="smooth_l1")
    parser.add_argument("--use_log_target", action="store_true")
    parser.add_argument("--no_log_target", action="store_true")
    parser.add_argument("--scheduler", type=str, choices=["none", "plateau"], default="plateau")
    parser.add_argument("--scheduler_patience", type=int, default=3)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--min_improve_delta", type=float, default=1e-3)
    parser.add_argument("--enable_food101_cls", action="store_true")
    parser.add_argument("--food101_root", type=str, default="data")
    parser.add_argument("--food101_download", action="store_true")
    parser.add_argument("--cls_loss_weight", type=float, default=1.0)
    parser.add_argument("--cls_batch_size", type=int, default=32)
    parser.add_argument(
        "--no_food101_augment",
        action="store_true",
        help="Disable Food-101 train augmentations (RandomResizedCrop, flip, jitter, erasing).",
    )
    parser.add_argument(
        "--food101_val_ratio",
        type=float,
        default=0.1,
        help="Fraction of Food-101 official train split held out for cls validation.",
    )
    parser.add_argument(
        "--cls_label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for Food-101 cross-entropy (e.g. 0.05). 0 disables.",
    )
    parser.add_argument(
        "--food101_cls_passes",
        type=int,
        default=0,
        help="With --enable_food101_cls: 0 = run Food-101 every epoch; P>0 = run about P times spread over all --epochs (e.g. E=40 P=4 → epochs 1,14,27,40). Ignored for scheduling if --food101_every_n_epochs > 0.",
    )
    parser.add_argument(
        "--food101_every_n_epochs",
        type=int,
        default=0,
        help="With --enable_food101_cls: if N>0, run Food-101 every N epochs (1, N+1, 2N+1, …). Overrides --food101_cls_passes for scheduling. 0 = use cls_passes only.",
    )
    parser.add_argument("--max_depth_units", type=float, default=4000.0)
    parser.add_argument("--device", type=str, default="")
    if train:
        parser.add_argument("--epochs", type=int, default=20)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--val_ratio", type=float, default=0.1)
        parser.add_argument(
            "--no_train_augment",
            action="store_true",
            help="Disable train-only flips and color jitter (val/test unchanged).",
        )
        parser.add_argument(
            "--backbone_lr",
            type=float,
            default=None,
            help="If set with --head_lr, backbone uses this LR and heads use --head_lr.",
        )
        parser.add_argument(
            "--head_lr",
            type=float,
            default=None,
            help="Head LR when used with --backbone_lr; ignored otherwise.",
        )
    else:
        parser.add_argument("--checkpoint_path", type=str, required=True)
        parser.add_argument("--save_predictions_csv", type=str, default="")
    return parser


def config_from_args(args: argparse.Namespace, train: bool = True) -> Config:
    use_pretrained = True
    if getattr(args, "no_pretrained", False):
        use_pretrained = False
    elif getattr(args, "pretrained", False):
        use_pretrained = True

    use_log_target = True
    if getattr(args, "no_log_target", False):
        use_log_target = False
    elif getattr(args, "use_log_target", False):
        use_log_target = True

    cfg = Config(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode=args.mode,
        split_type=args.split_type,
        seed=args.seed,
        pretrained=use_pretrained,
        loss_type=args.loss_type,
        use_log_target=use_log_target,
        scheduler=args.scheduler,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        early_stop_patience=args.early_stop_patience,
        min_improve_delta=args.min_improve_delta,
        augment_train=not getattr(args, "no_train_augment", False) if train else False,
        backbone_lr=getattr(args, "backbone_lr", None) if train else None,
        head_lr=getattr(args, "head_lr", None) if train else None,
        enable_food101_cls=args.enable_food101_cls,
        food101_root=args.food101_root,
        food101_download=args.food101_download,
        cls_loss_weight=args.cls_loss_weight,
        cls_batch_size=args.cls_batch_size,
        food101_augment=not getattr(args, "no_food101_augment", False) if train else True,
        food101_val_ratio=getattr(args, "food101_val_ratio", 0.1),
        cls_label_smoothing=getattr(args, "cls_label_smoothing", 0.0),
        food101_cls_passes=max(0, getattr(args, "food101_cls_passes", 0)) if train else 0,
        food101_every_n_epochs=max(0, getattr(args, "food101_every_n_epochs", 0)) if train else 0,
        food101_epoch_interval=1,
        max_depth_units=args.max_depth_units,
        device=args.device,
    )
    if train:
        cfg.epochs = args.epochs
        cfg.lr = args.lr
        cfg.weight_decay = args.weight_decay
        cfg.val_ratio = args.val_ratio
        if cfg.enable_food101_cls:
            cfg.food101_epoch_interval = compute_food101_epoch_interval(
                cfg.epochs, cfg.food101_every_n_epochs, cfg.food101_cls_passes
            )
        else:
            cfg.food101_epoch_interval = 1
    else:
        cfg.checkpoint_path = args.checkpoint_path
        cfg.save_predictions_csv = args.save_predictions_csv
    return cfg
