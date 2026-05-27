from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import atexit
import csv
import json
import os
import platform
import random
import subprocess
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

_caffeinate_proc: Optional[subprocess.Popen] = None


def prevent_system_sleep_while_running() -> None:
    """On macOS, run ``caffeinate`` so the machine stays awake while this process runs.

    Uses ``caffeinate -dims -w <pid>`` (idle, display, disk, system on AC). No-op on other OS.
    """
    global _caffeinate_proc
    if _caffeinate_proc is not None:
        return
    if platform.system() != "Darwin":
        return
    try:
        _caffeinate_proc = subprocess.Popen(
            ["caffeinate", "-dims", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("macOS: preventing system sleep during training (caffeinate -dims).")
    except FileNotFoundError:
        print("[warn] caffeinate not found; macOS may sleep during long training.")
        return

    def _cleanup_caffeinate() -> None:
        global _caffeinate_proc
        proc = _caffeinate_proc
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        _caffeinate_proc = None

    atexit.register(_cleanup_caffeinate)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(user_device: str = "") -> torch.device:
    if user_device:
        return torch.device(user_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mae(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(preds - targets))


def rmse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((preds - targets) ** 2))


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count


def save_checkpoint(state: Dict, output_dir: str, filename: str) -> str:
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / filename
    torch.save(state, path)
    return str(path)


def log_epoch(output_dir: str, row: Dict) -> None:
    log_path = Path(output_dir) / "logs" / "train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_run_config(output_dir: str, cfg_obj) -> None:
    cfg_path = Path(output_dir) / "logs" / "config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(cfg_obj)
    payload["timestamp"] = datetime.utcnow().isoformat() + "Z"
    with cfg_path.open("w") as f:
        json.dump(payload, f, indent=2)


def write_predictions_csv(path: str, dish_ids: Iterable[str], preds: Iterable[float], targets: Iterable[float]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dish_id", "predicted_calories", "target_calories", "abs_error"])
        for dish_id, pred, target in zip(dish_ids, preds, targets):
            writer.writerow([dish_id, float(pred), float(target), abs(float(pred) - float(target))])
