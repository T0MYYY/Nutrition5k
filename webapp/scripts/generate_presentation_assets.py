#!/usr/bin/env python3
"""Build EDA plots, pipeline diagrams, training/metric tables, and eval figures.

No model training: reads Nutrition5k metadata + split logic, and any files under
``--run_dir`` (and optional ``--extra_run_dirs``).

Default output: ``<repo>/presentation/slide_assets/`` — copy PNG/CSV into PowerPoint.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, UnidentifiedImageError

from data_loader import DishSample, build_split_samples

# Slide-friendly palette
_C = {
    "blue": "#4C72B0",
    "orange": "#DD8452",
    "green": "#55A868",
    "purple": "#8172B3",
    "red": "#C44E52",
    "gray": "#8C8C8C",
}


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _fig_table(df: pd.DataFrame, title: str, out_path: Path, fontsize: int = 9) -> None:
    nrows, ncols = df.shape
    fig_w = min(16, 1.8 + ncols * 1.6)
    fig_h = max(2.0, 0.42 * (nrows + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=12, fontweight="bold")
    table = ax.table(
        cellText=df.astype(str).values,
        colLabels=list(df.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.05, 1.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _samples_to_df(splits: Dict[str, List[DishSample]]) -> pd.DataFrame:
    rows = []
    for name, samples in splits.items():
        for s in samples:
            rows.append(
                {
                    "split": name,
                    "dish_id": s.dish_id,
                    "calories": float(s.calories),
                    "has_depth": s.depth_path is not None,
                }
            )
    return pd.DataFrame(rows)


# --- Pipeline & architecture (static diagrams) ---


def _plot_pipeline_diagram(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("End-to-end pipeline (Nutrition5k → calories)", fontsize=13, fontweight="bold", pad=8)

    def box(xy: Tuple[float, float], w: float, h: float, text: str, color: str) -> None:
        x, y = xy
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#333",
            facecolor=color,
            alpha=0.92,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, wrap=True)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#444",
            )
        )

    # Row 1: data
    box((0.2, 2.3), 1.5, 0.9, "Metadata\n(total kcal)", "#E8F4FC")
    box((2.0, 2.3), 1.5, 0.9, "Overhead\nRGB / depth", "#E8F4FC")
    box((3.8, 2.3), 1.5, 0.9, "Official\nsplit IDs", "#E8F4FC")
    arrow(1.7, 2.75, 2.0, 2.75)
    arrow(3.5, 2.75, 3.8, 2.75)

    # Row 2: prep
    box((5.6, 2.3), 1.7, 0.9, "Train / val / test\n(local filter)", "#FFF3E0")
    box((7.6, 2.3), 1.6, 0.9, "Resize +\nImageNet norm", "#FFF3E0")
    box((9.4, 2.3), 1.4, 0.9, "RGB-D\n4ch (opt.)", "#FFF3E0")
    arrow(5.3, 2.75, 5.6, 2.75)
    arrow(7.3, 2.75, 7.6, 2.75)
    arrow(9.2, 2.75, 9.4, 2.75)

    # Model
    box((11.0, 2.3), 1.5, 0.9, "ResNet-18\nbackbone", "#E8F5E9")
    box((12.7, 2.55), 1.1, 0.55, "MLP → kcal", "#C8E6C9")
    box((12.7, 1.35), 1.1, 0.55, "Food-101\n(opt.)", "#C8E6C9")
    arrow(10.8, 2.75, 11.0, 2.75)
    arrow(12.5, 2.75, 12.7, 2.82)
    arrow(12.5, 2.75, 12.7, 1.62)

    # Training note
    box((0.3, 0.35), 6.2, 1.35, "Training: SmoothL1/MSE, log1p target (opt.), plateau LR,\nearly stop on val MAE; auxiliary Food-101 CE on scheduled epochs", "#F5F5F5")
    box((6.8, 0.35), 6.9, 1.35, "Inference: load best.pt → forward → kcal;\nevaluate.py → MAE / RMSE / prediction CSV", "#F5F5F5")

    fig.tight_layout()
    fig.savefig(out_dir / "pipeline_end_to_end.png", dpi=180, facecolor="white")
    plt.close(fig)


def _arch_box(
    ax: plt.Axes,
    xy: Tuple[float, float],
    w: float,
    h: float,
    text: str,
    face: str,
    edge: str = "#2C3E50",
    fontsize: float = 9.5,
    shadow: bool = True,
) -> Tuple[float, float, float, float]:
    """Draw rounded box with optional drop shadow; returns (cx, cy, w, h)."""
    x, y = xy
    if shadow:
        sh = FancyBboxPatch(
            (x + 0.06, y - 0.06),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            facecolor=(0, 0, 0, 0.08),
            edgecolor="none",
            zorder=1,
        )
        ax.add_patch(sh)
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.4,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#1a1a1a", zorder=3)
    return x, y, w, h


def _arch_arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float, style: str = "-") -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.6,
            color="#455A64",
            linestyle=style,
            zorder=0,
        )
    )


def _plot_model_architecture(out_dir: Path, mode: str = "rgbd") -> None:
    is_rgbd = mode == "rgbd"
    in_ch = 4 if is_rgbd else 3
    fig_h = 4.6 if is_rgbd else 4.0
    fig, ax = plt.subplots(figsize=(12.5, fig_h))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFBFC")
    ax.set_facecolor("#FAFBFC")

    subtitle = "RGB + depth → 4-channel conv1" if is_rgbd else "3-channel overhead RGB"
    ax.text(
        0.35,
        4.35,
        f"CalorieRegressor · {mode.upper()}",
        fontsize=15,
        fontweight="bold",
        color="#1A237E",
        ha="left",
    )
    ax.text(0.35, 4.02, subtitle, fontsize=10, color="#546E7A", ha="left")

    y_main = 2.05
    h_box = 1.15

    # --- Input column ---
    if is_rgbd:
        _arch_box(ax, (0.35, y_main + 0.55), 1.05, 0.55, "RGB\n3×224²", "#E3F2FD")
        _arch_box(ax, (0.35, y_main - 0.15), 1.05, 0.55, "Depth\n1×224²", "#E0F7FA")
        _arch_arrow(ax, 1.45, y_main + 0.82, 1.85, y_main + 0.95)
        _arch_arrow(ax, 1.45, y_main + 0.12, 1.85, y_main + 0.35)
        _arch_box(ax, (1.85, y_main + 0.35), 1.35, 0.95, "Concat\n4×224²", "#FFF8E1", fontsize=9)
        x_after_in = 3.35
    else:
        _arch_box(ax, (0.45, y_main + 0.15), 1.25, 0.95, f"Input\n{in_ch}×224×224", "#E3F2FD")
        x_after_in = 1.95

    # --- Backbone ---
    _arch_box(ax, (x_after_in, y_main + 0.05), 2.35, 1.05, "ResNet-18 backbone\nconv1 → layer4", "#90CAF9", fontsize=9.5)
    if is_rgbd:
        _arch_arrow(ax, 3.2, y_main + 0.58, x_after_in, y_main + 0.58)
    else:
        _arch_arrow(ax, 0.45 + 1.25, y_main + 0.58, x_after_in, y_main + 0.58)

    x_pool = x_after_in + 2.55
    _arch_box(ax, (x_pool, y_main + 0.2), 1.45, 0.75, "Global avg pool\n512-d features", "#64B5F6", fontsize=9)
    _arch_arrow(ax, x_after_in + 2.35, y_main + 0.58, x_pool, y_main + 0.58)

    fork_x = x_pool + 1.55
    fork_y = y_main + 0.58

    # --- Regression head (main path) ---
    x_reg = fork_x + 0.35
    _arch_box(ax, (x_reg, y_main + 0.35), 1.65, 0.85, "Regression head\n512 → 128 → 1", "#A5D6A7", fontsize=9)
    _arch_arrow(ax, fork_x, fork_y, x_reg, y_main + 0.78)

    x_out = x_reg + 1.95
    _arch_box(ax, (x_out, y_main + 0.42), 1.35, 0.7, "kcal\n(scalar)", "#C8E6C9", edge="#2E7D32", fontsize=10)
    _arch_arrow(ax, x_reg + 1.65, y_main + 0.78, x_out, y_main + 0.78)

    # --- Optional classifier branch ---
    y_cls = 0.55
    _arch_box(ax, (x_reg, y_cls), 1.65, 0.85, "Classifier head\n512 → 256 → 101", "#FFE082", fontsize=9)
    _arch_arrow(ax, fork_x, fork_y - 0.15, x_reg + 0.2, y_cls + 0.85, style="--")
    _arch_box(ax, (x_out, y_cls + 0.12), 1.35, 0.7, "Food-101\nlogits", "#FFF9C4", edge="#F9A825", fontsize=9)
    _arch_arrow(ax, x_reg + 1.65, y_cls + 0.42, x_out, y_cls + 0.48, style="--")

    # Legend strip
    ax.add_patch(
        FancyBboxPatch(
            (0.35, 0.08),
            11.5,
            0.38,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor="#ECEFF1",
            edgecolor="#CFD8DC",
            linewidth=0.8,
            zorder=0,
        )
    )
    ax.text(
        0.55,
        0.27,
        "Solid arrows: forward pass for calorie regression  ·  Dashed: optional auxiliary Food-101 (scheduled epochs)",
        fontsize=8.5,
        color="#546E7A",
        va="center",
    )

    fig.savefig(out_dir / f"model_architecture_{mode}.png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# --- EDA ---


def _plot_calorie_eda(splits: Dict[str, List[DishSample]], out_dir: Path) -> None:
    df = _samples_to_df(splits)
    if df.empty:
        return

    order = ["train", "val", "test"]
    colors = [_C["blue"], _C["orange"], _C["green"]]

    # 1) Hist + counts (original)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split, g in df.groupby("split"):
        axes[0].hist(g["calories"], bins=40, alpha=0.55, label=split, density=True)
    axes[0].set_xlabel("Total calories (label)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Calorie distribution by split")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    counts = df.groupby("split").size().reindex(order).fillna(0).astype(int)
    axes[1].bar(counts.index.astype(str), counts.values, color=colors)
    axes[1].set_ylabel("Dish count")
    axes[1].set_title("Samples per split (local RGB available)")
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + max(counts.values, default=1) * 0.02, str(int(v)), ha="center", fontsize=10)
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "eda_calories_and_counts.png", dpi=180, facecolor="white")
    plt.close(fig)

    # 2) Summary table
    summary = (
        df.groupby("split")["calories"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    summary.to_csv(out_dir / "eda_calories_summary.csv", index=False)
    _fig_table(summary.round(2), "Calorie stats by split", out_dir / "eda_calories_summary_table.png")

    # 3) Boxplot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [df.loc[df["split"] == s, "calories"].values for s in order if s in df["split"].unique()]
    labels = [s for s in order if s in df["split"].unique()]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors[: len(labels)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    ax.set_ylabel("Calories (kcal)")
    ax.set_title("Calorie spread by split (boxplot)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "eda_calories_boxplot.png", dpi=180, facecolor="white")
    plt.close(fig)

    # 4) Log-scale histogram (all splits)
    fig, ax = plt.subplots(figsize=(7, 4))
    cal = df["calories"].clip(lower=1.0)
    ax.hist(np.log1p(cal), bins=45, color=_C["purple"], alpha=0.85, edgecolor="white")
    ax.set_xlabel("log(1 + calories)")
    ax.set_ylabel("Count")
    ax.set_title("Overall calorie distribution (log1p scale)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "eda_calories_log1p_hist.png", dpi=180, facecolor="white")
    plt.close(fig)

    # 5) Calorie bins
    bins = [0, 100, 200, 300, 400, 500, 700, 1000, 2000]
    labels_bin = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    df["cal_bin"] = pd.cut(df["calories"], bins=bins, labels=labels_bin, include_lowest=True)
    bin_counts = df.groupby("cal_bin", observed=False).size()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(bin_counts)), bin_counts.values, color=_C["blue"], alpha=0.85)
    ax.set_xticks(range(len(bin_counts)))
    ax.set_xticklabels(bin_counts.index.astype(str), rotation=35, ha="right")
    ax.set_ylabel("Dish count")
    ax.set_title("Dishes per calorie range (all splits)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "eda_calorie_bins.png", dpi=180, facecolor="white")
    plt.close(fig)

    # 6) Depth availability
    depth_by_split = df.groupby("split")["has_depth"].agg(["sum", "count"])
    depth_by_split["pct"] = 100.0 * depth_by_split["sum"] / depth_by_split["count"]
    depth_by_split.to_csv(out_dir / "eda_depth_availability.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    splits_present = [s for s in order if s in depth_by_split.index]
    pcts = [depth_by_split.loc[s, "pct"] for s in splits_present]
    ax.bar(splits_present, pcts, color=colors[: len(splits_present)])
    ax.set_ylim(0, 105)
    ax.set_ylabel("% dishes with local depth file")
    ax.set_title("Depth map availability (local download)")
    for i, (s, p) in enumerate(zip(splits_present, pcts)):
        n = int(depth_by_split.loc[s, "sum"])
        tot = int(depth_by_split.loc[s, "count"])
        ax.text(i, p + 2, f"{n}/{tot}", ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "eda_depth_availability.png", dpi=180, facecolor="white")
    plt.close(fig)

    # 7) CDF
    fig, ax = plt.subplots(figsize=(7, 4))
    for split, c in zip(order, colors):
        sub = df[df["split"] == split]["calories"].sort_values()
        if sub.empty:
            continue
        y = np.arange(1, len(sub) + 1) / len(sub)
        ax.plot(sub.values, y, label=split, color=c, linewidth=2)
    ax.set_xlabel("Calories (kcal)")
    ax.set_ylabel("CDF")
    ax.set_title("Cumulative distribution of calories by split")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "eda_calories_cdf.png", dpi=180, facecolor="white")
    plt.close(fig)


def _plot_sample_dishes(splits: Dict[str, List[DishSample]], out_dir: Path, n: int = 9) -> None:
    all_samples: List[DishSample] = []
    for key in ("train", "val", "test"):
        all_samples.extend(splits.get(key, []))
    if not all_samples:
        return
    rng = np.random.default_rng(42)
    idx = rng.choice(len(all_samples), size=min(n, len(all_samples)), replace=False)
    picked = [all_samples[i] for i in idx]

    cols = 3
    rows = int(np.ceil(len(picked) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.2 * rows))
    axes_flat = np.atleast_1d(axes).flatten()
    for ax, s in zip(axes_flat, picked):
        try:
            img = Image.open(s.rgb_path).convert("RGB")
        except (OSError, UnidentifiedImageError):
            ax.axis("off")
            continue
        ax.imshow(img)
        depth_tag = " +depth" if s.depth_path else ""
        ax.set_title(f"{s.dish_id}\n{s.calories:.0f} kcal{depth_tag}", fontsize=8)
        ax.axis("off")
    for ax in axes_flat[len(picked) :]:
        ax.axis("off")
    fig.suptitle("Random overhead RGB samples (Nutrition5k)", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "eda_sample_dishes_grid.png", dpi=160, facecolor="white")
    plt.close(fig)


# --- Training curves ---


def _plot_training_curves(train_log: Path, out_dir: Path, prefix: str = "") -> None:
    if not train_log.exists():
        return
    df = pd.read_csv(train_log)
    if df.empty:
        return
    pfx = f"{prefix}_" if prefix else ""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(df["epoch"], df["train_loss"], label="train", color=_C["blue"], linewidth=2)
    axes[0].plot(df["epoch"], df["val_loss"], label="val", color=_C["orange"], linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend(loc="upper right", framealpha=0.95)
    axes[0].grid(True, alpha=0.25)

    ax_mae = axes[1]
    ax_mae.plot(df["epoch"], df["train_mae"], label="train MAE", color=_C["blue"], linewidth=2)
    ax_mae.plot(df["epoch"], df["val_mae"], label="val MAE", color=_C["orange"], linewidth=2)
    best_label = None
    if "val_mae" in df.columns and df["val_mae"].notna().any():
        best_i = int(df["val_mae"].idxmin())
        best_ep = int(df.loc[best_i, "epoch"])
        best_mae = float(df.loc[best_i, "val_mae"])
        ax_mae.axvline(best_ep, color=_C["red"], ls=":", alpha=0.75, linewidth=1.5)
        ax_mae.scatter([best_ep], [best_mae], color=_C["red"], zorder=5, s=50, edgecolors="white", linewidths=0.8)
        best_label = f"best val @ ep {best_ep} ({best_mae:.1f} kcal)"
    ax_mae.set_xlabel("Epoch")
    ax_mae.set_ylabel("MAE (kcal)")
    ax_mae.set_title("Mean absolute error")
    ax_mae.grid(True, alpha=0.25)

    # Food-101 accuracy lives in training_food101_accuracy.png — keep MAE panel single-axis to avoid legend clash.
    handles, labels = ax_mae.get_legend_handles_labels()
    if best_label:
        handles.append(plt.Line2D([0], [0], color=_C["red"], ls=":", linewidth=1.5))
        labels.append(best_label)
    ax_mae.legend(handles, labels, loc="lower right", framealpha=0.95, fontsize=9)

    fig.suptitle("Training curves", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88, wspace=0.28)
    fig.savefig(out_dir / f"{pfx}training_curves.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if "lr" in df.columns and df["lr"].notna().any():
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(df["epoch"], df["lr"], color=_C["purple"], marker="o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_title("Learning rate schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"{pfx}training_lr.png", dpi=180, facecolor="white")
        plt.close(fig)

    fcols = [c for c in df.columns if c.startswith("food101_") and "acc" in c]
    if fcols and df[fcols].notna().any().any():
        fig, ax = plt.subplots(figsize=(7, 4))
        for c in fcols:
            sub = df[["epoch", c]].dropna()
            if not sub.empty:
                ax.plot(sub["epoch"], sub[c], marker="o", markersize=4, label=c.replace("food101_", ""))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Food-101 auxiliary classification accuracy")
        ax.legend()
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0, 1.02)
        fig.tight_layout()
        fig.savefig(out_dir / f"{pfx}training_food101_accuracy.png", dpi=180, facecolor="white")
        plt.close(fig)


# --- Metrics tables ---


def _collect_run_metrics(run_dir: Path, label: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {"run": label}
    cfg = _read_json(run_dir / "logs" / "config.json")
    summ = _read_json(run_dir / "logs" / "train_summary.json")
    ev = _read_json(run_dir / "logs" / "eval_metrics.json")
    if cfg:
        row.update(
            {
                "mode": cfg.get("mode"),
                "split_type": cfg.get("split_type"),
                "loss": cfg.get("loss_type"),
                "log1p": cfg.get("use_log_target"),
                "epochs": cfg.get("epochs"),
            }
        )
    if summ:
        row["best_val_mae"] = summ.get("best_val_mae")
        row["best_epoch"] = summ.get("best_epoch")
    if ev:
        row["test_mae"] = ev.get("mae")
        row["test_rmse"] = ev.get("rmse")
    return row


def _metrics_table(primary_run: Path, out_dir: Path, all_runs: Sequence[Tuple[str, Path]]) -> None:
    rows = [_collect_run_metrics(rd, name) for name, rd in all_runs]
    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics_all_runs.csv", index=False)

    display_cols = [c for c in ["run", "mode", "split_type", "best_val_mae", "best_epoch", "test_mae", "test_rmse"] if c in df.columns]
    _fig_table(df[display_cols].round(3), "All training runs — key metrics", out_dir / "metrics_all_runs_table.png")

    # Bar chart: test MAE / RMSE per run
    if "test_mae" in df.columns and df["test_mae"].notna().any():
        fig, ax = plt.subplots(figsize=(max(6, 2 * len(df)), 4))
        x = np.arange(len(df))
        w = 0.35
        mae = df["test_mae"].fillna(0).values
        rmse = df["test_rmse"].fillna(0).values if "test_rmse" in df.columns else np.zeros(len(df))
        ax.bar(x - w / 2, mae, w, label="Test MAE", color=_C["blue"])
        ax.bar(x + w / 2, rmse, w, label="Test RMSE", color=_C["orange"])
        ax.set_xticks(x)
        ax.set_xticklabels(df["run"], rotation=15, ha="right")
        ax.set_ylabel("kcal")
        ax.set_title("Held-out test error by run")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.25)
        for i, v in enumerate(mae):
            if v > 0:
                ax.text(i - w / 2, v + 3, f"{v:.1f}", ha="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "metrics_test_mae_rmse_bars.png", dpi=180, facecolor="white")
        plt.close(fig)

    # Primary run detail tables (legacy names)
    primary = primary_run
    cfg = _read_json(primary / "logs" / "config.json")
    summ = _read_json(primary / "logs" / "train_summary.json")
    ev = _read_json(primary / "logs" / "eval_metrics.json")

    flat: Dict[str, Any] = {}
    if cfg:
        flat.update(
            {
                "run_mode": cfg.get("mode"),
                "split_type": cfg.get("split_type"),
                "loss_type": cfg.get("loss_type"),
                "use_log_target": cfg.get("use_log_target"),
                "epochs_configured": cfg.get("epochs"),
                "food101_cls": cfg.get("enable_food101_cls"),
            }
        )
    if summ:
        flat["best_val_mae_kcal"] = summ.get("best_val_mae")
        flat["best_epoch"] = summ.get("best_epoch")
    if ev:
        flat["test_mae_kcal"] = ev.get("mae")
        flat["test_rmse_kcal"] = ev.get("rmse")
    if flat:
        summary_df = pd.DataFrame([{k: flat[k] for k in sorted(flat.keys())}])
        summary_df.to_csv(out_dir / "metrics_one_row_summary.csv", index=False)
        _fig_table(summary_df.round(4), "Primary run summary (for slides)", out_dir / "metrics_one_row_table.png")


def _snapshot_run_logs(run_dir: Path, assets_dir: Path, prefix: str = "primary") -> None:
    logs = run_dir / "logs"
    mapping = [
        (logs / "eval_metrics.json", assets_dir / f"{prefix}_eval_metrics.json"),
        (logs / "train_summary.json", assets_dir / f"{prefix}_train_summary.json"),
        (logs / "config.json", assets_dir / f"{prefix}_config.json"),
    ]
    for src, dst in mapping:
        if src.is_file():
            shutil.copy2(src, dst)


# --- Predictions / error analysis ---


def _plot_predictions(pred_csv: Path, out_dir: Path, tag: str = "primary") -> None:
    if not pred_csv.exists():
        return
    df = pd.read_csv(pred_csv)
    if df.empty or not {"predicted_calories", "target_calories"}.issubset(df.columns):
        return

    y = df["target_calories"].to_numpy(dtype=np.float64)
    p = df["predicted_calories"].to_numpy(dtype=np.float64)
    e = p - y
    ae = np.abs(e)
    if "abs_error" not in df.columns:
        df["abs_error"] = ae
    df.to_csv(out_dir / f"{tag}_predictions_with_errors.csv", index=False)

    mae = float(np.mean(ae))
    rmse = float(np.sqrt(np.mean(e**2)))
    med_ae = float(np.median(ae))
    p90 = float(np.percentile(ae, 90))

    # Scatter + residual
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
    axes[0].scatter(y, p, s=12, alpha=0.5, c=ae, cmap="YlOrRd", edgecolors="none")
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1, label="ideal")
    axes[0].set_xlabel("True calories (kcal)")
    axes[0].set_ylabel("Predicted calories (kcal)")
    axes[0].set_title(f"Prediction vs truth ({tag})")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.25)
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].hist(e, bins=40, color=_C["purple"], alpha=0.85, edgecolor="white")
    axes[1].axvline(0.0, color="k", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Error (pred − true), kcal")
    axes[1].set_title("Residual distribution")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(
        f"Test set | MAE={mae:.1f} | RMSE={rmse:.1f} | median |e|={med_ae:.1f} | p90={p90:.1f}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_test_scatter_residuals.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Abs error vs true
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y, ae, s=10, alpha=0.45, color=_C["blue"], edgecolors="none")
    ax.set_xlabel("True calories (kcal)")
    ax.set_ylabel("|Error| (kcal)")
    ax.set_title("Absolute error vs true calorie level")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_test_abs_error_vs_true.png", dpi=180, facecolor="white")
    plt.close(fig)

    # Error by calorie bin
    bins = [0, 100, 200, 300, 400, 600, 900, 2000]
    bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    bin_idx = pd.cut(y, bins=bins, labels=bin_labels, include_lowest=True)
    err_by_bin = pd.DataFrame({"bin": bin_idx, "ae": ae}).groupby("bin", observed=False)["ae"].agg(["mean", "count"])
    err_by_bin.to_csv(out_dir / f"{tag}_error_by_calorie_bin.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(err_by_bin))
    ax.bar(x, err_by_bin["mean"].values, color=_C["orange"], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(err_by_bin.index.astype(str), rotation=30, ha="right")
    ax.set_ylabel("Mean |error| (kcal)")
    ax.set_title("Mean absolute error by true-calorie bin")
    for i, (_, row) in enumerate(err_by_bin.iterrows()):
        ax.text(i, row["mean"] + 2, f"n={int(row['count'])}", ha="center", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_test_error_by_calorie_bin.png", dpi=180, facecolor="white")
    plt.close(fig)

    # Percentile curve
    sorted_ae = np.sort(ae)
    pct = 100.0 * (np.arange(1, len(sorted_ae) + 1) / len(sorted_ae))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pct, sorted_ae, color=_C["green"], linewidth=2)
    ax.set_xlabel("% of test dishes")
    ax.set_ylabel("|Error| (kcal)")
    ax.set_title("CDF of absolute error (test set)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_test_abs_error_cdf.png", dpi=180, facecolor="white")
    plt.close(fig)

    # Best / worst table
    show = df.sort_values("abs_error", ascending=False).head(8)[
        ["dish_id", "target_calories", "predicted_calories", "abs_error"]
    ].round(1)
    show.columns = ["dish_id", "true_kcal", "pred_kcal", "|error|"]
    _fig_table(show, f"Largest errors on test set ({tag})", out_dir / f"{tag}_test_worst_errors_table.png", fontsize=8)

    best = df.sort_values("abs_error", ascending=True).head(8)[
        ["dish_id", "target_calories", "predicted_calories", "abs_error"]
    ].round(1)
    best.columns = ["dish_id", "true_kcal", "pred_kcal", "|error|"]
    _fig_table(best, f"Smallest errors on test set ({tag})", out_dir / f"{tag}_test_best_errors_table.png", fontsize=8)


def _write_manifest(out_dir: Path, written: List[str], meta: Dict[str, Any]) -> None:
    lines = [
        "# Slide assets (copy into PowerPoint)",
        "",
        "Generated by: `python scripts/generate_presentation_assets.py`",
        "",
        "## Files",
        "",
    ]
    groups = {
        "Pipeline & model": ["pipeline_", "model_architecture_"],
        "EDA": ["eda_"],
        "Training": ["training_", "_training_"],
        "Metrics": ["metrics_"],
        "Test / predictions": ["test_", "_test_", "predictions"],
        "JSON snapshots": [".json"],
    }
    used = set()
    for title, prefixes in groups.items():
        lines.append(f"### {title}")
        for name in sorted(written):
            if name in used:
                continue
            if any(name.startswith(p) or p in name for p in prefixes):
                lines.append(f"- `{name}`")
                used.add(name)
        lines.append("")
    for name in sorted(written):
        if name not in used:
            lines.append(f"- `{name}`")
    lines.append("")
    lines.append("## Run metadata")
    lines.append("```json")
    lines.append(json.dumps(meta, indent=2))
    lines.append("```")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


# (source under slide_assets, dest filename, one-line note for README)
CURATED_PICKS: List[Tuple[str, str, str]] = [
    ("eda_calories_and_counts.png", "01_data_split_and_calorie_distribution.png", "Split counts + calorie density (subset on disk)"),
    ("eda_calories_summary_table.png", "02_data_calorie_stats_table.png", "Calorie stats table per split"),
    ("eda_depth_availability.png", "03_data_depth_coverage.png", "Local depth file coverage (RGB-D)"),
    ("eda_sample_dishes_grid.png", "04_data_sample_dishes.png", "Sample overhead RGB dishes + labels"),
    ("pipeline_end_to_end.png", "05_pipeline_end_to_end.png", "End-to-end pipeline diagram"),
    ("model_architecture_rgb.png", "06_model_architecture_RGB.png", "Baseline architecture (RGB)"),
    ("model_architecture_rgbd.png", "07_model_architecture_RGBD.png", "Baseline architecture (RGB-D)"),
    ("metrics_all_runs_table.png", "08_metrics_all_models_table.png", "Val/test MAE, best epoch — all runs"),
    ("metrics_test_mae_rmse_bars.png", "09_metrics_test_mae_rmse_bars.png", "Test MAE / RMSE bar chart"),
    ("metrics_all_runs.csv", "08_metrics_all_models.csv", "Metrics table (CSV)"),
    ("training_curves.png", "10_train_curves_RGBD.png", "Primary run (RGB-D): loss + val MAE"),
    ("outputs_food101_4passes_training_curves.png", "11_train_curves_RGB.png", "Comparison run (RGB): loss + val MAE"),
    ("training_food101_accuracy.png", "12_train_food101_aux_accuracy.png", "Food-101 auxiliary accuracy"),
    ("primary_test_scatter_residuals.png", "20_test_scatter_RGBD.png", "RGB-D test: pred vs truth + residuals"),
    ("outputs_food101_4passes_test_scatter_residuals.png", "21_test_scatter_RGB.png", "RGB test: pred vs truth + residuals"),
    ("primary_test_error_by_calorie_bin.png", "22_test_error_by_calorie_bin_RGBD.png", "RGB-D: mean |error| by calorie bin"),
    ("outputs_food101_4passes_test_error_by_calorie_bin.png", "23_test_error_by_calorie_bin_RGB.png", "RGB: mean |error| by calorie bin"),
    ("primary_test_abs_error_cdf.png", "24_test_error_cdf_RGBD.png", "RGB-D: CDF of absolute error"),
    ("primary_test_worst_errors_table.png", "25_test_worst_errors_RGBD.png", "RGB-D: largest test errors"),
]


def _build_curated_picks(slide_assets_dir: Path, picks_dir: Path) -> List[str]:
    """Copy a small, numbered set of figures for quick PPT selection."""
    picks_dir.mkdir(parents=True, exist_ok=True)
    # Clear old picks so removed items do not linger
    for old in picks_dir.iterdir():
        if old.is_file():
            old.unlink()

    copied: List[str] = []
    readme_rows: List[str] = [
        "# Curated slide figures",
        "",
        "Copied from `slide_assets/` for PowerPoint. Numbered in recommended presentation order.",
        "Regenerate with `scripts/generate_presentation_assets.py`.",
        "Pipeline context: [docs/RESEARCH_PIPELINE.md](../docs/RESEARCH_PIPELINE.md).",
        "",
        "| File | Description |",
        "|------|-------------|",
    ]
    for src_name, dst_name, note in CURATED_PICKS:
        src = slide_assets_dir / src_name
        if not src.is_file():
            continue
        dst = picks_dir / dst_name
        shutil.copy2(src, dst)
        copied.append(dst_name)
        readme_rows.append(f"| `{dst_name}` | {note} |")

    readme_rows.extend(
        [
            "",
            "## Run mapping",
            "",
            "- **RGB-D (primary):** `outputs_train_rgbd_food101` — files `10_*`, `20_*`–`25_*`",
            "- **RGB (comparison):** `outputs_food101_4passes` — files `11_*`, `21_*`–`23_*`",
            "",
            "Full asset library: `../slide_assets/`.",
        ]
    )
    (picks_dir / "README.md").write_text("\n".join(readme_rows), encoding="utf-8")
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PPT slide figures from existing logs (no training).")
    parser.add_argument("--dataset_root", type=str, required=True, help="Nutrition5k root (for EDA splits).")
    parser.add_argument("--run_dir", type=str, required=True, help="Primary training run (logs/).")
    parser.add_argument(
        "--extra_run_dirs",
        type=str,
        nargs="*",
        default=[],
        help="Additional runs for comparison tables (name=path or just path).",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_type", type=str, default="auto", choices=["auto", "rgb", "depth"])
    parser.add_argument("--predictions_csv", type=str, default="")
    parser.add_argument(
        "--assets_dir",
        type=str,
        default="",
        help="Output folder (default: presentation/slide_assets/).",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser()
    if not dataset_root.is_dir():
        raise SystemExit(f"--dataset_root must exist: {dataset_root}")

    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = (_REPO_ROOT / run_dir).resolve()
    else:
        run_dir = run_dir.resolve()
    if not (run_dir / "logs").is_dir() and not run_dir.exists():
        raise SystemExit(f"--run_dir not found: {run_dir}")

    if args.assets_dir:
        out_dir = Path(args.assets_dir).expanduser().resolve()
    else:
        out_dir = (_REPO_ROOT / "presentation" / "slide_assets").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse extra runs: "label=path" or "path"
    all_runs: List[Tuple[str, Path]] = [("primary", run_dir)]
    for spec in args.extra_run_dirs:
        if "=" in spec:
            name, path_s = spec.split("=", 1)
        else:
            path_s = spec
            name = Path(path_s).name
        p = Path(path_s).expanduser()
        if not p.is_absolute():
            p = (_REPO_ROOT / p).resolve()
        else:
            p = p.resolve()
        if p.is_dir():
            all_runs.append((name, p))

    splits = build_split_samples(
        dataset_root=str(dataset_root),
        val_ratio=args.val_ratio,
        seed=args.seed,
        split_type=args.split_type,
    )

    _plot_pipeline_diagram(out_dir)
    _plot_model_architecture(out_dir, "rgb")
    _plot_model_architecture(out_dir, "rgbd")
    _plot_calorie_eda(splits, out_dir)
    _plot_sample_dishes(splits, out_dir)

    for name, rd in all_runs:
        prefix = "" if name == "primary" else name
        _plot_training_curves(rd / "logs" / "train_log.csv", out_dir, prefix=prefix)
        pred = rd / "logs" / "test_predictions.csv"
        if pred.exists():
            _plot_predictions(pred, out_dir, tag=name if name != "primary" else "primary")

    _metrics_table(run_dir, out_dir, all_runs)

    pred_path = Path(args.predictions_csv) if args.predictions_csv else run_dir / "logs" / "test_predictions.csv"
    if pred_path.exists() and ("primary", run_dir) in all_runs:
        _plot_predictions(pred_path, out_dir, tag="primary")

    for name, rd in all_runs:
        _snapshot_run_logs(rd, out_dir, prefix=name)

    # Copy pipeline markdown for reference
    pipeline_md = _REPO_ROOT / "presentation" / "MODEL_PIPELINE.md"
    if pipeline_md.is_file():
        shutil.copy2(pipeline_md, out_dir / "MODEL_PIPELINE.md")

    written = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    meta = {
        "dataset_root": str(dataset_root),
        "primary_run": str(run_dir),
        "extra_runs": [str(p) for _, p in all_runs[1:]],
        "split_type": args.split_type,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
    }
    (out_dir / "index.json").write_text(
        json.dumps({"assets_dir": str(out_dir), "written": written, **meta}, indent=2),
        encoding="utf-8",
    )
    _write_manifest(out_dir, written, meta)

    picks_dir = out_dir.parent / "slide_picks"
    picked = _build_curated_picks(out_dir, picks_dir)

    print(f"Wrote {len(written)} files to:\n  {out_dir}")
    print(f"Curated {len(picked)} picks for PPT:\n  {picks_dir}")
    print("See README.md in each folder.")


if __name__ == "__main__":
    main()
