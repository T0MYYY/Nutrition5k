---
title: Food Calorie App
emoji: 🍱
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Nutrition5k Calorie Prediction — Repository Manual

End-to-end documentation for this codebase: **data layout**, **configuration**, **model**, **training**, **evaluation**, **Gradio demo**, **Hugging Face Spaces**, **helper scripts**, and **presentation exports**. Regression target is **`total_calories`** per dish from Nutrition5k metadata; inputs are **overhead RGB** or **RGB + depth** tensors.

**One-line pipeline:** `overhead image → ResNet-18 → MLP head → calories (kcal)` (+ optional Food-101 `classify` head).

**Research pipeline:** **Stage 0** — download official metadata/splits + a **storage-limited overhead subset** → **Stage 1** — ResNet-18 **baseline** (this repo) with fixed official **test** MAE/RMSE → **Stage 2** — future stronger models on the **same test set**. The README is the main manual; the consolidated data-scope, download, and research-pipeline notes live in [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md).

**Live Space:** [https://austinwang10-food-calorie-app.hf.space/](https://austinwang10-food-calorie-app.hf.space/)

---

## Table of contents

1. [Quick start](#1-quick-start)
2. [Repository layout](#2-repository-layout)
3. [Dependencies (`requirements.txt`)](#3-dependencies-requirementstxt)
4. [Nutrition5k on-disk layout](#4-nutrition5k-on-disk-layout)
   - [Dataset scope (subset justification)](#dataset-scope-subset-justification)
5. [Data loading and splits (`data_loader.py`)](#5-data-loading-and-splits-data_loaderpy)
6. [Model (`model.py`)](#6-model-modelpy)
7. [Configuration CLI (`config.py`)](#7-configuration-cli-configpy)
8. [Training (`train.py`)](#8-training-trainpy)
9. [Nutrition5k test evaluation (`evaluate.py`)](#9-nutrition5k-test-evaluation-evaluatepy)
10. [Food-101 test accuracy (`evaluate_food101.py`)](#10-food-101-test-accuracy-evaluate_food101py)
11. [Utilities (`utils.py`)](#11-utilities-utilspy)
12. [Scripts under `scripts/`](#12-scripts-under-scripts)
13. [Gradio web app (`web_app.py`)](#13-gradio-web-app-web_apppy)
14. [Hugging Face entry (`app.py`)](#14-hugging-face-entry-apppy)
15. [Run artifacts: directories and files](#15-run-artifacts-directories-and-files)
16. [Presentation bundle (`presentation/`)](#16-presentation-bundle-presentation)
17. [Consolidated project guide (`docs/PROJECT_GUIDE.md`)](#17-consolidated-project-guide-docsproject_guidemd)
18. [Benchmark alignment vs approximations](#18-benchmark-alignment-vs-approximations)
19. [Troubleshooting](#19-troubleshooting)
20. [Further reading](#20-further-reading)

---

## 1. Quick start

**Prerequisites:** Install [Google Cloud SDK](https://cloud.google.com/storage/docs/gsutil_install) (`gsutil`) and download **as much overhead data as your disk allows**. Metadata and split files are small; overhead imagery is the storage-heavy part.

```bash
# Example: partial download (legacy cap) or use download_nutrition5k.py --only_missing
python scripts/download_nutrition5k.py \
  --dataset_root ~/data/nutrition5k_mini \
  --tier essentials

python scripts/download_more_overhead.py \
  --dataset_root ~/data/nutrition5k_mini \
  --target_total 3000
```

Then train / evaluate (use **your** `--dataset_root` path):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train.py \
  --dataset_root ~/data/nutrition5k_mini \
  --mode rgb --split_type rgb --epochs 40 \
  --output_dir outputs_rgb \
  --loss_type smooth_l1 --scheduler plateau \
  --use_log_target --pretrained

python evaluate.py \
  --dataset_root ~/data/nutrition5k_mini \
  --mode rgb --split_type rgb \
  --checkpoint_path outputs_rgb/checkpoints/best.pt \
  --output_dir outputs_rgb \
  --save_predictions_csv outputs_rgb/logs/test_predictions.csv

python scripts/generate_presentation_assets.py \
  --dataset_root ~/data/nutrition5k_mini \
  --run_dir outputs_rgb \
  --split_type rgb --val_ratio 0.1 --seed 42
```

Slide figures land in **`presentation/slide_assets/`** (full set) and **`presentation/slide_picks/`** (curated). The current browser-ready presentation is **`presentation/PRESENTATION.html`**.

---

## 2. Repository layout

| File | Responsibility |
|------|----------------|
| `config.py` | `Config` dataclass; `build_arg_parser(train=…)`; `config_from_args`; Food-101 schedule helpers `food101_scheduled_epoch_list`, `compute_food101_epoch_interval`, `food101_schedule_explain`. |
| `data_loader.py` | CSV calorie maps; split file discovery; `build_split_samples`; `Nutrition5kCalorieDataset`; `create_dataloaders`; Food-101 loaders. |
| `model.py` | `CalorieRegressor`: torchvision ResNet-18, regression head, optional classifier, RGB-D first conv adaptation. |
| `train.py` | Training / validation loops; optional Food-101 epochs; checkpoints; `train_log.csv`; `train_summary.json`. |
| `evaluate.py` | Test-set forward pass; MAE, RMSE, MSE; optional per-dish CSV. |
| `evaluate_food101.py` | Food-101 **official test** top-1 / top-k (requires classifier in checkpoint). |
| `utils.py` | Seeds, device, MAE/RMSE, meters, checkpoint paths, CSV logging, predictions CSV, macOS caffeinate helper. |
| `web_app.py` | Gradio UI, dual checkpoints, MiDaS / heuristic / upload depth, top-K class text. |
| `app.py` | HF Spaces: optional checkpoint URL download, then `web_app.main()`. |
| `scripts/download_nutrition5k.py` | **Recommended** — essentials / overhead / full tier downloads from GCS. |
| `scripts/download_more_overhead.py` | Legacy incremental overhead download (e.g. `--target_total 1000`). |
| `scripts/check_overhead_integrity.py` | Counts complete/partial overhead folders by fixed filenames. |
| `scripts/generate_presentation_assets.py` | Figures + tables → `presentation/slide_assets/` and `slide_picks/`. |
| `docs/PROJECT_GUIDE.md` | Consolidated data-scope, download, and research-pipeline notes preserved from the old docs. |
| `presentation/PRESENTATION.md` | Editable source presentation. |
| `presentation/PRESENTATION.html` | Browser-ready presentation with embedded plots. |
| `presentation/deck_style.css` | CSS for the browser-ready deck. |

There is **no** `scripts/run_full_pipeline.sh` in the repo; use **`download_nutrition5k.py`** (§12.1) then `train.py` / `evaluate.py`.

---

## 3. Dependencies (`requirements.txt`)

| Package | Used for |
|---------|-----------|
| `torch`, `torchvision` | Model, dataloaders, `resnet18`, pretrained weights, Food-101 dataset. |
| `tqdm` | Progress bars in train/eval. |
| `Pillow` | Image I/O; `ImageOps.exif_transpose` in web app. |
| `numpy` | Depth array ops (`data_loader`, `web_app` preview). |
| `pandas`, `matplotlib` | `generate_presentation_assets.py` only. |
| `gradio` | `web_app.py`. |
| `timm` | Declared dependency; **core model uses torchvision ResNet-18 only** (`model.py`). |
| `opencv-python-headless` | Declared; training path is PIL-based. |

---

## 4. Nutrition5k on-disk layout

Expected under `--dataset_root` (e.g. `nutrition5k_mini` on a laptop, or `nutrition5k_dataset` when disk allows):

```text
metadata/
  dish_metadata_cafe1.csv
  dish_metadata_cafe2.csv
  ingredient_metadata.csv    # unused by train/eval in this repo
imagery/realsense_overhead/
  dish_xxxxxxxxxx/
    <rgb image>              # filename matched by keywords
    <depth image>            # optional in rgb mode; used in rgbd
dish_ids/splits/             # required for official benchmark splits
  rgb_train_ids.txt / rgb_test_ids.txt
  depth_train_ids.txt / depth_test_ids.txt
  …
```

**Labels:** `_read_calories_map` scans both cafe metadata files. Each CSV row: column 0 = dish id string, column 1 = **`total_calories`** (float). Header row `dish_id` is skipped. Invalid floats skipped.

**Download:** on limited storage, download **metadata + splits first**, then add overhead dishes incrementally (`download_more_overhead.py --target_total N` or `download_nutrition5k.py --only_missing`).

### Dataset scope (summary)

We use **official Nutrition5k labels and splits** on a **local overhead subset** (`official IDs ∩ downloaded RGB`). This supports a controlled baseline under laptop storage limits. All compared models should use the **same fixed local test set**, same seed/split configuration, and same MAE/RMSE metrics. Report actual train/val/test counts from `train.py` or EDA CSV in slides, not paper full-data leaderboard numbers.

For the full reviewer-facing justification, download commands, and staged research logic, see **[`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md)**.

---

## 5. Data loading and splits (`data_loader.py`)

### 5.1 Split file selection (`_locate_split_files`)

- Recursive search under `dish_ids/splits` for `*.csv` and `*.txt`.
- **`split_type rgb`:** prefers files named `rgb_train_ids.txt` and `rgb_test_ids.txt` (case-insensitive).
- **`split_type depth`:** prefers `depth_train_ids.txt` / `depth_test_ids.txt`.
- **`split_type auto`:** if `rgb_train_ids.txt` and `rgb_test_ids.txt` exist, use them; else first file with `train` in name and first with `test` in name.

### 5.2 `build_split_samples`

1. Load calories map; read train/test IDs from files (or fallback §5.3).
2. **`has_local_rgb`:** train/val pool only includes IDs that have a resolvable RGB file under `imagery/realsense_overhead/<id>/` (`_find_rgb_depth_paths`). On a partial download, official train IDs without local folders are **skipped** (subset training).
3. Shuffle available train IDs with `random.Random(seed)`; validation size = `max(1, int(len(available_train_ids) * val_ratio))`; remainder = train. **Test** list is built from official test IDs independently.
4. **`make_samples`:** skips IDs with no RGB file; depth optional; `_find_rgb_depth_paths` matches keywords (`rgb.png`, `rgb.jpg`, `color`, `depth_raw`, `raw_depth`, `depth`, …).

### 5.3 Fallback without `dish_ids/splits`

If the splits directory is missing, **all** IDs from the calorie map are shuffled with `seed` and split **80% train / 20% test** internally. This is **not** the official benchmark split—ship split files for publication numbers.

### 5.4 Tensor pipeline (`Nutrition5kCalorieDataset`)

- RGB: resize `image_size`, ToTensor, ImageNet normalize.
- Train augment (Nutrition5k only): 50% hflip on RGB+depth; `ColorJitter` on RGB.
- **RGB-D:** concat RGB (3×H×W) with depth channel (1×H×W). Missing/unreadable depth → zeros (+ one-time warning per bad path).
- **`depth_image_to_tensor`:** if max pixel > 255, divide by `max_depth_units` (default 4000); else `/255`. Per-image 1–99 percentile linear stretch to [0,1], clip.

### 5.5 `create_dataloaders`

Returns `(train_loader, val_loader, test_loader)` with `pin_memory=True`, train `shuffle=True`.

### 5.6 Food-101 loaders

- **`create_food101_dataloaders`:** Official `Food101(split="train")`; permuted indices; `food101_val_ratio` held out as val; train uses stronger augment when enabled.
- **`create_food101_test_dataloader`:** Official **`split="test"`** for `evaluate_food101.py`.

---

## 6. Model (`model.py`)

### 6.1 `CalorieRegressor.__init__`

- `mode`: `"rgb"` or `"rgbd"`.
- `pretrained`: if True, `resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`.
- `num_classes`: if > 0, builds `cls_head`; else `cls_head` is `None`.
- Replaces `backbone.fc` with `Identity` (512-D features).
- **Regression head:** `512→128` ReLU Dropout(0.2) `→1`.
- **Classifier head (optional):** `512→256` ReLU Dropout(0.2) `→num_classes`.

### 6.2 RGB-D stem

When `mode=="rgbd"`, `conv1` becomes `Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)`. Weights: channels 0–2 copy pretrained RGB; channel 3 = **mean of RGB filters** over input channels.

### 6.3 Methods

- **`forward`:** `reg_head(backbone(x))` — calorie branch only.
- **`classify`:** requires `cls_head`; `cls_head(backbone(x))`.
- **`has_classifier`:** property, True if `cls_head` is not None.

---

## 7. Configuration CLI (`config.py`)

### 7.1 Training-only arguments (`build_arg_parser(train=True)`)

Core: `--dataset_root` (required), `--output_dir`, `--image_size`, `--batch_size`, `--num_workers`, `--mode {rgb,rgbd}`, `--split_type {auto,rgb,depth}`, `--seed`, `--epochs`, `--lr`, `--weight_decay`, `--val_ratio`, `--max_depth_units`, `--device`.

**Pretrained ResNet:** `--pretrained` / `--no_pretrained` (default: **pretrained on** if neither flag).

**Target transform:** `--use_log_target` / `--no_log_target` (default: **log1p on** if neither flag).

**Loss / schedule / early stop:** `--loss_type {smooth_l1,mse}`, `--scheduler {none,plateau}`, `--scheduler_patience`, `--scheduler_factor`, `--early_stop_patience`, `--min_improve_delta`.

**Augmentation:** `--no_train_augment` disables Nutrition5k train flips/jitter.

**Per-group LR:** `--backbone_lr` and `--head_lr` must be **set together** (otherwise `ValueError`); else single `--lr` for all parameters.

**Food-101 multi-task:** `--enable_food101_cls`, `--food101_root`, `--food101_download`, `--cls_loss_weight`, `--cls_batch_size`, `--no_food101_augment`, `--food101_val_ratio`, `--cls_label_smoothing`, `--food101_cls_passes`, `--food101_every_n_epochs`.

**Food-101 schedule:** `food101_epoch_interval` is computed: if `food101_every_n_epochs > 0` → stride `max(1, that)`; else if `food101_cls_passes <= 0` → every epoch; else ~`P` evenly spaced epochs over `epochs` (see `compute_food101_epoch_interval`).

### 7.2 Evaluation-only arguments (`build_arg_parser(train=False)`)

- `--checkpoint_path` (required), `--save_predictions_csv` (optional path).

---

## 8. Training (`train.py`)

### 8.1 Startup

`config_from_args` → `ensure_paths` → `set_seed` → optional **`prevent_system_sleep_while_running`** (macOS `caffeinate -dims -w <pid>`) → dataloaders → optional Food-101 loaders + schedule print → model/optimizer/loss/scheduler → **`save_run_config`** → epoch loop.

### 8.2 `run_epoch` (Nutrition5k)

- Forward `preds = model(images)`.
- Loss target: `log1p(targets)` if `use_log_target`, else raw calories; criterion SmoothL1 or MSE.
- Reported MAE in **kcal**: `expm1(clamp(pred, max=12))` then clamp ≥0, compare to `targets`.

### 8.3 Food-101 epochs

When `enable_food101_cls` and `(epoch - 1) % food101_epoch_interval == 0`:

- `run_cls_epoch` train (with optional label smoothing) and val (no smoothing).
- `CrossEntropy * cls_loss_weight`; accuracy = mean correct.

**Epoch 1 always matches** `(epoch-1) % interval == 0`, so Food-101 runs on the first epoch whenever enabled and a classifier exists.

### 8.4 Logging row (`train_log.csv`)

Columns always include `timestamp`, `epoch`, `train_loss`, `train_mae`, `val_loss`, `val_mae`, `lr`. When Food-101 is enabled, **`food101_train_loss`, `food101_train_acc`, `food101_val_loss`, `food101_val_acc`** are written every epoch: numeric when that epoch ran Food-101, **empty cells** when the schedule skipped Food-101—so CSV columns stay aligned for plotting tools.

### 8.5 Checkpoints

Each epoch saves **`checkpoints/last.pt`**. If `val_mae` improves by at least `min_improve_delta`, saves **`checkpoints/best.pt`**.

**`state` dict keys:** `epoch`, `model_state_dict`, `optimizer_state_dict`, `val_mae`, `mode`, `split_type`, `image_size`, `max_depth_units`, `loss_type`, `use_log_target`, `has_classifier`, `food101_classes`, `cls_label_smoothing`, `food101_epoch_interval`, `food101_cls_passes`, `food101_every_n_epochs`.

### 8.6 End of training

Writes **`logs/train_summary.json`**: `best_val_mae`, `best_epoch`, `mode`, `split_type`, `loss_type`, `use_log_target`, `has_classifier`.

---

## 9. Nutrition5k test evaluation (`evaluate.py`)

1. `create_dataloaders(..., val_ratio=0.1, augment_train=False)` — **note:** `val_ratio` is **hardcoded to `0.1`** here; only the **test** split is used for metrics, so this matches training when training also used `val_ratio=0.1` for the train/val carve-out from the same official train list.
2. Load checkpoint; verify `checkpoint["mode"]` matches `--mode` and `checkpoint["split_type"]` matches `--split_type` (prevents evaluating an RGB-D checkpoint as RGB).
3. `use_log_target` from checkpoint (fallback CLI).
4. Aggregate preds/targets; **MAE**, **RMSE**, **MSE** in kcal space (same `expm1` path as training).
5. Prints JSON; writes **`logs/eval_metrics.json`** under `--output_dir`.
6. Optional **`write_predictions_csv`**: columns `dish_id`, `predicted_calories`, `target_calories`, `abs_error`.

---

## 10. Food-101 test accuracy (`evaluate_food101.py`)

Requires `has_classifier` and non-empty `food101_classes` in the checkpoint. Loads **`Food101(split="test")`**; asserts `ds.classes == food101_classes` (same root/version as training). RGB-D batches are **zero-padded to 4 channels** via `_expand_for_rgbd_if_needed`. Reports **`top1_accuracy`** and **`top{k}_accuracy`** (dynamic key name) for `k = min(top_k, 101)`. Optional **`logs/food101_test_metrics.json`** under `--output_dir`.

---

## 11. Utilities (`utils.py`)

| Symbol | Behavior |
|--------|-----------|
| `prevent_system_sleep_while_running` | macOS only: spawns `caffeinate -dims -w <pid>`, terminates on exit. |
| `set_seed` | Python/NumPy/torch seeds; `cudnn.deterministic=True`, `benchmark=False`. |
| `pick_device` | `cuda` if available else CPU; `--device` overrides. |
| `mae`, `rmse` | Mean absolute error, root mean square error on tensors. |
| `AverageMeter` | Running weighted average for loss/MAE meters. |
| `save_checkpoint` | `output_dir/checkpoints/<filename>.pt`. |
| `log_epoch` | Append one CSV row to `logs/train_log.csv` (header on first create). |
| `save_run_config` | `logs/config.json` from `dataclasses.asdict(cfg)` + UTC timestamp. |
| `write_predictions_csv` | Eval predictions export. |

---

## 12. Scripts under `scripts/`

### 12.1 `scripts/download_nutrition5k.py` (recommended)

Downloads from `gs://nutrition5k_dataset/nutrition5k_dataset/`:

| Tier | Contents |
|------|----------|
| `essentials` | `metadata/`, `dish_ids/` (official splits) |
| `overhead` | `imagery/realsense_overhead/` (~3.5k dishes — **what `train.py` uses**) |
| `full` | Entire bucket (~181 GB, includes side-angle videos) |

```bash
python scripts/download_nutrition5k.py --dataset_root ~/data/nutrition5k_dataset --tier essentials overhead
python scripts/download_nutrition5k.py --dataset_root ~/data/nutrition5k_dataset --tier overhead --only_missing
```

Requires **`gsutil`**.

### 12.2 `scripts/download_more_overhead.py` (legacy)

Incremental downloader with `--target_total` (default 1000). Use **`--target_total 0`** to fetch all missing split dishes, or prefer §12.1 for full overhead.

### 12.3 `scripts/check_overhead_integrity.py`

For each subdirectory of `imagery/realsense_overhead/`, checks presence of **`rgb.png`**, **`depth_raw.png`**, **`depth_color.png`** exactly. Reports totals and up to 20 partial examples. **Note:** training code accepts **other** RGB/depth filenames via keywords; a “partial” here does not always mean training will skip the dish—only that this strict triplet is missing.

### 12.4 `scripts/generate_presentation_assets.py`

- **Inputs:** `--dataset_root` (full Nutrition5k root), `--run_dir`, matching `--split_type`, `--val_ratio`, `--seed`; optional `--extra_run_dirs` for multi-model comparison.
- **Outputs:** `presentation/slide_assets/` (all figures) and `presentation/slide_picks/` (curated numbered PNGs for PPT). **No training.**

---

## 13. Gradio web app (`web_app.py`)

### 13.1 CLI

| Argument | Purpose |
|----------|---------|
| `--checkpoint_rgb`, `--checkpoint_rgbd` | Paths to `.pt` files (at least one required, or legacy `--checkpoint_path` + `--mode`). |
| `--image_size`, `--max_depth_units` | Fallbacks if missing from checkpoint (checkpoint values win when present). |
| `--host`, `--port` | `demo.launch(server_name, server_port)`. |
| `--auto_depth_backend {midas,heuristic}` | Default auto-depth backend for RGB-D when MiDaS load fails, falls back to heuristic. |
| `--cls_top_k`, `--cls_conf_threshold` | Top-K softmax lines in text output; if none above threshold, still shows top-1. |

### 13.2 `Predictor`

- Loads checkpoint; builds `CalorieRegressor(pretrained=False, num_classes=…)`; `eval()` mode.
- RGB preprocessing: EXIF transpose, resize, ImageNet normalize (same as training eval path for RGB).
- **RGB-D depth modes:** MiDaS `DPT_Hybrid` via `torch.hub` + `dpt_transform`; heuristic = grayscale luminance; real upload uses `depth_image_to_tensor` with checkpoint `max_depth_units`.
- **Calorie decode:** `expm1` + clamp if `use_log_target`.
- **Classes:** `classify` + softmax; filters by `cls_conf_threshold`.

### 13.3 UI

`AppRuntime.predict` switches between loaded predictors; RGB-D shows optional depth preview controls.

---

## 14. Hugging Face entry (`app.py`)

Used as **`app_file`** in Space metadata.

1. Reads **`RGB_CKPT_URL`** / **`RGBD_CKPT_URL`** env vars; if set, downloads to **`outputs_food101_4passes/checkpoints/best.pt`** and **`outputs_train_rgbd_food101/checkpoints/best.pt`** respectively (relative to process cwd).
2. Builds `sys.argv` for `web_app.main()`: `--checkpoint_rgb` / `--checkpoint_rgbd` when files exist, `--host 0.0.0.0`, `--port` from **`PORT`** env (default `7860`).
3. Raises **`RuntimeError`** if no checkpoint path ends up in argv.

---

## 15. Run artifacts: directories and files

Under each `--output_dir` (a valid **`--run_dir`** for the presentation script):

```text
checkpoints/
  last.pt
  best.pt
logs/
  config.json           # full training Config + timestamp
  train_log.csv         # per-epoch metrics (+ Food-101 columns when enabled)
  train_summary.json    # written at end of training
  eval_metrics.json     # from evaluate.py
  test_predictions.csv  # optional, from evaluate.py
  food101_test_metrics.json  # optional, from evaluate_food101.py
```

`.gitignore` ignores `outputs*/`, `*.pt`, and root-level `logs/` patterns—local runs may be untracked by default.

---

## 16. Presentation bundle (`presentation/`)

| Path | Role |
|------|------|
| `PRESENTATION.html` | **Current presentation deck** — browser-ready, self-contained HTML with embedded plots. |
| `PRESENTATION.md` | Editable source deck with audience-facing explanations, EDA, pipeline, model, metrics, demo link, and data-scope limitation. |
| `deck_style.css` | CSS used when exporting the HTML deck; keeps plots large in browser view. |
| `slide_assets/` | Figure library used by the deck (EDA, pipeline, model diagrams, metrics, errors, and one RGB/depth example). |
| `slide_picks/` | Curated numbered PNGs for optional PowerPoint export. |

Open the current deck directly:

```bash
open presentation/PRESENTATION.html
```

After editing `presentation/PRESENTATION.md`, rebuild the browser-ready deck:

```bash
cd presentation
pandoc PRESENTATION.md -f gfm -t html5 --standalone --embed-resources \
  --css deck_style.css \
  --metadata pagetitle="Nutrition5k Calorie Prediction" \
  -o PRESENTATION.html
```

---

## 17. Consolidated project guide (`docs/PROJECT_GUIDE.md`)

The old standalone notes for data scope, download instructions, and research-pipeline planning have been merged into **[`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md)**. This keeps the markdown structure small while preserving the important reviewer-facing material:

| Topic | Where it is now |
|-------|-----------------|
| Why a local Nutrition5k subset is valid | `docs/PROJECT_GUIDE.md`, section 2 |
| How to download essentials and overhead imagery | `docs/PROJECT_GUIDE.md`, section 4 |
| Stage 1 baseline vs future model comparison | `docs/PROJECT_GUIDE.md`, sections 1, 3, and 5 |
| Fair test-set rules | `docs/PROJECT_GUIDE.md`, section 5 |

---

## 18. Benchmark alignment vs approximations

**Aligned:** official dish calorie labels; official train/test ID files; evaluation on official **test** IDs that exist locally; same preprocessing and metrics for all **baseline** runs in this repo.

**Documented limitation (storage):** training covers a **subset** of official train IDs: the intersection with downloaded overhead RGB. The full Nutrition5k release is too large for the group laptops to store and train on comfortably, so results are presented as a controlled local-subset baseline rather than a full-data reproduction.

**Staged evaluation:** Stage 1 = this repo's ResNet-18 baseline; later models should use the same `test_predictions.csv` contract where possible (`--split_type`, `--seed`, and test dish IDs unchanged).

**Engineering / paper gap:** validation is a random hold-out from official train IDs (seeded, `--val_ratio`); RGB/depth filenames resolved by keyword heuristics; ResNet-18 + shallow head is a compact baseline, not the full Nutrition5k paper system; RGB-D fusion is early concat; web demo RGB-D may use estimated depth (MiDaS), not necessarily the same as training depth maps.

---

## 19. Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| `ValueError` checkpoint mode / split_type mismatch in `evaluate.py` | `--mode` / `--split_type` disagree with checkpoint; match training flags. |
| `evaluate_food101.py` class list mismatch | Wrong `--food101_root` or different torchvision cache than training. |
| Food-101 cls `ValueError` on evaluate | Train with `--enable_food101_cls` first. |
| `gsutil` errors in downloader | GCS credentials / `gsutil` not installed; bucket path changed. |
| MiDaS warning in web app | Torch hub download blocked; app falls back to heuristic depth. |
| Empty train loader | No local overhead RGB for train IDs—download more dishes or fix `--dataset_root`. |
| `OSError: read-only file system: '/path'` when generating slides | You used README **placeholder** paths; pass real `--dataset_root` and `--run_dir`. |

---

## 20. Further reading

- Nutrition5k: [google-research-datasets/Nutrition5k](https://github.com/google-research-datasets/Nutrition5k)
- RGB-D nutrition example: [SightVanish/NutritionEstimation](https://github.com/SightVanish/NutritionEstimation)
- Structured pipeline reference: [Lyce24/NutriFusionNet](https://github.com/Lyce24/NutriFusionNet)

---

## Appendix A — Example commands (RGB, RGB-D, Food-101)

**RGB train/eval** — use `--split_type rgb` with `--mode rgb`.

**RGB-D train/eval** — use `--split_type depth` with `--mode rgbd` so split IDs align with depth modality lists from the paper setup.

**Food-101 auxiliary train** — add `--enable_food101_cls --food101_root <Food101_root> [--food101_download]`.

**Food-101 test report:**

```bash
python evaluate_food101.py \
  --checkpoint_path outputs_rgb_food101/checkpoints/best.pt \
  --food101_root /path/to/food-101 \
  --mode rgb \
  --output_dir outputs_rgb_food101
```

**Gradio (two checkpoints):**

```bash
python web_app.py \
  --checkpoint_rgb outputs_food101_4passes/checkpoints/best.pt \
  --checkpoint_rgbd outputs_train_rgbd_food101/checkpoints/best.pt \
  --host 127.0.0.1 --port 7860
```
