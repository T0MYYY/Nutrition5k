---
title: Food Calorie App
emoji: 🍱
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Nutrition5k Calorie Prediction (Reproducible Baseline)

This project provides a clean, reproducible pipeline for calorie regression on Nutrition5k:

`image (RGB or RGB-D) -> ResNet regressor -> calories`

## Official resources used

- Nutrition5k dataset and official repo: [google-research-datasets/Nutrition5k](https://github.com/google-research-datasets/Nutrition5k)
- RGB-D baseline example: [SightVanish/NutritionEstimation](https://github.com/SightVanish/NutritionEstimation)
- Structured training pipeline reference: [Lyce24/NutriFusionNet](https://github.com/Lyce24/NutriFusionNet)

## Nutrition5k structure summary

Based on the official Nutrition5k release:

- `imagery/side_angles/`: 4 camera side-angle videos per dish (`A-D`).
- `imagery/realsense_overhead/`: overhead RGB + depth images by dish ID.
- `metadata/ingredient_metadata.csv`: ingredient nutritional metadata.
- `metadata/dish_metadata_cafe1.csv`, `metadata/dish_metadata_cafe2.csv`: dish-level nutrition labels (`total_calories`, mass, macros, per-ingredient fields).
- `dish_ids/splits/`: official train/test dish ID split files.

Train/test split design in official benchmark:

- Dish IDs are split so incremental scans of the same plate do not leak across train/test.
- This project uses those official split files when present.
- Validation split is created from the official train split using a fixed seed (`--seed`) and `--val_ratio`.

## Expected local dataset layout

Place the extracted dataset as:

```text
nutrition5k_dataset/
  imagery/
    realsense_overhead/
      dish_XXXXXXXXXX/
        <rgb file>
        <depth file>
  metadata/
    dish_metadata_cafe1.csv
    dish_metadata_cafe2.csv
    ingredient_metadata.csv
  dish_ids/
    splits/
      <train split file>
      <test split file>
```

## Download and setup

From the official bucket (example):

```bash
gsutil -m cp -r "gs://nutrition5k_dataset/nutrition5k_dataset" .
```

Or download the full archive from the official Nutrition5k page and extract it.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project files

- `config.py`: central CLI/config object
- `data_loader.py`: Nutrition5k parsing, split loading, RGB/RGB-D dataset
- `model.py`: pretrained ResNet18 calorie regressor (RGB + RGB-D mode)
- `train.py`: train/validation loop, checkpointing, CSV logging
- `evaluate.py`: Nutrition5k test evaluation (MAE, RMSE) + optional predictions CSV
- `evaluate_food101.py`: Food-101 official **test** split top-1 / top-k accuracy (requires cls checkpoint)
- `utils.py`: seed/device/metrics/checkpoint helpers
- `web_app.py`: local upload-and-predict web demo (Gradio)
- `scripts/download_more_overhead.py`: incremental overhead-data downloader for mini setups

## Optional: grow mini dataset

If you start with a small subset, you can incrementally download more overhead dishes:

```bash
python scripts/download_more_overhead.py \
  --dataset_root /Users/yiouwang/data/nutrition5k_mini \
  --target_total 1000 \
  --seed 42
```

Optional integrity check after downloading:

```bash
python scripts/check_overhead_integrity.py \
  --dataset_root /Users/yiouwang/data/nutrition5k_mini
```

## Full pipeline: expand data and train RGB + RGB-D

If you have roughly 1000 local overhead dishes, a practical next step is to grow to **about 3000** (`--target_total` in the downloader) before training: enough gain to help the model without requiring the full release. The helper script (a) tops up with `gsutil` until the local `realsense_overhead` count reaches that target, then (b) trains **RGB** with `--split_type rgb` and **RGB-D** with `--split_type depth` so each run matches the official list for that modality.

```bash
chmod +x scripts/run_full_pipeline.sh
export NUTRITION5K_ROOT="/path/to/nutrition5k_dataset"   # must contain metadata, dish_ids/splits, imagery/
./scripts/run_full_pipeline.sh
```

Environment overrides:

- `TARGET_TOTAL=3000` (default) — set lower (e.g. `2000`) if disk is tight; set `SKIP_DOWNLOAD=1` to only train on data you already have.
- `EPOCHS=40` (default), `BATCH=32` — reduce batch size on CPU or if you hit OOM.

Checkpoints: `outputs_full_rgb/checkpoints/best.pt` and `outputs_full_rgbd/checkpoints/best.pt`. Evaluate test sets with `evaluate.py` using the same `--mode` and `--split_type` as each training run.

## Phase 1: RGB baseline

- Input: overhead RGB only
- Backbone: pretrained **ResNet-18** only (fixed in code for speed and memory on typical laptops)
- Head: small MLP regression head
- Loss: SmoothL1 (default), optional MSE
- Metrics: MAE (during train/val), MAE + RMSE (test)
- **Train-only data augmentation** (on by default): random horizontal flip + mild `ColorJitter`; RGB-D mode applies the same flip to depth. Disable with `--no_train_augment`.
- **Optional** split learning rates: `--backbone_lr` and `--head_lr` together (e.g. `3e-5` and `3e-4`); if omitted, `--lr` applies to the whole model.
- Default stabilization: `log1p` target training + ReduceLROnPlateau + early stopping

Train:

```bash
python train.py \
  --dataset_root /path/to/nutrition5k_dataset \
  --mode rgb \
  --split_type rgb \
  --epochs 40 \
  --batch_size 32 \
  --output_dir outputs_rgb \
  --loss_type smooth_l1 \
  --scheduler plateau \
  --use_log_target \
  --pretrained
```

Evaluate:

```bash
python evaluate.py \
  --dataset_root /path/to/nutrition5k_dataset \
  --mode rgb \
  --split_type rgb \
  --checkpoint_path outputs_rgb/checkpoints/best.pt \
  --output_dir outputs_rgb \
  --save_predictions_csv outputs_rgb/logs/test_predictions.csv
```

## Phase 2: RGB-D extension

Implemented using a simple, stable approach:

- 4-channel input (`RGB + depth`) to the same ResNet backbone.
- First conv layer is adapted from pretrained RGB weights; depth channel is initialized from mean RGB filters.

Train:

```bash
python train.py \
  --dataset_root /path/to/nutrition5k_dataset \
  --mode rgbd \
  --split_type depth \
  --epochs 40 \
  --batch_size 32 \
  --output_dir outputs_rgbd \
  --loss_type smooth_l1 \
  --scheduler plateau \
  --use_log_target \
  --pretrained
```

Evaluate:

```bash
python evaluate.py \
  --dataset_root /path/to/nutrition5k_dataset \
  --mode rgbd \
  --split_type depth \
  --checkpoint_path outputs_rgbd/checkpoints/best.pt \
  --output_dir outputs_rgbd \
  --save_predictions_csv outputs_rgbd/logs/test_predictions.csv
```

## Optional: Food-101 category + confidence (multi-task)

You can train an additional Food-101 classification head on top of the same backbone so the app outputs both:
- predicted calories
- predicted food category candidates with softmax confidence

**Food-101 training defaults (quality-oriented):**
- **Train augmentations** (on by default): `RandomResizedCrop`, horizontal flip, `ColorJitter`, `RandomErasing`, then ImageNet normalization. Disable with `--no_food101_augment`.
- **Validation**: a reproducible **hold-out slice of the official Food-101 train split** (`--food101_val_ratio`, default `0.1`). The official **test** split is not used during the training loop, so you are not tuning on test labels every epoch.
- **Label smoothing** (optional): `--cls_label_smoothing 0.05` on the classification loss during Food-101 **train** steps only (validation uses plain cross-entropy for a standard accuracy readout).
- **Food-101 frequency vs. Nutrition5k** (Nutrition5k still every epoch):
  - **`--food101_every_n_epochs N`** with `N>0`: run Food-101 on epochs **1, N+1, 2N+1, …** (fixed stride). This **overrides** `--food101_cls_passes` for scheduling.
  - **`--food101_cls_passes P`** with `P=0` (default): Food-101 **every** epoch. `P>0` with `--epochs E`: about **P** runs total, evenly spaced (e.g. `E=40`, `P=4` → **1, 14, 27, 40**). Use this when you care about “roughly P Food-101 passes” rather than a fixed stride.
  - At startup, `train.py` prints how many Food-101 steps there are and **which epochs** (truncated if the list is long).

Example:

```bash
python train.py \
  --dataset_root /path/to/nutrition5k_dataset \
  --mode rgb \
  --split_type rgb \
  --epochs 20 \
  --output_dir outputs_rgb_food101 \
  --enable_food101_cls \
  --food101_root /path/to/food101_root \
  --food101_download \
  --cls_batch_size 32 \
  --cls_loss_weight 1.0 \
  --cls_label_smoothing 0.05 \
  --pretrained
```

**Official Food-101 test accuracy** (after training with `--enable_food101_cls`):

```bash
python evaluate_food101.py \
  --checkpoint_path outputs_rgb_food101/checkpoints/best.pt \
  --food101_root /path/to/food101_root \
  --mode rgb \
  --output_dir outputs_rgb_food101
```

Uses the torchvision **test** split only (no augmentation). For RGB-D checkpoints, pass `--mode rgbd` (Food-101 images are still RGB; depth is zero-padded like in training). Optional: `--top_k 5` (default), `--food101_download` if data is missing.

Note: Food-101 is a single-label dataset. The app can show multiple high-confidence class candidates (top-K), but that is candidate ranking, not true multi-label supervision.

## Reproducibility details

- Fixed random seed with `--seed` for Python, NumPy, and PyTorch.
- No hardcoded dataset paths; use `--dataset_root`.
- Configurable mode (`rgb` or `rgbd`), batch size, image size, workers, LR, epochs.
- Configurable split family via `--split_type` (`rgb`, `depth`, or `auto`).
- Checkpoint stores `mode` and `split_type`; evaluation validates both to avoid accidental mismatch.
- Configurable optimization behavior (`--loss_type`, `--scheduler`, `--use_log_target`, early stopping).
- Logging:
  - Epoch metrics: `output_dir/logs/train_log.csv`
  - Training config: `output_dir/logs/config.json`
  - Eval metrics: `output_dir/logs/eval_metrics.json`
- Checkpoints:
  - latest: `output_dir/checkpoints/last.pt`
  - best by validation MAE: `output_dir/checkpoints/best.pt`

## Web demo: upload your own food image

Install/update dependencies:

```bash
pip install -r requirements.txt
```

Run one web app with both checkpoints loaded:

```bash
python web_app.py \
  --checkpoint_rgb outputs_food101_4passes/checkpoints/best.pt \
  --checkpoint_rgbd outputs_train_rgbd_food101/checkpoints/best.pt \
  --host 127.0.0.1 \
  --port 7860
```

Then open [http://127.0.0.1:7860](http://127.0.0.1:7860).

In the UI:
- Choose `rgb` or `rgbd` from the **Inference Mode** dropdown (no terminal mode switch needed).
- The app always preprocesses uploaded RGB photos automatically (EXIF correction, resize, normalization).

### RGB-D depth source options (dropdown)

When `rgbd` mode is selected, choose one depth source in UI:

1. `Auto from RGB (MiDaS)`  
   Uses MiDaS monocular depth estimation.
2. `Auto from RGB (Heuristic)`  
   Uses grayscale pseudo-depth fallback.
3. `Upload real depth image`  
   Upload a real aligned depth image (PNG preferred) and the app converts it to model input format.

For best results, use overhead food photos similar to Nutrition5k capture style.

## What is official benchmark-aligned vs approximation

Benchmark-aligned:

- Uses official Nutrition5k dish metadata CSV labels (`total_calories`).
- Uses official train/test split files when available in `dish_ids/splits`.
- Uses overhead RGB(-D) imagery in `imagery/realsense_overhead`.

Approximation (engineering choices in this baseline):

- Validation split is derived from train split (official dataset gives train/test, not fixed val).
- Filename matching for RGB/depth files is robust by keyword because exact file naming can vary in local exports.
- Model architecture is a compact ResNet18 regressor baseline, not the full Nutrition5k paper model pipeline.
- Depth handling uses single-channel normalized depth concatenated to RGB (simple and stable, not advanced fusion).
