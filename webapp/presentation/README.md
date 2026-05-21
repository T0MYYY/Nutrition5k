# Presentation materials

## PPT asset folder (recommended)

Run the generator:

- **`presentation/slide_picks/`** — **精选**（约 15–20 张），按编号排序，优先看这个文件夹挑 PPT 图。
- **`presentation/slide_assets/`** — 完整素材库（全部 PNG/CSV）。

```bash
python scripts/generate_presentation_assets.py \
  --dataset_root "/path/to/nutrition5k_dataset" \
  --run_dir outputs_train_rgbd_food101 \
  --extra_run_dirs outputs_food101_4passes \
  --split_type depth \
  --val_ratio 0.1 \
  --seed 42
```

See **`slide_assets/README.md`** after generation for a full file index.

## Other paths

| Path | What it is |
|------|----------------|
| **`slide_assets/`** | **Generated** figures + tables for slides (default output). |
| **`MODEL_PIPELINE.md`** | Mermaid pipeline; also copied into `slide_assets/` when you regenerate. |
