# Presentation materials

Figures are generated from training runs on the **full Nutrition5k overhead dataset** (`~/data/nutrition5k_dataset` or your `--dataset_root`).

## Folders

| Path | Use |
|------|-----|
| **`slide_picks/`** | Curated, numbered PNGs for PowerPoint — start here. |
| **`slide_assets/`** | Full export (EDA, all runs, CSV/JSON). |

## Regenerate

```bash
python scripts/generate_presentation_assets.py \
  --dataset_root ~/data/nutrition5k_dataset \
  --run_dir outputs_train_rgbd_food101 \
  --extra_run_dirs outputs_food101_4passes \
  --split_type depth \
  --val_ratio 0.1 \
  --seed 42
```

Refreshes both `slide_assets/` and `slide_picks/`. See `slide_assets/README.md` for the file index.

## Pipeline diagram

`MODEL_PIPELINE.md` — Mermaid source; export PNG via [mermaid.live](https://mermaid.live) if needed.
