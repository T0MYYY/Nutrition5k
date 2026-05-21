# Presentation materials

Figures are generated from your training runs. Quote **actual** train/val/test counts from EDA (storage-limited subset is expected). See **[../docs/RESEARCH_PIPELINE.md](../docs/RESEARCH_PIPELINE.md)**.

## Folders

| Path | Use |
|------|-----|
| **`slide_picks/`** | Curated numbered PNGs for slides — start here. |
| **`slide_assets/`** | Full export (EDA, metrics, all runs). |
| **`MODEL_PIPELINE.md`** | Mermaid diagram; export via [mermaid.live](https://mermaid.live). |

## Regenerate

```bash
python scripts/generate_presentation_assets.py \
  --dataset_root ~/data/nutrition5k_mini \
  --run_dir outputs_train_rgbd_food101 \
  --extra_run_dirs outputs_food101_4passes \
  --split_type depth \
  --val_ratio 0.1 \
  --seed 42
```

Refreshes `slide_assets/` and `slide_picks/`.
