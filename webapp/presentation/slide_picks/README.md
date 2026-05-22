# Curated slide figures

Copied from `slide_assets/` for PowerPoint. Numbered in recommended presentation order.
Regenerate with `scripts/generate_presentation_assets.py`.
Pipeline context: [docs/RESEARCH_PIPELINE.md](../docs/RESEARCH_PIPELINE.md).

| File | Description |
|------|-------------|
| `01_data_split_and_calorie_distribution.png` | Split counts + calorie density (subset on disk) |
| `02_data_calorie_stats_table.png` | Calorie stats table per split |
| `03_data_depth_coverage.png` | Local depth file coverage (RGB-D) |
| `04_data_sample_dishes.png` | Sample overhead RGB dishes + labels |
| `05_pipeline_end_to_end.png` | End-to-end pipeline diagram |
| `06_model_architecture_RGB.png` | Baseline architecture (RGB) |
| `07_model_architecture_RGBD.png` | Baseline architecture (RGB-D) |
| `08_metrics_all_models_table.png` | Val/test MAE, best epoch — all runs |
| `09_metrics_test_mae_rmse_bars.png` | Test MAE / RMSE bar chart |
| `08_metrics_all_models.csv` | Metrics table (CSV) |
| `10_train_curves_RGBD.png` | Primary run (RGB-D): loss + val MAE |
| `11_train_curves_RGB.png` | Comparison run (RGB): loss + val MAE |
| `12_train_food101_aux_accuracy.png` | Food-101 auxiliary accuracy |
| `20_test_scatter_RGBD.png` | RGB-D test: pred vs truth + residuals |
| `21_test_scatter_RGB.png` | RGB test: pred vs truth + residuals |
| `22_test_error_by_calorie_bin_RGBD.png` | RGB-D: mean |error| by calorie bin |
| `23_test_error_by_calorie_bin_RGB.png` | RGB: mean |error| by calorie bin |
| `24_test_error_cdf_RGBD.png` | RGB-D: CDF of absolute error |
| `25_test_worst_errors_RGBD.png` | RGB-D: largest test errors |

## Run mapping

- **RGB-D (primary):** `outputs_train_rgbd_food101` — files `10_*`, `20_*`–`25_*`
- **RGB (comparison):** `outputs_food101_4passes` — files `11_*`, `21_*`–`23_*`

Full asset library: `../slide_assets/`.