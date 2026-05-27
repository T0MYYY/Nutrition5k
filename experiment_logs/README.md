# Experiment logs

Per-run training logs and final metrics from the Track B reproduction. Training ran on Colab (A100); the full datasets and `best.pt` checkpoints live on Google Drive and are intentionally not committed here.

| Run | Description |
|---|---|
| `exp1/` | Exp 1 — per-gram, side-angle (InceptionV3 multi-task) |
| `exp2/` | Exp 2 — direct, side-angle |
| `exp3/` | Exp 3 — direct, overhead RGB-D |
| `exp4/` | Exp 4 — volume-scalar mass pipeline |
| `exp1_convnext/` | Exp 1 ablation — ConvNeXt-Small backbone |
| `dpf/` | DPF-Nutrition — ImageNet ResNet-101 init |
| `dpf_food2k/` | DPF-Nutrition — Food2K ResNet-101 init |

- `train_log.csv` — per-epoch `train_loss`, `val_loss`, `lr`, `elapsed_s`.
- `metrics.json` (DPF runs) — final test MAE and MAE% per nutrient (calories/mass/fat/carb/protein).
