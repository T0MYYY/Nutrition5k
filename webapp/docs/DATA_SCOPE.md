# Data scope and methodological justification

Companion to **[RESEARCH_PIPELINE.md](RESEARCH_PIPELINE.md)**. Use this page when a reviewer asks: *“Why only part of Nutrition5k? Is that still valid?”*

---

## Summary

We train on **official Nutrition5k labels and split files**, restricted to dishes whose **overhead RGB** exists on disk. That is a **storage-limited subset**, not a random mini-dataset. The current model is a **Stage-1 baseline** (ResNet-18); a **VLM** is planned for Stage 2 with the **same fixed test set** and the same MAE/RMSE metrics.

---

## 1. Hardware constraint

| Resource | Official release | Our laptop setup |
|----------|------------------|------------------|
| Full `.tar.gz` | ~181 GB (includes side-angle video) | Not downloaded |
| Overhead RGB-D on GCS | ~3,490 dish folders | **Subset** that fits available disk |
| What we need for `train.py` | `realsense_overhead/` + metadata + splits | Incremental download |

We do not need side-angle video for the baseline in this repository.

---

## 2. What remains methodologically sound

1. **Official calorie labels** — `total_calories` from `metadata/dish_metadata_cafe{1,2}.csv`.
2. **Official split IDs** — `dish_ids/splits/` (`rgb_*` or `depth_*` train/test lists).
3. **No test leakage** — Test = official test file only; val = subset of official train IDs (`--val_ratio`, `--seed`).
4. **Transparent filter** — `build_split_samples()` uses only dishes with local RGB; skips are logged and appear in EDA.
5. **Fixed evaluation for comparisons** — `evaluate.py` + `test_predictions.csv` lock the test dishes for baseline vs future VLM.

**Formal definition:**

```text
train/val/test lists = official_split_ids  ∩  { id | local overhead RGB exists }
```

---

## 3. Why subset training is enough for Stage 1

| Argument | Explanation |
|----------|-------------|
| Transfer learning | ImageNet-pretrained ResNet-18; head learns scalar kcal mapping only |
| Scale | Typically ~2k+ train dishes—enough for a **baseline**, not the full Nutrition5k paper system |
| Official test | Metrics on held-out **test** IDs (local), not training memorization |
| Reproducibility | Fixed `--seed`, saved `config.json`, same `dataset_root` → same lists |
| Roadmap | Subset stabilizes baseline; VLM compared on **same test CSV** |

We **do not** claim identical numbers to the CVPR 2021 table trained on Google's full media.

---

## 4. Baseline now, VLM later — same test set

### Stage 1 (this repository)

- ResNet-18 + MLP calorie head; optional RGB-D and Food-101 auxiliary head.
- Purpose: working pipeline, RGB vs RGB-D ablation, **reference MAE/RMSE**.

### Stage 2 (planned)

- Vision–language model (image + text → calories).
- Same `total_calories`, same **test `dish_id` list**, same MAE/RMSE.
- Baseline predictions from Stage 1 are the benchmark to beat.

### Locked after baseline eval

Keep `logs/test_predictions.csv`, `logs/config.json`, and `logs/eval_metrics.json`. Do **not** change `--split_type` or `--seed` between stages unless you define a new benchmark.

**Logic chain for slides:**

```text
Storage limit → official overhead subset for training
       ↓
Official labels + splits (no ad-hoc split)
       ↓
Stage 1: ResNet baseline → MAE/RMSE on fixed test (CSV)
       ↓
Stage 2: VLM → same test dishes → fair improvement
```

---

## 5. Report wording

**Data scope**

> We use Nutrition5k overhead imagery with official calorie labels and train/test split files. Because of limited laptop storage, training uses the subset of official train IDs with locally downloaded RGB. Test metrics use the official test ID list (dishes available locally). This supports our Stage-1 baseline.

**Model roadmap**

> Stage 1 implements a reproducible ResNet-18 baseline to validate the pipeline and record reference test MAE/RMSE. Stage 2 will add a vision–language model on the **same fixed test set** (same dish IDs, targets, and metrics). We will not change the test split between stages; gains are attributed to the advanced model. Baseline outputs are saved with `evaluate.py --save_predictions_csv`.

---

## 6. Scaling up

When disk space allows, download more overhead folders (`download_nutrition5k.py --only_missing`). Point `--dataset_root` at the larger tree and re-run training—**no loader code changes** required.

---

## 7. Code references

- `data_loader.build_split_samples()`
- `scripts/download_nutrition5k.py`
- `scripts/generate_presentation_assets.py` (EDA counts for slides)
