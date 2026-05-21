# Downloading Nutrition5k

Training uses the **official split** plus a **local overhead subset** that fits your disk. Why that is methodologically sound: **[DATA_SCOPE.md](DATA_SCOPE.md)**. Full pipeline context: **[RESEARCH_PIPELINE.md](RESEARCH_PIPELINE.md)**.

- GCS overhead: ~**3,490** dish folders  
- Full `.tar.gz`: ~**181 GB** (includes side-angle video; **not required** for this repo)

Bucket: `gs://nutrition5k_dataset/nutrition5k_dataset/`

---

## Prerequisites

```bash
# macOS example
brew install --cask google-cloud-sdk
gcloud auth login   # optional; public bucket is usually readable
gsutil ls gs://nutrition5k_dataset/nutrition5k_dataset/
```

---

## Recommended: essentials + overhead

```bash
export N5K_ROOT="$HOME/data/nutrition5k_mini"   # or nutrition5k_dataset
mkdir -p "$N5K_ROOT"

cd "/path/to/food calorie"
python scripts/download_nutrition5k.py \
  --dataset_root "$N5K_ROOT" \
  --tier essentials overhead
```

| Tier | Contents |
|------|----------|
| `essentials` | `metadata/`, `dish_ids/splits/` (small) |
| `overhead` | `imagery/realsense_overhead/` (large) |

Re-run the same command to resume; `gsutil` skips existing files.

---

## Laptop with limited storage

Download metadata and splits first, then grow imagery incrementally:

```bash
# Cap download size (legacy helper)
python scripts/download_more_overhead.py \
  --dataset_root "$N5K_ROOT" \
  --target_total 3000

# Or resume only missing dishes
python scripts/download_nutrition5k.py \
  --dataset_root "$N5K_ROOT" \
  --tier overhead --only_missing
```

All missing IDs in split files:

```bash
python scripts/download_more_overhead.py \
  --dataset_root "$N5K_ROOT" \
  --target_total 0
```

---

## Full 181 GB bundle (optional)

Includes side-angle video. Not used by `train.py` in this project.

```bash
python scripts/download_nutrition5k.py \
  --dataset_root "$HOME/data/nutrition5k_full" \
  --tier full
```

---

## After download

```bash
python scripts/check_overhead_integrity.py --dataset_root "$N5K_ROOT"

python train.py \
  --dataset_root "$N5K_ROOT" \
  --mode rgbd --split_type depth \
  --output_dir outputs_rgbd \
  --loss_type smooth_l1 --scheduler plateau \
  --use_log_target --pretrained
```

No changes to `data_loader.py` or `train.py` are required—the loaders always use official splits and only dishes with local RGB.

---

## Scripts

| Script | Use |
|--------|-----|
| `download_nutrition5k.py` | **Preferred** — essentials / overhead / full |
| `download_more_overhead.py` | Legacy incremental download (`--target_total N`, or `0` for all missing in splits) |
