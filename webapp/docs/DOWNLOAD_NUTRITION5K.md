# Nutrition5k 全量下载指南

本仓库的训练与评估默认使用 **官方 GCS 上的完整 overhead 数据**（`imagery/realsense_overhead/`，约 **3490** 道菜），配合 `metadata/` 与 `dish_ids/splits/`。不需要 181 GB 侧拍视频，除非复现 side-angle 实验。

## 数据规模（官方）

| 内容 | 说明 |
|------|------|
| 论文/metadata | ~5006 道菜 |
| GCS overhead（本仓库使用） | **约 3490** 个 `dish_*` 文件夹 |
| 完整 tar.gz | `nutrition5k_dataset.tar.gz`，约 **181 GB** |
| 推荐本地路径 | `~/data/nutrition5k_dataset`（与 README / `train.py` 示例一致） |

官方 bucket：`gs://nutrition5k_dataset/nutrition5k_dataset/`

## 0. 前置：安装 gsutil

```bash
brew install --cask google-cloud-sdk   # macOS
gcloud auth login                      # 可选，公开 bucket 通常可直接读
gsutil ls gs://nutrition5k_dataset/nutrition5k_dataset/
```

## 1. 标准下载（metadata + 全部 overhead）

```bash
export N5K_ROOT="$HOME/data/nutrition5k_dataset"
mkdir -p "$N5K_ROOT"

cd "/path/to/food calorie"
python scripts/download_nutrition5k.py \
  --dataset_root "$N5K_ROOT" \
  --tier essentials overhead
```

- **essentials**：`metadata/` + `dish_ids/splits/`
- **overhead**：全部 `imagery/realsense_overhead/`

中断后 **重新执行同一条命令** 即可续传（`gsutil` 跳过已有文件）。

## 2. 补全缺失（resume）

若已有部分数据，只下本地没有的 dish：

```bash
python scripts/download_nutrition5k.py \
  --dataset_root "$N5K_ROOT" \
  --tier overhead \
  --only_missing
```

补全云端全部 overhead（不限 split 列表）：

```bash
python scripts/download_nutrition5k.py \
  --dataset_root "$N5K_ROOT" \
  --tier overhead --only_missing --all_remote
```

## 3. 完整 181 GB（含侧拍视频）

```bash
python scripts/download_nutrition5k.py \
  --dataset_root "$HOME/data/nutrition5k_full" \
  --tier full
```

## 4. 下载后检查与训练

```bash
python scripts/check_overhead_integrity.py --dataset_root "$N5K_ROOT"

python train.py \
  --dataset_root "$N5K_ROOT" \
  --mode rgbd --split_type depth \
  --output_dir outputs_rgbd_full \
  --loss_type smooth_l1 --scheduler plateau \
  --use_log_target --pretrained
```

## 5. 代码是否需要修改？

**不需要改 `data_loader.py` / `train.py`。** 它们会：

- 使用官方 `dish_ids/splits/`
- 只训练本地已有 RGB 的样本（全量 overhead 下载后与 split 基本对齐）

## 6. 旧脚本

`scripts/download_more_overhead.py` 曾用于 `--target_total 1000` 的增量下载；全量请用 `download_nutrition5k.py`。若仍用旧目录，可 `--target_total 0` 补全 split 内缺失项。
