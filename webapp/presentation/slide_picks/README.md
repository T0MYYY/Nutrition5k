# 精选可视化（PPT 快速挑选）

从 `slide_assets/` 自动复制的 **最有用** 图/表，文件名按推荐放映顺序编号。
重新运行 `generate_presentation_assets.py` 会刷新本目录。

| 文件 | 说明 |
|------|------|
| `01_data_split_and_calorie_distribution.png` | 划分后各 split 样本数 + 热量密度分布 |
| `02_data_calorie_stats_table.png` | train/val/test 热量统计表 |
| `03_data_depth_coverage.png` | 本地 depth 文件覆盖率（RGB-D 相关） |
| `04_data_sample_dishes.png` | 随机样例俯拍图 + 标签热量 |
| `05_pipeline_end_to_end.png` | 数据 → 预处理 → ResNet → 输出 |
| `06_model_architecture_RGB.png` | RGB 模型结构示意 |
| `07_model_architecture_RGBD.png` | RGB-D 四通道模型结构示意 |
| `08_metrics_all_models_table.png` | 各次训练：val/test MAE、best epoch 等 |
| `09_metrics_test_mae_rmse_bars.png` | 测试集 MAE / RMSE 柱状对比 |
| `08_metrics_all_models.csv` | 上表原始数据（可贴 Excel） |
| `10_train_curves_RGBD.png` | 主 run（RGB-D）：loss + val MAE |
| `11_train_curves_RGB.png` | 对比 run（RGB）：loss + val MAE |
| `12_train_food101_aux_accuracy.png` | Food-101 辅助分类准确率 |
| `20_test_scatter_RGBD.png` | RGB-D 测试集：预测 vs 真值 + 残差 |
| `21_test_scatter_RGB.png` | RGB 测试集：预测 vs 真值 + 残差 |
| `22_test_error_by_calorie_bin_RGBD.png` | RGB-D：按真值热量分箱的平均误差 |
| `23_test_error_by_calorie_bin_RGB.png` | RGB：按热量分箱的平均误差 |
| `24_test_error_cdf_RGBD.png` | RGB-D：绝对误差累积分布 |
| `25_test_worst_errors_RGBD.png` | RGB-D：误差最大的若干道菜 |

## 对应训练目录

- **RGB-D（主 run）**：`outputs_train_rgbd_food101` → 文件名含 `RGBD` / `10_` / `20_`–`25_`
- **RGB（对比）**：`outputs_food101_4passes` → 文件名含 `RGB` / `11_` / `21_`–`23_`

完整素材库见上一级目录 `slide_assets/`。