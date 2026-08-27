# 正式结果索引

| 模块 | 唯一正式目录 | 关键入口 |
|---|---|---|
| 任务一：二分类连续打标 | `results/task1_annotation_detection/` | `README_zh.md`、`figures/model_performance/`、`event_atlas/`、`release_manifest.json` |
| 任务二：PG36 形态学分类 | `results/task2_morphology_pg36/` | `taxonomy/`、`figures/`、`classification/all_window_model_comparison.csv`、`release_manifest.json` |

任务一权重在 `results/task1_annotation_detection/models/<model>/model.pt`。任务二权重在 `results/task2_morphology_pg36/classification/<model>/best_model.pt`。正式结果不包括 `_staging_task2_previous/`，也不包括源项目的旧 45 类、历史 21 类或旧 decoder 目录。

图形含义按论文式的“数据/标签 → 表征 → 分类 → 审计”顺序组织：任务一先看模型性能图，再看按覆盖类别分开的事件 PDF；任务二先看逐鼠独立通道 PCA/波形，再看 36 个全局质心和支持矩阵，最后看五个模型的全类别 PR/ROC 与混淆矩阵。
