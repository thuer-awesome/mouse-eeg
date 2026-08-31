# 仓库冻结快照索引

当前外部审计 run 的唯一交接入口是 [`docs/CURRENT_DELIVERY_zh.md`](docs/CURRENT_DELIVERY_zh.md)。本页只说明仓库内可提交的冻结快照；不要把它们与外部 run 中的旧目录按名称拼接。

| 模块 | 唯一正式目录 | 关键入口 |
|---|---|---|
| 任务一：二分类连续打标冻结快照 | `results/task1_annotation_detection/` | `README_zh.md`、`figures/model_performance/`、`event_atlas/`、`release_manifest.json` |
| 任务二：PG36 形态学分类冻结快照 | `results/task2_morphology_pg36/` | `taxonomy/`、`figures/`、`classification/all_window_model_comparison.csv`、`release_manifest.json` |

任务一权重在 `results/task1_annotation_detection/models/<model>/model.pt`。任务二权重在 `results/task2_morphology_pg36/classification/<model>/best_model.pt`。这些是仓库冻结快照；任务二当前外部 PG36 表与任务一严格 protocol 的键/split 对齐未通过，详见交接索引，因此不能把两者写成同一独立测试集结果。正式入口不包括 `_staging_task2_previous/`，也不包括源项目的旧 45 类、历史 21 类或旧 decoder 目录。

图形含义按论文式的“数据/标签 → 表征 → 分类 → 审计”顺序组织：任务一先看模型性能图，再看按覆盖类别分开的事件 PDF；任务二先看逐鼠独立通道 PCA/波形，再看 36 个全局质心和支持矩阵，最后看五个模型的全类别 PR/ROC 与混淆矩阵。
