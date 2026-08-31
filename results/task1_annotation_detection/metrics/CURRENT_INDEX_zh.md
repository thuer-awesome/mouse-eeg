# 任务一 corrected 输出索引

## 当前连续起止打标图集

`../event_atlas/` 是任务一 corrected interval-aware decoder 的完整临床图集。每个模型目录包含人工事件覆盖、部分覆盖、低重叠、完全漏检、长事件漏检、边界偏移和标签外候选的代表图及完整 PDF 图集；图层包括原始波形、原始/平滑概率和 decoded state。

图集分母为全部 `957` 个测试人工通道事件。正式完整可观测 interval-aware 指标使用 `953` 个事件，另外 4 个形式审计排除事件仍保留在图集和审计表中。

## 任务一模型性能总览

模型训练曲线、窗口级 PR/ROC 和你要求的“三模型混淆矩阵总览图”集中在：

`../figures/model_performance/`

- `figure_1_window_classification_confusion.*`：与参考图相同的上排 PR/ROC/窗口指标、下排三个模型窗口混淆矩阵。PR/ROC 使用 corrected 输出的 `split_safe_probability`；混淆矩阵和固定窗口指标使用 corrected decoder 的 `decoded_mask`。
- `figure_0_baseline_window_classification_confusion.*`：修正前 `events_baseline` 图，仅用于逐格历史核对。
- `figure_s1_training_diagnostics_frozen_models.*`：三个已训练模型的 train/validation 曲线，未重新训练。
- `figure_2_corrected_event_detection_audit.*`：修正解码后的事件敏感度、精确率、F1、IoU、覆盖率、边界误差，以及人工事件漏检和预测候选未匹配。事件级区间匹配没有自然 TN，因此这里不把它伪装成普通 2×2 TN 混淆矩阵。

所有对应数值、来源文件哈希和分母定义见 `../figures/model_performance/performance_manifest.json`、`../figures/model_performance/source_data/` 和该目录的 `README_zh.md`。

## 任务一模型和数值结果

本仓库冻结快照中的训练、held-out 预测和模型元数据位于 `../models/`；训练曲线、PR/ROC、窗口混淆矩阵和 corrected 事件审计图位于 `../figures/model_performance/`。该 release 的主结果使用验证集选择的 corrected interval-aware decoder（各模型的 `../models/<model>/events_corrected_v1_test/`）；训练期 `RUN_SUMMARY.json` 中的 `events_baseline` 只记录旧 baseline lineage，`figure_0` 也仅用于历史核对，不能与 corrected 指标混用。

外部完整运行目录的绝对来源路径记录在 `../release_manifest.json` 和本目录的 `manifest.json` 中；这些路径属于 provenance，不要求在仓库快照中存在。

## 任务二分离

任务二多分类图不在此目录。仓库内当前正式任务二入口是 `../../task2_morphology_pg36/`，应从其 `taxonomy/`、`classification/`、`figures/` 和 `reports/` 开始阅读。外部完整 run 的 `04_label_freeze_pooled/`、`05_classification_pooled/` 等路径只作为 provenance，具体来源见任务二 `release_manifest.json`，不要把它们拼接成仓库内相对路径。
