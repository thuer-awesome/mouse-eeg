# 任务一 corrected 输出索引

## 当前连续起止打标图集

`figures/clinical_event_atlas/` 是任务一 corrected interval-aware decoder 的完整临床图集。每个模型目录包含人工事件覆盖、部分覆盖、低重叠、完全漏检、长事件漏检、边界偏移和标签外候选的代表图及完整 PDF 图集；图层包括原始波形、原始/平滑概率和 decoded state。

图集分母为全部 `957` 个测试人工通道事件。正式完整可观测 interval-aware 指标使用 `953` 个事件，另外 4 个形式审计排除事件仍保留在图集和审计表中。

## 任务一模型性能总览

模型训练曲线、窗口级 PR/ROC 和你要求的“三模型混淆矩阵总览图”集中在：

`figures/model_performance/`

- `figure_1_window_classification_confusion.*`：与参考图相同的上排 PR/ROC/窗口指标、下排三个模型窗口混淆矩阵。PR/ROC 使用 corrected 输出的 `split_safe_probability`；混淆矩阵和固定窗口指标使用 corrected decoder 的 `decoded_mask`。
- `figure_0_baseline_window_classification_confusion.*`：修正前 `events_baseline` 图，仅用于逐格历史核对。
- `figure_s1_training_diagnostics_frozen_models.*`：三个已训练模型的 train/validation 曲线，未重新训练。
- `figure_2_corrected_event_detection_audit.*`：修正解码后的事件敏感度、精确率、F1、IoU、覆盖率、边界误差，以及人工事件漏检和预测候选未匹配。事件级区间匹配没有自然 TN，因此这里不把它伪装成普通 2×2 TN 混淆矩阵。

所有对应数值、来源文件哈希和分母定义见 `figures/model_performance/performance_manifest.json`、`figures/model_performance/source_data/` 和该目录的 `README_zh.md`。

## 任务一模型和数值结果

合并数据的训练、held-out 预测、canonical baseline decoder 以及训练曲线/PR/ROC/混淆矩阵位于：

`../../04_model_runs/combined_old_plus_new/`

该目录的 canonical decoder 是验证集选择的 `events_baseline`。`events_repaired` 和本目录的 corrected interval-aware 输出是独立的审计/敏感性分析，不能在同一指标表中混用。

## 任务二分离

任务二多分类图不在此目录。当前正式任务二入口是：

`../../../02_morphology_taxonomy/all_mice_old_plus_new/strict_all_channel_v1/06_figures/paper_ready_morphology_pooled_v2/`
