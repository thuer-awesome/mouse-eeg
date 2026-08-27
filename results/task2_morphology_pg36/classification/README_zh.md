# Pooled PG 多分类输出

这是任务二的唯一正式模型输出目录。输入标签为 `04_label_freeze_pooled/pooled_channel_window_labels.parquet`，共 PG01-PG36 36 个探索性形态类别；每个有效通道仍是独立样本，21 epoch 上下文只在同一事件、同一通道内取值。Physio-TF/Physio+statistics 使用完整 21 epoch 序列；EEGWaveNet-SC、EEGNetv4 和 EEGConformer 保持论文基线的单窗口输入，序列加载器只用于统一数据契约和边界填充，不把上下文偷偷拼入这些基线。

五个完整模型：`eegwavenet_sc`、`physio_tf`、`physio_stats`、`eegnetv4`、`eegconformer`。每个模型目录包含：

- `best_model.pt`：按验证集 Macro-F1 选择的 checkpoint；
- `history.csv`：训练/验证 NLL 与 Macro-F1；
- `test_predictions.parquet`：事件平衡 capped 敏感性结果；
- `all_window_test_predictions.parquet`：全部 32,583 个正式测试窗口的预测；
- `all_window_metrics.json`：正式窗口级和事件级指标；
- `confusion_matrix_all_windows.csv`：完整 PG36×PG36 整数混淆矩阵；
- `per_class_metrics_all_windows.csv`：每个 PG 类的 support、precision、recall、F1、AP、AUROC；
- `event_level_test_predictions.parquet`：按事件×通道聚合的预测。

目录根部的 `all_window_model_comparison.csv` 是正式模型比较表，`per_animal_channel_metrics.csv` 是独立鼠/通道稳健性审计。正式分母不是 capped 敏感性样本，而是完整测试窗口 32,583 个。

该模型输出的是 PG 形态类别，不是任务一的临床发作起始/结束打标器；二分类连续解码和边界修正保留在独立任务一目录。
