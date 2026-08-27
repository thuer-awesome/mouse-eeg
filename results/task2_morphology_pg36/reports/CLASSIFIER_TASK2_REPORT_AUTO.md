# 冻结形态类别分类器自动摘要

分类标签：`04_label_freeze/frozen_channel_window_labels.parquet`；模型：eegwavenet_sc, physio_tf, physio_stats, eegnetv4, eegconformer；类别数：36；序列上下文：21 epoch。
事件平衡敏感性结果写入 `05_classification/model_comparison.csv`；全窗口主结果由 `evaluate_strict_classifiers_all_windows.py` 写入 `05_classification/all_window_model_comparison.csv`。
