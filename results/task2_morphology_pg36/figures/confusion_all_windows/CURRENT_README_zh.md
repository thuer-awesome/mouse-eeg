# 当前 pooled 多分类完整模型图

本目录使用 `PG01-PG36`，每个有效通道是独立样本；E77/CH7 已排除。所有图的正式分母都是完整 held-out test windows，而不是 capped event-balanced sensitivity sample。

每个模型子目录均包含：

- `figure_confusion_matrix_all_windows`：完整 PG36 x PG36 行归一化混淆矩阵；
- `figure_class_support_recall`：36 个类别的测试支持度和召回率；
- `figure_pr_roc`：多类别 PR/ROC 汇总；

根目录的 `figure_5_per_animal_channel_metrics` 是独立 `animal/channel` 流的 Macro-F1 审计。数值来源和完整预测保存在 `../../05_classification_pooled/`。

任务一连续事件起止解码器不在此目录，修正版事件图只从 `01_annotation_detection/06_decoder_audit_v2/corrected_combined_old_plus_new/` 读取。
