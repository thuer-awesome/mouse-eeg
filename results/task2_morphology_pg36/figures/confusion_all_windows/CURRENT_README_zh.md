# 当前 pooled 多分类完整模型图

本目录使用 `PG01-PG36`，每个有效通道是独立样本；E77/CH7 已排除。所有图的正式分母都是完整 held-out test windows，而不是 capped event-balanced sensitivity sample。

重要对齐限制：PG36 使用自身的事件划分（6,393/1,375/1,376），与任务一严格 protocol（4,495/960/957）不同；测试键完全一致的只有 111/1,376，另有 642 条 split 不一致、591 条缺失。因此这些图是形态探索快照，不能作为任务一严格跨域泛化结果。

每个模型子目录均包含：

- `figure_confusion_matrix_all_windows`：完整 PG36 x PG36 行归一化混淆矩阵；
- `figure_class_support_recall`：36 个类别的测试支持度和召回率；
- `figure_pr_roc`：多类别 PR/ROC 汇总；

根目录的 `figure_5_per_animal_channel_metrics` 是独立 `animal/channel` 流的 Macro-F1 审计。数值来源和完整预测保存在 `../../classification/`。

任务一连续事件起止解码器不在此目录，仓库内修正版事件图位于 `../../../task1_annotation_detection/event_atlas/`；外部完整运行路径仅作为 release manifest 中的 provenance。
