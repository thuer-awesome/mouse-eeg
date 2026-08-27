# 任务一模型性能与 corrected 事件解码

这是任务一 corrected 输出的唯一模型性能入口。三类证据必须分开解释：

1. `figure_1_window_classification_confusion.*`：与用户示例同一结构。上排是 held-out PR、ROC 和窗口指标；下排是三个模型的二分类混淆矩阵。PR/ROC 使用 corrected 输出的 `split_safe_probability`，固定窗口指标和混淆矩阵使用 corrected interval-aware decoder 的 `decoded_mask`。
2. `figure_0_baseline_window_classification_confusion.*`：修正前 `events_baseline` 的历史参考图，保留用于逐格核对，不作为 corrected 主结果。
3. `figure_2_corrected_event_detection_audit.*`：corrected interval-aware decoder 的事件级结果。上排是事件敏感度/精确率/F1、IoU/覆盖和边界误差；下排分别展示人工事件的检测/漏检，以及预测候选的匹配/未匹配。两行分母不同，所以它不是带 TN 的普通混淆矩阵。
4. `figure_s1_training_diagnostics_frozen_models.*`：同一批已训练权重的训练/验证曲线。解码修正是后处理，不会重新生成训练曲线。

为什么没有“corrected 事件级 2x2 TN 混淆矩阵”：连续事件检测中 TP、FP 和 FN 可由区间匹配定义，但 TN 取决于人为选择的时间离散单位。若改用 4 秒窗口定义 TN，就退化为 `figure_1` 的窗口级混淆矩阵；不能把两种分母混在一起。

所有精确数值和来源哈希位于 `source_data/` 与 `performance_manifest.json`。完整逐事件波形证据仍在相邻目录 `../clinical_event_atlas/`。
