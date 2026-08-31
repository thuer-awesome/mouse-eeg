# 分类图索引

- 每个模型的 `figure_model_diagnostics.*`：训练/验证损失与敏感性测试混淆矩阵。
- 每个模型的 `figure_pr_roc.*`：全部冻结类别的一对多 PR/ROC；饱和线标出支持数最大的八类，其余类别以淡线保留。
- 每个模型的 `figure_class_support_recall.*`：PG01–PG36 这 36 个冻结类别的完整支持量与逐类召回率。
- 每个模型的 `figure_confusion_matrix_all_windows.*`：全测试窗口归一化 PG36×PG36 混淆矩阵。
- `per_animal_channel_metrics.csv` 与 `figure_5_per_animal_channel_metrics.*`：逐鼠逐通道全窗口 macro-F1。
- 图形源优先读取 `all_window_test_predictions.parquet`；若缺失才回退到 8 窗口敏感性结果。

这些图使用 PG36 自己的事件划分；测试键与任务一严格 protocol 完全一致的只有 111/1,376，因此不能把本图集解读为任务一严格跨域泛化结果。
