# 分类图索引

- 每个模型的 `figure_model_diagnostics.*`：训练/验证损失与敏感性测试混淆矩阵。
- 每个模型的 `figure_pr_roc.*`：全部冻结类别的一对多 PR/ROC；饱和线标出支持数最大的八类，其余类别以淡线保留。
- 每个模型的 `figure_class_support_recall.*`：45 类完整支持量与逐类召回率。
- 每个模型的 `figure_confusion_matrix_all_windows.*`：全测试窗口归一化 45×45 混淆矩阵。
- `per_animal_channel_metrics.csv` 与 `06_figures/figure_5_per_animal_channel_metrics.*`：逐鼠逐通道全窗口 macro-F1。
- 图形源优先读取 `all_window_test_predictions.parquet`；若缺失才回退到 8 窗口敏感性结果。
