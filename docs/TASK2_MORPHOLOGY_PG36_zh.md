# 任务二：PG36 形态学多分类

## 标签冻结

正式标签文件为 `results/task2_morphology_pg36/taxonomy/pooled_channel_window_labels.parquet`，taxonomy version 为 `pooled_global_v1_constrained_cross_mouse`。共 7 只鼠、9,144 个事件×通道观察、194,530 个窗口、40 个 pooled mouse prototypes，约束合并后冻结 36 个全局 PG 类别。PG11、PG19、PG27 在多个鼠出现；因此逐鼠类别数相加为 40，而全局 set union 为 36。

每个有效通道独立作为观察，CH7 与 CH8 只在图中用不同颜色/标记显示，不拼成一个 paired classifier 输入。E77/CH7 沿用坏道屏蔽。低置信度事件和无支持类别均显式写入审计表。

## 图形顺序

`figures/01_per_mouse_pg_pca/` 是逐鼠独立通道散点；`02_per_mouse_pg_waveforms/` 是每个已观察 PG 行的代表波形；`03_global_pg_centroid_pca/` 恰好 36 个冻结质心点；`04_pg_support_presence/` 是 PG×鼠/通道支持矩阵；`06_classifier_pr_roc_all36/` 保留五个模型的全部 36 个 one-vs-rest PR/ROC，不把“典型类别”以外的类涂灰或删除。

## 模型与分母

五个模型为 EEGWaveNet-SC、Physio-TF、Physio + statistics、EEGNetv4、EEGConformer。`classification/all_window_model_comparison.csv` 是正式主表，全部 32,583 个合格测试窗口；`test_predictions.parquet` 和 `model_comparison.csv` 的每事件最多 8 窗口结果仅作平衡敏感性分析。模型的 Macro-F1、balanced accuracy、Macro-AUPRC、Macro-AUROC 和 PG36 混淆矩阵均按该口径解释。

当前 PG36 分类准确率并不代表临床亚型识别能力；类别是从当前数据形态统计中得到的探索性标签，必须由临床和后续独立数据验证其语义。

## 训练命令

```powershell
epilepsy-eeg train-pg36 --out <taxonomy-root> --label-path <taxonomy-root>\taxonomy\pooled_channel_window_labels.parquet --output-subdir 05_classification_pooled --epochs 10
epilepsy-eeg render-pg36 --taxonomy-root <taxonomy-root> --output <figures>
```

仓库中的训练入口会先检查 `split` 列并按事件级 train/val/test 分开读取，再开始训练；如果一个事件同时出现在多个 split 会直接报错。`results/task2_morphology_pg36/` 中的正式指标仍是已审计的冻结快照，修正后的入口不会悄悄覆盖该快照；重新训练必须写入新的输出目录并重新运行全窗口评估。
