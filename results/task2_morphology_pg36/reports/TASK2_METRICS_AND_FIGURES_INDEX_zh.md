# 任务二指标与图形入口

## 正式多分类指标

正式分类使用 pooled taxonomy 的 `PG01–PG36`，但每个有效通道窗口仍是独立输入样本。测试集分母为全部合格窗口 `n=32,583`；不要把事件平衡敏感性分析的 5,331 个窗口当作正式分母。

| 模型 | Accuracy | Balanced accuracy | Macro-F1 | Macro-AUPRC | Event Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Physio + statistics | 0.6053 | 0.5928 | 0.4653 | 0.5134 | 0.6246 |
| Physio-TF | 0.5222 | 0.5817 | 0.4341 | 0.4975 | 0.6475 |
| EEGConformer | 0.2664 | 0.3649 | 0.2075 | 0.2788 | 0.4091 |
| EEGNetv4 | 0.2292 | 0.3284 | 0.1878 | 0.2460 | 0.3689 |
| EEGWaveNet-SC | 0.2189 | 0.2603 | 0.1540 | 0.2207 | 0.2467 |

机器可读原表：`05_classification_pooled/all_window_model_comparison.csv`。每模型完整指标、逐类指标和混淆矩阵分别在 `05_classification_pooled/<model>/metrics.json`、`per_class_metrics.csv` 和 `confusion_matrix_all_windows.{png,pdf,svg}`。逐动物、逐独立通道审计在 `05_classification/per_animal_channel_metrics.csv`。

低 accuracy 是 36 类长尾多分类的结果，不能按二分类的直觉解读；Physio+statistics 的窗口 Macro-F1 最高，Physio-TF 的事件聚合 Macro-F1 最高。Macro-AUROC 只能作为补充，不能单独作为临床结论。

## 聚类与 PCA 图

- 独立通道鼠内散点/PCA：`06_figures/paper_ready_morphology_pooled/01_independent_channel_pca/`。
- 同鼠 pooled 散点/PCA：`06_figures/paper_ready_morphology_pooled/02_pooled_within_mouse/`；每个点仍是一个独立 event×channel，Pxx 只是同鼠联合拟合的形态学参照。
- 全部有效通道联合 PCA（全空间和中心 zoom）：`06_figures/paper_ready_morphology_pooled/03_joint_pca/`；结构化坐标在 `03_cross_mouse/paper_ready_joint_coordinates.parquet`。
- 聚类 BIC、稳定性、3D PCA 和类别支持：`06_figures/paper_ready_morphology_pooled/06_cluster_diagnostics/`。

## 独立代表波形

为避免把两个通道误读为一个模型样本，当前正式 PG 分类的波形图集在 `06_figures/channel_independent_waveforms_v1/`：

- `per_mouse_local_categories/`：每只鼠、每个有效通道分别给出局部类别代表波形；一个 panel 只画一个通道。
- `global_categories/`：全部 PG01–PG36 × 有效通道的代表波形和 PSD inset；每个 panel 只对应一个 `PG × channel`，不把 CH7/CH8 配对或叠加。
- `source_data/*.csv`：每个代表 panel 的 event、channel、支持数和原始缓存索引。

`06_figures/combined_channel_views/` 仍保留 pooled 双通道视图，用来回答“同鼠两个有效通道是否共享原型”，不能替代上述独立通道主审计图。

## 类别数量的口径

`04_label_freeze/` 的 G01–G45 是独立通道全局探索性冻结；`04_label_freeze_pooled/` 的 PG01–PG36 是同鼠 pooled 后的当前正式分类标签。两套标签不是同一分类问题，不应在一个混淆矩阵中混用。历史 21 类只读保存在 `06_figures/historical_21_read_only/`。
