# 完整流程

> **可复现入口**：需要从原始数据重新得到训练日志、权重、预测、指标和图时，使用 `epilepsy-eeg run-task1` 或 `epilepsy-eeg run-pg36`。这两个命令默认执行全链路和从头训练；`annotate` 只用于已有权重的离线回放。实现清单见 `docs/MODELS_zh.md`，运行契约见 `docs/REPRODUCIBILITY_zh.md`。

## 1. 数据与通道

输入是原始 Parquet/CSV/NPY/NPZ 或可选 EDF/BDF/FIF；临床标签可以是单个 CSV/Parquet，也可以是包含多张标签表的目录。正式研究只接受 CH7、CH8；坏道按动物-通道配置屏蔽（当前 E07/CH7、E77/CH7）。ADC 到微伏的绝对标定尚未确认，所有幅值图和统计量以相对基线或 a.u. 表示。

## 2. 预处理

离线科研流程先把连续信号重采样到 250 Hz，再做四阶 0.5–70 Hz 零相位带通和 50 Hz 陷波（Q=30）。随后执行有限值、饱和轨、平坦线和连续性检查，按通道计算中位数/MAD 基线。标签不参与信号网格构造，避免用标签边界“修复”波形。

## 3. 任务一

临床区间先按独立事件分组，目标 70/15/15 混合划分；4 秒窗口、1 秒步长，跨 split 的原始窗口重叠 purge。三模型共享通道盲输入，不把动物或通道身份作为特征。训练完成后只在验证集选择一个全局 decoder；测试集只做一次评估。

模型概率以物理 4 秒窗支持，连续阳性窗取并集并应用中位数平滑、滞回、持续性和最短事件约束。输出包括窗口表、预测事件表、人工事件覆盖表、候选外部区间和三层 PDF（原始 EEG、概率、decoded state）。

## 4. 任务二

每个事件×有效通道独立提取时域、频域和基线相对特征。先在每只鼠内拟合局部 GMM/原型，再以约束 complete-linkage 在 pooled 原型空间合并，冻结 PG01–PG36。跨鼠 PCA/UMAP 只使用冻结原型和独立通道点；每个 PG 类别在全局质心图中恰好一个特征点。五个分类器的正式分母为当前运行 `all_window_model_comparison.csv` 中记录的全部合格测试窗口；最多 8 窗口/事件只作为敏感性分析，不能写死为某个历史快照数量。

## 5. 结果审计

优先阅读 `results/task1_annotation_detection/metrics/` 和 `results/task1_annotation_detection/figures/model_performance/`，再阅读 `event_atlas/`。任务二的仓库冻结快照先看 `results/task2_morphology_pg36/taxonomy/pooled_label_freeze_manifest.json`、`tables/`、`figures/` 和 `classification/all_window_model_comparison.csv`。如果检查外部完整 run，则对应阅读 `04_label_freeze_pooled/`、`05_classification_pooled/`、`06_figures/` 和 `07_reports/`；不要把两种目录布局混用。任何旧 45 类、历史 21 类或旧 decoder 目录都不属于正式入口。
