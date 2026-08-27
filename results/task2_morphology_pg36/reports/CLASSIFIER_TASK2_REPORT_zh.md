# 冻结形态类别多分类器（正式审计版）

## 任务定义

分类器学习的是 `04_label_freeze/frozen_channel_window_labels.parquet` 中冻结的 **45 个探索性形态类别 G01–G45**。每个有效通道独立输入；模型没有看到动物 ID、通道 ID 或临床发作亚型。该结果衡量的是对冻结形态表示的可重复分类能力，不等同于临床亚型诊断。

统计单位需区分：窗口层 n=32,583 是描述性部署负荷，不是 32,583 只独立动物；事件层 n=1,376 是独立事件×通道观察，动物层只有 7 只。报告因此不对窗口逐行做生物学显著性推断，也不把 accuracy/AUROC 写成临床效应量。

## 数据、上下文和训练方式

- 正式分母是事件级 70/15/15 混合划分后的全部合格测试窗口：**32,583 个窗口、1,376 个事件×通道观察**。
- 每个样本保留 **21 个连续 4 s epoch** 的上下文（中心窗口 4 s）；上下文只在同一事件、同一通道内搜索，缺失位置零填充。
- 为防止长事件支配训练，另做了事件平衡敏感性运行：每个事件×通道最多 8 个代表窗口，得到 train/val/test = **24,771/5,349/5,331**。该运行不能替代全窗口测试分母。
- 训练采用类别加权 NLL；`physio_stats` 另有统计量辅助回归损失；验证集以 macro-F1 早停（patience=3）。无预训练权重。

比较的五个模型为 `eegwavenet_sc`、`physio_tf`、`physio_stats`、Braindecode 0.8.1 的 `eegnetv4` 和 `eegconformer`。每个模型目录包含 `best_model.pt`、`history.csv`、逐类指标、混淆矩阵和预测 Parquet。

训练环境记录在 `08_audit/classifier_environment.json`（项目 `.venv`，PyTorch 2.8.0+cu128、Braindecode 0.8.1）；当前 shell 的系统 Python 不代表训练环境。

## 主结果：全部合格测试窗口

主表为 `05_classification/all_window_model_comparison.csv`；`split=test_all_eligible_windows` 明确表示没有 8 窗口上限。

| 模型 | n | Accuracy | Balanced accuracy | Macro-F1 | Macro-AUPRC | Macro-AUROC | 事件 Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| EEGWaveNet-SC | 32,583 | 0.2290 | 0.1840 | 0.0861 | 0.1536 | 0.8669 | 0.1511 |
| Physio-TF | 32,583 | 0.5153 | 0.4804 | 0.3197 | 0.3899 | 0.9619 | 0.5013 |
| Physio+statistics | 32,583 | **0.5196** | **0.4820** | **0.3343** | 0.3876 | 0.9610 | **0.5079** |
| EEGNetv4 | 32,583 | 0.1433 | 0.2712 | 0.1434 | 0.1918 | 0.8852 | 0.3204 |
| EEGConformer | 32,583 | 0.2566 | 0.2912 | 0.1888 | 0.2247 | 0.9005 | 0.3090 |

事件级列来自同一文件的 `event_*` 字段：每个事件×通道先对窗口概率取均值再评估，作为临床事件层面的补充，不能与窗口层 accuracy 混为一谈。

| 模型 | 事件 Accuracy | 事件 Balanced accuracy | 事件 Macro-F1 | 事件 Macro-AUPRC | 事件 Macro-AUROC |
|---|---:|---:|---:|---:|---:|
| EEGWaveNet-SC | 0.3110 | 0.2040 | 0.1511 | 0.2780 | 0.9235 |
| Physio-TF | 0.5225 | 0.5212 | 0.5013 | 0.5425 | 0.9679 |
| Physio+statistics | **0.5451** | **0.5329** | **0.5079** | **0.5484** | **0.9687** |
| EEGNetv4 | 0.3590 | 0.3479 | 0.3204 | 0.3695 | 0.9321 |
| EEGConformer | 0.3852 | 0.3435 | 0.3090 | 0.3919 | 0.9419 |

## 敏感性结果：事件平衡上限 8

`05_classification/model_comparison.csv`（与 `model_comparison_45class.csv` 内容相同）只用于比较采样策略；它的测试 n=5,331，不是正式全测试分母。

| 模型 | n | Accuracy | Balanced accuracy | Macro-F1 | Macro-AUPRC | Macro-AUROC |
|---|---:|---:|---:|---:|---:|---:|
| EEGWaveNet-SC | 5,331 | 0.2795 | 0.1782 | 0.1273 | 0.1959 | 0.9049 |
| Physio-TF | 5,331 | 0.4963 | 0.4870 | 0.4581 | 0.5029 | 0.9664 |
| Physio+statistics | 5,331 | **0.5153** | **0.5005** | **0.4806** | **0.5063** | **0.9677** |
| EEGNetv4 | 5,331 | 0.2968 | 0.2832 | 0.2511 | 0.2653 | 0.9054 |
| EEGConformer | 5,331 | 0.3328 | 0.3019 | 0.2798 | 0.2961 | 0.9256 |

两张表的差异主要来自长事件在全窗口表中的真实权重和 45 类严重不均衡；不能选择较高的一张表作为“最终准确率”。正式报告和临床沟通应优先引用全窗口表，同时把事件平衡表标为敏感性分析。

## 如何阅读图和文件

- `05_classification/<model>/figure_model_diagnostics.*`：该模型的 train/validation loss、macro-F1 和测试混淆矩阵；看曲线是否收敛时要结合 `history.csv` 的实际 epoch 数。
- `05_classification/<model>/figure_pr_roc.*`：45 类 one-vs-rest PR/ROC；随机基线随类别支持变化，不能用 0.5 作为所有类别的统一判断线。
- `05_classification/<model>/confusion_matrix_all_windows.csv`：主结果的 45×45 混淆矩阵；`confusion_matrix.csv` 是 8 窗口敏感性运行。
- `05_classification/per_animal_channel_metrics.csv`：逐鼠逐通道审计。某分层只有单一真实类别时，AUROC/AUPRC 为 NaN 或 macro 指标不稳定，应看支持数 `n`，不能据此排名动物。

当前最稳妥的模型是 `physio_stats`（全窗口 accuracy、balanced accuracy、macro-F1 和事件 macro-F1 最高；Physio-TF 的全窗口 Macro-AUPRC 略高）。EEGWaveNet-SC/EEGNetv4/EEGConformer 的较低 F1 反映了 45 类边界和极端支持不均衡，不能据此推断它们无法区分二分类癫痫任务。

## 重要限制

1. 45 类由同一组统计/频谱形态特征发现并冻结，分类器高分只说明该表示可被波形重建，不能证明临床标签正确。
2. G04、G12、G26 为单鼠支持类别；G12 还有较多低置信度事件，必须由临床复核。
3. 绝对幅值仍是相对基线/a.u.；不能从分类结果反推 µV 阈值。
4. `05_classification/legacy_two_category_sensitivity/` 以及 `04_label_freeze/frozen_window_labels.parquet` 仅为历史二类敏感性材料，不能用于当前 45 类模型或最终临床结论。

## 待临床/统计确认

- G01–G45 与急性、非急性及具体发作类型的对应关系尚未定义，不能由模型指标自动推断。
- 若要给出动物层置信区间或显著性比较，应以动物/事件为重采样单位另行制定分析方案；当前文件只报告固定测试快照的描述性指标。
