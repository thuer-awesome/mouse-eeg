# 模型与源码清单

所有用于正式结果的模型结构、损失、优化器和数据集代码都在本仓库。Git LFS 中的权重只是一个可核对的训练快照；不依赖预训练权重，完整入口会重新初始化模型并从训练集开始优化。

## 任务一：二分类连续打标

| CLI 名称 | 源码构造 | 训练入口 | 输入 | 输出 |
|---|---|---|---|---|
| `eegwavenet_sc` | `mouse_sleep.epilepsy.single_channel_models.build_single_channel_model` → `EEGWaveNetSC` | `single_channel_train.run_single_channel_training` | 一个有效通道的 4 s 窗口；不拼接 CH7/CH8 | `seizure`/`non-seizure` log-probability |
| `physio_sequence_no_aux` | `PhysioSequenceNet`，时域多尺度卷积 + STFT 频域 + 跨模态注意力 + epoch TCN + Transformer | 同上 | 独立通道的居中 21-window context | 二分类概率 |
| `physio_sequence_stats` | 同一 Physio 主干，增加可微统计特征输入和训练期生理回归辅助头 | 同上 | 同上 + 统计特征 | 二分类概率 + 训练期辅助输出 |

主要实现文件：

- `src/mouse_sleep/epilepsy/single_channel_models.py`：网络层、输入校验和构造器；包含 EEGWaveNet paper-faithful wrapper 与 Physio 分支。
- `src/mouse_sleep/epilepsy/single_channel_train.py`：数据集、类别权重、训练循环、早停、ReduceLROnPlateau、梯度裁剪、混合精度、最佳 checkpoint 和 history。
- `src/mouse_sleep/epilepsy/single_channel_inference.py`：连续窗口/上下文推理和概率表。
- `src/mouse_sleep/epilepsy/continuous_single_channel.py`：事件级 split、背景块和 raw-window purge。
- `src/mouse_sleep/epilepsy/seizure_annotation.py`、`scripts/run_corrected_event_decoder_v1.py`：validation-only interval-aware 解码。

任务一从头训练时，模型权重由 `build_single_channel_model` 初始化；`results/task1_annotation_detection/models/*/model.pt` 只是在相同源码和配置下保存的冻结实例。

## 任务二：PG36 探索性形态分类

| CLI 名称 | 实现 |
|---|---|
| `eegwavenet_sc` | 复用独立通道 EEGWaveNet |
| `physio_tf` | `PhysioSequenceNet(include_statistics_input=False, auxiliary_statistics=False)` |
| `physio_stats` | `PhysioSequenceNet(include_statistics_input=True, auxiliary_statistics=True)` |
| `eegnetv4` | `braindecode.models.EEGNetv4`，由 `scripts/train_pg36.py::BraindecodeAdapter` 适配单通道 4 s 输入 |
| `eegconformer` | `braindecode.models.EEGConformer`，由同一适配器构造；配置启用 log-softmax 输出 |

任务二完整训练脚本是 `scripts/train_pg36.py`，全测试窗口评估是 `scripts/evaluate_strict_classifiers_all_windows.py`。`max_per_event=8` 只限制训练/敏感性采样，正式评估重新读取所有合格 `split=test` 窗口。

## 为什么仓库同时保留权重

权重用于三种用途：

1. 快速复核发布结果，不必等待完整训练；
2. 回归测试网络输出、指标和图形是否发生漂移；
3. 在 `annotate` 离线回放中复现临床事件图。

它们不替代训练源码。要复现实验或改变数据/参数，使用 `run-task1`/`run-pg36`，不要把 `annotate` 当成训练入口。

## 依赖与版本

基础预处理不需要 PyTorch；从头训练任务一需要 `[deep]`，任务二的 EEGNetv4/EEGConformer 需要 `[pg36]`（Braindecode）。推荐 Python 3.11，并按目标 CUDA 驱动安装匹配的 PyTorch wheel。依赖分组和版本范围见 `pyproject.toml` 与 `environment/`。
