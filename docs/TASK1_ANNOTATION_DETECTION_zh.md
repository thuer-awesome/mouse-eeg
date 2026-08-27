# 任务一：二分类连续预测打标

## 正式输入

训练协议来自旧数据与新批次的合并缓存：7 只鼠、12 个有效通道流，E07/CH7 和 E77/CH7 排除；6,412 个通道事件；训练/验证/测试事件目标比例 70/15/15；跨 split 原始 4 秒窗 purge。输入通道独立，模型看不到动物名和通道名。

过滤为 0.5–70 Hz 零相位带通 + 50 Hz 陷波，采样率 250 Hz。每个窗口 4 秒、步长 1 秒；Physio 模型使用居中 21 窗上下文。幅值单位为相对基线/a.u.。

## 冻结模型与结果

三个权重位于 `results/task1_annotation_detection/models/<model>/model.pt`。窗口级 corrected 表、混淆矩阵、PR/ROC 和训练诊断在 `figures/model_performance/`。事件级 corrected 结果在各模型的 `events_corrected_v1_test/`；临床图集在 `event_atlas/`。

测试窗口 179,403 的 corrected 窗口指标：EEGWaveNet-SC AUPRC 0.5158/AUROC 0.8575/F1 0.5051；Physio-Sequence 0.5361/0.8940/0.5503；Physio + statistics 0.5409/0.8927/0.5609。事件级图集分母为 957 个测试通道事件；正式 interval-aware 指标分母为 953 个完全可观测事件，4 个形式审计排除行仍在图集中保留。

事件 precision 的分母是模型产生的所有候选区间，不是人工事件数。因此“检出约八百个人工事件”与 precision 较低并不矛盾：候选外部区间较多，且外部候选仍可能是漏标或边界差异，需临床复核。

## 训练与解码

```powershell
epilepsy-eeg train-task1 --protocol-dir <protocol> --output-root <run> --models eegwavenet_sc physio_sequence_no_aux physio_sequence_stats --epochs 20 --early-stopping-patience 4
epilepsy-eeg annotate --protocol-dir <protocol> --checkpoint-root <run> --decoder-root <corrected> --output <review>
```

`train-task1` 的 early stopping 只看验证集指标；当前冻结结果是在 12 epoch 上限内的最佳 checkpoint，不应写成“已证明收敛”。`annotate` 可以直接接受原始文件，自动生成 label-independent cache；没有标签时只输出候选区间，不能计算灵敏度/精确率。

## 临床边界

当前 PDF 是离线/回放和医生在环原型。离线零相位滤波不能直接用于严格实时；Physio 居中上下文还引入约 14 秒等待，且仓库没有采集机 SDK、LSL/TCP 输入、断流处理或医生修订审计。因此不能宣称无人值守临床部署。
