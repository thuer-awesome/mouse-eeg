# 合并数据训练日志

## 固定输入协议

- 7 只鼠、12 个有效通道流；E07/CH7 和 E77/CH7 排除。
- 6,412 个通道事件；测试集 957 个通道事件。
- 21-window context，4 s/epoch，1 s stride。
- 事件/背景块 70/15/15 目标划分；跨 split 原始窗口 purge 后保留重叠为 0。
- 过滤：0.5–70 Hz 四阶零相位带通 + 50 Hz notch（Q=30）。

## 已完成的 EEGWaveNet-SC

- 12 epochs；all-window sampler；batch size 1024；seed 20260822；CUDA mixed precision。
- 测试窗口：AUPRC 0.51584，AUROC 0.85753，窗口阳性比例 0.12578。
- 训练期 canonical decoder：`events_baseline`（验证集选择；未用测试集调参）。
- 测试事件：sensitivity 0.90136，precision 0.11865，F1 0.20969，coverage 0.78049；完整漏检 28，其中 >=30 s 漏检 4；FP 62.15/channel-hour。
- repaired decoder 作为保留消融：sensitivity 0.95908，但 precision/F1 降至 0.09640/0.17520，故未提升为 canonical。

## Physio 快速比较（已归档）

- `physio_sequence_no_aux` 与 `physio_sequence_stats`：1 epoch、2,048 samples/epoch、weighted smoke sampler。
- 该运行仅用于检查生理分支和统计辅助头是否能工作，不能与 EEGWaveNet 的 12-epoch 结果等价比较。
- 快速结果已移至外部历史归档；本 release 不复制该目录，来源位置由训练 run manifest 记录。

## 正式 Physio 训练与测试

- 运行配置：12 epochs、262,144 weighted event-region samples/epoch、batch size 256、seed 20260822；按验证宏平均 AUPRC 保存最佳 checkpoint。
- Physio-Sequence：best epoch 10，测试 AUPRC 0.53609，AUROC 0.89400；训练期 canonical 为 `events_baseline`。
- Physio + statistics：best epoch 11，测试 AUPRC 0.54094，AUROC 0.89266；训练期 canonical 为 `events_baseline`。
- 两条正式模型均已完成连续推理、baseline/repaired 解码和测试事件审计。

这里的 `events_baseline` 仅记录训练期 decoder lineage；当前 release 的主事件结果采用独立的 corrected interval-aware decoder，详见 `../event_atlas/` 和 `CORRECTED_DECODER_FINAL_REPORT_zh.md`。

## Epoch 与早停审计

- `epochs=12` 是本次正式运行的最大 epoch 上限，不是按合并后的窗口数机械推导的值。
- 早停已启用：每轮只用验证集宏平均动物 AUPRC 选择 checkpoint，`patience=4`；学习率调度器使用同一验证指标，`patience=2`。测试集不参与停止或调参。
- 三个模型本次都在达到 12 的上限时结束，尚未触发早停：EEGWaveNet-SC 和 Physio-Sequence 的最佳 epoch 均为 10（随后停滞 2 轮），Physio + statistics 的最佳 epoch 为 11（随后停滞 1 轮）。
- 因此当前结果应表述为“固定上限内的最佳验证 checkpoint”，不能表述为“早停已证明收敛”。合并数据本身不要求增加 epoch；若后续新批次再次正式训练，建议把上限提高到 20–24、保留 `patience=4`，并只报告新的验证最佳 checkpoint。
