# 报告审计变更

本次只核对和整理文件，没有重跑训练。

- 正式形态类别：55 个局部原型合并为 45 个全局 G 类别；G04/G12/G26 标记为鼠特异。
- 数据：源事件 6,225；完整窗口门槛 0.99 后 194,530 个独立通道窗口、9,144 个事件×通道观察；1,381 个事件无完整窗口并保留在审计中。
- 划分：事件级 train/val/test = 4,357/934/934；合格事件×通道 = 6,393/1,375/1,376；窗口 = 130,216/31,731/32,583。跨 split 事件重叠审计为 0 对。
- 主指标：`05_classification/all_window_model_comparison.csv`，全测试窗口 n=32,583；敏感性表 `model_comparison.csv` n=5,331（每事件×通道最多 8 窗口）。
- 标签防混淆：正式输入是 `04_label_freeze/frozen_channel_window_labels.parquet`；同目录的 `frozen_window_labels.parquet` 是旧二类遗留文件，报告中已明确禁止用于 45 类训练。
- 清理：形式上已完成的 EEGNetv4 旧 `FAILED.txt` 移到 `08_audit/legacy_failed_attempts/`，保留错误文本但不再污染正式模型目录。
- 未来重跑：`run_strict_all_channel_morphology.py` 改为写入 `MORPHOLOGY_TASK2_REPORT_AUTO.md`，并使用正确的正式标签路径，避免覆盖人工审计报告。
