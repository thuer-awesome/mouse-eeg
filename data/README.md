# 数据目录

原始 EEG 与机构标签不随代码仓库提交。请在本机建立 `data/raw/` 或通过 CLI 传入外部路径，并记录去标识化来源和 SHA256。推荐先运行 `epilepsy-eeg prepare --input ...` 生成 250 Hz 滤波缓存，再用 `epilepsy-eeg build-task1-protocol` 生成事件级划分和 purge 后监督协议。输入列契约见 [docs/DATA_CONTRACT_zh.md](../docs/DATA_CONTRACT_zh.md)。缺失标签不会被隐式当作 non-seizure。

`prepare` 输出的 `protocol/channel_events.parquet` 是逐通道复核表，`protocol/event_components.parquet` 是同一临床事件跨通道共享 ID 的训练协议输入。未经 `build-task1-protocol` 审计的推理缓存不能直接作为监督训练集。

最终结果的原始来源路径记录在 `data/manifests/source_locations_zh.json`；这些绝对路径只用于本地追溯，换机器时应由数据管理员更新。
