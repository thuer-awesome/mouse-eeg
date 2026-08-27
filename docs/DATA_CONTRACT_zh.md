# 数据契约

## 原始输入

表格输入必须包含时间列 `timestamp_ms`（或显式提供采样率）以及 CH7/CH8 数值列。数组输入为二维 `[channels, samples]` 或 `[samples, channels]`，必须通过 `--sampling-rate` 指定采样率。EDF/BDF/FIF 需要可选依赖并且通道名必须能解析为 CH7/CH8。

标签输入可以是一个 CSV/Parquet 文件，也可以是只包含标签表的目录。每行必须包含 `start_time_ms`/`end_time_ms`，可选 `channel_name`、`status`、`event_id`、`animal_id` 和 `session_id`。没有通道列时，区间会复制到每个通过 QC 的通道；`--labels-time-base relative_ms` 可将相对记录起点的坐标转换为绝对毫秒。空标签不是 non-seizure 标签，只表示没有人工参考。

多原始记录运行时，标签表必须能唯一映射到记录：推荐提供 `animal_id`/`session_id` 列，或使用含唯一 `E##` 动物标识的文件名。无法唯一映射的无身份标签目录会在预处理前直接报错，不会静默复制到每只鼠；映射行数、无效时间行和通道 QC 丢弃行写入 `protocol/label_alignment.{json,csv}`。

## 规范化缓存

适配器输出 `protocol/full_index.parquet`、`protocol/channel_events.parquet`、`protocol/event_components.parquet`、`protocol/training_baseline.json`、`protocol/split_regions.parquet` 和 `cache/shards/*`。每行带有动物、记录、通道、窗口起止、连续性 ID、缓存路径和 QC 状态。`event_components.parquet` 将同一临床事件在 CH7/CH8 的观察合并为共享事件 ID，供严格协议构建器做事件级划分；任何跨连续性或跨 split 的上下文都不得借用。

## 隐私与发布

原始记录、完整缓存和含敏感时间轴的表默认不进 Git。`data/manifests/` 记录机构保存位置、文件大小和 SHA256；需要公开时使用去标识化派生数据或 Git LFS，并在获得伦理/机构许可后再推送。
