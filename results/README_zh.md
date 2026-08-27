# 正式结果入口

这里仅保留两条已经审核的结果树：

- `task1_annotation_detection/`：二分类连续预测、corrected interval-aware decoder、窗口/事件指标和临床事件图集；
- `task2_morphology_pg36/`：逐通道形态学分析、冻结 PG01–PG36、五模型分类和全部测试窗口图表。

每个子目录都有 `release_manifest.json` 和验证 JSON。旧 21 类、45 类、smoke、failed 和历史 decoder 输出不在正式入口。重新训练或重新解码请使用新的工作目录，完成审计后再生成新的 release。
