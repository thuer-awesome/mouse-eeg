# 配置文件

`annotation_detection_final.yaml` 和 `morphology_pg36_final.yaml` 是正式结果对应的参数基线。配置只描述方法和训练参数，不包含机构原始路径；路径通过命令行传入。

任务一配置节：

- `preprocessing`：采样率、窗口、步长、带通/陷波和列名；
- `protocol`：动物集合、事件级 70/15/15、上下文边界和跨 split purge；
- `decoder`：只在验证集拟合的连续区间解码约束；
- `training` 与 `model_overrides`：三模型的优化、采样、早停和设备。

任务二 `training` 节控制五个 PG36 分类器的 epoch、batch、学习率、早停和输出子目录。显式 CLI 参数覆盖 YAML；改变正式参数时必须写入新的 run 目录并重新审计。

`model_registry.yaml` 是源码模型清单，记录每个 CLI 名称对应的构造器、输入形状、上下文和可选依赖；它不包含权重路径。权重只在具体 run 的 `models/` 或 `05_classification_pooled/` 下由训练阶段产生。
