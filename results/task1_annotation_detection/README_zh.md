# 任务一：二分类连续预测打标

本目录对应 corrected interval-aware decoder。三个模型共享验证集拟合的全局解码参数，测试集不参与选择。每个有效通道独立输入；`event_atlas/` 保存完整人工事件、模型候选和波形/概率/状态 PDF；`figures/model_performance/` 保存窗口级混淆矩阵、PR/ROC、训练诊断和事件级汇总。外部候选只是临床复核对象，不自动等同于误报。
