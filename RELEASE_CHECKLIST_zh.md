# 发布检查清单

这份清单用于重新训练、接入新批次或把结果交给其他研究人员时的最后签字。未通过任一硬性项时，不应把结果复制到 `results/` 正式入口。

## 数据与协议

- [ ] 原始文件和标签的时间单位、起止边界和记录 ID 已核对。
- [ ] 只保留 CH7/CH8 有效通道；E07/CH7、E77/CH7 等坏道屏蔽写入配置和审计。
- [ ] 250 Hz、0.5–70 Hz 零相位带通、50 Hz 陷波（Q=30）和中位数/MAD 归一化已记录。
- [ ] 每个有效通道是独立输入；没有把 CH7/CH8 拼成 paired feature。
- [ ] 事件级 70/15/15 划分完成，重叠窗口 purge 计数为零或有书面解释。
- [ ] ADC 到微伏未确认时，图和表只写相对基线或 a.u.。

## 任务一

- [ ] 三个二分类权重、训练历史和配置签名齐全。
- [ ] corrected decoder 只用验证集选择；测试集没有参与阈值或边界调参。
- [ ] 窗口指标、事件覆盖、候选外部区间和临床 PDF 来自同一 frozen probability 版本。
- [ ] 事件 atlas 分母、正式完全可观测分母和排除记录可由表格逐行追溯。
- [ ] 结果报告明确“离线/回放、医生在环”，未宣称实时临床部署。

## 任务二

- [ ] pooled taxonomy manifest 明确 PG01–PG36、观察单位和同鼠合并约束。
- [ ] 逐鼠类别计数与全局类别 union 分开报告；不存在把 40 个鼠内原型误写成 36 个全局类。
- [ ] 五个模型均有 history、权重、全部类别混淆矩阵和 one-vs-rest PR/ROC。
- [ ] 正式测试分母从当前运行的 `all_window_model_comparison.csv` 读取，并在五个模型间一致；每事件最多 8 窗口只标为敏感性分析。
- [ ] PG36 被标为探索性形态标签，不被写成临床确认亚型。

## 发布与版本

- [ ] `epilepsy-eeg audit --task1-root ... --task2-root ...` 返回 `ok=true`。
- [ ] 正式结果树中没有 `legacy`、`smoke`、`failed` 或 `archive` 文件。
- [ ] `python -m compileall -q src scripts` 和 `python -m pytest -q` 通过。
- [ ] 新实验使用新输出目录；没有用 `--allow-existing` 覆盖不同签名的结果。
- [ ] 大文件按 Git LFS 规则处理，原始敏感数据仍只在 manifest 中引用。
