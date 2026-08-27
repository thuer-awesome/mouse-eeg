# 小鼠癫痫 EEG 双模块研究仓库

这是整理后的可复现版本，只保留两条正式主线：

- **任务一：二分类连续预测打标。** 每个有效通道独立输入，`seizure=1`、`non-seizure=0`。仓库保留 EEGWaveNet-SC、Physio-Sequence、Physio + statistics 的审计快照，同时提供同一源码下从随机初始化重训的入口；验证集拟合的 interval-aware decoder 将 4 秒物理窗口并集为连续起止区间，并生成临床复核 PDF。
- **任务二：PG36 形态学多分类。** 每个事件×有效通道是独立观察；先逐鼠聚类，再做 pooled 原型、跨鼠 PCA/UMAP、代表波形和五个分类模型。PG01–PG36 是探索性形态类别，不等于临床确认的发作亚型。

原始 EEG、大型缓存和敏感数据不默认上传。数据位置、输入列契约、哈希和正式分母均记录在 `data/manifests/` 与 `results/**/release_manifest.json`。可用 Git LFS 管理确需发布的权重、Parquet 和 PDF。

目录约定：`configs/` 是可覆盖的实验配置，`src/` 是可复用实现，`scripts/` 是论文结果脚本，`results/` 只保留两条干净 release，`docs/` 是流程/API/限制说明，`tests/` 是轻量审计测试，`environment/` 是依赖快照。

安装：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[all,dev]"
```

安装后可直接使用 `epilepsy-eeg` 命令。刚克隆的源码在尚未安装时也可以
先设置 `$env:PYTHONPATH = (Resolve-Path .\\src)`，再运行
`python -m epilepsy_eeg` 做接口检查；它与 console script 使用同一套 CLI，
不会自动读取发布权重。

命令行入口还包括两个**从原始数据到最终结果的完整流程**：`epilepsy-eeg run-task1` 和 `epilepsy-eeg run-pg36`。它们默认从源码重新初始化并训练模型，终端显示训练日志，同时把日志、配置、checkpoint、预测、指标、混淆矩阵和论文图写入新的 run 目录；不需要预先提供权重。`annotate` 只用于已有权重的离线回放，不能替代完整训练流程。其余 `prepare`、`build-task1-protocol`、`train-task1`、`render-task1`、`materialize-*`、`pg36-report`、`train-pg36`、`render-pg36` 和 `audit` 是可单独调试的阶段入口。每条命令都支持 `--help`；默认参数与 `configs/` 中的 YAML 对齐，显式命令行参数覆盖 YAML。

完整复现说明见 [`docs/REPRODUCIBILITY_zh.md`](docs/REPRODUCIBILITY_zh.md)，模型源码与权重边界见 [`docs/MODELS_zh.md`](docs/MODELS_zh.md)。

完整流程说明见 `docs/`，尤其是 `docs/CONFIGURATION_zh.md`。任务一的 corrected decoder 与任务二 PG36 分类严格分开，不能用其中一条线的结果替代另一条线。

正式结果索引在 `RESULTS_INDEX_zh.md`，仓库贡献和复现实验检查见 `CONTRIBUTING.md`、`RELEASE_CHECKLIST_zh.md`。
