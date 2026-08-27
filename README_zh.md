# 小鼠癫痫 EEG 双模块研究仓库

这是整理后的可复现版本，只保留两条正式主线：

- **任务一：二分类连续预测打标。** 每个有效通道独立输入，`seizure=1`、`non-seizure=0`。保留 EEGWaveNet-SC、Physio-Sequence、Physio + statistics 三个冻结权重；验证集拟合的 interval-aware decoder 将 4 秒物理窗口并集为连续起止区间，并生成临床复核 PDF。
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

命令行入口：`epilepsy-eeg prepare`、`build-task1-protocol`、`annotate`、`train-task1`、`render-task1`、`materialize-task1`、`materialize-task2`、`pg36-report`、`train-pg36`、`render-pg36` 和 `audit`。每条命令都支持 `--help`；默认参数与 `configs/` 中的 YAML 对齐，显式命令行参数覆盖 YAML。`prepare` 负责预处理缓存，`build-task1-protocol` 负责事件级划分和跨 split purge。

完整流程说明见 `docs/`，尤其是 `docs/CONFIGURATION_zh.md`。任务一的 corrected decoder 与任务二 PG36 分类严格分开，不能用其中一条线的结果替代另一条线。
