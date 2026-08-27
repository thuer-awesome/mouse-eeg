# 从原始数据到结果的完整复现

本仓库发布的是**源码优先**的科研流水线。`results/` 中的权重、预测表和图是已审计的冻结快照，用于快速核对；它们不是唯一实现，也不是完整流程的替代品。别人拿到原始 EEG 和临床标签后，可以在新的输出目录中从头执行预处理、训练、推理、解码和绘图。

## 两条完整入口

安装依赖后，在仓库根目录执行：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)

# 任务一：二分类连续打标（默认会从头训练三个模型）
epilepsy-eeg run-task1 `
  --input <raw-file-or-directory> `
  --labels <label-file-or-directory> `
  --output runs/task1_retrain_YYYYMMDD `
  --config configs/annotation_detection_final.yaml

# 任务二：PG36 探索性形态类别分类（默认会从头训练五个模型）
epilepsy-eeg run-pg36 `
  --input <raw-file-or-directory> `
  --labels <label-file-or-directory> `
  --output runs/pg36_retrain_YYYYMMDD `
  --config configs/morphology_pg36_final.yaml
```

`run-task1` 和 `run-pg36` 是全流程命令，不要求预先提供 checkpoint。只有显式指定 `--skip-training` 时，才会读取**同一个输出目录**中已经存在的权重；程序会检查权重是否齐全。`annotate` 是离线回放/医生在环推理入口，适合已有权重后的单次复核，不应与全流程命令混淆。

## 任务一阶段

```text
原始 EEG/标签
  -> raw_adapter：格式检查、CH7/CH8 解析、重采样、零相位滤波、QC
  -> continuous_single_channel：事件级 70/15/15、背景块、跨 split raw-window purge
  -> single_channel_train：三个模型从头训练、验证集早停、checkpoint/history/metrics
  -> single_channel_inference：所有有效通道窗口概率和 21-window context 推理
  -> run_corrected_event_decoder_v1：只用 validation 选择 interval-aware decoder
  -> render_task1_corrected_model_performance + render_corrected_event_atlas：指标图、混淆矩阵、临床波形 PDF
```

模型与训练实现均在仓库内：

- `src/mouse_sleep/epilepsy/single_channel_models.py`
- `src/mouse_sleep/epilepsy/single_channel_train.py`
- `src/mouse_sleep/epilepsy/single_channel_inference.py`
- `src/mouse_sleep/epilepsy/single_channel_visualization.py`
- `scripts/run_corrected_event_decoder_v1.py`
- `scripts/render_task1_corrected_model_performance.py`
- `scripts/render_corrected_event_atlas.py`

## 任务二阶段

```text
原始 EEG/标签
  -> 任务一同一套预处理和事件协议
  -> morphology_adapter：event x valid-channel 特征和物理窗口表
  -> fit_pooled_mouse_morphology：每只鼠仅用 train 事件拟合独立通道 pooled 聚类
  -> fit_pooled_global_taxonomy：跨鼠约束合并并冻结 PG01–PG36
  -> train_pg36：五个模型从头训练、验证集早停和 checkpoint/history/metrics
  -> evaluate_strict_classifiers_all_windows：全部合格 test windows 主分母评估
  -> render_task2_pg36_corrected_v2：逐鼠波形/PCA、全局质心、支持矩阵、混淆矩阵和 PR/ROC
```

任务二的模型适配、训练和评估源码在：

- `scripts/train_pg36.py`
- `scripts/evaluate_strict_classifiers_all_windows.py`
- `src/mouse_sleep/epilepsy/single_channel_models.py`
- `scripts/render_task2_pg36_corrected_v2.py`

## 日志、权重和图在哪里

每次完整运行都使用独立的 `<output>` 目录，不覆盖 `results/`：

```text
<output>/
├── config_input.yaml          # 输入 YAML 原样复制
├── config_resolved.json       # YAML + CLI 后的实际有效参数
├── command.json               # 完整命令、Python、仓库路径
├── logs/                      # 与终端同步的每阶段日志
├── preprocessing/             # 滤波缓存、QC、输入 manifest
├── protocol/                  # 任务一严格协议（任务二为 task1_protocol/）
├── models/                    # 任务一 model.pt、history、预测和指标
├── decoder/                   # 任务一连续解码和事件审计
├── figures/                   # 任务一性能图和临床 atlas
├── 01_tables/                # 任务二形态特征
├── 03_cross_mouse/           # 任务二逐鼠聚类/原型
├── 04_label_freeze_pooled/   # PG36 冻结标签
├── 05_classification_pooled/ # 任务二五模型 checkpoint/预测/指标
├── 06_figures/               # 任务二论文图
├── metrics/                  # 根级小型指标索引
├── reports/                  # RUN_REPORT_zh.md
└── run_manifest.json         # 完成状态、输入哈希和阶段清单
```

训练阶段的 stdout/stderr 会同时显示在终端并写入 `logs/03_train_*.log`（任务二为 `logs/06_train_pg36.log`）。每个模型目录含 `model.pt` 或 `best_model.pt`、`history.csv`、`metrics.json`、预测表和混淆矩阵；渲染脚本只读取这些派生结果，不会偷偷重新训练。

## 输入与隐私

原始记录可以是 Parquet/CSV/NPY/NPZ，或安装 `[edf]` 后使用 EDF/BDF/FIF。标签可以是一个 CSV/Parquet，也可以是只包含标签表的目录；多记录目录会先做唯一身份映射预检，不能唯一映射的表不会被静默复用。`protocol/label_alignment.{json,csv}` 保存每个来源文件的读取、匹配、无效时间和通道 QC 计数。输入列和时间单位见 `docs/DATA_CONTRACT_zh.md`。

原始 EEG、机构标签和完整缓存不会自动复制到 Git 仓库。run manifest 记录路径、大小和可选 SHA256，便于数据管理员在本地复核。发布派生数据或权重前仍需伦理和机构许可。

## 轻量验证与正式训练

```powershell
python -m compileall -q src scripts
python -m pytest -q

# 仅检查接口，不产生正式结果
epilepsy-eeg run-task1 --input C:\dummy.raw --labels C:\dummy.csv --output C:\planned --dry-run
epilepsy-eeg run-pg36 --taxonomy-root C:\frozen\taxonomy --output C:\planned_pg36 --dry-run
```

`--smoke` 只用于确认数据契约和模型接口（单 epoch/小评估），不能作为论文指标。正式训练请使用新的输出目录，并保留 `run_manifest.json`、日志和配置作为实验记录。

## 实时性边界

当前 `annotate` 和 `run-task1` 产生的是离线/回放、医生在环结果。零相位滤波和居中 21-window context 不能直接宣称为采集机实时部署；真正接入图机还需要因果滤波、缓存延迟、断流处理、时间同步、医生修订审计和独立前瞻验证。
