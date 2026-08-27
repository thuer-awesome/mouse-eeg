# 环境、配置与命令行

## 研究仓库应包含什么

SCI/CVPR 风格的可复现实验仓库通常不只放模型文件，还应同时提供：

- 明确的 Python 版本、依赖分组和可选 CUDA/Braindecode 依赖；
- 一个能描述正式实验的 YAML 配置，而不是把关键参数藏在脚本默认值中；
- 从原始数据到缓存、标签、划分、训练、解码和绘图的可执行入口；
- 结果清单、数据契约、分母定义、排除规则、哈希和审计脚本；
- 最小单元测试和持续集成检查，防止通道数、类别数或测试分母悄悄改变；
- 不上传原始敏感数据的 manifest，以及权重/大表/PDF 的 Git LFS 规则。

本仓库按这个约定组织。`results/` 只放两条已经审计的正式 release；旧 21 类、45 类、smoke 和失败实验不作为命令行默认输入。

## 环境安装

基础审计（不加载深度模型）：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[parquet,viz,dev]"
```

训练和 EEGNet/EEGConformer：

```powershell
python -m pip install -e ".[all,dev]"
```

GPU 机器应根据驱动选择匹配的 PyTorch CUDA wheel，并在运行前用 `python -c "import torch; print(torch.cuda.is_available())"` 检查。`environment/environment.yml` 和 `environment/requirements-*.txt` 是环境快照，不会自动替用户选择 CUDA 版本。

## 参数优先级

每个模块遵循同一规则：

`代码内安全默认值 < --config YAML < 显式命令行参数`

渲染、物化和 `audit` 不会触发训练；它们只读取已存在的权重、概率表和冻结标签。训练输出目录若已有不同签名的 `completed.json`，程序会拒绝覆盖，以免把两个实验混成一个结果。

## 任务一：预处理和严格协议

先把一段原始记录转成离线推理缓存。输入可为带时间列的 Parquet/CSV，或带 `--sampling-rate` 的 NPY/NPZ；EDF/BDF/FIF 需要 `[edf]` 依赖。

```powershell
epilepsy-eeg prepare `
  --input <recording.parquet> `
  --labels <events.parquet> `
  --output <work/prepared> `
  --config configs/annotation_detection_final.yaml
```

`prepare` 的可覆盖参数包括目标采样率、窗口/步长、0.5–70 Hz 带通、50 Hz 陷波、通道列名、动物/记录名和标签时间基准。任务一和 PG36 训练会从缓存元数据推导实际采样点数与采样率，并拒绝混合形状；不再在训练/推理阶段偷偷写死 `1000/250 Hz`。它会做有限值、饱和轨、平坦通道和 CH7/CH8 基本 QC；E07/CH7 与 E77/CH7 的正式屏蔽仍由协议配置控制。

标签参数既可以指向单个 CSV/Parquet，也可以指向包含多张标签表的目录；目录内文件会按稳定路径排序合并，再按动物和记录过滤。

### 一条命令完整从头重跑

需要让终端输出训练日志并在最后生成全部结果时，使用顶层编排器，而不是只调用 `annotate`：

```powershell
epilepsy-eeg run-task1 `
  --input <raw-file-or-directory> `
  --labels <label-file-or-directory> `
  --output runs/task1_retrain_YYYYMMDD `
  --config configs/annotation_detection_final.yaml

epilepsy-eeg run-pg36 `
  --input <raw-file-or-directory> `
  --labels <label-file-or-directory> `
  --output runs/pg36_retrain_YYYYMMDD `
  --config configs/morphology_pg36_final.yaml
```

默认会从源码构造模型并训练；`logs/` 中的日志与终端同步，权重/历史/预测/指标在 `models/` 或 `05_classification_pooled/`，性能图和临床 PDF 在 `figures/`/`06_figures/`。这两个命令不读取 `results/` 的冻结权重，除非显式使用 `--skip-training`。

完整入口也接受与 `prepare` 相同的滤波覆盖参数：`--highpass-hz`、`--lowpass-hz`、`--notch-hz`、`--notch-quality-factor` 和 `--bandpass-order`。任务一还可直接覆盖 `--eegwavenet-batch-size`、`--learning-rate-factor`、`--gradient-clip-norm`、`--normalization-clip`、`--maximum-evaluation-windows` 及 `--[no-]mixed-precision`；这些值会同时写入 `config_resolved.json` 和 `run_manifest`，并实际用于缓存/训练，而不是只作为显示参数。

若要构建训练用的事件分组、背景块和跨 split 原始窗口 purge：

```powershell
epilepsy-eeg build-task1-protocol `
  --filtered-manifest <work/prepared/complete_filtered_manifest.csv> `
  --event-components <work/prepared/protocol/event_components.parquet> `
  --output <work/task1_protocol> `
  --config configs/annotation_detection_final.yaml
```

该步骤按独立临床事件做 70/15/15 划分，并输出 `full_index.parquet`、`supervised_centers.parquet`、`training_baseline.json`、`protocol_audit.json`。动物和通道身份不进入模型特征。

## 任务一：训练、解码和临床复核

```powershell
epilepsy-eeg train-task1 `
  --protocol-dir <work/task1_protocol> `
  --output-root <work/task1_run> `
  --config configs/annotation_detection_final.yaml

epilepsy-eeg annotate `
  --input <recording.parquet> `
  --checkpoint-root results/task1_annotation_detection/models `
  --decoder-root results/task1_annotation_detection `
  --output <work/clinical_review> `
  --config configs/annotation_detection_final.yaml
```

训练配置可以分别控制三模型、batch size、学习率、权重衰减、采样模式、类别权重、混合精度、worker 数、梯度裁剪、验证集早停和最大评估窗口。任务一的 Physio 模型默认保留事件加权采样；正式全窗口分母只能使用审计过的全测试表，不能用 capped-at-8 采样结果替代。

`annotate` 当前是离线/回放和医生在环接口：输出逐通道窗口表、连续区间表和三层 PDF（EEG、概率、状态）。离线零相位滤波和居中 21 窗上下文不等于严格实时采集部署；临床接入仍需因果滤波、断流处理、延迟测试和医生修订审计。

## 任务二：PG36 形态学分类

```powershell
epilepsy-eeg train-pg36 `
  --out <task2_root> `
  --label-path results/task2_morphology_pg36/taxonomy/pooled_channel_window_labels.parquet `
  --config configs/morphology_pg36_final.yaml

epilepsy-eeg render-pg36 `
  --taxonomy-root results/task2_morphology_pg36 `
  --output <work/pg36_figures>
```

五个模型、学习率、早停、事件窗口上限和设备均可由 YAML 或命令行覆盖。每个事件×有效通道仍是独立观察，CH7/CH8 不拼成 paired input；`max_per_event=8` 只用于训练采样或敏感性分析，正式测试主分母必须从当前运行的全窗口评估表读取，不能沿用历史固定数量。

## 审计和交接

```powershell
epilepsy-eeg audit `
  --task1-root results/task1_annotation_detection `
  --task2-root results/task2_morphology_pg36 `
  --output work/release_audit.json
```

审计必须报告：任务一三个模型和 decoder 配置、任务二 PG01–PG36 和五个模型、当前运行的全窗口测试分母，以及没有 legacy/smoke/failed/archive 文件进入正式结果树。任何参数变更都应生成新的输出目录和 manifest，不直接覆盖 `results/`。
