# CLI 与 Python API

## 完整从头运行（推荐）

如果目标是从原始数据得到一套可复核的结果，直接使用下面两个入口。它们会调用仓库内的预处理、严格协议、模型构造、训练循环、推理/评估和绘图代码；默认从随机初始化开始，不要求预先提供权重：

```powershell
epilepsy-eeg run-task1 --input <raw-file-or-dir> --labels <label-file-or-dir> `
  --output <new-run> --config configs/annotation_detection_final.yaml

epilepsy-eeg run-pg36 --input <raw-file-or-dir> --labels <label-file-or-dir> `
  --output <new-run> --config configs/morphology_pg36_final.yaml
```

训练 stdout/stderr 会同时显示在终端和 `<new-run>/logs/`；源码生成的 checkpoint、`history.csv`、预测表、指标、混淆矩阵和论文图均写入该 run。`--skip-training` 是显式例外，只能在同一个 run 目录已有完整权重时使用。

两个完整入口的滤波 CLI 覆盖项为 `--highpass-hz`、`--lowpass-hz`、`--notch-hz`、`--notch-quality-factor`、`--bandpass-order`。任务一额外支持 `--eegwavenet-batch-size`、`--learning-rate-factor`、`--gradient-clip-norm`、`--normalization-clip`、`--maximum-evaluation-windows` 和 `--[no-]mixed-precision`；解析后的值会保存在 run provenance 中并传入实际构造器。

## `prepare`

`prepare` 接受一段原始记录，完成重采样、0.5–70 Hz 零相位带通、50 Hz 陷波、通道 QC 和 4 秒/1 秒窗口化。`--config` 指向 `configs/annotation_detection_final.yaml` 时，命令行仍可逐项覆盖 preprocessing 参数。输出是离线推理缓存和 `protocol/` 元数据；它不会把缺失标签当作 non-seizure。

## `build-task1-protocol`

`build-task1-protocol` 调用严格的事件分组器，生成 `full_index.parquet`、`supervised_centers.parquet`、`split_regions.parquet`、`channel_events.parquet` 和审计 JSON。训练协议必须由这个步骤或等价的已审计源流程产生，不能直接把原始推理缓存当作监督训练集。

## `annotate`

`--protocol-dir` 与 `--input` 二选一。前者用于已构建的严格协议；后者接受原始文件/目录，自动完成重采样、滤波、QC、窗口化和基线估计。`--checkpoint-root` 下应有 `<model>/model.pt`，`--decoder-root` 可提供冻结 `models/<model>/decoder_fit.json`。输出每个模型的 `inference/`、`models/<model>/predicted_events.parquet`、`decoded_tracks.parquet`、`clinical_event_atlas.pdf` 和 `run_manifest.json`。`annotate` 是已有权重的离线回放入口，不是训练入口；从头训练请使用 `run-task1`。原始输入路径模式适合离线回放；它不替代采集机实时 SDK。

## `train-task1`

支持 `--config` 以及 `--epochs`、`--batch-size`、`--physio-samples-per-epoch`、`--sampling-mode`、`--class-weight-mode`、学习率/权重衰减、早停、混合精度、`--device`、`--seed` 和 `--num-workers`。训练函数直接调用 `mouse_sleep.epilepsy.single_channel_train.SingleChannelTrainingConfig`，因此协议、权重和指标语义与冻结结果一致。YAML 中可用 `model_overrides` 为三个模型指定不同 batch size 或采样方式。

## `train-pg36`

调用仓库内 `scripts/train_pg36.py`，支持 `--config`、五个模型、epoch、batch、每事件窗口上限、学习率、早停、seed、device 和输出子目录。训练脚本会按标签表中的 `split` 列分离 train/val/test；PG36 标签路径必须显式指定，避免误用旧 45 类或二分类遗留标签。

## Python 调用

```python
from epilepsy_eeg.raw_adapter import build_protocol
from epilepsy_eeg.task1 import run_annotation

protocol = build_protocol("recording.parquet", "work/run", sampling_rate_hz=1000)
run_annotation(
    protocol_dir=protocol,
    source=None,
    labels=None,
    checkpoint_root="results/task1_annotation_detection/models",
    decoder_root="results/task1_annotation_detection",
    output_dir="work/clinical_review",
)
```

## 返回与审计

所有高层命令都写 JSON manifest；`epilepsy-eeg audit --task1-root ... --task2-root ...` 会验证模型数量、标签类别数、测试分母和关键文件是否存在。命令退出码为 0 才表示审计通过。
