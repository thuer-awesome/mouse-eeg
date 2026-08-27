"""Command line interface for the two reproducible epilepsy EEG workflows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .paths import MODEL_ORDER_TASK1, MODEL_ORDER_TASK2, project_root, read_yaml, write_json
from .raw_adapter import build_protocol
from .release import materialize_task1, validate_task1
from .task2 import materialize_pg36, validate_pg36


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _config(path: Path | None, section: str | None = None) -> dict[str, Any]:
    values = read_yaml(path) if path else {}
    if section is None:
        return values
    selected = values.get(section, {})
    if selected is None:
        return {}
    if not isinstance(selected, dict):
        raise ValueError(f"配置节 {section!r} 必须是 mapping")
    return dict(selected)


def _pick(cli_value: Any, values: dict[str, Any], key: str, default: Any) -> Any:
    return cli_value if cli_value is not None else values.get(key, default)


def _filter_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "highpass_hz": getattr(args, "highpass_hz", None),
        "lowpass_hz": getattr(args, "lowpass_hz", None),
        "notch_hz": getattr(args, "notch_hz", None),
        "notch_quality_factor": getattr(args, "notch_quality_factor", None),
        "bandpass_order": getattr(args, "bandpass_order", None),
    }


def _add_preprocessing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sampling-rate", type=float, help="无时间轴输入的原始采样率 Hz")
    parser.add_argument("--target-rate", type=float, help="重采样目标 Hz；默认 250")
    parser.add_argument("--window-seconds", type=float, help="物理窗口长度；默认 4 s")
    parser.add_argument("--stride-seconds", type=float, help="窗口步长；默认 1 s")
    parser.add_argument("--highpass-hz", type=float, help="离线零相位带通下限")
    parser.add_argument("--lowpass-hz", type=float, help="离线零相位带通上限")
    parser.add_argument("--notch-hz", type=float, help="陷波频率")
    parser.add_argument("--notch-quality-factor", type=float, help="陷波 Q 值")
    parser.add_argument("--bandpass-order", type=int, help="Butterworth 带通阶数")
    parser.add_argument("--ch7-column", help="表格中 CH7 的实际列名")
    parser.add_argument("--ch8-column", help="表格中 CH8 的实际列名")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="epilepsy-eeg", description="小鼠癫痫 EEG 双模块可复现研究流水线")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="原始 EEG -> 离线滤波缓存与推理协议")
    prepare.add_argument("--input", type=_path, required=True)
    prepare.add_argument("--labels", type=_path, help="可选事件标签表；只用于复核，不用于修复信号网格")
    prepare.add_argument("--output", type=_path, required=True)
    prepare.add_argument("--config", type=_path)
    prepare.add_argument("--animal-id")
    prepare.add_argument("--session-id")
    prepare.add_argument("--labels-time-base", choices=("absolute_ms", "relative_ms"), default="absolute_ms")
    _add_preprocessing_arguments(prepare)

    build1 = sub.add_parser("build-task1-protocol", help="从滤波缓存和事件组件构建泄漏安全的任务一训练协议")
    build1.add_argument("--filtered-manifest", type=_path, required=True)
    build1.add_argument("--event-components", type=_path, required=True)
    build1.add_argument("--output", type=_path, required=True)
    build1.add_argument("--config", type=_path)
    build1.add_argument("--animals", nargs="+")
    build1.add_argument("--sequence-length", type=int)
    build1.add_argument("--train-fraction", type=float)
    build1.add_argument("--validation-fraction", type=float)
    build1.add_argument("--test-fraction", type=float)
    build1.add_argument("--positive-overlap-fraction", type=float)
    build1.add_argument("--event-context-margin-seconds", type=float)
    build1.add_argument("--background-block-seconds", type=float)
    build1.add_argument("--normalization-clip", type=float)
    build1.add_argument("--seed", type=int)
    build1.add_argument("--exclude-animal-channel", action="append", default=[], metavar="ANIMAL:CHANNEL")
    build1.add_argument("--purge-cross-split-windows", action=argparse.BooleanOptionalAction, default=None)

    annotate = sub.add_parser("annotate", help="任务一：逐有效通道二分类推理并生成临床 PDF")
    annotate.add_argument("--input", type=_path, help="原始 Parquet/CSV/NPY/NPZ/EDF/BDF/FIF；与 --protocol-dir 二选一")
    annotate.add_argument("--labels", type=_path, help="可选事件标签表；需要 start/end 时间列")
    annotate.add_argument("--protocol-dir", type=_path, help="已有的完整协议目录")
    annotate.add_argument("--checkpoint-root", type=_path, required=True, help="包含三个模型子目录和 model.pt 的目录")
    annotate.add_argument("--decoder-root", type=_path, help="corrected 输出根目录；读取 models/<model>/decoder_fit.json")
    annotate.add_argument("--output", type=_path, required=True)
    annotate.add_argument("--models", nargs="+", choices=MODEL_ORDER_TASK1)
    annotate.add_argument("--config", type=_path)
    annotate.add_argument("--device")
    annotate.add_argument("--labels-time-base", choices=("absolute_ms", "relative_ms"), default="absolute_ms")
    annotate.add_argument("--page-seconds", type=float, default=120.0)
    annotate.add_argument("--allow-existing", action="store_true")
    _add_preprocessing_arguments(annotate)

    train1 = sub.add_parser("train-task1", help="任务一：按协议训练二分类模型")
    train1.add_argument("--protocol-dir", type=_path, required=True)
    train1.add_argument("--output-root", type=_path, required=True)
    train1.add_argument("--config", type=_path)
    train1.add_argument("--models", nargs="+", choices=MODEL_ORDER_TASK1)
    train1.add_argument("--epochs", type=int)
    train1.add_argument("--batch-size", type=int)
    train1.add_argument("--eegwavenet-batch-size", type=int)
    train1.add_argument("--physio-samples-per-epoch", type=int)
    train1.add_argument("--sampling-mode", choices=("all_windows", "weighted_smoke"))
    train1.add_argument("--class-weight-mode", choices=("none", "balanced", "balanced_mouse_sqrt", "hierarchical_event_region"))
    train1.add_argument("--learning-rate", type=float)
    train1.add_argument("--weight-decay", type=float)
    train1.add_argument("--early-stopping-patience", type=int)
    train1.add_argument("--learning-rate-patience", type=int)
    train1.add_argument("--learning-rate-factor", type=float)
    train1.add_argument("--gradient-clip-norm", type=float)
    train1.add_argument("--normalization-clip", type=float)
    train1.add_argument("--maximum-evaluation-windows", type=int)
    train1.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=None)
    train1.add_argument("--device")
    train1.add_argument("--seed", type=int)
    train1.add_argument("--num-workers", type=int)

    render1 = sub.add_parser("render-task1", help="任务一：从已有 corrected 输出重绘临床图集")
    render1.add_argument("--corrected-root", type=_path, required=True)
    render1.add_argument("--protocol-dir", type=_path, required=True)
    render1.add_argument("--model-run-root", type=_path, help="已有模型运行目录；默认取 corrected-root 的同级 models")
    render1.add_argument("--output", type=_path, required=True)
    render1.add_argument("--models", nargs="+", choices=MODEL_ORDER_TASK1, default=list(MODEL_ORDER_TASK1))
    render1.add_argument("--allow-existing", action="store_true")

    mat1 = sub.add_parser("materialize-task1", help="复制任务一最终结果到干净 release 目录")
    mat1.add_argument("--model-run-root", type=_path, required=True)
    mat1.add_argument("--corrected-root", type=_path, required=True)
    mat1.add_argument("--output", type=_path, required=True)
    mat1.add_argument("--include-full-traces", action="store_true")
    mat1.add_argument("--include-tiff", action="store_true")
    mat1.add_argument("--allow-existing", action="store_true")

    mat2 = sub.add_parser("materialize-task2", help="复制任务二最终 PG36 结果到干净 release 目录")
    mat2.add_argument("--source-root", type=_path, required=True)
    mat2.add_argument("--output", type=_path, required=True)
    mat2.add_argument("--include-tiff", action="store_true")
    mat2.add_argument("--without-predictions", action="store_true")
    mat2.add_argument("--allow-existing", action="store_true")

    pg = sub.add_parser("pg36-report", help="任务二：验证/复制冻结 PG36 结果")
    pg.add_argument("--source-root", type=_path, required=True)
    pg.add_argument("--output", type=_path)
    pg.add_argument("--include-tiff", action="store_true")
    pg.add_argument("--without-predictions", action="store_true")
    pg.add_argument("--allow-existing", action="store_true")

    render2 = sub.add_parser("render-pg36", help="任务二：按冻结 PG36 数据重绘全部形态学/分类图")
    render2.add_argument("--taxonomy-root", type=_path, required=True)
    render2.add_argument("--output", type=_path, required=True)
    render2.add_argument("--allow-capped-fallback", action="store_true", help="缺少全窗口预测时允许显式回退到事件平衡敏感性结果")
    render2.add_argument("--include-tiff", action="store_true")
    render2.add_argument("--allow-existing", action="store_true")

    train2 = sub.add_parser("train-pg36", help="任务二：训练五个 PG36 分类器（可配置）")
    train2.add_argument("--out", type=_path, required=True)
    train2.add_argument("--label-path", type=_path, required=True)
    train2.add_argument("--config", type=_path)
    train2.add_argument("--models", nargs="+", choices=MODEL_ORDER_TASK2)
    train2.add_argument("--epochs", type=int)
    train2.add_argument("--batch-size", type=int)
    train2.add_argument("--max-per-event", type=int)
    train2.add_argument("--learning-rate", type=float)
    train2.add_argument("--weight-decay", type=float)
    train2.add_argument("--early-stopping-patience", type=int)
    train2.add_argument("--learning-rate-patience", type=int)
    train2.add_argument("--num-workers", type=int)
    train2.add_argument("--seed", type=int)
    train2.add_argument("--device")
    train2.add_argument("--output-subdir")
    train2.add_argument("--fail-on-error", action="store_true", help="任一模型失败时返回非零")
    train2.add_argument("--allow-non36", action="store_true", help="仅调试用：允许类别数不是 36")

    run1 = sub.add_parser(
        "run-task1",
        help="任务一完整流水线：原始数据 -> 预处理/协议 -> 训练 -> 解码 -> 指标与临床图",
    )
    run1.add_argument("--input", type=_path, required=True)
    run1.add_argument("--labels", type=_path, required=True)
    run1.add_argument("--output", type=_path, required=True)
    run1.add_argument("--config", type=_path)
    run1.add_argument("--models", nargs="+", choices=MODEL_ORDER_TASK1)
    run1.add_argument("--animals", nargs="+")
    run1.add_argument("--sampling-rate", type=float)
    run1.add_argument("--target-rate", type=float)
    run1.add_argument("--window-seconds", type=float)
    run1.add_argument("--stride-seconds", type=float)
    run1.add_argument("--highpass-hz", type=float, help="离线零相位带通下限")
    run1.add_argument("--lowpass-hz", type=float, help="离线零相位带通上限")
    run1.add_argument("--notch-hz", type=float, help="陷波频率")
    run1.add_argument("--notch-quality-factor", type=float, help="陷波 Q 值")
    run1.add_argument("--bandpass-order", type=int, help="Butterworth 带通阶数")
    run1.add_argument("--labels-time-base", choices=("absolute_ms", "relative_ms"), default="absolute_ms")
    run1.add_argument("--ch7-column", help="表格中 CH7 的实际列名")
    run1.add_argument("--ch8-column", help="表格中 CH8 的实际列名")
    run1.add_argument("--device")
    run1.add_argument("--epochs", type=int)
    run1.add_argument("--batch-size", type=int)
    run1.add_argument("--eegwavenet-batch-size", type=int)
    run1.add_argument("--physio-samples-per-epoch", type=int)
    run1.add_argument("--sampling-mode", choices=("all_windows", "weighted_smoke"))
    run1.add_argument("--class-weight-mode", choices=("none", "balanced", "balanced_mouse_sqrt", "hierarchical_event_region"))
    run1.add_argument("--learning-rate", type=float)
    run1.add_argument("--weight-decay", type=float)
    run1.add_argument("--early-stopping-patience", type=int)
    run1.add_argument("--learning-rate-patience", type=int)
    run1.add_argument("--learning-rate-factor", type=float)
    run1.add_argument("--gradient-clip-norm", type=float)
    run1.add_argument("--normalization-clip", type=float)
    run1.add_argument("--maximum-evaluation-windows", type=int)
    run1.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=None)
    run1.add_argument("--num-workers", type=int)
    run1.add_argument("--seed", type=int)
    run1.add_argument("--primary-iou", type=float)
    run1.add_argument("--minimum-sensitivity", type=float)
    run1.add_argument("--minimum-precision", type=float)
    run1.add_argument("--maximum-fp-per-hour", type=float)
    run1.add_argument("--skip-training", action="store_true", help="只在已有本次 run 的权重时跳过训练")
    run1.add_argument("--skip-render", action="store_true")
    run1.add_argument("--smoke", action="store_true", help="单 epoch/小评估，用于接口验证，不是正式结果")
    run1.add_argument("--allow-existing", action="store_true")
    run1.add_argument("--no-input-hash", action="store_true")
    run1.add_argument("--dry-run", action="store_true")

    run2 = sub.add_parser(
        "run-pg36",
        help="任务二完整流水线：原始数据 -> 形态特征/PG36 -> 五模型 -> 全窗指标与图",
    )
    run2.add_argument("--input", type=_path, help="raw EEG；与 --taxonomy-root 二选一")
    run2.add_argument("--labels", type=_path, help="raw 模式的临床事件标签")
    run2.add_argument("--taxonomy-root", type=_path, help="已有冻结 taxonomy；不重新提取原始形态特征")
    run2.add_argument("--output", type=_path, required=True)
    run2.add_argument("--config", type=_path)
    run2.add_argument("--models", nargs="+", choices=MODEL_ORDER_TASK2)
    run2.add_argument("--animals", nargs="+")
    run2.add_argument("--sampling-rate", type=float)
    run2.add_argument("--target-rate", type=float)
    run2.add_argument("--window-seconds", type=float)
    run2.add_argument("--stride-seconds", type=float)
    run2.add_argument("--highpass-hz", type=float, help="离线零相位带通下限")
    run2.add_argument("--lowpass-hz", type=float, help="离线零相位带通上限")
    run2.add_argument("--notch-hz", type=float, help="陷波频率")
    run2.add_argument("--notch-quality-factor", type=float, help="陷波 Q 值")
    run2.add_argument("--bandpass-order", type=int, help="Butterworth 带通阶数")
    run2.add_argument("--labels-time-base", choices=("absolute_ms", "relative_ms"), default="absolute_ms")
    run2.add_argument("--ch7-column", help="表格中 CH7 的实际列名")
    run2.add_argument("--ch8-column", help="表格中 CH8 的实际列名")
    run2.add_argument("--device")
    run2.add_argument("--epochs", type=int)
    run2.add_argument("--batch-size", type=int)
    run2.add_argument("--max-per-event", type=int)
    run2.add_argument("--learning-rate", type=float)
    run2.add_argument("--weight-decay", type=float)
    run2.add_argument("--early-stopping-patience", type=int)
    run2.add_argument("--learning-rate-patience", type=int)
    run2.add_argument("--num-workers", type=int)
    run2.add_argument("--seed", type=int)
    run2.add_argument("--threshold", type=float)
    run2.add_argument("--expected-classes", type=int, default=36)
    run2.add_argument("--allow-non36", action="store_true", help="仅调试用：允许冻结类别数不是 36")
    run2.add_argument("--baseline-samples-per-stream", type=int, default=256)
    run2.add_argument("--skip-training", action="store_true")
    run2.add_argument("--skip-render", action="store_true")
    run2.add_argument("--smoke", action="store_true", help="单 epoch，用于接口验证，不是正式结果")
    run2.add_argument("--allow-existing", action="store_true")
    run2.add_argument("--no-input-hash", action="store_true")
    run2.add_argument("--dry-run", action="store_true")

    audit = sub.add_parser("audit", help="核对 release 的冻结标签、模型和主分母")
    audit.add_argument("--task1-root", type=_path)
    audit.add_argument("--task2-root", type=_path)
    audit.add_argument("--output", type=_path)

    return parser


def _run_train_task1(args: argparse.Namespace) -> int:
    from mouse_sleep.epilepsy.single_channel_train import SingleChannelTrainingConfig, run_single_channel_training

    values = _config(args.config, "training")
    model_values = _config(args.config, "model_overrides")
    model_names = tuple(args.models or values.get("models") or MODEL_ORDER_TASK1)
    unknown_models = set(model_names) - set(MODEL_ORDER_TASK1)
    if unknown_models:
        raise ValueError(f"未知任务一模型: {sorted(unknown_models)}")
    common_batch = _pick(args.batch_size, values, "batch_size", 256)
    default_device = _pick(args.device, values, "device", "auto")
    default_seed = _pick(args.seed, values, "seed", 20260822)
    default_workers = _pick(args.num_workers, values, "num_workers", 0)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for model in model_names:
        override = model_values.get(model, {}) if isinstance(model_values.get(model, {}), dict) else {}
        is_stats = model == "physio_sequence_stats"
        is_physio = model != "eegwavenet_sc"
        if model == "eegwavenet_sc":
            batch_size = int(_pick(override.get("batch_size"), values, "eegwavenet_batch_size", _pick(args.eegwavenet_batch_size, values, "eegwavenet_batch_size", 1024)))
        else:
            batch_size = int(_pick(override.get("batch_size"), {"batch_size": common_batch}, "batch_size", common_batch))
        samples_per_epoch = _pick(args.physio_samples_per_epoch, override, "samples_per_epoch", _pick(args.physio_samples_per_epoch, values, "physio_samples_per_epoch", 262144))
        sampling_mode = _pick(args.sampling_mode, override, "sampling_mode", _pick(args.sampling_mode, values, "sampling_mode", "weighted_smoke" if is_physio else "all_windows"))
        auxiliary_weight = float(override.get("auxiliary_statistics_weight", 0.15 if is_stats else 0.0))
        config = SingleChannelTrainingConfig(
            model_name=model,
            epochs=int(_pick(args.epochs, override, "epochs", values.get("epochs", 20))),
            batch_size=batch_size,
            samples_per_epoch=int(samples_per_epoch) if is_physio and samples_per_epoch is not None else None,
            sampling_mode=str(sampling_mode),
            class_weight_mode=str(_pick(args.class_weight_mode, override, "class_weight_mode", values.get("class_weight_mode", "hierarchical_event_region"))),
            learning_rate=float(_pick(args.learning_rate, override, "learning_rate", values.get("learning_rate", 3e-4))),
            weight_decay=float(_pick(args.weight_decay, override, "weight_decay", values.get("weight_decay", 1e-4))),
            auxiliary_statistics_weight=auxiliary_weight,
            early_stopping_patience=int(_pick(args.early_stopping_patience, override, "early_stopping_patience", values.get("early_stopping_patience", 4))),
            learning_rate_patience=int(_pick(args.learning_rate_patience, override, "learning_rate_patience", values.get("learning_rate_patience", 2))),
            learning_rate_factor=float(_pick(args.learning_rate_factor, override, "learning_rate_factor", values.get("learning_rate_factor", 0.5))),
            gradient_clip_norm=float(_pick(args.gradient_clip_norm, override, "gradient_clip_norm", values.get("gradient_clip_norm", 1.0))),
            normalization_clip=float(_pick(args.normalization_clip, override, "normalization_clip", values.get("normalization_clip", 20.0))),
            maximum_evaluation_windows=_pick(args.maximum_evaluation_windows, override, "maximum_evaluation_windows", values.get("maximum_evaluation_windows")),
            mixed_precision=bool(_pick(args.mixed_precision, override, "mixed_precision", values.get("mixed_precision", True))),
            seed=int(_pick(args.seed, override, "seed", default_seed)),
            device=str(_pick(args.device, override, "device", default_device)),
            num_workers=int(_pick(args.num_workers, override, "num_workers", default_workers)),
        )
        out = args.output_root / model
        print(f"[train-task1] {model} -> {out}", flush=True)
        run_single_channel_training(args.protocol_dir, out, config)
    return 0


def _run_render_task1(args: argparse.Namespace) -> int:
    corrected_root = args.corrected_root.resolve()
    output = args.output.resolve()
    model_root = (args.model_run_root.resolve() if args.model_run_root is not None else corrected_root.parent / "models")
    if output.exists() and any(output.iterdir()) and not args.allow_existing:
        raise FileExistsError(f"输出目录非空: {output}; 如确认覆盖请使用 --allow-existing")
    output.mkdir(parents=True, exist_ok=True)
    performance = output / "model_performance"
    performance_command = [
        sys.executable,
        str(project_root() / "scripts" / "render_task1_corrected_model_performance.py"),
        "--run-root",
        str(model_root),
        "--corrected-root",
        str(corrected_root),
        "--output-root",
        str(performance),
        "--models",
        *args.models,
        "--allow-existing",
    ]
    subprocess.run(performance_command, check=True)
    atlas = output / "clinical_event_atlas"
    atlas_command = [
        sys.executable,
        str(project_root() / "scripts" / "render_corrected_event_atlas.py"),
        "--corrected-root",
        str(corrected_root),
        "--protocol-dir",
        str(args.protocol_dir.resolve()),
        "--output-dir",
        str(atlas),
        "--models",
        *args.models,
        "--allow-existing",
    ]
    subprocess.run(atlas_command, check=True)
    return 0


def _run_train_pg36(args: argparse.Namespace) -> int:
    script = project_root() / "scripts" / "train_pg36.py"
    values = _config(args.config, "training")
    model_names = tuple(args.models or values.get("models") or MODEL_ORDER_TASK2)
    unknown_models = set(model_names) - set(MODEL_ORDER_TASK2)
    if unknown_models:
        raise ValueError(f"未知任务二模型: {sorted(unknown_models)}")
    epochs = int(_pick(args.epochs, values, "epochs", 10))
    batch_size = int(_pick(args.batch_size, values, "batch_size", 64))
    max_per_event = int(_pick(args.max_per_event, values, "max_per_event", 8))
    seed = int(_pick(args.seed, values, "seed", 20260823))
    device = str(_pick(args.device, values, "device", "auto"))
    output_subdir = str(_pick(args.output_subdir, values, "output_subdir", "05_classification_pooled"))
    learning_rate = float(_pick(args.learning_rate, values, "learning_rate", 3e-4))
    weight_decay = float(_pick(args.weight_decay, values, "weight_decay", 1e-4))
    early_stopping_patience = int(_pick(args.early_stopping_patience, values, "early_stopping_patience", 3))
    learning_rate_patience = int(_pick(args.learning_rate_patience, values, "learning_rate_patience", 1))
    num_workers = int(_pick(args.num_workers, values, "num_workers", 0))
    command = [sys.executable, str(script), "--out", str(args.out), "--label-path", str(args.label_path), "--models", *model_names, "--epochs", str(epochs), "--batch-size", str(batch_size), "--max-per-event", str(max_per_event), "--seed", str(seed), "--device", device, "--output-subdir", output_subdir, "--learning-rate", str(learning_rate), "--weight-decay", str(weight_decay), "--early-stopping-patience", str(early_stopping_patience), "--learning-rate-patience", str(learning_rate_patience), "--num-workers", str(num_workers)]
    if args.fail_on_error:
        command.append("--fail-on-error")
    if args.allow_non36:
        command.append("--allow-non36")
    subprocess.run(command, check=True)
    return 0


def _run_prepare(args: argparse.Namespace) -> int:
    values = _config(args.config, "preprocessing")
    from mouse_sleep.epilepsy.signal_processing import EEGFilterConfig

    target_rate = float(_pick(args.target_rate, values, "target_rate_hz", 250.0))
    window = float(_pick(args.window_seconds, values, "window_seconds", 4.0))
    stride = float(_pick(args.stride_seconds, values, "stride_seconds", 1.0))
    root_values = _config(args.config)
    filter_values = dict(values.get("filter", root_values.get("filter", {})))
    filter_values.update({key: value for key, value in _filter_overrides(args).items() if value is not None})
    protocol = build_protocol(
        args.input,
        args.output,
        labels=args.labels,
        animal_id=args.animal_id,
        session_id=args.session_id,
        sampling_rate_hz=args.sampling_rate,
        target_rate_hz=target_rate,
        window_seconds=window,
        stride_seconds=stride,
        labels_time_base=args.labels_time_base,
        ch7_column=args.ch7_column or values.get("ch7_column"),
        ch8_column=args.ch8_column or values.get("ch8_column"),
        filter_config=EEGFilterConfig(**filter_values) if filter_values else None,
    )
    print(protocol)
    return 0


def _run_build_task1_protocol(args: argparse.Namespace) -> int:
    from mouse_sleep.epilepsy.continuous_single_channel import (
        ContinuousSingleChannelConfig,
        build_continuous_single_channel_protocol,
    )
    from mouse_sleep.epilepsy.signal_processing import EEGFilterConfig

    values = _config(args.config, "protocol")
    animals = tuple(args.animals or values.get("animals", ()))
    if not animals:
        raise ValueError("必须通过 --animals 或 protocol.animals 指定动物集合")
    excluded_raw = args.exclude_animal_channel or values.get("excluded_animal_channels", (("E77", "CH7"),))
    excluded: list[tuple[str, str]] = []
    for item in excluded_raw:
        if isinstance(item, str):
            parts = item.replace("/", ":").split(":")
            if len(parts) != 2:
                raise ValueError(f"无效 --exclude-animal-channel: {item}")
            excluded.append((parts[0], parts[1]))
        else:
            excluded.append((str(item[0]), str(item[1])))
    filter_values = dict(values.get("filter", {}))
    filter_config = EEGFilterConfig(**filter_values) if filter_values else EEGFilterConfig()
    config = ContinuousSingleChannelConfig(
        animals=animals,
        train_fraction=float(_pick(args.train_fraction, values, "train_fraction", 0.70)),
        validation_fraction=float(_pick(args.validation_fraction, values, "validation_fraction", 0.15)),
        test_fraction=float(_pick(args.test_fraction, values, "test_fraction", 0.15)),
        sequence_length=int(_pick(args.sequence_length, values, "sequence_length", 21)),
        positive_overlap_fraction=float(_pick(args.positive_overlap_fraction, values, "positive_overlap_fraction", 0.50)),
        event_context_margin_seconds=float(_pick(args.event_context_margin_seconds, values, "event_context_margin_seconds", 12.0)),
        background_block_seconds=float(_pick(args.background_block_seconds, values, "background_block_seconds", 300.0)),
        normalization_clip=float(_pick(args.normalization_clip, values, "normalization_clip", 20.0)),
        random_seed=int(_pick(args.seed, values, "random_seed", 20260814)),
        excluded_animal_channels=tuple(excluded),
        purge_cross_split_windows=bool(_pick(args.purge_cross_split_windows, values, "purge_cross_split_windows", True)),
        filter_config=filter_config,
    )
    result = build_continuous_single_channel_protocol(args.filtered_manifest, args.event_components, args.output, config)
    print(result)
    return 0


def _run_render_pg36(args: argparse.Namespace) -> int:
    if args.output.exists() and any(args.output.iterdir()) and not args.allow_existing:
        raise FileExistsError(f"输出目录非空: {args.output}; 如确认覆盖请使用 --allow-existing")
    taxonomy_root = args.taxonomy_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    morphology = output / "paper_ready_morphology_pooled"
    available_models = [model for model in MODEL_ORDER_TASK2 if ((taxonomy_root / "classification" if (taxonomy_root / "taxonomy").is_dir() else taxonomy_root / "05_classification_pooled") / model).is_dir()]
    if not available_models:
        available_models = list(MODEL_ORDER_TASK2)
    subprocess.run(
        [
            sys.executable,
            str(project_root() / "scripts" / "render_task2_pg36_corrected_v2.py"),
            "--taxonomy-root",
            str(taxonomy_root),
            "--output-dir",
            str(morphology),
            "--models",
            *available_models,
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(project_root() / "scripts" / "render_pooled_cluster_diagnostics.py"),
            "--root",
            str(taxonomy_root),
            "--figure-dir",
            str(morphology),
        ],
        check=True,
    )
    classifier = output / "paper_ready_classifier_pooled"
    subprocess.run(
        [
            sys.executable,
            str(project_root() / "scripts" / "render_pooled_split_audit.py"),
            "--root",
            str(taxonomy_root),
            "--figure-dir",
            str(classifier),
        ],
        check=True,
    )
    if (taxonomy_root / "taxonomy").is_dir():
        label_path = taxonomy_root / "taxonomy" / "pooled_channel_window_labels.parquet"
        class_dir = taxonomy_root / "classification"
    else:
        label_path = taxonomy_root / "04_label_freeze_pooled" / "pooled_channel_window_labels.parquet"
        class_dir = taxonomy_root / "05_classification_pooled"
    subprocess.run(
        [
            sys.executable,
            str(project_root() / "scripts" / "render_pg36_classifier_summary.py"),
            "--class-dir",
            str(class_dir),
            "--figure-dir",
            str(classifier),
            "--label-prefix",
            "PG",
            "--allow-existing",
        ],
        check=True,
    )
    full = output / "strict_classifier_full_pooled"
    full_command = [
        sys.executable,
        str(project_root() / "scripts" / "render_pg36_classifier_full.py"),
        "--class-dir",
        str(class_dir),
        "--label-path",
        str(label_path),
        "--figure-root",
        str(full),
        "--label-prefix",
        "PG",
    ]
    if args.allow_capped_fallback:
        full_command.append("--allow-capped-fallback")
    elif not all((class_dir / model / "all_window_test_predictions.parquet").is_file() for model in MODEL_ORDER_TASK2 if (class_dir / model).is_dir()):
        full_command.append("--require-full-window")
    if args.allow_existing:
        full_command.append("--allow-existing")
    subprocess.run(full_command, check=True)
    return 0


def _run_audit(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {"schema_version": 1, "ok": True, "checks": {}}
    if args.task1_root:
        try:
            report["checks"]["task1"] = validate_task1(args.task1_root)
        except Exception as exc:
            report["ok"] = False; report["checks"]["task1"] = {"ok": False, "error": repr(exc)}
    if args.task2_root:
        try:
            report["checks"]["task2"] = validate_pg36(args.task2_root)
        except Exception as exc:
            report["ok"] = False; report["checks"]["task2"] = {"ok": False, "error": repr(exc)}
    if not report["checks"]:
        raise ValueError("audit 至少需要 --task1-root 或 --task2-root")
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        return _run_prepare(args)
    if args.command == "build-task1-protocol":
        return _run_build_task1_protocol(args)
    if args.command == "annotate":
        from .task1 import run_annotation
        config_values = _config(args.config)
        preprocessing = dict(config_values.get("preprocessing", {}))
        model_names = tuple(args.models or config_values.get("models") or MODEL_ORDER_TASK1)
        unknown_models = set(model_names) - set(MODEL_ORDER_TASK1)
        if unknown_models:
            raise ValueError(f"未知任务一模型: {sorted(unknown_models)}")
        run_annotation(protocol_dir=args.protocol_dir, source=args.input, labels=args.labels, checkpoint_root=args.checkpoint_root, output_dir=args.output, models=model_names, decoder_root=args.decoder_root, device=args.device or str(preprocessing.get("device", "auto")), sampling_rate_hz=args.sampling_rate, labels_time_base=args.labels_time_base, page_seconds=args.page_seconds, allow_existing=args.allow_existing, config_path=args.config, target_rate_hz=args.target_rate, window_seconds=args.window_seconds, stride_seconds=args.stride_seconds, ch7_column=args.ch7_column, ch8_column=args.ch8_column, filter_overrides=_filter_overrides(args))
        return 0
    if args.command == "train-task1":
        return _run_train_task1(args)
    if args.command == "render-task1":
        return _run_render_task1(args)
    if args.command == "materialize-task1":
        materialize_task1(args.model_run_root, args.corrected_root, args.output, include_full_traces=args.include_full_traces, include_tiff=args.include_tiff, allow_existing=args.allow_existing); return 0
    if args.command in {"materialize-task2", "pg36-report"}:
        output = args.output if args.command == "materialize-task2" else args.output
        if output is None:
            validate_pg36(args.source_root); return 0
        materialize_pg36(args.source_root, output, include_tiff=args.include_tiff, include_predictions=not args.without_predictions, allow_existing=args.allow_existing); validate_pg36(output); return 0
    if args.command == "train-pg36":
        return _run_train_pg36(args)
    if args.command == "run-task1":
        from .pipeline import run_task1_pipeline

        run_task1_pipeline(
                input_path=args.input,
                labels_path=args.labels,
                output_root=args.output,
                config_path=args.config,
                models=tuple(args.models or MODEL_ORDER_TASK1),
                animals=args.animals,
                sampling_rate_hz=args.sampling_rate,
                target_rate_hz=args.target_rate,
                window_seconds=args.window_seconds,
                stride_seconds=args.stride_seconds,
                labels_time_base=args.labels_time_base,
                ch7_column=args.ch7_column,
                ch8_column=args.ch8_column,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                eegwavenet_batch_size=args.eegwavenet_batch_size,
                physio_samples_per_epoch=args.physio_samples_per_epoch,
                sampling_mode=args.sampling_mode,
                class_weight_mode=args.class_weight_mode,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                early_stopping_patience=args.early_stopping_patience,
                learning_rate_patience=args.learning_rate_patience,
                learning_rate_factor=args.learning_rate_factor,
                gradient_clip_norm=args.gradient_clip_norm,
                normalization_clip=args.normalization_clip,
                maximum_evaluation_windows=args.maximum_evaluation_windows,
                mixed_precision=args.mixed_precision,
                num_workers=args.num_workers,
                seed=args.seed,
                filter_overrides=_filter_overrides(args),
                primary_iou=args.primary_iou,
                minimum_sensitivity=args.minimum_sensitivity,
                minimum_precision=args.minimum_precision,
                maximum_fp_per_hour=args.maximum_fp_per_hour,
                skip_training=args.skip_training,
                skip_render=args.skip_render,
                smoke=args.smoke,
                allow_existing=args.allow_existing,
                hash_inputs=not args.no_input_hash,
                command=sys.argv,
                dry_run=args.dry_run,
        )
        return 0
    if args.command == "run-pg36":
        from .pipeline import run_pg36_pipeline

        run_pg36_pipeline(
            output_root=args.output,
            input_path=args.input,
            labels_path=args.labels,
            taxonomy_root=args.taxonomy_root,
            config_path=args.config,
            models=tuple(args.models or MODEL_ORDER_TASK2),
            animals=args.animals,
            sampling_rate_hz=args.sampling_rate,
            target_rate_hz=args.target_rate,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
            labels_time_base=args.labels_time_base,
            ch7_column=args.ch7_column,
            ch8_column=args.ch8_column,
            filter_overrides=_filter_overrides(args),
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_per_event=args.max_per_event,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.early_stopping_patience,
            learning_rate_patience=args.learning_rate_patience,
            num_workers=args.num_workers,
            seed=args.seed,
            threshold=args.threshold,
            expected_classes=args.expected_classes,
            allow_non36=args.allow_non36,
            baseline_samples_per_stream=args.baseline_samples_per_stream,
            skip_training=args.skip_training,
            skip_render=args.skip_render,
            smoke=args.smoke,
            allow_existing=args.allow_existing,
            hash_inputs=not args.no_input_hash,
            command=sys.argv,
            dry_run=args.dry_run,
        )
        return 0
    if args.command == "render-pg36":
        return _run_render_pg36(args)
    if args.command == "audit":
        return _run_audit(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
