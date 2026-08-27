"""Top-level, provenance-preserving orchestration for both research tasks.

The lower-level commands remain useful for debugging and for rerunning one
stage.  This module provides the two public end-to-end entry points used by
``epilepsy-eeg run-task1`` and ``epilepsy-eeg run-pg36``.  A run is deliberately
isolated from the frozen ``results/`` releases and contains its own resolved
configuration, command record, stage logs, checkpoints, predictions, figures
and manifest.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from .paths import MODEL_ORDER_TASK1, MODEL_ORDER_TASK2, project_root, read_yaml, write_json
from .raw_adapter import build_protocol, discover_files, discover_label_files


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


class _Tee:
    """Write a stage log and keep the same output visible in the terminal."""

    def __init__(self, log: Any, console: Any) -> None:
        self.log = log
        self.console = console

    def write(self, value: str) -> int:
        self.log.write(value)
        self.log.flush()
        _write_console(self.console, value)
        return len(value)

    def flush(self) -> None:
        self.log.flush()
        self.console.flush()


def _write_console(stream: Any, value: str) -> None:
    """Write text without letting a legacy Windows code page abort a run."""

    try:
        stream.write(value)
    except UnicodeEncodeError:
        # The stage log is UTF-8 and keeps the original text.  Only the live
        # console view is downgraded when its active code page cannot encode a
        # character (for example ``gbk`` on a Chinese Windows terminal).
        encoding = getattr(stream, "encoding", None) or "utf-8"
        fallback = value.encode(encoding, errors="replace").decode(
            encoding, errors="replace"
        )
        stream.write(fallback)
    stream.flush()


@dataclass
class StageRecord:
    name: str
    command: list[str]
    log: str
    started_at_utc: str
    finished_at_utc: str | None = None
    status: str = "running"
    error: str | None = None


def _ensure_output_root(path: Path, allow_existing: bool) -> None:
    path = path.resolve()
    if path.exists() and any(path.iterdir()) and not allow_existing:
        raise FileExistsError(
            f"输出目录非空: {path}; 为避免混合两个实验，请换新目录或使用 --allow-existing"
        )
    path.mkdir(parents=True, exist_ok=True)


def _source_inventory(
    source: str | Path,
    *,
    hash_files: bool = True,
    exclude: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Record source files without copying sensitive raw recordings."""

    files = discover_files(source)
    if exclude is not None:
        excluded = {path.resolve() for path in discover_label_files(exclude)}
        files = [path for path in files if path.resolve() not in excluded]
    rows: list[dict[str, Any]] = []
    for path in files:
        item: dict[str, Any] = {
            "path": str(path.resolve()),
            "name": path.name,
            "suffix": path.suffix.lower(),
            "bytes": int(path.stat().st_size),
        }
        if hash_files:
            item["sha256"] = _sha256(path)
        rows.append(item)
    return rows


def _label_inventory(source: str | Path, *, hash_files: bool = True) -> list[dict[str, Any]]:
    """Record label file provenance without copying clinical annotations."""

    rows: list[dict[str, Any]] = []
    for path in discover_label_files(source):
        item: dict[str, Any] = {
            "path": str(path.resolve()),
            "name": path.name,
            "suffix": path.suffix.lower(),
            "bytes": int(path.stat().st_size),
        }
        if hash_files:
            item["sha256"] = _sha256(path)
        rows.append(item)
    return rows


def _parse_excluded(values: Iterable[Any] | None, default: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    """Normalize YAML/CLI ``ANIMAL:CHANNEL`` exclusion forms."""

    source = default if values is None else values
    result: list[tuple[str, str]] = []
    for value in source:
        if isinstance(value, str):
            parts = value.replace("/", ":").split(":")
            if len(parts) != 2:
                raise ValueError(f"无效通道排除项: {value}")
            result.append((parts[0].upper(), parts[1].upper()))
        else:
            if len(value) != 2:
                raise ValueError(f"无效通道排除项: {value}")
            result.append((str(value[0]).upper(), str(value[1]).upper()))
    return tuple(result)


_FILTER_KEYS = (
    "highpass_hz",
    "lowpass_hz",
    "notch_hz",
    "notch_quality_factor",
    "bandpass_order",
)


def _resolve_filter_values(
    base: Mapping[str, Any] | None,
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge YAML and CLI filter settings with explicit ``None`` handling.

    The full-run commands expose the same five filter controls as ``prepare``.
    Keeping the merge in one place prevents a provenance/configuration drift
    where a value is shown in ``config_resolved.json`` but not used to create
    the filtered cache.
    """

    result = dict(base or {})
    for key in _FILTER_KEYS:
        value = (overrides or {}).get(key)
        if value is not None:
            result[key] = value
    return result


def _script(name: str) -> Path:
    path = project_root() / "scripts" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _run_subprocess(
    name: str,
    command: Sequence[str],
    root: Path,
    stages: list[dict[str, Any]],
) -> None:
    log_path = root / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = StageRecord(name=name, command=[str(x) for x in command], log=str(log_path), started_at_utc=_utc_now())
    stages.append(asdict(record))
    print(f"\n[pipeline] {name} ...", flush=True)
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                [str(x) for x in command],
                cwd=str(project_root()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                _write_console(sys.stdout, line)
            return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
        stages[-1].update({"status": "completed", "finished_at_utc": _utc_now()})
    except Exception as exc:
        stages[-1].update({"status": "failed", "finished_at_utc": _utc_now(), "error": repr(exc)})
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n[pipeline] ERROR\n")
            traceback.print_exc(file=handle)
        # Keep a machine-readable failure marker even when the caller aborts
        # before it can assemble the final success manifest.
        _write_json(
            root / "run_manifest.failed.json",
            {
                "schema_version": 1,
                "status": "failed",
                "failed_stage": name,
                "created_at_utc": _utc_now(),
                "output_root": str(root),
                "stages": stages,
                "error": repr(exc),
            },
        )
        raise


def _run_callable(
    name: str,
    command: Sequence[str],
    root: Path,
    stages: list[dict[str, Any]],
    function: Callable[[], Any],
) -> Any:
    log_path = root / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = StageRecord(name=name, command=[str(x) for x in command], log=str(log_path), started_at_utc=_utc_now())
    stages.append(asdict(record))
    print(f"\n[pipeline] {name} ...", flush=True)
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            stream = _Tee(handle, sys.stdout)
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                result = function()
        stages[-1].update({"status": "completed", "finished_at_utc": _utc_now()})
        return result
    except Exception as exc:
        stages[-1].update({"status": "failed", "finished_at_utc": _utc_now(), "error": repr(exc)})
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n[pipeline] ERROR\n")
            traceback.print_exc(file=handle)
        _write_json(
            root / "run_manifest.failed.json",
            {
                "schema_version": 1,
                "status": "failed",
                "failed_stage": name,
                "created_at_utc": _utc_now(),
                "output_root": str(root),
                "stages": stages,
                "error": repr(exc),
            },
        )
        raise


def _copy_config(config_path: Path | None, output: Path, resolved: dict[str, Any]) -> None:
    if config_path is not None and config_path.is_file():
        shutil.copy2(config_path, output / "config_input.yaml")
    _write_json(output / "config_resolved.json", resolved)


def _failure_manifest(task: str, output_key: str):
    """Persist a durable failure record without masking the original error.

    A failed run must never look like a successful/partial release.  The
    decorator writes ``run_manifest.json`` for a fresh output directory and a
    sidecar ``run_manifest.failed.json`` when an existing release is being
    inspected with ``--allow-existing``.
    """

    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                output_value = kwargs.get(output_key)
                if output_value is not None:
                    output = Path(output_value).expanduser().resolve()
                    if output.is_dir():
                        target = output / "run_manifest.json"
                        if target.exists():
                            target = output / "run_manifest.failed.json"
                        payload = {
                            "schema_version": 1,
                            "task": task,
                            "status": "failed",
                            "created_at_utc": _utc_now(),
                            "output_root": str(output),
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                            "note": "This run is incomplete and must not be used as a formal result.",
                        }
                        try:
                            _write_json(target, payload)
                        except Exception:
                            # Preserve the original exception if a read-only
                            # output tree prevents writing the diagnostic.
                            pass
                raise

        return wrapped

    return decorate


def _task1_training_config(
    model: str,
    values: dict[str, Any],
    model_values: dict[str, Any],
    overrides: dict[str, Any],
    *,
    smoke: bool,
):
    from mouse_sleep.epilepsy.single_channel_train import SingleChannelTrainingConfig

    override = model_values.get(model, {}) if isinstance(model_values.get(model, {}), dict) else {}
    merged = dict(values)
    merged.update(override)
    if smoke:
        epochs = 1
        if model == "eegwavenet_sc":
            smoke_batch = (
                overrides.get("eegwavenet_batch_size")
                if overrides.get("eegwavenet_batch_size") is not None
                else merged.get("eegwavenet_batch_size", merged.get("batch_size", 32))
            )
        else:
            smoke_batch = overrides.get("batch_size") or merged.get("batch_size", 32)
        batch_size = min(int(smoke_batch), 32)
        samples_per_epoch = 1024 if model != "eegwavenet_sc" else None
        sampling_mode = "weighted_smoke" if model != "eegwavenet_sc" else "all_windows"
        max_eval = 512
    else:
        epochs = int(overrides.get("epochs") if overrides.get("epochs") is not None else merged.get("epochs", 12))
        if model == "eegwavenet_sc":
            batch_value = (
                overrides.get("eegwavenet_batch_size")
                if overrides.get("eegwavenet_batch_size") is not None
                else merged.get("eegwavenet_batch_size", 1024)
            )
        else:
            batch_value = (
                overrides.get("batch_size")
                if overrides.get("batch_size") is not None
                else merged.get("batch_size", 256)
            )
        batch_size = int(batch_value)
        samples_per_epoch = overrides.get("physio_samples_per_epoch") if model != "eegwavenet_sc" else None
        if samples_per_epoch is None and model != "eegwavenet_sc":
            samples_per_epoch = merged.get("samples_per_epoch", merged.get("physio_samples_per_epoch", 262144))
        sampling_mode = str(overrides.get("sampling_mode") or merged.get("sampling_mode", "all_windows" if model == "eegwavenet_sc" else "weighted_smoke"))
        max_eval = (
            overrides.get("maximum_evaluation_windows")
            if overrides.get("maximum_evaluation_windows") is not None
            else merged.get("maximum_evaluation_windows")
        )
    stats_model = model == "physio_sequence_stats"
    return SingleChannelTrainingConfig(
        model_name=model,
        epochs=epochs,
        batch_size=batch_size,
        samples_per_epoch=int(samples_per_epoch) if samples_per_epoch is not None else None,
        sampling_mode=sampling_mode,
        class_weight_mode=str(overrides.get("class_weight_mode") or merged.get("class_weight_mode", "hierarchical_event_region")),
        learning_rate=float(overrides.get("learning_rate") if overrides.get("learning_rate") is not None else merged.get("learning_rate", 3e-4)),
        weight_decay=float(overrides.get("weight_decay") if overrides.get("weight_decay") is not None else merged.get("weight_decay", 1e-4)),
        auxiliary_statistics_weight=float(override.get("auxiliary_statistics_weight", 0.15 if stats_model else 0.0)),
        early_stopping_patience=int(overrides.get("early_stopping_patience") if overrides.get("early_stopping_patience") is not None else merged.get("early_stopping_patience", 4)),
        learning_rate_patience=int(overrides.get("learning_rate_patience") if overrides.get("learning_rate_patience") is not None else merged.get("learning_rate_patience", 2)),
        learning_rate_factor=float(overrides.get("learning_rate_factor") if overrides.get("learning_rate_factor") is not None else merged.get("learning_rate_factor", 0.5)),
        gradient_clip_norm=float(overrides.get("gradient_clip_norm") if overrides.get("gradient_clip_norm") is not None else merged.get("gradient_clip_norm", 1.0)),
        normalization_clip=float(overrides.get("normalization_clip") if overrides.get("normalization_clip") is not None else merged.get("normalization_clip", 20.0)),
        maximum_evaluation_windows=max_eval,
        mixed_precision=bool(overrides.get("mixed_precision") if overrides.get("mixed_precision") is not None else merged.get("mixed_precision", True)),
        seed=int(overrides.get("seed") if overrides.get("seed") is not None else merged.get("seed", 20260822)),
        device=str(overrides.get("device") or merged.get("device", "auto")),
        num_workers=int(overrides.get("num_workers") if overrides.get("num_workers") is not None else merged.get("num_workers", 0)),
    )


def _task1_training_summary(
    model: str,
    values: dict[str, Any],
    model_values: dict[str, Any],
    overrides: dict[str, Any],
    *,
    smoke: bool,
) -> dict[str, Any]:
    """Resolve training values without importing torch.

    The summary is written before a run starts so another researcher can see
    the exact effective settings even on an audit-only machine.  The actual
    dataclass used by the trainer is built by ``_task1_training_config`` from
    the same values immediately before training.
    """

    override = model_values.get(model, {}) if isinstance(model_values.get(model, {}), dict) else {}
    merged = dict(values)
    merged.update(override)
    is_physio = model != "eegwavenet_sc"
    is_stats = model == "physio_sequence_stats"
    if smoke:
        epochs = 1
        if not is_physio:
            smoke_batch = (
                overrides.get("eegwavenet_batch_size")
                if overrides.get("eegwavenet_batch_size") is not None
                else merged.get("eegwavenet_batch_size", merged.get("batch_size", 32))
            )
        else:
            smoke_batch = overrides.get("batch_size") or merged.get("batch_size", 32)
        batch = min(int(smoke_batch), 32)
        samples = 1024 if is_physio else None
        mode = "weighted_smoke" if is_physio else "all_windows"
        max_eval = 512
    else:
        epochs = int(overrides.get("epochs") if overrides.get("epochs") is not None else merged.get("epochs", 12))
        if not is_physio:
            batch_value = (
                overrides.get("eegwavenet_batch_size")
                if overrides.get("eegwavenet_batch_size") is not None
                else merged.get("eegwavenet_batch_size", 1024)
            )
        else:
            batch_value = (
                overrides.get("batch_size")
                if overrides.get("batch_size") is not None
                else merged.get("batch_size", 256)
            )
        batch = int(batch_value)
        samples = overrides.get("physio_samples_per_epoch") if is_physio else None
        if samples is None and is_physio:
            samples = merged.get("samples_per_epoch", merged.get("physio_samples_per_epoch", 262144))
        mode = str(overrides.get("sampling_mode") or merged.get("sampling_mode", "all_windows" if not is_physio else "weighted_smoke"))
        max_eval = (
            overrides.get("maximum_evaluation_windows")
            if overrides.get("maximum_evaluation_windows") is not None
            else merged.get("maximum_evaluation_windows")
        )
    return {
        "model_name": model,
        "epochs": epochs,
        "batch_size": batch,
        "samples_per_epoch": int(samples) if samples is not None else None,
        "sampling_mode": mode,
        "class_weight_mode": str(overrides.get("class_weight_mode") or merged.get("class_weight_mode", "hierarchical_event_region")),
        "learning_rate": float(overrides.get("learning_rate") if overrides.get("learning_rate") is not None else merged.get("learning_rate", 3e-4)),
        "weight_decay": float(overrides.get("weight_decay") if overrides.get("weight_decay") is not None else merged.get("weight_decay", 1e-4)),
        "auxiliary_statistics_weight": float(override.get("auxiliary_statistics_weight", 0.15 if is_stats else 0.0)),
        "early_stopping_patience": int(overrides.get("early_stopping_patience") if overrides.get("early_stopping_patience") is not None else merged.get("early_stopping_patience", 4)),
        "learning_rate_patience": int(overrides.get("learning_rate_patience") if overrides.get("learning_rate_patience") is not None else merged.get("learning_rate_patience", 2)),
        "learning_rate_factor": float(overrides.get("learning_rate_factor") if overrides.get("learning_rate_factor") is not None else merged.get("learning_rate_factor", 0.5)),
        "gradient_clip_norm": float(overrides.get("gradient_clip_norm") if overrides.get("gradient_clip_norm") is not None else merged.get("gradient_clip_norm", 1.0)),
        "normalization_clip": float(overrides.get("normalization_clip") if overrides.get("normalization_clip") is not None else merged.get("normalization_clip", 20.0)),
        "maximum_evaluation_windows": max_eval,
        "mixed_precision": bool(overrides.get("mixed_precision") if overrides.get("mixed_precision") is not None else merged.get("mixed_precision", True)),
        "seed": int(overrides.get("seed") if overrides.get("seed") is not None else merged.get("seed", 20260822)),
        "device": str(overrides.get("device") or merged.get("device", "auto")),
        "num_workers": int(overrides.get("num_workers") if overrides.get("num_workers") is not None else merged.get("num_workers", 0)),
    }


@_failure_manifest("annotation_detection", "output_root")
def run_task1_pipeline(
    *,
    input_path: str | Path,
    labels_path: str | Path,
    output_root: str | Path,
    config_path: str | Path | None = None,
    models: Iterable[str] = MODEL_ORDER_TASK1,
    animals: Iterable[str] | None = None,
    sampling_rate_hz: float | None = None,
    target_rate_hz: float | None = None,
    window_seconds: float | None = None,
    stride_seconds: float | None = None,
    labels_time_base: str = "absolute_ms",
    ch7_column: str | None = None,
    ch8_column: str | None = None,
    device: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    eegwavenet_batch_size: int | None = None,
    physio_samples_per_epoch: int | None = None,
    sampling_mode: str | None = None,
    class_weight_mode: str | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    early_stopping_patience: int | None = None,
    learning_rate_patience: int | None = None,
    learning_rate_factor: float | None = None,
    gradient_clip_norm: float | None = None,
    normalization_clip: float | None = None,
    maximum_evaluation_windows: int | None = None,
    mixed_precision: bool | None = None,
    num_workers: int | None = None,
    seed: int | None = None,
    filter_overrides: Mapping[str, Any] | None = None,
    primary_iou: float | None = None,
    minimum_sensitivity: float | None = None,
    minimum_precision: float | None = None,
    maximum_fp_per_hour: float | None = None,
    skip_training: bool = False,
    skip_render: bool = False,
    smoke: bool = False,
    allow_existing: bool = False,
    hash_inputs: bool = True,
    command: Sequence[str] | None = None,
    dry_run: bool = False,
) -> Path:
    """Run raw EEG -> Task-1 protocol -> models -> decoder -> figures."""

    input_path = Path(input_path).expanduser().resolve()
    labels_path = Path(labels_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    animal_override = tuple(str(value) for value in animals) if animals is not None else None
    if dry_run:
        # A dry run is a planning operation; paths are intentionally allowed
        # to be placeholders so CI and data managers can inspect the resolved
        # command before mounting sensitive recordings.
        input_exists = input_path.exists()
        labels_exists = labels_path.exists()
    else:
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        if not labels_path.exists():
            raise FileNotFoundError(labels_path)
    selected_models = tuple(models)
    unknown = set(selected_models) - set(MODEL_ORDER_TASK1)
    if unknown:
        raise ValueError(f"未知任务一模型: {sorted(unknown)}")
    config = read_yaml(config_path) if config_path else {}
    prep_values = dict(config.get("preprocessing", {}))
    protocol_values = dict(config.get("protocol", {}))
    training_values = dict(config.get("training", {}))
    model_values = dict(config.get("model_overrides", {}))
    filter_values = _resolve_filter_values(
        prep_values.get("filter", config.get("filter", {})),
        filter_overrides,
    )
    train_overrides = {
        "epochs": epochs,
        "batch_size": batch_size,
        "eegwavenet_batch_size": eegwavenet_batch_size,
        "physio_samples_per_epoch": physio_samples_per_epoch,
        "sampling_mode": sampling_mode,
        "class_weight_mode": class_weight_mode,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "early_stopping_patience": early_stopping_patience,
        "learning_rate_patience": learning_rate_patience,
        "learning_rate_factor": learning_rate_factor,
        "gradient_clip_norm": gradient_clip_norm,
        "normalization_clip": normalization_clip,
        "maximum_evaluation_windows": maximum_evaluation_windows,
        "mixed_precision": mixed_precision,
        "num_workers": num_workers,
        "seed": seed,
        "device": device,
    }
    effective_training = {
        model: _task1_training_summary(
            model,
            training_values,
            model_values,
            train_overrides,
            smoke=smoke,
        )
        for model in selected_models
    }
    decoder_values = dict(config.get("decoder", {}))
    effective_decoder = {
        **decoder_values,
        "primary_match_iou": primary_iou if primary_iou is not None else decoder_values.get("primary_match_iou", 0.30),
        "minimum_sensitivity": minimum_sensitivity if minimum_sensitivity is not None else 0.85,
        "minimum_precision": minimum_precision if minimum_precision is not None else 0.15,
        "maximum_fp_per_hour": maximum_fp_per_hour if maximum_fp_per_hour is not None else 40.0,
        "selection_split": decoder_values.get("selection_split", "val"),
        "test_used_for_selection": False,
    }
    effective_protocol = dict(protocol_values)
    effective_protocol["filter"] = _resolve_filter_values(
        protocol_values.get("filter", filter_values),
        filter_overrides,
    )
    if normalization_clip is not None:
        effective_protocol["normalization_clip"] = normalization_clip
    resolved = {
        "task": "annotation_detection",
        "input": str(input_path),
        "labels": str(labels_path),
        "models": list(selected_models),
        # Keep every command-line/API override in the provenance record.  The
        # nested config sections below contain YAML values; this explicit
        # section records which values were supplied for this particular run.
        "cli_overrides": {
            "animals": list(animal_override) if animal_override is not None else None,
            "sampling_rate_hz": sampling_rate_hz,
            "target_rate_hz": target_rate_hz,
            "window_seconds": window_seconds,
            "stride_seconds": stride_seconds,
            "labels_time_base": labels_time_base,
            "ch7_column": ch7_column,
            "ch8_column": ch8_column,
            "device": device,
            "epochs": epochs,
            "batch_size": batch_size,
            "eegwavenet_batch_size": eegwavenet_batch_size,
            "physio_samples_per_epoch": physio_samples_per_epoch,
            "sampling_mode": sampling_mode,
            "class_weight_mode": class_weight_mode,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "early_stopping_patience": early_stopping_patience,
            "learning_rate_patience": learning_rate_patience,
            "learning_rate_factor": learning_rate_factor,
            "gradient_clip_norm": gradient_clip_norm,
            "normalization_clip": normalization_clip,
            "maximum_evaluation_windows": maximum_evaluation_windows,
            "mixed_precision": mixed_precision,
            "num_workers": num_workers,
            "seed": seed,
            "filter": dict(filter_overrides or {}),
            "primary_iou": primary_iou,
            "minimum_sensitivity": minimum_sensitivity,
            "minimum_precision": minimum_precision,
            "maximum_fp_per_hour": maximum_fp_per_hour,
            "skip_training": skip_training,
            "skip_render": skip_render,
            "smoke": smoke,
            "allow_existing": allow_existing,
            "hash_inputs": hash_inputs,
        },
        "preprocessing": {
            "sampling_rate_hz": sampling_rate_hz,
            "target_rate_hz": target_rate_hz if target_rate_hz is not None else prep_values.get("target_rate_hz", 250.0),
            "window_seconds": window_seconds if window_seconds is not None else prep_values.get("window_seconds", 4.0),
            "stride_seconds": stride_seconds if stride_seconds is not None else prep_values.get("stride_seconds", 1.0),
            "labels_time_base": labels_time_base,
            "ch7_column": ch7_column or prep_values.get("ch7_column"),
            "ch8_column": ch8_column or prep_values.get("ch8_column"),
            "filter": filter_values,
        },
        "protocol": protocol_values,
        "effective_protocol": effective_protocol,
        "training": training_values,
        "decoder": decoder_values,
        "effective_training": effective_training,
        "effective_decoder": effective_decoder,
        "smoke": smoke,
    }
    if dry_run:
        resolved["dry_run_path_exists"] = {"input": input_exists, "labels": labels_exists}
        print(json.dumps(_json_safe({"command": "run-task1", "output": str(output), "resolved": resolved}), ensure_ascii=False, indent=2))
        return output
    _ensure_output_root(output, allow_existing)
    _copy_config(Path(config_path).resolve() if config_path else None, output, resolved)
    stages: list[dict[str, Any]] = []
    source_info = _source_inventory(input_path, hash_files=hash_inputs, exclude=labels_path)
    label_info = {
        "path": str(labels_path),
        "files": _label_inventory(labels_path, hash_files=hash_inputs),
    }
    _write_json(output / "command.json", {"argv": list(command or []), "python": sys.executable, "cwd": str(project_root()), "created_at_utc": _utc_now()})

    from .raw_adapter import EEGFilterConfig

    prepared = output / "preprocessing"
    protocol = _run_callable(
        "01_prepare",
        ["epilepsy-eeg", "prepare", "--input", str(input_path), "--labels", str(labels_path), "--output", str(prepared)],
        output,
        stages,
        lambda: build_protocol(
            input_path,
            prepared,
            labels=labels_path,
            sampling_rate_hz=sampling_rate_hz,
            target_rate_hz=float(target_rate_hz if target_rate_hz is not None else prep_values.get("target_rate_hz", 250.0)),
            window_seconds=float(window_seconds if window_seconds is not None else prep_values.get("window_seconds", 4.0)),
            stride_seconds=float(stride_seconds if stride_seconds is not None else prep_values.get("stride_seconds", 1.0)),
            labels_time_base=labels_time_base,
            ch7_column=ch7_column or prep_values.get("ch7_column"),
            ch8_column=ch8_column or prep_values.get("ch8_column"),
            filter_config=EEGFilterConfig(**filter_values) if filter_values else None,
        ),
    )
    manifest = pd.read_csv(prepared / "complete_filtered_manifest.csv")
    inferred_animals = sorted(manifest["animal_id"].astype(str).unique())
    selected_animals = tuple(str(x) for x in (animal_override or inferred_animals))
    if not selected_animals:
        raise ValueError("预处理结果没有动物记录")
    excluded_raw = _parse_excluded(
        protocol_values.get("excluded_animal_channels"),
        (("E07", "CH7"), ("E77", "CH7")),
    )
    strict_protocol = output / "protocol"
    _run_callable(
        "02_build_protocol",
        ["epilepsy-eeg", "build-task1-protocol", "--filtered-manifest", str(prepared / "complete_filtered_manifest.csv"), "--event-components", str(protocol / "event_components.parquet"), "--output", str(strict_protocol)],
        output,
        stages,
        lambda: __import__("mouse_sleep.epilepsy.continuous_single_channel", fromlist=["build_continuous_single_channel_protocol"]).build_continuous_single_channel_protocol(
            prepared / "complete_filtered_manifest.csv",
            protocol / "event_components.parquet",
            strict_protocol,
            __import__("mouse_sleep.epilepsy.continuous_single_channel", fromlist=["ContinuousSingleChannelConfig"]).ContinuousSingleChannelConfig(
                animals=selected_animals,
                train_fraction=float(protocol_values.get("train_fraction", 0.70)),
                validation_fraction=float(protocol_values.get("validation_fraction", 0.15)),
                test_fraction=float(protocol_values.get("test_fraction", 0.15)),
                sequence_length=int(protocol_values.get("sequence_length", 21)),
                positive_overlap_fraction=float(protocol_values.get("positive_overlap_fraction", 0.50)),
                event_context_margin_seconds=float(protocol_values.get("event_context_margin_seconds", 12.0)),
                background_block_seconds=float(protocol_values.get("background_block_seconds", 300.0)),
                normalization_clip=float(
                    normalization_clip
                    if normalization_clip is not None
                    else protocol_values.get("normalization_clip", 20.0)
                ),
                random_seed=int(seed if seed is not None else protocol_values.get("random_seed", 20260814)),
                excluded_animal_channels=excluded_raw,
                purge_cross_split_windows=bool(protocol_values.get("purge_cross_split_windows", True)),
                filter_config=EEGFilterConfig(
                    **_resolve_filter_values(
                        protocol_values.get("filter", filter_values),
                        filter_overrides,
                    )
                ),
            ),
        ),
    )
    model_root = output / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    if not skip_training:
        from mouse_sleep.epilepsy.single_channel_train import run_single_channel_training

        for model in selected_models:
            model_config = _task1_training_config(model, training_values, model_values, train_overrides, smoke=smoke)
            _run_callable(
                f"03_train_{model}",
                ["epilepsy-eeg", "train-task1", "--protocol-dir", str(strict_protocol), "--output-root", str(model_root), "--models", model],
                output,
                stages,
                lambda model_config=model_config, model=model: run_single_channel_training(strict_protocol, model_root / model, model_config),
            )
    else:
        missing = [str(model_root / model / "model.pt") for model in selected_models if not (model_root / model / "model.pt").is_file()]
        if missing:
            raise FileNotFoundError("--skip-training 需要已有权重: " + ", ".join(missing))

    from mouse_sleep.epilepsy.single_channel_inference import run_continuous_single_channel_inference

    for model in selected_models:
        _run_callable(
            f"04_infer_{model}",
            ["continuous-inference", model],
            output,
            stages,
            lambda model=model: run_continuous_single_channel_inference(strict_protocol, model_root / model, model_root / model / "continuous", device=str(device or training_values.get("device", "auto"))),
        )

    corrected = output / "decoder"
    decoder_command = [sys.executable, str(_script("run_corrected_event_decoder_v1.py")), "--run-root", str(model_root), "--protocol-dir", str(strict_protocol), "--output-root", str(corrected), "--models", *selected_models]
    decoder_command += ["--primary-iou", str(primary_iou if primary_iou is not None else config.get("decoder", {}).get("primary_match_iou", 0.30)), "--minimum-sensitivity", str(minimum_sensitivity if minimum_sensitivity is not None else 0.85), "--minimum-precision", str(minimum_precision if minimum_precision is not None else 0.15), "--maximum-fp-per-hour", str(maximum_fp_per_hour if maximum_fp_per_hour is not None else 40.0), "--allow-existing"]
    _run_subprocess("05_corrected_decoder", decoder_command, output, stages)

    if not skip_render:
        performance = output / "figures" / "model_performance"
        render_perf = [sys.executable, str(_script("render_task1_corrected_model_performance.py")), "--run-root", str(model_root), "--corrected-root", str(corrected), "--output-root", str(performance), "--allow-existing"]
        _run_subprocess("06_render_performance", render_perf, output, stages)
        atlas = output / "figures" / "clinical_event_atlas"
        render_atlas = [sys.executable, str(_script("render_corrected_event_atlas.py")), "--corrected-root", str(corrected), "--protocol-dir", str(strict_protocol), "--output-dir", str(atlas), "--models", *selected_models, "--allow-existing"]
        _run_subprocess("07_render_event_atlas", render_atlas, output, stages)

    _write_task1_indexes(output, stages)
    run_manifest = {
        "schema_version": 1,
        "task": "annotation_detection",
        "status": "completed",
        "created_at_utc": _utc_now(),
        "repository": str(project_root()),
        "git_revision": _git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "input_files": source_info,
        "label_file": label_info,
        "animals": list(selected_animals),
        "models": list(selected_models),
        "output_root": str(output),
        "stages": stages,
        "directories": {
            "preprocessing": str(prepared),
            "protocol": str(strict_protocol),
            "models": str(model_root),
            "decoder": str(corrected),
            "figures": str(output / "figures"),
            "metrics": str(output / "metrics"),
            "reports": str(output / "reports"),
            "logs": str(output / "logs"),
        },
        "offline_only": True,
        "raw_data_not_copied": True,
        "test_used_for_decoder_selection": False,
    }
    _write_json(output / "run_manifest.json", run_manifest)
    (output / "README_zh.md").write_text(
        "# Task 1 完整重跑\n\n"
        "本目录由 `epilepsy-eeg run-task1` 生成，链路为：原始 EEG -> 250 Hz/0.5–70 Hz/50 Hz 陷波与 QC -> 事件级 70/15/15 + raw-window purge -> 三模型训练 -> 全连续推理 -> validation-only corrected decoder -> 模型指标、混淆矩阵和临床事件 PDF。\n\n"
        "每个阶段的终端等价日志在 `logs/`，解析后的配置在 `config_resolved.json`，命令在 `command.json`，权重/历史/预测在 `models/`。原始数据和敏感标签只记录路径与哈希，不复制到仓库。\n",
        encoding="utf-8",
    )
    return output


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root(), text=True).strip()
    except Exception:
        return None


def _copy_if_exists(source: Path, target: Path) -> bool:
    """Copy a derived index file when a stage produced it."""

    if not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def _write_task1_indexes(output: Path, stages: list[dict[str, Any]]) -> None:
    """Create a small, stable root index without duplicating large artifacts."""

    metrics = output / "metrics"
    reports = output / "reports"
    metrics.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    candidates = (
        (output / "figures" / "model_performance" / "source_data" / "corrected_window_metrics.csv", metrics / "corrected_window_metrics.csv"),
        (output / "figures" / "model_performance" / "source_data" / "corrected_event_metrics.csv", metrics / "corrected_event_metrics.csv"),
        (output / "figures" / "model_performance" / "source_data" / "window_metrics.csv", metrics / "window_metrics.csv"),
        (output / "figures" / "model_performance" / "source_data" / "confusion_matrices.csv", metrics / "confusion_matrices.csv"),
        (output / "decoder" / "combined_decoder_comparison.csv", metrics / "combined_decoder_comparison.csv"),
    )
    copied = [str(dst.relative_to(output)) for src, dst in candidates if _copy_if_exists(src, dst)]
    rows = []
    for stage in stages:
        rows.append(
            f"| {stage.get('name', '')} | {stage.get('status', '')} | `{stage.get('log', '')}` |"
        )
    report = (
        "# Task 1 run report\n\n"
        "该目录由 `epilepsy-eeg run-task1` 生成。默认路径是从原始 EEG 完整执行预处理、严格事件划分、三模型从头训练、连续推理、validation-only decoder 和论文图；只有显式传入 `--skip-training` 才会复用本次 run 已存在的权重。\n\n"
        "## 关键入口\n\n"
        "- `preprocessing/`：重采样、滤波、QC 和缓存。\n"
        "- `protocol/`：事件级 70/15/15 划分与跨 split raw-window purge。\n"
        "- `models/<model>/`：源码训练产生的 `model.pt`、`history.csv`、预测和指标。\n"
        "- `decoder/`：独立的连续概率解码与事件级审计。\n"
        "- `figures/model_performance/`、`figures/clinical_event_atlas/`：由表格重新绘制的图。\n"
        "- `logs/`：与终端同步的逐阶段日志。\n"
        f"- 根级指标索引：{', '.join(f'`{item}`' for item in copied) if copied else '阶段尚未生成指标'}。\n\n"
        "## 阶段日志\n\n"
        "| 阶段 | 状态 | 日志 |\n|---|---|---|\n"
        + "\n".join(rows)
        + "\n\n"
        "原始记录和临床标签不会复制到 run；`run_manifest.json` 仅保存路径、大小和 SHA256。该流程是离线/回放和医生在环研究接口，不等同于已验证的实时临床部署。\n"
    )
    (reports / "RUN_REPORT_zh.md").write_text(report, encoding="utf-8")


def _write_pg36_indexes(output: Path, stages: list[dict[str, Any]]) -> None:
    """Create a compact root index for a full PG36 rerun."""

    metrics = output / "metrics"
    reports = output / "reports"
    metrics.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    candidates = (
        (output / "05_classification_pooled" / "all_window_model_comparison.csv", metrics / "all_window_model_comparison.csv"),
        (output / "05_classification_pooled" / "model_comparison.csv", metrics / "event_balanced_model_comparison.csv"),
        (output / "01_tables" / "pg36_input_manifest.json", metrics / "pg36_input_manifest.json"),
        (output / "04_label_freeze_pooled" / "pooled_label_freeze_manifest.json", metrics / "pooled_label_freeze_manifest.json"),
    )
    copied = [str(dst.relative_to(output)) for src, dst in candidates if _copy_if_exists(src, dst)]
    rows = [
        f"| {stage.get('name', '')} | {stage.get('status', '')} | `{stage.get('log', '')}` |"
        for stage in stages
    ]
    report = (
        "# PG36 run report\n\n"
        "该目录由 `epilepsy-eeg run-pg36` 生成。raw 模式从原始 EEG 开始，依次执行预处理、严格协议、独立事件×通道特征、逐鼠聚类、PG36 冻结、五模型从头训练、全测试窗口评估和图形渲染。\n\n"
        "- `01_tables/`：独立通道特征与输入审计。\n"
        "- `03_cross_mouse/`：逐鼠 pooled 聚类和原型。\n"
        "- `04_label_freeze_pooled/`：冻结 PG01–PG36 标签。\n"
        "- `05_classification_pooled/<model>/`：源码训练产生的 checkpoint、history、预测、混淆矩阵和指标。\n"
        "- `06_figures/`：逐鼠形态、全局 PCA、支持矩阵和五模型全类别图。\n"
        "- `logs/`：与终端同步的逐阶段日志。\n"
        f"- 根级指标索引：{', '.join(f'`{item}`' for item in copied) if copied else '阶段尚未生成指标'}。\n\n"
        "PG 类别是探索性形态类别，不是临床确证亚型；原始数据和敏感标签不会复制到 run。\n\n"
        "## 阶段日志\n\n"
        "| 阶段 | 状态 | 日志 |\n|---|---|---|\n"
        + "\n".join(rows)
        + "\n"
    )
    (reports / "RUN_REPORT_zh.md").write_text(report, encoding="utf-8")


@_failure_manifest("morphology_pg36", "output_root")
def run_pg36_pipeline(
    *,
    output_root: str | Path,
    input_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    taxonomy_root: str | Path | None = None,
    config_path: str | Path | None = None,
    models: Iterable[str] = MODEL_ORDER_TASK2,
    animals: Iterable[str] | None = None,
    sampling_rate_hz: float | None = None,
    target_rate_hz: float | None = None,
    window_seconds: float | None = None,
    stride_seconds: float | None = None,
    labels_time_base: str = "absolute_ms",
    ch7_column: str | None = None,
    ch8_column: str | None = None,
    filter_overrides: Mapping[str, Any] | None = None,
    device: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    max_per_event: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    early_stopping_patience: int | None = None,
    learning_rate_patience: int | None = None,
    num_workers: int | None = None,
    seed: int | None = None,
    threshold: float | None = None,
    expected_classes: int = 36,
    allow_non36: bool = False,
    baseline_samples_per_stream: int = 256,
    skip_training: bool = False,
    skip_render: bool = False,
    smoke: bool = False,
    allow_existing: bool = False,
    hash_inputs: bool = True,
    command: Sequence[str] | None = None,
    dry_run: bool = False,
) -> Path:
    """Run raw EEG/protocol -> PG36 taxonomy -> five classifiers -> figures."""

    output = Path(output_root).expanduser().resolve()
    selected_models = tuple(models)
    animal_override = tuple(str(value) for value in animals) if animals is not None else None
    unknown = set(selected_models) - set(MODEL_ORDER_TASK2)
    if unknown:
        raise ValueError(f"未知任务二模型: {sorted(unknown)}")
    config = read_yaml(config_path) if config_path else {}
    preprocessing = dict(config.get("preprocessing", {}))
    training = dict(config.get("training", {}))
    filter_values = _resolve_filter_values(
        preprocessing.get("filter", config.get("filter", {})),
        filter_overrides,
    )
    effective_training = {
        "epochs": int(1 if smoke else (epochs if epochs is not None else training.get("epochs", 10))),
        "batch_size": int(batch_size if batch_size is not None else (32 if smoke else training.get("batch_size", 64))),
        "max_per_event": int(max_per_event if max_per_event is not None else training.get("max_per_event", 8)),
        "learning_rate": float(learning_rate if learning_rate is not None else training.get("learning_rate", 3e-4)),
        "weight_decay": float(weight_decay if weight_decay is not None else training.get("weight_decay", 1e-4)),
        "early_stopping_patience": int(early_stopping_patience if early_stopping_patience is not None else training.get("early_stopping_patience", 3)),
        "learning_rate_patience": int(learning_rate_patience if learning_rate_patience is not None else training.get("learning_rate_patience", 1)),
        "num_workers": int(num_workers if num_workers is not None else training.get("num_workers", 0)),
        "seed": int(seed if seed is not None else training.get("seed", 20260823)),
        "device": str(device or training.get("device", "auto")),
    }
    resolved = {
        "task": "morphology_pg36",
        "output": str(output),
        "input": str(Path(input_path).resolve()) if input_path else None,
        "labels": str(Path(labels_path).resolve()) if labels_path else None,
        "taxonomy_root": str(Path(taxonomy_root).resolve()) if taxonomy_root else None,
        "models": list(selected_models),
        "cli_overrides": {
            "animals": list(animal_override) if animal_override is not None else None,
            "sampling_rate_hz": sampling_rate_hz,
            "target_rate_hz": target_rate_hz,
            "window_seconds": window_seconds,
            "stride_seconds": stride_seconds,
            "labels_time_base": labels_time_base,
            "ch7_column": ch7_column,
            "ch8_column": ch8_column,
            "filter": dict(filter_overrides or {}),
            "device": device,
            "epochs": epochs,
            "batch_size": batch_size,
            "max_per_event": max_per_event,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "early_stopping_patience": early_stopping_patience,
            "learning_rate_patience": learning_rate_patience,
            "num_workers": num_workers,
            "seed": seed,
            "threshold": threshold,
            "expected_classes": expected_classes,
            "allow_non36": allow_non36,
            "baseline_samples_per_stream": baseline_samples_per_stream,
            "skip_training": skip_training,
            "skip_render": skip_render,
            "smoke": smoke,
            "allow_existing": allow_existing,
            "hash_inputs": hash_inputs,
        },
        "training": training,
        "effective_training": effective_training,
        "preprocessing": {
            "sampling_rate_hz": sampling_rate_hz,
            "target_rate_hz": target_rate_hz if target_rate_hz is not None else preprocessing.get("target_rate_hz", 250.0),
            "window_seconds": window_seconds if window_seconds is not None else preprocessing.get("window_seconds", 4.0),
            "stride_seconds": stride_seconds if stride_seconds is not None else preprocessing.get("stride_seconds", 1.0),
            "filter": filter_values,
        },
        "input_mode": "taxonomy_retrain" if taxonomy_root else "raw_to_result",
        "labels_time_base": labels_time_base,
        "channel_columns": {
            "CH7": ch7_column or preprocessing.get("ch7_column"),
            "CH8": ch8_column or preprocessing.get("ch8_column"),
        },
        "expected_classes": expected_classes,
        "allow_non36": allow_non36,
        "smoke": smoke,
    }
    if dry_run:
        print(json.dumps(_json_safe({"command": "run-pg36", "resolved": resolved}), ensure_ascii=False, indent=2))
        return output
    if taxonomy_root is not None and (input_path is not None or labels_path is not None):
        raise ValueError(
            "--taxonomy-root 是复用冻结形态表的模式，不能同时传 --input/--labels；"
            "如需从原始数据完整重跑，请只使用 --input 和 --labels"
        )
    if taxonomy_root is None and (input_path is None or labels_path is None):
        raise ValueError("run-pg36 的 raw 模式需要同时提供 --input 和 --labels；已有结果模式使用 --taxonomy-root")
    _ensure_output_root(output, allow_existing)
    _copy_config(Path(config_path).resolve() if config_path else None, output, resolved)
    stages: list[dict[str, Any]] = []
    source_info = (
        _source_inventory(input_path, hash_files=hash_inputs, exclude=labels_path)
        if input_path
        else []
    )
    label_info = None
    if labels_path:
        label_file = Path(labels_path).resolve()
        if not label_file.exists():
            raise FileNotFoundError(label_file)
        label_info = {
            "path": str(label_file),
            "files": _label_inventory(label_file, hash_files=hash_inputs),
        }
    _write_json(output / "command.json", {"argv": list(command or []), "python": sys.executable, "cwd": str(project_root()), "created_at_utc": _utc_now()})

    frozen_taxonomy_input = False
    if taxonomy_root is not None:
        source_root = Path(taxonomy_root).expanduser().resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(source_root)
        # Existing taxonomy mode keeps the source read-only and copies only
        # the small table/manifest layer needed for a self-contained rerun.
        # Accept both the working-tree layout and the compact release layout.
        # The latter stores the frozen taxonomy under ``taxonomy/`` and tables
        # under ``tables/``; they are expanded into the canonical run layout.
        if (source_root / "taxonomy" / "pooled_channel_window_labels.parquet").is_file():
            frozen_taxonomy_input = True
            mapping = {
                "tables": "01_tables",
                "taxonomy": "04_label_freeze_pooled",
            }
            for source_name, target_name in mapping.items():
                source = source_root / source_name
                target = output / target_name
                if source.exists() and not target.exists():
                    shutil.copytree(source, target)
            # The global freeze stage uses this pre-freeze alias only for
            # auditing; a compact release already contains final PG labels.
            final_labels = output / "04_label_freeze_pooled" / "pooled_channel_window_labels.parquet"
            prefreeze = output / "04_label_freeze" / "frozen_channel_window_labels.parquet"
            if final_labels.is_file() and not prefreeze.is_file():
                prefreeze.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(final_labels, prefreeze)
        else:
            for relative in ("01_tables", "03_cross_mouse", "04_label_freeze", "04_label_freeze_pooled"):
                source = source_root / relative
                if source.exists() and not (output / relative).exists():
                    shutil.copytree(source, output / relative)
            frozen_taxonomy_input = (output / "04_label_freeze_pooled" / "pooled_channel_window_labels.parquet").is_file()
        protocol = None
    else:
        from .raw_adapter import EEGFilterConfig
        prep_values = dict(config.get("preprocessing", {}))
        prepared = output / "preprocessing"
        protocol = _run_callable(
            "01_prepare",
            ["epilepsy-eeg", "prepare", "--input", str(input_path), "--labels", str(labels_path), "--output", str(prepared)],
            output,
            stages,
            lambda: build_protocol(
                Path(input_path),
                prepared,
                labels=Path(labels_path),
                sampling_rate_hz=sampling_rate_hz,
                target_rate_hz=float(target_rate_hz if target_rate_hz is not None else prep_values.get("target_rate_hz", 250.0)),
                window_seconds=float(window_seconds if window_seconds is not None else prep_values.get("window_seconds", 4.0)),
                stride_seconds=float(stride_seconds if stride_seconds is not None else prep_values.get("stride_seconds", 1.0)),
                labels_time_base=labels_time_base,
                ch7_column=ch7_column or prep_values.get("ch7_column"),
                ch8_column=ch8_column or prep_values.get("ch8_column"),
                filter_config=EEGFilterConfig(**filter_values) if filter_values else None,
            ),
        )
        manifest = pd.read_csv(prepared / "complete_filtered_manifest.csv")
        selected_animals = tuple(str(x) for x in (animal_override or sorted(manifest["animal_id"].astype(str).unique())))
        protocol_values = dict(config.get("protocol", {}))
        strict_protocol = output / "task1_protocol"
        _run_callable(
            "02_build_protocol",
            ["epilepsy-eeg", "build-task1-protocol", "--filtered-manifest", str(prepared / "complete_filtered_manifest.csv"), "--event-components", str(protocol / "event_components.parquet"), "--output", str(strict_protocol)],
            output,
            stages,
            lambda: __import__("mouse_sleep.epilepsy.continuous_single_channel", fromlist=["build_continuous_single_channel_protocol"]).build_continuous_single_channel_protocol(
                prepared / "complete_filtered_manifest.csv",
                protocol / "event_components.parquet",
                strict_protocol,
                __import__("mouse_sleep.epilepsy.continuous_single_channel", fromlist=["ContinuousSingleChannelConfig"]).ContinuousSingleChannelConfig(
                    animals=selected_animals,
                    train_fraction=float(protocol_values.get("train_fraction", 0.70)),
                    validation_fraction=float(protocol_values.get("validation_fraction", 0.15)),
                    test_fraction=float(protocol_values.get("test_fraction", 0.15)),
                    sequence_length=int(protocol_values.get("sequence_length", 21)),
                    positive_overlap_fraction=float(protocol_values.get("positive_overlap_fraction", 0.50)),
                    event_context_margin_seconds=float(protocol_values.get("event_context_margin_seconds", 12.0)),
                    background_block_seconds=float(protocol_values.get("background_block_seconds", 300.0)),
                    random_seed=int(seed if seed is not None else protocol_values.get("random_seed", 20260814)),
                    excluded_animal_channels=_parse_excluded(
                        protocol_values.get("excluded_animal_channels"),
                        (("E07", "CH7"), ("E77", "CH7")),
                    ),
                    filter_config=EEGFilterConfig(
                        **_resolve_filter_values(
                            protocol_values.get("filter", filter_values),
                            filter_overrides,
                        )
                    ),
                ),
            ),
        )
        _run_subprocess(
            "03_build_pg36_tables",
            [sys.executable, str(_script("prepare_pg36_inputs.py")), "--protocol-dir", str(strict_protocol), "--output-root", str(output), "--baseline-samples-per-stream", str(baseline_samples_per_stream)],
            output,
            stages,
        )
    label_path = output / "04_label_freeze_pooled" / "pooled_channel_window_labels.parquet"
    if taxonomy_root is not None:
        # A copied frozen taxonomy already has this file.  If the caller passed
        # a root in the older layout, use its canonical alias when available.
        if not label_path.is_file():
            candidate = output / "04_label_freeze_pooled" / "pooled_channel_window_labels.parquet"
            if candidate.is_file():
                label_path = candidate
    if not frozen_taxonomy_input:
        _run_subprocess("04_fit_pooled_mouse", [sys.executable, str(_script("fit_pooled_mouse_morphology.py")), "--root", str(output), "--seed", str(seed if seed is not None else 20260823)], output, stages)
        _run_subprocess("05_freeze_pg36", [sys.executable, str(_script("fit_pooled_global_taxonomy.py")), "--root", str(output), "--threshold", str(threshold if threshold is not None else 0.90), "--seed", str(seed if seed is not None else 20260823)], output, stages)
    if not label_path.is_file():
        label_path = output / "04_label_freeze_pooled" / "pooled_channel_window_labels.parquet"
    if not label_path.is_file():
        raise FileNotFoundError(f"PG36 freeze did not produce {label_path}")
    # A taxonomy-only rerun still needs the filtered waveform cache referenced
    # by each label row.  Fail early with a precise message instead of letting
    # a DataLoader fail much later with an opaque file-not-found exception.
    if taxonomy_root is not None:
        try:
            label_probe = pd.read_parquet(label_path)
        except Exception as exc:
            raise RuntimeError(f"无法读取冻结 PG36 标签表: {label_path}") from exc
        cache_column = "filtered_cache_dir" if "filtered_cache_dir" in label_probe.columns else "cache_dir" if "cache_dir" in label_probe.columns else None
        if cache_column is None:
            raise ValueError("冻结 PG36 标签表缺少 filtered_cache_dir/cache_dir 列")
        missing_cache = sorted({str(Path(value)) for value in label_probe[cache_column].dropna().unique() if not Path(str(value)).is_dir()})
        if missing_cache:
            preview = ", ".join(missing_cache[:3])
            suffix = " ..." if len(missing_cache) > 3 else ""
            raise FileNotFoundError(
                "--taxonomy-root 只复制标签/特征表，不包含原始波形缓存；"
                f"以下 cache_dir 不存在: {preview}{suffix}。"
                "请改用 raw 模式，或先把相同缓存挂载到原路径。"
            )
    freeze_manifest = output / "04_label_freeze_pooled" / "pooled_label_freeze_manifest.json"
    if freeze_manifest.is_file():
        try:
            frozen_classes = int(json.loads(freeze_manifest.read_text(encoding="utf-8")).get("n_global_categories", -1))
        except (OSError, ValueError, json.JSONDecodeError):
            frozen_classes = -1
        if frozen_classes != expected_classes and not allow_non36:
            raise ValueError(
                f"PG36 流程要求 {expected_classes} 个全局类别，实际冻结为 {frozen_classes}；"
                "如需调试其他类别数请显式使用 --allow-non36"
            )

    class_root = output / "05_classification_pooled"
    if not skip_training:
        train_command = [
            sys.executable,
            str(_script("train_pg36.py")),
            "--out", str(output),
            "--label-path", str(label_path),
            "--output-subdir", "05_classification_pooled",
            "--models", *selected_models,
            "--epochs", str(effective_training["epochs"]),
            "--batch-size", str(effective_training["batch_size"]),
            "--max-per-event", str(effective_training["max_per_event"]),
            "--learning-rate", str(effective_training["learning_rate"]),
            "--weight-decay", str(effective_training["weight_decay"]),
            "--early-stopping-patience", str(effective_training["early_stopping_patience"]),
            "--learning-rate-patience", str(effective_training["learning_rate_patience"]),
            "--num-workers", str(effective_training["num_workers"]),
            "--seed", str(effective_training["seed"]),
            "--device", effective_training["device"],
            "--fail-on-error",
        ]
        if allow_non36:
            train_command.append("--allow-non36")
        _run_subprocess("06_train_pg36", train_command, output, stages)
    else:
        missing = [str(class_root / model / "best_model.pt") for model in selected_models if not (class_root / model / "best_model.pt").is_file()]
        if missing:
            raise FileNotFoundError("--skip-training 需要已有 PG36 权重: " + ", ".join(missing))

    eval_command = [
        sys.executable,
        str(_script("evaluate_strict_classifiers_all_windows.py")),
        "--out", str(output),
        "--label-path", str(label_path),
        "--output-subdir", "05_classification_pooled",
        "--models", *selected_models,
        "--device", effective_training["device"],
        "--seed", str(effective_training["seed"]),
        "--num-workers", str(effective_training["num_workers"]),
    ]
    if allow_non36:
        eval_command.append("--allow-non36")
    _run_subprocess("07_evaluate_all_windows", eval_command, output, stages)
    if not skip_render:
        figure_root = output / "06_figures" / "paper_ready_morphology_pooled"
        render_command = [sys.executable, str(_script("render_task2_pg36_corrected_v2.py")), "--taxonomy-root", str(output), "--output-dir", str(figure_root)]
        _run_subprocess("08_render_pg36", render_command, output, stages)
        _run_subprocess("09_write_pg36_report", [sys.executable, str(_script("write_pooled_task2_report.py")), "--root", str(output)], output, stages)

    _write_pg36_indexes(output, stages)
    run_manifest = {
        "schema_version": 1,
        "task": "morphology_pg36",
        "status": "completed",
        "created_at_utc": _utc_now(),
        "repository": str(project_root()),
        "git_revision": _git_revision(),
        "python": sys.version,
        "input_files": source_info,
        "label_file": label_info,
        "models": list(selected_models),
        "output_root": str(output),
        "stages": stages,
        "directories": {
            "preprocessing": str(output / "preprocessing") if (output / "preprocessing").exists() else None,
            "task1_protocol": str(output / "task1_protocol") if (output / "task1_protocol").exists() else None,
            "tables": str(output / "01_tables"),
            "cross_mouse": str(output / "03_cross_mouse"),
            "taxonomy": str(output / "04_label_freeze_pooled"),
            "classification": str(output / "05_classification_pooled"),
            "figures": str(output / "06_figures"),
            "metrics": str(output / "metrics"),
            "reports": str(output / "reports"),
            "logs": str(output / "logs"),
        },
        "taxonomy": f"PG01-PG{expected_classes:02d} exploratory morphology categories; not clinical subtype labels",
        "channel_independence": True,
        "raw_data_not_copied": True,
    }
    _write_json(output / "run_manifest.json", run_manifest)
    (output / "README_zh.md").write_text(
        "# PG36 完整重跑\n\n"
        "本目录由 `epilepsy-eeg run-pg36` 生成。raw 模式依次执行预处理、严格协议、独立事件×通道特征、逐鼠 pooled 聚类、PG01–PG36 冻结、五模型训练、全测试窗口评估和论文图；`--taxonomy-root` 模式只复用已冻结的表并重新训练/评估/渲染。\n\n"
        "PG 标签是探索性形态类别，不是临床亚型。每个有效 CH7/CH8 通道是独立观察，通道不会拼接。\n",
        encoding="utf-8",
    )
    return output


__all__ = ["run_task1_pipeline", "run_pg36_pipeline"]
