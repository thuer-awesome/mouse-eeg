"""Materialize and audit the two frozen result bundles."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import MODEL_ORDER_TASK1, sha256, write_json
from .figure_manifest import figure_family, write_figure_manifest


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_filtered(source: Path, target: Path, *, extensions: set[str] | None = None, exclude_names: set[str] | None = None) -> list[str]:
    if not source.exists():
        raise FileNotFoundError(source)
    copied: list[str] = []
    for item in source.rglob("*"):
        if not item.is_file():
            continue
        if extensions is not None and item.suffix.lower() not in extensions:
            continue
        if exclude_names and item.name in exclude_names:
            continue
        relative = item.relative_to(source)
        destination = target / relative
        _copy_file(item, destination)
        copied.append(str(destination))
    return copied


def materialize_task1(model_run_root: str | Path, corrected_root: str | Path, output_root: str | Path, *, include_full_traces: bool = False, include_tiff: bool = False, allow_existing: bool = False) -> Path:
    """Copy the final Task 1 artifacts into a stable release directory.

    The huge historical ``decoder_comparison.json`` and full decoded tracks
    are omitted by default because the compact test tables plus the PDF atlas
    are sufficient for review.  The omission is recorded in the manifest and
    can be changed with ``--include-full-traces``.
    """
    source_models = Path(model_run_root).resolve()
    source_corrected = Path(corrected_root).resolve()
    target = Path(output_root).resolve()
    if target.exists() and any(target.iterdir()) and not allow_existing:
        raise FileExistsError(f"输出目录非空: {target}")
    target.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in ("RUN_SUMMARY.json", "TRAINING_LOG_zh.md", "combined_decoder_comparison.csv"):
        path = source_models / name
        if path.is_file():
            _copy_file(path, target / "metrics" / name)
            copied.append(str(path))
    for model in MODEL_ORDER_TASK1:
        src = source_models / model
        dst = target / "models" / model
        for name in ("model.pt", "history.csv", "metrics.json", "run_manifest.json"):
            path = src / name
            if path.is_file():
                _copy_file(path, dst / name)
                copied.append(str(path))
        corrected = source_corrected / "models" / model
        for name in ("decoder_fit.json", "decoder_candidates.csv"):
            path = corrected / name
            if path.is_file():
                _copy_file(path, dst / name)
                copied.append(str(path))
        for scope in ("events_corrected_v1_validation", "events_corrected_v1_test", "events_corrected_v1_full_recording"):
            src_scope = corrected / scope
            dst_scope = dst / scope
            if not src_scope.is_dir():
                continue
            allowed = {".parquet", ".csv", ".json"}
            for item in src_scope.iterdir():
                if not item.is_file() or item.suffix.lower() not in allowed:
                    continue
                if item.name == "decoded_tracks.parquet" and scope.endswith("full_recording") and not include_full_traces:
                    continue
                if item.name == "event_metrics.json" and scope.endswith("full_recording") and not include_full_traces:
                    continue
                _copy_file(item, dst_scope / item.name)
                copied.append(str(item))
    # The corrected decoder manifest/report are small and define the formal denominator.
    for name in ("manifest.json", "CURRENT_INDEX_zh.md", "README_zh.md", "CORRECTED_DECODER_FINAL_REPORT_zh.md"):
        path = source_corrected / name
        if path.is_file():
            _copy_file(path, target / "metrics" / name)
            copied.append(str(path))
    perf_source = source_corrected / "figures" / "model_performance"
    perf_target = target / "figures" / "model_performance"
    copied.extend(_copy_filtered(perf_source, perf_target, extensions={".png", ".pdf", ".svg", ".csv", ".json", ".md", ".txt"} | ({".tiff"} if include_tiff else set())))
    atlas_source = source_corrected / "figures" / "clinical_event_atlas"
    atlas_target = target / "event_atlas"
    copied.extend(_copy_filtered(atlas_source, atlas_target, extensions={".png", ".pdf", ".svg", ".csv", ".json", ".md", ".txt"} | ({".tiff"} if include_tiff else set())))
    audit = json.loads((source_corrected / "manifest.json").read_text(encoding="utf-8")) if (source_corrected / "manifest.json").is_file() else {}
    manual_counts = audit.get("manual_event_counts", {}) if isinstance(audit, dict) else {}
    clinical_denominator = int(manual_counts.get("test", 0) or 0)
    atlas_manifest_data: dict[str, Any] = {}
    if not clinical_denominator:
        atlas_manifest = target / "event_atlas" / "corrected_atlas_manifest.json"
        if atlas_manifest.is_file():
            try:
                atlas_manifest_data = json.loads(atlas_manifest.read_text(encoding="utf-8"))
                clinical_denominator = int(atlas_manifest_data.get("clinical_atlas_denominator", 0) or 0)
            except (OSError, ValueError, json.JSONDecodeError):
                clinical_denominator = 0
    formal_denominator = (
        int(audit.get("formal_metric_denominator", 0) or 0)
        if isinstance(audit, dict)
        else 0
    )
    if not formal_denominator and isinstance(audit, dict):
        formal_denominator = int(audit.get("formal_event_denominator", 0) or 0)
    if not atlas_manifest_data:
        atlas_manifest = target / "event_atlas" / "corrected_atlas_manifest.json"
        if atlas_manifest.is_file():
            try:
                atlas_manifest_data = json.loads(atlas_manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                atlas_manifest_data = {}
    # The formal metric denominator can be smaller than the clinical atlas
    # denominator because a small number of events fail the metric gate.  Do
    # not silently replace that value with the larger atlas count.
    if not formal_denominator:
        formal_denominator = int(atlas_manifest_data.get("formal_metric_denominator", 0) or 0)
    if not formal_denominator:
        formal_denominator = clinical_denominator
    (target / "README_zh.md").write_text(
        "# 任务一：二分类连续预测打标\n\n"
        "本目录对应 corrected interval-aware decoder。三个模型共享验证集拟合的全局解码参数，测试集不参与选择。"
        "每个有效通道独立输入；`event_atlas/` 保存完整人工事件、模型候选和波形/概率/状态 PDF；"
        "`figures/model_performance/` 保存窗口级混淆矩阵、PR/ROC、训练诊断和事件级汇总。"
        "外部候选只是临床复核对象，不自动等同于误报。\n",
        encoding="utf-8",
    )
    artifacts = []
    for path in target.rglob("*"):
        if path.is_file() and path.name != "release_manifest.json":
            artifacts.append({"path": str(path.relative_to(target)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256(path)})
    release = {
        "schema_version": 1,
        "task": "annotation_detection",
        "decoder_kind": "interval_aware_full_window_union",
        "models": list(MODEL_ORDER_TASK1),
        "model_run_root_source": str(source_models),
        "corrected_root_source": str(source_corrected),
        "clinical_atlas_denominator_test_channel_events": clinical_denominator,
        "formal_metric_denominator_test_events": formal_denominator,
        "include_full_traces": bool(include_full_traces),
        "include_tiff": bool(include_tiff),
        "legacy_directories_excluded": True,
        "omitted_by_default": ["decoder_comparison.json", "full_recording decoded_tracks.parquet", "full_recording event_metrics.json"],
        "artifacts": artifacts,
    }
    write_json(target / "release_manifest.json", release)
    # A release gets its own compact lineage manifest; absolute source paths
    # are intentionally not copied into the release's figure semantics.
    families = [
        figure_family(
            root=target,
            name="model_performance",
            relative_path="figures/model_performance",
            renderer="scripts/render_task1_corrected_model_performance.py",
            source_inputs=(target / "models", target / "metrics"),
            denominator="supervised held-out 4-s windows; event TN is undefined",
            semantics="corrected window metrics, confusion, PR/ROC and training diagnostics",
            requested=True,
        ),
        figure_family(
            root=target,
            name="clinical_event_atlas",
            relative_path="event_atlas",
            renderer="scripts/render_corrected_event_atlas.py",
            source_inputs=(target / "models", target / "metrics"),
            denominator="all manual test event×channel rows",
            semantics="raw waveform, probability and interval-aware decoded state",
            requested=True,
        ),
    ]
    write_figure_manifest(root=target, task="annotation_detection", families=families, path=target / "figures" / "FIGURE_MANIFEST.json")
    release["artifacts"] = [
        {
            "path": str(path.relative_to(target)).replace("\\", "/"),
            "bytes": int(path.stat().st_size),
            "sha256": sha256(path),
        }
        for path in target.rglob("*")
        if path.is_file() and path.name != "release_manifest.json"
    ]
    write_json(target / "release_manifest.json", release)
    return target


def validate_task1(root: str | Path) -> dict[str, Any]:
    target = Path(root).resolve()
    errors: list[str] = []
    models = []
    for model in MODEL_ORDER_TASK1:
        path = target / "models" / model
        if not (path / "model.pt").is_file():
            errors.append(f"missing weight: {model}")
        if not (path / "decoder_fit.json").is_file():
            errors.append(f"missing decoder config: {model}")
        models.append(model)
    comparison = target / "figures" / "model_performance" / "source_data" / "corrected_event_metrics.csv"
    event_rows = pd.read_csv(comparison) if comparison.is_file() else pd.DataFrame()
    if not event_rows.empty and "model" in event_rows and set(event_rows.model.astype(str)) != set(MODEL_ORDER_TASK1):
        errors.append("corrected event metric table does not contain exactly three models")
    figure_manifest = target / "figures" / "FIGURE_MANIFEST.json"
    if not figure_manifest.is_file():
        errors.append("missing figures/FIGURE_MANIFEST.json")
    else:
        try:
            payload = json.loads(figure_manifest.read_text(encoding="utf-8"))
            if payload.get("task") not in {None, "annotation_detection"}:
                errors.append("figure manifest task is not annotation_detection")
            families = payload.get("figure_families")
            if not isinstance(families, list) or not families:
                errors.append("figure manifest has no figure_families")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            errors.append("invalid figures/FIGURE_MANIFEST.json")
    for relative in ("figures/model_performance", "event_atlas"):
        if not (target / relative).is_dir():
            errors.append(f"missing formal figure directory: {relative}")
    result = {"root": str(target), "models": models, "ok": not errors, "errors": errors}
    write_json(target / "task1_validation.json", result)
    if errors:
        raise ValueError("Task 1 validation failed: " + "; ".join(errors))
    return result
