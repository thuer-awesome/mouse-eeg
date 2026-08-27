"""Read-only reporting for same-day mouse epilepsy LOSO runs.

The training launcher owns the run directories.  This module only consumes
completed artifacts and deliberately keeps fold-level and pooled estimates
separate: fold summaries describe between-animal variability, whereas pooled
metrics describe all available out-of-fold test windows together.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from mouse_sleep.epilepsy.metrics import window_metrics

__all__ = [
    "LosoReportTables",
    "build_loso_report_tables",
    "generate_loso_report",
    "render_clinical_markdown",
]


_WINDOW_METRICS = (
    "accuracy",
    "sensitivity",
    "specificity",
    "f1",
    "balanced_accuracy",
    "mcc",
    "auroc",
    "auprc",
)
_EVENT_RATE_METRICS = (
    "event_sensitivity",
    "false_alarms_per_hour",
    "median_session_latency_s",
)
_REQUIRED_PREDICTION_COLUMNS = {
    "animal_id",
    "session_id",
    "start_time_ms",
    "label",
    "seizure_probability",
}


@dataclass(frozen=True)
class _CompletedRun:
    model: str
    fold: int
    test_animal: str
    run_dir: Path
    threshold: float
    metrics: dict[str, Any]
    predictions: pd.DataFrame


@dataclass(frozen=True)
class LosoReportTables:
    """All machine-readable tables needed by the clinical Markdown report."""

    inventory: pd.DataFrame
    per_animal: pd.DataFrame
    per_session: pd.DataFrame
    fold_summary: pd.DataFrame
    pooled: pd.DataFrame
    confusion_matrices: pd.DataFrame
    protocol: dict[str, Any]


def build_loso_report_tables(
    runs_root: str | Path,
    *,
    window_seconds: float = 4.0,
) -> LosoReportTables:
    """Collect completed LOSO artifacts without changing any training state."""

    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    root = Path(runs_root).resolve()
    queue_path = root / "queue_manifest.json"
    queue = _read_json(queue_path)
    tasks = queue.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"{queue_path} must contain a non-empty tasks list")

    inventory_rows: list[dict[str, Any]] = []
    completed: list[_CompletedRun] = []
    expected_animals: set[str] = set()
    expected_by_model: dict[str, int] = {}

    for task in sorted(
        tasks,
        key=lambda item: (str(item.get("model", "")), int(item.get("fold", 0))),
    ):
        model = _required_text(task, "model", queue_path)
        fold = int(task.get("fold", 0))
        if fold < 1:
            raise ValueError(f"invalid fold in {queue_path}: {task.get('fold')!r}")
        test_animal = _required_text(task, "test_animal", queue_path)
        expected_animals.add(test_animal)
        expected_by_model[model] = expected_by_model.get(model, 0) + 1
        task_dir = root / model / f"fold_{fold:02d}_test_{test_animal}"
        status_path = task_dir / "launcher_status.json"
        status_payload: dict[str, Any] = {}
        status = "pending"
        if status_path.is_file():
            status_payload = _read_json(status_path)
            status = str(status_payload.get("status", "unknown"))

        run_dir: Path | None = None
        artifact_state = "not_available"
        if status == "completed":
            run_dir = _resolve_completed_run_dir(task_dir, status_payload)
            run = _load_completed_run(
                model=model,
                fold=fold,
                test_animal=test_animal,
                run_dir=run_dir,
            )
            completed.append(run)
            artifact_state = "complete"

        inventory_rows.append(
            {
                "model": model,
                "fold": fold,
                "test_animal": test_animal,
                "validation_animal": task.get("val_animal"),
                "launcher_status": status,
                "artifact_state": artifact_state,
                "run_dir": str(run_dir) if run_dir is not None else None,
            }
        )

    if not completed:
        raise ValueError(f"no completed LOSO artifacts found under {root}")

    per_animal_rows: list[dict[str, Any]] = []
    per_session_rows: list[dict[str, Any]] = []
    prediction_parts: dict[str, list[pd.DataFrame]] = {}

    for run in completed:
        test = _required_mapping(run.metrics, "test", run.run_dir / "metrics.json")
        source_window = _required_mapping(
            test, "window", run.run_dir / "metrics.json"
        )
        source_event = test.get("event")
        if source_event is not None and not isinstance(source_event, Mapping):
            raise ValueError(f"test.event must be an object in {run.run_dir}")
        source_sessions = test.get("sessions", {})
        if not isinstance(source_sessions, Mapping):
            raise ValueError(f"test.sessions must be an object in {run.run_dir}")

        test_predictions = run.predictions.copy()
        test_predictions["predicted_label"] = (
            test_predictions["seizure_probability"].to_numpy(dtype=float)
            >= run.threshold
        ).astype(np.int8)
        test_predictions["fold"] = run.fold
        test_predictions["threshold"] = run.threshold
        prediction_parts.setdefault(run.model, []).append(
            test_predictions[
                [
                    "label",
                    "seizure_probability",
                    "predicted_label",
                    "session_id",
                    "start_time_ms",
                ]
            ].copy()
        )

        session_durations: list[float] = []
        for session_id, group in test_predictions.groupby("session_id", sort=True):
            session_id = str(session_id)
            group_animal = set(group["animal_id"].astype(str))
            if group_animal != {run.test_animal}:
                raise ValueError(
                    f"session {session_id} in {run.run_dir} does not belong only "
                    f"to test animal {run.test_animal}: {sorted(group_animal)}"
                )
            window = window_metrics(
                group["label"].to_numpy(),
                group["seizure_probability"].to_numpy(),
                run.threshold,
            )
            duration_hours = _duration_hours(
                group["start_time_ms"].to_numpy(dtype=float), window_seconds
            )
            session_durations.append(duration_hours)
            event = source_sessions.get(session_id)
            if event is not None and not isinstance(event, Mapping):
                raise ValueError(
                    f"test.sessions.{session_id} must be an object in {run.run_dir}"
                )
            row: dict[str, Any] = {
                "model": run.model,
                "fold": run.fold,
                "test_animal": run.test_animal,
                "session_id": session_id,
                "threshold_selected_on_validation": run.threshold,
                "duration_hours": duration_hours,
                "run_dir": str(run.run_dir),
            }
            row.update({key: window.get(key) for key in ("n_windows", "positive_windows", *_WINDOW_METRICS)})
            row.update(
                {
                    "true_events": _metric_or_none(event, "true_events"),
                    "predicted_events": _metric_or_none(event, "predicted_events"),
                    "false_alarms": _metric_or_none(event, "false_alarms"),
                    "event_sensitivity": _metric_or_none(
                        event, "event_sensitivity"
                    ),
                    "false_alarms_per_hour": _metric_or_none(
                        event, "false_alarms_per_hour"
                    ),
                    "median_latency_s": _metric_or_none(
                        event, "median_latency_s"
                    ),
                }
            )
            per_session_rows.append(row)

        expected_windows = _metric_or_none(source_window, "n_windows")
        if expected_windows is not None and int(expected_windows) != len(test_predictions):
            raise ValueError(
                f"test prediction count does not match metrics.json in {run.run_dir}: "
                f"{len(test_predictions)} != {int(expected_windows)}"
            )
        animal_row: dict[str, Any] = {
            "model": run.model,
            "fold": run.fold,
            "test_animal": run.test_animal,
            "n_sessions": int(test_predictions["session_id"].nunique()),
            "threshold_selected_on_validation": run.threshold,
            "duration_hours": float(sum(session_durations)),
            "run_dir": str(run.run_dir),
        }
        animal_row.update(
            {
                key: _metric_or_none(source_window, key)
                for key in ("n_windows", "positive_windows", "tp", "tn", "fp", "fn", *_WINDOW_METRICS)
            }
        )
        animal_row.update(
            {
                "true_events": _metric_or_none(source_event, "true_events"),
                "predicted_events": _metric_or_none(
                    source_event, "predicted_events"
                ),
                "false_alarms": _metric_or_none(source_event, "false_alarms"),
                "event_sensitivity": _metric_or_none(
                    source_event, "event_sensitivity"
                ),
                "false_alarms_per_hour": _metric_or_none(
                    source_event, "false_alarms_per_hour"
                ),
                "median_session_latency_s": _metric_or_none(
                    source_event, "median_session_latency_s"
                ),
            }
        )
        per_animal_rows.append(animal_row)

    inventory = pd.DataFrame(inventory_rows)
    per_animal = pd.DataFrame(per_animal_rows).sort_values(
        ["model", "fold"], ignore_index=True
    )
    per_session = pd.DataFrame(per_session_rows).sort_values(
        ["model", "fold", "session_id"], ignore_index=True
    )
    fold_summary = _fold_summary(per_animal, expected_by_model)
    pooled, confusions = _pooled_tables(
        per_animal,
        per_session,
        prediction_parts,
        expected_by_model,
    )
    protocol = {
        "design": "animal-level LOSO",
        "cohort": "acute_post_drug_single_day_20260701",
        "collection_date": "2026-07-01",
        "same_day_acute_post_drug": True,
        "cross_date_evidence": False,
        "split_unit": "animal_id",
        "expected_animals": sorted(expected_animals),
        "expected_animal_count": len(expected_animals),
        "completed_runs": len(completed),
        "expected_runs": len(tasks),
        "window_seconds_used_for_reported_duration": float(window_seconds),
        "source_queue_manifest": str(queue_path),
    }
    return LosoReportTables(
        inventory=inventory,
        per_animal=per_animal,
        per_session=per_session,
        fold_summary=fold_summary,
        pooled=pooled,
        confusion_matrices=confusions,
        protocol=protocol,
    )


def generate_loso_report(
    runs_root: str | Path,
    output_dir: str | Path | None = None,
    *,
    window_seconds: float = 4.0,
) -> dict[str, Path]:
    """Write CSV tables, protocol metadata, and a Chinese clinical report."""

    root = Path(runs_root).resolve()
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "clinical_report"
    )
    tables = build_loso_report_tables(root, window_seconds=window_seconds)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "inventory": destination / "run_inventory.csv",
        "per_animal": destination / "per_animal_metrics.csv",
        "per_session": destination / "per_session_metrics.csv",
        "fold_summary": destination / "model_fold_summary.csv",
        "pooled": destination / "model_pooled_metrics.csv",
        "confusion_matrices": destination / "confusion_matrices.csv",
        "protocol": destination / "report_manifest.json",
        "clinical_report": destination / "clinical_report_zh.md",
    }
    for name in (
        "inventory",
        "per_animal",
        "per_session",
        "fold_summary",
        "pooled",
        "confusion_matrices",
    ):
        getattr(tables, name).to_csv(paths[name], index=False, encoding="utf-8-sig")
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": tables.protocol,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["protocol"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    paths["clinical_report"].write_text(
        render_clinical_markdown(tables), encoding="utf-8", newline="\n"
    )
    return paths


def render_clinical_markdown(tables: LosoReportTables) -> str:
    """Render a concise, claim-safe Chinese report for clinical review."""

    protocol = tables.protocol
    animals = ", ".join(protocol["expected_animals"])
    lines = [
        "# 小鼠癫痫检测 LOSO 结果汇总",
        "",
        "## 研究边界",
        "",
        f"- 本实验包含 {protocol['expected_animal_count']} 只动物：{animals}。划分单位是动物，不是文件片段。",
        "- 所有记录均来自 2026-07-01 同一天，为急性给药后采集（same-day acute post-drug）。",
        "- 本结果只检验同日条件下对未见动物的迁移；没有跨日期证据（no cross-date evidence）。",
        f"- 当前报告纳入 {protocol['completed_runs']}/{protocol['expected_runs']} 个已完成模型折次；未完成折次不进入指标计算。",
        "- 每折分类阈值只由该折验证动物选择，再原样用于测试动物。",
        "",
        "## 模型总览",
        "",
        "下表为动物折之间的非加权均值 ± 样本标准差，反映不同测试动物之间的波动。只有 1 折时标准差记为未提供。",
        "",
        "| 模型 | 完成折/计划折 | Accuracy | Sensitivity | Specificity | F1 | Balanced Acc. | MCC | AUROC | AUPRC | Event sensitivity | False alarms/h | Latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tables.fold_summary.to_dict(orient="records"):
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row["model"]),
                    f"{int(row['completed_folds'])}/{int(row['expected_folds'])}",
                    _mean_std(row, "accuracy"),
                    _mean_std(row, "sensitivity"),
                    _mean_std(row, "specificity"),
                    _mean_std(row, "f1"),
                    _mean_std(row, "balanced_accuracy"),
                    _mean_std(row, "mcc"),
                    _mean_std(row, "auroc"),
                    _mean_std(row, "auprc"),
                    _mean_std(row, "event_sensitivity"),
                    _mean_std(row, "false_alarms_per_hour"),
                    _mean_std(row, "median_session_latency_s"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 每只测试动物",
            "",
            "| 模型 | 折 | 动物 | 片段数 | 阈值 | Accuracy | Sensitivity | Specificity | F1 | AUROC | AUPRC | Event sensitivity | False alarms/h | Latency (s) |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in tables.per_animal.to_dict(orient="records"):
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row["model"]),
                    str(int(row["fold"])),
                    _md(row["test_animal"]),
                    str(int(row["n_sessions"])),
                    _format_number(row["threshold_selected_on_validation"]),
                    _format_number(row.get("accuracy")),
                    _format_number(row.get("sensitivity")),
                    _format_number(row.get("specificity")),
                    _format_number(row.get("f1")),
                    _format_number(row.get("auroc")),
                    _format_number(row.get("auprc")),
                    _format_number(row.get("event_sensitivity")),
                    _format_number(row.get("false_alarms_per_hour")),
                    _format_number(row.get("median_session_latency_s")),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Pooled 留一动物测试结果", ""])
    for row in tables.pooled.to_dict(orient="records"):
        lines.extend(
            [
                f"### {_md(row['model'])}",
                "",
                f"纳入 {int(row['animals_evaluated'])} 只测试动物、{int(row['n_sessions'])} 个记录片段、{int(row['n_windows']):,} 个窗口。",
                "",
                "| Accuracy | Sensitivity | Specificity | F1 | Balanced Acc. | MCC | AUROC | AUPRC | Event sensitivity | False alarms/h | Session latency median (s) |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                "| "
                + " | ".join(
                    _format_number(row.get(key))
                    for key in (
                        "accuracy",
                        "sensitivity",
                        "specificity",
                        "f1",
                        "balanced_accuracy",
                        "mcc",
                        "auroc",
                        "auprc",
                        "event_sensitivity",
                        "false_alarms_per_hour",
                        "median_of_session_median_latency_s",
                    )
                )
                + " |",
                "",
            ]
        )
        confusion = tables.confusion_matrices.loc[
            tables.confusion_matrices["model"].eq(row["model"])
        ].iloc[0]
        lines.extend(
            [
                "混淆矩阵（行是真实类别，列是模型预测）：",
                "",
                "| 真实 \\ 预测 | 非发作 | 发作 |",
                "|---|---:|---:|",
                f"| 非发作 | {int(confusion['tn']):,} | {int(confusion['fp']):,} |",
                f"| 发作 | {int(confusion['fn']):,} | {int(confusion['tp']):,} |",
                "",
            ]
        )

    lines.extend(
        [
            "## 指标口径与临床解读",
            "",
            "- Accuracy 是全部窗口的正确比例；类别不平衡时不能单独使用。Sensitivity 表示真实发作窗口检出的比例，Specificity 表示真实非发作窗口排除的比例。",
            "- F1 同时考虑发作检出和误报；Balanced Accuracy 对发作与非发作同等计权；MCC 在类别不平衡时更稳健。",
            "- AUROC/AUPRC 使用连续发作概率。AUPRC 对发作样本较少的任务通常更有解释力。",
            "- pooled 的离散分类指标使用每折各自在验证动物上选定的阈值，再合并所有留一动物测试预测；没有用测试动物重新调阈值。",
            "- 每个 LOSO 折都是独立训练的模型，折间概率校准可能不同，因此 pooled AUROC/AUPRC 仅作描述；动物折均值 ± 标准差是判断跨动物稳定性的主要结果。",
            "- Event sensitivity、false alarms/hour 和 latency 沿用训练产物中的事件合并算法。pooled latency 是各记录片段 latency 中位数的再次取中位数，不等于所有单次发作延迟的精确总体中位数。",
            "- 4 秒窗口以 1 秒步长重叠，窗口彼此相关，不能把窗口数当作独立动物样本量。真正的生物学样本量仍是 6 只动物。",
            "- 当前若尚未完成全部折次，本报告只是训练进度快照，不能作为两个模型的最终排名。",
            "",
        ]
    )
    return "\n".join(lines)


def _fold_summary(
    per_animal: pd.DataFrame, expected_by_model: Mapping[str, int]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = (
        "threshold_selected_on_validation",
        *_WINDOW_METRICS,
        *_EVENT_RATE_METRICS,
    )
    for model in sorted(expected_by_model):
        group = per_animal.loc[per_animal["model"].eq(model)]
        row: dict[str, Any] = {
            "model": model,
            "completed_folds": int(len(group)),
            "expected_folds": int(expected_by_model[model]),
            "complete": bool(len(group) == expected_by_model[model]),
        }
        for metric in metrics:
            values = pd.to_numeric(group.get(metric), errors="coerce")
            values = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else np.nan
            )
            row[f"{metric}_n"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows)


def _pooled_tables(
    per_animal: pd.DataFrame,
    per_session: pd.DataFrame,
    prediction_parts: Mapping[str, list[pd.DataFrame]],
    expected_by_model: Mapping[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for model, parts in sorted(prediction_parts.items()):
        predictions = pd.concat(parts, ignore_index=True)
        y_true = predictions["label"].to_numpy(dtype=np.int8)
        probability = predictions["seizure_probability"].to_numpy(dtype=float)
        predicted = predictions["predicted_label"].to_numpy(dtype=np.int8)
        tn = int(np.sum((y_true == 0) & (predicted == 0)))
        fp = int(np.sum((y_true == 0) & (predicted == 1)))
        fn = int(np.sum((y_true == 1) & (predicted == 0)))
        tp = int(np.sum((y_true == 1) & (predicted == 1)))
        discrete = _metrics_from_confusion(tn=tn, fp=fp, fn=fn, tp=tp)
        if len(np.unique(y_true)) == 2:
            auroc = float(roc_auc_score(y_true, probability))
            auprc = float(average_precision_score(y_true, probability))
        else:
            auroc = np.nan
            auprc = np.nan

        animal = per_animal.loc[per_animal["model"].eq(model)].copy()
        sessions = per_session.loc[per_session["model"].eq(model)].copy()
        event_available = animal["true_events"].notna()
        event_animals = animal.loc[event_available]
        true_events = int(event_animals["true_events"].sum()) if len(event_animals) else 0
        predicted_events = (
            int(event_animals["predicted_events"].sum()) if len(event_animals) else 0
        )
        false_alarms = (
            int(event_animals["false_alarms"].sum()) if len(event_animals) else 0
        )
        weighted_matches = (
            event_animals["event_sensitivity"].astype(float)
            * event_animals["true_events"].astype(float)
        ).sum()
        event_sensitivity = (
            float(weighted_matches / true_events) if true_events else np.nan
        )
        event_duration = float(event_animals["duration_hours"].sum())
        false_alarms_per_hour = (
            float(false_alarms / event_duration) if event_duration > 0 else np.nan
        )
        session_latencies = pd.to_numeric(
            sessions.get("median_latency_s"), errors="coerce"
        )
        session_latencies = session_latencies[np.isfinite(session_latencies)]
        latency = (
            float(np.median(session_latencies)) if len(session_latencies) else np.nan
        )
        thresholds = animal["threshold_selected_on_validation"].astype(float)
        row: dict[str, Any] = {
            "model": model,
            "completed_folds": int(len(animal)),
            "expected_folds": int(expected_by_model[model]),
            "complete": bool(len(animal) == expected_by_model[model]),
            "animals_evaluated": int(animal["test_animal"].nunique()),
            "n_sessions": int(len(sessions)),
            "n_windows": int(len(y_true)),
            "positive_windows": int(y_true.sum()),
            "threshold_mode": "per_fold_validation_selected",
            "threshold_mean": float(thresholds.mean()),
            "threshold_std": (
                float(thresholds.std(ddof=1)) if len(thresholds) > 1 else np.nan
            ),
            **discrete,
            "auroc": auroc,
            "auprc": auprc,
            "event_folds_available": int(len(event_animals)),
            "true_events": true_events if len(event_animals) else np.nan,
            "predicted_events": predicted_events if len(event_animals) else np.nan,
            "false_alarms": false_alarms if len(event_animals) else np.nan,
            "event_sensitivity": event_sensitivity,
            "false_alarms_per_hour": false_alarms_per_hour,
            "median_of_session_median_latency_s": latency,
            "duration_hours": float(animal["duration_hours"].sum()),
        }
        pooled_rows.append(row)
        confusion_rows.append(
            {
                "model": model,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "nonseizure_recall": discrete["specificity"],
                "seizure_recall": discrete["sensitivity"],
                "n_windows": int(len(y_true)),
                "threshold_mode": "per_fold_validation_selected",
            }
        )
    return pd.DataFrame(pooled_rows), pd.DataFrame(confusion_rows)


def _metrics_from_confusion(*, tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    total = tn + fp + fn + tp
    sensitivity = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    denominator = math.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": _safe_ratio(tp + tn, total),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
        "mcc": float((tp * tn - fp * fn) / denominator)
        if denominator
        else np.nan,
    }


def _load_completed_run(
    *, model: str, fold: int, test_animal: str, run_dir: Path
) -> _CompletedRun:
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.csv"
    run_manifest = _read_json(run_dir / "run_manifest.json")
    manifest_model = run_manifest.get("model")
    if manifest_model is not None and str(manifest_model) != model:
        raise ValueError(
            f"model in {run_dir / 'run_manifest.json'} does not match queue task: "
            f"{manifest_model!r} != {model!r}"
        )
    metrics = _read_json(metrics_path)
    threshold = _finite_float(
        metrics.get("threshold_selected_on_validation"),
        f"threshold_selected_on_validation in {metrics_path}",
    )
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be between 0 and 1 in {metrics_path}")
    predictions = pd.read_csv(predictions_path, encoding="utf-8-sig")
    missing = sorted(_REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns))
    if missing:
        raise ValueError(
            f"{predictions_path} is missing columns: {', '.join(missing)}"
        )
    if "split" in predictions.columns:
        predictions = predictions.loc[predictions["split"].astype(str).eq("test")]
    if predictions.empty:
        raise ValueError(f"{predictions_path} contains no test predictions")
    prediction_animals = set(predictions["animal_id"].astype(str))
    if prediction_animals != {test_animal}:
        raise ValueError(
            f"test predictions in {predictions_path} do not match frozen test animal "
            f"{test_animal}: {sorted(prediction_animals)}"
        )
    if not predictions["label"].isin([0, 1]).all():
        raise ValueError(f"{predictions_path} contains labels outside 0/1")
    probabilities = pd.to_numeric(
        predictions["seizure_probability"], errors="coerce"
    ).to_numpy()
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError(f"{predictions_path} contains invalid seizure probabilities")
    return _CompletedRun(
        model=model,
        fold=fold,
        test_animal=test_animal,
        run_dir=run_dir,
        threshold=threshold,
        metrics=metrics,
        predictions=predictions.reset_index(drop=True),
    )


def _resolve_completed_run_dir(
    task_dir: Path, status_payload: Mapping[str, Any]
) -> Path:
    raw_run_dir = status_payload.get("run_dir")
    if raw_run_dir:
        candidate = Path(str(raw_run_dir))
        if _is_complete_artifact_dir(candidate):
            return candidate.resolve()
    candidates = [
        path.parent
        for path in (task_dir / "artifacts").glob("*/metrics.json")
        if _is_complete_artifact_dir(path.parent)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"launcher marks task completed but no complete artifact directory exists: {task_dir}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def _is_complete_artifact_dir(path: Path) -> bool:
    return path.is_dir() and all(
        (path / name).is_file()
        for name in ("metrics.json", "predictions.csv", "history.csv", "run_manifest.json")
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"top-level JSON in {path} must be an object")
    return value


def _required_mapping(
    value: Mapping[str, Any], key: str, source: Path
) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"{source} must contain an object at {key}")
    return item


def _required_text(value: Mapping[str, Any], key: str, source: Path) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{source} must contain a non-empty string at {key}")
    return item.strip()


def _metric_or_none(value: Any, key: str) -> float | int | None:
    if not isinstance(value, Mapping) or key not in value:
        return None
    raw = value[key]
    if raw is None or isinstance(raw, bool):
        return None
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if isinstance(raw, int):
        return int(raw)
    return number


def _finite_float(value: Any, label: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _duration_hours(start_time_ms: np.ndarray, window_seconds: float) -> float:
    starts = np.asarray(start_time_ms, dtype=float)
    if len(starts) == 0:
        return 0.0
    return float(
        (starts.max() - starts.min() + window_seconds * 1000.0) / 3_600_000.0
    )


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def _mean_std(row: Mapping[str, Any], metric: str) -> str:
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if not _is_finite(mean):
        return "未提供"
    if not _is_finite(std):
        return f"{float(mean):.3f}"
    return f"{float(mean):.3f} ± {float(std):.3f}"


def _format_number(value: Any) -> str:
    return f"{float(value):.3f}" if _is_finite(value) else "未提供"


def _is_finite(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
