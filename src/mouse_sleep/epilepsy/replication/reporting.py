"""Structured, claim-safe reporting for public-dataset replication runs."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "build_replication_summary",
    "generate_replication_report",
    "render_chinese_markdown",
]


_REPORTED_METRICS = {
    "accuracy": "accuracy",
    "macro_f1": "macro_f1",
    "balanced_accuracy": "balanced_accuracy",
    "auroc": "auroc",
    "auprc": "auprc",
    "seizure_recall": "recall_class_1",
}
_STATS = ("mean", "std", "min", "max", "n")
_THRESHOLD_MODES = (
    {
        "key": "default_argmax",
        "label_zh": "默认 argmax（二分类等价于 0.5）",
        "prefix": "test",
        "threshold_source": "default_argmax",
    },
    {
        "key": "validation_calibrated",
        "label_zh": "验证集校准（Youden J）",
        "prefix": "test_calibrated",
        "threshold_source": "validation_youden_j",
    },
)


def build_replication_summary(
    aggregate_paths: str | Path | Iterable[str | Path],
) -> pd.DataFrame:
    """Convert one or more ``aggregate_metrics.json`` files to tidy rows.

    Each source run contributes two adjacent rows: its default argmax decision
    and its validation-calibrated decision.  A missing calibrated result remains
    an explicit row with ``metrics_available=False`` and missing metric values.
    This makes old and current runs comparable without silently substituting
    zero for an experiment that was never run.
    """

    paths = _coerce_paths(aggregate_paths)
    rows: list[dict[str, Any]] = []
    for path in paths:
        manifest = _read_manifest(path)
        aggregate = manifest["aggregate_metrics"]
        default_means: dict[str, float | None] = {}
        run_rows: list[dict[str, Any]] = []

        for mode in _THRESHOLD_MODES:
            prefix = str(mode["prefix"])
            row: dict[str, Any] = {
                "source_path": str(path.resolve()),
                "run_id": path.parent.name,
                "created_utc": manifest.get("created_utc"),
                "model": manifest.get("model"),
                "normalization": manifest.get("normalization"),
                "folds": manifest.get("folds"),
                "threshold_mode": mode["key"],
                "threshold_label_zh": mode["label_zh"],
                "threshold_source": mode["threshold_source"],
                "replication_status": manifest.get("replication_status"),
                "paper_performance_claim_allowed": manifest.get(
                    "paper_performance_claim_allowed"
                ),
            }
            if mode["key"] == "default_argmax":
                row.update(
                    {
                        "decision_threshold_mean": 0.5,
                        "decision_threshold_std": 0.0,
                        "decision_threshold_min": 0.5,
                        "decision_threshold_max": 0.5,
                        "decision_threshold_n": manifest.get("folds"),
                    }
                )
            else:
                threshold_key = f"{prefix}_decision_threshold"
                threshold_stats = aggregate.get(threshold_key)
                if threshold_stats is None:
                    threshold_stats = aggregate.get(
                        "validation_calibrated_decision_threshold"
                    )
                _copy_stats(row, "decision_threshold", threshold_stats)

            for output_name, aggregate_name in _REPORTED_METRICS.items():
                metric_stats = aggregate.get(f"{prefix}_{aggregate_name}")
                _copy_stats(row, output_name, metric_stats)
                if mode["key"] == "default_argmax":
                    default_means[output_name] = row[f"{output_name}_mean"]

            row["metrics_available"] = row["accuracy_mean"] is not None
            run_rows.append(row)

        for row in run_rows:
            for metric_name in _REPORTED_METRICS:
                value = row[f"{metric_name}_mean"]
                baseline = default_means.get(metric_name)
                row[f"{metric_name}_delta_vs_default"] = (
                    float(value) - float(baseline)
                    if value is not None and baseline is not None
                    else None
                )
        rows.extend(run_rows)

    return pd.DataFrame(rows, columns=_summary_columns())


def render_chinese_markdown(summary: pd.DataFrame) -> str:
    """Render the tidy summary as a clinician-readable Chinese Markdown report."""

    required = set(_summary_columns())
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"summary is missing required columns: {', '.join(missing)}")
    if summary.empty:
        raise ValueError("summary must contain at least one result row")

    lines = [
        "# 癫痫模型公共数据复现结果汇总",
        "",
        "> 本表只汇总输入文件中已经保存的结果与科研状态，不会据此自动认定论文复现成功。",
        "",
        "## 测试集结果",
        "",
        "| 运行 | 模型 | 归一化 | 阈值方案 | 阈值 | Accuracy | Macro-F1 | Balanced Accuracy | AUROC | AUPRC | 发作 Recall | 复现状态 | 可声明论文性能 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary.to_dict(orient="records"):
        available = bool(row["metrics_available"])
        metrics = [
            _format_mean_std(row, "accuracy") if available else "未提供",
            _format_mean_std(row, "macro_f1") if available else "未提供",
            _format_mean_std(row, "balanced_accuracy") if available else "未提供",
            _format_mean_std(row, "auroc") if available else "未提供",
            _format_mean_std(row, "auprc") if available else "未提供",
            _format_mean_std(row, "seizure_recall") if available else "未提供",
        ]
        values = [
            _markdown_escape(row["run_id"]),
            _markdown_escape(row["model"]),
            _markdown_escape(row["normalization"]),
            _markdown_escape(row["threshold_label_zh"]),
            _format_mean_std(row, "decision_threshold"),
            *metrics,
            _markdown_escape(row["replication_status"]),
            _claim_label(row["paper_performance_claim_allowed"]),
        ]
        lines.append("| " + " | ".join(values) + " |")

    statuses = sorted(
        {
            str(value)
            for value in summary["replication_status"].dropna().tolist()
        }
    )
    claim_values = summary["paper_performance_claim_allowed"].dropna().tolist()
    lines.extend(
        [
            "",
            "## 结果边界",
            "",
            f"- 复现状态：{', '.join(statuses) if statuses else '未声明'}。",
            "- 论文性能声明："
            + (
                "所有输入运行均允许。"
                if claim_values and all(value is True for value in claim_values)
                else "至少一个运行不允许或未明确允许，因此不能把本表直接作为达到原论文性能的证据。"
            ),
            "",
            "## 阈值与指标说明",
            "",
            "- 默认方案使用最大概率类别；在二分类 Softmax 输出下等价于发作概率阈值 0.5。",
            "- 验证集校准方案只使用验证集标签按 Youden J 选择阈值，再原样应用到测试集；没有校准结果时显示“未提供”。",
            "- Accuracy 是总体正确率；Macro-F1 对两类等权；Balanced Accuracy 是两类召回率的平均值。",
            "- AUROC 和 AUPRC 衡量连续发作概率的排序能力，不随单个分类阈值改变，因此两种阈值方案中通常相同。",
            "- 发作 Recall 是标签 1 的召回率，表示真实发作片段中被识别出的比例。数值均为跨折均值 ± 样本标准差。",
            "",
        ]
    )
    return "\n".join(lines)


def generate_replication_report(
    aggregate_paths: str | Path | Iterable[str | Path],
    output_dir: str | Path,
    *,
    csv_name: str = "summary.csv",
    markdown_name: str = "summary_zh.md",
) -> tuple[Path, Path]:
    """Write a UTF-8 CSV and Chinese Markdown report, returning both paths."""

    if Path(csv_name).name != csv_name or not csv_name.lower().endswith(".csv"):
        raise ValueError("csv_name must be a plain .csv filename")
    if Path(markdown_name).name != markdown_name or not markdown_name.lower().endswith(
        ".md"
    ):
        raise ValueError("markdown_name must be a plain .md filename")

    summary = build_replication_summary(aggregate_paths)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / csv_name
    markdown_path = destination / markdown_name
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    markdown_path.write_text(
        render_chinese_markdown(summary), encoding="utf-8", newline="\n"
    )
    return csv_path, markdown_path


def _coerce_paths(
    aggregate_paths: str | Path | Iterable[str | Path],
) -> list[Path]:
    if isinstance(aggregate_paths, str | Path):
        paths = [Path(aggregate_paths)]
    else:
        paths = [Path(path) for path in aggregate_paths]
    if not paths:
        raise ValueError("at least one aggregate_metrics.json path is required")
    return paths


def _read_manifest(path: Path) -> dict[str, Any]:
    if path.name != "aggregate_metrics.json":
        raise ValueError(f"expected aggregate_metrics.json, received {path.name!r}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"top-level JSON in {path} must be an object")
    if not isinstance(value.get("aggregate_metrics"), dict):
        raise ValueError(f"{path} does not contain an aggregate_metrics object")
    claim = value.get("paper_performance_claim_allowed")
    if claim is not None and not isinstance(claim, bool):
        raise ValueError(
            f"paper_performance_claim_allowed in {path} must be boolean or null"
        )
    return value


def _copy_stats(
    row: dict[str, Any], name: str, value: Any
) -> None:
    if value is None:
        for stat in _STATS:
            row[f"{name}_{stat}"] = None
        return
    if isinstance(value, Mapping):
        for stat in _STATS:
            raw = value.get(stat)
            row[f"{name}_{stat}"] = _finite_number_or_none(raw)
        return
    row[f"{name}_mean"] = _finite_number_or_none(value)
    for stat in _STATS[1:]:
        row[f"{name}_{stat}"] = None


def _finite_number_or_none(value: Any) -> float | int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    if isinstance(value, int):
        return int(value)
    return numeric


def _format_mean_std(row: Mapping[str, Any], metric: str) -> str:
    mean = _finite_number_or_none(row.get(f"{metric}_mean"))
    std = _finite_number_or_none(row.get(f"{metric}_std"))
    if mean is None:
        return "未提供"
    if std is None:
        return f"{float(mean):.3f}"
    return f"{float(mean):.3f} ± {float(std):.3f}"


def _markdown_escape(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "未声明"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _claim_label(value: Any) -> str:
    if value is True:
        return "允许"
    if value is False:
        return "不允许"
    return "未声明"


def _summary_columns() -> list[str]:
    columns = [
        "source_path",
        "run_id",
        "created_utc",
        "model",
        "normalization",
        "folds",
        "threshold_mode",
        "threshold_label_zh",
        "threshold_source",
        "metrics_available",
        "replication_status",
        "paper_performance_claim_allowed",
    ]
    for metric in ("decision_threshold", *_REPORTED_METRICS):
        columns.extend(f"{metric}_{stat}" for stat in _STATS)
    columns.extend(
        f"{metric}_delta_vs_default" for metric in _REPORTED_METRICS
    )
    return columns
