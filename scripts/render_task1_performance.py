#!/usr/bin/env python
"""Assemble the canonical Task-1 performance figures beside corrected decoding.

The corrected decoder changes continuous event intervals, not the frozen model
probabilities.  This renderer therefore keeps window discrimination and event
detection in separate figures and records the source of every artifact.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "legend.fontsize": 6.2,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    PROJECT_ROOT
    / "output"
    / "epilepsy_program_20260821"
    / "01_annotation_detection"
    / "04_model_runs"
    / "combined_old_plus_new"
)
CORRECTED_ROOT = (
    PROJECT_ROOT
    / "output"
    / "epilepsy_program_20260821"
    / "01_annotation_detection"
    / "06_decoder_audit_v2"
    / "corrected_combined_old_plus_new"
)
OUT_ROOT = CORRECTED_ROOT / "figures" / "model_performance"
SOURCE_ROOT = OUT_ROOT / "source_data"

MODELS = ("eegwavenet_sc", "physio_sequence_no_aux", "physio_sequence_stats")
LABELS = {
    "eegwavenet_sc": "EEGWaveNet-SC",
    "physio_sequence_no_aux": "Physio-Sequence",
    "physio_sequence_stats": "Physio + statistics",
}
COLORS = {
    "eegwavenet_sc": "#59636F",
    "physio_sequence_no_aux": "#2F6B8A",
    "physio_sequence_stats": "#C56B3D",
}
MARKERS = {
    "eegwavenet_sc": "o",
    "physio_sequence_no_aux": "s",
    "physio_sequence_stats": "^",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_axis(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.55, pad=2)
    if grid:
        ax.grid(axis=grid, color="#D7DADD", linewidth=0.45, alpha=0.8)
        ax.set_axisbelow(True)


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def _copy_window_figure_and_sources() -> dict[str, object]:
    """Keep the old baseline figure as an explicitly labelled reference."""

    source_dir = RUN_ROOT / "figures" / "deep_learning_metrics"
    source_stem = source_dir / "figure_3_deep_learning_metrics"
    target_stem = OUT_ROOT / "figure_0_baseline_window_classification_confusion"
    copied: list[str] = []
    source_hashes: dict[str, str] = {}
    for suffix in (".png", ".pdf", ".svg"):
        source = source_stem.with_suffix(suffix)
        target = target_stem.with_suffix(suffix)
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, target)
        copied.append(str(target))
        source_hashes[str(source)] = _sha256(source)

    with Image.open(target_stem.with_suffix(".png")) as image:
        image.save(
            target_stem.with_suffix(".tiff"),
            compression="tiff_lzw",
            dpi=(300, 300),
        )
    copied.append(str(target_stem.with_suffix(".tiff")))

    for filename in (
        "window_metrics.csv",
        "confusion_matrices.csv",
        "calibration_bins.csv",
        "per_animal_window_metrics.csv",
    ):
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, SOURCE_ROOT / filename)
            source_hashes[str(source)] = _sha256(source)

    return {
        "semantic_level": "held-out 4-s window classification",
        "version": "canonical events_baseline reference",
        "decoder_dependency": "none",
        "source_stem": str(source_stem),
        "source_sha256": source_hashes,
        "outputs": copied,
    }


def _read_corrected_windows(model: str) -> pd.DataFrame:
    """Read the one supervised-center row per model window from corrected output."""

    path = (
        CORRECTED_ROOT
        / "models"
        / model
        / "events_corrected_v1_test"
        / "decoded_tracks.parquet"
    )
    columns = [
        "sample_id",
        "label",
        "split_safe_probability",
        "decoded_mask",
        "supervised_center_eligible",
        "formal_metric_eligible_effective",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame.loc[
        frame["supervised_center_eligible"].astype(bool)
        & frame["formal_metric_eligible_effective"].astype(bool)
    ].copy()
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce").astype(int)
    frame["split_safe_probability"] = pd.to_numeric(
        frame["split_safe_probability"], errors="coerce"
    ).clip(0, 1)
    frame["decoded_mask"] = pd.to_numeric(frame["decoded_mask"], errors="coerce").fillna(0).astype(int)
    frame = frame.sort_values("sample_id", kind="stable").reset_index(drop=True)
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"corrected test windows are not unique for {model}")
    if frame.empty or frame["label"].nunique() < 2:
        raise ValueError(f"corrected test windows are incomplete for {model}")
    return frame


def _render_corrected_window_figure() -> dict[str, object]:
    """Render the reference 2x3 layout from corrected probabilities and masks."""

    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    predictions = {model: _read_corrected_windows(model) for model in MODELS}
    rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(2, 3, figsize=(7.2047, 5.75), constrained_layout=False)
    pr_ax, roc_ax, metric_ax, *cm_axes = axes.ravel()
    for model in MODELS:
        frame = predictions[model]
        y = frame["label"].to_numpy(np.int8)
        score = frame["split_safe_probability"].to_numpy(float)
        pred = frame["decoded_mask"].to_numpy(np.int8)
        tn = int(((y == 0) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        tp = int(((y == 1) & (pred == 1)).sum())
        n = max(len(y), 1)
        pos = max(int((y == 1).sum()), 1)
        neg = max(int((y == 0).sum()), 1)
        sensitivity = tp / pos
        specificity = tn / neg
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)
        balanced_accuracy = 0.5 * (sensitivity + specificity)
        mcc_den = float(np.sqrt(float(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))))
        mcc = ((tp * tn) - (fp * fn)) / mcc_den
        auprc = float(average_precision_score(y, score))
        auroc = float(roc_auc_score(y, score))
        rows.append(
            {
                "model": model,
                "model_label": LABELS[model],
                "prediction_state": "corrected_decoded_mask",
                "n_windows": n,
                "positive_windows": int((y == 1).sum()),
                "negative_windows": int((y == 0).sum()),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "accuracy": (tp + tn) / n,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "precision": precision,
                "f1": f1,
                "balanced_accuracy": balanced_accuracy,
                "mcc": mcc,
                "auroc": auroc,
                "auprc": auprc,
            }
        )
        confusion_rows.extend(
            [
                {"model": model, "true_label": 0, "predicted_label": 0, "count": tn},
                {"model": model, "true_label": 0, "predicted_label": 1, "count": fp},
                {"model": model, "true_label": 1, "predicted_label": 0, "count": fn},
                {"model": model, "true_label": 1, "predicted_label": 1, "count": tp},
            ]
        )
        precision_curve, recall_curve, _ = precision_recall_curve(y, score)
        pr_ax.plot(recall_curve, precision_curve, color=COLORS[model], linewidth=1.35, label=f"{LABELS[model]}  {auprc:.3f}")
        fpr, tpr, _ = roc_curve(y, score)
        roc_ax.plot(fpr, tpr, color=COLORS[model], linewidth=1.35, label=f"{LABELS[model]}  {auroc:.3f}")
        metric_ax.plot(
            ["AUPRC", "AUROC", "F1", "BA", "MCC"],
            [auprc, auroc, f1, balanced_accuracy, mcc],
            marker="o",
            markersize=3.5,
            linewidth=1.0,
            color=COLORS[model],
            label=LABELS[model],
        )

    prevalence = float(predictions[MODELS[0]]["label"].mean())
    pr_ax.axhline(prevalence, color="#858B90", linestyle=(0, (3, 2)), linewidth=0.75, label=f"Prevalence {prevalence:.3f}")
    pr_ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision", title="Held-out precision–recall")
    pr_ax.legend(frameon=False, loc="lower left", handlelength=1.6)
    _clean_axis(pr_ax, "both")
    roc_ax.plot((0, 1), (0, 1), color="#858B90", linestyle=(0, (3, 2)), linewidth=0.75)
    roc_ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False-positive rate", ylabel="True-positive rate", title="Held-out ROC")
    roc_ax.legend(frameon=False, loc="lower right", handlelength=1.6)
    _clean_axis(roc_ax, "both")
    metric_ax.set_ylim(-0.05, 1.05)
    metric_ax.set_ylabel("Score")
    metric_ax.set_title("Window metrics after corrected decoder")
    metric_ax.legend(frameon=False, loc="lower left", handlelength=1.4)
    metric_ax.axhline(0, color="#B8BCC0", linewidth=0.55)
    _clean_axis(metric_ax, "y")

    cmap = LinearSegmentedColormap.from_list("confusion_blue", ("#F2F5F7", "#7EA9C8", "#174A6E"))
    metrics_frame = pd.DataFrame(rows).set_index("model")
    for ax, model in zip(cm_axes, MODELS, strict=True):
        values = metrics_frame.loc[model, ["tn", "fp", "fn", "tp"]].to_numpy(float).reshape(2, 2)
        fractions = values / np.maximum(values.sum(axis=1, keepdims=True), 1)
        ax.imshow(fractions, cmap=cmap, vmin=0, vmax=1, aspect="equal")
        for i in range(2):
            for j in range(2):
                value = fractions[i, j]
                ax.text(j, i, f"{int(values[i, j]):,}\n{value * 100:.1f}%", ha="center", va="center", fontsize=6.4, color="white" if value > 0.55 else "#2B2B2B", fontweight="bold" if i == j else "normal")
        ax.set_xticks((0, 1), ("Non-seizure", "Seizure"))
        ax.set_yticks((0, 1), ("Non-seizure", "Seizure"))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(LABELS[model] + "  |  corrected state")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.text(0.01, 0.006, "PR/ROC use corrected split-safe probabilities; fixed-window metrics and matrices use the corrected interval-aware decoded state. Rows are normalized within true class.", fontsize=6.2, color="#55585C")
    fig.subplots_adjust(left=0.075, right=0.995, top=0.93, bottom=0.10, wspace=0.40, hspace=0.52)
    stem = OUT_ROOT / "figure_1_window_classification_confusion"
    _save(fig, stem)
    metrics_frame.reset_index().to_csv(SOURCE_ROOT / "corrected_window_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(SOURCE_ROOT / "corrected_confusion_matrices.csv", index=False)
    return {
        "semantic_level": "held-out 4-s window classification after corrected decoder",
        "decoder_dependency": "corrected interval-aware full-window union",
        "probability_source": "decoded_tracks.parquet::split_safe_probability",
        "state_source": "decoded_tracks.parquet::decoded_mask",
        "outputs": [str(stem.with_suffix(suffix)) for suffix in (".png", ".pdf", ".svg", ".tiff")],
        "metrics": str(SOURCE_ROOT / "corrected_window_metrics.csv"),
        "confusion": str(SOURCE_ROOT / "corrected_confusion_matrices.csv"),
    }


def _copy_training_diagnostics() -> dict[str, object]:
    """Place the unchanged training diagnostics beside the corrected audit."""

    source_dir = RUN_ROOT / "figures" / "legacy_style_deep_learning"
    source_stem = source_dir / "figure_s1_training_diagnostics"
    target_stem = OUT_ROOT / "figure_s1_training_diagnostics_frozen_models"
    outputs: list[str] = []
    source_hashes: dict[str, str] = {}
    for suffix in (".png", ".pdf", ".svg", ".tiff"):
        source = source_stem.with_suffix(suffix)
        if not source.exists():
            continue
        target = target_stem.with_suffix(suffix)
        shutil.copy2(source, target)
        outputs.append(str(target))
        source_hashes[str(source)] = _sha256(source)

    for model in MODELS:
        source = RUN_ROOT / model / "history.csv"
        if source.exists():
            shutil.copy2(source, SOURCE_ROOT / f"history_{model}.csv")
            source_hashes[str(source)] = _sha256(source)

    return {
        "semantic_level": "model optimization history",
        "decoder_dependency": "none",
        "source_stem": str(source_stem),
        "source_sha256": source_hashes,
        "outputs": outputs,
    }


def _load_corrected_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        path = CORRECTED_ROOT / "models" / model / "events_corrected_v1_test" / "event_metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        pooled = payload["metrics"]["pooled"]
        true_events = int(pooled["true_events"])
        predicted_events = int(pooled["predicted_events"])
        matched_events = int(pooled["matched_events"])
        rows.append(
            {
                "model": model,
                "model_label": LABELS[model],
                "true_events": true_events,
                "predicted_events": predicted_events,
                "matched_events": matched_events,
                "missed_events": true_events - matched_events,
                "false_positive_events": int(pooled["false_positive_events"]),
                "event_sensitivity": float(pooled["event_sensitivity"]),
                "event_precision": float(pooled["event_precision"]),
                "event_f1": float(pooled["event_f1"]),
                "mean_iou": float(pooled["mean_iou"]),
                "median_iou": float(pooled["median_iou"]),
                "mean_annotated_event_coverage": float(pooled["mean_annotated_event_coverage"]),
                "median_absolute_onset_error_seconds": float(pooled["median_absolute_onset_error_seconds"]),
                "median_absolute_offset_error_seconds": float(pooled["median_absolute_offset_error_seconds"]),
                "median_absolute_duration_error_seconds": float(pooled["median_absolute_duration_error_seconds"]),
                "false_positive_events_per_hour": float(pooled["false_positive_events_per_hour"]),
                "zero_coverage_events": int(pooled["zero_coverage_events"]),
                "zero_coverage_rate": float(pooled["zero_coverage_rate"]),
                "long_events_ge_30s": int(pooled["long_events_ge_30s"]),
                "long_events_ge_30s_zero_coverage": int(pooled["long_events_ge_30s_zero_coverage"]),
                "long_events_ge_30s_zero_coverage_rate": float(pooled["long_events_ge_30s_zero_coverage_rate"]),
                "fragmentation_rate": float(pooled["fragmentation_rate"]),
                "merged_prediction_rate": float(pooled["merged_prediction_rate"]),
                "formal_event_denominator": int(payload["formal_event_denominator"]),
                "visual_event_denominator": int(payload["clinical_event_denominator"]),
                "source_path": str(path),
                "source_sha256": _sha256(path),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(SOURCE_ROOT / "corrected_event_metrics.csv", index=False)
    return frame


def _plot_lines(
    ax: plt.Axes,
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    labels: tuple[str, ...],
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    x = np.arange(len(columns))
    for _, row in frame.iterrows():
        model = str(row["model"])
        ax.plot(
            x,
            [float(row[column]) for column in columns],
            color=COLORS[model],
            marker=MARKERS[model],
            linewidth=1.1,
            markersize=3.5,
            label=LABELS[model],
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    _clean_axis(ax, "y")


def _build_corrected_event_figure(frame: pd.DataFrame) -> dict[str, object]:
    """Render event metrics plus an honest TP/FN and TP/FP audit matrix."""

    fig, axes = plt.subplots(2, 3, figsize=(7.2047, 5.85), constrained_layout=False)
    metric_ax, overlap_ax, boundary_ax, *audit_axes = axes.ravel()

    _plot_lines(
        metric_ax,
        frame,
        ("event_sensitivity", "event_precision", "event_f1"),
        ("Sensitivity", "Precision", "F1"),
        "Corrected event detection",
        "Score",
        (0, 1.02),
    )
    metric_ax.legend(frameon=False, loc="lower left", handlelength=1.5)

    _plot_lines(
        overlap_ax,
        frame,
        ("mean_iou", "median_iou", "mean_annotated_event_coverage"),
        ("Mean IoU", "Median IoU", "Mean coverage"),
        "Temporal agreement",
        "Score",
        (0, 1.02),
    )

    _plot_lines(
        boundary_ax,
        frame,
        (
            "median_absolute_onset_error_seconds",
            "median_absolute_offset_error_seconds",
            "median_absolute_duration_error_seconds",
        ),
        ("Onset", "Offset", "Duration"),
        "Median absolute boundary error",
        "Error (s; lower is better)",
    )

    cmap = LinearSegmentedColormap.from_list("event_audit_blue", ("#F2F5F7", "#7EA9C8", "#174A6E"))
    for ax, (_, row) in zip(audit_axes, frame.iterrows(), strict=True):
        values = np.array(
            [
                [row["matched_events"], row["missed_events"]],
                [row["matched_events"], row["false_positive_events"]],
            ],
            dtype=float,
        )
        fractions = values / np.maximum(values.sum(axis=1, keepdims=True), 1)
        ax.imshow(fractions, cmap=cmap, vmin=0, vmax=1, aspect="equal")
        for i in range(2):
            for j in range(2):
                value = fractions[i, j]
                ax.text(
                    j,
                    i,
                    f"{int(values[i, j]):,}\n{value * 100:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color="white" if value > 0.55 else "#2B2B2B",
                    fontweight="bold" if j == 0 else "normal",
                )
        ax.set_xticks((0, 1), ("Detected/matched", "Missed/unmatched"))
        ax.set_yticks((0, 1), ("Manual events", "Predicted candidates"))
        ax.set_title(
            f"{row['model_label']}  |  corrected\nFP {row['false_positive_events_per_hour']:.2f}/channel-hour"
        )
        ax.tick_params(length=0)
        for label in ax.get_xticklabels():
            label.set_rotation(12)
            label.set_ha("right")
        for spine in ax.spines.values():
            spine.set_visible(False)

    for label, ax in zip("abcdef", axes.ravel(), strict=True):
        ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

    fig.text(
        0.01,
        0.006,
        "Held-out corrected interval-aware decoder; n=953 formally observable manual channel-events. "
        "Bottom rows have different denominators (manual events versus predicted candidates), so they are event-audit matrices, not TN-based confusion matrices.",
        fontsize=6.1,
        color="#55585C",
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.93, bottom=0.12, wspace=0.43, hspace=0.58)
    stem = OUT_ROOT / "figure_2_corrected_event_detection_audit"
    _save(fig, stem)
    return {
        "semantic_level": "held-out continuous event detection",
        "decoder_dependency": "corrected interval-aware full-window union",
        "formal_manual_event_denominator": int(frame["formal_event_denominator"].iloc[0]),
        "visual_atlas_denominator": int(frame["visual_event_denominator"].iloc[0]),
        "outputs": [str(stem.with_suffix(suffix)) for suffix in (".png", ".pdf", ".svg", ".tiff")],
    }


def _write_readme() -> None:
    text = """# 任务一模型性能与 corrected 事件解码

这是任务一 corrected 输出的唯一模型性能入口。三类证据必须分开解释：

1. `figure_1_window_classification_confusion.*`：与用户示例同一结构。上排是 held-out PR、ROC 和窗口指标；下排是三个模型的二分类混淆矩阵。PR/ROC 使用 corrected 输出的 `split_safe_probability`，固定窗口指标和混淆矩阵使用 corrected interval-aware decoder 的 `decoded_mask`。
2. `figure_0_baseline_window_classification_confusion.*`：修正前 `events_baseline` 的历史参考图，保留用于逐格核对，不作为 corrected 主结果。
3. `figure_2_corrected_event_detection_audit.*`：corrected interval-aware decoder 的事件级结果。上排是事件敏感度/精确率/F1、IoU/覆盖和边界误差；下排分别展示人工事件的检测/漏检，以及预测候选的匹配/未匹配。两行分母不同，所以它不是带 TN 的普通混淆矩阵。
4. `figure_s1_training_diagnostics_frozen_models.*`：同一批已训练权重的训练/验证曲线。解码修正是后处理，不会重新生成训练曲线。

为什么没有“corrected 事件级 2x2 TN 混淆矩阵”：连续事件检测中 TP、FP 和 FN 可由区间匹配定义，但 TN 取决于人为选择的时间离散单位。若改用 4 秒窗口定义 TN，就退化为 `figure_1` 的窗口级混淆矩阵；不能把两种分母混在一起。

所有精确数值和来源哈希位于 `source_data/` 与 `performance_manifest.json`。完整逐事件波形证据仍在相邻目录 `../clinical_event_atlas/`。
"""
    (OUT_ROOT / "README_zh.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    baseline_window = _copy_window_figure_and_sources()
    window = _render_corrected_window_figure()
    training = _copy_training_diagnostics()
    corrected_frame = _load_corrected_metrics()
    event = _build_corrected_event_figure(corrected_frame)
    _write_readme()
    manifest = {
        "schema_version": 1,
        "purpose": "canonical Task-1 performance figures beside corrected event decoding",
        "model_run_root": str(RUN_ROOT),
        "corrected_decoder_root": str(CORRECTED_ROOT),
        "window_classification": window,
        "baseline_window_reference": baseline_window,
        "corrected_event_detection": event,
        "training_diagnostics": training,
        "critical_semantics": {
            "window_confusion_matrix": "valid TN/FP/FN/TP using corrected interval-aware decoded_mask on supervised 4-s windows",
            "window_pr_roc": "corrected split_safe_probability; decoder-independent ranking",
            "event_detection": "TP/FP/FN interval matching; TN intentionally undefined",
            "model_retrained_by_this_renderer": False,
        },
    }
    (OUT_ROOT / "performance_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
