#!/usr/bin/env python
"""Build compact QC and deep-learning figures for the pooled epilepsy run.

This renderer is deliberately read-only with respect to training artifacts.  It
uses the frozen test probabilities and protocol audit already produced by the
formal run; no model is retrained and no existing figure is overwritten.
"""

from __future__ import annotations

import json
import sys
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
        "legend.fontsize": 6.4,
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
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

RUN_ROOT = PROJECT_ROOT / "output" / "epilepsy_program_20260821" / "01_annotation_detection" / "04_model_runs" / "combined_old_plus_new"
PROTOCOL_ROOT = PROJECT_ROOT / "output" / "epilepsy_program_20260821" / "01_annotation_detection" / "03_combined_old_plus_new" / "protocol"
FIG_ROOT = RUN_ROOT / "figures"

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
SPLIT_COLORS = {"train": "#547C9C", "val": "#9CA7AE", "test": "#C98249"}
CLASS_COLORS = ("#6D7785", "#B5523B")


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
    plt.close(fig)


def _read_predictions() -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for model in MODELS:
        path = RUN_ROOT / model / "predictions.parquet"
        frame = pd.read_parquet(path)
        frame = frame.loc[frame["split"].astype(str).eq("test")].copy()
        frame["label"] = pd.to_numeric(frame["label"], errors="coerce").astype(int)
        frame["seizure_probability"] = pd.to_numeric(frame["seizure_probability"], errors="coerce").clip(0, 1)
        result[model] = frame
    return result


def build_deep_learning_metrics(out: Path) -> dict[str, str]:
    """Build PR/ROC/calibration, metric table and three confusion matrices."""

    predictions = _read_predictions()
    rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(2, 3, figsize=(7.2047, 5.75), constrained_layout=False)
    pr_ax, roc_ax, metric_ax, *cm_axes = axes.ravel()
    for model in MODELS:
        frame = predictions[model]
        y = frame["label"].to_numpy(np.int8)
        score = frame["seizure_probability"].to_numpy(float)
        pred = (score >= 0.5).astype(np.int8)
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
        # Cast before sqrt: the product can exceed NumPy's platform integer
        # range for the pooled windows and otherwise becomes an object scalar.
        mcc_den = float(np.sqrt(float(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))))
        mcc = ((tp * tn) - (fp * fn)) / mcc_den
        auprc = float(average_precision_score(y, score))
        auroc = float(roc_auc_score(y, score))
        rows.append(
            {
                "model": model,
                "model_label": LABELS[model],
                "threshold": 0.5,
                "n_windows": n,
                "positive_windows": int((y == 1).sum()),
                "negative_windows": int((y == 0).sum()),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
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
        pr_ax.plot(
            recall_curve,
            precision_curve,
            color=COLORS[model],
            linewidth=1.35,
            label=f"{LABELS[model]}  {auprc:.3f}",
        )
        fpr, tpr, _ = roc_curve(y, score)
        roc_ax.plot(fpr, tpr, color=COLORS[model], linewidth=1.35, label=f"{LABELS[model]}  {auroc:.3f}")
        observed, predicted = calibration_curve(y, score, n_bins=10, strategy="quantile")
        for i, (pp, oo) in enumerate(zip(predicted, observed, strict=True)):
            calibration_rows.append({"model": model, "bin": i, "mean_predicted": float(pp), "observed_fraction": float(oo)})
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
    metric_ax.set_title("Window metrics at p ≥ 0.5")
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
        ax.set_title(LABELS[model] + "  |  p ≥ 0.5")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.text(0.01, 0.006, "Test windows are held out by clinical event/background region; rows are normalized within true class in confusion matrices.", fontsize=6.2, color="#55585C")
    fig.subplots_adjust(left=0.075, right=0.995, top=0.93, bottom=0.10, wspace=0.40, hspace=0.52)
    stem = out / "figure_3_deep_learning_metrics"
    _save(fig, stem)
    metrics_frame.reset_index().to_csv(out / "window_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(out / "confusion_matrices.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(out / "calibration_bins.csv", index=False)
    return {"figure": str(stem.with_suffix(".png")), "metrics": str(out / "window_metrics.csv")}


def build_per_animal_metrics(out: Path) -> str:
    """Render mouse-level AUPRC/AUROC/support without hiding null classes."""

    rows: list[dict[str, object]] = []
    for model in MODELS:
        metrics = json.loads((RUN_ROOT / model / "metrics.json").read_text(encoding="utf-8"))
        for animal, values in metrics["test"]["per_animal"].items():
            rows.append({"model": model, "animal_id": str(animal), **values})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "per_animal_window_metrics.csv", index=False)
    animals = sorted(frame["animal_id"].astype(str).unique())
    fig, axes = plt.subplots(1, 3, figsize=(7.2047, 3.25), constrained_layout=False)
    for ax, metric, title, cmap, vmin, vmax in zip(
        axes,
        ("auprc", "auroc", "positive_fraction"),
        ("Mouse-specific AUPRC", "Mouse-specific AUROC", "Test positive fraction"),
        ("YlOrRd", "YlGnBu", "Greys"),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.5),
        strict=True,
    ):
        matrix = frame.pivot(index="model", columns="animal_id", values=metric).reindex(index=MODELS, columns=animals).to_numpy(float)
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(animals)), animals, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(MODELS)), [LABELS[m] for m in MODELS])
        ax.set_title(title, loc="left", fontweight="bold")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                text = "NA" if not np.isfinite(value) else f"{value:.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=5.9, color="#2B2B2B" if (not np.isfinite(value) or value > (vmax * 0.58)) else "white")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
        cbar.ax.tick_params(labelsize=5.5, length=2)
        cbar.set_label(metric, fontsize=5.8)
    fig.text(0.01, 0.006, "Values are pooled within each mouse on the held-out windows; NA denotes a mouse/channel aggregate with no positive test windows.", fontsize=6.1, color="#55585C")
    fig.subplots_adjust(left=0.12, right=0.985, top=0.86, bottom=0.23, wspace=0.54)
    stem = out / "figure_s3_per_animal_window_metrics"
    _save(fig, stem)
    return str(stem.with_suffix(".png"))


def build_split_qc(out: Path) -> dict[str, str]:
    """Build a compact frozen-protocol split and channel contribution figure."""

    audit = json.loads((PROTOCOL_ROOT / "protocol_audit.json").read_text(encoding="utf-8"))
    centers = pd.read_parquet(PROTOCOL_ROOT / "supervised_centers.parquet", columns=["animal_id", "channel_name", "split", "label"])
    centers["label_name"] = centers["label"].map({0: "non-seizure", 1: "seizure"})
    split_counts = centers.groupby(["split", "label_name"], sort=False).size().unstack(fill_value=0).reindex(index=("train", "val", "test"), columns=("non-seizure", "seizure"), fill_value=0)
    split_counts.to_csv(out / "split_window_counts.csv")
    channel_counts = centers.groupby(["animal_id", "channel_name", "split", "label_name"], sort=False).size().unstack(fill_value=0).reset_index()
    for col in ("non-seizure", "seizure"):
        if col not in channel_counts.columns:
            channel_counts[col] = 0
    channel_counts["stream"] = channel_counts["animal_id"].astype(str) + "/" + channel_counts["channel_name"].astype(str)
    channel_counts.to_csv(out / "channel_split_window_counts.csv", index=False)
    events = pd.read_parquet(PROTOCOL_ROOT / "channel_events.parquet", columns=["split", "animal_id", "channel_name"])
    event_counts = events.groupby("split").size().reindex(("train", "val", "test"), fill_value=0).rename("channel_events")
    independent = audit.get("event_split_summary", {})

    fig, axes = plt.subplots(2, 2, figsize=(7.2047, 5.35), constrained_layout=False)
    ax = axes[0, 0]
    x = np.arange(3)
    for label_name in ("non-seizure", "seizure"):
        vals = split_counts[label_name].to_numpy(float)
        bottom = np.zeros(3) if label_name == "non-seizure" else split_counts["non-seizure"].to_numpy(float)
        ax.bar(x, vals, bottom=bottom, color=CLASS_COLORS[0] if label_name == "non-seizure" else CLASS_COLORS[1], width=0.62, label=label_name)
    ax.set_xticks(x, ("Train", "Validation", "Test"))
    ax.set_ylabel("Supervised centers")
    ax.set_title("Frozen split composition", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    _clean_axis(ax, "y")

    ax = axes[0, 1]
    channel_event_values = [int(event_counts.get(s, 0)) for s in ("train", "val", "test")]
    independent_values = [int(independent.get(s, {}).get("independent_events", 0)) for s in ("train", "val", "test")]
    width = 0.35
    ax.bar(x - width / 2, independent_values, width, color="#5E7E95", label="Independent events")
    ax.bar(x + width / 2, channel_event_values, width, color="#C98249", label="Channel events")
    ax.set_xticks(x, ("Train", "Validation", "Test"))
    ax.set_ylabel("Event count")
    ax.set_title("Event-level allocation", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    _clean_axis(ax, "y")

    ax = axes[1, 0]
    channel_pivot = channel_counts.pivot_table(index="stream", columns="split", values="seizure", aggfunc="sum", fill_value=0).reindex(columns=("train", "val", "test"), fill_value=0)
    image = ax.imshow(np.log10(channel_pivot.to_numpy(float) + 1), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(3), ("Train", "Validation", "Test"))
    ax.set_yticks(np.arange(len(channel_pivot)), channel_pivot.index, fontsize=5.4)
    ax.set_title("Positive-window support by channel", loc="left", fontweight="bold")
    ax.set_xlabel("Split")
    ax.set_ylabel("Independent channel stream")
    for i in range(channel_pivot.shape[0]):
        for j in range(channel_pivot.shape[1]):
            ax.text(j, i, f"{int(channel_pivot.iloc[i, j]):,}", ha="center", va="center", fontsize=5.0, color="#2B2B2B" if channel_pivot.iloc[i, j] < channel_pivot.to_numpy().max() * 0.55 else "white")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("log10(count + 1)", fontsize=6.0)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax = axes[1, 1]
    ax.axis("off")
    cfg = audit.get("config", {})
    lines = [
        "Protocol audit",
        f"Animals: {len(cfg.get('animals', []))}   valid streams: {len(channel_pivot)}",
        f"Events: {audit.get('channel_events', 0):,} channel / {audit.get('unique_clinical_events', 0):,} independent",
        f"Full index: {audit.get('full_rows', 0):,} rows",
        f"Supervised centers: {audit.get('supervised_centers', audit.get('eligible_centers', 0)):,.0f}",
        f"Raw-window purge retained overlap: {audit.get('cross_split_raw_overlap_pairs_retained_centers', 'NA')}",
        "Context: 21 epochs × 4 s; stride 1 s",
        "Filter: 0.5–70 Hz zero-phase + 50 Hz notch",
        "Excluded: E07/CH7, E77/CH7",
        "Amplitude: training-baseline normalized a.u.",
    ]
    ax.text(0.02, 0.96, "\n".join(lines), va="top", ha="left", fontsize=7.0, linespacing=1.55, family="DejaVu Sans Mono")
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(0.01, 0.006, "Counts are protocol-audit denominators, not model test performance denominators. Splits are assigned to independent events/background regions before window purge.", fontsize=6.2, color="#55585C")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.10, wspace=0.34, hspace=0.44)
    stem = out / "figure_s2_data_split_qc"
    _save(fig, stem)
    (out / "protocol_audit_snapshot.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"figure": str(stem.with_suffix(".png")), "split_counts": str(out / "split_window_counts.csv")}


def build_prediction_index(out: Path) -> dict[str, str]:
    """Index all continuous prediction tracks and expose the existing overview atlas."""

    full = pd.read_parquet(PROTOCOL_ROOT / "full_index.parquet", columns=["animal_id", "session_id", "channel_name", "split", "channel_valid"])
    track_keys = ["animal_id", "session_id", "channel_name"]
    tracks = (
        full.groupby(track_keys, sort=False)
        .agg(
            channel_valid=("channel_valid", "max"),
            manual_rows=("channel_valid", "size"),
            splits_present=("split", lambda values: ";".join(sorted(set(values.astype(str))))),
        )
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for model in MODELS:
        pred_path = RUN_ROOT / model / "continuous" / "continuous_predictions.parquet"
        pred = pd.read_parquet(pred_path, columns=["animal_id", "session_id", "channel_name", "split", "start_time_ms", "end_time_ms"])
        summary = pred.groupby(["animal_id", "session_id", "channel_name"], sort=False).agg(prediction_rows=("start_time_ms", "size"), prediction_start_ms=("start_time_ms", "min"), prediction_end_ms=("end_time_ms", "max"), test_rows=("split", lambda x: int((x.astype(str) == "test").sum()))).reset_index()
        merged = tracks.merge(summary, on=["animal_id", "session_id", "channel_name"], how="left")
        merged["model"] = model
        merged["prediction_status"] = np.where(merged["prediction_rows"].notna(), "continuous_prediction_available", "no_continuous_rows")
        rows.extend(merged.to_dict(orient="records"))
    index = pd.DataFrame(rows)
    index.to_csv(out / "all_channel_prediction_index.csv", index=False)
    overview_dir = RUN_ROOT / "figures" / "legacy_style_deep_learning" / "whole_recording_overviews"
    overview_png = len(list(overview_dir.glob("*.png"))) if overview_dir.exists() else 0
    payload = {
        "protocol_tracks": int(len(tracks)),
        "models": list(MODELS),
        "continuous_prediction_tracks_by_model": {model: int(index.loc[index.model.eq(model) & index.prediction_rows.notna(), ["animal_id", "session_id", "channel_name"]].drop_duplicates().shape[0]) for model in MODELS},
        "existing_physio_no_aux_overview_png": overview_png,
        "existing_overview_directory": str(overview_dir),
        "clinical_event_atlas_directory": str(FIG_ROOT / "clinical_event_atlas"),
        "note": "A missing continuous row means no eligible prediction trace was written for that protocol track; it is not silently counted as a negative event.",
    }
    (out / "all_channel_prediction_index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"index": str(out / "all_channel_prediction_index.csv"), "manifest": str(out / "all_channel_prediction_index.json")}


def write_readmes(metric_dir: Path, qc_dir: Path, index_dir: Path) -> None:
    metric_dir.joinpath("README_zh.md").write_text(
        "# 合并数据深度学习指标图\n\n"
        "`figure_3_deep_learning_metrics.*` 汇总当前 7 只小鼠合并协议的三个正式模型：PR、ROC、固定 p≥0.5 的窗口指标、以及逐模型混淆矩阵。混淆矩阵中的百分比按真实类别行归一化；测试窗口共用同一冻结划分。\n\n"
        "`figure_s3_per_animal_window_metrics.*` 展示逐鼠 AUPRC、AUROC 和阳性支持；没有阳性测试窗口的格子保留为 NA。\n\n"
        "数值明细见 `window_metrics.csv`、`per_animal_window_metrics.csv`、`confusion_matrices.csv` 和 `calibration_bins.csv`。\n",
        encoding="utf-8",
    )
    qc_dir.joinpath("README_zh.md").write_text(
        "# 合并协议划分质控图\n\n"
        "`figure_s2_data_split_qc.*` 展示训练/验证/测试的监督窗口类别组成、独立事件与通道事件分配、各独立通道流的阳性窗口支持，以及原始窗口 purge 和预处理摘要。它是协议审计图，不是模型性能图。\n\n"
        "原始明细见 `split_window_counts.csv`、`channel_split_window_counts.csv` 和 `protocol_audit_snapshot.json`。\n",
        encoding="utf-8",
    )
    index_dir.joinpath("README_zh.md").write_text(
        "# 全通道连续打标索引\n\n"
        "`all_channel_prediction_index.csv` 按模型、动物、记录和通道列出连续预测覆盖情况。现有 `legacy_style_deep_learning/whole_recording_overviews/` 是 Physio-Sequence 的整段记录图；`clinical_event_atlas/` 则是三个模型的测试事件覆盖、漏检、边界偏移和标签外候选图。\n\n"
        "没有连续预测行的协议轨迹被标记为 `no_continuous_rows`，不会被当成 non-seizure。\n",
        encoding="utf-8",
    )


def main() -> None:
    metric_dir = FIG_ROOT / "deep_learning_metrics"
    qc_dir = FIG_ROOT / "protocol_qc"
    index_dir = FIG_ROOT / "continuous_prediction_index"
    for path in (metric_dir, qc_dir, index_dir):
        path.mkdir(parents=True, exist_ok=True)
    result = {
        "deep_learning_metrics": build_deep_learning_metrics(metric_dir),
        "per_animal_window_metrics": build_per_animal_metrics(metric_dir),
        "protocol_qc": build_split_qc(qc_dir),
        "continuous_prediction_index": build_prediction_index(index_dir),
    }
    write_readmes(metric_dir, qc_dir, index_dir)
    manifest = {
        "run_root": str(RUN_ROOT),
        "protocol_root": str(PROTOCOL_ROOT),
        "models": list(MODELS),
        "outputs": result,
        "training_started": False,
    }
    (FIG_ROOT / "COMBINED_FIGURE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
