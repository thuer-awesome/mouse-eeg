#!/usr/bin/env python
"""Render complete all-class diagnostics from finished PG classifier outputs.

The renderer is deliberately separate from training and evaluation.  It reads
``all_window_test_predictions.parquet`` produced by the full-window evaluator
and writes publication figures plus source tables.  A capped event-balanced
table is accepted only when ``--allow-capped-fallback`` is explicitly given and
is labelled sensitivity-only in the manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


MODEL_ORDER = ("eegwavenet_sc", "physio_tf", "physio_stats", "eegnetv4", "eegconformer")
DISPLAY = {
    "eegwavenet_sc": "EEGWaveNet-SC",
    "physio_tf": "Physio-TF",
    "physio_stats": "Physio + statistics",
    "eegnetv4": "EEGNetv4",
    "eegconformer": "EEGConformer",
}
COLOURS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#6A3D9A", "#333333")
MID_GREY = "#66717A"
INK = "#252A2E"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def _natural(value: object) -> int:
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else 10**9


def _save(fig: plt.Figure, stem: Path, *, include_tiff: bool = False) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    if include_tiff:
        fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _files(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    return [
        {"path": str(path.relative_to(root)).replace("\\", "/"), "bytes": int(path.stat().st_size)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".png", ".pdf", ".svg", ".tiff", ".eps"}
    ]


def _probability_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns = [column for column in frame.columns if str(column).lower().startswith("prob_")]
    if prefix:
        preferred = [column for column in columns if str(column).split("_", 1)[1].upper().startswith(prefix.upper())]
        if preferred:
            columns = preferred
    ordered = sorted(columns, key=lambda column: _natural(str(column).split("_", 1)[-1]))
    if len(set(ordered)) != len(ordered):
        raise ValueError("prediction table contains duplicate probability column names")
    return ordered


def _category_column(frame: pd.DataFrame) -> str:
    for name in ("global_category", "pooled_global_category", "y_true", "true"):
        if name in frame:
            return name
    raise ValueError("label table has no global_category/pooled_global_category column")


def _load_labels(path: Path) -> tuple[pd.DataFrame, int]:
    labels = pd.read_parquet(path)
    category = _category_column(labels)
    if category != "global_category":
        labels = labels.rename(columns={category: "global_category"})
    if "split" not in labels:
        raise ValueError(f"label table has no split column: {path}")
    labels["global_category"] = pd.to_numeric(labels["global_category"], errors="raise").astype(int)
    classes = int(labels["global_category"].max()) + 1
    observed = sorted(labels["global_category"].unique().tolist())
    if observed != list(range(classes)):
        raise ValueError(f"global_category values must be contiguous from zero; observed={observed}")
    labels = labels.loc[labels["split"].astype(str).str.lower().eq("test")].copy()
    if labels.empty:
        raise ValueError(f"test split is empty: {path}")
    return labels.reset_index(drop=True), classes


def _model_names(class_dir: Path) -> list[str]:
    comparison = class_dir / "all_window_model_comparison.csv"
    names: set[str] = set()
    comparison_names: set[str] = set()
    if comparison.is_file():
        frame = pd.read_csv(comparison)
        if "model" in frame:
            comparison_names.update(str(value) for value in frame["model"].dropna())
    if comparison_names:
        names = comparison_names
    else:
        names.update(path.name for path in class_dir.iterdir() if path.is_dir() and (path / "best_model.pt").is_file())
    ordered = [name for name in MODEL_ORDER if name in names]
    ordered.extend(sorted(names - set(ordered)))
    if not ordered:
        raise FileNotFoundError(f"no model checkpoints found below {class_dir}")
    return ordered


def _capped_rows(labels: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    keys = [key for key in ("animal_id", "session_id", "event_id", "channel_name") if key in labels]
    if not keys:
        return labels.copy()
    chunks: list[pd.DataFrame] = []
    for _, group in labels.groupby(keys, sort=False):
        if len(group) <= limit:
            chunks.append(group)
        else:
            indices = np.linspace(0, len(group) - 1, limit).round().astype(int)
            chunks.append(group.iloc[np.unique(indices)])
    return pd.concat(chunks, ignore_index=True) if chunks else labels.iloc[0:0].copy()


def _load_prediction(model_dir: Path, labels: pd.DataFrame, prefix: str, allow_capped: bool, classes: int) -> tuple[pd.DataFrame, str, list[str]]:
    full_path = model_dir / "all_window_test_predictions.parquet"
    if full_path.is_file():
        pred = pd.read_parquet(full_path).reset_index(drop=True)
        mode = "full_window"
    else:
        if not allow_capped:
            raise FileNotFoundError(f"{full_path} is missing; rerun full-window evaluation or pass --allow-capped-fallback")
        legacy = model_dir / "test_predictions.parquet"
        if not legacy.is_file():
            raise FileNotFoundError(f"missing prediction table for {model_dir.name}: {legacy}")
        pred = pd.read_parquet(legacy).reset_index(drop=True)
        capped = _capped_rows(labels)
        if len(pred) != len(capped):
            raise ValueError(f"capped prediction length mismatch for {model_dir.name}: {len(pred)} != {len(capped)}")
        pred = pd.concat([capped.reset_index(drop=True), pred], axis=1)
        mode = "capped_sensitivity"
    if "y_true" not in pred:
        pred["y_true"] = labels["global_category"].to_numpy()[: len(pred)]
    if "y_pred" not in pred:
        if "pred" in pred:
            pred["y_pred"] = pred["pred"]
        else:
            raise ValueError(f"prediction table has no y_pred/pred: {model_dir}")
    pred["y_true"] = pd.to_numeric(pred["y_true"], errors="raise").astype(int)
    pred["y_pred"] = pd.to_numeric(pred["y_pred"], errors="raise").astype(int)
    if pred["y_true"].lt(0).any() or pred["y_true"].ge(classes).any():
        raise ValueError(f"prediction labels are outside the frozen category range: {model_dir}")
    columns = _probability_columns(pred, prefix)
    if not columns:
        raise ValueError(f"prediction table has no probability columns: {model_dir}")
    if not np.isfinite(pred[columns].to_numpy(float)).all():
        raise ValueError(f"prediction table contains NaN/Inf probabilities: {model_dir}")
    return pred, mode, columns


def _metrics(pred: pd.DataFrame, columns: list[str], classes: int, prefix: str) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    y = pred["y_true"].to_numpy(int); yp = pred["y_pred"].to_numpy(int); prob = pred[columns].to_numpy(float)
    if prob.shape[1] != classes:
        raise ValueError(f"probability columns={prob.shape[1]} but labels have classes={classes}")
    if len(y) != len(yp) or len(y) != len(prob):
        raise ValueError("prediction columns have unequal lengths")
    cm = confusion_matrix(y, yp, labels=np.arange(classes))
    support = cm.sum(axis=1)
    per_rows: list[dict[str, Any]] = []
    aps: list[float] = []; aucs: list[float] = []
    for category in range(classes):
        positive = (y == category).astype(int)
        precision = recall = ap = auroc = float("nan")
        if positive.sum() > 0:
            tp = int(((yp == category) & (y == category)).sum())
            recall = tp / int(positive.sum())
            predicted_positive = int((yp == category).sum())
            precision = tp / predicted_positive if predicted_positive else 0.0
            if positive.sum() < len(y):
                ap = float(average_precision_score(positive, prob[:, category])); auroc = float(roc_auc_score(positive, prob[:, category])); aps.append(ap); aucs.append(auroc)
        f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall else 0.0
        per_rows.append({"category": category, "label": f"{prefix}{category + 1:02d}", "support": int(support[category]), "precision": precision, "recall": recall, "f1": f1, "average_precision": ap, "auroc": auroc})
    recalls = np.divide(np.diag(cm), support, out=np.zeros(classes, dtype=float), where=support > 0)
    metrics = {"n": int(len(y)), "accuracy": float((y == yp).mean()), "balanced_accuracy": float(recalls[support > 0].mean()) if np.any(support > 0) else float("nan"), "macro_f1": float(f1_score(y, yp, labels=np.arange(classes), average="macro", zero_division=0)), "weighted_f1": float(f1_score(y, yp, labels=np.arange(classes), average="weighted", zero_division=0)), "macro_auprc": float(np.mean(aps)) if aps else float("nan"), "macro_auroc": float(np.mean(aucs)) if aucs else float("nan")}
    return metrics, pd.DataFrame(per_rows), cm


def _plot_pr_roc(pred: pd.DataFrame, columns: list[str], model: str, classes: int, mode: str, root: Path, prefix: str, *, include_tiff: bool) -> None:
    y = pred["y_true"].to_numpy(int); probability = pred[columns].to_numpy(float); cols = 6 if classes >= 6 else max(1, classes); rows = int(math.ceil(classes / cols));
    for kind in ("pr", "roc"):
        fig, axes = plt.subplots(rows, cols, figsize=(183 / 25.4, max(55, rows * 30) / 25.4), squeeze=False, sharex=True, sharey=True, constrained_layout=True); flat = axes.ravel()
        for category in range(classes):
            ax = flat[category]; positive = (y == category).astype(int); support = int(positive.sum()); label = f"{prefix}{category + 1:02d}"
            if support == 0 or support == len(y):
                ax.text(0.5, 0.5, f"{label}\nnot evaluable", ha="center", va="center", transform=ax.transAxes, fontsize=5.5, color=MID_GREY)
            elif kind == "pr":
                precision, recall, _ = precision_recall_curve(positive, probability[:, category]); ap = average_precision_score(positive, probability[:, category]); ax.plot(recall, precision, color=COLOURS[category % len(COLOURS)], lw=0.85); ax.axhline(positive.mean(), color="#B8BEC4", lw=0.45, ls="--"); ax.set_title(f"{label}  n={support}\nAP={ap:.3f}", fontsize=5.2, loc="left")
            else:
                fpr, tpr, _ = roc_curve(positive, probability[:, category]); auroc = roc_auc_score(positive, probability[:, category]); ax.plot(fpr, tpr, color=COLOURS[category % len(COLOURS)], lw=0.85); ax.plot([0, 1], [0, 1], color="#B8BEC4", lw=0.45, ls="--"); ax.set_title(f"{label}  n={support}\nAUROC={auroc:.3f}", fontsize=5.2, loc="left")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.tick_params(labelsize=4.8, length=2); ax.grid(color="#EEF0F1", lw=0.3)
            if category % cols == 0: ax.set_ylabel("Precision" if kind == "pr" else "TPR", fontsize=5.5)
            if category >= classes - cols: ax.set_xlabel("Recall" if kind == "pr" else "FPR", fontsize=5.5)
        for ax in flat[classes:]: ax.axis("off")
        fig.suptitle(f"{DISPLAY.get(model, model)} | all {classes} {prefix} categories | one-vs-rest {kind.upper()} | {mode}", x=0.01, ha="left", fontsize=8.5, fontweight="bold")
        fig.text(0.01, 0.006, "Every category is retained as its own panel. A pale dashed line is the per-category prevalence baseline, not a typicality filter.", fontsize=5.5, color=MID_GREY)
        _save(fig, root / model / f"figure_{kind}_all_classes", include_tiff=include_tiff)


def _plot_support_recall(per_class: pd.DataFrame, model: str, mode: str, root: Path, *, include_tiff: bool) -> None:
    ordered = per_class.sort_values(["support", "category"], ascending=[False, True]).reset_index(drop=True); y = np.arange(len(ordered)); fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, max(70, 30 + 5 * len(ordered)) / 25.4), gridspec_kw={"width_ratios": [1.15, 1.0]}, constrained_layout=True)
    axes[0].barh(y, ordered.support, color="#9BB7C8", edgecolor="none"); axes[0].set_yticks(y, ordered.label, fontsize=5.5); axes[0].invert_yaxis(); axes[0].set_xlabel("Test-window support"); axes[0].set_title("Class support", loc="left", fontweight="bold")
    axes[1].barh(y, ordered.recall.fillna(0), color="#0072B2", alpha=0.85, edgecolor="none"); axes[1].set_yticks(y, ordered.label, fontsize=5.5); axes[1].invert_yaxis(); axes[1].set_xlim(0, 1); axes[1].set_xlabel("Per-class recall"); axes[1].set_title("Recall", loc="left", fontweight="bold"); axes[1].axvline(1 / max(1, len(ordered)), color="#999999", ls="--", lw=0.7, label=f"1 / {len(ordered)}"); axes[1].legend(frameon=False, fontsize=5.5, loc="lower right")
    fig.suptitle(f"{DISPLAY.get(model, model)} | support and recall for all {len(ordered)} categories | {mode}", fontsize=8.5, fontweight="bold"); _save(fig, root / model / "figure_class_support_recall", include_tiff=include_tiff)


def _plot_confusion(cm: np.ndarray, labels: list[str], model: str, mode: str, root: Path, *, include_tiff: bool) -> None:
    normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0); classes = len(labels); ticks = np.arange(0, classes, max(1, int(math.ceil(classes / 12)))); fig, ax = plt.subplots(figsize=(110 / 25.4, 105 / 25.4), constrained_layout=True); image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues", interpolation="nearest", aspect="equal"); ax.set_xticks(ticks, [labels[index] for index in ticks], rotation=45, ha="right"); ax.set_yticks(ticks, [labels[index] for index in ticks]); ax.set_xlabel("Predicted category"); ax.set_ylabel("True category"); ax.set_title(f"{DISPLAY.get(model, model)} | {classes}×{classes} confusion | {mode}", loc="left", fontweight="bold"); fig.colorbar(image, ax=ax, fraction=0.04, pad=0.025, label="Row fraction"); _save(fig, root / model / "figure_confusion_matrix_all_windows", include_tiff=include_tiff)


def _plot_streams(rows: list[dict[str, Any]], models: list[str], root: Path, *, include_tiff: bool) -> None:
    if not rows: return
    data = pd.DataFrame(rows); pivot = data.pivot_table(index=["animal_id", "channel_name"], columns="model", values="macro_f1").reindex(columns=models); values = pivot.to_numpy(float); fig, ax = plt.subplots(figsize=(183 / 25.4, max(70, 30 + 9 * len(pivot)) / 25.4), constrained_layout=True); image = ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto"); ax.set_xticks(np.arange(len(models)), [DISPLAY.get(model, model) for model in models], rotation=25, ha="right", fontsize=6); ax.set_yticks(np.arange(len(pivot)), [f"{animal}/{channel}" for animal, channel in pivot.index], fontsize=6); ax.set_xlabel("Model"); ax.set_ylabel("Independent channel stream"); ax.set_title("Per-stream Macro-F1 on full test windows", loc="left", fontweight="bold"); fig.colorbar(image, ax=ax, fraction=.04, pad=.03, label="Macro-F1"); _save(fig, root / "figure_5_per_animal_channel_metrics", include_tiff=include_tiff)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-dir", type=Path, required=True)
    parser.add_argument("--label-path", type=Path, required=True)
    parser.add_argument("--figure-root", type=Path, required=True)
    parser.add_argument("--label-prefix", default="PG")
    parser.add_argument("--allow-capped-fallback", action="store_true", help="allow legacy event-balanced test_predictions.parquet; marked sensitivity-only")
    parser.add_argument("--require-full-window", action="store_true", help="fail if any model lacks all_window_test_predictions.parquet")
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--include-tiff", action="store_true")
    args = parser.parse_args(argv)
    class_dir, label_path, figure_root = args.class_dir.resolve(), args.label_path.resolve(), args.figure_root.resolve()
    if not class_dir.is_dir(): raise FileNotFoundError(class_dir)
    if not label_path.is_file(): raise FileNotFoundError(label_path)
    if figure_root.exists() and any(figure_root.iterdir()) and not args.allow_existing: raise FileExistsError(f"figure root is non-empty: {figure_root}; use --allow-existing")
    figure_root.mkdir(parents=True, exist_ok=True)
    labels, classes = _load_labels(label_path); models = _model_names(class_dir); all_metrics: list[dict[str, Any]] = []; all_class: list[pd.DataFrame] = []; stream_rows: list[dict[str, Any]] = []; modes: dict[str, str] = {}; sources: dict[str, str] = {}
    for model in models:
        pred, mode, columns = _load_prediction(class_dir / model, labels, str(args.label_prefix), args.allow_capped_fallback, classes)
        if args.require_full_window and mode != "full_window": raise RuntimeError(f"{model} is {mode}; full-window predictions are required")
        if mode == "full_window" and len(pred) != len(labels): raise ValueError(f"full prediction length mismatch for {model}: {len(pred)} != {len(labels)}")
        key_columns = [column for column in ("animal_id", "session_id", "event_id", "channel_name", "start_time_ms", "end_time_ms") if column in pred.columns and column in labels.columns]
        if mode == "full_window":
            required_keys = ("animal_id", "session_id", "event_id", "channel_name")
            missing_keys = [column for column in required_keys if column not in key_columns]
            if missing_keys:
                raise ValueError(f"full-window prediction table is missing identity keys for {model}: {missing_keys}")
            expected = labels[key_columns + ["global_category"]].copy()
            expected["_key"] = expected[key_columns].astype(str).agg("|".join, axis=1)
            actual = pred[key_columns + ["y_true"]].copy()
            actual["_key"] = actual[key_columns].astype(str).agg("|".join, axis=1)
            if expected["_key"].duplicated().any() or actual["_key"].duplicated().any():
                raise ValueError(f"full-window prediction keys are not unique for {model}")
            joined = expected.set_index("_key")["global_category"].reindex(actual["_key"])
            if joined.isna().any() or not np.array_equal(joined.to_numpy(int), actual["y_true"].to_numpy(int)):
                raise ValueError(f"full-window prediction keys/labels do not match the frozen test table for {model}")
        modes[model] = mode; sources[model] = str((class_dir / model / ("all_window_test_predictions.parquet" if mode == "full_window" else "test_predictions.parquet")))
        metrics, per_class, cm = _metrics(pred, columns, classes, str(args.label_prefix)); metrics.update({"model": model, "prediction_mode": mode}); all_metrics.append(metrics); per_class.insert(0, "model", model); all_class.append(per_class); labels_text = [f"{args.label_prefix}{index + 1:02d}" for index in range(classes)]; figure_root.joinpath("source_data").mkdir(parents=True, exist_ok=True); pd.DataFrame(cm, index=labels_text, columns=labels_text).to_csv(figure_root / "source_data" / f"{model}_confusion_matrix_all_windows.csv", encoding="utf-8-sig"); _plot_pr_roc(pred, columns, model, classes, mode, figure_root, str(args.label_prefix), include_tiff=args.include_tiff); _plot_support_recall(per_class, model, mode, figure_root, include_tiff=args.include_tiff); _plot_confusion(cm, labels_text, model, mode, figure_root, include_tiff=args.include_tiff)
        if {"animal_id", "channel_name"}.issubset(pred.columns):
            for (animal, channel), group in pred.groupby(["animal_id", "channel_name"], sort=True):
                y = group.y_true.to_numpy(int); yp = group.y_pred.to_numpy(int); support = np.unique(y); recalls = [float(((yp == category) & (y == category)).sum() / (y == category).sum()) for category in support if (y == category).sum()]; stream_rows.append({"model": model, "animal_id": str(animal), "channel_name": str(channel), "n": int(len(group)), "n_classes_present": int(len(support)), "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"), "macro_f1": float(f1_score(y, yp, labels=np.arange(classes), average="macro", zero_division=0))})
    metrics_frame = pd.DataFrame(all_metrics); per_class_frame = pd.concat(all_class, ignore_index=True); figure_root.joinpath("source_data").mkdir(parents=True, exist_ok=True); metrics_frame.to_csv(figure_root / "source_data" / "all_window_model_comparison.csv", index=False, encoding="utf-8-sig"); per_class_frame.to_csv(figure_root / "source_data" / "per_class_metrics_all_models.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(stream_rows).to_csv(figure_root / "source_data" / "per_animal_channel_metrics.csv", index=False, encoding="utf-8-sig"); _plot_streams(stream_rows, models, figure_root, include_tiff=args.include_tiff)
    manifest = {"schema_version": 2, "renderer": "scripts/render_pg36_classifier_full.py", "models": models, "classes": classes, "label_prefix": str(args.label_prefix), "formal_denominator": "all eligible held-out test windows", "prediction_mode_by_model": modes, "prediction_source_by_model": sources, "allow_capped_fallback": bool(args.allow_capped_fallback), "retrained_by_renderer": False, "label_path": str(label_path), "class_dir": str(class_dir), "outputs": _files(figure_root), "figure_contract": {"core_claim": "show every frozen category under the same held-out-window denominator", "data_unit": "one complete test window; channel streams remain independent", "visual_encoding": "one-vs-rest panel per category plus row-normalized confusion", "reviewer_risk": "capped event-balanced outputs are sensitivity-only and never primary"}}
    (figure_root / "figure_generation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (figure_root / "FIGURE_INDEX_zh.md").write_text("# PG 全类别分类图索引\n\n" + f"- 类别数：{classes}；模型：{', '.join(models)}。\n" + "- 每个模型的 `figure_pr_all_classes`、`figure_roc_all_classes`：每个类别一个 one-vs-rest panel，所有类别都保留。\n" + "- 每个模型的 `figure_class_support_recall`：完整支持量和召回率。\n" + "- 每个模型的 `figure_confusion_matrix_all_windows`：完整类别混淆矩阵及其 CSV。\n" + "- `source_data/` 保存指标、逐类指标和逐鼠逐通道指标。\n" + "- 正式分母是全部合格测试窗口；若使用 `--allow-capped-fallback`，manifest 会明确写出 sensitivity-only。\n", encoding="utf-8")
    print(json.dumps({"figure_root": str(figure_root), "models": models, "classes": classes, "prediction_mode_by_model": modes}, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
