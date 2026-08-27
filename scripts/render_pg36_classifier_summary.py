#!/usr/bin/env python
"""Render the publication-facing summary for a completed PG classifier run.

This stage only consumes completed training products (history, metrics and
confusion tables). It never retrains a model and never silently substitutes an
event-balanced sensitivity denominator for the complete held-out-window table.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from string import ascii_lowercase
from typing import Any

import matplotlib as mpl

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


MODEL_ORDER = ("eegwavenet_sc", "physio_tf", "physio_stats", "eegnetv4", "eegconformer")
DISPLAY = {
    "eegwavenet_sc": "EEGWaveNet-SC",
    "physio_tf": "Physio-TF",
    "physio_stats": "Physio + statistics",
    "eegnetv4": "EEGNetv4",
    "eegconformer": "EEGConformer",
}
MODEL_COLOURS = {
    "eegwavenet_sc": "#4D4D4D",
    "physio_tf": "#0072B2",
    "physio_stats": "#D55E00",
    "eegnetv4": "#009E73",
    "eegconformer": "#CC79A7",
}
BLUE = "#0072B2"
ORANGE = "#D55E00"
INK = "#252A2E"
MID_GREY = "#66717A"
LIGHT_GREY = "#D9DEE3"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "axes.linewidth": 0.65,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def _save(fig: plt.Figure, stem: Path, *, include_tiff: bool = False) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for suffix, kwargs in (
        (".svg", {"bbox_inches": "tight", "facecolor": "white"}),
        (".pdf", {"bbox_inches": "tight", "facecolor": "white"}),
        (".png", {"dpi": 600, "bbox_inches": "tight", "facecolor": "white"}),
    ):
        path = stem.with_suffix(suffix)
        fig.savefig(path, **kwargs)
        outputs.append(str(path))
    if include_tiff:
        path = stem.with_suffix(".tiff")
        fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def _files(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".png", ".pdf", ".svg", ".tiff", ".eps"}:
            rows.append({"path": str(path.relative_to(root)).replace("\\", "/"), "bytes": int(path.stat().st_size)})
    return rows


def _model_names(class_dir: Path, comparison: pd.DataFrame | None) -> list[str]:
    names: set[str] = set()
    if comparison is not None and "model" in comparison:
        names.update(str(value) for value in comparison["model"].dropna())
    if not names:
        names.update(path.name for path in class_dir.iterdir() if path.is_dir() and (path / "best_model.pt").is_file())
    ordered = [name for name in MODEL_ORDER if name in names]
    ordered.extend(sorted(names - set(ordered)))
    if not ordered:
        raise FileNotFoundError(f"no classifier checkpoint found below {class_dir}")
    return ordered


def _read_comparison(class_dir: Path) -> tuple[pd.DataFrame, str, Path]:
    full = class_dir / "all_window_model_comparison.csv"
    capped = class_dir / "model_comparison.csv"
    if full.is_file():
        frame, denominator, source = pd.read_csv(full), "all eligible held-out test windows", full
    elif capped.is_file():
        frame, denominator, source = pd.read_csv(capped), "event-balanced sensitivity windows (not the primary denominator)", capped
    else:
        raise FileNotFoundError(f"missing all_window_model_comparison.csv below {class_dir}")
    if "model" not in frame:
        raise ValueError(f"comparison table has no model column: {source}")
    return frame, denominator, source


def _history(class_dir: Path, model: str) -> pd.DataFrame | None:
    path = class_dir / model / "history.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if "epoch" not in frame:
        raise ValueError(f"history table has no epoch column: {path}")
    return frame


def _best_epoch(class_dir: Path, model: str, history: pd.DataFrame) -> int | None:
    for name in ("all_window_metrics.json", "metrics.json"):
        path = class_dir / model / name
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8")).get("best_epoch")
            except (OSError, ValueError, json.JSONDecodeError):
                value = None
            if value is not None:
                return int(value)
    if "val_macro_f1" in history and history["val_macro_f1"].notna().any():
        return int(history.loc[history["val_macro_f1"].idxmax(), "epoch"])
    return None


def _labels_from_confusion(matrix: pd.DataFrame, prefix: str) -> list[str]:
    labels = [str(value) for value in matrix.index]
    if len(labels) == matrix.shape[0] and any(label.startswith(prefix) for label in labels):
        return labels
    return [f"{prefix}{idx + 1:02d}" for idx in range(matrix.shape[0])]


def _read_confusion(class_dir: Path, model: str, prefix: str) -> tuple[np.ndarray, list[str], str, Path]:
    model_dir = class_dir / model
    full = model_dir / "confusion_matrix_all_windows.csv"
    capped = model_dir / "confusion_matrix.csv"
    path = full if full.is_file() else capped if capped.is_file() else None
    if path is None:
        raise FileNotFoundError(f"missing confusion matrix for {model}")
    frame = pd.read_csv(path, index_col=0)
    values = frame.apply(pd.to_numeric, errors="raise").to_numpy(float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"confusion matrix is not square: {path}")
    return values, _labels_from_confusion(frame, prefix), "all_window" if path == full else "capped_sensitivity", path


def _render_learning_curves(class_dir: Path, models: list[str], figure_dir: Path, *, metric: str, ylabel: str, stem: str, include_tiff: bool) -> list[str]:
    histories = {model: _history(class_dir, model) for model in models}
    n = len(models)
    cols = min(3, max(1, n))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(183 / 25.4, max(58, 50 * rows) / 25.4), squeeze=False, constrained_layout=True)
    flat = axes.ravel()
    for index, model in enumerate(models):
        ax = flat[index]
        ax.text(-0.14, 1.07, ascii_lowercase[index] if index < len(ascii_lowercase) else str(index + 1), transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
        history = histories[model]
        if history is None:
            ax.text(0.5, 0.5, "history unavailable", ha="center", va="center", color=MID_GREY, transform=ax.transAxes)
            ax.set_title(DISPLAY.get(model, model), loc="left")
            ax.set_axis_off()
            continue
        train_col, val_col = f"train_{metric}", f"val_{metric}"
        if train_col not in history or val_col not in history:
            ax.text(0.5, 0.5, f"{train_col}/{val_col}\nmissing", ha="center", va="center", color=MID_GREY, transform=ax.transAxes)
            ax.set_axis_off()
            continue
        x = pd.to_numeric(history["epoch"], errors="coerce")
        train = pd.to_numeric(history[train_col], errors="coerce")
        val = pd.to_numeric(history[val_col], errors="coerce")
        if metric == "loss":
            train, val = train.clip(lower=1e-6), val.clip(lower=1e-6)
            ax.set_yscale("log")
        ax.plot(x, train, color=BLUE, lw=1.15, label="Training")
        ax.plot(x, val, color=ORANGE, lw=1.15, ls="--", label="Validation")
        selected = _best_epoch(class_dir, model, history)
        if selected is not None:
            row = history.iloc[int(np.argmin(np.abs(x.to_numpy(float) - selected)))]
            value = max(float(row[val_col]), 1e-6) if metric == "loss" else float(row[val_col])
            ax.scatter([row["epoch"]], [value], marker="*", s=50, color=INK, zorder=5)
            ax.annotate(f"best={selected}", (row["epoch"], value), xytext=(4, 5), textcoords="offset points", fontsize=5.5)
        if metric == "macro_f1":
            ax.set_ylim(0, 1)
        ax.set_title(DISPLAY.get(model, model), loc="left", fontsize=7.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D7DADD", linewidth=0.45)
        ax.set_axisbelow(True)
    for ax in flat[n:]:
        ax.axis("off")
    fig.legend(handles=[Line2D([], [], color=BLUE, lw=1.2, label="Training"), Line2D([], [], color=ORANGE, lw=1.2, ls="--", label="Validation"), Line2D([], [], marker="*", linestyle="none", color=INK, markersize=7, label="Selected checkpoint")], loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=6.5)
    fig.suptitle("Learning dynamics for frozen morphology classification", fontsize=9, fontweight="bold", y=1.08)
    fig.text(0.01, -0.015, "Curves are shown exactly as logged; no smoothing or post-hoc checkpoint selection is applied.", fontsize=5.8, color=MID_GREY)
    return _save(fig, figure_dir / stem, include_tiff=include_tiff)


def _render_model_comparison(frame: pd.DataFrame, models: list[str], figure_dir: Path, denominator: str, *, include_tiff: bool) -> list[str]:
    table = frame.set_index("model").reindex(models).reset_index()
    metrics = [(column, title) for column, title in (("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro-F1"), ("macro_auprc", "Macro-AUPRC")) if column in table]
    if not metrics:
        raise ValueError("comparison table has none of the primary metrics")
    fig, axes = plt.subplots(1, len(metrics), figsize=(183 / 25.4, 62 / 25.4), sharey=True, squeeze=False, constrained_layout=True)
    flat, y = axes.ravel(), np.arange(len(table))
    for index, (ax, (column, title)) in enumerate(zip(flat, metrics)):
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(float)
        colours = [MODEL_COLOURS.get(str(model), BLUE) for model in table.model]
        ax.hlines(y, 0, np.nan_to_num(values, nan=0.0), color=LIGHT_GREY, linewidth=0.9, zorder=1)
        ax.scatter(values, y, s=30, color=colours, edgecolor="white", linewidth=0.45, zorder=3)
        for row, value in enumerate(values):
            if np.isfinite(value):
                ax.text(min(float(value) + 0.025, 0.92), row, f"{value:.3f}", va="center", fontsize=5.6)
        ax.set_xlim(0, 1); ax.set_xlabel("Score"); ax.set_title(title, loc="left", fontsize=7.5, fontweight="bold")
        ax.grid(axis="x", color="#D7DADD", linewidth=0.45); ax.set_axisbelow(True)
        ax.text(-0.14, 1.06, ascii_lowercase[index], transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    flat[0].set_yticks(y, [DISPLAY.get(str(model), str(model)) for model in table.model], fontsize=6.2); flat[0].invert_yaxis()
    n = int(pd.to_numeric(table["n"], errors="coerce").max()) if "n" in table and table["n"].notna().any() else 0
    fig.suptitle(f"Held-out model comparison | {denominator} | n={n:,}", fontsize=9, fontweight="bold", y=1.03)
    fig.text(0.01, -0.035, "Primary metrics are kept separate from the event-balanced sensitivity table.", fontsize=5.8, color=MID_GREY)
    return _save(fig, figure_dir / "figure_3_model_comparison_primary_metrics", include_tiff=include_tiff)


def _render_best_summary(class_dir: Path, model: str, figure_dir: Path, prefix: str, *, include_tiff: bool) -> tuple[list[str], int, str]:
    matrix, labels, mode, source = _read_confusion(class_dir, model, prefix)
    support = matrix.sum(axis=1); normalized = matrix / np.maximum(support[:, None], 1.0); recall = np.diag(normalized); classes = matrix.shape[0]
    ticks = np.arange(0, classes, max(1, int(math.ceil(classes / 12))))
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 92 / 25.4), gridspec_kw={"width_ratios": [1.22, 0.92]}, constrained_layout=True)
    im = axes[0].imshow(normalized, vmin=0, vmax=1, cmap="Blues", interpolation="nearest", aspect="equal")
    axes[0].set_xticks(ticks, [labels[index] for index in ticks], rotation=45, ha="right"); axes[0].set_yticks(ticks, [labels[index] for index in ticks]); axes[0].set_xlabel("Predicted category"); axes[0].set_ylabel("True category"); axes[0].set_title("Row-normalized confusion", loc="left", fontsize=7.5, fontweight="bold"); axes[0].text(-0.14, 1.04, "a", transform=axes[0].transAxes, fontsize=9, fontweight="bold", va="top")
    fig.colorbar(im, ax=axes[0], fraction=0.04, pad=0.025, label="Row fraction")
    valid = support > 0
    axes[1].scatter(support[valid], recall[valid], s=20, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.4)
    if valid.any():
        axes[1].set_xscale("log"); axes[1].set_xlim(max(1, float(support[valid].min()) * 0.7), max(2, float(support[valid].max()) * 1.5))
    axes[1].set_ylim(0, 1); axes[1].set_xlabel("Test-window support (log scale)"); axes[1].set_ylabel("Per-category recall"); axes[1].set_title("Support versus recall", loc="left", fontsize=7.5, fontweight="bold"); axes[1].grid(axis="y", color="#D7DADD", linewidth=0.45); axes[1].set_axisbelow(True)
    top = np.argsort(-support)[: min(8, classes)]
    for index, category in enumerate(top):
        if support[category] > 0:
            axes[1].annotate(labels[category], (support[category], recall[category]), xytext=(4, 4 if index % 2 == 0 else -9), textcoords="offset points", fontsize=5.5, fontweight="bold", arrowprops={"arrowstyle": "-", "lw": 0.35, "color": MID_GREY})
    axes[1].text(-0.14, 1.04, "b", transform=axes[1].transAxes, fontsize=9, fontweight="bold", va="top")
    fig.suptitle(f"{DISPLAY.get(model, model)} | all {classes} categories | n={int(support.sum()):,}", fontsize=9, fontweight="bold", y=1.02); fig.text(0.01, -0.02, f"Confusion source: {source.name}; denominator mode: {mode}.", fontsize=5.8, color=MID_GREY)
    return _save(fig, figure_dir / f"figure_4_{model}_confusion_and_support", include_tiff=include_tiff), classes, mode


def _render_per_stream(class_dir: Path, figure_dir: Path, models: list[str], *, include_tiff: bool) -> tuple[list[str], str | None]:
    path = class_dir / "per_animal_channel_metrics.csv"
    if not path.is_file():
        return [], None
    data = pd.read_csv(path); required = {"animal_id", "channel_name", "model", "macro_f1"}
    if not required.issubset(data.columns):
        return [], f"per-stream table missing columns: {sorted(required - set(data.columns))}"
    data["stream"] = data["animal_id"].astype(str) + "/" + data["channel_name"].astype(str)
    pivot = data.pivot_table(index="stream", columns="model", values="macro_f1").reindex(columns=models); values = pivot.to_numpy(float)
    fig, ax = plt.subplots(figsize=(125 / 25.4, max(70, 14 + 9 * len(pivot)) / 25.4), constrained_layout=True)
    im = ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto"); ax.set_xticks(np.arange(len(models)), [DISPLAY.get(model, model) for model in models], rotation=30, ha="right", fontsize=6); ax.set_yticks(np.arange(len(pivot)), pivot.index, fontsize=6.2); ax.set_xlabel("Model (cell labels = Macro-F1)"); ax.set_ylabel("Independent channel stream"); ax.set_title("Per-stream Macro-F1 on complete test windows", loc="left", fontsize=7.5, fontweight="bold")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=5.0, color="white" if value > 0.55 else INK)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Macro-F1")
    return _save(fig, figure_dir / "figure_5_per_stream_macro_f1", include_tiff=include_tiff), None


def _render_supplementary(class_dir: Path, figure_dir: Path, models: list[str], prefix: str, *, include_tiff: bool) -> list[str]:
    root = figure_dir / "supplementary_confusion_matrices"; outputs: list[str] = []
    for model in models:
        matrix, labels, mode, _ = _read_confusion(class_dir, model, prefix); normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0); classes = matrix.shape[0]; ticks = np.arange(0, classes, max(1, int(math.ceil(classes / 12))))
        fig, ax = plt.subplots(figsize=(110 / 25.4, 105 / 25.4), constrained_layout=True); im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues", interpolation="nearest", aspect="equal"); ax.set_xticks(ticks, [labels[index] for index in ticks], rotation=45, ha="right"); ax.set_yticks(ticks, [labels[index] for index in ticks]); ax.set_xlabel("Predicted category"); ax.set_ylabel("True category"); ax.set_title(f"{DISPLAY.get(model, model)} | {classes}×{classes} | {mode}", loc="left", fontsize=7.5, fontweight="bold"); fig.colorbar(im, ax=ax, fraction=0.04, pad=0.025, label="Row fraction"); outputs.extend(_save(fig, root / f"confusion_{model}", include_tiff=include_tiff))
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-dir", type=Path, required=True, help="completed classifier output directory")
    parser.add_argument("--figure-dir", type=Path, required=True, help="new figure directory")
    parser.add_argument("--label-prefix", default="PG")
    parser.add_argument("--best-model", default=None)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--include-tiff", action="store_true")
    args = parser.parse_args(argv)
    class_dir, figure_dir = args.class_dir.resolve(), args.figure_dir.resolve()
    if not class_dir.is_dir():
        raise FileNotFoundError(class_dir)
    if figure_dir.exists() and any(figure_dir.iterdir()) and not args.allow_existing:
        raise FileExistsError(f"figure directory is non-empty: {figure_dir}; use --allow-existing")
    figure_dir.mkdir(parents=True, exist_ok=True)
    comparison, denominator, comparison_path = _read_comparison(class_dir); models = _model_names(class_dir, comparison)
    if args.best_model is None:
        candidates = comparison.loc[comparison.model.astype(str).isin(models)].copy()
        if "macro_f1" in candidates:
            candidates["macro_f1"] = pd.to_numeric(candidates["macro_f1"], errors="coerce"); candidates = candidates.sort_values("macro_f1", ascending=False, na_position="last")
        best_model = str(candidates.iloc[0].model) if len(candidates) else models[0]
    else:
        best_model = str(args.best_model)
        if best_model not in models:
            raise ValueError(f"--best-model {best_model!r} is not in the model table")
    outputs: list[str] = []
    outputs.extend(_render_learning_curves(class_dir, models, figure_dir, metric="loss", ylabel="NLL (log scale)", stem="figure_1_training_loss", include_tiff=args.include_tiff))
    outputs.extend(_render_learning_curves(class_dir, models, figure_dir, metric="macro_f1", ylabel="Macro-F1", stem="figure_2_training_macro_f1", include_tiff=args.include_tiff))
    outputs.extend(_render_model_comparison(comparison, models, figure_dir, denominator, include_tiff=args.include_tiff))
    best_outputs, classes, best_mode = _render_best_summary(class_dir, best_model, figure_dir, str(args.label_prefix), include_tiff=args.include_tiff); outputs.extend(best_outputs)
    stream_outputs, stream_note = _render_per_stream(class_dir, figure_dir, models, include_tiff=args.include_tiff); outputs.extend(stream_outputs)
    outputs.extend(_render_supplementary(class_dir, figure_dir, models, str(args.label_prefix), include_tiff=args.include_tiff))
    manifest = {
        "schema_version": 2,
        "renderer": "scripts/render_pg36_classifier_summary.py",
        "models": models,
        "classes": int(classes),
        "label_prefix": str(args.label_prefix),
        "best_model": best_model,
        "formal_denominator": denominator,
        "best_confusion_denominator_mode": best_mode,
        "comparison_source": str(comparison_path),
        "checkpoint_selection": "validation Macro-F1 when available",
        "retrained_by_renderer": False,
        "stream_note": stream_note,
        "outputs": _files(figure_dir),
        "figure_contract": {
            "core_claim": "compare the completed classifiers under one explicit held-out-window denominator",
            "data_unit": "window rows; each channel stream remains independent",
            "transform": "row-normalized confusion and unsmoothed logged learning curves",
            "reviewer_risk": "event-balanced sensitivity results must not be presented as the primary test denominator",
        },
    }
    (figure_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (figure_dir / "README_zh.md").write_text(
        "# PG 分类器论文版汇总图\n\n"
        f"- 当前模型：{', '.join(models)}。冻结类别数：{classes}，标签前缀：`{args.label_prefix}`。\n"
        f"- 正式分母：{denominator}。渲染器不会训练或改变 checkpoint。\n"
        "- `figure_1_training_loss`、`figure_2_training_macro_f1`：逐模型训练/验证曲线，黑星为已记录的验证集 checkpoint。\n"
        "- `figure_3_model_comparison_primary_metrics`：balanced accuracy、Macro-F1、Macro-AUPRC。\n"
        f"- `figure_4_{best_model}_confusion_and_support`：最佳模型的完整 {classes}×{classes} 混淆和支持量-召回关系。\n"
        "- `figure_5_per_stream_macro_f1`：逐鼠逐通道独立流稳健性（若源表存在）。\n"
        "- `supplementary_confusion_matrices/`：每个模型的完整类别混淆矩阵。\n"
        "- 所有输出文件、输入表和分母写入 `manifest.json`；PG 类别是探索性形态标签，不是临床亚型。\n",
        encoding="utf-8",
    )
    print(json.dumps({"figure_dir": str(figure_dir), "models": models, "classes": classes, "outputs": len(outputs)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
