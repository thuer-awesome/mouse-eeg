"""Publication figures for completed same-day mouse epilepsy LOSO runs.

This module is deliberately read-only with respect to training artifacts.  It
uses the clinical report tables as the metric contract, accepts histories only
from launcher tasks marked complete, and keeps animal-level variability
separate from pooled window-level summaries.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from string import ascii_uppercase
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

__all__ = [
    "EpilepsySciFigureData",
    "generate_epilepsy_sci_figures",
    "load_epilepsy_sci_figure_data",
]


OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
    "gray": "#8F8F8F",
    "light_gray": "#D0D0D0",
}

MODEL_LABELS = {
    "eegwavenet_paper_exact": "EEGWaveNet",
    "eegnet_8_2_paper_exact": "EEGNet-8,2",
}
MODEL_COLORS = {
    "eegwavenet_paper_exact": OKABE_ITO["blue"],
    "eegnet_8_2_paper_exact": OKABE_ITO["orange"],
}
MODEL_MARKERS = {
    "eegwavenet_paper_exact": "o",
    "eegnet_8_2_paper_exact": "s",
}

WINDOW_METRICS_FIG2 = ("accuracy", "sensitivity", "specificity", "f1")
WINDOW_METRICS_FIG6 = ("auroc", "auprc", "balanced_accuracy", "mcc")
OPERATING_METRICS = (
    "threshold_selected_on_validation",
    "event_sensitivity",
    "false_alarms_per_hour",
)
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "f1": "F1 score",
    "balanced_accuracy": "Balanced accuracy",
    "mcc": "Matthews correlation coefficient",
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "threshold_selected_on_validation": "Validation-selected threshold",
    "event_sensitivity": "Event sensitivity",
    "false_alarms_per_hour": "False alarms per hour",
}

_KEY_COLUMNS = ["model", "fold", "test_animal"]
_REQUIRED_REPORT_FILES = {
    "inventory": "run_inventory.csv",
    "per_animal": "per_animal_metrics.csv",
    "pooled": "model_pooled_metrics.csv",
    "confusions": "confusion_matrices.csv",
}


@dataclass(frozen=True)
class EpilepsySciFigureData:
    """Validated source tables used by the publication figures."""

    runs_root: Path
    clinical_report_dir: Path
    inventory: pd.DataFrame
    per_animal: pd.DataFrame
    pooled: pd.DataFrame
    confusions: pd.DataFrame
    split_design: pd.DataFrame
    animal_scale: pd.DataFrame
    fold_scale: pd.DataFrame
    histories: pd.DataFrame
    input_paths: tuple[Path, ...]

    @property
    def models(self) -> list[str]:
        return _ordered_models(self.per_animal["model"].astype(str).unique())

    @property
    def animals(self) -> list[str]:
        return sorted(
            self.split_design["animal_id"].astype(str).unique(), key=_natural_id_key
        )

    @property
    def folds(self) -> list[int]:
        return sorted(self.split_design["fold"].astype(int).unique().tolist())


def load_epilepsy_sci_figure_data(
    runs_root: str | Path,
    clinical_report_dir: str | Path | None = None,
) -> EpilepsySciFigureData:
    """Load and validate completed-fold inputs without touching training state."""

    root = Path(runs_root).resolve()
    report_dir = (
        Path(clinical_report_dir).resolve()
        if clinical_report_dir is not None
        else root / "clinical_report"
    )
    queue_path = root / "queue_manifest.json"
    if not queue_path.is_file():
        raise FileNotFoundError(f"missing queue manifest: {queue_path}")

    report_paths = {
        name: report_dir / filename for name, filename in _REQUIRED_REPORT_FILES.items()
    }
    tables = {name: _read_csv(path) for name, path in report_paths.items()}
    inventory = tables["inventory"].copy()
    _require_columns(
        inventory,
        [*_KEY_COLUMNS, "validation_animal", "launcher_status", "artifact_state", "run_dir"],
        report_paths["inventory"],
    )
    inventory["fold"] = pd.to_numeric(inventory["fold"], errors="raise").astype(int)
    complete_mask = inventory["launcher_status"].astype(str).eq("completed") & inventory[
        "artifact_state"
    ].astype(str).eq("complete")
    completed = inventory.loc[complete_mask].copy()
    if completed.empty:
        raise ValueError("no launcher-completed folds are available for figures")
    _reject_duplicate_keys(completed, report_paths["inventory"])

    per_animal_all = tables["per_animal"].copy()
    _require_columns(
        per_animal_all,
        [
            *_KEY_COLUMNS,
            "n_sessions",
            "n_windows",
            "positive_windows",
            "duration_hours",
            *WINDOW_METRICS_FIG2,
            *WINDOW_METRICS_FIG6,
            *OPERATING_METRICS,
        ],
        report_paths["per_animal"],
    )
    per_animal_all["fold"] = pd.to_numeric(
        per_animal_all["fold"], errors="raise"
    ).astype(int)
    _reject_duplicate_keys(per_animal_all, report_paths["per_animal"])
    complete_keys = completed[_KEY_COLUMNS]
    missing_metrics = complete_keys.merge(
        per_animal_all[_KEY_COLUMNS], on=_KEY_COLUMNS, how="left", indicator=True
    )
    if missing_metrics["_merge"].ne("both").any():
        missing = missing_metrics.loc[
            missing_metrics["_merge"].ne("both"), _KEY_COLUMNS
        ].to_dict(orient="records")
        raise ValueError(f"completed folds missing per-animal metrics: {missing}")
    per_animal = per_animal_all.merge(
        complete_keys, on=_KEY_COLUMNS, how="inner", validate="one_to_one"
    ).sort_values(["model", "fold"], ignore_index=True)
    _coerce_finite_numeric(
        per_animal,
        [
            "n_sessions",
            "n_windows",
            "positive_windows",
            "duration_hours",
            *WINDOW_METRICS_FIG2,
            *WINDOW_METRICS_FIG6,
            *OPERATING_METRICS,
        ],
        report_paths["per_animal"],
    )

    model_counts = completed.groupby("model", sort=False)["fold"].nunique()
    pooled = tables["pooled"].copy()
    _require_columns(
        pooled,
        ["model", "completed_folds", "n_windows", *WINDOW_METRICS_FIG2, *WINDOW_METRICS_FIG6],
        report_paths["pooled"],
    )
    pooled = _filter_one_row_per_model(
        pooled, model_counts.index, report_paths["pooled"]
    )
    for row in pooled.itertuples(index=False):
        expected = int(model_counts.loc[str(row.model)])
        if int(row.completed_folds) != expected:
            raise ValueError(
                f"stale pooled report for {row.model}: completed_folds="
                f"{row.completed_folds}, inventory={expected}"
            )

    confusions = tables["confusions"].copy()
    _require_columns(
        confusions,
        ["model", "tn", "fp", "fn", "tp", "n_windows"],
        report_paths["confusions"],
    )
    confusions = _filter_one_row_per_model(
        confusions, model_counts.index, report_paths["confusions"]
    )
    _coerce_finite_numeric(
        confusions, ["tn", "fp", "fn", "tp", "n_windows"], report_paths["confusions"]
    )

    queue = json.loads(queue_path.read_text(encoding="utf-8-sig"))
    split_design = _build_split_design(queue, completed)
    animal_scale = _build_animal_scale(per_animal)
    fold_scale = _build_fold_scale(split_design, animal_scale)
    histories, history_paths = _load_completed_histories(completed)

    input_paths = tuple(
        dict.fromkeys([queue_path, *report_paths.values(), *history_paths])
    )
    return EpilepsySciFigureData(
        runs_root=root,
        clinical_report_dir=report_dir,
        inventory=completed.sort_values(["model", "fold"], ignore_index=True),
        per_animal=per_animal,
        pooled=pooled,
        confusions=confusions,
        split_design=split_design,
        animal_scale=animal_scale,
        fold_scale=fold_scale,
        histories=histories,
        input_paths=input_paths,
    )


def generate_epilepsy_sci_figures(
    runs_root: str | Path,
    output_dir: str | Path | None = None,
    *,
    clinical_report_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Generate six manuscript figures and their reproducibility bundle."""

    data = load_epilepsy_sci_figure_data(runs_root, clinical_report_dir)
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else data.runs_root / "sci_figures"
    )
    destination.mkdir(parents=True, exist_ok=True)
    source_dir = destination / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    sources: dict[str, pd.DataFrame] = {}
    figures: dict[str, mpl.figure.Figure] = {}
    figures["figure_1_loso_design_and_scale"], figure_1_sources = _figure_1(data)
    sources.update(figure_1_sources)
    figures["figure_2_window_metrics_by_animal"], figure_2_sources = _figure_2(data)
    sources.update(figure_2_sources)
    figures["figure_3_pooled_confusion_matrices"], figure_3_sources = _figure_3(data)
    sources.update(figure_3_sources)
    figures["figure_4_training_validation_loss"], figure_4_sources = _figure_4(data)
    sources.update(figure_4_sources)
    figures["figure_5_operating_characteristics"], figure_5_sources = _figure_5(data)
    sources.update(figure_5_sources)
    figures["figure_6_discrimination_by_animal"], figure_6_sources = _figure_6(data)
    sources.update(figure_6_sources)

    output_paths: dict[str, Path] = {}
    figure_outputs: dict[str, list[Path]] = {}
    for stem, figure in figures.items():
        exported = _export_figure(figure, destination / stem)
        plt.close(figure)
        figure_outputs[stem] = exported
        for path in exported:
            output_paths[f"{stem}_{path.suffix.lstrip('.')}"] = path

    source_paths: dict[str, Path] = {}
    for name, frame in sources.items():
        path = source_dir / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        source_paths[name] = path
        output_paths[name] = path

    captions_path = destination / "figure_captions.md"
    captions_path.write_text(_render_captions(data), encoding="utf-8", newline="\n")
    output_paths["figure_captions"] = captions_path

    manifest_path = destination / "figure_manifest.json"
    manifest = _build_manifest(
        data=data,
        destination=destination,
        figure_outputs=figure_outputs,
        source_paths=source_paths,
        captions_path=captions_path,
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    output_paths["figure_manifest"] = manifest_path
    return output_paths


def _figure_1(
    data: EpilepsySciFigureData,
) -> tuple[mpl.figure.Figure, dict[str, pd.DataFrame]]:
    with mpl.rc_context(_publication_rc()):
        fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, width_ratios=(1.12, 1.0))
        ax_design = fig.add_subplot(grid[:, 0])
        ax_animals = fig.add_subplot(grid[0, 1])
        ax_folds = fig.add_subplot(grid[1, 1])

        animals = data.animals
        folds = data.folds
        role_code = {"train": 0, "validation": 1, "test": 2}
        role_text = {"train": "Tr", "validation": "Va", "test": "Te"}
        role_colors = [
            OKABE_ITO["sky_blue"],
            OKABE_ITO["orange"],
            OKABE_ITO["vermillion"],
        ]
        matrix = np.full((len(folds), len(animals)), np.nan)
        role_lookup: dict[tuple[int, str], str] = {}
        for row in data.split_design.itertuples(index=False):
            i = folds.index(int(row.fold))
            j = animals.index(str(row.animal_id))
            role = str(row.role)
            matrix[i, j] = role_code[role]
            role_lookup[(int(row.fold), str(row.animal_id))] = role
        masked = np.ma.masked_invalid(matrix)
        ax_design.imshow(
            masked,
            cmap=ListedColormap(role_colors),
            vmin=-0.5,
            vmax=2.5,
            aspect="auto",
        )
        for i, fold in enumerate(folds):
            for j, animal in enumerate(animals):
                role = role_lookup.get((fold, animal))
                if role is None:
                    continue
                text_color = "white" if role == "test" else "black"
                ax_design.text(
                    j,
                    i,
                    role_text[role],
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7,
                    fontweight="bold",
                )
        ax_design.set_xticks(range(len(animals)), animals)
        ax_design.set_yticks(range(len(folds)), [f"Fold {fold}" for fold in folds])
        ax_design.set_xlabel("Animal")
        ax_design.set_ylabel("LOSO fold")
        ax_design.set_title("Animal-level allocation")
        ax_design.set_xticks(np.arange(-0.5, len(animals), 1), minor=True)
        ax_design.set_yticks(np.arange(-0.5, len(folds), 1), minor=True)
        ax_design.grid(which="minor", color="white", linewidth=1.3)
        ax_design.tick_params(which="minor", bottom=False, left=False)
        ax_design.legend(
            handles=[
                Patch(facecolor=role_colors[0], edgecolor="black", label="Train (Tr)"),
                Patch(facecolor=role_colors[1], edgecolor="black", label="Validation (Va)"),
                Patch(facecolor=role_colors[2], edgecolor="black", label="Test (Te)"),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            frameon=False,
            handlelength=1.1,
        )

        scale = data.animal_scale.set_index("animal_id").reindex(animals).reset_index()
        x = np.arange(len(scale))
        nonseizure = scale["nonseizure_windows"].to_numpy(dtype=float) / 1000.0
        seizure = scale["positive_windows"].to_numpy(dtype=float) / 1000.0
        ax_animals.bar(
            x,
            nonseizure,
            color=OKABE_ITO["gray"],
            edgecolor="black",
            linewidth=0.5,
            label="Non-seizure",
        )
        ax_animals.bar(
            x,
            seizure,
            bottom=nonseizure,
            color=OKABE_ITO["vermillion"],
            edgecolor="black",
            linewidth=0.5,
            hatch="///",
            label="Seizure",
        )
        ax_animals.set_xticks(x, scale["animal_id"].astype(str))
        ax_animals.set_ylabel("Windows (thousands)")
        ax_animals.set_title("Natural test distribution by animal")
        ax_animals.legend(frameon=False, ncol=2, loc="upper right")
        _despine(ax_animals)

        role_order = ["train", "validation", "test"]
        width = 0.24
        fold_scale = data.fold_scale.copy()
        for offset, role in enumerate(role_order):
            values = (
                fold_scale.loc[fold_scale["role"].eq(role)]
                .set_index("fold")
                .reindex(folds)["n_windows"]
                .to_numpy(dtype=float)
                / 1000.0
            )
            ax_folds.bar(
                np.arange(len(folds)) + (offset - 1) * width,
                values,
                width,
                color=role_colors[offset],
                edgecolor="black",
                linewidth=0.5,
                hatch=("//", "..", "xx")[offset],
                label=role.title(),
            )
        ax_folds.set_xticks(range(len(folds)), [str(fold) for fold in folds])
        ax_folds.set_xlabel("LOSO fold")
        ax_folds.set_ylabel("Windows (thousands)")
        ax_folds.set_title("Window count by split role")
        ax_folds.legend(frameon=False, ncol=3, loc="upper right")
        _despine(ax_folds)

        _panel_labels([ax_design, ax_animals, ax_folds])
        fig.suptitle("Figure 1 | Fixed six-fold LOSO design and data scale", fontsize=10)
    return fig, {
        "figure_1_split_design": data.split_design.copy(),
        "figure_1_animal_scale": data.animal_scale.copy(),
        "figure_1_fold_scale": data.fold_scale.copy(),
    }


def _figure_2(
    data: EpilepsySciFigureData,
) -> tuple[mpl.figure.Figure, dict[str, pd.DataFrame]]:
    long, summary = _metric_source_tables(data.per_animal, WINDOW_METRICS_FIG2)
    with mpl.rc_context(_publication_rc()):
        fig, axes = plt.subplots(
            2, 2, figsize=(7.2, 5.2), constrained_layout=True, squeeze=False
        )
        for ax, metric in zip(axes.flat, WINDOW_METRICS_FIG2, strict=True):
            _paired_animal_panel(ax, long, summary, metric, data.models)
            ax.set_ylim(-0.05, 1.05)
        _panel_labels(list(axes.flat))
        fig.suptitle(
            "Figure 2 | Window-level performance across held-out animals", fontsize=10
        )
    return fig, {
        "figure_2_window_metrics": long,
        "figure_2_window_metric_summary": summary,
    }


def _figure_3(
    data: EpilepsySciFigureData,
) -> tuple[mpl.figure.Figure, dict[str, pd.DataFrame]]:
    source_rows: list[dict[str, Any]] = []
    matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model in data.models:
        row = data.confusions.loc[data.confusions["model"].astype(str).eq(model)].iloc[0]
        counts = np.asarray([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=float)
        denominators = counts.sum(axis=1, keepdims=True)
        normalized = np.divide(
            counts,
            denominators,
            out=np.full_like(counts, np.nan),
            where=denominators > 0,
        )
        matrices[model] = counts, normalized
        for true_index, true_class in enumerate(("Non-seizure", "Seizure")):
            for pred_index, predicted_class in enumerate(("Non-seizure", "Seizure")):
                source_rows.append(
                    {
                        "model": model,
                        "true_class": true_class,
                        "predicted_class": predicted_class,
                        "count": int(counts[true_index, pred_index]),
                        "row_fraction": normalized[true_index, pred_index],
                    }
                )
    source = pd.DataFrame(source_rows)
    with mpl.rc_context(_publication_rc()):
        fig, axes = plt.subplots(
            len(data.models),
            2,
            figsize=(7.2, 2.55 * len(data.models)),
            constrained_layout=True,
            squeeze=False,
        )
        raw_vmax = max(float(np.nanmax(pair[0])) for pair in matrices.values())
        for row_index, model in enumerate(data.models):
            counts, normalized = matrices[model]
            _confusion_panel(
                axes[row_index, 0],
                counts,
                normalized=False,
                vmax=raw_vmax,
                title=f"{_model_label(model)} | pooled counts",
            )
            _confusion_panel(
                axes[row_index, 1],
                normalized,
                normalized=True,
                vmax=1.0,
                title=f"{_model_label(model)} | row-normalized",
            )
        _panel_labels(list(axes.flat))
        fig.suptitle(
            "Figure 3 | Pooled out-of-fold confusion matrices", fontsize=10
        )
    return fig, {"figure_3_confusion_matrices": source}


def _figure_4(
    data: EpilepsySciFigureData,
) -> tuple[mpl.figure.Figure, dict[str, pd.DataFrame]]:
    histories = data.histories.copy()
    best = (
        histories.loc[histories["is_best_epoch"]]
        [
            [
                "model",
                "fold",
                "test_animal",
                "validation_animal",
                "epoch",
                "train_loss",
                "val_loss",
            ]
        ]
        .rename(columns={"epoch": "best_epoch", "val_loss": "best_val_loss"})
        .sort_values(["model", "fold"], ignore_index=True)
    )
    folds = data.folds
    models = data.models
    with mpl.rc_context(_publication_rc(small_multiples=True)):
        fig, axes = plt.subplots(
            len(models),
            len(folds),
            figsize=(11.2, 2.5 * len(models)),
            constrained_layout=True,
            squeeze=False,
        )
        for model_index, model in enumerate(models):
            for fold_index, fold in enumerate(folds):
                ax = axes[model_index, fold_index]
                group = histories.loc[
                    histories["model"].astype(str).eq(model)
                    & histories["fold"].astype(int).eq(fold)
                ].sort_values("epoch")
                if group.empty:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, "Not completed", ha="center", va="center")
                    continue
                ax.plot(
                    group["epoch"],
                    group["train_loss"],
                    color=OKABE_ITO["blue"],
                    linestyle="-",
                    linewidth=1.2,
                )
                ax.plot(
                    group["epoch"],
                    group["val_loss"],
                    color=OKABE_ITO["vermillion"],
                    linestyle="--",
                    linewidth=1.2,
                )
                best_row = group.loc[group["is_best_epoch"]].iloc[0]
                ax.axvline(
                    float(best_row["epoch"]),
                    color=OKABE_ITO["black"],
                    linestyle=":",
                    linewidth=0.9,
                )
                ax.scatter(
                    [best_row["epoch"]],
                    [best_row["val_loss"]],
                    marker="*",
                    s=35,
                    color=OKABE_ITO["yellow"],
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=5,
                )
                ax.set_yscale("log")
                ax.set_title(
                    f"Fold {fold} | Test {best_row['test_animal']}\n"
                    f"Best epoch {int(best_row['epoch'])}",
                    fontsize=6.5,
                )
                ax.set_xlabel("Epoch")
                if fold_index == 0:
                    ax.set_ylabel(f"{_model_label(model)}\nWeighted CE loss")
                ax.set_xlim(1, max(30, int(group["epoch"].max())))
                ax.set_xticks([1, 10, 20, 30])
                ax.grid(axis="y", color="#E5E5E5", linewidth=0.45)
                _despine(ax)
        _panel_labels([ax for ax in axes.flat if ax.axison])
        fig.legend(
            handles=[
                Line2D([0], [0], color=OKABE_ITO["blue"], linestyle="-", label="Train"),
                Line2D(
                    [0], [0], color=OKABE_ITO["vermillion"], linestyle="--", label="Validation"
                ),
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle=":",
                    marker="*",
                    markerfacecolor=OKABE_ITO["yellow"],
                    label="Minimum validation loss",
                ),
            ],
            loc="outside lower center",
            ncol=3,
            frameon=False,
        )
        fig.suptitle(
            "Figure 4 | Train and validation loss for every completed fold (log scale)",
            fontsize=10,
        )
    return fig, {
        "figure_4_loss_curves": histories,
        "figure_4_best_epochs": best,
    }


def _figure_5(
    data: EpilepsySciFigureData,
) -> tuple[mpl.figure.Figure, dict[str, pd.DataFrame]]:
    long, summary = _metric_source_tables(data.per_animal, OPERATING_METRICS)
    with mpl.rc_context(_publication_rc()):
        fig, axes = plt.subplots(
            1, 3, figsize=(7.2, 2.85), constrained_layout=True, squeeze=False
        )
        for ax, metric in zip(axes.flat, OPERATING_METRICS, strict=True):
            _paired_animal_panel(ax, long, summary, metric, data.models)
            if metric in {"threshold_selected_on_validation", "event_sensitivity"}:
                ax.set_ylim(-0.05, 1.05)
            else:
                ax.set_ylim(bottom=0)
        _panel_labels(list(axes.flat))
        fig.suptitle(
            "Figure 5 | Validation threshold and event-level operating characteristics",
            fontsize=10,
        )
    return fig, {
        "figure_5_operating_metrics": long,
        "figure_5_operating_metric_summary": summary,
    }


def _figure_6(
    data: EpilepsySciFigureData,
) -> tuple[mpl.figure.Figure, dict[str, pd.DataFrame]]:
    long, summary = _metric_source_tables(data.per_animal, WINDOW_METRICS_FIG6)
    with mpl.rc_context(_publication_rc()):
        fig, axes = plt.subplots(
            2, 2, figsize=(7.2, 5.2), constrained_layout=True, squeeze=False
        )
        for ax, metric in zip(axes.flat, WINDOW_METRICS_FIG6, strict=True):
            _paired_animal_panel(ax, long, summary, metric, data.models)
            if metric == "mcc":
                ax.axhline(0, color="#808080", linewidth=0.7, linestyle=":")
                ax.set_ylim(-1.05, 1.05)
            else:
                ax.axhline(0.5, color="#808080", linewidth=0.7, linestyle=":")
                ax.set_ylim(-0.05, 1.05)
        _panel_labels(list(axes.flat))
        fig.suptitle(
            "Figure 6 | Discrimination and imbalance-aware metrics by animal",
            fontsize=10,
        )
    return fig, {
        "figure_6_discrimination_metrics": long,
        "figure_6_discrimination_metric_summary": summary,
    }


def _metric_source_tables(
    per_animal: pd.DataFrame, metrics: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    identifier_columns = ["model", "fold", "test_animal"]
    long = per_animal[identifier_columns + list(metrics)].melt(
        id_vars=identifier_columns,
        value_vars=list(metrics),
        var_name="metric",
        value_name="value",
    )
    long["metric_label"] = long["metric"].map(METRIC_LABELS)
    long = long.sort_values(["metric", "model", "fold"], ignore_index=True)
    summary = (
        long.groupby(["model", "metric", "metric_label"], sort=False, dropna=False)[
            "value"
        ]
        .agg(mean="mean", sample_sd="std", n="count")
        .reset_index()
    )
    summary["sample_sd_ddof"] = 1
    return long, summary


def _paired_animal_panel(
    ax: mpl.axes.Axes,
    long: pd.DataFrame,
    summary: pd.DataFrame,
    metric: str,
    models: Sequence[str],
) -> None:
    subset = long.loc[long["metric"].eq(metric)].copy()
    animals = sorted(subset["test_animal"].astype(str).unique(), key=_natural_id_key)
    jitter = {
        animal: offset
        for animal, offset in zip(
            animals, np.linspace(-0.08, 0.08, max(len(animals), 1)), strict=True
        )
    }
    for animal in animals:
        animal_rows = subset.loc[subset["test_animal"].astype(str).eq(animal)]
        xs: list[float] = []
        ys: list[float] = []
        for model_index, model in enumerate(models):
            values = animal_rows.loc[animal_rows["model"].astype(str).eq(model), "value"]
            if len(values) != 1:
                continue
            xs.append(model_index + jitter[animal])
            ys.append(float(values.iloc[0]))
        if len(xs) > 1:
            ax.plot(xs, ys, color=OKABE_ITO["light_gray"], linewidth=0.7, zorder=1)
    for model_index, model in enumerate(models):
        model_rows = subset.loc[subset["model"].astype(str).eq(model)]
        xs = [model_index + jitter[str(animal)] for animal in model_rows["test_animal"]]
        ax.scatter(
            xs,
            model_rows["value"],
            s=22,
            marker=MODEL_MARKERS.get(model, "o"),
            color=MODEL_COLORS.get(model, OKABE_ITO["gray"]),
            edgecolor="black",
            linewidth=0.45,
            alpha=0.9,
            zorder=3,
        )
        summary_row = summary.loc[
            summary["model"].astype(str).eq(model) & summary["metric"].eq(metric)
        ]
        if len(summary_row) != 1:
            continue
        mean = float(summary_row.iloc[0]["mean"])
        sd = float(summary_row.iloc[0]["sample_sd"])
        error = 0.0 if math.isnan(sd) else sd
        ax.errorbar(
            model_index,
            mean,
            yerr=error,
            fmt=MODEL_MARKERS.get(model, "o"),
            markersize=7,
            markerfacecolor=MODEL_COLORS.get(model, OKABE_ITO["gray"]),
            markeredgecolor="black",
            markeredgewidth=0.8,
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
            zorder=5,
        )
    ax.set_xticks(range(len(models)), [_model_label(model) for model in models])
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(METRIC_LABELS[metric])
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.6)
    _despine(ax)


def _confusion_panel(
    ax: mpl.axes.Axes,
    matrix: np.ndarray,
    *,
    normalized: bool,
    vmax: float,
    title: str,
) -> None:
    image = ax.imshow(matrix, cmap="cividis", vmin=0, vmax=vmax)
    for row in range(2):
        for column in range(2):
            value = float(matrix[row, column])
            label = f"{value:.1%}" if normalized else f"{int(value):,}"
            threshold = 0.45 * vmax
            color = "white" if value < threshold else "black"
            ax.text(column, row, label, ha="center", va="center", color=color, fontsize=8)
    ax.set_xticks([0, 1], ["Non-seizure", "Seizure"])
    ax.set_yticks([0, 1], ["Non-seizure", "Seizure"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row fraction" if normalized else "Window count")


def _build_split_design(queue: Mapping[str, Any], completed: pd.DataFrame) -> pd.DataFrame:
    folds = queue.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("queue_manifest.json must contain a non-empty folds list")
    used_folds = set(completed["fold"].astype(int))
    rows: list[dict[str, Any]] = []
    for payload in folds:
        if not isinstance(payload, Mapping):
            raise ValueError("queue fold entries must be objects")
        fold = int(payload.get("fold", 0))
        if fold not in used_folds:
            continue
        train = [str(value) for value in payload.get("train_animals", [])]
        validation = str(payload.get("val_animal", ""))
        test = str(payload.get("test_animal", ""))
        if not train or not validation or not test:
            raise ValueError(f"fold {fold} is missing train/validation/test animals")
        role_sets = [set(train), {validation}, {test}]
        if any(role_sets[i] & role_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f"fold {fold} assigns an animal to multiple split roles")
        for animal in train:
            rows.append({"fold": fold, "animal_id": animal, "role": "train"})
        rows.append({"fold": fold, "animal_id": validation, "role": "validation"})
        rows.append({"fold": fold, "animal_id": test, "role": "test"})
    design = pd.DataFrame(rows)
    if design.empty:
        raise ValueError("no completed fold is represented in queue fold definitions")
    return design.sort_values(["fold", "animal_id"], key=lambda col: col.map(_natural_id_key) if col.name == "animal_id" else col, ignore_index=True)


def _build_animal_scale(per_animal: pd.DataFrame) -> pd.DataFrame:
    columns = ["n_sessions", "n_windows", "positive_windows", "duration_hours"]
    rows: list[dict[str, Any]] = []
    for animal, group in per_animal.groupby("test_animal", sort=False):
        for column in columns:
            values = group[column].dropna().to_numpy(dtype=float)
            if values.size and not np.allclose(values, values[0], rtol=0, atol=1e-9):
                raise ValueError(
                    f"data scale differs between models for animal {animal}: {column}"
                )
        first = group.iloc[0]
        n_windows = int(first["n_windows"])
        positive = int(first["positive_windows"])
        rows.append(
            {
                "animal_id": str(animal),
                "n_sessions": int(first["n_sessions"]),
                "n_windows": n_windows,
                "positive_windows": positive,
                "nonseizure_windows": n_windows - positive,
                "seizure_prevalence": positive / n_windows if n_windows else np.nan,
                "duration_hours": float(first["duration_hours"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "animal_id", key=lambda col: col.map(_natural_id_key), ignore_index=True
    )


def _build_fold_scale(
    split_design: pd.DataFrame, animal_scale: pd.DataFrame
) -> pd.DataFrame:
    merged = split_design.merge(animal_scale, on="animal_id", how="left", validate="many_to_one")
    numeric = [
        "n_sessions",
        "n_windows",
        "positive_windows",
        "nonseizure_windows",
        "duration_hours",
    ]
    grouped = (
        merged.groupby(["fold", "role"], sort=False, dropna=False)[numeric]
        .sum(min_count=1)
        .reset_index()
    )
    grouped["animals_with_scale"] = (
        merged.groupby(["fold", "role"], sort=False)["n_windows"].count().to_numpy()
    )
    grouped["animals_in_role"] = (
        merged.groupby(["fold", "role"], sort=False)["animal_id"].count().to_numpy()
    )
    grouped["scale_complete"] = grouped["animals_with_scale"].eq(grouped["animals_in_role"])
    return grouped.sort_values(["fold", "role"], ignore_index=True)


def _load_completed_histories(
    completed: pd.DataFrame,
) -> tuple[pd.DataFrame, list[Path]]:
    parts: list[pd.DataFrame] = []
    paths: list[Path] = []
    for row in completed.sort_values(["model", "fold"]).itertuples(index=False):
        run_dir = Path(str(row.run_dir)).resolve()
        history_path = run_dir / "history.csv"
        if not history_path.is_file():
            raise FileNotFoundError(
                f"completed fold is missing history.csv: {row.model} fold {row.fold}"
            )
        history = _read_csv(history_path)
        _require_columns(history, ["epoch", "train_loss", "val_loss"], history_path)
        _coerce_finite_numeric(history, ["epoch", "train_loss", "val_loss"], history_path)
        history["epoch"] = history["epoch"].astype(int)
        if history["epoch"].duplicated().any() or not history["epoch"].is_monotonic_increasing:
            raise ValueError(f"epochs must be unique and increasing in {history_path}")
        if (history[["train_loss", "val_loss"]] <= 0).any().any():
            raise ValueError(f"loss values must be positive in {history_path}")
        best_index = history["val_loss"].idxmin()
        history["model"] = str(row.model)
        history["fold"] = int(row.fold)
        history["test_animal"] = str(row.test_animal)
        history["validation_animal"] = str(row.validation_animal)
        history["is_best_epoch"] = False
        history.loc[best_index, "is_best_epoch"] = True
        history["history_path"] = str(history_path)
        parts.append(history)
        paths.append(history_path)
    combined = pd.concat(parts, ignore_index=True)
    columns = [
        "model",
        "fold",
        "test_animal",
        "validation_animal",
        "epoch",
        "train_loss",
        "val_loss",
        "is_best_epoch",
        "history_path",
    ]
    return combined[columns].sort_values(["model", "fold", "epoch"], ignore_index=True), paths


def _render_captions(data: EpilepsySciFigureData) -> str:
    assignment_rows: list[str] = []
    for fold in data.folds:
        fold_rows = data.split_design.loc[data.split_design["fold"].eq(fold)]
        train = ", ".join(
            sorted(
                fold_rows.loc[fold_rows["role"].eq("train"), "animal_id"].astype(str),
                key=_natural_id_key,
            )
        )
        validation = fold_rows.loc[
            fold_rows["role"].eq("validation"), "animal_id"
        ].iloc[0]
        test = fold_rows.loc[fold_rows["role"].eq("test"), "animal_id"].iloc[0]
        assignment_rows.append(f"| {fold} | {train} | {validation} | {test} |")
    completed = data.inventory.groupby("model")["fold"].nunique().to_dict()
    completion_text = ", ".join(
        f"{_model_label(model)} {completed[model]} folds" for model in data.models
    )
    return "\n".join(
        [
            "# Mouse epilepsy LOSO figure captions / 小鼠癫痫 LOSO 图注",
            "",
            "## Fixed data partition / 固定数据划分",
            "",
            "| Fold | Training animals | Validation animal | Test animal |",
            "|---:|---|---|---|",
            *assignment_rows,
            "",
            f"Only launcher-completed runs are included: {completion_text}. All recordings were acquired on 2026-07-01 after acute drug administration; these results provide same-day held-out-animal evidence and no cross-date evidence.",
            "",
            f"仅纳入训练队列中状态为 completed 且产物完整的结果：{completion_text}。所有记录均来自 2026-07-01 急性给药后同一天，因此只能解释为同日条件下的留一动物泛化，不能声称跨日期泛化。",
            "",
            "## Figure 1",
            "",
            "**English.** Fixed animal-level LOSO allocation and data scale. (A) Each fold trains on four animals, selects the checkpoint and probability threshold using one validation animal, and evaluates once on one untouched test animal. Color and cell abbreviations redundantly encode train (Tr), validation (Va), and test (Te). (B) Natural, unrebalanced seizure and non-seizure window counts for each animal. (C) Window counts entering each split role. Recording segments from the same animal remain together.",
            "",
            "**中文。** 固定动物级 LOSO 划分及数据规模。A，每折用 4 只动物训练、1 只动物验证并选择模型与阈值、1 只完全未见动物测试；颜色与 Tr/Va/Te 缩写同时编码三种角色。B，每只动物未经重平衡的发作/非发作窗口数。C，每折各角色的窗口规模。同一动物的全部记录片段始终处于同一划分中。",
            "",
            "## Figure 2",
            "",
            "**English.** Accuracy, sensitivity, specificity, and F1 score for each held-out animal. Small symbols are individual animals; gray lines connect the same animal across models. Large symbols and error bars show the unweighted mean ± sample standard deviation across six animals (n=6, SD with ddof=1). Each test animal is the statistical unit; overlapping windows are not treated as independent biological replicates.",
            "",
            "**中文。** 每只留出测试动物的 Accuracy、Sensitivity、Specificity 与 F1。小点表示单只动物，灰线连接同一动物在两个模型上的结果；大点和误差线表示 6 只动物的非加权均值 ± 样本标准差（n=6，ddof=1）。统计单位是动物，不把重叠窗口视为独立生物学重复。",
            "",
            "## Figure 3",
            "",
            "**English.** Pooled out-of-fold confusion matrices across all held-out test windows. Left panels report raw counts; right panels normalize each true-class row. Each fold uses its own validation-selected threshold before predictions are pooled. Test distributions are not rebalanced. Pooled matrices describe window-level operating behavior and do not replace animal-level uncertainty in Figures 2, 5, and 6.",
            "",
            "**中文。** 汇总全部留出测试窗口的 pooled 混淆矩阵。左列为原始计数，右列按真实类别逐行归一化。每折先使用各自验证动物选择的阈值，再合并预测；测试集未做重平衡。该图描述窗口级运行表现，不能替代图 2、5、6 的动物级不确定性。",
            "",
            "## Figure 4",
            "",
            "**English.** Weighted cross-entropy loss for training and validation sets in every completed model-fold run (12 panels). The y-axis is logarithmic. A dotted vertical line and star mark the epoch with minimum validation loss, which defines checkpoint selection. Training uses class-aware weighted sampling and weighted cross-entropy; validation retains its natural distribution.",
            "",
            "**中文。** 两个模型全部完成折次的训练集与验证集加权交叉熵损失（共 12 个小面板），纵轴使用对数尺度。虚线和星号标出验证损失最低的 epoch，并据此选择 checkpoint。训练阶段使用类别感知加权采样与加权交叉熵，验证集保持自然分布。",
            "",
            "## Figure 5",
            "",
            "**English.** Validation-selected probability threshold, event sensitivity, and false alarms per hour for each held-out animal. Symbols and error bars follow Figure 2 (unweighted mean ± sample SD, n=6 animals). Event metrics depend on the prespecified temporal event-merging procedure and should be interpreted jointly with false-alarm burden.",
            "",
            "**中文。** 每只留出动物的验证集选定阈值、事件敏感度和每小时误报数。点与误差线定义同图 2（6 只动物的非加权均值 ± 样本标准差）。事件指标依赖预先设定的时间合并规则，需与误报负担共同解释。",
            "",
            "## Figure 6",
            "",
            "**English.** AUROC, AUPRC, balanced accuracy, and MCC across held-out animals. Individual animals and paired lines expose between-animal heterogeneity; large symbols show the unweighted mean ± sample SD (n=6). Horizontal reference lines indicate 0.5 for AUROC/AUPRC/balanced accuracy and 0 for MCC. No inferential significance test is claimed for this six-animal exploratory comparison.",
            "",
            "**中文。** 每只留出动物的 AUROC、AUPRC、Balanced Accuracy 与 MCC。单动物点和配对线直接展示动物间异质性，大点为 6 只动物的非加权均值 ± 样本标准差。参考线分别为 AUROC/AUPRC/BA 的 0.5 和 MCC 的 0。本研究仅 6 只动物，因此该模型比较不进行推断性显著性声明。",
            "",
        ]
    )


def _build_manifest(
    *,
    data: EpilepsySciFigureData,
    destination: Path,
    figure_outputs: Mapping[str, Sequence[Path]],
    source_paths: Mapping[str, Path],
    captions_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs_root": str(data.runs_root),
        "clinical_report_dir": str(data.clinical_report_dir),
        "output_dir": str(destination),
        "protocol": {
            "design": "animal-level LOSO",
            "collection_date": "2026-07-01",
            "cohort": "acute_post_drug_single_day_20260701",
            "same_day_held_out_animal_evidence": True,
            "cross_date_evidence": False,
            "statistical_unit_for_error_bars": "held-out animal",
            "error_bar": "unweighted mean +/- sample SD (ddof=1)",
            "overlapping_windows_are_independent_replicates": False,
            "test_distribution_rebalanced": False,
        },
        "completed_runs": {
            model: int(count)
            for model, count in data.inventory.groupby("model")["fold"].nunique().items()
        },
        "models": [
            {"id": model, "display_name": _model_label(model)} for model in data.models
        ],
        "animals": data.animals,
        "folds": data.folds,
        "style": {
            "palette": "Okabe-Ito",
            "redundant_encodings": ["color", "marker", "line style", "hatch", "cell abbreviation"],
            "continuous_colormap": "cividis",
            "axes_language": "English",
            "raster_dpi": 300,
            "formats": ["png", "pdf", "svg"],
        },
        "inputs": [_file_record(path, data.runs_root) for path in data.input_paths],
        "figures": {
            stem: [_file_record(path, destination) for path in paths]
            for stem, paths in figure_outputs.items()
        },
        "source_data": {
            name: _file_record(path, destination) for name, path in source_paths.items()
        },
        "captions": _file_record(captions_path, destination),
    }


def _export_figure(figure: mpl.figure.Figure, base_path: Path) -> list[Path]:
    paths: list[Path] = []
    for suffix in (".png", ".pdf", ".svg"):
        path = base_path.with_suffix(suffix)
        kwargs: dict[str, Any] = {
            "bbox_inches": "tight",
            "facecolor": "white",
        }
        if suffix == ".png":
            kwargs["dpi"] = 300
        figure.savefig(path, **kwargs)
        paths.append(path)
    return paths


def _publication_rc(*, small_multiples: bool = False) -> dict[str, Any]:
    base_size = 6.0 if small_multiples else 7.5
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": base_size,
        "axes.titlesize": base_size + 0.5,
        "axes.labelsize": base_size + 0.5,
        "xtick.labelsize": base_size - 0.5,
        "ytick.labelsize": base_size - 0.5,
        "legend.fontsize": base_size - 0.5,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.2,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }


def _panel_labels(axes: Sequence[mpl.axes.Axes]) -> None:
    for index, ax in enumerate(axes):
        label = ascii_uppercase[index] if index < len(ascii_uppercase) else str(index + 1)
        ax.text(
            -0.15,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
            ha="left",
        )


def _despine(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing required figure input: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def _reject_duplicate_keys(frame: pd.DataFrame, source: Path) -> None:
    duplicated = frame.duplicated(_KEY_COLUMNS, keep=False)
    if duplicated.any():
        rows = frame.loc[duplicated, _KEY_COLUMNS].to_dict(orient="records")
        raise ValueError(f"duplicate model/fold/test-animal rows in {source}: {rows}")


def _coerce_finite_numeric(
    frame: pd.DataFrame, columns: Iterable[str], source: Path
) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{source} contains non-finite values in {column}")


def _filter_one_row_per_model(
    frame: pd.DataFrame, models: Iterable[str], source: Path
) -> pd.DataFrame:
    expected = {str(model) for model in models}
    filtered = frame.loc[frame["model"].astype(str).isin(expected)].copy()
    counts = filtered["model"].astype(str).value_counts().to_dict()
    invalid = {model: counts.get(model, 0) for model in expected if counts.get(model, 0) != 1}
    if invalid:
        raise ValueError(f"{source} must contain one row per completed model: {invalid}")
    return filtered.sort_values("model", ignore_index=True)


def _ordered_models(models: Iterable[str]) -> list[str]:
    order = {model: index for index, model in enumerate(MODEL_LABELS)}
    return sorted({str(model) for model in models}, key=lambda value: (order.get(value, 999), value))


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " "))


def _natural_id_key(value: Any) -> tuple[str, int, str]:
    text = str(value)
    prefix = text.rstrip("0123456789")
    suffix = text[len(prefix) :]
    return prefix, int(suffix) if suffix else -1, text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, relative_root: Path) -> dict[str, Any]:
    try:
        relative = str(path.resolve().relative_to(relative_root.resolve()))
    except ValueError:
        relative = None
    return {
        "path": str(path.resolve()),
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
