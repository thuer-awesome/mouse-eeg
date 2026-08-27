"""Publication-grade figures for the pooled-mouse binary seizure study.

The module is read-only with respect to data and training artifacts. It treats
independent event complexes/background blocks as the uncertainty unit, keeps
window-level and mouse-macro summaries separate, and never reports continuous
monitoring metrics from the capped-window evaluation view.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.signal import periodogram
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

__all__ = [
    "MixedBinaryFigureData",
    "generate_mixed_binary_visualizations",
    "load_mixed_binary_figure_data",
]


PRIMARY_MODELS = (
    "eegnet82",
    "eegwavenet",
    "physio_no_aux",
    "physio_stats_aux",
)
SENSITIVITY_MODELS = (
    "physio_no_aux_mouse_balanced",
    "physio_stats_aux_mouse_balanced",
)
ALL_MODELS = PRIMARY_MODELS + SENSITIVITY_MODELS
ANIMALS = ("E50", "E76", "E77", "E79", "E81", "E82")
SPLITS = ("train", "val", "test")
KEY_COLUMNS = (
    "animal_id",
    "session_id",
    "group_id",
    "start_time_ms",
    "end_time_ms",
    "label",
    "split",
)

MODEL_LABELS = {
    "eegnet82": "EEGNet-8,2",
    "eegwavenet": "EEGWaveNet",
    "physio_no_aux": "Physio (no aux)",
    "physio_stats_aux": "Physio (+ stats aux)",
    "physio_no_aux_mouse_balanced": "Physio (mouse-balanced)",
    "physio_stats_aux_mouse_balanced": "Physio (aux + mouse-balanced)",
}
MODEL_SHORT_LABELS = {
    "eegnet82": "EEGNet",
    "eegwavenet": "EEGWaveNet",
    "physio_no_aux": "Physio",
    "physio_stats_aux": "Physio + aux",
    "physio_no_aux_mouse_balanced": "Physio + mouse weight",
    "physio_stats_aux_mouse_balanced": "Aux + mouse weight",
}
MODEL_COLORS = {
    "eegnet82": "#0072B2",
    "eegwavenet": "#E69F00",
    "physio_no_aux": "#009E73",
    "physio_stats_aux": "#CC79A7",
    "physio_no_aux_mouse_balanced": "#006B4F",
    "physio_stats_aux_mouse_balanced": "#8E4F78",
}
MODEL_MARKERS = {
    "eegnet82": "o",
    "eegwavenet": "s",
    "physio_no_aux": "^",
    "physio_stats_aux": "D",
    "physio_no_aux_mouse_balanced": "v",
    "physio_stats_aux_mouse_balanced": "P",
}
MODEL_LINESTYLES = {
    "eegnet82": "-",
    "eegwavenet": "--",
    "physio_no_aux": "-.",
    "physio_stats_aux": ":",
    "physio_no_aux_mouse_balanced": "--",
    "physio_stats_aux_mouse_balanced": ":",
}

NEUTRAL_DARK = "#2B2B2B"
NEUTRAL_MID = "#777777"
NEUTRAL_LIGHT = "#D9D9D9"
PANEL_BG = "#FAFAFA"
SEIZURE_COLOR = "#B5523B"
NONSEIZURE_COLOR = "#6D7785"
SPLIT_COLORS = {"train": "#4C78A8", "val": "#A5A5A5", "test": "#E5A44D"}

METRIC_LABELS = {
    "auprc": "AUPRC",
    "auroc": "AUROC",
    "balanced_accuracy": "Balanced accuracy",
    "mcc": "MCC",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "f1": "F1 score",
    "macro_mouse_auprc": "Macro-mouse AUPRC",
    "macro_mouse_balanced_accuracy": "Macro-mouse BA",
}
BOOTSTRAP_METRICS = (
    "auprc",
    "auroc",
    "balanced_accuracy",
    "mcc",
    "macro_mouse_auprc",
    "macro_mouse_balanced_accuracy",
)
CONTRAST_LABELS = {
    "physio_no_aux_minus_eegwavenet": "Physio - EEGWaveNet",
    "physio_stats_aux_minus_physio_no_aux": "Aux - no aux",
    "physio_no_aux_mouse_balanced_minus_physio_no_aux": "Mouse weight - base",
    "physio_stats_aux_mouse_balanced_minus_physio_no_aux_mouse_balanced": (
        "Aux - no aux (mouse weight)"
    ),
}
CONTRAST_COLORS = {
    "physio_no_aux_minus_eegwavenet": "#0072B2",
    "physio_stats_aux_minus_physio_no_aux": "#CC79A7",
    "physio_no_aux_mouse_balanced_minus_physio_no_aux": "#009E73",
    "physio_stats_aux_mouse_balanced_minus_physio_no_aux_mouse_balanced": "#E69F00",
}

FIGURE_SPECS: tuple[tuple[str, str], ...] = (
    ("fig00_study_model_overview", "Study design and physiology-informed model"),
    ("fig01_data_split_qc", "Frozen split and preprocessing quality control"),
    ("fig02_preprocessing_examples", "Representative signal preprocessing"),
    ("fig03_overall_performance", "Overall test performance and operating trade-offs"),
    ("fig04_discrimination_curves", "Discrimination, calibration, and score separation"),
    ("fig05_confusion_matrices", "Validation-threshold test confusion matrices"),
    ("fig06_per_mouse_performance", "Cross-mouse heterogeneity"),
    ("fig07_per_mouse_pr_curves", "Mouse-specific precision-recall curves"),
    ("fig08_training_histories", "Training and checkpoint-selection histories"),
    ("fig09_auxiliary_ablation", "Physiological statistics auxiliary-loss ablation"),
    ("fig10_mouse_balanced_sensitivity", "Mouse-balanced loss sensitivity analysis"),
    ("fig11_multiseed_stability", "Three-seed paired stability analysis"),
    ("fig12_bootstrap_forest", "Independent-group bootstrap summary"),
    ("fig13_results_overview", "Integrated results overview"),
)

MAIN_FIGURE_SPECS: tuple[tuple[str, str], ...] = (
    ("main_fig01_data_model", "Data integrity and physiology-informed model"),
    ("main_fig02_binary_detection", "Binary seizure discrimination on the frozen test set"),
    ("main_fig03_mouse_heterogeneity", "Cross-mouse performance and evidence support"),
    ("main_fig04_design_evidence", "Evidence for model components and loss design"),
)
ALL_FIGURE_SPECS = MAIN_FIGURE_SPECS + FIGURE_SPECS


@dataclass(frozen=True)
class MixedBinaryFigureData:
    """Validated inputs used to generate the complete figure suite."""

    project_root: Path
    run_root: Path
    output_dir: Path
    split_summary: pd.DataFrame
    group_summary: pd.DataFrame
    split_audit: Mapping[str, Any]
    metadata: pd.DataFrame
    quality_summary: pd.DataFrame
    training_baseline: Mapping[str, Any]
    view_manifest: Mapping[str, Any]
    overall: pd.DataFrame
    per_mouse: pd.DataFrame
    bootstrap: Mapping[str, Any]
    predictions: Mapping[str, pd.DataFrame]
    histories: Mapping[str, pd.DataFrame]
    metrics: Mapping[str, Mapping[str, Any]]
    manifests: Mapping[str, Mapping[str, Any]]
    seed_metrics: pd.DataFrame
    seed_per_mouse: pd.DataFrame
    seed_differences: pd.DataFrame
    input_paths: tuple[Path, ...]

    @property
    def test_overall(self) -> pd.DataFrame:
        frame = self.overall.loc[
            self.overall["split"].eq("test")
            & self.overall["model"].isin(PRIMARY_MODELS)
        ].copy()
        return _order_frame(frame, "model", PRIMARY_MODELS)

    @property
    def test_prevalence(self) -> float:
        first = self.test_overall.iloc[0]
        return float(first["positive_windows"] / first["n_windows"])


def load_mixed_binary_figure_data(
    project_root: str | Path,
    output_dir: str | Path | None = None,
) -> MixedBinaryFigureData:
    """Load and validate all completed binary-study artifacts."""

    root = Path(project_root).resolve()
    run_root = root / "runs" / "epilepsy_binary_mixed_v1"
    split_root = root / "data" / "epilepsy_binary_mixed_split_v1"
    filtered_root = root / "data" / "epilepsy_binary_mixed_filtered_v1"
    comparison_root = run_root / "comparison_extended"
    seed_root = root / "runs" / "epilepsy_binary_mixed_physio_multiseed"
    destination = (
        Path(output_dir).resolve() if output_dir else run_root / "figures"
    )

    paths = {
        "split_summary": split_root / "split_summary.csv",
        "group_summary": split_root / "group_summary.csv",
        "split_audit": split_root / "split_audit.json",
        "metadata": filtered_root / "metadata.parquet",
        "quality_summary": filtered_root / "quality_summary.csv",
        "training_baseline": filtered_root / "training_baseline.json",
        "view_manifest": filtered_root / "view_manifest.json",
        "overall": comparison_root / "overall_metrics.csv",
        "per_mouse": comparison_root / "per_mouse_metrics.csv",
        "bootstrap": comparison_root / "paired_group_bootstrap.json",
        "seed_metrics": seed_root / "seed_metrics.csv",
        "seed_per_mouse": seed_root / "seed_per_mouse_metrics.csv",
        "seed_differences": seed_root / "paired_seed_differences.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing figure inputs: {missing}")

    split_summary = pd.read_csv(paths["split_summary"], encoding="utf-8-sig")
    group_summary = pd.read_csv(paths["group_summary"], encoding="utf-8-sig")
    metadata = pd.read_parquet(paths["metadata"])
    quality_summary = pd.read_csv(paths["quality_summary"], encoding="utf-8-sig")
    overall = pd.read_csv(paths["overall"], encoding="utf-8-sig")
    per_mouse = pd.read_csv(paths["per_mouse"], encoding="utf-8-sig")
    seed_metrics = pd.read_csv(paths["seed_metrics"], encoding="utf-8-sig")
    seed_per_mouse = pd.read_csv(paths["seed_per_mouse"], encoding="utf-8-sig")
    seed_differences = pd.read_csv(
        paths["seed_differences"], encoding="utf-8-sig"
    )
    split_audit = _read_json(paths["split_audit"])
    training_baseline = _read_json(paths["training_baseline"])
    view_manifest = _read_json(paths["view_manifest"])
    bootstrap = _read_json(paths["bootstrap"])

    _require_columns(
        split_summary,
        (
            "split",
            "animal_id",
            "windows",
            "seizure_windows",
            "nonseizure_windows",
            "seizure_groups",
            "background_groups",
        ),
        paths["split_summary"],
    )
    _require_columns(
        group_summary,
        ("group_id", "animal_id", "session_id", "label", "split", "windows"),
        paths["group_summary"],
    )
    _require_columns(
        metadata,
        (
            "animal_id",
            "label",
            "split",
            "cache_dir",
            "cache_local_index",
            "filtered_cache_dir",
            "filtered_local_index",
            "ch7_valid",
            "ch8_valid",
            "maximum_channel_jump_z",
        ),
        paths["metadata"],
    )
    _require_columns(
        overall,
        (
            "model",
            "split",
            "parameters",
            "best_epoch",
            "threshold",
            "auprc",
            "auroc",
            "balanced_accuracy",
            "mcc",
            "positive_group_sensitivity",
            "false_positive_background_group_fraction",
            "macro_mouse_auprc",
            "macro_mouse_balanced_accuracy",
        ),
        paths["overall"],
    )
    _require_columns(
        per_mouse,
        (
            "model",
            "split",
            "animal_id",
            "positive_windows",
            "sensitivity",
            "balanced_accuracy",
            "auprc",
        ),
        paths["per_mouse"],
    )
    if set(PRIMARY_MODELS) - set(overall["model"].astype(str)):
        raise ValueError("overall_metrics.csv does not contain all primary models")
    if set(ALL_MODELS) - set(per_mouse["model"].astype(str)):
        raise ValueError("per_mouse_metrics.csv does not contain all required models")
    if int(split_audit.get("schema_version", -1)) != 1:
        raise ValueError("unsupported split audit schema")
    if int(view_manifest.get("schema_version", -1)) != 1:
        raise ValueError("unsupported filtered-view schema")
    if int(bootstrap.get("schema_version", -1)) != 1:
        raise ValueError("unsupported bootstrap schema")
    if bool(split_audit.get("subtype_used_for_split", True)):
        raise ValueError("the frozen binary split unexpectedly used seizure subtype")
    if not bool(split_audit.get("event_groups_disjoint", False)):
        raise ValueError("independent groups are not disjoint across splits")
    if int(split_audit.get("overlapping_raw_windows_cross_splits", -1)) != 0:
        raise ValueError("overlapping raw windows cross split boundaries")

    predictions: dict[str, pd.DataFrame] = {}
    histories: dict[str, pd.DataFrame] = {}
    metrics: dict[str, Mapping[str, Any]] = {}
    manifests: dict[str, Mapping[str, Any]] = {}
    run_input_paths: list[Path] = []
    aligned_keys: pd.DataFrame | None = None
    for model in ALL_MODELS:
        model_root = run_root / model
        required = {
            "completed": model_root / "completed.json",
            "predictions": model_root / "predictions.parquet",
            "history": model_root / "history.csv",
            "metrics": model_root / "metrics.json",
            "manifest": model_root / "run_manifest.json",
        }
        absent = [str(path) for path in required.values() if not path.is_file()]
        if absent:
            raise FileNotFoundError(f"incomplete model run {model}: {absent}")
        frame = pd.read_parquet(required["predictions"])
        _validate_predictions(frame, required["predictions"])
        keys = frame.loc[:, KEY_COLUMNS].reset_index(drop=True)
        if aligned_keys is None:
            aligned_keys = keys
        elif not aligned_keys.equals(keys):
            raise ValueError(f"prediction rows are not aligned for {model}")
        history = pd.read_csv(required["history"], encoding="utf-8-sig")
        _require_columns(
            history,
            (
                "epoch",
                "train_loss",
                "train_classification_loss",
                "train_statistics_loss",
                "validation_loss",
                "validation_classification_loss",
                "validation_statistics_loss",
                "validation_auprc",
                "validation_auroc",
            ),
            required["history"],
        )
        prediction_metrics = _read_json(required["metrics"])
        manifest = _read_json(required["manifest"])
        if int(prediction_metrics.get("schema_version", -1)) != 1:
            raise ValueError(f"unsupported metrics schema for {model}")
        if bool(prediction_metrics.get("continuous_monitoring_metrics_valid", True)):
            raise ValueError(f"capped-window run {model} claims continuous metrics")
        predictions[model] = frame
        histories[model] = history
        metrics[model] = prediction_metrics
        manifests[model] = manifest
        run_input_paths.extend(required.values())

    _validate_metric_reproduction(predictions, metrics)
    return MixedBinaryFigureData(
        project_root=root,
        run_root=run_root,
        output_dir=destination,
        split_summary=split_summary,
        group_summary=group_summary,
        split_audit=split_audit,
        metadata=metadata,
        quality_summary=quality_summary,
        training_baseline=training_baseline,
        view_manifest=view_manifest,
        overall=overall,
        per_mouse=per_mouse,
        bootstrap=bootstrap,
        predictions=predictions,
        histories=histories,
        metrics=metrics,
        manifests=manifests,
        seed_metrics=seed_metrics,
        seed_per_mouse=seed_per_mouse,
        seed_differences=seed_differences,
        input_paths=tuple(paths.values()) + tuple(run_input_paths),
    )


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], source: Path
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def _validate_predictions(frame: pd.DataFrame, source: Path) -> None:
    _require_columns(frame, (*KEY_COLUMNS, "seizure_probability"), source)
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"duplicate prediction keys in {source}")
    probabilities = pd.to_numeric(frame["seizure_probability"], errors="raise")
    if not np.isfinite(probabilities.to_numpy(float)).all():
        raise ValueError(f"non-finite probabilities in {source}")
    if not probabilities.between(0.0, 1.0).all():
        raise ValueError(f"probabilities outside [0, 1] in {source}")
    if set(frame["split"].astype(str)) != {"val", "test"}:
        raise ValueError(f"prediction splits are not val/test in {source}")


def _validate_metric_reproduction(
    predictions: Mapping[str, pd.DataFrame],
    metrics: Mapping[str, Mapping[str, Any]],
) -> None:
    for model in PRIMARY_MODELS:
        test = predictions[model].loc[predictions[model]["split"].eq("test")]
        labels = test["label"].to_numpy(np.int8)
        scores = test["seizure_probability"].to_numpy(float)
        expected = metrics[model]["test"]["window"]
        checks = {
            "auprc": average_precision_score(labels, scores),
            "auroc": roc_auc_score(labels, scores),
        }
        for name, value in checks.items():
            if not math.isclose(float(expected[name]), float(value), abs_tol=1e-12):
                raise ValueError(f"{model} {name} does not reproduce from predictions")


def _order_frame(
    frame: pd.DataFrame, column: str, order: Sequence[str]
) -> pd.DataFrame:
    ranks = {value: index for index, value in enumerate(order)}
    output = frame.copy()
    output["__order"] = output[column].astype(str).map(ranks)
    return output.sort_values("__order", kind="stable").drop(columns="__order")


def _publication_rc(*, compact: bool = False) -> dict[str, Any]:
    base = 6.2 if compact else 7.2
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": base,
        "axes.titlesize": base + 0.5,
        "axes.labelsize": base + 0.3,
        "xtick.labelsize": base - 0.4,
        "ytick.labelsize": base - 0.4,
        "legend.fontsize": base - 0.5,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.25,
        "lines.markersize": 4.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }


def _panel_labels(axes: Sequence[mpl.axes.Axes]) -> None:
    for index, ax in enumerate(axes):
        label = chr(ord("a") + index) if index < 26 else str(index + 1)
        ax.text(
            -0.12,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            ha="left",
            va="top",
            clip_on=False,
        )


def _style_axis(ax: mpl.axes.Axes, *, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=2.5)
    if grid is not None:
        ax.grid(axis=grid, color="#E6E6E6", linewidth=0.55, zorder=0)


def _model_legend(models: Sequence[str] = PRIMARY_MODELS) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linestyle=MODEL_LINESTYLES[model],
            label=MODEL_LABELS[model],
            linewidth=1.4,
            markersize=4.2,
        )
        for model in models
    ]


def _export_figure(
    figure: mpl.figure.Figure, base_path: Path, dpi: int
) -> list[Path]:
    outputs: list[Path] = []
    export_rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with mpl.rc_context(export_rc):
        for suffix in (".png", ".pdf", ".svg"):
            path = base_path.with_suffix(suffix)
            kwargs: dict[str, Any] = {
                "bbox_inches": "tight",
                "pad_inches": 0.04,
                "facecolor": "white",
            }
            if suffix == ".png":
                kwargs["dpi"] = max(int(dpi), 300)
            figure.savefig(path, **kwargs)
            outputs.append(path)
    plt.close(figure)
    return outputs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_line_power(signal: np.ndarray, sampling_rate_hz: float) -> float:
    values = np.asarray(signal, dtype=np.float64)
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / sampling_rate_hz)
    power = np.abs(np.fft.rfft(values - np.mean(values))) ** 2
    line = (frequencies >= 48.0) & (frequencies <= 52.0)
    total = (frequencies >= 0.5) & (frequencies <= 70.0)
    denominator = float(power[total].sum())
    return float(power[line].sum() / denominator) if denominator > 0.0 else 0.0


def _compute_line_noise_qc(metadata: pd.DataFrame) -> pd.DataFrame:
    selected = metadata.iloc[
        np.linspace(0, len(metadata) - 1, 512, dtype=np.int64)
    ]
    rows: list[dict[str, Any]] = []
    sample_index = 0
    for row in selected.itertuples(index=False):
        raw = np.load(Path(row.cache_dir) / "signals.npy", mmap_mode="r")[
            int(row.cache_local_index)
        ]
        filtered = np.load(
            Path(row.filtered_cache_dir) / "signals.npy", mmap_mode="r"
        )[int(row.filtered_local_index)]
        for channel_index, (channel_name, valid) in enumerate(
            (("CH7", row.ch7_valid), ("CH8", row.ch8_valid))
        ):
            if not bool(valid):
                continue
            rows.append(
                {
                    "sample_index": sample_index,
                    "animal_id": str(row.animal_id),
                    "channel": channel_name,
                    "sampling_rate_hz": float(row.sampling_rate_hz),
                    "raw_relative_power_48_52": _relative_line_power(
                        raw[channel_index], float(row.sampling_rate_hz)
                    ),
                    "filtered_relative_power_48_52": _relative_line_power(
                        filtered[channel_index], float(row.sampling_rate_hz)
                    ),
                }
            )
            sample_index += 1
    output = pd.DataFrame(rows)
    if len(output) != 958:
        raise ValueError(f"expected 958 valid sampled channels, received {len(output)}")
    return output


def _baseline_lookup(
    training_baseline: Mapping[str, Any], animal_id: str
) -> dict[str, tuple[float, float, bool]]:
    result: dict[str, tuple[float, float, bool]] = {}
    for item in training_baseline["channels"]:
        if str(item["animal_id"]) == animal_id:
            result[str(item["channel_name"])] = (
                float(item["center"]),
                float(item["scale"]),
                bool(item["valid"]),
            )
    return result


def _representative_signal_rows(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for animal, label in (("E50", 0), ("E50", 1), ("E77", 1)):
        subset = metadata.loc[
            metadata["animal_id"].eq(animal)
            & metadata["split"].eq("test")
            & metadata["label"].eq(label)
        ].copy()
        median = float(subset["maximum_channel_jump_z"].median())
        index = (subset["maximum_channel_jump_z"] - median).abs().idxmin()
        rows.append(subset.loc[index])
    return pd.DataFrame(rows).reset_index(drop=True)


def _load_signal_pair(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(
        np.load(Path(row["cache_dir"]) / "signals.npy", mmap_mode="r")[
            int(row["cache_local_index"])
        ],
        dtype=np.float64,
    )
    filtered = np.asarray(
        np.load(
            Path(row["filtered_cache_dir"]) / "signals.npy", mmap_mode="r"
        )[int(row["filtered_local_index"])],
        dtype=np.float64,
    )
    return raw, filtered


def _normalized_signal_pair(
    row: pd.Series, training_baseline: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    raw, filtered = _load_signal_pair(row)
    lookup = _baseline_lookup(training_baseline, str(row["animal_id"]))
    raw_scaled = np.zeros_like(raw)
    filtered_scaled = np.zeros_like(filtered)
    for channel_index, channel in enumerate(("CH7", "CH8")):
        center, scale, valid = lookup[channel]
        if valid:
            raw_scaled[channel_index] = (
                raw[channel_index] - np.median(raw[channel_index])
            ) / scale
            filtered_scaled[channel_index] = (
                filtered[channel_index] - center
            ) / scale
    return raw_scaled, filtered_scaled


def _flatten_bootstrap(bootstrap: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, metrics in bootstrap["performance"].items():
        for metric, values in metrics.items():
            rows.append(
                {
                    "kind": "performance",
                    "comparison": model,
                    "metric": metric,
                    **values,
                    "bootstrap_probability_greater_than_zero": np.nan,
                }
            )
    for contrast, metrics in bootstrap["contrasts"].items():
        for metric, values in metrics.items():
            rows.append(
                {
                    "kind": "contrast",
                    "comparison": contrast,
                    "metric": metric,
                    **values,
                }
            )
    return pd.DataFrame(rows)


def _curve_source_data(data: MixedBinaryFigureData) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for model in PRIMARY_MODELS:
        test = data.predictions[model].loc[
            data.predictions[model]["split"].eq("test")
        ]
        labels = test["label"].to_numpy(np.int8)
        scores = test["seizure_probability"].to_numpy(float)
        precision, recall, pr_threshold = precision_recall_curve(labels, scores)
        rows.append(
            pd.DataFrame(
                {
                    "model": model,
                    "curve": "precision_recall",
                    "x": recall,
                    "y": precision,
                    "threshold": np.r_[pr_threshold, np.nan],
                    "animal_id": "pooled",
                }
            )
        )
        false_positive_rate, true_positive_rate, roc_threshold = roc_curve(
            labels, scores
        )
        rows.append(
            pd.DataFrame(
                {
                    "model": model,
                    "curve": "roc",
                    "x": false_positive_rate,
                    "y": true_positive_rate,
                    "threshold": roc_threshold,
                    "animal_id": "pooled",
                }
            )
        )
        calibrated, predicted = calibration_curve(
            labels, scores, n_bins=12, strategy="quantile"
        )
        rows.append(
            pd.DataFrame(
                {
                    "model": model,
                    "curve": "calibration",
                    "x": predicted,
                    "y": calibrated,
                    "threshold": np.nan,
                    "animal_id": "pooled",
                }
            )
        )
        for animal in ANIMALS:
            mouse = test.loc[test["animal_id"].eq(animal)]
            mouse_labels = mouse["label"].to_numpy(np.int8)
            mouse_scores = mouse["seizure_probability"].to_numpy(float)
            mouse_precision, mouse_recall, mouse_threshold = precision_recall_curve(
                mouse_labels, mouse_scores
            )
            rows.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "curve": "precision_recall",
                        "x": mouse_recall,
                        "y": mouse_precision,
                        "threshold": np.r_[mouse_threshold, np.nan],
                        "animal_id": animal,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _confusion_source_data(data: MixedBinaryFigureData) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in PRIMARY_MODELS:
        values = data.metrics[model]["test"]["window"]
        matrix = np.asarray(
            [[values["tn"], values["fp"]], [values["fn"], values["tp"]]],
            dtype=int,
        )
        for true_label in range(2):
            denominator = int(matrix[true_label].sum())
            for predicted_label in range(2):
                rows.append(
                    {
                        "model": model,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": int(matrix[true_label, predicted_label]),
                        "row_fraction": float(
                            matrix[true_label, predicted_label] / denominator
                        ),
                        "threshold": float(data.metrics[model]["threshold"]),
                    }
                )
    return pd.DataFrame(rows)


def _representative_source_data(
    data: MixedBinaryFigureData, representative_rows: pd.DataFrame
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for item in representative_rows.itertuples(index=False):
        row = pd.Series(item._asdict())
        raw, filtered = _normalized_signal_pair(row, data.training_baseline)
        time = np.arange(raw.shape[-1], dtype=float) / float(row["sampling_rate_hz"])
        for channel_index, channel in enumerate(("CH7", "CH8")):
            rows.append(
                pd.DataFrame(
                    {
                        "animal_id": row["animal_id"],
                        "label": int(row["label"]),
                        "group_id": row["group_id"],
                        "start_time_ms": float(row["start_time_ms"]),
                        "channel": channel,
                        "time_seconds": time,
                        "raw_centered_baseline_scale_au": raw[channel_index],
                        "filtered_baseline_normalized_au": filtered[channel_index],
                        "channel_valid": bool(row[channel.lower() + "_valid"]),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _draw_box(
    ax: mpl.axes.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = "#4A4A4A",
    fontsize: float = 6.6,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=0.75,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=NEUTRAL_DARK,
    )
    return patch


def _draw_arrow(
    ax: mpl.axes.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#666666",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def _figure_study_model_overview(data: MixedBinaryFigureData) -> mpl.figure.Figure:
    with mpl.rc_context(_publication_rc()):
        figure = plt.figure(figsize=(7.4, 5.15), constrained_layout=True)
        grid = figure.add_gridspec(2, 1, height_ratios=(0.82, 1.3), hspace=0.13)
        design = figure.add_subplot(grid[0])
        architecture = figure.add_subplot(grid[1])
        for ax in (design, architecture):
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

        design.text(0.0, 1.03, "a", fontsize=8.5, fontweight="bold", va="top")
        architecture.text(
            0.0, 1.03, "b", fontsize=8.5, fontweight="bold", va="top"
        )

        design_boxes = (
            (0.01, "6 mice\n8 sessions", "#E7EDF4"),
            (0.18, "Binary labels\nseizure / non-seizure", "#F3E8E4"),
            (0.38, "Independent groups\nevent complexes / blocks", "#E8F1ED"),
            (0.59, "70 / 15 / 15 split\nno shared group or overlap", "#F4EEDC"),
            (0.79, "Frozen test view\n25,376 windows; 502 groups", "#E9E5F0"),
        )
        widths = (0.14, 0.17, 0.18, 0.17, 0.2)
        for index, (x, text, color) in enumerate(design_boxes):
            _draw_box(design, (x, 0.38), widths[index], 0.36, text, facecolor=color)
            if index < len(design_boxes) - 1:
                _draw_arrow(
                    design,
                    (x + widths[index], 0.56),
                    (design_boxes[index + 1][0], 0.56),
                )
        design.text(
            0.5,
            0.16,
            "Seen-mouse, independent-group generalization; seizure morphology subtypes excluded from the main label and split",
            ha="center",
            va="center",
            fontsize=6.4,
            color=NEUTRAL_MID,
        )

        _draw_box(
            architecture,
            (0.01, 0.39),
            0.13,
            0.23,
            "2-channel EEG\n4 s x 250 Hz\n+ channel mask",
            facecolor="#E7EDF4",
        )
        _draw_box(
            architecture,
            (0.2, 0.65),
            0.17,
            0.18,
            "Temporal branch\ndepthwise kernels\n9 / 31 / 101",
            facecolor="#DDEEE8",
        )
        _draw_box(
            architecture,
            (0.2, 0.18),
            0.17,
            0.18,
            "Spectral branch\ndifferentiable STFT\n0.5-70 Hz",
            facecolor="#F1E7D6",
        )
        _draw_arrow(architecture, (0.14, 0.505), (0.2, 0.74))
        _draw_arrow(architecture, (0.14, 0.505), (0.2, 0.27))
        _draw_box(
            architecture,
            (0.44, 0.39),
            0.15,
            0.23,
            "Bidirectional\ntime-frequency\ncross-attention",
            facecolor="#E9E5F0",
        )
        _draw_arrow(architecture, (0.37, 0.74), (0.44, 0.54))
        _draw_arrow(architecture, (0.37, 0.27), (0.44, 0.47))
        _draw_box(
            architecture,
            (0.64, 0.39),
            0.14,
            0.23,
            "Gated residual fusion\n+ dilated TCN\n+ Transformer",
            facecolor="#E5EFE7",
        )
        _draw_arrow(architecture, (0.59, 0.505), (0.64, 0.505))
        _draw_box(
            architecture,
            (0.84, 0.62),
            0.15,
            0.16,
            "Binary head\nP(seizure)",
            facecolor="#F2DFD9",
            edgecolor=SEIZURE_COLOR,
        )
        _draw_box(
            architecture,
            (0.84, 0.22),
            0.15,
            0.2,
            "Training-only head\n14 continuous EEG\nstatistics",
            facecolor="#F0E4ED",
            edgecolor=MODEL_COLORS["physio_stats_aux"],
        )
        _draw_arrow(architecture, (0.78, 0.52), (0.84, 0.7))
        _draw_arrow(architecture, (0.78, 0.48), (0.84, 0.32))
        architecture.text(
            0.205,
            0.94,
            "Physiology-informed seizure model",
            ha="left",
            va="center",
            fontsize=7.8,
            fontweight="bold",
        )
        architecture.text(
            0.205,
            0.06,
            "Training-baseline median/MAD normalization preserves relative amplitude; E77 CH7 is masked before both branches",
            ha="left",
            va="center",
            fontsize=6.4,
            color=NEUTRAL_MID,
        )
        return figure


def _figure_data_split_qc(
    data: MixedBinaryFigureData, line_noise: pd.DataFrame
) -> mpl.figure.Figure:
    with mpl.rc_context(_publication_rc(compact=True)):
        figure, axes = plt.subplots(
            2, 3, figsize=(7.4, 4.75), constrained_layout=True
        )
        ax_windows, ax_groups, ax_mouse, ax_events, ax_channels, ax_line = axes.ravel()

        pooled = (
            data.split_summary.groupby("split", as_index=False)[
                ["seizure_windows", "nonseizure_windows"]
            ]
            .sum()
            .set_index("split")
            .loc[list(SPLITS)]
        )
        x = np.arange(len(SPLITS))
        negative = pooled["nonseizure_windows"].to_numpy(float) / 1000.0
        positive = pooled["seizure_windows"].to_numpy(float) / 1000.0
        ax_windows.bar(
            x,
            negative,
            color=NONSEIZURE_COLOR,
            width=0.62,
            label="Non-seizure",
            zorder=2,
        )
        ax_windows.bar(
            x,
            positive,
            bottom=negative,
            color=SEIZURE_COLOR,
            width=0.62,
            label="Seizure",
            zorder=2,
        )
        for index, total in enumerate(negative + positive):
            ax_windows.text(index, total + 2.2, f"{int(total * 1000):,}", ha="center")
        ax_windows.set_xticks(x, ("Train", "Validation", "Test"))
        ax_windows.set_ylabel("Windows (thousands)")
        ax_windows.set_title("Frozen window allocation")
        ax_windows.legend(loc="upper right", handlelength=1.1)
        _style_axis(ax_windows, grid="y")

        groups = (
            data.group_summary.groupby(["split", "label"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=SPLITS, columns=[0, 1], fill_value=0)
        )
        ax_groups.bar(
            x,
            groups[0],
            color=NONSEIZURE_COLOR,
            width=0.62,
            label="Background blocks",
            zorder=2,
        )
        ax_groups.bar(
            x,
            groups[1],
            bottom=groups[0],
            color=SEIZURE_COLOR,
            width=0.62,
            label="Event complexes",
            zorder=2,
        )
        for index, total in enumerate(groups.sum(axis=1)):
            ax_groups.text(index, total + 24, f"{int(total):,}", ha="center")
        ax_groups.set_xticks(x, ("Train", "Validation", "Test"))
        ax_groups.set_ylabel("Independent groups")
        ax_groups.set_title("Leakage-safe grouping")
        ax_groups.text(
            0.97,
            0.05,
            "0 shared groups\n0 cross-split overlaps",
            transform=ax_groups.transAxes,
            ha="right",
            va="bottom",
            color=NEUTRAL_MID,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        )
        _style_axis(ax_groups, grid="y")

        animal_totals = (
            data.split_summary.pivot(index="animal_id", columns="split", values="windows")
            .reindex(index=ANIMALS, columns=SPLITS)
            .fillna(0)
        )
        bottom = np.zeros(len(ANIMALS), dtype=float)
        for split in SPLITS:
            values = animal_totals[split].to_numpy(float) / 1000.0
            ax_mouse.bar(
                np.arange(len(ANIMALS)),
                values,
                bottom=bottom,
                width=0.68,
                color=SPLIT_COLORS[split],
                label={"train": "Train", "val": "Validation", "test": "Test"}[split],
                zorder=2,
            )
            bottom += values
        ax_mouse.set_xticks(np.arange(len(ANIMALS)), ANIMALS)
        ax_mouse.set_ylabel("Windows (thousands)")
        ax_mouse.set_title("Contribution by mouse")
        ax_mouse.legend(ncol=3, loc="upper right", handlelength=1.0, columnspacing=0.8)
        _style_axis(ax_mouse, grid="y")

        test_summary = (
            data.split_summary.loc[data.split_summary["split"].eq("test")]
            .set_index("animal_id")
            .reindex(ANIMALS)
        )
        event_values = test_summary["seizure_groups"].to_numpy(float)
        window_values = test_summary["seizure_windows"].to_numpy(float)
        positions = np.arange(len(ANIMALS))
        ax_events.bar(
            positions,
            window_values,
            width=0.58,
            color="#D6A193",
            edgecolor=SEIZURE_COLOR,
            linewidth=0.7,
            label="Positive windows",
            zorder=2,
        )
        ax_events.set_yscale("log")
        ax_events.set_xticks(positions, ANIMALS)
        ax_events.set_ylabel("Positive windows (log scale)")
        event_axis = ax_events.twinx()
        event_axis.plot(
            positions,
            event_values,
            color=NEUTRAL_DARK,
            marker="o",
            linewidth=1.0,
            label="Positive event groups",
        )
        event_axis.set_yscale("log")
        event_axis.set_ylabel("Positive groups (log scale)")
        event_axis.spines["top"].set_visible(False)
        for index, groups_count in enumerate(event_values):
            event_axis.text(index, groups_count * 1.18, f"{int(groups_count)}", ha="center")
        ax_events.set_title("Test positive support")
        handles = [
            Line2D([0], [0], color="#D6A193", linewidth=5, label="Windows"),
            Line2D([0], [0], color=NEUTRAL_DARK, marker="o", label="Event groups"),
        ]
        ax_events.legend(handles=handles, loc="upper right")
        _style_axis(ax_events, grid="y")

        validity = np.zeros((len(ANIMALS), 2), dtype=int)
        for animal_index, animal in enumerate(ANIMALS):
            subset = data.metadata.loc[data.metadata["animal_id"].eq(animal)]
            validity[animal_index, 0] = int(subset["ch7_valid"].all())
            validity[animal_index, 1] = int(subset["ch8_valid"].all())
        cmap = ListedColormap(["#D7D7D7", "#5AAE91"])
        ax_channels.imshow(validity, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax_channels.set_xticks((0, 1), ("CH7", "CH8"))
        ax_channels.set_yticks(np.arange(len(ANIMALS)), ANIMALS)
        for row_index in range(validity.shape[0]):
            for column_index in range(validity.shape[1]):
                ax_channels.text(
                    column_index,
                    row_index,
                    "valid" if validity[row_index, column_index] else "masked",
                    ha="center",
                    va="center",
                    color="white" if validity[row_index, column_index] else NEUTRAL_DARK,
                    fontsize=5.8,
                )
        ax_channels.set_title("Channel validity")
        for spine in ax_channels.spines.values():
            spine.set_visible(False)

        raw_values = line_noise["raw_relative_power_48_52"].to_numpy(float)
        filtered_values = line_noise["filtered_relative_power_48_52"].to_numpy(float)
        medians = (float(np.median(raw_values)), float(np.median(filtered_values)))
        box = ax_line.boxplot(
            [raw_values, filtered_values],
            positions=[0, 1],
            widths=0.38,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": NEUTRAL_DARK, "linewidth": 1.2},
            boxprops={"linewidth": 0.7},
            whiskerprops={"linewidth": 0.7},
            capprops={"linewidth": 0.7},
        )
        for patch, color in zip(box["boxes"], ("#C7CDD4", "#85BBAA")):
            patch.set_facecolor(color)
        display_index = np.arange(0, len(line_noise), 12)
        for position, values, color in (
            (0, raw_values, NONSEIZURE_COLOR),
            (1, filtered_values, MODEL_COLORS["physio_no_aux"]),
        ):
            jitter = 0.035 * np.sin(np.arange(len(display_index)) * 2.399963)
            ax_line.scatter(
                np.full(len(display_index), position) + jitter,
                values[display_index],
                s=4,
                alpha=0.28,
                color=color,
                linewidths=0,
                zorder=3,
            )
        reduction = 1.0 - medians[1] / medians[0]
        ax_line.set_yscale("log")
        ax_line.set_xticks((0, 1), ("Before", "After"))
        ax_line.set_ylabel("48-52 Hz relative power")
        ax_line.set_title("Zero-phase notch check")
        ax_line.text(
            0.5,
            0.98,
            f"median reduction {reduction * 100:.1f}%\nn={len(line_noise)} valid channel-windows",
            transform=ax_line.transAxes,
            ha="center",
            va="top",
            color=NEUTRAL_MID,
        )
        _style_axis(ax_line, grid="y")
        _panel_labels(list(axes.ravel()))
        return figure


def _plot_two_channel_trace(
    ax: mpl.axes.Axes,
    values: np.ndarray,
    sampling_rate_hz: float,
    *,
    title: str,
    valid: tuple[bool, bool] = (True, True),
) -> None:
    time = np.arange(values.shape[-1]) / sampling_rate_hz
    robust = np.nanpercentile(np.abs(values[np.asarray(valid)]), 98.0)
    spacing = max(float(robust) * 2.4, 2.5)
    offsets = (spacing / 2.0, -spacing / 2.0)
    for index, (channel, color) in enumerate(
        (("CH7", "#4C78A8"), ("CH8", "#D9893D"))
    ):
        if valid[index]:
            ax.plot(time, values[index] + offsets[index], color=color, linewidth=0.65)
        else:
            ax.axhline(offsets[index], color=NEUTRAL_MID, linestyle="--", linewidth=0.7)
        ax.text(
            0.01,
            offsets[index],
            channel + ("" if valid[index] else " masked"),
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=5.8,
            color=color if valid[index] else NEUTRAL_MID,
        )
    ax.set_xlim(0.0, values.shape[-1] / sampling_rate_hz)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title(title)
    _style_axis(ax)


def _plot_psd_pair(
    ax: mpl.axes.Axes,
    raw: np.ndarray,
    filtered: np.ndarray,
    sampling_rate_hz: float,
    valid: tuple[bool, bool],
    *,
    title: str,
) -> None:
    raw_spectra: list[np.ndarray] = []
    filtered_spectra: list[np.ndarray] = []
    frequencies: np.ndarray | None = None
    for channel_index, is_valid in enumerate(valid):
        if not is_valid:
            continue
        frequencies, raw_power = periodogram(
            raw[channel_index], fs=sampling_rate_hz, detrend="constant", scaling="density"
        )
        _, filtered_power = periodogram(
            filtered[channel_index],
            fs=sampling_rate_hz,
            detrend="constant",
            scaling="density",
        )
        raw_spectra.append(raw_power)
        filtered_spectra.append(filtered_power)
    if frequencies is None:
        raise ValueError("PSD panel has no valid channel")
    mask = (frequencies >= 0.5) & (frequencies <= 70.0)
    raw_median = np.median(np.vstack(raw_spectra), axis=0)
    filtered_median = np.median(np.vstack(filtered_spectra), axis=0)
    ax.plot(
        frequencies[mask],
        raw_median[mask],
        color=NONSEIZURE_COLOR,
        linewidth=0.8,
        label="Before",
    )
    ax.plot(
        frequencies[mask],
        filtered_median[mask],
        color=MODEL_COLORS["physio_no_aux"],
        linewidth=0.9,
        label="After",
    )
    ax.axvspan(48.0, 52.0, color="#E7C6B8", alpha=0.45, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlim(0.5, 70.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power density (a.u.$^2$/Hz)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    _style_axis(ax, grid="y")


def _figure_preprocessing_examples(
    data: MixedBinaryFigureData, representative_rows: pd.DataFrame
) -> mpl.figure.Figure:
    with mpl.rc_context(_publication_rc(compact=True)):
        figure = plt.figure(figsize=(7.4, 5.6), constrained_layout=True)
        grid = figure.add_gridspec(3, 3, height_ratios=(1, 1, 0.76))
        axes: list[mpl.axes.Axes] = []
        for row_position, label in enumerate((0, 1)):
            row = representative_rows.loc[
                representative_rows["animal_id"].eq("E50")
                & representative_rows["label"].eq(label)
            ].iloc[0]
            raw, filtered = _normalized_signal_pair(row, data.training_baseline)
            valid = (bool(row["ch7_valid"]), bool(row["ch8_valid"]))
            raw_axis = figure.add_subplot(grid[row_position, 0])
            filtered_axis = figure.add_subplot(grid[row_position, 1])
            psd_axis = figure.add_subplot(grid[row_position, 2])
            class_label = "Non-seizure" if label == 0 else "Seizure"
            _plot_two_channel_trace(
                raw_axis,
                raw,
                float(row["sampling_rate_hz"]),
                title=f"{class_label}: before filtering",
                valid=valid,
            )
            _plot_two_channel_trace(
                filtered_axis,
                filtered,
                float(row["sampling_rate_hz"]),
                title=f"{class_label}: after filtering",
                valid=valid,
            )
            _plot_psd_pair(
                psd_axis,
                raw,
                filtered,
                float(row["sampling_rate_hz"]),
                valid,
                title=f"{class_label}: power spectrum",
            )
            axes.extend((raw_axis, filtered_axis, psd_axis))

        e77_row = representative_rows.loc[
            representative_rows["animal_id"].eq("E77")
        ].iloc[0]
        _, e77_filtered = _normalized_signal_pair(e77_row, data.training_baseline)
        e77_axis = figure.add_subplot(grid[2, :])
        _plot_two_channel_trace(
            e77_axis,
            e77_filtered,
            float(e77_row["sampling_rate_hz"]),
            title="E77 seizure example after filtering: invalid CH7 excluded, CH8 retained",
            valid=(False, True),
        )
        e77_axis.text(
            0.995,
            0.06,
            "Raw traces are median-centered for display; amplitudes use the training-baseline scale",
            transform=e77_axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.7,
            color=NEUTRAL_MID,
        )
        axes.append(e77_axis)
        _panel_labels(axes)
        return figure


def _performance_interval_panel(
    ax: mpl.axes.Axes,
    data: MixedBinaryFigureData,
    metric: str,
    *,
    show_labels: bool,
) -> None:
    y = np.arange(len(PRIMARY_MODELS))[::-1]
    for y_value, model in zip(y, PRIMARY_MODELS):
        values = data.bootstrap["performance"][model][metric]
        estimate = float(values["observed"])
        lower = float(values["ci95_lower"])
        upper = float(values["ci95_upper"])
        ax.errorbar(
            estimate,
            y_value,
            xerr=np.asarray([[estimate - lower], [upper - estimate]]),
            fmt=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model],
            capsize=2,
            linewidth=1.0,
            markersize=4.0,
            zorder=3,
        )
    ax.set_yticks(
        y,
        [MODEL_SHORT_LABELS[model] for model in PRIMARY_MODELS] if show_labels else [],
    )
    ax.set_xlabel("Test score")
    ax.set_title(METRIC_LABELS[metric])
    _style_axis(ax, grid="x")


def _figure_overall_performance(data: MixedBinaryFigureData) -> mpl.figure.Figure:
    with mpl.rc_context(_publication_rc(compact=True)):
        figure = plt.figure(figsize=(7.4, 5.0), constrained_layout=True)
        grid = figure.add_gridspec(2, 4, height_ratios=(1.0, 1.18))
        axes: list[mpl.axes.Axes] = []
        for index, metric in enumerate(
            ("auprc", "auroc", "balanced_accuracy", "mcc")
        ):
            axis = figure.add_subplot(grid[0, index])
            _performance_interval_panel(
                axis, data, metric, show_labels=index == 0
            )
            axes.append(axis)

        parameter_axis = figure.add_subplot(grid[1, :2])
        tradeoff_axis = figure.add_subplot(grid[1, 2:])
        test = data.test_overall.set_index("model")
        for model in PRIMARY_MODELS:
            row = test.loc[model]
            parameter_axis.scatter(
                float(row["parameters"]),
                float(row["auprc"]),
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                s=28,
                zorder=3,
            )
            offsets = {
                "eegnet82": (5, 5),
                "eegwavenet": (5, 5),
                "physio_no_aux": (7, -9),
                "physio_stats_aux": (7, 7),
            }
            parameter_axis.annotate(
                MODEL_SHORT_LABELS[model],
                (float(row["parameters"]), float(row["auprc"])),
                xytext=offsets[model],
                textcoords="offset points",
                fontsize=5.8,
            )
        parameter_axis.axhline(
            data.test_prevalence,
            color=NEUTRAL_MID,
            linestyle="--",
            linewidth=0.8,
            label=f"Random-ranking AUPRC = {data.test_prevalence:.3f}",
        )
        parameter_axis.set_xscale("log")
        parameter_axis.set_xlabel("Trainable parameters (log scale)")
        parameter_axis.set_ylabel("Test AUPRC")
        parameter_axis.set_title("Model size and ranking performance")
        parameter_axis.set_ylim(0.08, 0.88)
        parameter_axis.legend(loc="lower right")
        _style_axis(parameter_axis, grid="y")

        for model in PRIMARY_MODELS:
            row = test.loc[model]
            tradeoff_axis.scatter(
                float(row["false_positive_background_group_fraction"]),
                float(row["positive_group_sensitivity"]),
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                s=30,
                zorder=3,
            )
            offsets = {
                "eegnet82": (5, 5),
                "eegwavenet": (6, -13),
                "physio_no_aux": (6, 6),
                "physio_stats_aux": (6, 7),
            }
            tradeoff_axis.annotate(
                MODEL_SHORT_LABELS[model],
                (
                    float(row["false_positive_background_group_fraction"]),
                    float(row["positive_group_sensitivity"]),
                ),
                xytext=offsets[model],
                textcoords="offset points",
                fontsize=5.8,
            )
        tradeoff_axis.set_xlabel("Background groups with >=1 positive window")
        tradeoff_axis.set_ylabel("Positive groups with >=1 detected window")
        tradeoff_axis.set_title("Capped-group operating view")
        tradeoff_axis.set_xlim(0.3, 0.52)
        tradeoff_axis.set_ylim(0.78, 1.025)
        tradeoff_axis.text(
            0.02,
            0.02,
            "Not false alarms/hour or full-event sensitivity",
            transform=tradeoff_axis.transAxes,
            color=NEUTRAL_MID,
            fontsize=5.7,
        )
        _style_axis(tradeoff_axis, grid="both")
        axes.extend((parameter_axis, tradeoff_axis))
        _panel_labels(axes)
        return figure


def _figure_discrimination_curves(data: MixedBinaryFigureData) -> mpl.figure.Figure:
    with mpl.rc_context(_publication_rc(compact=True)):
        figure, axes = plt.subplots(
            2, 2, figsize=(7.4, 5.5), constrained_layout=True
        )
        pr_axis, roc_axis, calibration_axis, scores_axis = axes.ravel()
        score_groups: list[np.ndarray] = []
        score_positions: list[float] = []
        score_colors: list[str] = []
        for model_index, model in enumerate(PRIMARY_MODELS):
            test = data.predictions[model].loc[
                data.predictions[model]["split"].eq("test")
            ]
            labels = test["label"].to_numpy(np.int8)
            scores = test["seizure_probability"].to_numpy(float)
            precision, recall, _ = precision_recall_curve(labels, scores)
            false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
            pr_axis.plot(
                recall,
                precision,
                color=MODEL_COLORS[model],
                linestyle=MODEL_LINESTYLES[model],
                label=(
                    f"{MODEL_SHORT_LABELS[model]} "
                    f"({average_precision_score(labels, scores):.3f})"
                ),
            )
            roc_axis.plot(
                false_positive_rate,
                true_positive_rate,
                color=MODEL_COLORS[model],
                linestyle=MODEL_LINESTYLES[model],
                label=(
                    f"{MODEL_SHORT_LABELS[model]} "
                    f"({roc_auc_score(labels, scores):.3f})"
                ),
            )
            observed, predicted = calibration_curve(
                labels, scores, n_bins=12, strategy="quantile"
            )
            calibration_axis.plot(
                predicted,
                observed,
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                linestyle=MODEL_LINESTYLES[model],
                markersize=3.0,
            )
            for label_index in (0, 1):
                score_groups.append(scores[labels == label_index])
                score_positions.append(model_index * 3.0 + label_index)
                score_colors.append(
                    NONSEIZURE_COLOR if label_index == 0 else SEIZURE_COLOR
                )

        pr_axis.axhline(
            data.test_prevalence,
            color=NEUTRAL_MID,
            linestyle=(0, (3, 2)),
            linewidth=0.8,
            label=f"Prevalence ({data.test_prevalence:.3f})",
        )
        pr_axis.set_xlim(0, 1)
        pr_axis.set_ylim(0, 1.02)
        pr_axis.set_xlabel("Recall")
        pr_axis.set_ylabel("Precision")
        pr_axis.set_title("Precision-recall (AUPRC)")
        pr_axis.legend(loc="lower left")
        _style_axis(pr_axis, grid="both")

        roc_axis.plot((0, 1), (0, 1), color=NEUTRAL_MID, linestyle="--", linewidth=0.8)
        roc_axis.set_xlim(0, 1)
        roc_axis.set_ylim(0, 1.02)
        roc_axis.set_xlabel("False-positive rate")
        roc_axis.set_ylabel("True-positive rate")
        roc_axis.set_title("Receiver operating characteristic (AUROC)")
        roc_axis.legend(loc="lower right")
        _style_axis(roc_axis, grid="both")

        calibration_axis.plot(
            (0, 1), (0, 1), color=NEUTRAL_MID, linestyle="--", linewidth=0.8
        )
        calibration_axis.set_xlim(0, 1)
        calibration_axis.set_ylim(0, 1)
        calibration_axis.set_xlabel("Mean predicted probability")
        calibration_axis.set_ylabel("Observed seizure fraction")
        calibration_axis.set_title("Window-level calibration")
        _style_axis(calibration_axis, grid="both")

        box = scores_axis.boxplot(
            score_groups,
            positions=score_positions,
            widths=0.72,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "white", "linewidth": 0.9},
            boxprops={"linewidth": 0.6},
            whiskerprops={"linewidth": 0.6},
            capprops={"linewidth": 0.6},
        )
        for patch, color in zip(box["boxes"], score_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.82)
        scores_axis.set_xticks(
            np.arange(len(PRIMARY_MODELS)) * 3.0 + 0.5,
            [MODEL_SHORT_LABELS[model] for model in PRIMARY_MODELS],
            rotation=18,
            ha="right",
        )
        scores_axis.set_ylabel("Predicted seizure probability")
        scores_axis.set_title("Score separation by true class")
        scores_axis.legend(
            handles=[
                Line2D([0], [0], color=NONSEIZURE_COLOR, linewidth=5, label="Non-seizure"),
                Line2D([0], [0], color=SEIZURE_COLOR, linewidth=5, label="Seizure"),
            ],
            loc="upper center",
            ncol=2,
        )
        _style_axis(scores_axis, grid="y")
        _panel_labels(list(axes.ravel()))
        return figure


def _figure_confusion_matrices(data: MixedBinaryFigureData) -> mpl.figure.Figure:
    with mpl.rc_context(_publication_rc(compact=True)):
        figure, axes = plt.subplots(
            2, 2, figsize=(6.2, 5.3), constrained_layout=True
        )
        cmap = LinearSegmentedColormap.from_list(
            "nature_blue", ("#F2F5F7", "#7EA9C8", "#174A6E")
        )
        for axis, model in zip(axes.ravel(), PRIMARY_MODELS):
            values = data.metrics[model]["test"]["window"]
            matrix = np.asarray(
                [[values["tn"], values["fp"]], [values["fn"], values["tp"]]],
                dtype=int,
            )
            fractions = matrix / matrix.sum(axis=1, keepdims=True)
            axis.imshow(fractions, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
            for row in range(2):
                for column in range(2):
                    value = float(fractions[row, column])
                    axis.text(
                        column,
                        row,
                        f"{matrix[row, column]:,}\n{value * 100:.1f}%",
                        ha="center",
                        va="center",
                        color="white" if value > 0.55 else NEUTRAL_DARK,
                        fontsize=7.2,
                        fontweight="bold" if row == column else "normal",
                    )
            axis.set_xticks((0, 1), ("Non-seizure", "Seizure"))
            axis.set_yticks((0, 1), ("Non-seizure", "Seizure"))
            axis.set_xlabel("Predicted label")
            axis.set_ylabel("True label")
            axis.set_title(
                f"{MODEL_LABELS[model]}  |  threshold={float(data.metrics[model]['threshold']):.3f}"
            )
            axis.tick_params(length=0)
            for spine in axis.spines.values():
                spine.set_visible(False)
        _panel_labels(list(axes.ravel()))
        return figure


def _annotated_heatmap(
    ax: mpl.axes.Axes,
    matrix: np.ndarray,
    *,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    title: str,
    cmap: str = "cividis",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> mpl.image.AxesImage:
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(x_labels)), x_labels)
    ax.set_yticks(np.arange(len(y_labels)), y_labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix[row, column])
            color = "white" if value < 0.38 or value > 0.78 else NEUTRAL_DARK
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=5.7,
            )
    ax.set_title(title)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def _figure_per_mouse_performance(data: MixedBinaryFigureData) -> mpl.figure.Figure:
    with mpl.rc_context(_publication_rc(compact=True)):
        figure, axes = plt.subplots(
            2, 2, figsize=(7.4, 5.15), constrained_layout=True
        )
        auprc_axis, ba_axis, sensitivity_axis, support_axis = axes.ravel()
        test = data.per_mouse.loc[
            data.per_mouse["split"].eq("test")
            & data.per_mouse["model"].isin(PRIMARY_MODELS)
        ].copy()
        mouse_labels = ("E50", "E76", "E77", "E79", "E81*", "E82*")
        y_labels = [MODEL_SHORT_LABELS[model] for model in PRIMARY_MODELS]
        auprc_matrix = (
            test.pivot(index="model", columns="animal_id", values="auprc")
            .reindex(index=PRIMARY_MODELS, columns=ANIMALS)
            .to_numpy(float)
        )
        ba_matrix = (
            test.pivot(index="model", columns="animal_id", values="balanced_accuracy")
            .reindex(index=PRIMARY_MODELS, columns=ANIMALS)
            .to_numpy(float)
        )
        image = _annotated_heatmap(
            auprc_axis,
            auprc_matrix,
            x_labels=mouse_labels,
            y_labels=y_labels,
            title="Mouse-specific AUPRC",
        )
        colorbar = figure.colorbar(image, ax=auprc_axis, fraction=0.04, pad=0.02)
        colorbar.set_label("AUPRC")
        image_ba = _annotated_heatmap(
            ba_axis,
            ba_matrix,
            x_labels=mouse_labels,
            y_labels=y_labels,
            title="Mouse-specific balanced accuracy",
        )
        colorbar_ba = figure.colorbar(image_ba, ax=ba_axis, fraction=0.04, pad=0.02)
        colorbar_ba.set_label("Balanced accuracy")

        x = np.arange(len(ANIMALS))
        for model in PRIMARY_MODELS:
            values = (
                test.loc[test["model"].eq(model)]
                .set_index("animal_id")
                .reindex(ANIMALS)["sensitivity"]
                .to_numpy(float)
            )
            sensitivity_axis.plot(
                x,
                values,
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                linestyle=MODEL_LINESTYLES[model],
                label=MODEL_SHORT_LABELS[model],
            )
        sensitivity_axis.set_xticks(x, mouse_labels)
        sensitivity_axis.set_ylim(-0.02, 1.02)
        sensitivity_axis.set_xlabel("Mouse")
        sensitivity_axis.set_ylabel("Sensitivity")
        sensitivity_axis.set_title("Fixed-threshold sensitivity")
        sensitivity_axis.legend(ncol=2, loc="lower left")
        _style_axis(sensitivity_axis, grid="y")

        summary = (
            data.split_summary.loc[data.split_summary["split"].eq("test")]
            .set_index("animal_id")
            .reindex(ANIMALS)
        )
        windows = summary["seizure_windows"].to_numpy(float)
        events = summary["seizure_groups"].to_numpy(float)
        support_axis.bar(
            x - 0.18,
            windows,
            width=0.36,
            color="#D6A193",
            edgecolor=SEIZURE_COLOR,
            linewidth=0.6,
            label="Positive windows",
            zorder=2,
        )
        support_axis.bar(
            x + 0.18,
            events,
            width=0.36,
            color="#B8BEC5",
            edgecolor=NEUTRAL_MID,
            linewidth=0.6,
            label="Positive event groups",
            zorder=2,
        )
        support_axis.set_yscale("log")
        support_axis.set_xticks(x, mouse_labels)
        support_axis.set_xlabel("Mouse")
        support_axis.set_ylabel("Count (log scale)")
        support_axis.set_title("Test positive support")
        support_axis.legend(loc="upper right")
        for index in (4, 5):
            support_axis.text(
                index + 0.18,
                events[index] * 1.45,
                "n=1",
                ha="center",
                fontsize=5.7,
                color=NEUTRAL_DARK,
            )
        _style_axis(support_axis, grid="y")
        figure.text(
            0.5,
            -0.005,
            "* E81 and E82 each contain one positive test event complex (64 positive windows).",
            ha="center",
            fontsize=5.9,
            color=NEUTRAL_MID,
        )
        _panel_labels(list(axes.ravel()))
        return figure


def _figure_per_mouse_pr_curves(data: MixedBinaryFigureData) -> mpl.figure.Figure:
    with mpl.rc_context(_publication_rc(compact=True)):
        figure, axes = plt.subplots(
            2, 3, figsize=(7.4, 4.65), constrained_layout=True
        )
        support = (
            data.split_summary.loc[data.split_summary["split"].eq("test")]
            .set_index("animal_id")
            .reindex(ANIMALS)
        )
        for axis, animal in zip(axes.ravel(), ANIMALS):
            prevalence = None
            for model in PRIMARY_MODELS:
                mouse = data.predictions[model].loc[
                    data.predictions[model]["split"].eq("test")
                    & data.predictions[model]["animal_id"].eq(animal)
                ]
                labels = mouse["label"].to_numpy(np.int8)
                scores = mouse["seizure_probability"].to_numpy(float)
                precision, recall, _ = precision_recall_curve(labels, scores)
                prevalence = float(labels.mean())
                axis.plot(
                    recall,
                    precision,
                    color=MODEL_COLORS[model],
                    linestyle=MODEL_LINESTYLES[model],
                    linewidth=1.0,
                )
            if prevalence is None:
                raise ValueError(f"no test predictions for {animal}")
            axis.axhline(
                prevalence,
                color=NEUTRAL_MID,
                linestyle=(0, (3, 2)),
                linewidth=0.7,
            )
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1.02)
            axis.set_xlabel("Recall")
            axis.set_ylabel("Precision")
            event_count = int(support.loc[animal, "seizure_groups"])
            event_word = "event group" if event_count == 1 else "event groups"
            axis.set_title(
                f"{animal}  |  {event_count} {event_word}; "
                f"{int(support.loc[animal, 'seizure_windows'])} positive windows"
            )
            _style_axis(axis, grid="both")
        figure.legend(
            handles=_model_legend(),
            loc="outside upper center",
            ncol=4,
            handlelength=2.2,
        )
        _panel_labels(list(axes.ravel()))
        return figure


def _figure_training_histories(data: MixedBinaryFigureData) -> mpl.figure.Figure:
    with mpl.rc_context(_publication_rc(compact=True)):
        figure, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), constrained_layout=True)
        physio_models = ("physio_no_aux", "physio_stats_aux")
        for axis, model in zip(axes[:2], physio_models):
            history = data.histories[model]
            best_epoch = int(data.metrics[model]["best_epoch"])
            final_epoch = int(history["epoch"].iloc[-1])
            axis.plot(
                history["epoch"],
                history["train_loss"],
                color=MODEL_COLORS[model],
                linewidth=1.0,
                label="Training loss",
            )
            axis.plot(
                history["epoch"],
                history["validation_loss"],
                color=NEUTRAL_DARK,
                linestyle="--",
                linewidth=1.0,
                label="Validation loss",
            )
            axis.axvspan(
                best_epoch,
                final_epoch,
                color="#E9E9E9",
                alpha=0.65,
                linewidth=0,
                label="Post-best epochs",
            )
            axis.axvline(
                best_epoch, color=NEUTRAL_DARK, linestyle=":", linewidth=0.8
            )
            best_row = history.loc[history["epoch"].eq(best_epoch)].iloc[0]
            final_row = history.iloc[-1]
            axis.annotate(
                (
                    f"best epoch {best_epoch}\n"
                    f"val loss {float(best_row['validation_loss']):.3f} -> "
                    f"{float(final_row['validation_loss']):.3f}"
                ),
                xy=(best_epoch, float(best_row["validation_loss"])),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=5.5,
                color=NEUTRAL_MID,
            )
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Optimization loss")
            axis.set_title(MODEL_LABELS[model])
            axis.legend(loc="upper right")
            _style_axis(axis, grid="y")

        metric_axis = axes[2]
        for model in physio_models:
            history = data.histories[model]
            best_epoch = int(data.metrics[model]["best_epoch"])
            best_row = history.loc[history["epoch"].eq(best_epoch)].iloc[0]
            metric_axis.plot(
                history["epoch"],
                history["validation_auprc"],
                color=MODEL_COLORS[model],
                linestyle=MODEL_LINESTYLES[model],
                label=MODEL_SHORT_LABELS[model],
            )
            metric_axis.scatter(
                best_epoch,
                float(best_row["validation_auprc"]),
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                s=24,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            metric_axis.annotate(
                f"epoch {best_epoch}",
                (best_epoch, float(best_row["validation_auprc"])),
                xytext=(4, 5 if model == "physio_no_aux" else -10),
                textcoords="offset points",
                fontsize=5.5,
                color=MODEL_COLORS[model],
            )
        metric_axis.set_xlabel("Epoch")
        metric_axis.set_ylabel("Validation AUPRC")
        metric_axis.set_ylim(0.77, 0.825)
        metric_axis.set_title("Checkpoint-selection metric")
        metric_axis.legend(loc="lower right")
        _style_axis(metric_axis, grid="y")
        figure.text(
            0.5,
            -0.015,
            "After the selected checkpoint, training loss continues to fall while validation loss rises or fluctuates; the final test uses the validation-selected checkpoint, not the last epoch.",
            ha="center",
            fontsize=5.7,
            color=NEUTRAL_MID,
        )
        _panel_labels(list(axes))
        return figure


def _contrast_forest_panel(
    ax: mpl.axes.Axes,
    data: MixedBinaryFigureData,
    contrast: str,
    *,
    title: str,
    metrics: Sequence[str] = BOOTSTRAP_METRICS,
) -> None:
    values = data.bootstrap["contrasts"][contrast]
    y = np.arange(len(metrics))[::-1]
    for position, metric in zip(y, metrics):
        summary = values[metric]
        estimate = float(summary["observed"])
        lower = float(summary["ci95_lower"])
        upper = float(summary["ci95_upper"])
        ax.errorbar(
            estimate,
            position,
            xerr=np.asarray([[estimate - lower], [upper - estimate]]),
            fmt="o",
            color=CONTRAST_COLORS[contrast],
            capsize=2,
            linewidth=1.0,
            markersize=3.8,
            zorder=3,
        )
    ax.axvline(0.0, color=NEUTRAL_MID, linestyle="--", linewidth=0.8)
    ax.set_yticks(y, [METRIC_LABELS[metric] for metric in metrics])
    ax.set_xlabel("Paired difference (95% group-bootstrap CI)")
    ax.set_title(title)
    _style_axis(ax, grid="x")


def _figure_auxiliary_ablation(data: MixedBinaryFigureData) -> mpl.figure.Figure:
    with mpl.rc_context(_publication_rc(compact=True)):
        figure, axes = plt.subplots(
            2, 2, figsize=(7.4, 5.4), constrained_layout=True
        )
        main_axis, forest_axis, seeds_axis, stats_axis = axes.ravel()
        test = data.overall.loc[
            data.overall["split"].eq("test")
            & data.overall["model"].isin(("physio_no_aux", "physio_stats_aux"))
        ].set_index("model")
        metrics = (
            "auprc",
            "auroc",
            "balanced_accuracy",
            "mcc",
            "macro_mouse_auprc",
            "macro_mouse_balanced_accuracy",
        )
        x = np.arange(len(metrics))
        for model in ("physio_no_aux", "physio_stats_aux"):
            values = [float(test.loc[model, metric]) for metric in metrics]
            main_axis.plot(
                x,
                values,
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                linestyle=MODEL_LINESTYLES[model],
                label=MODEL_LABELS[model],
            )
        for index, metric in enumerate(metrics):
            main_axis.plot(
                [index, index],
                [float(test.loc["physio_no_aux", metric]), float(test.loc["physio_stats_aux", metric])],
                color="#CFCFCF",
                linewidth=0.7,
                zorder=0,
            )
        main_axis.set_xticks(
            x,
            ("AUPRC", "AUROC", "BA", "MCC", "Macro AUPRC", "Macro BA"),
            rotation=20,
            ha="right",
        )
        main_axis.set_ylim(0.45, 0.98)
        main_axis.set_ylabel("Test score")
        main_axis.set_title("Main-seed paired comparison")
        main_axis.legend(loc="lower left")
        _style_axis(main_axis, grid="y")

        _contrast_forest_panel(
            forest_axis,
            data,
            "physio_stats_aux_minus_physio_no_aux",
            title="Auxiliary minus no-aux",
        )

        seed_frame = data.seed_metrics.loc[
            data.seed_metrics["model"].isin(("physio_no_aux", "physio_stats_aux"))
        ]
        for seed, group in seed_frame.groupby("seed", sort=True):
            ordered = group.set_index("model").reindex(
                ("physio_no_aux", "physio_stats_aux")
            )
            seeds_axis.plot(
                (0, 1),
                ordered["test_auprc"].to_numpy(float),
                color="#8B8B8B",
                marker="o",
                linewidth=0.8,
                markersize=3.5,
            )
            seeds_axis.text(
                1.04,
                float(ordered.loc["physio_stats_aux", "test_auprc"]),
                str(int(seed)),
                va="center",
                fontsize=5.5,
                color=NEUTRAL_MID,
            )
        mean_values = (
            seed_frame.groupby("model")["test_auprc"]
            .mean()
            .reindex(("physio_no_aux", "physio_stats_aux"))
            .to_numpy(float)
        )
        seeds_axis.plot(
            (0, 1),
            mean_values,
            color=MODEL_COLORS["physio_stats_aux"],
            marker="D",
            linewidth=1.8,
            markersize=4.5,
            label="Three-seed mean",
        )
        seeds_axis.set_xticks((0, 1), ("No aux", "+ stats aux"))
        seeds_axis.set_xlim(-0.18, 1.24)
        seeds_axis.set_ylim(0.78, 0.82)
        seeds_axis.set_ylabel("Test AUPRC")
        seeds_axis.set_title("Three paired training seeds")
        seeds_axis.legend(loc="lower center")
        _style_axis(seeds_axis, grid="y")

        for model, color, linestyle in (
            ("physio_no_aux", MODEL_COLORS["physio_no_aux"], "--"),
            ("physio_stats_aux", MODEL_COLORS["physio_stats_aux"], "-"),
        ):
            history = data.histories[model]
            stats_axis.plot(
                history["epoch"],
                history["train_statistics_loss"],
                color=color,
                linestyle=linestyle,
                linewidth=1.0,
                label=f"{MODEL_SHORT_LABELS[model]} train",
            )
            stats_axis.plot(
                history["epoch"],
                history["validation_statistics_loss"],
                color=color,
                linestyle=":" if linestyle == "-" else "-.",
                linewidth=0.9,
                label=f"{MODEL_SHORT_LABELS[model]} validation",
            )
        stats_axis.set_yscale("log")
        stats_axis.set_xlabel("Epoch")
        stats_axis.set_ylabel("Unweighted statistics loss")
        stats_axis.set_title("Continuous-statistics reconstruction")
        stats_axis.legend(ncol=2, loc="upper right")
        stats_axis.text(
            0.02,
            0.03,
            "No-aux loss is diagnostic only (optimization weight = 0).",
            transform=stats_axis.transAxes,
            fontsize=5.6,
            color=NEUTRAL_MID,
        )
        _style_axis(stats_axis, grid="y")
        _panel_labels(list(axes.ravel()))
        return figure


def _figure_mouse_balanced_sensitivity(
    data: MixedBinaryFigureData,
) -> mpl.figure.Figure:
    with mpl.rc_context(_publication_rc(compact=True)):
        figure, axes = plt.subplots(
            2, 2, figsize=(7.4, 5.3), constrained_layout=True
        )
        summary_axis, sensitivity_axis, auprc_axis, forest_axis = axes.ravel()
        models = ("physio_no_aux", "physio_no_aux_mouse_balanced")
        test_overall = data.overall.loc[
            data.overall["split"].eq("test") & data.overall["model"].isin(models)
        ].set_index("model")
        metric_pairs = (
            ("auprc", "Pooled AUPRC"),
            ("macro_mouse_auprc", "Macro AUPRC"),
            ("balanced_accuracy", "Pooled BA"),
            ("macro_mouse_balanced_accuracy", "Macro BA"),
        )
        x = np.arange(len(metric_pairs))
        for model in models:
            values = [float(test_overall.loc[model, metric]) for metric, _ in metric_pairs]
            summary_axis.plot(
                x,
                values,
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                linestyle=MODEL_LINESTYLES[model],
                label=MODEL_LABELS[model],
            )
        summary_axis.set_xticks(
            x, [label for _, label in metric_pairs], rotation=18, ha="right"
        )
        summary_axis.set_ylim(0.48, 0.88)
        summary_axis.set_ylabel("Test score")
        summary_axis.set_title("Pooled and mouse-macro trade-off")
        summary_axis.legend(loc="lower left")
        _style_axis(summary_axis, grid="y")

        test_mouse = data.per_mouse.loc[
            data.per_mouse["split"].eq("test")
            & data.per_mouse["model"].isin(models)
        ]
        positions = np.arange(len(ANIMALS))
        for axis, metric, title in (
            (sensitivity_axis, "sensitivity", "Per-mouse sensitivity"),
            (auprc_axis, "auprc", "Per-mouse AUPRC"),
        ):
            for mouse_index, animal in enumerate(ANIMALS):
                values = (
                    test_mouse.loc[test_mouse["animal_id"].eq(animal)]
                    .set_index("model")
                    .reindex(models)[metric]
                    .to_numpy(float)
                )
                axis.plot(
                    (mouse_index - 0.12, mouse_index + 0.12),
                    values,
                    color="#BDBDBD",
                    linewidth=0.8,
                    zorder=1,
                )
                for offset, model, value in zip((-0.12, 0.12), models, values):
                    axis.scatter(
                        mouse_index + offset,
                        value,
                        color=MODEL_COLORS[model],
                        marker=MODEL_MARKERS[model],
                        s=22,
                        zorder=3,
                    )
            axis.set_xticks(positions, ANIMALS)
            axis.set_ylim(-0.02, 1.02)
            axis.set_xlabel("Mouse")
            axis.set_ylabel(METRIC_LABELS.get(metric, metric))
            axis.set_title(title)
            _style_axis(axis, grid="y")
        sensitivity_axis.legend(
            handles=_model_legend(models), loc="upper right", handlelength=2.0
        )

        _contrast_forest_panel(
            forest_axis,
            data,
            "physio_no_aux_mouse_balanced_minus_physio_no_aux",
            title="Mouse-balanced minus base Physio",
        )
        _panel_labels(list(axes.ravel()))
        return figure


def _figure_multiseed_stability(data: MixedBinaryFigureData) -> mpl.figure.Figure:
    with mpl.rc_context(_publication_rc(compact=True)):
        figure, axes = plt.subplots(
            3, 2, figsize=(7.4, 7.0), constrained_layout=True
        )
        metrics = (
            ("test_auprc", "Test AUPRC"),
            ("test_auroc", "Test AUROC"),
            ("test_balanced_accuracy", "Test balanced accuracy"),
            ("test_mcc", "Test MCC"),
            ("macro_mouse_auprc", "Macro-mouse AUPRC"),
            ("macro_mouse_balanced_accuracy", "Macro-mouse BA"),
        )
        models = ("physio_no_aux", "physio_stats_aux")
        frame = data.seed_metrics.loc[data.seed_metrics["model"].isin(models)]
        for axis, (metric, label) in zip(axes.ravel(), metrics):
            for seed, group in frame.groupby("seed", sort=True):
                ordered = group.set_index("model").reindex(models)
                values = ordered[metric].to_numpy(float)
                axis.plot(
                    (0, 1),
                    values,
                    color="#A4A4A4",
                    marker="o",
                    linewidth=0.8,
                    markersize=3.5,
                )
                axis.annotate(
                    str(int(seed))[-2:],
                    (1, values[1]),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=5.5,
                    color=NEUTRAL_MID,
                )
            aggregate = frame.groupby("model")[metric].agg(["mean", "std"]).reindex(models)
            axis.errorbar(
                (0, 1),
                aggregate["mean"].to_numpy(float),
                yerr=aggregate["std"].to_numpy(float),
                color=MODEL_COLORS["physio_stats_aux"],
                marker="D",
                linewidth=1.7,
                capsize=3,
                markersize=4.2,
                label="Mean +/- sample SD",
                zorder=4,
            )
            values_all = frame[metric].to_numpy(float)
            margin = max(float(np.ptp(values_all)) * 0.35, 0.006)
            axis.set_ylim(float(values_all.min()) - margin, float(values_all.max()) + margin)
            axis.set_xticks((0, 1), ("No aux", "+ stats aux"))
            axis.set_xlim(-0.18, 1.23)
            axis.set_ylabel(label)
            axis.set_title(label)
            _style_axis(axis, grid="y")
        axes[0, 0].legend(loc="lower center")
        figure.text(
            0.5,
            -0.004,
            "Three paired seeds are descriptive repeats; error bars are sample SD, not 95% confidence intervals.",
            ha="center",
            fontsize=5.9,
            color=NEUTRAL_MID,
        )
        _panel_labels(list(axes.ravel()))
        return figure


def _figure_bootstrap_forest(data: MixedBinaryFigureData) -> mpl.figure.Figure:
    with mpl.rc_context(_publication_rc(compact=True)):
        figure, axes = plt.subplots(
            len(BOOTSTRAP_METRICS),
            2,
            figsize=(7.4, 8.2),
            constrained_layout=True,
        )
        contrast_names = tuple(data.bootstrap["contrasts"])
        for row_index, metric in enumerate(BOOTSTRAP_METRICS):
            performance_axis = axes[row_index, 0]
            contrast_axis = axes[row_index, 1]
            y_model = np.arange(len(PRIMARY_MODELS))[::-1]
            performance_lows: list[float] = []
            performance_highs: list[float] = []
            for position, model in zip(y_model, PRIMARY_MODELS):
                summary = data.bootstrap["performance"][model][metric]
                estimate = float(summary["observed"])
                lower = float(summary["ci95_lower"])
                upper = float(summary["ci95_upper"])
                performance_lows.append(lower)
                performance_highs.append(upper)
                performance_axis.errorbar(
                    estimate,
                    position,
                    xerr=np.asarray([[estimate - lower], [upper - estimate]]),
                    fmt=MODEL_MARKERS[model],
                    color=MODEL_COLORS[model],
                    capsize=2,
                    linewidth=0.95,
                    markersize=3.6,
                )
            lower_limit = max(0.0, min(performance_lows) - 0.04)
            upper_limit = min(1.0, max(performance_highs) + 0.04)
            performance_axis.set_xlim(lower_limit, upper_limit)
            performance_axis.set_yticks(
                y_model,
                [MODEL_SHORT_LABELS[model] for model in PRIMARY_MODELS],
            )
            performance_axis.set_xlabel("Observed score (95% CI)")
            performance_axis.set_title(METRIC_LABELS[metric])
            _style_axis(performance_axis, grid="x")

            y_contrast = np.arange(len(contrast_names))[::-1]
            bounds: list[float] = []
            for position, contrast in zip(y_contrast, contrast_names):
                summary = data.bootstrap["contrasts"][contrast][metric]
                estimate = float(summary["observed"])
                lower = float(summary["ci95_lower"])
                upper = float(summary["ci95_upper"])
                bounds.extend((abs(lower), abs(upper)))
                contrast_axis.errorbar(
                    estimate,
                    position,
                    xerr=np.asarray([[estimate - lower], [upper - estimate]]),
                    fmt="o",
                    color=CONTRAST_COLORS[contrast],
                    capsize=2,
                    linewidth=0.95,
                    markersize=3.5,
                )
            maximum = max(bounds) * 1.18
            contrast_axis.set_xlim(-maximum, maximum)
            contrast_axis.axvline(
                0.0, color=NEUTRAL_MID, linestyle="--", linewidth=0.75
            )
            contrast_axis.set_yticks(
                y_contrast,
                [CONTRAST_LABELS[name] for name in contrast_names],
            )
            contrast_axis.set_xlabel("Paired difference (95% CI)")
            contrast_axis.set_title(f"Planned contrasts: {METRIC_LABELS[metric]}")
            _style_axis(contrast_axis, grid="x")
        _panel_labels(list(axes.ravel()))
        return figure


def _draw_compact_workflow(ax: mpl.axes.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = (
        (0.01, 0.15, "6 mice\npooled", "#E7EDF4"),
        (0.2, 0.18, "Group-level\n70/15/15 split", "#E8F1ED"),
        (0.42, 0.18, "0.5-70 Hz\n+ 50 Hz notch", "#F1E7D6"),
        (0.64, 0.16, "4 primary\nmodel runs", "#E9E5F0"),
        (0.84, 0.15, "Frozen test\n+ group bootstrap", "#F2DFD9"),
    )
    for index, (x, width, text, color) in enumerate(boxes):
        _draw_box(ax, (x, 0.26), width, 0.48, text, facecolor=color, fontsize=6.2)
        if index < len(boxes) - 1:
            _draw_arrow(ax, (x + width, 0.5), (boxes[index + 1][0], 0.5))


def _figure_results_overview(data: MixedBinaryFigureData) -> mpl.figure.Figure:
    with mpl.rc_context(_publication_rc(compact=True)):
        figure = plt.figure(figsize=(7.4, 6.0), constrained_layout=True)
        grid = figure.add_gridspec(
            3, 4, height_ratios=(0.58, 1.0, 1.0), width_ratios=(1.1, 1.1, 1.0, 0.72)
        )
        workflow_axis = figure.add_subplot(grid[0, :])
        pr_axis = figure.add_subplot(grid[1:, :2])
        baseline_axis = figure.add_subplot(grid[1, 2:])
        mouse_axis = figure.add_subplot(grid[2, 2])
        seed_axis = figure.add_subplot(grid[2, 3])
        _draw_compact_workflow(workflow_axis)

        for model in PRIMARY_MODELS:
            test = data.predictions[model].loc[
                data.predictions[model]["split"].eq("test")
            ]
            labels = test["label"].to_numpy(np.int8)
            scores = test["seizure_probability"].to_numpy(float)
            precision, recall, _ = precision_recall_curve(labels, scores)
            pr_axis.plot(
                recall,
                precision,
                color=MODEL_COLORS[model],
                linestyle=MODEL_LINESTYLES[model],
                label=f"{MODEL_SHORT_LABELS[model]} ({average_precision_score(labels, scores):.3f})",
            )
        pr_axis.axhline(
            data.test_prevalence,
            color=NEUTRAL_MID,
            linestyle="--",
            linewidth=0.8,
            label=f"Random ranking ({data.test_prevalence:.3f})",
        )
        pr_axis.set_xlim(0, 1)
        pr_axis.set_ylim(0, 1.02)
        pr_axis.set_xlabel("Recall")
        pr_axis.set_ylabel("Precision")
        pr_axis.set_title("Pooled test precision-recall")
        pr_axis.legend(loc="lower left")
        _style_axis(pr_axis, grid="both")

        metric_names = ("accuracy", "sensitivity", "balanced_accuracy", "auprc")
        x = np.arange(len(metric_names))
        dummy = np.asarray(
            [1.0 - data.test_prevalence, 0.0, 0.5, data.test_prevalence],
            dtype=float,
        )
        baseline_axis.plot(
            x,
            dummy,
            color=NEUTRAL_DARK,
            marker="x",
            linestyle="--",
            linewidth=0.9,
            label="Always non-seizure",
        )
        overall = data.test_overall.set_index("model")
        for model in ("physio_no_aux", "physio_stats_aux"):
            values = overall.loc[model, list(metric_names)].to_numpy(float)
            baseline_axis.plot(
                x,
                values,
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                linestyle=MODEL_LINESTYLES[model],
                label=MODEL_SHORT_LABELS[model],
            )
        baseline_axis.set_xticks(x, ("Accuracy", "Sensitivity", "BA", "AUPRC"))
        baseline_axis.set_ylim(-0.03, 1.0)
        baseline_axis.set_ylabel("Test score")
        baseline_axis.set_title("Why accuracy alone is misleading")
        baseline_axis.legend(loc="lower right")
        baseline_axis.text(
            0.02,
            0.84,
            f"All-negative accuracy = {dummy[0] * 100:.2f}%\nbut sensitivity = 0",
            transform=baseline_axis.transAxes,
            ha="left",
            va="top",
            color=NEUTRAL_DARK,
            fontsize=5.8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.3},
        )
        _style_axis(baseline_axis, grid="y")

        physio_mouse = (
            data.per_mouse.loc[
                data.per_mouse["split"].eq("test")
                & data.per_mouse["model"].eq("physio_no_aux")
            ]
            .set_index("animal_id")
            .reindex(ANIMALS)
        )
        mouse_matrix = np.vstack(
            [
                physio_mouse["auprc"].to_numpy(float),
                physio_mouse["balanced_accuracy"].to_numpy(float),
                physio_mouse["sensitivity"].to_numpy(float),
            ]
        )
        _annotated_heatmap(
            mouse_axis,
            mouse_matrix,
            x_labels=("E50", "E76", "E77", "E79", "E81*", "E82*"),
            y_labels=("AUPRC", "BA", "Sensitivity"),
            title="Physio heterogeneity",
        )

        difference = data.seed_differences.set_index("metric")
        selected = difference.loc[
            ["test_auprc", "test_balanced_accuracy", "macro_mouse_auprc"]
        ]
        values = selected["mean_aux_minus_no_aux"].to_numpy(float)
        colors = [
            MODEL_COLORS["physio_stats_aux"] if value >= 0 else MODEL_COLORS["eegwavenet"]
            for value in values
        ]
        y = np.arange(len(values))[::-1]
        seed_axis.barh(y, values, color=colors, height=0.56)
        seed_axis.axvline(0, color=NEUTRAL_MID, linewidth=0.7)
        seed_axis.set_yticks(y, ("AUPRC", "BA", "Macro AUPRC"))
        seed_axis.set_xlabel("Mean aux - no aux")
        seed_axis.set_title("Three-seed auxiliary effect")
        seed_axis.set_xlim(-0.04, 0.01)
        seed_axis.set_xticks((-0.03, 0.0))
        seed_axis.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
        _style_axis(seed_axis, grid="x")

        axes = (workflow_axis, pr_axis, baseline_axis, mouse_axis, seed_axis)
        _panel_labels(axes[:4])
        seed_axis.text(
            -0.34,
            1.08,
            "e",
            transform=seed_axis.transAxes,
            fontsize=8.5,
            fontweight="bold",
            ha="left",
            va="top",
            clip_on=False,
        )
        return figure


def _draw_compact_architecture(ax: mpl.axes.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _draw_box(
        ax,
        (0.01, 0.36),
        0.13,
        0.28,
        "4 s EEG\n2 channels\n+ mask",
        facecolor="#E7EDF4",
        fontsize=6.2,
    )
    _draw_box(
        ax,
        (0.21, 0.61),
        0.17,
        0.23,
        "Temporal kernels\n9 / 31 / 101",
        facecolor="#DDEEE8",
        fontsize=6.2,
    )
    _draw_box(
        ax,
        (0.21, 0.16),
        0.17,
        0.23,
        "Differentiable STFT\n0.5-70 Hz",
        facecolor="#F1E7D6",
        fontsize=6.2,
    )
    _draw_box(
        ax,
        (0.46, 0.36),
        0.15,
        0.28,
        "Bidirectional\ncross-attention",
        facecolor="#E9E5F0",
        fontsize=6.2,
    )
    _draw_box(
        ax,
        (0.68, 0.36),
        0.14,
        0.28,
        "Gated fusion\nTCN + Transformer",
        facecolor="#E5EFE7",
        fontsize=6.2,
    )
    _draw_box(
        ax,
        (0.88, 0.57),
        0.11,
        0.2,
        "P(seizure)",
        facecolor="#F2DFD9",
        edgecolor=SEIZURE_COLOR,
        fontsize=6.2,
    )
    _draw_box(
        ax,
        (0.88, 0.22),
        0.11,
        0.22,
        "14 EEG stats\n(training only)",
        facecolor="#F0E4ED",
        edgecolor=MODEL_COLORS["physio_stats_aux"],
        fontsize=5.8,
    )
    _draw_arrow(ax, (0.14, 0.5), (0.21, 0.72))
    _draw_arrow(ax, (0.14, 0.5), (0.21, 0.27))
    _draw_arrow(ax, (0.38, 0.72), (0.46, 0.55))
    _draw_arrow(ax, (0.38, 0.27), (0.46, 0.45))
    _draw_arrow(ax, (0.61, 0.5), (0.68, 0.5))
    _draw_arrow(ax, (0.82, 0.53), (0.88, 0.67))
    _draw_arrow(ax, (0.82, 0.47), (0.88, 0.33))
    ax.text(
        0.01,
        0.96,
        "Physiology-informed binary seizure model",
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=7.4,
    )
    ax.text(
        0.01,
        0.02,
        "Training-nonseizure median/MAD scaling preserves relative amplitude; invalid channels are masked before both branches.",
        ha="left",
        va="bottom",
        fontsize=5.6,
        color=NEUTRAL_MID,
    )


def _line_noise_compact_panel(
    ax: mpl.axes.Axes, line_noise: pd.DataFrame
) -> None:
    before = line_noise["raw_relative_power_48_52"].to_numpy(float)
    after = line_noise["filtered_relative_power_48_52"].to_numpy(float)
    box = ax.boxplot(
        [before, after],
        positions=(0, 1),
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": NEUTRAL_DARK, "linewidth": 1.0},
        boxprops={"linewidth": 0.6},
        whiskerprops={"linewidth": 0.6},
        capprops={"linewidth": 0.6},
    )
    for patch, color in zip(box["boxes"], ("#C7CDD4", "#85BBAA")):
        patch.set_facecolor(color)
    ax.set_yscale("log")
    ax.set_xticks((0, 1), ("Before", "After"))
    ax.set_ylabel("48-52 Hz relative power")
    reduction = 1.0 - float(np.median(after) / np.median(before))
    ax.set_title(f"Notch QC: median -{reduction * 100:.1f}% (n={len(line_noise)})")
    _style_axis(ax, grid="y")


def _main_figure_data_model(
    data: MixedBinaryFigureData,
    line_noise: pd.DataFrame,
    representative_rows: pd.DataFrame,
) -> mpl.figure.Figure:
    with mpl.rc_context(_publication_rc(compact=True)):
        figure = plt.figure(figsize=(7.4, 6.0), constrained_layout=True)
        grid = figure.add_gridspec(3, 3, height_ratios=(0.58, 1.0, 1.0))
        workflow = figure.add_subplot(grid[0, :])
        architecture = figure.add_subplot(grid[1, :])
        background_axis = figure.add_subplot(grid[2, 0])
        seizure_axis = figure.add_subplot(grid[2, 1])
        line_axis = figure.add_subplot(grid[2, 2])
        _draw_compact_workflow(workflow)
        _draw_compact_architecture(architecture)
        for axis, label, title in (
            (background_axis, 0, "Representative non-seizure input"),
            (seizure_axis, 1, "Representative seizure input"),
        ):
            row = representative_rows.loc[
                representative_rows["animal_id"].eq("E50")
                & representative_rows["label"].eq(label)
            ].iloc[0]
            _, filtered = _normalized_signal_pair(row, data.training_baseline)
            _plot_two_channel_trace(
                axis,
                filtered,
                float(row["sampling_rate_hz"]),
                title=title,
                valid=(True, True),
            )
        _line_noise_compact_panel(line_axis, line_noise)
        figure.text(
            0.5,
            -0.005,
            "Representative windows are selected deterministically by within-class signal-QC median, not by model score; amplitude is in a.u.",
            ha="center",
            fontsize=5.7,
            color=NEUTRAL_MID,
        )
        _panel_labels((workflow, architecture, background_axis, seizure_axis, line_axis))
        return figure


def _draw_single_confusion(
    ax: mpl.axes.Axes, data: MixedBinaryFigureData, model: str
) -> None:
    values = data.metrics[model]["test"]["window"]
    matrix = np.asarray(
        [[values["tn"], values["fp"]], [values["fn"], values["tp"]]], dtype=int
    )
    fractions = matrix / matrix.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list(
        "main_confusion", ("#F2F5F7", "#7EA9C8", "#174A6E")
    )
    ax.imshow(fractions, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    for row in range(2):
        for column in range(2):
            fraction = float(fractions[row, column])
            ax.text(
                column,
                row,
                f"{matrix[row, column]:,}\n{fraction * 100:.1f}%",
                ha="center",
                va="center",
                color="white" if fraction > 0.55 else NEUTRAL_DARK,
                fontweight="bold" if row == column else "normal",
                fontsize=6.8,
            )
    ax.set_xticks((0, 1), ("Non-seizure", "Seizure"))
    ax.set_yticks((0, 1), ("Non-seizure", "Seizure"))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"Recommended Physio checkpoint\nvalidation-selected threshold={float(data.metrics[model]['threshold']):.3f}"
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _main_performance_interval_panel(
    ax: mpl.axes.Axes, data: MixedBinaryFigureData
) -> None:
    metrics = ("auprc", "auroc", "balanced_accuracy", "mcc")
    y_base = np.arange(len(metrics))[::-1]
    offsets = np.linspace(-0.24, 0.24, len(PRIMARY_MODELS))
    for model_index, model in enumerate(PRIMARY_MODELS):
        for metric_index, metric in enumerate(metrics):
            summary = data.bootstrap["performance"][model][metric]
            estimate = float(summary["observed"])
            lower = float(summary["ci95_lower"])
            upper = float(summary["ci95_upper"])
            position = y_base[metric_index] + offsets[model_index]
            ax.errorbar(
                estimate,
                position,
                xerr=np.asarray([[estimate - lower], [upper - estimate]]),
                fmt=MODEL_MARKERS[model],
                color=MODEL_COLORS[model],
                capsize=1.5,
                linewidth=0.8,
                markersize=3.2,
            )
    ax.set_yticks(y_base, ("AUPRC", "AUROC", "Balanced accuracy", "MCC"))
    ax.set_xlim(0.48, 0.99)
    ax.set_xlabel("Test score (95% independent-group bootstrap CI)")
    ax.set_title("Performance with sampling uncertainty")
    ax.text(
        0.02,
        0.02,
        (
            f"Always non-seizure: accuracy {(1-data.test_prevalence)*100:.2f}%, "
            f"sensitivity 0, BA 0.50, AUPRC {data.test_prevalence:.3f}"
        ),
        transform=ax.transAxes,
        fontsize=5.3,
        color=NEUTRAL_MID,
        va="bottom",
    )
    _style_axis(ax, grid="x")


def _main_figure_binary_detection(data: MixedBinaryFigureData) -> mpl.figure.Figure:
    with mpl.rc_context(_publication_rc(compact=True)):
        figure = plt.figure(figsize=(7.4, 5.2), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, width_ratios=(1.32, 1.0))
        pr_axis = figure.add_subplot(grid[:, 0])
        confusion_axis = figure.add_subplot(grid[0, 1])
        interval_axis = figure.add_subplot(grid[1, 1])
        for model in PRIMARY_MODELS:
            test = data.predictions[model].loc[
                data.predictions[model]["split"].eq("test")
            ]
            labels = test["label"].to_numpy(np.int8)
            scores = test["seizure_probability"].to_numpy(float)
            precision, recall, _ = precision_recall_curve(labels, scores)
            pr_axis.plot(
                recall,
                precision,
                color=MODEL_COLORS[model],
                linestyle=MODEL_LINESTYLES[model],
                label=f"{MODEL_SHORT_LABELS[model]}  AUPRC={average_precision_score(labels, scores):.3f}",
            )
        pr_axis.axhline(
            data.test_prevalence,
            color=NEUTRAL_MID,
            linestyle="--",
            linewidth=0.8,
            label=f"Random ranking={data.test_prevalence:.3f}",
        )
        pr_axis.set_xlim(0, 1)
        pr_axis.set_ylim(0, 1.02)
        pr_axis.set_xlabel("Recall")
        pr_axis.set_ylabel("Precision")
        pr_axis.set_title("Seizure ranking beyond class prevalence")
        pr_axis.legend(loc="lower left")
        _style_axis(pr_axis, grid="both")
        _draw_single_confusion(confusion_axis, data, "physio_no_aux")
        _main_performance_interval_panel(interval_axis, data)
        _panel_labels((pr_axis, confusion_axis, interval_axis))
        return figure


def _main_figure_mouse_heterogeneity(data: MixedBinaryFigureData) -> mpl.figure.Figure:
    with mpl.rc_context(_publication_rc(compact=True)):
        figure = plt.figure(figsize=(7.4, 4.6), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.05))
        heatmap_axis = figure.add_subplot(grid[0, :])
        sensitivity_axis = figure.add_subplot(grid[1, 0])
        support_axis = figure.add_subplot(grid[1, 1])
        test = data.per_mouse.loc[
            data.per_mouse["split"].eq("test")
            & data.per_mouse["model"].isin(PRIMARY_MODELS)
        ]
        matrix = (
            test.pivot(index="model", columns="animal_id", values="auprc")
            .reindex(index=PRIMARY_MODELS, columns=ANIMALS)
            .to_numpy(float)
        )
        image = _annotated_heatmap(
            heatmap_axis,
            matrix,
            x_labels=("E50", "E76", "E77", "E79", "E81*", "E82*"),
            y_labels=[MODEL_SHORT_LABELS[model] for model in PRIMARY_MODELS],
            title="Mouse-specific AUPRC exposes heterogeneous difficulty",
        )
        colorbar = figure.colorbar(image, ax=heatmap_axis, fraction=0.025, pad=0.015)
        colorbar.set_label("AUPRC")

        x = np.arange(len(ANIMALS))
        for model in PRIMARY_MODELS:
            values = (
                test.loc[test["model"].eq(model)]
                .set_index("animal_id")
                .reindex(ANIMALS)["sensitivity"]
                .to_numpy(float)
            )
            sensitivity_axis.plot(
                x,
                values,
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                linestyle=MODEL_LINESTYLES[model],
                label=MODEL_SHORT_LABELS[model],
            )
        sensitivity_axis.set_xticks(x, ("E50", "E76", "E77", "E79", "E81*", "E82*"))
        sensitivity_axis.set_ylim(-0.02, 1.02)
        sensitivity_axis.set_ylabel("Sensitivity")
        sensitivity_axis.set_title("Validation-threshold sensitivity")
        sensitivity_axis.legend(ncol=2, loc="lower left")
        _style_axis(sensitivity_axis, grid="y")

        support = (
            data.split_summary.loc[data.split_summary["split"].eq("test")]
            .set_index("animal_id")
            .reindex(ANIMALS)
        )
        windows = support["seizure_windows"].to_numpy(float)
        groups = support["seizure_groups"].to_numpy(float)
        support_axis.bar(
            x - 0.18,
            windows,
            width=0.36,
            color="#D6A193",
            edgecolor=SEIZURE_COLOR,
            linewidth=0.6,
            label="Positive windows",
        )
        support_axis.bar(
            x + 0.18,
            groups,
            width=0.36,
            color="#B8BEC5",
            edgecolor=NEUTRAL_MID,
            linewidth=0.6,
            label="Independent positive groups",
        )
        support_axis.set_yscale("log")
        support_axis.set_xticks(x, ("E50", "E76", "E77", "E79", "E81*", "E82*"))
        support_axis.set_ylabel("Test support (log count)")
        support_axis.set_title("Evidence supporting each mouse estimate")
        support_axis.legend(loc="upper right")
        support_axis.text(4.18, 1.4, "n=1", ha="center", fontsize=5.6)
        support_axis.text(5.18, 1.4, "n=1", ha="center", fontsize=5.6)
        _style_axis(support_axis, grid="y")
        figure.text(
            0.5,
            -0.005,
            "* E81 and E82 each contain one positive test event complex (64 positive windows); their mouse-specific estimates are unstable.",
            ha="center",
            fontsize=5.7,
            color=NEUTRAL_MID,
        )
        _panel_labels((heatmap_axis, sensitivity_axis, support_axis))
        return figure


def _seed_difference_panel(ax: mpl.axes.Axes, data: MixedBinaryFigureData) -> None:
    metrics = (
        ("test_auprc", "AUPRC"),
        ("test_balanced_accuracy", "Balanced accuracy"),
        ("test_mcc", "MCC"),
        ("macro_mouse_auprc", "Macro-mouse AUPRC"),
    )
    frame = data.seed_metrics
    pivot = frame.pivot(index="seed", columns="model")
    y = np.arange(len(metrics))[::-1]
    for position, (metric, _) in zip(y, metrics):
        differences = (
            pivot[(metric, "physio_stats_aux")]
            - pivot[(metric, "physio_no_aux")]
        ).to_numpy(float)
        ax.scatter(
            differences,
            np.full(len(differences), position),
            facecolor="white",
            edgecolor=MODEL_COLORS["physio_stats_aux"],
            linewidth=0.8,
            s=20,
            zorder=3,
        )
        ax.scatter(
            [float(np.mean(differences))],
            [position],
            color=MODEL_COLORS["physio_stats_aux"],
            marker="D",
            s=24,
            zorder=4,
        )
    ax.axvline(0, color=NEUTRAL_MID, linestyle="--", linewidth=0.8)
    ax.set_yticks(y, [label for _, label in metrics])
    ax.set_xlabel("Auxiliary minus no-aux (three paired seeds)")
    ax.set_title("Training-seed stability of the auxiliary effect")
    _style_axis(ax, grid="x")


def _main_figure_design_evidence(data: MixedBinaryFigureData) -> mpl.figure.Figure:
    with mpl.rc_context(_publication_rc(compact=True)):
        figure, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), constrained_layout=True)
        _contrast_forest_panel(
            axes[0, 0],
            data,
            "physio_no_aux_minus_eegwavenet",
            title="Physio encoder minus EEGWaveNet",
        )
        _contrast_forest_panel(
            axes[0, 1],
            data,
            "physio_stats_aux_minus_physio_no_aux",
            title="Statistics auxiliary loss minus no-aux",
        )
        _contrast_forest_panel(
            axes[1, 0],
            data,
            "physio_no_aux_mouse_balanced_minus_physio_no_aux",
            title="Mouse-balanced loss minus base Physio",
        )
        _seed_difference_panel(axes[1, 1], data)
        figure.text(
            0.5,
            -0.005,
            "Intervals are paired independent-group bootstrap CIs. Three-seed points are descriptive repeats, not confidence intervals or significance tests.",
            ha="center",
            fontsize=5.7,
            color=NEUTRAL_MID,
        )
        _panel_labels(list(axes.ravel()))
        return figure


def _write_source_data(
    data: MixedBinaryFigureData,
    source_dir: Path,
    line_noise: pd.DataFrame,
    representative_rows: pd.DataFrame,
) -> dict[str, Path]:
    source_dir.mkdir(parents=True, exist_ok=True)
    histories = pd.concat(
        [frame.assign(model=model) for model, frame in data.histories.items()],
        ignore_index=True,
    )
    source_tables = {
        "split_summary": data.split_summary,
        "group_summary": data.group_summary,
        "line_noise_qc": line_noise,
        "representative_signals": _representative_source_data(
            data, representative_rows
        ),
        "overall_metrics": data.overall,
        "per_mouse_metrics": data.per_mouse,
        "curve_points": _curve_source_data(data),
        "confusion_matrices": _confusion_source_data(data),
        "training_histories": histories,
        "bootstrap_summary": _flatten_bootstrap(data.bootstrap),
        "seed_metrics": data.seed_metrics,
        "seed_per_mouse_metrics": data.seed_per_mouse,
        "seed_differences": data.seed_differences,
    }
    outputs: dict[str, Path] = {}
    for name, frame in source_tables.items():
        path = source_dir / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        outputs[name] = path
    return outputs


def _file_record(path: Path, relative_root: Path) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(relative_root.resolve())
        relative_path = relative.as_posix()
    except ValueError:
        relative_path = None
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "relative_path": relative_path,
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }
    if path.suffix.lower() == ".png":
        image = plt.imread(path)
        record["width_px"] = int(image.shape[1])
        record["height_px"] = int(image.shape[0])
        record["channel_count"] = int(image.shape[2]) if image.ndim == 3 else 1
        record["pixel_standard_deviation"] = float(np.std(image))
    return record


def _verify_figure_outputs(
    figure_outputs: Mapping[str, Sequence[Path]],
) -> None:
    expected_stems = {stem for stem, _ in ALL_FIGURE_SPECS}
    if set(figure_outputs) != expected_stems:
        raise ValueError("figure output set does not match the declared suite")
    for stem, paths in figure_outputs.items():
        suffixes = {path.suffix.lower() for path in paths}
        if suffixes != {".png", ".pdf", ".svg"}:
            raise ValueError(f"{stem} is missing an export format")
        for path in paths:
            if not path.is_file() or path.stat().st_size < 1024:
                raise ValueError(f"empty or missing figure output: {path}")
        png = next(path for path in paths if path.suffix.lower() == ".png")
        image = plt.imread(png)
        if image.shape[0] < 600 or image.shape[1] < 900:
            raise ValueError(f"PNG is too small for publication review: {png}")
        if float(np.std(image)) < 0.01:
            raise ValueError(f"PNG appears blank: {png}")


def _render_figure_index(
    figure_outputs: Mapping[str, Sequence[Path]],
    data: MixedBinaryFigureData,
    line_noise: pd.DataFrame,
) -> str:
    captions = {
        "main_fig01_data_model": (
            "从冻结分组、连续信号滤波、训练基线归一化到时域/频域双分支模型，形成一条完整的数据完整性与模型输入证据链。"
            "代表窗按类内质量指标中位数确定，不按模型得分挑选；工频质控由原始缓存现场重算。"
        ),
        "main_fig02_binary_detection": (
            f"主结论图：测试集随机排序 AUPRC 基线为 {data.test_prevalence:.3f}，Physio 无辅助达到 0.793。"
            "推荐 checkpoint 的 sensitivity 为 74.8%、specificity 为 95.0%；永远预测非发作虽有 87.05% accuracy，但 sensitivity=0、BA=0.50，因此不能用 accuracy 单独判断模型。"
        ),
        "main_fig03_mouse_heterogeneity": (
            "逐鼠 AUPRC、固定阈值 sensitivity 与阳性事件支持量放在同一图中，直接说明总体结果主要由 E50/E76/E77 支撑。"
            "E81/E82 各只有一个阳性测试事件，低性能与高不确定性必须同时报告。"
        ),
        "main_fig04_design_evidence": (
            "四个面板只回答模型设计问题：生理编码器相对 EEGWaveNet 的增益、统计辅助头是否稳定、逐鼠加权的取舍，以及辅助头在三随机种子中的方向一致性。"
            "结果支持 Physio 无辅助作为默认模型，不支持把统计辅助约束作为稳定增益。"
        ),
        "fig00_study_model_overview": (
            "六只小鼠先汇总，再以独立发作事件复合组和连续背景块为单位进行冻结划分。"
            "生理融合模型同时编码多尺度时域波形和可微频谱，统计回归头只在训练期作为可选约束。"
        ),
        "fig01_data_split_qc": (
            "展示窗口、独立组和各鼠贡献。E77 的 CH7 全程屏蔽；其余通道有效。"
            f"对 {len(line_noise)} 个有效通道窗系统抽样重算后，48-52 Hz 相对功率中位数从 "
            f"{line_noise['raw_relative_power_48_52'].median():.6f} 降至 "
            f"{line_noise['filtered_relative_power_48_52'].median():.6f}。"
        ),
        "fig02_preprocessing_examples": (
            "E50 测试集中按类内 jump-z 中位数确定代表窗，不按模型得分挑图。"
            "原始波形仅为显示而减去各窗中位数，幅值均为相对训练基线的 a.u.；E77 原始 CH7 是硬件无效信号，滤波视图中严格屏蔽。"
        ),
        "fig03_overall_performance": (
            "四个主模型的点估计和 95% 区间来自按小鼠与二分类标签分层、以独立组为单位的配对 bootstrap（1,000 次）。"
            "组级横轴是受限窗口视图中的背景组阳性比例，不是每小时误报。"
        ),
        "fig04_discrimination_curves": (
            f"测试集发作窗比例为 {data.test_prevalence:.4f}，即 PR 随机排序基线。"
            "PR/ROC 使用连续概率；校准和分数分布均为窗口级描述。"
        ),
        "fig05_confusion_matrices": (
            "测试集硬分类使用各模型在验证集上按 MCC 选择的固定阈值。每个单元格同时给出计数和按真实类别归一化百分比。"
        ),
        "fig06_per_mouse_performance": (
            "整体性能并不代表每只鼠相同。E81 和 E82 测试集各只有 1 个阳性事件复合组、64 个阳性窗，逐鼠 AUPRC 和 sensitivity 的不确定性很高。"
        ),
        "fig07_per_mouse_pr_curves": (
            "六只鼠分别绘制 PR 曲线并给出各自 prevalence 基线。该图直接显示 E50/E76/E77 与 E79/E81/E82 之间的难度差异。"
        ),
        "fig08_training_histories": (
            "checkpoint 由验证 AUPRC 选择，竖虚线和实心点使用 metrics.json 中的最终 best_epoch；history.csv 的累计 best 标志未被误当成唯一最终点。"
        ),
        "fig09_auxiliary_ablation": (
            "单种子中统计辅助头提高 pooled balanced accuracy，但 AUPRC/MCC 区间跨零，macro-mouse AUPRC 下降。三种子 AUPRC 提升方向不稳定；无辅助统计损失仅是诊断量，不参与优化。"
        ),
        "fig10_mouse_balanced_sensitivity": (
            "逐鼠加权提高 macro-mouse balanced accuracy 及 E79/E82 sensitivity，但降低 pooled AUPRC 和 macro-mouse AUPRC，且未解决 E81。"
        ),
        "fig11_multiseed_stability": (
            "三个配对随机种子仅作描述性稳定性分析；粗线为均值，误差条为样本 SD，不是 95% CI，也不进行显著性检验。"
        ),
        "fig12_bootstrap_forest": (
            "左列汇总六项性能的独立组 bootstrap 95% 区间，右列汇总所有预设配对对比并以 0 为参考线。bootstrap P(差值>0) 未当作传统 p 值或显著性星号。"
        ),
        "fig13_results_overview": (
            f"总览图专门区分类别不平衡造成的 accuracy 假象：永远预测非发作可得 {(1-data.test_prevalence)*100:.2f}% accuracy，"
            "但 sensitivity=0、balanced accuracy=0.5、AUPRC 等于 prevalence；Physio 模型在 sensitivity、BA 和 AUPRC 上获得实质提升，同时仍存在跨鼠异质性。"
        ),
    }
    captions["fig08_training_histories"] = (
        "训练诊断只保留两条 Physio 配置。无辅助模型在第 9 轮后训练损失继续下降，但验证损失由 0.221 升至 0.240、验证 AUPRC 由 0.817 降至 0.805；"
        "辅助模型在第 21 轮后验证损失由 0.256 升至 0.317、验证 AUPRC 由 0.816 降至 0.800。灰区表示最佳 checkpoint 后的轮次，说明早停阻止了继续过拟合。"
    )
    lines = [
        "# 混合小鼠癫痫二分类图件索引",
        "",
        "全部图件使用同一 Python/Matplotlib 后端、白底紧凑多面板和色盲友好配色。PNG 为 300 dpi 审阅版，PDF/SVG 为可编辑矢量版；图内坐标使用英文，解释和限制写在本索引中。",
        "",
        "## 统计与解释边界",
        "",
        "- 主任务只有 seizure 与 non-seizure 两类，既往 21 种形态未参与主标签、划分或主评价。",
        "- 训练/验证/测试均可包含六只已见小鼠；结论是已见小鼠内独立事件/背景块泛化，不是新小鼠泛化。",
        "- 95% 区间来自独立组 bootstrap；三随机种子误差条为样本 SD。",
        "- 当前每组最多抽取 64 个窗口，不能据此报告 false alarms/hour、检测延迟或完整连续事件敏感度。",
        "- ADC 到微伏的标定未确认，所有波形幅值只写 a.u.。现有基础 QC 未覆盖全部运动/接触伪迹。",
        "",
    ]
    lines.extend(["## 主文图：四条紧凑证据链", ""])
    title_lookup = dict(ALL_FIGURE_SPECS)
    for index, (stem, _) in enumerate(MAIN_FIGURE_SPECS, start=1):
        paths = {path.suffix.lower(): path for path in figure_outputs[stem]}
        png = paths[".png"].name
        pdf = paths[".pdf"].name
        svg = paths[".svg"].name
        lines.extend(
            [
                f"## Main Figure {index} | {title_lookup[stem]}",
                "",
                captions[stem],
                "",
                f"[PNG]({png}) | [PDF]({pdf}) | [SVG]({svg})",
                "",
                f"![{title_lookup[stem]}]({png})",
                "",
            ]
        )
    lines.extend(
        [
            "## 补充与审计图",
            "",
            "以下图件保留完整 QC、曲线、逐鼠和统计审计信息，不建议全部塞入主文。训练诊断图只保留两条 Physio 配置，并明确标出最佳 checkpoint 后的过拟合区间。",
            "",
        ]
    )
    for index, (stem, _) in enumerate(FIGURE_SPECS, start=1):
        paths = {path.suffix.lower(): path for path in figure_outputs[stem]}
        png = paths[".png"].name
        pdf = paths[".pdf"].name
        svg = paths[".svg"].name
        lines.extend(
            [
                f"### Supplementary Figure S{index:02d} | {title_lookup[stem]}",
                "",
                captions[stem],
                "",
                f"[PNG]({png}) | [PDF]({pdf}) | [SVG]({svg})",
                "",
                f"![{title_lookup[stem]}]({png})",
                "",
            ]
        )
    lines.extend(
        [
            "## 复现",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\build_epilepsy_binary_visualizations.py",
            "```",
            "",
            "机器可读清单见 `figure_manifest.json`、`figure_manifest.csv` 和 `source_data_manifest.csv`；逐图源数据位于 `source_data/`。",
            "",
        ]
    )
    return "\n".join(lines)


def _render_qa_notes(
    data: MixedBinaryFigureData,
    figure_outputs: Mapping[str, Sequence[Path]],
    line_noise: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# 癫痫二分类图件 QA 记录",
            "",
            f"- 图件数量：{len(figure_outputs)} 张，每张均含 PNG/PDF/SVG，共 {len(figure_outputs) * 3} 个图文件。",
            "- 后端：仅使用 Python/Matplotlib；SVG 文本保持可编辑，PDF 使用 TrueType fonttype 42。",
            "- 画布：主图宽 7.4 inch（约 188 mm，双栏）；正文字号不低于 5.3 pt；PNG 为 300 dpi 审阅版。",
            "- TIFF：未额外导出。所有定量图均有 PDF/SVG 矢量版，PNG 只作快速审阅；正式投稿若期刊强制 TIFF，可从同一 Python 后端追加 600 dpi TIFF。",
            "- 配色：固定 Okabe-Ito 模型颜色，并同时使用 marker/线型；未使用 rainbow/jet。",
            "- 数据完整性：六只鼠、八个 session、3,371 个独立组；跨集合共享组和重叠原始窗均为 0。",
            f"- 工频 QC：从 512 个等距窗口重新读取原始/滤波缓存；屏蔽 E77 CH7 后得到 n={len(line_noise)} 个有效通道窗。",
            "- 代表信号：按 E50 测试集类内 jump-z 中位数确定，不依据模型得分；原始显示仅减去窗内中位数，滤波信号按训练非发作基线缩放，单位为 a.u.。",
            "- 统计区间：性能和差值区间均来自按小鼠与二分类标签分层、以独立事件复合组/背景块为单位的 1,000 次配对 bootstrap。",
            "- 三随机种子：只报告配对点、均值和样本 SD；未进行显著性检验或添加星号。",
            "- 训练诊断：Physio 无辅助最佳轮次为 9，统计辅助版为 21；灰色区明确标出最佳 checkpoint 之后的过拟合轮次。",
            "- E81/E82：测试集各只有 1 个阳性事件复合组（64 个阳性窗），已在主图和图注明示。",
            "- 连续监测边界：未报告 false alarms/hour、检测延迟或完整连续事件敏感度。",
            "- 视觉验收：已逐张检查主图、QC、波形、曲线、混淆矩阵、逐鼠图、消融图与 bootstrap 森林图；修复了柱顶说明、散点标签、低事件注释和窄轴科学计数法重叠。",
            "",
        ]
    )


def generate_mixed_binary_visualizations(
    project_root: str | Path,
    output_dir: str | Path | None = None,
    *,
    dpi: int = 300,
) -> dict[str, Any]:
    """Generate, source, index, and verify the complete figure suite."""

    if dpi < 300:
        raise ValueError("publication PNG exports require dpi >= 300")
    data = load_mixed_binary_figure_data(project_root, output_dir)
    destination = data.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    source_dir = destination / "source_data"
    line_noise = _compute_line_noise_qc(data.metadata)
    representative_rows = _representative_signal_rows(data.metadata)
    source_paths = _write_source_data(
        data, source_dir, line_noise, representative_rows
    )

    builders: dict[str, Any] = {
        "main_fig01_data_model": lambda: _main_figure_data_model(
            data, line_noise, representative_rows
        ),
        "main_fig02_binary_detection": lambda: _main_figure_binary_detection(data),
        "main_fig03_mouse_heterogeneity": lambda: _main_figure_mouse_heterogeneity(data),
        "main_fig04_design_evidence": lambda: _main_figure_design_evidence(data),
        "fig00_study_model_overview": lambda: _figure_study_model_overview(data),
        "fig01_data_split_qc": lambda: _figure_data_split_qc(data, line_noise),
        "fig02_preprocessing_examples": lambda: _figure_preprocessing_examples(
            data, representative_rows
        ),
        "fig03_overall_performance": lambda: _figure_overall_performance(data),
        "fig04_discrimination_curves": lambda: _figure_discrimination_curves(data),
        "fig05_confusion_matrices": lambda: _figure_confusion_matrices(data),
        "fig06_per_mouse_performance": lambda: _figure_per_mouse_performance(data),
        "fig07_per_mouse_pr_curves": lambda: _figure_per_mouse_pr_curves(data),
        "fig08_training_histories": lambda: _figure_training_histories(data),
        "fig09_auxiliary_ablation": lambda: _figure_auxiliary_ablation(data),
        "fig10_mouse_balanced_sensitivity": lambda: _figure_mouse_balanced_sensitivity(data),
        "fig11_multiseed_stability": lambda: _figure_multiseed_stability(data),
        "fig12_bootstrap_forest": lambda: _figure_bootstrap_forest(data),
        "fig13_results_overview": lambda: _figure_results_overview(data),
    }
    figure_outputs: dict[str, list[Path]] = {}
    for stem, _ in ALL_FIGURE_SPECS:
        figure_outputs[stem] = _export_figure(
            builders[stem](), destination / stem, dpi
        )
    _verify_figure_outputs(figure_outputs)

    index_path = destination / "FIGURE_INDEX_zh.md"
    index_path.write_text(
        _render_figure_index(figure_outputs, data, line_noise), encoding="utf-8"
    )
    qa_path = destination / "FIGURE_QA_zh.md"
    qa_path.write_text(
        _render_qa_notes(data, figure_outputs, line_noise), encoding="utf-8"
    )

    figure_rows: list[dict[str, Any]] = []
    title_lookup = dict(ALL_FIGURE_SPECS)
    for stem, paths in figure_outputs.items():
        for path in paths:
            record = _file_record(path, destination)
            figure_rows.append(
                {
                    "figure_id": stem,
                    "title": title_lookup[stem],
                    "format": path.suffix.lower().lstrip("."),
                    **record,
                }
            )
    figure_manifest_csv = destination / "figure_manifest.csv"
    pd.DataFrame(figure_rows).to_csv(
        figure_manifest_csv, index=False, encoding="utf-8-sig"
    )

    source_rows = [
        {"source_table": name, **_file_record(path, destination)}
        for name, path in source_paths.items()
    ]
    source_manifest_csv = destination / "source_data_manifest.csv"
    pd.DataFrame(source_rows).to_csv(
        source_manifest_csv, index=False, encoding="utf-8-sig"
    )

    input_records = []
    for path in data.input_paths:
        if path.is_file():
            input_records.append(_file_record(path, data.project_root))
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": "pooled_mouse_binary_seizure_classification",
        "conclusion_scope": (
            "independent event-complex/background-block generalization within six observed mice"
        ),
        "protocol": {
            "classes": ["non-seizure", "seizure"],
            "seizure_subtype_used": False,
            "split_unit": "independent_event_complex_or_background_block",
            "cross_split_group_count": 0,
            "cross_split_overlapping_raw_windows": 0,
            "continuous_monitoring_metrics_valid": False,
            "amplitude_unit": "a.u.",
        },
        "statistics": {
            "performance_interval": data.bootstrap["method"],
            "bootstrap_repetitions": int(data.bootstrap["repetitions"]),
            "bootstrap_resampling_unit": data.bootstrap["resampling_unit"],
            "training_seed_repeat_summary": "paired descriptive mean and sample SD; n=3",
        },
        "style": {
            "target": "Nature/Cell and EEG/neural-engineering conference style",
            "backend": "Python/Matplotlib",
            "palette": "Okabe-Ito with neutral supporting tones",
            "font": "Arial/DejaVu Sans",
            "editable_svg_text": True,
            "pdf_fonttype": 42,
            "png_dpi": dpi,
            "formats": ["png", "pdf", "svg"],
        },
        "figures": figure_rows,
        "source_data": source_rows,
        "inputs": input_records,
        "index": _file_record(index_path, destination),
        "qa_notes": _file_record(qa_path, destination),
    }
    manifest_path = destination / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return {
        "output_dir": destination,
        "figure_outputs": figure_outputs,
        "figure_index": index_path,
        "qa_notes": qa_path,
        "figure_manifest_json": manifest_path,
        "figure_manifest_csv": figure_manifest_csv,
        "source_data_manifest": source_manifest_csv,
        "source_data": source_paths,
    }
