from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.0,
    }
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, Rectangle
from sklearn.metrics import precision_recall_curve


MODEL_ORDER = ("eegwavenet_sc", "physio_sequence_no_aux", "physio_sequence_stats")
MODEL_LABELS = {
    "eegwavenet_sc": "EEGWaveNet-SC",
    "physio_sequence_no_aux": "Physio-Sequence",
    "physio_sequence_stats": "Physio + statistics",
}
MODEL_COLORS = {
    "eegwavenet_sc": "#59636F",
    "physio_sequence_no_aux": "#2F6B8A",
    "physio_sequence_stats": "#C56B3D",
}
ANIMAL_COLORS = {"E50": "#1F5A75", "E76": "#B05A36", "E77": "#4C7A52"}
# Keep the historical colours above, then use a deterministic, colour-blind
# friendly extension for animals introduced by a combined run.
ANIMAL_COLOR_CYCLE = (
    "#1F5A75",
    "#B05A36",
    "#4C7A52",
    "#CC79A7",
    "#D55E00",
    "#56B4E9",
    "#7A6A3A",
    "#6C6C6C",
)
TRUE_COLOR = "#1E5B8C"
PRED_COLOR = "#C45A32"
PROBABILITY_COLOR = "#2D6F68"
MM_PER_INCH = 25.4
FIGURE_WIDTH_IN = 7.2047  # 183 mm, standard double-column width.


def _animal_colors(animals: Iterable[str]) -> dict[str, str]:
    """Return stable colours while preserving the legacy E50/E76/E77 map."""

    names = sorted({str(animal) for animal in animals if pd.notna(animal)})
    colors = dict(ANIMAL_COLORS)
    unused = [color for color in ANIMAL_COLOR_CYCLE if color not in colors.values()]
    for animal in names:
        if animal in colors:
            continue
        if unused:
            colors[animal] = unused.pop(0)
        else:
            colors[animal] = ANIMAL_COLOR_CYCLE[len(colors) % len(ANIMAL_COLOR_CYCLE)]
    return colors


@dataclass
class FigureInputs:
    run_root: Path
    protocol_dir: Path
    output_dir: Path

    @property
    def source_dir(self) -> Path:
        path = self.output_dir / "source_data"
        path.mkdir(parents=True, exist_ok=True)
        return path


class TraceStore:
    def __init__(self, protocol_dir: Path) -> None:
        self.full_index = pd.read_parquet(protocol_dir / "full_index.parquet")
        self.tracks = {
            tuple(map(str, key)): group.sort_values("start_time_ms", kind="stable")
            for key, group in self.full_index.groupby(
                ["animal_id", "session_id", "channel_name"], sort=False
            )
        }
        self.track_bounds = {
            key: (
                track["start_time_ms"].to_numpy(float),
                track["end_time_ms"].to_numpy(float),
            )
            for key, track in self.tracks.items()
        }
        baseline = json.loads(
            (protocol_dir / "training_baseline.json").read_text(encoding="utf-8")
        )
        self.stats = {
            (str(row["animal_id"]), str(row["channel_name"])): (
                float(row["center"]), float(row["scale"])
            )
            for row in baseline["channels"]
        }
        self.arrays: dict[str, np.ndarray] = {}

    def load(
        self,
        animal: str,
        session: str,
        channel: str,
        start_ms: float,
        end_ms: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        track_key = (str(animal), str(session), str(channel))
        track = self.tracks.get(track_key)
        if track is None or track.empty:
            raise ValueError(f"missing trace {animal}/{session}/{channel}")
        starts, ends = self.track_bounds[track_key]
        left = int(np.searchsorted(ends, start_ms, side="left"))
        right = int(np.searchsorted(starts, end_ms, side="right"))
        selected = track.iloc[left:right]
        return self._load_selected_rows(
            track_key,
            selected,
            start_ms,
            end_ms,
            require_one_physical_run=False,
        )

    def load_decoder_segment(
        self,
        decoder_segment: pd.DataFrame,
        start_ms: float,
        end_ms: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load waveform samples only from one decoded physical/QC segment.

        The continuous atlas receives a probability trace that may omit
        rejected or unreviewed rows.  Selecting raw waveform rows by time
        alone could reintroduce those omitted rows into a figure.  The trace
        ``sample_id`` values are therefore treated as an exact allow-list,
        and the selected rows must still form one physical continuity run.
        """

        required = {
            "animal_id",
            "session_id",
            "channel_name",
            "sample_id",
        }
        missing = sorted(required - set(decoder_segment.columns))
        if missing:
            raise ValueError(
                f"decoder segment is missing waveform identity columns: {missing}"
            )
        if decoder_segment.empty:
            raise ValueError("cannot load an empty decoder segment")
        identity = decoder_segment.loc[:, ["animal_id", "session_id", "channel_name"]]
        keys = {
            tuple(map(str, row))
            for row in identity.drop_duplicates().itertuples(index=False, name=None)
        }
        if len(keys) != 1:
            raise ValueError("decoder segment must contain exactly one channel track")
        track_key = next(iter(keys))
        track = self.tracks.get(track_key)
        if track is None or track.empty:
            raise ValueError(f"missing trace {'/'.join(track_key)}")
        expected_ids = decoder_segment["sample_id"].astype(str)
        if expected_ids.duplicated().any():
            raise ValueError("decoder segment contains duplicate sample_id values")
        allowed_ids = set(expected_ids.tolist())
        selected = track.loc[track["sample_id"].astype(str).isin(allowed_ids)].copy()
        selected_ids = set(selected["sample_id"].astype(str).tolist())
        if selected_ids != allowed_ids or len(selected) != len(allowed_ids):
            missing_ids = sorted(allowed_ids - selected_ids)
            raise ValueError(
                "decoder segment sample IDs do not exactly match the waveform index"
                + (f": missing {missing_ids[:3]}" if missing_ids else "")
            )
        return self._load_selected_rows(
            track_key,
            selected,
            start_ms,
            end_ms,
            require_one_physical_run=True,
        )

    def _load_selected_rows(
        self,
        track_key: tuple[str, str, str],
        selected: pd.DataFrame,
        start_ms: float,
        end_ms: float,
        *,
        require_one_physical_run: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if selected.empty:
            raise ValueError("requested trace interval has no cached windows")
        selected = selected.sort_values("start_time_ms", kind="stable")
        physical_runs = self._physical_run_ids(selected)
        if require_one_physical_run and int(physical_runs[-1]) != int(physical_runs[0]):
            raise ValueError(
                "decoder segment crosses a physical/QC waveform continuity boundary"
            )
        rate = float(selected["sampling_rate_hz"].iloc[0])
        stride_samples = int(round(rate * float(selected["stride_seconds"].iloc[0])))
        waveform_pieces = []
        time_pieces = []
        for _, continuity in selected.groupby(physical_runs, sort=False):
            for position, row in enumerate(continuity.itertuples(index=False)):
                root = str(row.filtered_cache_dir)
                if root not in self.arrays:
                    self.arrays[root] = np.load(Path(root) / "signals.npy", mmap_mode="r")
                values = np.asarray(
                    self.arrays[root][int(row.filtered_local_index), int(row.channel_index)],
                    dtype=np.float32,
                )
                if position == 0:
                    piece = values
                    piece_start = float(row.start_time_ms)
                else:
                    piece = values[-stride_samples:]
                    piece_start = float(row.end_time_ms) - 1000.0 * stride_samples / rate
                waveform_pieces.append(piece)
                time_pieces.append(
                    piece_start + np.arange(len(piece), dtype=float) * 1000.0 / rate
                )
        waveform = np.concatenate(waveform_pieces).astype(np.float32, copy=False)
        time_ms = np.concatenate(time_pieces)
        center, scale = self.stats[(track_key[0], track_key[2])]
        waveform = (waveform - center) / max(scale, 1e-6)
        keep = (time_ms >= start_ms) & (time_ms <= end_ms)
        order = np.argsort(time_ms[keep], kind="stable")
        return time_ms[keep][order], waveform[keep][order]

    @staticmethod
    def _physical_run_ids(selected: pd.DataFrame) -> np.ndarray:
        """Identify source-contiguous waveform runs without label metadata."""

        if selected.empty:
            return np.empty(0, dtype=np.int64)
        continuity = selected["continuity_id"].astype("string")
        same_continuity = continuity.eq(continuity.shift()).fillna(False).to_numpy(bool)
        if "continuity_position" in selected.columns:
            position = pd.to_numeric(selected["continuity_position"], errors="coerce")
            consecutive = (
                position.notna()
                & position.shift().notna()
                & position.eq(position.shift() + 1)
            ).to_numpy(bool)
        else:
            starts = selected["start_time_ms"].to_numpy(float)
            consecutive = np.zeros(len(selected), dtype=bool)
            if len(starts) > 1:
                deltas = np.diff(starts)
                positive = deltas[deltas > 0.0]
                if len(positive):
                    stride = float(np.median(positive))
                    consecutive[1:] = deltas <= stride * 1.5
        run_start = ~(same_continuity & consecutive)
        run_start[0] = True
        return np.cumsum(run_start, dtype=np.int64)


def generate_single_channel_figure_bundle(
    run_root: str | Path,
    protocol_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    inputs = FigureInputs(Path(run_root).resolve(), Path(protocol_dir).resolve(), Path(output_dir).resolve())
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    comparison, pr_curves, boundary, animal_metrics = _collect_model_sources(inputs)
    _plot_primary_summary(inputs, comparison, pr_curves, boundary, animal_metrics)
    _plot_training_diagnostics(inputs)
    representatives = _select_representative_events(inputs)
    _plot_clinical_waveform_evidence(inputs, representatives)
    _plot_whole_recording_overviews(inputs)
    _build_event_atlases(inputs)
    _write_figure_notes(inputs, representatives)
    return inputs.output_dir


def _collect_model_sources(
    inputs: FigureInputs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison_rows = []
    pr_rows = []
    boundary_frames = []
    animal_rows = []
    for model_name in MODEL_ORDER:
        model_root = inputs.run_root / model_name
        window_metrics = json.loads((model_root / "metrics.json").read_text(encoding="utf-8"))
        event_metrics = json.loads(
            (model_root / "events" / "event_metrics.json").read_text(encoding="utf-8")
        )
        test_window = window_metrics["test"]
        test_event = event_metrics["test"]["pooled"]
        predictions = pd.read_parquet(model_root / "predictions.parquet")
        test_predictions = predictions.loc[predictions["split"].astype(str).eq("test")]
        labels = pd.to_numeric(test_predictions["label"], errors="coerce").dropna()
        test_prevalence = float(labels.mean()) if not labels.empty else float("nan")
        comparison_rows.append(
            {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "window_auprc": test_window["pooled"]["auprc"],
                "window_auroc": test_window["pooled"]["auroc"],
                "macro_animal_auprc": test_window["macro_animal_auprc"],
                "test_prevalence": test_prevalence,
                **{f"event_{key}": value for key, value in test_event.items()},
            }
        )
        precision, recall, _ = precision_recall_curve(
            test_predictions["label"].to_numpy(),
            test_predictions["seizure_probability"].to_numpy(),
        )
        take = np.unique(np.linspace(0, len(precision) - 1, min(len(precision), 800), dtype=int))
        pr_rows.extend(
            {
                "model": model_name,
                "recall": float(recall[index]),
                "precision": float(precision[index]),
            }
            for index in take
        )
        matches = pd.read_parquet(model_root / "events" / "test_event_matches.parquet")
        matches["model"] = model_name
        boundary_frames.append(matches)
        for animal, values in event_metrics["test"]["per_animal"].items():
            animal_rows.append(
                {
                    "model": model_name,
                    "animal_id": animal,
                    **values,
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    pr_curves = pd.DataFrame(pr_rows)
    boundary = pd.concat(boundary_frames, ignore_index=True)
    animal_metrics = pd.DataFrame(animal_rows)
    comparison.to_csv(inputs.source_dir / "model_comparison.csv", index=False)
    pr_curves.to_csv(inputs.source_dir / "precision_recall_curves.csv", index=False)
    boundary.to_csv(inputs.source_dir / "test_boundary_matches.csv", index=False)
    animal_metrics.to_csv(inputs.source_dir / "per_animal_event_metrics.csv", index=False)
    return comparison, pr_curves, boundary, animal_metrics


def _plot_primary_summary(
    inputs: FigureInputs,
    comparison: pd.DataFrame,
    pr_curves: pd.DataFrame,
    boundary: pd.DataFrame,
    animal_metrics: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 6.4567))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=(0.75, 1.75, 1.65),
        width_ratios=(1.18, 1.0),
        hspace=0.54,
        wspace=0.42,
    )
    workflow = fig.add_subplot(grid[0, :])
    protocol_events = pd.read_parquet(inputs.protocol_dir / "channel_events.parquet")
    n_events = int(len(protocol_events))
    n_animals = int(protocol_events["animal_id"].astype(str).nunique())
    n_streams = int(
        protocol_events[["animal_id", "channel_name"]]
        .drop_duplicates()
        .shape[0]
    )
    n_test_events = int(protocol_events["split"].astype(str).eq("test").sum())
    observed_hours = float(
        pd.to_numeric(comparison["event_recording_duration_hours"], errors="coerce")
        .dropna()
        .median()
    ) if "event_recording_duration_hours" in comparison else float("nan")
    _draw_workflow(workflow, n_animals, n_streams, n_events, n_test_events)
    _panel_label(workflow, "a")

    pr_axis = fig.add_subplot(grid[1, 0])
    for model_name in MODEL_ORDER:
        curve = pr_curves.loc[pr_curves["model"].eq(model_name)].sort_values("recall")
        value = float(comparison.loc[comparison["model"].eq(model_name), "window_auprc"].iloc[0])
        pr_axis.plot(
            curve["recall"],
            curve["precision"],
            color=MODEL_COLORS[model_name],
            label=f"{MODEL_LABELS[model_name]}  {value:.3f}",
        )
    prevalence_series = (
        comparison["test_prevalence"]
        if "test_prevalence" in comparison
        else pd.Series(dtype=float)
    )
    prevalence_values = pd.to_numeric(prevalence_series, errors="coerce").dropna()
    if not prevalence_values.empty:
        # All models share the same held-out windows; median is robust to a
        # stale/partial prediction table in one model and avoids a hard-coded
        # legacy denominator.
        prevalence = float(prevalence_values.median())
        pr_axis.axhline(prevalence, color="#A0A4A8", linestyle="--", linewidth=0.8)
        pr_axis.text(
            0.70,
            min(prevalence + 0.018, 0.98),
            f"test prevalence {prevalence:.3f}",
            color="#73777B",
        )
    pr_axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
    pr_axis.set_title("Held-out window ranking", loc="left", fontweight="bold")
    pr_axis.legend(frameon=False, loc="lower left")
    _clean_axis(pr_axis)
    _panel_label(pr_axis, "b")

    operating = fig.add_subplot(grid[1, 1])
    label_offsets = {
        "eegwavenet_sc": (5, 7),
        "physio_sequence_no_aux": (-78, -22),
        "physio_sequence_stats": (5, -36),
    }
    for row in comparison.itertuples(index=False):
        operating.scatter(
            row.event_false_positive_events_per_hour,
            row.event_event_sensitivity,
            s=42,
            color=MODEL_COLORS[row.model],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        operating.annotate(
            row.model_label,
            (row.event_false_positive_events_per_hour, row.event_event_sensitivity),
            xytext=label_offsets[row.model],
            textcoords="offset points",
            fontsize=6.2,
        )
    # Keep the callout inside the data domain for both the historical and
    # pooled old+new runs.  The old fixed 0-12 x-range clipped all current
    # operating points (which are tens of events per channel-hour).
    fp_values = pd.to_numeric(
        comparison["event_false_positive_events_per_hour"], errors="coerce"
    ).dropna()
    sens_values = pd.to_numeric(
        comparison["event_event_sensitivity"], errors="coerce"
    ).dropna()
    fp_max = float(fp_values.max()) if not fp_values.empty else 1.0
    sens_min = float(sens_values.min()) if not sens_values.empty else 0.0
    sens_max = float(sens_values.max()) if not sens_values.empty else 1.0
    operating.set(
        xlabel="False-positive events per channel-hour",
        ylabel="Event sensitivity",
        xlim=(0, max(1.0, fp_max * 1.12)),
        ylim=(max(0.0, min(0.0, sens_min - 0.08)), min(1.05, max(1.0, sens_max * 1.08))),
    )
    operating.text(
        0.04,
        0.06,
        "lower-left = fewer false-positive events",
        transform=operating.transAxes,
        color="#55585C",
        fontsize=6.2,
    )
    operating.set(
        xlabel="False-positive events per channel-hour",
        ylabel="Event sensitivity",
    )
    operating.set_title("Validation-frozen event operating point", loc="left", fontweight="bold")
    _clean_axis(operating)
    _panel_label(operating, "c")

    boundary_axis = fig.add_subplot(grid[2, 0])
    positions = []
    values = []
    colors = []
    labels = []
    for model_index, model_name in enumerate(MODEL_ORDER):
        group = boundary.loc[boundary["model"].eq(model_name)]
        for offset, (column, label, color) in enumerate(
            (
                ("onset_error_seconds", "onset", TRUE_COLOR),
                ("offset_error_seconds", "offset", PRED_COLOR),
            )
        ):
            positions.append(model_index * 3 + offset)
            values.append(np.abs(group[column].to_numpy(float)))
            colors.append(color)
            labels.append(label)
    boxes = boundary_axis.boxplot(
        values,
        positions=positions,
        widths=0.68,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 1.0},
        whiskerprops={"linewidth": 0.7},
        capprops={"linewidth": 0.7},
    )
    for patch, color in zip(boxes["boxes"], colors, strict=True):
        patch.set(facecolor=color, edgecolor=color, alpha=0.88)
    boundary_axis.set_xticks(
        [0.5, 3.5, 6.5],
        ["EEGWaveNet", "Physio", "+ statistics"],
    )
    boundary_axis.set_ylabel("Absolute boundary error (s)")
    boundary_axis.set_ylim(0, 9)
    boundary_axis.set_title("Matched-event boundary error", loc="left", fontweight="bold")
    boundary_axis.plot([], [], color=TRUE_COLOR, linewidth=5, label="Onset")
    boundary_axis.plot([], [], color=PRED_COLOR, linewidth=5, label="Offset")
    boundary_axis.legend(frameon=False, ncol=2, loc="upper right")
    _clean_axis(boundary_axis)
    _panel_label(boundary_axis, "d")

    animal_axis = fig.add_subplot(grid[2, 1])
    x_positions = np.arange(len(MODEL_ORDER))
    animals = sorted(animal_metrics["animal_id"].astype(str).unique())
    animal_colors = _animal_colors(animals)
    model_names = list(MODEL_ORDER)
    for animal in animals:
        group = animal_metrics.loc[
            animal_metrics["animal_id"].astype(str).eq(animal)
        ].set_index("model")
        values = [
            float(group.loc[model, "event_sensitivity"]) if model in group.index else float("nan")
            for model in model_names
        ]
        animal_axis.plot(
            np.arange(len(model_names)),
            values,
            marker="o",
            markersize=4,
            color=animal_colors[animal],
            label=animal,
        )
    animal_axis.set_xticks(
        np.arange(len(model_names)),
        [MODEL_LABELS.get(model, model) for model in model_names],
    )
    animal_axis.set_ylabel("Event sensitivity")
    animal_axis.set_ylim(0.0, 1.05)
    animal_axis.set_title("Performance is not identical across mice", loc="left", fontweight="bold")
    animal_axis.legend(frameon=False, ncol=3, loc="lower left")
    _clean_axis(animal_axis)
    _panel_label(animal_axis, "e")

    fig.text(
        0.01,
        0.006,
        f"Held-out test: {n_test_events} channel-specific events, {observed_hours:.2f} channel-hours; "
        "one fixed split and seed. "
        "All decoder settings were selected on validation data only.",
        fontsize=6.3,
        color="#55585C",
    )
    _save_bundle(fig, inputs.output_dir / "figure_1_primary_summary")


def _draw_workflow(
    axis: plt.Axes,
    n_animals: int,
    n_streams: int,
    n_events: int,
    n_test_events: int,
) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    boxes = [
        (0.02, 0.18, 0.17, f"{n_animals} mice, {n_streams} streams\n{n_events:,} events"),
        (0.225, 0.18, 0.17, "Purged regions\n70 / 15 / 15"),
        (0.43, 0.18, 0.17, "Shared 1-channel model\nidentity hidden"),
        (0.635, 0.18, 0.15, "Global val decoder\nonset / offset"),
        (0.82, 0.18, 0.16, f"Held-out audit\n{n_test_events:,} events"),
    ]
    for left, bottom, width, text in boxes:
        axis.add_patch(
            Rectangle(
                (left, bottom),
                width,
                0.56,
                facecolor="#F3F4F5",
                edgecolor="#8D949A",
                linewidth=0.7,
            )
        )
        axis.text(left + width / 2, bottom + 0.28, text, ha="center", va="center", fontsize=5.9)
    for index in range(len(boxes) - 1):
        left, bottom, width, _ = boxes[index]
        next_left = boxes[index + 1][0]
        axis.add_patch(
            FancyArrowPatch(
                (left + width + 0.005, 0.46),
                (next_left - 0.005, 0.46),
                arrowstyle="-|>",
                mutation_scale=7,
                color="#62686D",
                linewidth=0.7,
            )
        )
    axis.text(0.02, 0.89, "Channel-blind continuous seizure annotation", fontweight="bold", fontsize=8)


def _plot_training_diagnostics(inputs: FigureInputs) -> None:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FIGURE_WIDTH_IN, 3.6220),
        sharex="col",
        constrained_layout=False,
    )
    source_frames = []
    for column, model_name in enumerate(MODEL_ORDER):
        history = pd.read_csv(inputs.run_root / model_name / "history.csv")
        history["model"] = model_name
        source_frames.append(history)
        best_epoch = int(history.loc[history["best"].astype(bool), "epoch"].max())
        top = axes[0, column]
        bottom = axes[1, column]
        top.plot(
            history["epoch"],
            history["train_loss"],
            color="#9AA0A5",
            linestyle="--",
            label="Train",
        )
        top.plot(history["epoch"], history["validation_loss"], color=MODEL_COLORS[model_name], label="Validation")
        top.axvline(best_epoch, color="#202326", linestyle=":", linewidth=0.8)
        bottom.plot(
            history["epoch"],
            history["validation_macro_animal_auprc"],
            color=MODEL_COLORS[model_name],
        )
        bottom.axvline(best_epoch, color="#202326", linestyle=":", linewidth=0.8)
        top.set_title(MODEL_LABELS[model_name], fontweight="bold")
        top.set_ylabel("Cross-entropy loss" if column == 0 else "")
        bottom.set_ylabel("Validation macro AUPRC" if column == 0 else "")
        bottom.set_xlabel("Epoch")
        top.text(
            0.98,
            0.93,
            f"best epoch {best_epoch}",
            transform=top.transAxes,
            ha="right",
            va="top",
            fontsize=6.2,
        )
        if column == 0:
            top.legend(frameon=False, loc="lower left")
        _clean_axis(top)
        _clean_axis(bottom)
    _panel_label(axes[0, 0], "a")
    _panel_label(axes[1, 0], "b")
    fig.subplots_adjust(left=0.085, right=0.99, top=0.92, bottom=0.13, wspace=0.28, hspace=0.18)
    pd.concat(source_frames, ignore_index=True).to_csv(
        inputs.source_dir / "training_histories.csv", index=False
    )
    _save_bundle(fig, inputs.output_dir / "figure_s1_training_diagnostics")


def _select_representative_events(inputs: FigureInputs) -> pd.DataFrame:
    model_root = inputs.run_root / "physio_sequence_no_aux" / "events"
    matches = pd.read_parquet(model_root / "test_event_matches.parquet")
    events = pd.read_parquet(inputs.protocol_dir / "channel_events.parquet")
    predicted = pd.read_parquet(model_root / "test_predicted_events.parquet")
    merged = matches.merge(
        events,
        on=["animal_id", "session_id", "channel_name", "event_id", "split"],
        how="left",
        validate="one_to_one",
    ).merge(
        predicted,
        on=[
            "animal_id",
            "session_id",
            "channel_name",
            "split",
            "decoder_segment_id",
            "predicted_event_index",
        ],
        how="left",
        suffixes=("_true", "_pred"),
        validate="many_to_one",
    )
    rows = []
    for animal, group in merged.groupby("animal_id", sort=True):
        median_iou = float(group["iou"].median())
        selected = (
            group.assign(distance_to_median=(group["iou"] - median_iou).abs())
            .sort_values(["distance_to_median", "event_id"], kind="stable")
            .iloc[0]
        )
        rows.append(selected)
    representatives = pd.DataFrame(rows).reset_index(drop=True)
    representatives.to_csv(inputs.source_dir / "representative_events.csv", index=False)
    return representatives


def _plot_clinical_waveform_evidence(
    inputs: FigureInputs, representatives: pd.DataFrame
) -> None:
    predictions = pd.read_parquet(
        inputs.run_root
        / "physio_sequence_no_aux"
        / "continuous"
        / "continuous_predictions.parquet"
    )
    decoder = json.loads(
        (
            inputs.run_root
            / "physio_sequence_no_aux"
            / "events"
            / "decoder_fit.json"
        ).read_text(encoding="utf-8")
    )["config"]
    traces = TraceStore(inputs.protocol_dir)
    n_rows = max(1, len(representatives))
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, max(7.4803, 2.25 * n_rows)))
    outer = fig.add_gridspec(
        n_rows,
        2,
        width_ratios=(1.75, 1.0),
        hspace=0.48,
        wspace=0.28,
    )
    waveform_source = []
    for row_number, row in enumerate(representatives.itertuples(index=False)):
        true_start = float(row.start_time_ms_true)
        true_end = float(row.end_time_ms_true)
        pred_start = float(row.start_time_ms_pred)
        pred_end = float(row.end_time_ms_pred)
        margin_ms = 10_000.0
        crop_start = min(true_start, pred_start) - margin_ms
        crop_end = max(true_end, pred_end) + margin_ms
        time_ms, waveform = traces.load(
            str(row.animal_id),
            str(row.session_id),
            str(row.channel_name),
            crop_start,
            crop_end,
        )
        relative_seconds = (time_ms - true_start) / 1000.0
        probability = predictions.loc[
            predictions["animal_id"].eq(row.animal_id)
            & predictions["session_id"].eq(row.session_id)
            & predictions["channel_name"].eq(row.channel_name)
            & predictions["start_time_ms"].between(crop_start, crop_end)
        ].sort_values("start_time_ms", kind="stable")

        left_grid = outer[row_number, 0].subgridspec(2, 1, height_ratios=(2.1, 0.85), hspace=0.06)
        trace_axis = fig.add_subplot(left_grid[0])
        probability_axis = fig.add_subplot(left_grid[1], sharex=trace_axis)
        plot_x, lower, upper = _waveform_display(relative_seconds, waveform, maximum_points=5000)
        if lower is None:
            trace_axis.plot(plot_x, upper, color="#202326", linewidth=0.55)
        else:
            trace_axis.fill_between(plot_x, lower, upper, color="#303438", linewidth=0, alpha=0.85)
        _event_spans(trace_axis, true_start, true_end, pred_start, pred_end, true_start)
        probability_x = (probability["center_time_ms"].to_numpy(float) - true_start) / 1000.0
        probability_axis.plot(
            probability_x,
            probability["seizure_probability"],
            color=PROBABILITY_COLOR,
            linewidth=0.9,
        )
        probability_axis.axhline(decoder["high_threshold"], color="#6F7478", linestyle="--", linewidth=0.65)
        probability_axis.axhline(decoder["low_threshold"], color="#A0A4A8", linestyle=":", linewidth=0.65)
        _event_spans(probability_axis, true_start, true_end, pred_start, pred_end, true_start)
        probability_axis.set(ylim=(0, 1.02), ylabel="P(seizure)", xlabel="Time from manual onset (s)")
        trace_axis.set_ylabel("Relative amplitude (a.u.)")
        trace_axis.tick_params(labelbottom=False)
        trace_axis.set_title(
            f"{row.animal_id}  blinded {row.channel_name}  |  IoU {row.iou:.2f}, "
            f"coverage {row.annotated_coverage:.2f}",
            loc="left",
            fontweight="bold",
        )
        _clean_axis(trace_axis)
        _clean_axis(probability_axis)
        _panel_label(trace_axis, chr(ord("a") + row_number))

        zoom_grid = outer[row_number, 1].subgridspec(2, 1, hspace=0.64)
        onset_axis = fig.add_subplot(zoom_grid[0])
        offset_axis = fig.add_subplot(zoom_grid[1])
        _plot_boundary_zoom(
            onset_axis,
            relative_seconds,
            waveform,
            true_boundary=0.0,
            predicted_boundary=(pred_start - true_start) / 1000.0,
            title=f"Onset  | error {row.onset_error_seconds:+.2f} s",
            show_xlabel=False,
        )
        _plot_boundary_zoom(
            offset_axis,
            relative_seconds,
            waveform,
            true_boundary=(true_end - true_start) / 1000.0,
            predicted_boundary=(pred_end - true_start) / 1000.0,
            title=f"Offset  | error {row.offset_error_seconds:+.2f} s",
            show_xlabel=True,
        )
        for time_value, amplitude in zip(relative_seconds[::25], waveform[::25], strict=False):
            waveform_source.append(
                {
                    "animal_id": row.animal_id,
                    "channel_name": row.channel_name,
                    "event_id": row.event_id,
                    "time_from_manual_onset_s": float(time_value),
                    "relative_amplitude_au": float(amplitude),
                }
            )
    fig.text(
        0.01,
        0.008,
        "Selection: matched test event nearest each mouse's median IoU. "
        "Shading: manual (blue), model (orange).\n"
        "Boundary uncertainty is +/-2 s; amplitude is relative to the training baseline (a.u.).",
        fontsize=6.0,
        color="#55585C",
        va="bottom",
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.975, bottom=0.07)
    pd.DataFrame(waveform_source).to_csv(
        inputs.source_dir / "representative_waveform_samples.csv", index=False
    )
    _save_bundle(fig, inputs.output_dir / "figure_2_clinical_waveform_evidence")


def _plot_whole_recording_overviews(inputs: FigureInputs) -> None:
    model_root = inputs.run_root / "physio_sequence_no_aux"
    predictions = pd.read_parquet(model_root / "continuous" / "continuous_predictions.parquet")
    manual_events = pd.read_parquet(inputs.protocol_dir / "channel_events.parquet")
    predicted_events = pd.read_parquet(model_root / "events" / "full_recording_predicted_events.parquet")
    full_index = pd.read_parquet(inputs.protocol_dir / "full_index.parquet")
    baseline = json.loads((inputs.protocol_dir / "training_baseline.json").read_text(encoding="utf-8"))
    scale_lookup = {
        (str(row["animal_id"]), str(row["channel_name"])): float(row["scale"])
        for row in baseline["channels"]
    }
    array_cache: dict[str, np.ndarray] = {}
    overview_dir = inputs.output_dir / "whole_recording_overviews"
    overview_dir.mkdir(parents=True, exist_ok=True)
    overview_source = []
    for key, group in predictions.groupby(["animal_id", "session_id", "channel_name"], sort=True):
        animal, session, channel = map(str, key)
        ordered = group.sort_values("start_time_ms", kind="stable")
        index_rows = full_index.loc[
            full_index["animal_id"].eq(animal)
            & full_index["session_id"].eq(session)
            & full_index["channel_name"].eq(channel)
        ].sort_values("start_time_ms", kind="stable").reset_index(drop=True)
        index_rows["_position"] = np.arange(len(index_rows), dtype=np.int64)
        rms = np.empty(len(index_rows), dtype=np.float32)
        scale = max(scale_lookup[(animal, channel)], 1e-6)
        for root, shard_rows in index_rows.groupby("filtered_cache_dir", sort=False):
            root = str(root)
            if root not in array_cache:
                array_cache[root] = np.load(Path(root) / "signals.npy", mmap_mode="r")
            local = shard_rows["filtered_local_index"].to_numpy(np.int64)
            channel_index = int(shard_rows["channel_index"].iloc[0])
            values = np.asarray(array_cache[root][local, channel_index], dtype=np.float32)
            rms[shard_rows["_position"].to_numpy(np.int64)] = np.sqrt(
                np.mean((values / scale) ** 2, axis=1)
            )
        time_hours = ordered["center_time_ms"].to_numpy(float) / 3_600_000.0
        minute_bin = np.floor(ordered["center_time_ms"].to_numpy(float) / 60_000.0).astype(np.int64)
        source = pd.DataFrame(
            {
                "bin": minute_bin,
                "time_hours": time_hours,
                "rms": rms[: len(ordered)],
                "probability": ordered["seizure_probability"].to_numpy(float),
            }
        ).groupby("bin", as_index=False).agg(
            time_hours=("time_hours", "median"),
            rms_median=("rms", "median"),
            rms_q90=("rms", lambda values: float(np.quantile(values, 0.90))),
            probability_q95=("probability", lambda values: float(np.quantile(values, 0.95))),
        )
        source["animal_id"] = animal
        source["session_id"] = session
        source["channel_name"] = channel
        overview_source.append(source)
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(FIGURE_WIDTH_IN, 3.0709),
            sharex=True,
            gridspec_kw={"height_ratios": (1.5, 1.0, 0.42), "hspace": 0.12},
        )
        time_values = source["time_hours"].to_numpy(float)
        breaks = np.r_[
            0,
            np.flatnonzero(np.diff(time_values) > 2.5 / 60.0) + 1,
            len(source),
        ]
        for left, right in zip(breaks[:-1], breaks[1:], strict=True):
            segment = source.iloc[int(left) : int(right)]
            axes[0].fill_between(
                segment["time_hours"],
                segment["rms_median"],
                segment["rms_q90"],
                color="#9AA1A6",
                alpha=0.55,
                linewidth=0,
            )
            axes[0].plot(
                segment["time_hours"],
                segment["rms_median"],
                color="#303438",
                linewidth=0.65,
            )
            axes[1].plot(
                segment["time_hours"],
                segment["probability_q95"],
                color=PROBABILITY_COLOR,
                linewidth=0.7,
            )
        axes[0].set_ylabel("Amplitude\nenvelope (a.u.)")
        axes[1].set(ylabel="Minute q95\nP(seizure)", ylim=(0, 1.02))
        manual = manual_events.loc[
            manual_events["animal_id"].eq(animal)
            & manual_events["session_id"].eq(session)
            & manual_events["channel_name"].eq(channel)
        ]
        detected = predicted_events.loc[
            predicted_events["animal_id"].eq(animal)
            & predicted_events["session_id"].eq(session)
            & predicted_events["channel_name"].eq(channel)
        ]
        for event in manual.itertuples(index=False):
            axes[2].axvspan(event.start_time_ms / 3_600_000, event.end_time_ms / 3_600_000, 0.55, 0.95, color=TRUE_COLOR, linewidth=0)
        for event in detected.itertuples(index=False):
            axes[2].axvspan(event.start_time_ms / 3_600_000, event.end_time_ms / 3_600_000, 0.05, 0.45, color=PRED_COLOR, linewidth=0)
        axes[2].text(0.002, 0.76, "manual", transform=axes[2].transAxes, va="center", fontsize=6.0)
        axes[2].text(0.002, 0.23, "model", transform=axes[2].transAxes, va="center", fontsize=6.0)
        axes[2].set(xlabel="Recording time (h)", yticks=[])
        axes[0].set_title(f"{animal} | blinded {channel} | {session}", loc="left", fontweight="bold")
        for axis in axes:
            _clean_axis(axis)
        fig.subplots_adjust(left=0.09, right=0.99, top=0.91, bottom=0.16)
        _save_bundle(fig, overview_dir / f"{animal}_{session}_{channel}_overview", include_tiff=False)
    pd.concat(overview_source, ignore_index=True).to_csv(
        inputs.source_dir / "whole_recording_minute_summary.csv", index=False
    )


def _build_event_atlases(inputs: FigureInputs) -> None:
    model_root = inputs.run_root / "physio_sequence_no_aux"
    predictions = pd.read_parquet(model_root / "continuous" / "continuous_predictions.parquet")
    true_events = pd.read_parquet(inputs.protocol_dir / "channel_events.parquet")
    test_true = true_events.loc[true_events["split"].eq("test")].sort_values(
        ["animal_id", "session_id", "channel_name", "start_time_ms"], kind="stable"
    )
    predicted = pd.read_parquet(model_root / "events" / "test_predicted_events.parquet")
    matches = pd.read_parquet(model_root / "events" / "test_event_matches.parquet")
    trace_store = TraceStore(inputs.protocol_dir)
    atlas_dir = inputs.output_dir / "review_atlases"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    _event_review_pdf(
        atlas_dir / "heldout_test_event_atlas.pdf",
        test_true,
        predicted,
        predictions,
        trace_store,
        title="Held-out annotated events",
    )
    match_keys = matches[
        ["animal_id", "session_id", "channel_name", "predicted_event_index"]
    ].drop_duplicates()
    false_positive = predicted.merge(
        match_keys.assign(matched=True),
        on=["animal_id", "session_id", "channel_name", "predicted_event_index"],
        how="left",
    )
    false_positive = false_positive.loc[false_positive["matched"].isna()].drop(columns="matched")
    _predicted_review_pdf(
        atlas_dir / "heldout_false_positive_candidate_atlas.pdf",
        false_positive,
        predictions,
        trace_store,
        title="Held-out unmatched model candidates",
    )
    false_positive.to_csv(inputs.source_dir / "heldout_false_positive_candidates.csv", index=False)


def _event_review_pdf(
    path: Path,
    events: pd.DataFrame,
    predicted: pd.DataFrame,
    probabilities: pd.DataFrame,
    traces: TraceStore,
    *,
    title: str,
) -> None:
    with PdfPages(path) as pdf:
        for page_start in range(0, len(events), 4):
            page = events.iloc[page_start : page_start + 4]
            fig, axes = plt.subplots(4, 1, figsize=(8.27, 11.69), squeeze=False)
            for panel_index, (axis, event) in enumerate(
                zip(axes[:, 0], page.itertuples(index=False), strict=False)
            ):
                overlaps = predicted.loc[
                    predicted["animal_id"].eq(event.animal_id)
                    & predicted["session_id"].eq(event.session_id)
                    & predicted["channel_name"].eq(event.channel_name)
                    & predicted["end_time_ms"].gt(event.start_time_ms)
                    & predicted["start_time_ms"].lt(event.end_time_ms)
                ]
                crop_start = float(event.start_time_ms) - 8_000.0
                crop_end = float(event.end_time_ms) + 8_000.0
                _review_panel(
                    axis,
                    event.animal_id,
                    event.session_id,
                    event.channel_name,
                    f"{event.event_id} [{event.split}]",
                    crop_start,
                    crop_end,
                    float(event.start_time_ms),
                    float(event.end_time_ms),
                    overlaps,
                    probabilities,
                    traces,
                    show_xlabel=panel_index == len(page) - 1,
                )
            for axis in axes[len(page) :, 0]:
                axis.axis("off")
            fig.suptitle(
                f"{title} | {page_start + 1}-{page_start + len(page)} of {len(events)}",
                fontsize=10,
                fontweight="bold",
                y=0.992,
            )
            fig.tight_layout(rect=(0.03, 0.025, 0.99, 0.95))
            pdf.savefig(fig, dpi=300)
            plt.close(fig)


def _predicted_review_pdf(
    path: Path,
    predicted: pd.DataFrame,
    probabilities: pd.DataFrame,
    traces: TraceStore,
    *,
    title: str,
) -> None:
    with PdfPages(path) as pdf:
        for page_start in range(0, len(predicted), 4):
            page = predicted.iloc[page_start : page_start + 4]
            fig, axes = plt.subplots(4, 1, figsize=(8.27, 11.69), squeeze=False)
            for panel_index, (axis, event) in enumerate(
                zip(axes[:, 0], page.itertuples(index=False), strict=False)
            ):
                crop_start = float(event.start_time_ms) - 8_000.0
                crop_end = float(event.end_time_ms) + 8_000.0
                _review_panel(
                    axis,
                    event.animal_id,
                    event.session_id,
                    event.channel_name,
                    f"candidate {event.predicted_event_index}",
                    crop_start,
                    crop_end,
                    None,
                    None,
                    pd.DataFrame([event._asdict()]),
                    probabilities,
                    traces,
                    show_xlabel=panel_index == len(page) - 1,
                )
            for axis in axes[len(page) :, 0]:
                axis.axis("off")
            fig.suptitle(
                f"{title} | {page_start + 1}-{page_start + len(page)} of {len(predicted)}",
                fontsize=10,
                fontweight="bold",
                y=0.992,
            )
            fig.tight_layout(rect=(0.03, 0.025, 0.99, 0.95))
            pdf.savefig(fig, dpi=300)
            plt.close(fig)


def _review_panel(
    axis: plt.Axes,
    animal: str,
    session: str,
    channel: str,
    identifier: str,
    crop_start: float,
    crop_end: float,
    true_start: float | None,
    true_end: float | None,
    predicted: pd.DataFrame,
    probabilities: pd.DataFrame,
    traces: TraceStore,
    *,
    show_xlabel: bool,
) -> None:
    time_ms, waveform = traces.load(animal, session, channel, crop_start, crop_end)
    reference = true_start if true_start is not None else crop_start
    x = (time_ms - reference) / 1000.0
    plot_x, lower, upper = _waveform_display(x, waveform, maximum_points=3000)
    if lower is None:
        axis.plot(plot_x, upper, color="#292D30", linewidth=0.45)
    else:
        axis.fill_between(plot_x, lower, upper, color="#404448", linewidth=0, alpha=0.8)
    if true_start is not None and true_end is not None:
        axis.axvspan(0, (true_end - true_start) / 1000.0, color=TRUE_COLOR, alpha=0.13, linewidth=0)
    for event in predicted.itertuples(index=False):
        axis.axvspan(
            (float(event.start_time_ms) - reference) / 1000.0,
            (float(event.end_time_ms) - reference) / 1000.0,
            color=PRED_COLOR,
            alpha=0.16,
            linewidth=0,
        )
    probability = probabilities.loc[
        probabilities["animal_id"].eq(animal)
        & probabilities["session_id"].eq(session)
        & probabilities["channel_name"].eq(channel)
        & probabilities["start_time_ms"].between(crop_start, crop_end)
    ]
    if not probability.empty:
        twin = axis.twinx()
        twin.plot(
            (probability["center_time_ms"].to_numpy(float) - reference) / 1000.0,
            probability["seizure_probability"],
            color=PROBABILITY_COLOR,
            linewidth=0.7,
            alpha=0.9,
        )
        twin.set(ylim=(0, 1.02), yticks=(0, 0.5, 1), ylabel="P")
        twin.spines["top"].set_visible(False)
    axis.set_title(f"{animal} | blinded {channel} | {identifier}", loc="left", fontsize=7.2, fontweight="bold")
    axis.set(
        xlabel="Time relative to manual onset or candidate crop (s)" if show_xlabel else "",
        ylabel="Relative amplitude (a.u.)",
    )
    _clean_axis(axis)


def _waveform_display(
    x: np.ndarray, y: np.ndarray, *, maximum_points: int
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    if len(x) <= maximum_points:
        return x, None, y
    bins = max(maximum_points // 2, 1)
    edges = np.linspace(0, len(x), bins + 1, dtype=int)
    centers = np.empty(bins, dtype=float)
    lower = np.empty(bins, dtype=float)
    upper = np.empty(bins, dtype=float)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        block = y[left:max(right, left + 1)]
        centers[index] = float(np.mean(x[left:max(right, left + 1)]))
        lower[index] = float(np.min(block))
        upper[index] = float(np.max(block))
    return centers, lower, upper


def _event_spans(
    axis: plt.Axes,
    true_start: float,
    true_end: float,
    pred_start: float,
    pred_end: float,
    reference: float,
) -> None:
    axis.axvspan(
        (true_start - reference) / 1000.0,
        (true_end - reference) / 1000.0,
        color=TRUE_COLOR,
        alpha=0.12,
        linewidth=0,
    )
    axis.axvspan(
        (pred_start - reference) / 1000.0,
        (pred_end - reference) / 1000.0,
        facecolor="none",
        edgecolor=PRED_COLOR,
        linewidth=0.9,
    )


def _plot_boundary_zoom(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    true_boundary: float,
    predicted_boundary: float,
    title: str,
    show_xlabel: bool,
) -> None:
    keep = (x >= true_boundary - 6) & (x <= true_boundary + 6)
    axis.plot(x[keep] - true_boundary, y[keep], color="#202326", linewidth=0.55)
    axis.axvline(0, color=TRUE_COLOR, linewidth=1.0, label="Manual")
    axis.axvline(predicted_boundary - true_boundary, color=PRED_COLOR, linewidth=1.0, label="Model")
    axis.set(
        xlim=(-6, 6),
        xlabel="Time from manual boundary (s)" if show_xlabel else "",
        ylabel="a.u.",
    )
    if not show_xlabel:
        axis.tick_params(labelbottom=False)
    axis.set_title(title, loc="left", fontsize=6.8, fontweight="bold")
    _clean_axis(axis)


def _write_figure_notes(inputs: FigureInputs, representatives: pd.DataFrame) -> None:
    protocol_events = pd.read_parquet(inputs.protocol_dir / "channel_events.parquet")
    test_events = protocol_events.loc[protocol_events["split"].astype(str).eq("test")]
    recording_hours = float("nan")
    event_metrics_path = (
        inputs.run_root / "eegwavenet_sc" / "events" / "event_metrics.json"
    )
    if event_metrics_path.is_file():
        try:
            event_metrics = json.loads(event_metrics_path.read_text(encoding="utf-8"))
            recording_hours = float(
                event_metrics["test"]["pooled"]["recording_duration_hours"]
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            recording_hours = float("nan")
    hours_text = f"{recording_hours:.2f}" if np.isfinite(recording_hours) else "NA"
    notes = {
        "schema_version": 1,
        "core_claim": (
            "A shared blinded single-channel physiology-informed model recovers most held-out "
            "annotated seizure events with approximately two-second boundary resolution, while "
            "unmatched detections remain candidates for label-versus-model review."
        ),
        "archetype": "asymmetric waveform evidence plus compact quantitative model comparison",
        "backend": "Python matplotlib only",
        "final_width_mm": 183,
        "editable_text": "SVG fonttype none; PDF fonttype 42",
        "split": "purged temporal regions, within each mouse, 70/15/15",
        "seeds_or_folds": "one fixed seed and one frozen split",
        "metric_definition": (
            "AUPRC at window level; one-to-one maximum-IoU event matching; false-positive "
            "events per observed channel-hour; boundary errors relative to formal annotations"
        ),
        "baseline_definition": "test positive fraction for PR; EEGWaveNet-SC for architecture",
        "n_definition": (
            f"{len(test_events)} held-out channel-specific clinical events over "
            f"{hours_text} channel-hours"
        ),
        "representative_selection": "nearest matched-event IoU to each mouse's median matched IoU",
        "representative_event_ids": representatives["event_id"].astype(str).tolist(),
        "image_integrity": {
            "raw_file": "complete filtered cache signals.npy",
            "processed_file": "relative train-background median/MAD amplitude",
            "crop": "declared 10-s margins around union of manual and model intervals",
            "brightness_contrast_gamma": "not applicable",
            "pseudo_color": "none",
            "scale_calibration": "250 Hz; amplitude remains a.u.",
            "stitching": "overlapping four-second windows reconstructed using one-second tails",
            "quantification_link": "source_data/representative_events.csv",
        },
    }
    (inputs.output_dir / "figure_qa_notes.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.10,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=2.5, pad=2)


def _save_bundle(fig: plt.Figure, stem: Path, *, include_tiff: bool = True) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the declared journal canvas; a tight bounding box changes its physical width.
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    if include_tiff:
        fig.savefig(stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig)


__all__ = ["generate_single_channel_figure_bundle"]
