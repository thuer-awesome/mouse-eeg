from __future__ import annotations

import json
import math
from dataclasses import asdict
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
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "legend.fontsize": 6.3,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
    }
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from mouse_sleep.epilepsy.seizure_annotation import (
    DecoderConfig,
    decode_window_probabilities,
)
from mouse_sleep.epilepsy.single_channel_visualization import TraceStore, _waveform_display


MODEL_ORDER = ("eegwavenet_sc", "physio_sequence_no_aux", "physio_sequence_stats")
MODEL_LABELS = {
    "eegwavenet_sc": "EEGWaveNet-SC",
    "physio_sequence_no_aux": "Physio-Sequence",
    "physio_sequence_stats": "Physio-Sequence + statistics",
}
MODEL_COLORS = {
    "eegwavenet_sc": "#59636F",
    "physio_sequence_no_aux": "#2F6B8A",
    "physio_sequence_stats": "#C56B3D",
}

FULL = "full_coverage_ge_0p9"
PARTIAL = "partial_coverage_0p5_to_0p9"
LOW = "low_overlap_gt_0_to_0p5"
MISSED = "missed_coverage_0"
LONG_MISSED = "missed_ge_30s"
BOUNDARY_SHIFT = "boundary_shift_gt_2s"
OUTSIDE = "label_external_prediction"

PRIMARY_CATEGORIES = (FULL, PARTIAL, LOW, MISSED)
SECONDARY_CATEGORIES = (LONG_MISSED, BOUNDARY_SHIFT)
CATEGORY_LABELS = {
    FULL: "Full detection: coverage >= 0.90",
    PARTIAL: "Partial detection: 0.50 <= coverage < 0.90",
    LOW: "Low overlap: 0 < coverage < 0.50",
    MISSED: "Complete miss: coverage = 0",
    LONG_MISSED: "Long complete miss: duration >= 30 s",
    BOUNDARY_SHIFT: "Boundary shift: |onset or offset error| > 2 s",
    OUTSIDE: "Prediction with zero overlap to every manual label",
}

MANUAL_COLOR = "#1E5B8C"
PREDICTED_COLOR = "#C45A32"
RAW_COLOR = "#8A9196"
SMOOTHED_COLOR = "#25746B"
MASK_COLOR = "#C45A32"
INK = "#252A2E"
MID_GREY = "#6C7378"
LIGHT_GREY = "#E6E9EB"
KEYS = ["animal_id", "session_id", "channel_name"]


def generate_event_coverage_atlas(
    run_root: str | Path,
    protocol_dir: str | Path,
    output_dir: str | Path,
    *,
    models: Iterable[str] = MODEL_ORDER,
    event_directory_name: str = "events_redecoded_v2",
) -> Path:
    """Render complete event-centric review atlases for frozen test outputs."""

    run_root = Path(run_root).resolve()
    protocol_dir = Path(protocol_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    manual = pd.read_parquet(protocol_dir / "channel_events.parquet")
    manual = manual.loc[manual["split"].astype(str).eq("test")].copy()
    manual = manual.sort_values(KEYS + ["start_time_ms", "event_id"], kind="stable")
    traces = TraceStore(protocol_dir)

    selected_models = tuple(models)
    if not selected_models or len(selected_models) != len(set(selected_models)):
        raise ValueError("models must be a non-empty collection without duplicates")
    unknown_models = sorted(set(selected_models) - set(MODEL_LABELS))
    if unknown_models:
        raise ValueError(f"unknown models: {unknown_models}")
    if not event_directory_name or Path(event_directory_name).name != event_directory_name:
        raise ValueError("event_directory_name must be one directory name")

    all_manual_rows: list[pd.DataFrame] = []
    all_candidate_rows: list[pd.DataFrame] = []
    all_index_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for model_name in selected_models:
        model_root = run_root / "models" / model_name
        event_dir = model_root / event_directory_name
        predictions = pd.read_parquet(
            model_root / "continuous" / "continuous_predictions.parquet"
        )
        predictions = predictions.loc[predictions["split"].astype(str).eq("test")].copy()
        predictions = predictions.sort_values(KEYS + ["start_time_ms"], kind="stable")
        predicted = pd.read_parquet(event_dir / "test_predicted_events.parquet")
        assessments = pd.read_parquet(event_dir / "test_true_event_assessments.parquet")
        decoder_record = json.loads(
            (event_dir / "decoder_fit.json").read_text(encoding="utf-8")
        )
        config = DecoderConfig(**decoder_record["config"])
        decoded = _decode_probability_tracks(predictions, config)
        _validate_decoded_intervals(decoded, predicted)

        event_table = _build_manual_event_table(manual, assessments, predicted, model_name)
        candidate_table = _build_external_candidate_table(manual, predicted, predictions, model_name)
        event_table.to_csv(
            source_dir / f"{model_name}_manual_event_coverage.csv",
            index=False,
            encoding="utf-8-sig",
        )
        candidate_table.to_csv(
            source_dir / f"{model_name}_label_external_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        all_manual_rows.append(event_table)
        all_candidate_rows.append(candidate_table)

        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        category_frames: dict[str, pd.DataFrame] = {
            category: event_table.loc[event_table["primary_category"].eq(category)].copy()
            for category in PRIMARY_CATEGORIES
        }
        category_frames[LONG_MISSED] = event_table.loc[event_table["long_missed"]].copy()
        category_frames[BOUNDARY_SHIFT] = event_table.loc[event_table["boundary_shift"]].copy()
        category_frames[OUTSIDE] = candidate_table.copy()

        for category, frame in category_frames.items():
            frame = _sort_review_frame(frame, category)
            pdf_path = model_dir / f"{category}.pdf"
            index_rows = _write_category_pdf(
                pdf_path,
                frame,
                category,
                model_name,
                manual,
                predicted,
                decoded,
                traces,
                config,
            )
            all_index_rows.extend(index_rows)
            if not frame.empty:
                exemplar = _select_exemplar(frame, category)
                _write_exemplar(
                    model_dir / f"example_{category}",
                    exemplar,
                    category,
                    model_name,
                    manual,
                    predicted,
                    decoded,
                    traces,
                    config,
                )
            summary_rows.append(
                {
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "category": category,
                    "category_label": CATEGORY_LABELS[category],
                    "count": int(len(frame)),
                }
            )

    manual_all = pd.concat(all_manual_rows, ignore_index=True)
    candidates_all = pd.concat(all_candidate_rows, ignore_index=True)
    atlas_index = pd.DataFrame(all_index_rows)
    summary = pd.DataFrame(summary_rows)
    manual_all.to_csv(source_dir / "all_models_manual_event_coverage.csv", index=False, encoding="utf-8-sig")
    candidates_all.to_csv(source_dir / "all_models_label_external_predictions.csv", index=False, encoding="utf-8-sig")
    atlas_index.to_csv(output_dir / "event_atlas_index.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(source_dir / "event_atlas_counts.csv", index=False, encoding="utf-8-sig")
    _plot_summary(summary, output_dir / "event_atlas_summary", selected_models)
    _write_readme(output_dir, summary, config_by_model={
        model: json.loads(
            (run_root / "models" / model / event_directory_name / "decoder_fit.json").read_text(encoding="utf-8")
        )["config"]
        for model in selected_models
    }, models=selected_models, event_directory_name=event_directory_name)
    _write_qa(
        output_dir,
        summary,
        len(manual),
        len(candidates_all),
        models=selected_models,
        event_directory_name=event_directory_name,
    )
    return output_dir


def _decode_probability_tracks(
    predictions: pd.DataFrame, config: DecoderConfig
) -> dict[tuple[str, str, str], pd.DataFrame]:
    output: dict[tuple[str, str, str], pd.DataFrame] = {}
    for key, group in predictions.groupby(KEYS, sort=True):
        ordered = group.sort_values("start_time_ms", kind="stable").copy()
        # Corrected audit tracks already contain the interval-aware decoder
        # output.  Preserve it verbatim so the historical renderer can be
        # reused without silently reintroducing centre-based boundaries.
        if "interval_aware_mask" in ordered.columns:
            if "raw_probability" not in ordered.columns:
                source = "split_safe_probability" if "split_safe_probability" in ordered else "seizure_probability"
                ordered["raw_probability"] = pd.to_numeric(ordered[source], errors="coerce")
            if "smoothed_probability" not in ordered.columns:
                ordered["smoothed_probability"] = ordered["raw_probability"]
            if "decoded_mask" not in ordered.columns:
                ordered["decoded_mask"] = ordered["interval_aware_mask"].astype(np.int8)
            if "center_time_ms" not in ordered.columns:
                ordered["center_time_ms"] = (
                    ordered["window_interval_start_ms"].astype(float)
                    + ordered["window_interval_end_ms"].astype(float)
                ) / 2.0
            output[tuple(map(str, key))] = ordered
            continue
        annotation = decode_window_probabilities(
            ordered["split_safe_probability"].to_numpy(float),
            ordered["start_time_ms"].to_numpy(float),
            config,
        )
        ordered["raw_probability"] = annotation.probability
        ordered["smoothed_probability"] = annotation.smoothed_probability
        ordered["decoded_mask"] = annotation.seizure_mask.astype(np.int8)
        output[tuple(map(str, key))] = ordered
    return output


def _validate_decoded_intervals(
    decoded: dict[tuple[str, str, str], pd.DataFrame], predicted: pd.DataFrame
) -> None:
    # The persisted intervals remain the source of truth. This check confirms that
    # re-decoding for the displayed mask used the same number of recording streams.
    predicted_keys = {tuple(map(str, key)) for key in predicted.groupby(KEYS).groups}
    if not predicted_keys.issubset(decoded):
        missing = sorted(predicted_keys - set(decoded))
        raise ValueError(f"predicted events have no probability track: {missing[:3]}")


def _interval_overlap(frame: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    return frame.loc[frame["end_time_ms"].gt(start) & frame["start_time_ms"].lt(end)]


def _build_manual_event_table(
    manual: pd.DataFrame,
    assessments: pd.DataFrame,
    predicted: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    assessment_columns = KEYS + [
        "event_id",
        "annotated_duration_ms",
        "covered_duration_ms",
        "coverage",
        "overlapping_predicted_segments",
    ]
    result = manual.merge(
        assessments[assessment_columns],
        on=KEYS + ["event_id"],
        how="left",
        validate="one_to_one",
    )
    if result["coverage"].isna().any():
        raise ValueError("manual events and assessment rows do not align")
    rows: list[dict[str, Any]] = []
    predicted_groups = {
        tuple(map(str, key)): group.sort_values("start_time_ms", kind="stable")
        for key, group in predicted.groupby(KEYS, sort=False)
    }
    for event in result.itertuples(index=False):
        key = (str(event.animal_id), str(event.session_id), str(event.channel_name))
        group = predicted_groups.get(key, predicted.iloc[0:0])
        overlaps = _interval_overlap(group, float(event.start_time_ms), float(event.end_time_ms))
        record = event._asdict()
        coverage = float(event.coverage)
        if coverage >= 0.9:
            category = FULL
        elif coverage >= 0.5:
            category = PARTIAL
        elif coverage > 0.0:
            category = LOW
        else:
            category = MISSED
        record["model"] = model_name
        record["model_label"] = MODEL_LABELS[model_name]
        record["primary_category"] = category
        record["long_missed"] = bool(coverage == 0.0 and float(event.duration_s) >= 30.0)
        if overlaps.empty:
            record.update(
                {
                    "prediction_envelope_start_ms": np.nan,
                    "prediction_envelope_end_ms": np.nan,
                    "envelope_onset_error_s": np.nan,
                    "envelope_offset_error_s": np.nan,
                    "maximum_pairwise_iou": 0.0,
                    "boundary_shift": False,
                }
            )
        else:
            pred_start = float(overlaps["start_time_ms"].min())
            pred_end = float(overlaps["end_time_ms"].max())
            onset_error = (pred_start - float(event.start_time_ms)) / 1000.0
            offset_error = (pred_end - float(event.end_time_ms)) / 1000.0
            intersection = np.maximum(
                0.0,
                np.minimum(overlaps["end_time_ms"].to_numpy(float), float(event.end_time_ms))
                - np.maximum(overlaps["start_time_ms"].to_numpy(float), float(event.start_time_ms)),
            )
            union = (
                overlaps["end_time_ms"].to_numpy(float)
                - overlaps["start_time_ms"].to_numpy(float)
                + float(event.end_time_ms)
                - float(event.start_time_ms)
                - intersection
            )
            record.update(
                {
                    "prediction_envelope_start_ms": pred_start,
                    "prediction_envelope_end_ms": pred_end,
                    "envelope_onset_error_s": onset_error,
                    "envelope_offset_error_s": offset_error,
                    "maximum_pairwise_iou": float(np.max(intersection / np.maximum(union, 1e-9))),
                    "boundary_shift": bool(max(abs(onset_error), abs(offset_error)) > 2.0),
                }
            )
        rows.append(record)
    return pd.DataFrame(rows)


def _build_external_candidate_table(
    manual: pd.DataFrame,
    predicted: pd.DataFrame,
    probabilities: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    manual_groups = {
        tuple(map(str, key)): group for key, group in manual.groupby(KEYS, sort=False)
    }
    probability_groups = {
        tuple(map(str, key)): group for key, group in probabilities.groupby(KEYS, sort=False)
    }
    rows: list[dict[str, Any]] = []
    for candidate in predicted.itertuples(index=False):
        key = (str(candidate.animal_id), str(candidate.session_id), str(candidate.channel_name))
        truth = manual_groups.get(key, manual.iloc[0:0])
        overlap = _interval_overlap(truth, float(candidate.start_time_ms), float(candidate.end_time_ms))
        if not overlap.empty:
            continue
        probability = probability_groups.get(key, probabilities.iloc[0:0])
        inside = probability.loc[
            probability["center_time_ms"].ge(float(candidate.start_time_ms))
            & probability["center_time_ms"].lt(float(candidate.end_time_ms))
        ]
        record = candidate._asdict()
        record["model"] = model_name
        record["model_label"] = MODEL_LABELS[model_name]
        record["primary_category"] = OUTSIDE
        record["peak_raw_probability"] = (
            float(inside["split_safe_probability"].max()) if not inside.empty else np.nan
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _sort_review_frame(frame: pd.DataFrame, category: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if category == OUTSIDE:
        columns = ["peak_raw_probability", "duration_s"]
        ascending = [False, False]
    elif category in {MISSED, LONG_MISSED}:
        columns = ["duration_s", "event_id"]
        ascending = [False, True]
    elif category == BOUNDARY_SHIFT:
        frame = frame.assign(
            _boundary_error=frame[["envelope_onset_error_s", "envelope_offset_error_s"]]
            .abs()
            .max(axis=1)
        )
        columns = ["_boundary_error", "event_id"]
        ascending = [False, True]
    else:
        columns = ["coverage", "duration_s", "event_id"]
        ascending = [False, False, True]
    return frame.sort_values(columns, ascending=ascending, kind="stable").reset_index(drop=True)


def _select_exemplar(frame: pd.DataFrame, category: str) -> pd.Series:
    if category == FULL:
        boundary_error = (
            frame[["envelope_onset_error_s", "envelope_offset_error_s"]]
            .abs()
            .max(axis=1)
            .fillna(np.inf)
        )
        return frame.iloc[int(boundary_error.to_numpy().argmin())]
    if category in {MISSED, LONG_MISSED, BOUNDARY_SHIFT, OUTSIDE}:
        return frame.iloc[0]
    target = float(frame["coverage"].median())
    return frame.iloc[(frame["coverage"] - target).abs().to_numpy().argmin()]


def _write_category_pdf(
    path: Path,
    frame: pd.DataFrame,
    category: str,
    model_name: str,
    manual: pd.DataFrame,
    predicted: pd.DataFrame,
    decoded: dict[tuple[str, str, str], pd.DataFrame],
    traces: TraceStore,
    config: DecoderConfig,
) -> list[dict[str, Any]]:
    index_rows: list[dict[str, Any]] = []
    with PdfPages(path) as pdf:
        if frame.empty:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.5, 0.53, CATEGORY_LABELS[category], ha="center", fontsize=12, fontweight="bold")
            fig.text(0.5, 0.48, "No events in this category", ha="center", color=MID_GREY)
            pdf.savefig(fig)
            plt.close(fig)
            return index_rows
        total_pages = math.ceil(len(frame) / 2)
        for page_index, page_start in enumerate(range(0, len(frame), 2), start=1):
            page = frame.iloc[page_start : page_start + 2]
            fig = plt.figure(figsize=(8.27, 11.69))
            outer = fig.add_gridspec(2, 1, left=0.08, right=0.965, top=0.945, bottom=0.055, hspace=0.28)
            for panel_index, (_, entry) in enumerate(page.iterrows()):
                axes = _add_review_axes(fig, outer[panel_index])
                _draw_review_unit(
                    axes,
                    entry,
                    category,
                    manual,
                    predicted,
                    decoded,
                    traces,
                    config,
                    panel_label="ab"[panel_index],
                )
                index_rows.append(
                    {
                        "model": model_name,
                        "category": category,
                        "entity": "predicted_interval" if category == OUTSIDE else "manual_event",
                        "animal_id": str(entry["animal_id"]),
                        "session_id": str(entry["session_id"]),
                        "channel_name": str(entry["channel_name"]),
                        "event_id": str(entry.get("event_id", "")),
                        "predicted_event_index": entry.get("predicted_event_index", pd.NA),
                        "pdf": str(path.relative_to(path.parents[1])).replace("\\", "/"),
                        "page": page_index,
                        "panel": "ab"[panel_index],
                    }
                )
            fig.suptitle(
                f"{MODEL_LABELS[model_name]} | {CATEGORY_LABELS[category]} | "
                f"{page_start + 1}-{page_start + len(page)} / {len(frame)}",
                x=0.08,
                y=0.985,
                ha="left",
                fontsize=9.5,
                fontweight="bold",
                color=MODEL_COLORS[model_name],
            )
            fig.text(0.965, 0.982, f"page {page_index}/{total_pages}", ha="right", va="top", fontsize=6.5, color=MID_GREY)
            fig.text(
                0.08,
                0.018,
                "Blue = manual label; orange = persisted model interval/mask; grey = raw probability; "
                "green = decoder-smoothed probability. Amplitude uses training-baseline a.u.",
                fontsize=5.8,
                color=MID_GREY,
            )
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
    return index_rows


def _add_review_axes(fig: plt.Figure, slot: Any) -> tuple[plt.Axes, plt.Axes, plt.Axes]:
    grid = slot.subgridspec(3, 1, height_ratios=(2.25, 1.25, 0.48), hspace=0.08)
    waveform = fig.add_subplot(grid[0])
    probability = fig.add_subplot(grid[1], sharex=waveform)
    mask = fig.add_subplot(grid[2], sharex=waveform)
    return waveform, probability, mask


def _draw_review_unit(
    axes: tuple[plt.Axes, plt.Axes, plt.Axes],
    entry: pd.Series,
    category: str,
    manual: pd.DataFrame,
    predicted: pd.DataFrame,
    decoded: dict[tuple[str, str, str], pd.DataFrame],
    traces: TraceStore,
    config: DecoderConfig,
    *,
    panel_label: str,
) -> None:
    waveform_axis, probability_axis, mask_axis = axes
    key = (str(entry["animal_id"]), str(entry["session_id"]), str(entry["channel_name"]))
    track = decoded[key]
    truth_group = manual.loc[
        manual["animal_id"].astype(str).eq(key[0])
        & manual["session_id"].astype(str).eq(key[1])
        & manual["channel_name"].astype(str).eq(key[2])
    ]
    prediction_group = predicted.loc[
        predicted["animal_id"].astype(str).eq(key[0])
        & predicted["session_id"].astype(str).eq(key[1])
        & predicted["channel_name"].astype(str).eq(key[2])
    ]

    if category == OUTSIDE:
        focus_start = float(entry["start_time_ms"])
        focus_end = float(entry["end_time_ms"])
        true_start = true_end = None
        reference = focus_start
    else:
        true_start = float(entry["start_time_ms"])
        true_end = float(entry["end_time_ms"])
        focus_start, focus_end = true_start, true_end
        reference = true_start
    overlapping_prediction = _interval_overlap(prediction_group, focus_start, focus_end)
    span_start = min([focus_start, *overlapping_prediction["start_time_ms"].astype(float).tolist()])
    span_end = max([focus_end, *overlapping_prediction["end_time_ms"].astype(float).tolist()])
    margin_ms = max(12_000.0, 0.08 * (span_end - span_start))
    crop_start = span_start - margin_ms
    crop_end = span_end + margin_ms
    crop_predictions = _interval_overlap(prediction_group, crop_start, crop_end)
    crop_truth = _interval_overlap(truth_group, crop_start, crop_end)
    crop_track = track.loc[
        track["center_time_ms"].ge(crop_start) & track["center_time_ms"].le(crop_end)
    ]
    time_ms, waveform = traces.load(*key, crop_start, crop_end)
    x_wave = (time_ms - reference) / 1000.0
    display_x, lower, upper = _waveform_display(x_wave, waveform, maximum_points=6500)
    if lower is None:
        waveform_axis.plot(display_x, upper, color=INK, linewidth=0.5)
    else:
        waveform_axis.fill_between(display_x, lower, upper, color=INK, linewidth=0, alpha=0.82)

    for event in crop_truth.itertuples(index=False):
        clipped_start = max(float(event.start_time_ms), crop_start)
        clipped_end = min(float(event.end_time_ms), crop_end)
        for axis in (waveform_axis, probability_axis):
            axis.axvspan(
                (clipped_start - reference) / 1000.0,
                (clipped_end - reference) / 1000.0,
                color=MANUAL_COLOR,
                alpha=0.12,
                linewidth=0,
            )
    for event in crop_predictions.itertuples(index=False):
        clipped_start = max(float(event.start_time_ms), crop_start)
        clipped_end = min(float(event.end_time_ms), crop_end)
        for axis in (waveform_axis, probability_axis):
            axis.axvspan(
                (clipped_start - reference) / 1000.0,
                (clipped_end - reference) / 1000.0,
                facecolor="none",
                edgecolor=PREDICTED_COLOR,
                linewidth=0.8,
            )

    probability_x = (crop_track["center_time_ms"].to_numpy(float) - reference) / 1000.0
    probability_axis.plot(
        probability_x,
        crop_track["raw_probability"],
        color=RAW_COLOR,
        linewidth=0.55,
        marker="o",
        markersize=1.6,
        label="Raw",
    )
    probability_axis.plot(
        probability_x,
        crop_track["smoothed_probability"],
        color=SMOOTHED_COLOR,
        linewidth=1.0,
        label="Smoothed",
    )
    probability_axis.axhline(config.high_threshold, color="#4B5155", linestyle="--", linewidth=0.6)
    probability_axis.axhline(config.low_threshold, color="#92989C", linestyle=":", linewidth=0.6)
    probability_axis.set(ylim=(-0.02, 1.02), yticks=(0, 0.5, 1), ylabel="P(seizure)")
    probability_axis.legend(loc="upper right", ncol=2, frameon=False)

    mask_axis.fill_between(
        probability_x,
        0,
        crop_track["decoded_mask"].to_numpy(float),
        step="mid",
        color=MASK_COLOR,
        alpha=0.78,
        linewidth=0,
    ) if "interval_aware_mask" not in crop_track.columns else None
    if "interval_aware_mask" in crop_track.columns:
        # A positive decision supports the complete physical EEG window. Draw
        # the union of those windows, rather than a one-stride centre trace.
        positive = crop_track.loc[crop_track["decoded_mask"].astype(bool)]
        spans: list[tuple[float, float]] = []
        for row in positive.itertuples(index=False):
            left = max(float(row.window_interval_start_ms), crop_start)
            right = min(float(row.window_interval_end_ms), crop_end)
            if right <= left:
                continue
            if spans and left <= spans[-1][1] + 1e-6:
                spans[-1] = (spans[-1][0], max(spans[-1][1], right))
            else:
                spans.append((left, right))
        for left, right in spans:
            mask_axis.axvspan(
                (left - reference) / 1000.0,
                (right - reference) / 1000.0,
                color=MASK_COLOR,
                alpha=0.78,
                linewidth=0,
            )
    mask_axis.set(ylim=(0, 1), yticks=(0, 1), yticklabels=("off", "on"), ylabel="Decoded", xlabel="Time from manual onset or candidate onset (s)")

    if true_start is not None and true_end is not None:
        for axis in axes:
            axis.axvline(0.0, color=MANUAL_COLOR, linestyle="--", linewidth=0.85)
            axis.axvline((true_end - reference) / 1000.0, color=MANUAL_COLOR, linestyle="--", linewidth=0.85)

    metadata = _metadata_text(entry, category)
    waveform_axis.set_title(
        metadata,
        loc="left",
        fontsize=6.2,
        fontweight="bold",
        wrap=True,
        pad=4,
    )
    waveform_axis.set_ylabel("Amplitude (a.u.)")
    waveform_axis.tick_params(labelbottom=False)
    probability_axis.tick_params(labelbottom=False)
    waveform_axis.text(-0.045, 1.08, panel_label, transform=waveform_axis.transAxes, fontsize=9, fontweight="bold", va="top")
    x_limits = (
        (crop_start - reference) / 1000.0,
        (crop_end - reference) / 1000.0,
    )
    for axis in axes:
        axis.set_xlim(x_limits)
        axis.tick_params(direction="out", length=2.5, pad=2)


def _metadata_text(entry: pd.Series, category: str) -> str:
    identity = f"{entry['animal_id']} | {entry['session_id']} | blinded {entry['channel_name']}"
    if category == OUTSIDE:
        return (
            f"{identity} | candidate {int(entry['predicted_event_index'])}\n"
            f"duration {float(entry['duration_s']):.1f} s | no manual overlap | "
            f"peak raw P {float(entry['peak_raw_probability']):.2f}"
        )
    onset = entry.get("envelope_onset_error_s", np.nan)
    offset = entry.get("envelope_offset_error_s", np.nan)
    errors = (
        "onset/offset n.a."
        if pd.isna(onset) or pd.isna(offset)
        else f"envelope onset/offset {float(onset):+.1f}/{float(offset):+.1f} s"
    )
    return (
        f"{identity} | {entry['event_id']}\n"
        f"duration {float(entry['duration_s']):.1f} s | "
        f"coverage {float(entry['coverage']):.2f} | max pair IoU {float(entry['maximum_pairwise_iou']):.2f} | "
        f"segments {int(entry['overlapping_predicted_segments'])} | {errors}"
    )


def _write_exemplar(
    stem: Path,
    entry: pd.Series,
    category: str,
    model_name: str,
    manual: pd.DataFrame,
    predicted: pd.DataFrame,
    decoded: dict[tuple[str, str, str], pd.DataFrame],
    traces: TraceStore,
    config: DecoderConfig,
) -> None:
    # Reserve a dedicated header band for the model/category title.  The
    # waveform panel carries a two-line event metadata title of its own; a
    # tight top margin made the two texts collide in the previous exemplars.
    # Keep the exemplar proportions aligned with the established clinical
    # review plate: the title and two-line event metadata sit immediately
    # above the waveform, leaving no oversized blank header band.  The
    # previous ``top=0.77`` setting pushed the metadata far away from the
    # traces and made the PNGs look unlike the validated v2 reference.
    # The slightly wider/shorter canvas compensates for the current
    # Matplotlib tight-bbox metrics and reproduces the proportions of the
    # validated v2 review plate.
    fig = plt.figure(figsize=(7.62, 3.70))
    grid = fig.add_gridspec(
        3,
        1,
        left=0.08,
        right=0.95,
        top=0.88,
        bottom=0.11,
        height_ratios=(2.35, 1.25, 0.48),
        hspace=0.10,
    )
    axes = tuple(fig.add_subplot(grid[index], sharex=None if index == 0 else None) for index in range(3))
    axes[1].sharex(axes[0])
    axes[2].sharex(axes[0])
    _draw_review_unit(axes, entry, category, manual, predicted, decoded, traces, config, panel_label="")
    fig.suptitle(
        f"{MODEL_LABELS[model_name]} | {CATEGORY_LABELS[category]}",
        x=0.08,
        y=0.975,
        ha="left",
        fontsize=8.5,
        fontweight="bold",
        color=MODEL_COLORS[model_name],
    )
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _plot_summary(
    summary: pd.DataFrame, stem: Path, models: tuple[str, ...]
) -> None:
    categories = list(PRIMARY_CATEGORIES) + [LONG_MISSED, BOUNDARY_SHIFT, OUTSIDE]
    labels = [
        "coverage >=0.9",
        "0.5-0.9",
        "0-0.5",
        "missed",
        ">=30 s missed",
        ">2 s boundary shift",
        "label-external",
    ]
    figure_width = max(3.5, 2.35 * len(models))
    fig, axes = plt.subplots(
        1, len(models), figsize=(figure_width, 2.75), sharey=True, squeeze=False
    )
    for model_index, (axis, model) in enumerate(zip(axes[0], models, strict=True)):
        values = summary.loc[summary["model"].eq(model)].set_index("category")["count"]
        counts = [int(values.get(category, 0)) for category in categories]
        y = np.arange(len(categories))
        axis.barh(y, counts, color=MODEL_COLORS[model], alpha=0.9)
        for position, value in zip(y, counts, strict=True):
            axis.text(value + max(counts + [1]) * 0.025, position, str(value), va="center", fontsize=6.2)
        axis.set_title(MODEL_LABELS[model], loc="left", fontweight="bold")
        axis.set_xlabel("Intervals")
        axis.set_yticks(y)
        if model_index == 0:
            axis.set_yticklabels(labels)
            axis.tick_params(labelleft=True)
        else:
            # The y-axis is shared; replacing the formatter on a later axis
            # would also erase labels from the first axis.
            axis.tick_params(labelleft=False)
        axis.invert_yaxis()
        axis.tick_params(direction="out", length=2.5)
    fig.suptitle("Held-out event coverage and label-external predictions", x=0.11, ha="left", fontsize=9, fontweight="bold")
    denominators = []
    for model in models:
        values = summary.loc[summary["model"].eq(model)].set_index("category")["count"]
        denominators.append(int(sum(values.get(category, 0) for category in PRIMARY_CATEGORIES)))
    denominator_text = str(denominators[0]) if denominators and len(set(denominators)) == 1 else ", ".join(map(str, denominators))
    fig.text(0.11, 0.015, f"Primary coverage classes are mutually exclusive over {denominator_text} held-out manual channel-events per model. Long misses and boundary shifts are flagged subsets.", fontsize=5.8, color=MID_GREY)
    fig.subplots_adjust(left=0.20, right=0.985, top=0.82, bottom=0.20, wspace=0.28)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    # Keep the compact summary submission-ready without rasterizing every atlas page.
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def _write_readme(
    output_dir: Path,
    summary: pd.DataFrame,
    config_by_model: dict[str, Any],
    *,
    models: tuple[str, ...],
    event_directory_name: str,
) -> None:
    corrected = event_directory_name == "events_corrected_v1_test"
    title = "Corrected interval-aware held-out event atlas" if corrected else "Strict held-out event atlas"
    scope_note = (
        "The release is self-contained. Read `corrected_atlas_manifest.json` and `DENOMINATOR_NOTE_zh.txt` "
        "for denominators, and consult the adjacent metrics report for split, purge and alignment status."
        if corrected
        else "These figures audit the frozen outputs supplied by the caller; consult that run's protocol and denominator manifests before making generalization claims."
    )
    lines = [
        f"# {title}",
        "",
        "This folder contains the complete test-set visual audit. Each valid EEG channel is an independent model stream. E77 CH7 is excluded.",
        f"The displayed decoded intervals and probabilities are read from each model's `{event_directory_name}` output; this is the frozen validation-selected re-decoding used for the audit.",
        scope_note,
        "",
        "## Reading order",
        "",
        "1. Open `event_atlas_summary.png` for the category counts.",
        "2. Open `<model>/example_<category>.png` for a one-event example.",
        "3. Open the corresponding category PDF for every event/candidate.",
        "4. Use `event_atlas_index.csv` to locate an event by model, ID, PDF page and panel.",
        "",
        "## Definitions",
        "",
        "- Coverage is the fraction of a manual interval covered by the union of all overlapping decoded model intervals.",
        "- Primary classes are mutually exclusive: coverage >=0.9; 0.5-0.9; 0-0.5; and coverage=0.",
        "- `>=30 s missed` is a subset of complete misses.",
        "- A count of zero for `>=30 s missed` means no held-out manual event lasting at least 30 s was completely missed under this decoder; it is not an omitted category.",
        "- Boundary shift is a subset of detected events whose prediction-envelope onset or offset differs from the manual boundary by more than 2 s, the half-window temporal uncertainty.",
        "- Label-external predictions have zero positive-time overlap with every manual label in the same animal, recording and channel. They are review candidates, not proof of either model error or missing labels.",
        "- Maximum pair IoU is descriptive only. Coverage classification does not use the previous Hungarian IoU>0 matching rule.",
        "- Raw and smoothed probabilities use the split-safe test trace. The orange decoded mask is reconstructed with the frozen validation-selected decoder.",
        "- EEG amplitude is relative to the training-background median/MAD and remains a.u.; ADC-to-microvolt calibration is unavailable.",
        "",
        "## Counts",
        "",
        "| model | full | partial | low | missed | >=30 s missed | boundary shift | label-external |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        values = summary.loc[summary["model"].eq(model)].set_index("category")["count"]
        lines.append(
            "| {label} | {full} | {partial} | {low} | {missed} | {long} | {shift} | {outside} |".format(
                label=MODEL_LABELS[model],
                full=int(values.get(FULL, 0)),
                partial=int(values.get(PARTIAL, 0)),
                low=int(values.get(LOW, 0)),
                missed=int(values.get(MISSED, 0)),
                long=int(values.get(LONG_MISSED, 0)),
                shift=int(values.get(BOUNDARY_SHIFT, 0)),
                outside=int(values.get(OUTSIDE, 0)),
            )
        )
    lines.extend(["", "## Frozen decoder settings", "", "```json", json.dumps(config_by_model, indent=2), "```", ""])
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_qa(
    output_dir: Path,
    summary: pd.DataFrame,
    manual_events: int,
    candidates: int,
    *,
    models: tuple[str, ...],
    event_directory_name: str,
) -> None:
    corrected = event_directory_name == "events_corrected_v1_test"
    qa = {
        "schema_version": 1,
        "core_conclusion": "The event atlas distinguishes complete recovery, partial coverage, complete misses, boundary shifts, and zero-label-overlap predictions without conflating any-overlap matching with full detection.",
        "archetype": "clinical waveform evidence atlas plus compact quantitative index",
        "backend": "Python/matplotlib only",
        "models_rendered": list(models),
        "event_directory_name": event_directory_name,
        "protocol_status": (
            "strict purged, validation-selected corrected-v1 held-out audit"
            if corrected
            else "caller-supplied frozen held-out audit"
        ),
        "final_protocol": "epilepsy_blind_single_channel_v3_strict_purged",
        "test_manual_channel_events_per_model": manual_events,
        "label_external_candidates_across_models": candidates,
        "primary_category_partition_check": {
            model: int(
                summary.loc[
                    summary["model"].eq(model) & summary["category"].isin(PRIMARY_CATEGORIES),
                    "count",
                ].sum()
            )
            for model in models
        },
        "boundary_shift_threshold_seconds": 2.0,
        "final_width_note": (
            "The compact summary is approximately double-column width; category PDFs are intentionally A4 portrait "
            "for indexed clinical review rather than manuscript placement."
        ),
        "scope_note": (
            "Figures audit the frozen corrected-v1 outputs in this release; consult corrected_atlas_manifest.json, "
            "DENOMINATOR_NOTE_zh.txt and the adjacent metrics report for split-boundary purge, denominator and alignment status."
            if corrected
            else "Figures audit the frozen outputs supplied by the caller; consult its protocol and denominator manifests before making generalization claims."
        ),
        "matching_note": "Coverage uses union overlap; maximum pair IoU is descriptive; category membership does not use Hungarian assignment.",
        "source_trace": "complete filtered cache with training-background median/MAD normalization",
        "amplitude_unit": "a.u.",
        "exports": "summary: PNG/PDF/SVG/TIFF; exemplars: PNG/PDF/SVG; complete categories: indexed multipage PDF",
    }
    (output_dir / "figure_qa_notes.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["generate_event_coverage_atlas"]
