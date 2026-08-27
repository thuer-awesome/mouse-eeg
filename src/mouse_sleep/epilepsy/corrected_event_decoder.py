"""Leakage-safe, interval-aware continuous event decoding.

This module is intentionally separate from the historical centre-based
decoder.  It is used for a new audit directory so old model runs and figures
remain reproducible.  Model probabilities and clinical labels are the only
inputs; no animal- or channel-specific operating point is fitted.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from mouse_sleep.epilepsy.seizure_annotation import (
    DecoderConfig,
    EventEvaluation,
    EventInterval,
    EventMatch,
    TrueEventAssessment,
    _pool_event_evaluations,
    decode_window_intervals,
    evaluate_event_annotations,
)


KEYS = ("animal_id", "session_id", "channel_name")


@dataclass(frozen=True)
class CorrectedSequence:
    """One physically continuous single-channel trace segment."""

    key: tuple[str, str, str]
    segment_id: str
    frame: pd.DataFrame
    truth: tuple[EventInterval, ...]
    truth_rows: pd.DataFrame
    recording_duration_seconds: float


def corrected_decoder_candidates() -> tuple[DecoderConfig, ...]:
    """Return a compact, auditable validation-only operating-point grid.

    The grid spans the observed precision/recall trade-off while keeping the
    physical window interval fixed at 4 s.  It deliberately excludes the
    low-threshold, one-window ``repair`` settings that produced large bursts
    of one-to-two-second false candidates in the previous audit.
    """

    candidates: list[DecoderConfig] = []
    # Keep the formal audit rerunnable on the complete combined trace.  The
    # prior 144-point grid included one-window medians and low persistence that
    # recreated the short false-candidate bursts being audited.
    for high in (0.70, 0.85, 0.95):
        for low in (0.10, 0.30):
            for median in (3,):
                for persistence in ((2, 3), (3, 5)):
                    for max_gap in (0.0,):
                        for minimum in (3.0, 6.0):
                            if low >= high:
                                continue
                            candidates.append(
                                DecoderConfig(
                                    high_threshold=high,
                                    low_threshold=low,
                                    median_width_windows=median,
                                    persistence_k=persistence[0],
                                    persistence_n=persistence[1],
                                    max_gap_seconds=max_gap,
                                    min_event_seconds=minimum,
                                    boundary_search_seconds=0.0,
                                )
                            )
    # A stable order makes candidate CSVs and tie-breaking reproducible.
    return tuple(candidates)


def build_sequences(
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    split: str,
) -> tuple[CorrectedSequence, ...]:
    """Build split-safe, physically continuous sequences from a trace table."""

    required_predictions = set(KEYS) | {
        "start_time_ms",
        "end_time_ms",
        "split",
        "split_safe_probability",
    }
    missing = sorted(required_predictions - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions missing required columns: {missing}")
    # ``all`` is an explicit full-recording audit scope.  Treating it as a
    # literal split silently returned an empty sequence and made the previous
    # full-recording decoder report meaningless zero-event tables.
    if split == "all":
        selected = predictions.copy()
    else:
        selected = predictions.loc[predictions["split"].astype(str).eq(split)].copy()
    selected = selected.sort_values([*KEYS, "start_time_ms"], kind="stable")
    event_split = (
        events.copy()
        if split == "all"
        else events.loc[events["split"].astype(str).eq(split)].copy()
    )
    sequences: list[CorrectedSequence] = []
    for key_values, group in selected.groupby(list(KEYS), sort=True):
        key = tuple(map(str, key_values))
        group = group.sort_values("start_time_ms", kind="stable").copy()
        truth_frame = event_split.loc[
            event_split["animal_id"].astype(str).eq(key[0])
            & event_split["session_id"].astype(str).eq(key[1])
            & event_split["channel_name"].astype(str).eq(key[2])
        ].sort_values("start_time_ms", kind="stable")
        segments = tuple(_decoder_segments(group))
        # A four-second sliding window can straddle a retained-window gap.  In
        # that case the same short clinical event may be fully contained by
        # both neighbouring decoder fragments.  Assign each event to exactly
        # one fragment (the one with the greatest physical support) so event
        # denominators and one-to-one matching cannot be inflated by a context
        # boundary.
        assigned_truth = _assign_truth_rows_to_segments(truth_frame, segments)
        for segment_index, (segment_id, segment) in enumerate(segments):
            contained = assigned_truth[segment_index]
            truth = tuple(
                EventInterval(float(row.start_time_ms), float(row.end_time_ms))
                for row in contained.itertuples(index=False)
            )
            sequences.append(
                CorrectedSequence(
                    key=key,
                    segment_id=str(segment_id),
                    frame=segment.copy(),
                    truth=truth,
                    truth_rows=contained.copy(),
                    recording_duration_seconds=_covered_duration_seconds(segment),
                )
            )
    return tuple(sequences)


def evaluate_sequences(
    sequences: Sequence[CorrectedSequence],
    config: DecoderConfig,
    *,
    probability_column: str = "split_safe_probability",
    split_label: str = "test",
    minimum_iou: float = 0.30,
    include_decoded_tracks: bool = True,
) -> dict[str, Any]:
    """Decode and evaluate every sequence, returning auditable tables."""

    evaluations: list[EventEvaluation] = []
    keys: list[tuple[str, str, str, str]] = []
    predicted_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    assessment_rows: list[dict[str, Any]] = []
    for sequence in sequences:
        frame = sequence.frame
        if probability_column not in frame.columns:
            raise ValueError(f"trace is missing probability column {probability_column!r}")
        decoded = decode_window_intervals(
            frame[probability_column].to_numpy(float),
            frame["start_time_ms"].to_numpy(float),
            config,
            frame["end_time_ms"].to_numpy(float),
        )
        evaluation = evaluate_event_annotations(
            sequence.truth,
            decoded.events,
            sequence.recording_duration_seconds,
            minimum_iou=minimum_iou,
        )
        evaluations.append(evaluation)
        key = (*sequence.key, sequence.segment_id)
        keys.append(key)
        for index, event in enumerate(decoded.events):
            predicted_rows.append(
                {
                    **dict(zip(KEYS, sequence.key, strict=True)),
                    "split": split_label,
                    "decoder_segment_id": sequence.segment_id,
                    "predicted_event_index": int(index),
                    "start_time_ms": float(event.start_ms),
                    "end_time_ms": float(event.end_ms),
                    "duration_s": float(event.duration_ms / 1000.0),
                    "interval_support": "full_model_window_union",
                    "window_seconds": float(config.window_seconds),
                }
            )
        for match in evaluation.matches:
            true_row = sequence.truth_rows.iloc[match.true_index]
            match_rows.append(_match_row(sequence, match, true_row, split_label, minimum_iou))
        for assessment in evaluation.true_event_assessments:
            true_row = sequence.truth_rows.iloc[assessment.true_index]
            assessment_rows.append(_assessment_row(sequence, assessment, true_row, split_label))

    pooled = _pool_event_evaluations(evaluations)
    per_animal: dict[str, Any] = {}
    for animal in sorted({key[0] for key in keys}):
        subset = [item for key, item in zip(keys, evaluations, strict=True) if key[0] == animal]
        per_animal[animal] = _json_clean(_pool_event_evaluations(subset))
    per_channel: dict[str, Any] = {}
    for key, evaluation in zip(keys, evaluations, strict=True):
        per_channel[":".join(key)] = _json_clean(evaluation.metrics)
    return {
        "metrics": {
            "pooled": _json_clean(pooled),
            "per_animal": per_animal,
            "per_recording_channel": per_channel,
        },
        "predicted_events": pd.DataFrame(predicted_rows),
        "matches": pd.DataFrame(match_rows),
        "true_assessments": pd.DataFrame(assessment_rows),
        "decoded_tracks": (
            _decoded_tracks(sequences, config, probability_column)
            if include_decoded_tracks
            else {}
        ),
    }


def fit_corrected_config(
    validation_sequences: Sequence[CorrectedSequence],
    candidates: Sequence[DecoderConfig] | None = None,
    *,
    minimum_sensitivity: float = 0.85,
    minimum_precision: float = 0.15,
    maximum_false_positive_events_per_hour: float = 40.0,
    minimum_iou: float = 0.30,
) -> tuple[DecoderConfig, dict[str, Any], pd.DataFrame]:
    """Fit one shared decoder using validation-only clinical constraints."""

    if not validation_sequences:
        raise ValueError("validation_sequences cannot be empty")
    if not 0.0 <= minimum_sensitivity <= 1.0:
        raise ValueError("minimum_sensitivity must lie in [0, 1]")
    if not 0.0 <= minimum_precision <= 1.0:
        raise ValueError("minimum_precision must lie in [0, 1]")
    if maximum_false_positive_events_per_hour <= 0.0:
        raise ValueError("maximum_false_positive_events_per_hour must be positive")
    candidates = tuple(candidates or corrected_decoder_candidates())
    rows: list[dict[str, Any]] = []
    results: list[tuple[DecoderConfig, dict[str, Any]]] = []
    for index, config in enumerate(candidates):
        evaluated = evaluate_sequences(
            validation_sequences,
            config,
            split_label="val",
            minimum_iou=minimum_iou,
            include_decoded_tracks=False,
        )
        metrics = evaluated["metrics"]["pooled"]
        sensitivity = _finite_metric(metrics, "event_sensitivity", default=0.0)
        precision = _finite_metric(metrics, "event_precision", default=0.0)
        false_positive_rate = _finite_metric(
            metrics, "false_positive_events_per_hour", default=np.inf
        )
        row = {
            **asdict(config),
            "candidate_index": int(index),
            **{f"metric_{key}": value for key, value in metrics.items()},
            "meets_sensitivity_floor": bool(sensitivity >= minimum_sensitivity),
            "meets_precision_floor": bool(precision >= minimum_precision),
            "meets_false_positive_cap": bool(false_positive_rate <= maximum_false_positive_events_per_hour),
        }
        row["eligible"] = bool(
            row["meets_sensitivity_floor"]
            and row["meets_precision_floor"]
            and row["meets_false_positive_cap"]
        )
        rows.append(_json_clean(row))
        results.append((config, metrics))
    table = pd.DataFrame(rows)
    eligible_indices = [index for index, row in enumerate(rows) if bool(row["eligible"])]
    if eligible_indices:
        pool = eligible_indices
        fallback = None
    else:
        # If a model cannot satisfy all three floors, retain the least-risk
        # operating point and record the fallback instead of silently choosing
        # a low-threshold high-recall decoder.
        pool = list(range(len(rows)))
        fallback = "no_candidate_met_all_validation_constraints"

    def _strict_f1_key(iou: float) -> str:
        # Metric keys use the stable ``0p3`` spelling (rather than relying on
        # Python's float string formatting in several call sites).
        token = f"{float(iou):.6f}".rstrip("0").rstrip(".").replace(".", "p")
        return f"f1_iou_ge_{token}"

    strict_f1_key = _strict_f1_key(minimum_iou)

    def rank(index: int) -> tuple[float, ...]:
        metrics = results[index][1]
        sensitivity = _finite_metric(metrics, "event_sensitivity", default=0.0)
        precision = _finite_metric(metrics, "event_precision", default=0.0)
        false_positive_rate = _finite_metric(
            metrics, "false_positive_events_per_hour", default=np.inf
        )
        violations = (
            max(0.0, minimum_sensitivity - sensitivity),
            max(0.0, minimum_precision - precision),
            max(0.0, false_positive_rate - maximum_false_positive_events_per_hour),
        )
        return (
            sum(violations),
            violations[0],
            violations[1],
            violations[2],
            _finite_metric(metrics, "long_events_ge_30s_zero_coverage", default=np.inf),
            _finite_metric(metrics, "low_coverage_events_lt_0p5", default=np.inf),
            -_finite_metric(
                metrics,
                strict_f1_key,
                default=_finite_metric(metrics, "event_f1", default=-np.inf),
            ),
            -_finite_metric(metrics, "event_f1", default=-np.inf),
            -_finite_metric(metrics, "mean_iou", default=-np.inf),
            false_positive_rate,
            float(results[index][0].high_threshold),
            float(results[index][0].low_threshold),
            int(results[index][0].median_width_windows),
            float(results[index][0].max_gap_seconds),
            float(results[index][0].min_event_seconds),
        )

    selected_index = min(pool, key=rank)
    selected_config, selected_metrics = results[selected_index]
    selected_row = table.loc[table["candidate_index"].eq(selected_index), "eligible"]
    table["selected"] = table["candidate_index"].eq(selected_index)
    record = {
        "schema_version": 1,
        "decoder_kind": "interval_aware_full_window_union",
        "fit_split": "val",
        "test_used_for_selection": False,
        "primary_match_iou": float(minimum_iou),
        "minimum_validation_sensitivity": float(minimum_sensitivity),
        "minimum_validation_precision": float(minimum_precision),
        "maximum_validation_false_positive_events_per_hour": float(
            maximum_false_positive_events_per_hour
        ),
        "constraint_satisfied": bool(selected_row.iloc[0]),
        "selection_fallback": fallback,
        "candidate_count": int(len(candidates)),
        "config": asdict(selected_config),
        "validation_metrics": _json_clean(selected_metrics),
        "selection_order": [
            "satisfy validation sensitivity/precision/false-positive constraints",
            "minimize long-event zero coverage",
            "minimize low-coverage events",
            "maximize strict IoU event F1",
            "maximize event F1 and mean IoU",
            "minimize false-positive event rate",
        ],
    }
    return selected_config, record, table


def _finite_metric(metrics: Mapping[str, Any], key: str, *, default: float) -> float:
    """Read a metric without allowing NaN to poison tuple ranking."""

    value = metrics.get(key, default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def write_evaluation_artifacts(
    output_dir: str | Path,
    evaluated: Mapping[str, Any],
    *,
    decoder_record: Mapping[str, Any],
    split: str,
) -> None:
    """Write one split's parquet tables and decoder metadata."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evaluated["predicted_events"].to_parquet(
        output / f"{split}_predicted_events.parquet", index=False
    )
    evaluated["matches"].to_parquet(
        output / f"{split}_event_matches.parquet", index=False
    )
    evaluated["true_assessments"].to_parquet(
        output / f"{split}_true_event_assessments.parquet", index=False
    )
    (output / "decoder_fit.json").write_text(
        json.dumps(_json_clean(decoder_record), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _decoded_tracks(
    sequences: Sequence[CorrectedSequence],
    config: DecoderConfig,
    probability_column: str,
) -> dict[tuple[str, str, str], pd.DataFrame]:
    """Return plotting tracks with interval-aware mask metadata."""

    output: dict[tuple[str, str, str], list[pd.DataFrame]] = {}
    for sequence in sequences:
        frame = sequence.frame.copy()
        decoded = decode_window_intervals(
            frame[probability_column].to_numpy(float),
            frame["start_time_ms"].to_numpy(float),
            config,
            frame["end_time_ms"].to_numpy(float),
        )
        frame["raw_probability"] = decoded.probability
        frame["smoothed_probability"] = decoded.smoothed_probability
        frame["decoded_mask"] = decoded.seizure_mask.astype(np.int8)
        frame["interval_aware_mask"] = True
        frame["window_interval_start_ms"] = decoded.window_start_time_ms
        frame["window_interval_end_ms"] = decoded.window_end_time_ms
        frame["decoder_segment_id"] = sequence.segment_id
        output.setdefault(sequence.key, []).append(frame)
    return {
        key: pd.concat(parts, ignore_index=True).sort_values("start_time_ms", kind="stable")
        for key, parts in output.items()
    }


def _match_row(
    sequence: CorrectedSequence,
    match: EventMatch,
    true_row: Any,
    split: str,
    minimum_iou: float,
) -> dict[str, Any]:
    return {
        **dict(zip(KEYS, sequence.key, strict=True)),
        "split": split,
        "decoder_segment_id": sequence.segment_id,
        "match_minimum_iou": float(minimum_iou),
        "event_id": str(true_row["event_id"]),
        "true_index": int(match.true_index),
        "predicted_event_index": int(match.predicted_index),
        "overlap_ms": float(match.overlap_ms),
        "iou": float(match.iou),
        "dice": float(match.dice),
        "annotated_coverage": float(match.annotated_coverage),
        "onset_error_seconds": float(match.onset_error_seconds),
        "offset_error_seconds": float(match.offset_error_seconds),
        "duration_error_seconds": float(match.duration_error_seconds),
        "duration_ratio": float(match.duration_ratio),
    }


def _assessment_row(
    sequence: CorrectedSequence,
    assessment: TrueEventAssessment,
    true_row: Any,
    split: str,
) -> dict[str, Any]:
    return {
        **dict(zip(KEYS, sequence.key, strict=True)),
        "split": split,
        "decoder_segment_id": sequence.segment_id,
        "event_id": str(true_row["event_id"]),
        "annotated_duration_ms": float(assessment.annotated_duration_ms),
        "covered_duration_ms": float(assessment.covered_duration_ms),
        "coverage": float(assessment.coverage),
        "overlapping_predicted_segments": int(assessment.overlapping_predicted_segments),
    }


def _assign_truth_rows_to_segments(
    truth_frame: pd.DataFrame,
    segments: Sequence[tuple[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, ...]:
    """Assign each track event to at most one physically observed segment.

    Adjacent decoder fragments can overlap in wall-clock support because the
    retained EEG windows are four seconds long even when one centre window is
    missing.  Merely filtering each fragment by its min/max timestamp then
    duplicates short events.  This helper chooses the segment with the largest
    overlap support; ties prefer the segment intersecting more model windows
    and then the nearest segment centre.  Events not fully contained by any
    segment remain excluded from formal metrics.
    """

    if truth_frame.empty:
        return tuple(truth_frame.copy() for _ in segments)
    assigned: list[list[int]] = [[] for _ in segments]
    for event_index, event in truth_frame.reset_index(drop=True).iterrows():
        start = float(event["start_time_ms"])
        end = float(event["end_time_ms"])
        # Keep the actual segment index as a separate field.  The previous
        # implementation selected the rank within this filtered candidate
        # list and used it as a segment index, which assigned events to the
        # wrong fragment whenever earlier fragments were not eligible.
        candidates: list[tuple[float, int, float, int, int]] = []
        for segment_index, (_, segment) in enumerate(segments):
            if segment.empty:
                continue
            seg_start = float(segment["start_time_ms"].min())
            seg_end = float(segment["end_time_ms"].max())
            if start < seg_start - 1e-6 or end > seg_end + 1e-6:
                continue
            overlap = min(end, seg_end) - max(start, seg_start)
            if overlap <= 0.0:
                continue
            starts = segment["start_time_ms"].to_numpy(float)
            ends = segment["end_time_ms"].to_numpy(float)
            window_support = int(((starts < end) & (ends > start)).sum())
            centre_distance = abs((seg_start + seg_end) / 2.0 - (start + end) / 2.0)
            # Maximise overlap, then supporting windows; minimise distance and
            # finally segment index for deterministic output.
            candidates.append(
                (
                    float(overlap),
                    window_support,
                    -float(centre_distance),
                    -segment_index,
                    segment_index,
                )
            )
        if candidates:
            chosen = max(candidates)[-1]
            assigned[chosen].append(int(event_index))
    truth = truth_frame.reset_index(drop=True)
    return tuple(truth.iloc[indexes].reset_index(drop=True) for indexes in assigned)


def _decoder_segments(group: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    """Split only at physical/QC/context discontinuities."""

    ordered = group.sort_values("start_time_ms", kind="stable")
    public_source = ordered.get("context_segment_id")
    # Some persisted v2/v3 traces retain a context column whose values are
    # entirely missing.  In that case ``DataFrame.get`` must not win over the
    # usable continuity ID; otherwise every row is treated as unknown and the
    # decoder silently emits no segment at all.
    if public_source is None or public_source.notna().sum() == 0:
        public_source = ordered.get("continuity_id")
    if public_source is None:
        public_source = pd.Series("legacy_continuity", index=ordered.index, dtype="string")
    boundary_source = ordered.get("inference_context_segment_id")
    if boundary_source is None or boundary_source.notna().sum() == 0:
        boundary_source = public_source
    public_ids = public_source.astype("string")
    boundary_ids = boundary_source.astype("string")
    known = (
        public_ids.notna()
        & public_ids.ne("")
        & boundary_ids.notna()
        & boundary_ids.ne("")
    ).fillna(False).to_numpy(bool)
    if not known.any():
        return
    if "continuity_position" in ordered.columns:
        positions = pd.to_numeric(ordered["continuity_position"], errors="coerce")
        previous = positions.shift()
        adjacent = (positions.notna() & previous.notna() & positions.eq(previous + 1)).to_numpy(bool)
        position_break = ~adjacent
    else:
        starts = ordered["start_time_ms"].to_numpy(float)
        position_break = np.zeros(len(starts), dtype=bool)
        if len(starts) > 1:
            differences = np.diff(starts)
            positive = differences[differences > 0.0]
            if len(positive):
                position_break[1:] = differences > float(np.median(positive)) * 1.5
    position_break[0] = True
    id_changed = boundary_ids.ne(boundary_ids.shift()).fillna(True).to_numpy(bool)
    previous_known = np.r_[False, known[:-1]]
    run_start = (~known) | (~previous_known) | id_changed | position_break
    run_ids = np.cumsum(run_start, dtype=np.int64)
    emitted: dict[str, int] = {}
    for _, segment in ordered.loc[known].groupby(run_ids[known], sort=False):
        source_id = str(public_source.loc[segment.index].iloc[0])
        occurrence = emitted.get(source_id, 0)
        emitted[source_id] = occurrence + 1
        segment_id = source_id if occurrence == 0 else f"{source_id}:fragment{occurrence:04d}"
        yield segment_id, segment


def _covered_duration_seconds(ordered: pd.DataFrame) -> float:
    starts = ordered["start_time_ms"].to_numpy(float)
    ends = ordered["end_time_ms"].to_numpy(float)
    if len(starts) == 0 or np.any(ends <= starts):
        raise ValueError("trace intervals must be finite and positive")
    total = 0.0
    current_start = float(starts[0])
    current_end = float(ends[0])
    for start, end in zip(starts[1:], ends[1:], strict=True):
        start = float(start)
        end = float(end)
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return (total + current_end - current_start) / 1000.0


def _truth_frame_overlapping_coverage(
    truth_frame: pd.DataFrame, ordered: pd.DataFrame
) -> pd.DataFrame:
    if truth_frame.empty:
        return truth_frame.copy()
    start = float(ordered["start_time_ms"].min())
    end = float(ordered["end_time_ms"].max())
    contained = truth_frame["start_time_ms"].ge(start - 1e-6) & truth_frame[
        "end_time_ms"
    ].le(end + 1e-6)
    return truth_frame.loc[contained].reset_index(drop=True)


def _json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


__all__ = [
    "CorrectedSequence",
    "build_sequences",
    "corrected_decoder_candidates",
    "evaluate_sequences",
    "fit_corrected_config",
    "write_evaluation_artifacts",
]
