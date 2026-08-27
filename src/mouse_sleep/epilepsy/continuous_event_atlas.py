"""Scoped, continuity-safe clinical event atlases for the frozen v5 study.

The historical atlas under ``runs/`` mixed a source-label audit with a formal
test result.  This module intentionally has two non-interchangeable scopes:

``formal_test``
    Uses the fully observable formal event denominator and the reviewed trace.
    It is the only scope that can be used for held-out performance reporting.

``clinical_review_all``
    Uses every source test label and the all-valid candidate trace.  It is an
    exhaustive clinical review aid, including labels that are not eligible for
    a formal metric and model-only candidates.  It is not a performance
    denominator.

The decoder segmentation is delegated to ``single_channel_inference`` at run
time.  That is important: probability tracks and pages are split at the same
QC/context/physical continuity boundaries used during event evaluation.  A
page is always clipped to one decoder segment and never draws a waveform or
decoded mask across an omitted window.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.0,
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
from mouse_sleep.epilepsy.single_channel_visualization import (
    TraceStore,
    _waveform_display,
)


AtlasScopeName = Literal["formal_test", "clinical_review_all"]
KEYS = ("animal_id", "session_id", "channel_name")

FULL = "full_coverage_ge_0p9"
PARTIAL = "partial_coverage_0p5_to_0p9"
LOW = "low_overlap_gt_0_to_0p5"
MISSED = "missed_coverage_0"
LONG_MISSED = "missed_ge_30s"
BOUNDARY_SHIFT = "boundary_shift_gt_2s"
OUTSIDE = "label_external_prediction"

PRIMARY_CATEGORIES = (FULL, PARTIAL, LOW, MISSED)
SECONDARY_CATEGORIES = (LONG_MISSED, BOUNDARY_SHIFT, OUTSIDE)
CATEGORY_ORDER = PRIMARY_CATEGORIES + SECONDARY_CATEGORIES
CATEGORY_LABELS = {
    FULL: "Full coverage (>=0.90)",
    PARTIAL: "Partial coverage (0.50-<0.90)",
    LOW: "Low overlap (>0-<0.50)",
    MISSED: "Complete miss (coverage = 0)",
    LONG_MISSED: "Long complete miss (>=30 s)",
    BOUNDARY_SHIFT: "Boundary shift (>2 s)",
    OUTSIDE: "Label-external model candidate",
}

MODEL_LABELS = {
    "eegwavenet_sc": "EEGWaveNet-SC",
    "physio_sequence_time_frequency": "Physio-TimeFrequency",
    "physio_sequence_stats_input": "Physio-StatisticsInput",
    "physio_sequence_stats_aux": "Physio-StatisticsAux",
    # Frozen historic aliases are retained only to make the reader robust when
    # inspecting an already-produced non-formal model directory.
    "physio_sequence_no_aux": "Physio-Sequence (historic)",
    "physio_sequence_stats": "Physio-Sequence+Stats (historic)",
}

INK = "#252A2E"
MID_GREY = "#6C7378"
LIGHT_GREY = "#E6E9EB"
MANUAL_COLOR = "#1E5B8C"
PREDICTED_COLOR = "#C45A32"
RAW_COLOR = "#7B8389"
SMOOTHED_COLOR = "#26746D"
MASK_COLOR = "#C45A32"


@dataclass(frozen=True)
class AtlasScope:
    """Immutable source contract for one atlas type."""

    name: AtlasScopeName
    event_filename: str
    trace_filename: str
    probability_column: str
    denominator_key: str
    reporting_status: str
    external_candidate_definition: str


SCOPES: Mapping[AtlasScopeName, AtlasScope] = {
    "formal_test": AtlasScope(
        name="formal_test",
        event_filename="formal_channel_events.parquet",
        trace_filename="reviewed_supervised_trace.parquet",
        probability_column="split_safe_probability",
        denominator_key="formal_channel_events",
        reporting_status="formal held-out performance denominator",
        external_candidate_definition=(
            "zero overlap with every formal-denominator event in the same "
            "animal, recording, and channel; not automatically a clinical false positive"
        ),
    ),
    "clinical_review_all": AtlasScope(
        name="clinical_review_all",
        event_filename="channel_events.parquet",
        trace_filename="all_valid_candidate_trace.parquet",
        probability_column="seizure_probability",
        denominator_key="source_channel_events",
        reporting_status="clinical review only; not a formal performance denominator",
        external_candidate_definition=(
            "zero overlap with every source clinical event in the same animal, "
            "recording, and channel; candidate requires clinical adjudication"
        ),
    ),
}


@dataclass(frozen=True)
class DecodedAtlas:
    """Decoded traces indexed by the QC/continuity-safe segment identity."""

    segment_frames: Mapping[tuple[str, str, str, str], pd.DataFrame]
    segment_index: pd.DataFrame
    predicted_events: pd.DataFrame


Segmenter = Callable[[pd.DataFrame], Iterable[tuple[str, pd.DataFrame]]]


def scope_specification(scope: AtlasScopeName) -> AtlasScope:
    """Return the explicit input contract for a supported atlas scope."""

    try:
        return SCOPES[scope]
    except KeyError as exc:
        raise ValueError(
            "scope must be one of: " + ", ".join(sorted(SCOPES))
        ) from exc


def generate_continuous_event_atlas(
    protocol_dir: str | Path,
    model_root: str | Path,
    output_dir: str | Path,
    *,
    scope: AtlasScopeName,
    boundary_shift_seconds: float = 2.0,
) -> Path:
    """Build one complete, indexed event atlas without touching ``runs``.

    ``formal_test`` and ``clinical_review_all`` intentionally cannot share the
    same output directory.  The caller must choose a clean directory for each
    scope, which prevents a visual atlas from silently mixing 131 formal
    events with 188 source-label review events.
    """

    if boundary_shift_seconds <= 0.0:
        raise ValueError("boundary_shift_seconds must be positive")
    spec = scope_specification(scope)
    protocol = Path(protocol_dir).resolve()
    model = Path(model_root).resolve()
    output = Path(output_dir).resolve()
    for label, path in (("protocol_dir", protocol), ("model_root", model), ("output_dir", output)):
        _reject_historical_runs(path, label)
    _require_clean_output_directory(output)

    audit = _read_json(protocol / "protocol_audit.json")
    manual_path = protocol / spec.event_filename
    trace_path = model / "continuous" / spec.trace_filename
    decoder_path = model / "events" / "decoder_fit.json"
    for path in (manual_path, trace_path, decoder_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manual = _load_scope_events(manual_path, audit, spec)
    trace = _load_scope_trace(trace_path, spec)
    decoder_config = _load_decoder_config(decoder_path)
    decoded = _decode_trace_segments(
        trace,
        decoder_config,
        probability_column=spec.probability_column,
        allow_unknown_rows=spec.name == "clinical_review_all",
    )
    if spec.name == "formal_test":
        _require_formal_events_in_one_decoder_segment(manual, decoded.segment_index)
    model_name = str(model.name)
    model_label = MODEL_LABELS.get(model_name, model_name.replace("_", " "))

    event_table = _build_manual_event_table(
        manual,
        decoded.predicted_events,
        decoded.segment_index,
        scope=spec.name,
        model_name=model_name,
        model_label=model_label,
        boundary_shift_seconds=boundary_shift_seconds,
    )
    candidate_table = _build_external_candidate_table(
        manual,
        decoded,
        scope=spec.name,
        model_name=model_name,
        model_label=model_label,
    )
    counts = _build_category_counts(event_table, candidate_table, spec, model_name, model_label)

    # An explicitly created empty target is a valid clean output directory.
    # Non-empty targets were rejected before any source data was read.
    output.mkdir(parents=True, exist_ok=True)
    source_data = output / "source_data"
    atlas_pages = output / "atlas_pages"
    source_data.mkdir()
    atlas_pages.mkdir()

    event_table.to_csv(
        source_data / "manual_event_assessment.csv", index=False, encoding="utf-8-sig"
    )
    candidate_table.to_csv(
        source_data / "label_external_candidates.csv", index=False, encoding="utf-8-sig"
    )
    decoded.predicted_events.to_csv(
        source_data / "decoded_predicted_events.csv", index=False, encoding="utf-8-sig"
    )
    decoded.segment_index.to_csv(
        source_data / "decoder_segment_index.csv", index=False, encoding="utf-8-sig"
    )
    counts.to_csv(
        source_data / "event_category_counts.csv", index=False, encoding="utf-8-sig"
    )

    _plot_category_summary(counts, output / "event_category_summary", spec, model_label)
    trace_store = TraceStore(protocol)
    index = _write_complete_atlas(
        atlas_pages,
        event_table,
        candidate_table,
        manual,
        decoded,
        trace_store,
        decoder_config,
        spec,
        model_name,
        model_label,
    )
    index.to_csv(output / "atlas_index.csv", index=False, encoding="utf-8-sig")
    manifest = _build_manifest(
        spec,
        protocol,
        model,
        audit,
        manual,
        trace,
        decoded,
        event_table,
        candidate_table,
        counts,
        index,
        decoder_config,
        boundary_shift_seconds,
    )
    (output / "atlas_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_readme(output, spec, model_label, counts, manifest)
    return output


def _reject_historical_runs(path: Path, label: str) -> None:
    """Prevent accidental input from or output to historical experiment roots."""

    if any(part.casefold() == "runs" for part in path.parts):
        raise ValueError(f"{label} must not be inside a historical runs directory: {path}")


def _require_clean_output_directory(output: Path) -> None:
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"atlas output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"atlas output directory is not empty: {output}; choose a new scope-specific directory"
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"expected an object in {path}")
    return data


def _load_scope_events(path: Path, audit: Mapping[str, Any], spec: AtlasScope) -> pd.DataFrame:
    events = pd.read_parquet(path)
    required = {*KEYS, "event_id", "start_time_ms", "end_time_ms", "split"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"event file is missing required columns: {missing}")
    events = events.loc[events["split"].astype(str).eq("test")].copy()
    events = events.sort_values([*KEYS, "start_time_ms", "event_id"], kind="stable")
    if events.empty:
        raise ValueError(f"{spec.name} has no test events")
    if (events["end_time_ms"].to_numpy(float) <= events["start_time_ms"].to_numpy(float)).any():
        raise ValueError("event file contains a non-positive interval")
    if spec.name == "formal_test":
        required_columns = {"event_gate_status", "event_gate_reason", "formal_decoder_segment_id"}
        missing_gate = sorted(required_columns - set(events.columns))
        if missing_gate:
            raise ValueError(f"formal event file is missing gate columns: {missing_gate}")
        if not events["event_gate_status"].astype(str).eq("included").all():
            raise ValueError("formal event file contains excluded channel events")
        if not events["event_gate_reason"].astype(str).eq(
            "fully_observable_one_decoder_segment"
        ).all():
            raise ValueError("formal event file contains incompletely observable events")
        if events["formal_decoder_segment_id"].isna().any() or not events[
            "formal_decoder_segment_id"
        ].astype(str).str.strip().ne("").all():
            raise ValueError("formal event file contains an empty decoder-segment identity")

    expected = _expected_scope_event_count(audit, spec)
    if len(events) != expected:
        raise ValueError(
            f"{spec.name} event denominator mismatch: observed {len(events)}, expected {expected}"
        )
    return events.reset_index(drop=True)


def _expected_scope_event_count(audit: Mapping[str, Any], spec: AtlasScope) -> int:
    try:
        value = audit["event_metric_denominator"]["split_counts"]["test"][
            spec.denominator_key
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"protocol audit does not declare the {spec.name} test denominator"
        ) from exc
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {spec.name} test denominator: {value!r}") from exc
    if count < 1:
        raise ValueError(f"invalid {spec.name} test denominator: {count}")
    return count


def _load_scope_trace(path: Path, spec: AtlasScope) -> pd.DataFrame:
    trace = pd.read_parquet(path)
    required = {
        *KEYS,
        "sample_id",
        "split",
        "continuity_id",
        "continuity_position",
        "start_time_ms",
        "end_time_ms",
        "center_time_ms",
        spec.probability_column,
    }
    missing = sorted(required - set(trace.columns))
    if missing:
        raise ValueError(f"trace file is missing required columns: {missing}")
    trace = trace.loc[trace["split"].astype(str).eq("test")].copy()
    if trace.empty:
        raise ValueError(f"{spec.name} trace has no test windows")
    probability = trace[spec.probability_column].to_numpy(float)
    if not np.isfinite(probability).all() or ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError(
            f"trace contains an invalid {spec.probability_column} seizure probability"
        )
    return trace.sort_values([*KEYS, "start_time_ms"], kind="stable").reset_index(drop=True)


def _require_formal_events_in_one_decoder_segment(
    events: pd.DataFrame, segment_index: pd.DataFrame
) -> None:
    """Fail fast if the formal event gate and decoded trace disagree.

    Formal event eligibility promises that each label is fully observable in
    one physical/QC decoder segment.  Checking that promise again after the
    actual trace has been segmented prevents a stale gate, a mismatched model
    trace, or a silently changed context rule from turning a censored event
    into a reported miss.
    """

    required = {
        *KEYS,
        "event_id",
        "formal_decoder_segment_id",
        "start_time_ms",
        "end_time_ms",
    }
    missing_events = sorted(required - set(events.columns))
    if missing_events:
        raise ValueError(f"formal event table is missing required columns: {missing_events}")
    required = {*KEYS, "decoder_segment_id", "start_time_ms", "end_time_ms"}
    missing = sorted(required - set(segment_index.columns))
    if missing:
        raise ValueError(f"decoder segment index is missing required columns: {missing}")
    grouped = {
        tuple(map(str, key)): group.sort_values("start_time_ms", kind="stable")
        for key, group in segment_index.groupby(list(KEYS), sort=False)
    }
    failures: list[str] = []
    tolerance_ms = 1e-6
    for event in events.itertuples(index=False):
        key = tuple(map(str, (event.animal_id, event.session_id, event.channel_name)))
        segments = grouped.get(key)
        if segments is None:
            failures.append(f"{event.event_id}: no decoded trace for {'/'.join(key)}")
            continue
        raw_expected_id = event.formal_decoder_segment_id
        if pd.isna(raw_expected_id) or not str(raw_expected_id).strip():
            failures.append(f"{event.event_id}: empty formal decoder-segment identity")
            continue
        expected_id = str(raw_expected_id)
        expected = segments.loc[
            segments["decoder_segment_id"].astype(str).eq(expected_id)
        ]
        if len(expected) != 1:
            failures.append(
                f"{event.event_id}: expected formal decoder segment {expected_id!r}, "
                f"found {len(expected)} matching decoded segments"
            )
            continue
        expected_segment = expected.iloc[0]
        expected_contains = (
            float(expected_segment["start_time_ms"])
            <= float(event.start_time_ms) + tolerance_ms
            and float(expected_segment["end_time_ms"])
            >= float(event.end_time_ms) - tolerance_ms
        )
        if not expected_contains:
            failures.append(
                f"{event.event_id}: formal decoder segment {expected_id!r} does not contain "
                "the annotated interval"
            )
            continue
        contained = segments.loc[
            segments["start_time_ms"].le(float(event.start_time_ms) + tolerance_ms)
            & segments["end_time_ms"].ge(float(event.end_time_ms) - tolerance_ms)
        ]
        if len(contained) != 1:
            failures.append(
                f"{event.event_id}: expected exactly one containing decoder segment, found "
                f"{len(contained)}"
            )
    if failures:
        example = "; ".join(failures[:5])
        raise ValueError(
            "formal event containment failed after trace segmentation; "
            "rerun the event gate or use the matching inference trace. "
            f"Examples: {example}"
        )


def _load_decoder_config(path: Path) -> DecoderConfig:
    record = _read_json(path)
    config = record.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"decoder fit does not contain a config object: {path}")
    try:
        return DecoderConfig(**dict(config))
    except TypeError as exc:
        raise ValueError(f"decoder fit config is incompatible with DecoderConfig: {path}") from exc


def _runtime_decoder_segmenter(group: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    """Import the shared inference segmenter lazily in the training runtime.

    Keeping the import local allows lightweight source-data tests to run in a
    plotting-only environment while production atlas construction still calls
    the exact implementation used by inference and event evaluation.
    """

    from mouse_sleep.epilepsy.single_channel_inference import _decoder_segments

    return _decoder_segments(group)


def _decode_trace_segments(
    trace: pd.DataFrame,
    config: DecoderConfig,
    *,
    probability_column: str,
    segmenter: Segmenter | None = None,
    allow_unknown_rows: bool = False,
) -> DecodedAtlas:
    """Decode each shared QC/continuity segment independently.

    ``segmenter`` exists for lightweight unit tests only.  Normal execution
    uses the inference module's private, shared segmentation implementation.
    """

    required = {*KEYS, "start_time_ms", "end_time_ms", "continuity_id", "continuity_position", probability_column}
    missing = sorted(required - set(trace.columns))
    if missing:
        raise ValueError(f"trace is missing decoder columns: {missing}")
    if trace.empty:
        raise ValueError("cannot decode an empty trace")
    segmenter = _runtime_decoder_segmenter if segmenter is None else segmenter
    working = trace.copy().reset_index(drop=True)
    working["__atlas_trace_row"] = np.arange(len(working), dtype=np.int64)
    # The clinical candidate trace intentionally contains unknown/unreviewed
    # windows with no context identity.  The shared decoder omits those rows
    # rather than decoding across an unbounded context; only the clinical
    # review scope may therefore allow this documented omission class.
    if "context_segment_id" in working.columns:
        public_ids = working["context_segment_id"].astype("string")
    else:
        public_ids = working["continuity_id"].astype("string")
    if "inference_context_segment_id" in working.columns:
        boundary_ids = working["inference_context_segment_id"].astype("string")
    else:
        boundary_ids = public_ids
    decoder_known = (
        public_ids.notna()
        & public_ids.ne("")
        & boundary_ids.notna()
        & boundary_ids.ne("")
    ).fillna(False).to_numpy(dtype=bool)
    segment_frames: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    segment_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    seen_rows: set[int] = set()
    global_candidate_index = 0

    for key_values, group in working.groupby(list(KEYS), sort=True):
        key = tuple(map(str, key_values))
        for segment_id, segment in segmenter(group):
            ordered = segment.sort_values("start_time_ms", kind="stable").copy()
            if ordered.empty:
                continue
            starts = ordered["start_time_ms"].to_numpy(float)
            if len(starts) > 1 and np.any(np.diff(starts) <= 0.0):
                raise ValueError("decoder segment has non-increasing window start times")
            row_ids = ordered["__atlas_trace_row"].to_numpy(np.int64)
            duplicate_rows = set(map(int, row_ids)) & seen_rows
            if duplicate_rows:
                raise ValueError("decoder segments overlap on trace rows")
            seen_rows.update(map(int, row_ids))
            annotation = decode_window_probabilities(
                ordered[probability_column].to_numpy(float), starts, config
            )
            identifier = str(segment_id)
            segment_key = (*key, identifier)
            if segment_key in segment_frames:
                raise ValueError(f"duplicate decoder segment key: {segment_key}")
            ordered["decoder_segment_id"] = identifier
            ordered["raw_probability"] = annotation.probability
            ordered["smoothed_probability"] = annotation.smoothed_probability
            ordered["decoded_mask"] = annotation.seizure_mask.astype(np.int8)
            ordered["decoder_center_time_ms"] = annotation.window_center_time_ms
            ordered = ordered.drop(columns="__atlas_trace_row")
            segment_frames[segment_key] = ordered
            segment_rows.append(
                {
                    "animal_id": key[0],
                    "session_id": key[1],
                    "channel_name": key[2],
                    "decoder_segment_id": identifier,
                    "start_time_ms": float(ordered["start_time_ms"].min()),
                    "end_time_ms": float(ordered["end_time_ms"].max()),
                    "windows": int(len(ordered)),
                    "first_continuity_id": str(ordered["continuity_id"].iloc[0]),
                    "first_continuity_position": int(ordered["continuity_position"].iloc[0]),
                    "last_continuity_position": int(ordered["continuity_position"].iloc[-1]),
                }
            )
            for local_index, event in enumerate(annotation.events):
                event_rows.append(
                    {
                        "animal_id": key[0],
                        "session_id": key[1],
                        "channel_name": key[2],
                        "decoder_segment_id": identifier,
                        "predicted_event_index": int(local_index),
                        "atlas_candidate_index": int(global_candidate_index),
                        "start_time_ms": float(event.start_ms),
                        "end_time_ms": float(event.end_ms),
                        "duration_s": float(event.duration_ms / 1000.0),
                    }
                )
                global_candidate_index += 1

    expected_rows = set(range(len(working)))
    omitted_rows = expected_rows - seen_rows
    if allow_unknown_rows:
        unknown_rows = set(np.flatnonzero(~decoder_known).tolist())
        omitted_rows -= unknown_rows
    if omitted_rows:
        missing_rows = len(omitted_rows)
        raise ValueError(f"decoder segmentation omitted {missing_rows} trace rows")
    segment_index = pd.DataFrame(segment_rows)
    predicted = pd.DataFrame(
        event_rows,
        columns=[
            *KEYS,
            "decoder_segment_id",
            "predicted_event_index",
            "atlas_candidate_index",
            "start_time_ms",
            "end_time_ms",
            "duration_s",
        ],
    )
    if segment_index.empty:
        raise ValueError("decoder segmentation produced no usable segments")
    return DecodedAtlas(
        segment_frames=segment_frames,
        segment_index=segment_index.sort_values([*KEYS, "start_time_ms"], kind="stable").reset_index(drop=True),
        predicted_events=predicted.sort_values([*KEYS, "start_time_ms"], kind="stable").reset_index(drop=True),
    )


def _build_manual_event_table(
    manual: pd.DataFrame,
    predicted: pd.DataFrame,
    segment_index: pd.DataFrame,
    *,
    scope: AtlasScopeName,
    model_name: str,
    model_label: str,
    boundary_shift_seconds: float,
) -> pd.DataFrame:
    """Classify every manual event using union coverage, never Hungarian matches."""

    manual_groups = {
        tuple(map(str, key)): group.sort_values("start_time_ms", kind="stable")
        for key, group in manual.groupby(list(KEYS), sort=False)
    }
    predicted_groups = {
        tuple(map(str, key)): group.sort_values("start_time_ms", kind="stable")
        for key, group in predicted.groupby(list(KEYS), sort=False)
    }
    segment_groups = {
        tuple(map(str, key)): group.sort_values("start_time_ms", kind="stable")
        for key, group in segment_index.groupby(list(KEYS), sort=False)
    }
    rows: list[dict[str, Any]] = []
    for event in manual.itertuples(index=False):
        record = event._asdict()
        key = tuple(map(str, (event.animal_id, event.session_id, event.channel_name)))
        start = float(event.start_time_ms)
        end = float(event.end_time_ms)
        duration_ms = end - start
        if duration_ms <= 0.0:
            raise ValueError(f"manual event has a non-positive interval: {record.get('event_id')}")
        prediction_group = predicted_groups.get(key, predicted.iloc[0:0])
        overlaps = _interval_overlap(prediction_group, start, end)
        covered_ms = _union_covered_duration(overlaps, start, end)
        coverage = covered_ms / duration_ms
        category = _primary_category(coverage)
        segment_group = segment_groups.get(key, segment_index.iloc[0:0])
        overlapping_segments = _interval_overlap(segment_group, start, end)
        display_segment, display_overlap_ms = _select_display_segment(
            overlapping_segments, start, end
        )

        record.update(
            {
                "scope": scope,
                "model": model_name,
                "model_label": model_label,
                "duration_s": float(duration_ms / 1000.0),
                "covered_duration_ms": float(covered_ms),
                "coverage": float(coverage),
                "overlapping_predicted_segments": int(len(overlaps)),
                "primary_category": category,
                "long_missed": bool(coverage == 0.0 and duration_ms >= 30_000.0),
                "decoder_segment_count_intersecting_truth": int(len(overlapping_segments)),
                "decoder_segments_intersecting_truth": "|".join(
                    overlapping_segments["decoder_segment_id"].astype(str).tolist()
                ),
                "crosses_decoder_boundary": bool(len(overlapping_segments) > 1),
                "display_decoder_segment_id": display_segment,
                "display_trace_overlap_ms": float(display_overlap_ms),
                "display_trace_visible_fraction": float(display_overlap_ms / duration_ms),
            }
        )
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
            envelope_start = float(overlaps["start_time_ms"].min())
            envelope_end = float(overlaps["end_time_ms"].max())
            onset_error = (envelope_start - start) / 1000.0
            offset_error = (envelope_end - end) / 1000.0
            record.update(
                {
                    "prediction_envelope_start_ms": envelope_start,
                    "prediction_envelope_end_ms": envelope_end,
                    "envelope_onset_error_s": float(onset_error),
                    "envelope_offset_error_s": float(offset_error),
                    "maximum_pairwise_iou": _maximum_pairwise_iou(overlaps, start, end),
                    "boundary_shift": bool(
                        max(abs(onset_error), abs(offset_error)) > boundary_shift_seconds
                    ),
                }
            )
        rows.append(record)
    result = pd.DataFrame(rows)
    if len(result) != len(manual):
        raise ValueError("manual event assessment did not preserve every source event")
    return result.sort_values([*KEYS, "start_time_ms", "event_id"], kind="stable").reset_index(drop=True)


def _build_external_candidate_table(
    manual: pd.DataFrame,
    decoded: DecodedAtlas,
    *,
    scope: AtlasScopeName,
    model_name: str,
    model_label: str,
) -> pd.DataFrame:
    """Return model intervals with zero positive-time label overlap in this scope."""

    manual_groups = {
        tuple(map(str, key)): group.sort_values("start_time_ms", kind="stable")
        for key, group in manual.groupby(list(KEYS), sort=False)
    }
    rows: list[dict[str, Any]] = []
    for candidate in decoded.predicted_events.itertuples(index=False):
        key = tuple(map(str, (candidate.animal_id, candidate.session_id, candidate.channel_name)))
        truth = manual_groups.get(key, manual.iloc[0:0])
        if not _interval_overlap(truth, float(candidate.start_time_ms), float(candidate.end_time_ms)).empty:
            continue
        segment = decoded.segment_frames[(
            key[0],
            key[1],
            key[2],
            str(candidate.decoder_segment_id),
        )]
        inside = segment.loc[
            segment["decoder_center_time_ms"].ge(float(candidate.start_time_ms))
            & segment["decoder_center_time_ms"].lt(float(candidate.end_time_ms))
        ]
        if inside.empty:
            inside = segment
        record = candidate._asdict()
        record.update(
            {
                "scope": scope,
                "model": model_name,
                "model_label": model_label,
                "primary_category": OUTSIDE,
                "peak_raw_probability": float(inside["raw_probability"].max()),
                "peak_smoothed_probability": float(inside["smoothed_probability"].max()),
                "manual_events_on_track": int(len(truth)),
            }
        )
        rows.append(record)
    return pd.DataFrame(
        rows,
        columns=[
            *KEYS,
            "decoder_segment_id",
            "predicted_event_index",
            "atlas_candidate_index",
            "start_time_ms",
            "end_time_ms",
            "duration_s",
            "scope",
            "model",
            "model_label",
            "primary_category",
            "peak_raw_probability",
            "peak_smoothed_probability",
            "manual_events_on_track",
        ],
    )


def _interval_overlap(frame: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    return frame.loc[frame["end_time_ms"].gt(start) & frame["start_time_ms"].lt(end)]


def _union_covered_duration(overlaps: pd.DataFrame, start: float, end: float) -> float:
    if overlaps.empty:
        return 0.0
    spans = sorted(
        (
            max(float(row.start_time_ms), start),
            min(float(row.end_time_ms), end),
        )
        for row in overlaps.itertuples(index=False)
    )
    total = 0.0
    left, right = spans[0]
    for next_left, next_right in spans[1:]:
        if next_left <= right:
            right = max(right, next_right)
        else:
            total += max(0.0, right - left)
            left, right = next_left, next_right
    total += max(0.0, right - left)
    return float(total)


def _maximum_pairwise_iou(overlaps: pd.DataFrame, start: float, end: float) -> float:
    values: list[float] = []
    manual_duration = end - start
    for row in overlaps.itertuples(index=False):
        candidate_start = float(row.start_time_ms)
        candidate_end = float(row.end_time_ms)
        intersection = max(0.0, min(candidate_end, end) - max(candidate_start, start))
        union = manual_duration + (candidate_end - candidate_start) - intersection
        values.append(intersection / max(union, 1e-9))
    return float(max(values, default=0.0))


def _primary_category(coverage: float) -> str:
    if coverage >= 0.90:
        return FULL
    if coverage >= 0.50:
        return PARTIAL
    if coverage > 0.0:
        return LOW
    return MISSED


def _select_display_segment(
    overlaps: pd.DataFrame, start: float, end: float
) -> tuple[str | None, float]:
    if overlaps.empty:
        return None, 0.0
    candidate = overlaps.copy()
    candidate["_overlap_ms"] = np.minimum(
        candidate["end_time_ms"].to_numpy(float), end
    ) - np.maximum(candidate["start_time_ms"].to_numpy(float), start)
    candidate = candidate.sort_values(
        ["_overlap_ms", "start_time_ms", "decoder_segment_id"],
        ascending=[False, True, True],
        kind="stable",
    )
    chosen = candidate.iloc[0]
    return str(chosen["decoder_segment_id"]), float(chosen["_overlap_ms"])


def _build_category_counts(
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    spec: AtlasScope,
    model_name: str,
    model_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    denominator = len(events)
    for category in PRIMARY_CATEGORIES:
        count = int(events["primary_category"].eq(category).sum())
        rows.append(_count_row(spec, model_name, model_label, category, count, denominator, "manual_event"))
    rows.append(
        _count_row(
            spec,
            model_name,
            model_label,
            LONG_MISSED,
            int(events["long_missed"].sum()),
            denominator,
            "manual_event_subset",
        )
    )
    rows.append(
        _count_row(
            spec,
            model_name,
            model_label,
            BOUNDARY_SHIFT,
            int(events["boundary_shift"].sum()),
            denominator,
            "manual_event_subset",
        )
    )
    rows.append(
        _count_row(
            spec,
            model_name,
            model_label,
            OUTSIDE,
            int(len(candidates)),
            denominator,
            "predicted_interval",
        )
    )
    counts = pd.DataFrame(rows)
    if int(counts.loc[counts["category"].isin(PRIMARY_CATEGORIES), "count"].sum()) != denominator:
        raise ValueError("primary category partition is not equal to the event denominator")
    return counts


def _count_row(
    spec: AtlasScope,
    model_name: str,
    model_label: str,
    category: str,
    count: int,
    denominator: int,
    entity: str,
) -> dict[str, Any]:
    return {
        "scope": spec.name,
        "model": model_name,
        "model_label": model_label,
        "category": category,
        "category_label": CATEGORY_LABELS[category],
        "count": count,
        "manual_event_denominator": denominator,
        "fraction_of_manual_event_denominator": float(count / denominator),
        "entity": entity,
    }


def _plot_category_summary(
    counts: pd.DataFrame,
    stem: Path,
    spec: AtlasScope,
    model_label: str,
) -> None:
    labels = [
        "Full\n>=0.90",
        "Partial\n0.50-<0.90",
        "Low\n>0-<0.50",
        "Missed\n0",
        "Long missed\n>=30 s",
        "Boundary shift\n>2 s",
        "Label-external\ncandidates",
    ]
    values = [
        int(counts.loc[counts["category"].eq(category), "count"].iloc[0])
        for category in CATEGORY_ORDER
    ]
    colors = ["#3B7187", "#79A6B8", "#C5D5DA", "#C85B3A", "#9D3324", "#8B6D45", "#6C7378"]
    fig, axis = plt.subplots(figsize=(7.2047, 2.8))
    positions = np.arange(len(values))
    bars = axis.bar(positions, values, color=colors, width=0.72)
    ceiling = max(values + [1])
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + ceiling * 0.025,
            str(value),
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    denominator = int(counts["manual_event_denominator"].iloc[0])
    scope_title = "Formal held-out event audit" if spec.name == "formal_test" else "Clinical review event audit"
    axis.set(
        xticks=positions,
        xticklabels=labels,
        ylabel="Events / candidates",
        ylim=(0, ceiling * 1.18),
    )
    axis.set_title(f"{scope_title} | {model_label}", loc="left", fontweight="bold")
    axis.tick_params(axis="x", length=0, pad=5)
    axis.grid(axis="y", color=LIGHT_GREY, linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    footnote = (
        f"Primary categories partition n={denominator} manual events. Long misses and boundary shifts are subsets. "
        f"{spec.reporting_status.capitalize()}."
    )
    fig.text(0.10, 0.02, footnote, fontsize=5.8, color=MID_GREY)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.84, bottom=0.27)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _write_complete_atlas(
    atlas_dir: Path,
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    manual: pd.DataFrame,
    decoded: DecodedAtlas,
    traces: TraceStore,
    config: DecoderConfig,
    spec: AtlasScope,
    model_name: str,
    model_label: str,
) -> pd.DataFrame:
    index_rows: list[dict[str, Any]] = []
    category_frames: dict[str, pd.DataFrame] = {
        FULL: events.loc[events["primary_category"].eq(FULL)].copy(),
        PARTIAL: events.loc[events["primary_category"].eq(PARTIAL)].copy(),
        LOW: events.loc[events["primary_category"].eq(LOW)].copy(),
        MISSED: events.loc[events["primary_category"].eq(MISSED)].copy(),
        LONG_MISSED: events.loc[events["long_missed"]].copy(),
        BOUNDARY_SHIFT: events.loc[events["boundary_shift"]].copy(),
        OUTSIDE: candidates.copy(),
    }
    for number, category in enumerate(CATEGORY_ORDER, start=1):
        frame = _sort_category_frame(category_frames[category], category)
        path = atlas_dir / f"{number:02d}_{category}.pdf"
        index_rows.extend(
            _write_category_pdf(
                path,
                frame,
                category,
                manual,
                decoded,
                traces,
                config,
                spec,
                model_name,
                model_label,
            )
        )
    return pd.DataFrame(
        index_rows,
        columns=[
            "scope",
            "model",
            "category",
            "entity",
            "animal_id",
            "session_id",
            "channel_name",
            "event_id",
            "atlas_candidate_index",
            "pdf",
            "page",
        ],
    )


def _sort_category_frame(frame: pd.DataFrame, category: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if category == OUTSIDE:
        return frame.sort_values(
            ["peak_raw_probability", "duration_s", "atlas_candidate_index"],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)
    if category in {MISSED, LONG_MISSED}:
        return frame.sort_values(
            ["duration_s", "event_id"], ascending=[False, True], kind="stable"
        ).reset_index(drop=True)
    if category == BOUNDARY_SHIFT:
        return frame.assign(
            _boundary_error=frame[["envelope_onset_error_s", "envelope_offset_error_s"]]
            .abs()
            .max(axis=1)
        ).sort_values(["_boundary_error", "event_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    return frame.sort_values(
        ["coverage", "duration_s", "event_id"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _write_category_pdf(
    path: Path,
    frame: pd.DataFrame,
    category: str,
    manual: pd.DataFrame,
    decoded: DecodedAtlas,
    traces: TraceStore,
    config: DecoderConfig,
    spec: AtlasScope,
    model_name: str,
    model_label: str,
) -> list[dict[str, Any]]:
    """Write exactly one review unit per A4 page and return its stable index."""

    index_rows: list[dict[str, Any]] = []
    with PdfPages(path) as pdf:
        if frame.empty:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.50, 0.53, CATEGORY_LABELS[category], ha="center", fontsize=12, fontweight="bold")
            fig.text(0.50, 0.48, "No intervals in this category", ha="center", color=MID_GREY)
            pdf.savefig(fig)
            plt.close(fig)
            return index_rows
        total = len(frame)
        for page, (_, entry) in enumerate(frame.iterrows(), start=1):
            fig = plt.figure(figsize=(8.27, 11.69))
            grid = fig.add_gridspec(
                3,
                1,
                left=0.085,
                right=0.965,
                top=0.92,
                bottom=0.095,
                height_ratios=(2.35, 1.25, 0.48),
                hspace=0.09,
            )
            axes = (
                fig.add_subplot(grid[0]),
                fig.add_subplot(grid[1]),
                fig.add_subplot(grid[2]),
            )
            axes[1].sharex(axes[0])
            axes[2].sharex(axes[0])
            _draw_review_unit(axes, entry, category, manual, decoded, traces, config)
            fig.suptitle(
                f"{model_label} | {spec.name} | {CATEGORY_LABELS[category]}",
                x=0.085,
                y=0.972,
                ha="left",
                fontsize=9.5,
                fontweight="bold",
                color=INK,
            )
            fig.text(0.965, 0.973, f"page {page}/{total}", ha="right", va="top", fontsize=6.2, color=MID_GREY)
            fig.text(
                0.085,
                0.027,
                "Blue = manual label; orange = decoded model interval/mask; grey = raw probability; green = smoothed probability. "
                "Amplitude is training-baseline normalized a.u. Every page is constrained to one QC/continuity decoder segment.",
                fontsize=5.5,
                color=MID_GREY,
            )
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
            index_rows.append(
                {
                    "scope": spec.name,
                    "model": model_name,
                    "category": category,
                    "entity": "predicted_interval" if category == OUTSIDE else "manual_event",
                    "animal_id": str(entry["animal_id"]),
                    "session_id": str(entry["session_id"]),
                    "channel_name": str(entry["channel_name"]),
                    "event_id": str(entry.get("event_id", "")),
                    "atlas_candidate_index": entry.get("atlas_candidate_index", pd.NA),
                    "pdf": f"atlas_pages/{path.name}",
                    "page": page,
                }
            )
    return index_rows


def _draw_review_unit(
    axes: tuple[plt.Axes, plt.Axes, plt.Axes],
    entry: pd.Series,
    category: str,
    manual: pd.DataFrame,
    decoded: DecodedAtlas,
    traces: TraceStore,
    config: DecoderConfig,
) -> None:
    waveform_axis, probability_axis, mask_axis = axes
    key = tuple(map(str, (entry["animal_id"], entry["session_id"], entry["channel_name"])))
    segment_id = str(
        entry["decoder_segment_id"]
        if category == OUTSIDE
        else entry.get("display_decoder_segment_id", "")
    )
    segment_key = (*key, segment_id)
    segment = decoded.segment_frames.get(segment_key)
    if not segment_id or segment is None:
        _draw_unobservable_unit(axes, entry, category)
        return

    is_candidate = category == OUTSIDE
    focus_start = float(entry["start_time_ms"])
    focus_end = float(entry["end_time_ms"])
    reference = focus_start
    segment_start = float(segment["start_time_ms"].min())
    segment_end = float(segment["end_time_ms"].max())
    prediction_group = decoded.predicted_events.loc[
        decoded.predicted_events["animal_id"].astype(str).eq(key[0])
        & decoded.predicted_events["session_id"].astype(str).eq(key[1])
        & decoded.predicted_events["channel_name"].astype(str).eq(key[2])
        & decoded.predicted_events["decoder_segment_id"].astype(str).eq(segment_id)
    ]
    nearby_predictions = _interval_overlap(prediction_group, focus_start, focus_end)
    span_start = min([focus_start, *nearby_predictions["start_time_ms"].astype(float).tolist()])
    span_end = max([focus_end, *nearby_predictions["end_time_ms"].astype(float).tolist()])
    margin = max(12_000.0, 0.08 * max(span_end - span_start, 1.0))
    crop_start = max(segment_start, span_start - margin)
    crop_end = min(segment_end, span_end + margin)
    if crop_end <= crop_start:
        _draw_unobservable_unit(axes, entry, category)
        return
    truth_group = manual.loc[
        manual["animal_id"].astype(str).eq(key[0])
        & manual["session_id"].astype(str).eq(key[1])
        & manual["channel_name"].astype(str).eq(key[2])
    ]
    crop_truth = _interval_overlap(truth_group, crop_start, crop_end)
    crop_predictions = _interval_overlap(prediction_group, crop_start, crop_end)
    crop_trace = segment.loc[
        segment["decoder_center_time_ms"].ge(crop_start)
        & segment["decoder_center_time_ms"].le(crop_end)
    ]

    try:
        # Load exactly the raw windows represented by this decoder segment.
        # A whole-track crop could otherwise draw across an omitted QC/context
        # interval even when the probability/mask panel is correctly split.
        time_ms, waveform = traces.load_decoder_segment(segment, crop_start, crop_end)
    except (OSError, ValueError, KeyError) as exc:
        waveform_axis.text(
            0.5,
            0.5,
            f"Waveform unavailable for this decoder segment\n{type(exc).__name__}",
            ha="center",
            va="center",
            transform=waveform_axis.transAxes,
            color=MID_GREY,
        )
    else:
        display_x, lower, upper = _waveform_display(
            (time_ms - reference) / 1000.0, waveform, maximum_points=6500
        )
        if lower is None:
            waveform_axis.plot(display_x, upper, color=INK, linewidth=0.48)
        else:
            waveform_axis.fill_between(display_x, lower, upper, color=INK, linewidth=0, alpha=0.82)

    for event in crop_truth.itertuples(index=False):
        _span(
            (waveform_axis, probability_axis),
            float(event.start_time_ms),
            float(event.end_time_ms),
            crop_start,
            crop_end,
            reference,
            facecolor=MANUAL_COLOR,
            alpha=0.13,
        )
    for event in crop_predictions.itertuples(index=False):
        _span(
            (waveform_axis, probability_axis),
            float(event.start_time_ms),
            float(event.end_time_ms),
            crop_start,
            crop_end,
            reference,
            facecolor="none",
            edgecolor=PREDICTED_COLOR,
            linewidth=0.8,
        )

    x_probability = (crop_trace["decoder_center_time_ms"].to_numpy(float) - reference) / 1000.0
    if len(crop_trace):
        probability_axis.plot(
            x_probability,
            crop_trace["raw_probability"].to_numpy(float),
            color=RAW_COLOR,
            linewidth=0.55,
            marker="o",
            markersize=1.4,
            label="Raw",
        )
        probability_axis.plot(
            x_probability,
            crop_trace["smoothed_probability"].to_numpy(float),
            color=SMOOTHED_COLOR,
            linewidth=1.0,
            label="Smoothed",
        )
        mask_axis.fill_between(
            x_probability,
            0,
            crop_trace["decoded_mask"].to_numpy(float),
            step="mid",
            color=MASK_COLOR,
            alpha=0.78,
            linewidth=0,
        )
    probability_axis.axhline(config.high_threshold, color="#4B5155", linestyle="--", linewidth=0.6)
    probability_axis.axhline(config.low_threshold, color="#92989C", linestyle=":", linewidth=0.6)
    probability_axis.set(ylim=(-0.02, 1.02), yticks=(0, 0.5, 1), ylabel="P(seizure)")
    if len(crop_trace):
        probability_axis.legend(loc="upper right", ncol=2, frameon=False)
    mask_axis.set(
        ylim=(0, 1),
        yticks=(0, 1),
        yticklabels=("off", "on"),
        ylabel="Decoded",
        xlabel="Time from label/candidate onset (s)",
    )
    if not is_candidate:
        for boundary in (focus_start, focus_end):
            if crop_start <= boundary <= crop_end:
                for axis in axes:
                    axis.axvline(
                        (boundary - reference) / 1000.0,
                        color=MANUAL_COLOR,
                        linestyle="--",
                        linewidth=0.85,
                    )
    waveform_axis.set_ylabel("Amplitude (a.u.)")
    waveform_axis.set_title(_metadata_text(entry, category), loc="left", fontsize=6.2, fontweight="bold", pad=4)
    waveform_axis.tick_params(labelbottom=False)
    probability_axis.tick_params(labelbottom=False)
    x_limits = ((crop_start - reference) / 1000.0, (crop_end - reference) / 1000.0)
    for axis in axes:
        axis.set_xlim(x_limits)
        axis.tick_params(direction="out", length=2.5, pad=2)


def _span(
    axes: Iterable[plt.Axes],
    start: float,
    end: float,
    crop_start: float,
    crop_end: float,
    reference: float,
    **kwargs: Any,
) -> None:
    clipped_start = max(start, crop_start)
    clipped_end = min(end, crop_end)
    if clipped_end <= clipped_start:
        return
    for axis in axes:
        axis.axvspan(
            (clipped_start - reference) / 1000.0,
            (clipped_end - reference) / 1000.0,
            **kwargs,
        )


def _draw_unobservable_unit(
    axes: tuple[plt.Axes, plt.Axes, plt.Axes], entry: pd.Series, category: str
) -> None:
    message = (
        "No single observable decoder segment covers this manual interval.\n"
        "This page is retained for clinical review and is excluded from the formal denominator."
    )
    for axis in axes:
        axis.set_axis_off()
    axes[0].text(0.5, 0.56, message, ha="center", va="center", transform=axes[0].transAxes, color=MID_GREY)
    axes[0].set_title(_metadata_text(entry, category), loc="left", fontsize=6.2, fontweight="bold", pad=4)


def _metadata_text(entry: pd.Series, category: str) -> str:
    identity = f"{entry['animal_id']} | {entry['session_id']} | blinded {entry['channel_name']}"
    if category == OUTSIDE:
        return (
            f"{identity} | candidate {int(entry['atlas_candidate_index'])}\n"
            f"duration {float(entry['duration_s']):.1f} s | no manual label overlap in this scope | "
            f"peak raw P {float(entry['peak_raw_probability']):.2f}"
        )
    onset = entry.get("envelope_onset_error_s", np.nan)
    offset = entry.get("envelope_offset_error_s", np.nan)
    boundary = "onset/offset n.a."
    if not (pd.isna(onset) or pd.isna(offset)):
        boundary = f"envelope onset/offset {float(onset):+.1f}/{float(offset):+.1f} s"
    clipped = ""
    if bool(entry.get("crosses_decoder_boundary", False)):
        clipped = " | multi-segment label; displayed segment is clipped"
    elif pd.isna(entry.get("display_decoder_segment_id", pd.NA)) or not str(
        entry.get("display_decoder_segment_id", "")
    ).strip():
        clipped = " | no observable decoder segment"
    return (
        f"{identity} | {entry['event_id']}\n"
        f"duration {float(entry['duration_s']):.1f} s | coverage {float(entry['coverage']):.2f} | "
        f"max pair IoU {float(entry['maximum_pairwise_iou']):.2f} | segments {int(entry['overlapping_predicted_segments'])} | "
        f"{boundary}{clipped}"
    )


def _build_manifest(
    spec: AtlasScope,
    protocol: Path,
    model: Path,
    audit: Mapping[str, Any],
    manual: pd.DataFrame,
    trace: pd.DataFrame,
    decoded: DecodedAtlas,
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    counts: pd.DataFrame,
    index: pd.DataFrame,
    config: DecoderConfig,
    boundary_shift_seconds: float,
) -> dict[str, Any]:
    denominator = int(len(manual))
    return {
        "schema_version": 1,
        "scope": spec.name,
        "reporting_status": spec.reporting_status,
        "event_source": spec.event_filename,
        "trace_source": spec.trace_filename,
        "manual_event_denominator": denominator,
        "expected_manual_event_denominator": _expected_scope_event_count(audit, spec),
        "model_root": str(model),
        "protocol": str(protocol),
        "probability_column": spec.probability_column,
        "decoder_config": asdict(config),
        "decoder_segmentation": (
            "single_channel_inference._decoder_segments; splits at context segment and "
            "physical/QC continuity-position boundaries, never clinical labels or event IDs"
        ),
        "trace_test_windows": int(len(trace)),
        "decoder_segments": int(len(decoded.segment_index)),
        "decoded_predicted_events": int(len(decoded.predicted_events)),
        "manual_events_with_no_observable_decoder_segment": int(
            events["display_decoder_segment_id"].isna().sum()
        ),
        "manual_events_crossing_decoder_boundary": int(events["crosses_decoder_boundary"].sum()),
        "label_external_candidates": int(len(candidates)),
        "external_candidate_definition": spec.external_candidate_definition,
        "boundary_shift_threshold_seconds": float(boundary_shift_seconds),
        "primary_category_partition": {
            category: int(
                counts.loc[counts["category"].eq(category), "count"].iloc[0]
            )
            for category in PRIMARY_CATEGORIES
        },
        "primary_category_partition_matches_denominator": int(
            counts.loc[counts["category"].isin(PRIMARY_CATEGORIES), "count"].sum()
        )
        == denominator,
        "indexed_pages": int(len(index)),
        "amplitude_unit": "training-baseline normalized arbitrary units (a.u.)",
        "historic_runs_used": False,
    }


def _write_readme(
    output: Path,
    spec: AtlasScope,
    model_label: str,
    counts: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> None:
    lines = [
        f"# Continuous event atlas: {spec.name}",
        "",
        f"Model: {model_label}.",
        f"Status: {spec.reporting_status}.",
        "",
        "## Scope contract",
        "",
        f"- Manual events: `{spec.event_filename}`; test-event denominator `{manifest['manual_event_denominator']}`.",
        f"- Probability trace: `{spec.trace_filename}` using `{spec.probability_column}`.",
        "- Probability traces are split with the same QC/context/physical-continuity segmentation used by inference. No page spans a decoder segment boundary.",
        "- Coverage is the union of decoded model intervals within a manual event. It is not a Hungarian IoU match count.",
        f"- Label-external candidate: {spec.external_candidate_definition}.",
        "",
        "## Reading order",
        "",
        "1. `event_category_summary.png` is the compact result figure.",
        "2. `source_data/manual_event_assessment.csv` contains one row for every manual event.",
        "3. `atlas_pages/` has one clinical review unit per PDF page; `atlas_index.csv` gives the stable page location.",
        "4. `source_data/label_external_candidates.csv` lists candidates for review, not automatically false positives.",
        "",
        "## Counts",
        "",
        "| category | count | denominator fraction | entity |",
        "|---|---:|---:|---|",
    ]
    for row in counts.itertuples(index=False):
        lines.append(
            f"| {row.category_label} | {int(row.count)} | {float(row.fraction_of_manual_event_denominator):.3f} | {row.entity} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guard",
            "",
            "`formal_test` is the only performance-reporting scope. `clinical_review_all` deliberately includes source labels that may be incomplete or not observable as one decoder trace; its counts are clinical audit descriptors rather than sensitivity, specificity, or false-positive-rate denominators.",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "AtlasScope",
    "AtlasScopeName",
    "DecodedAtlas",
    "generate_continuous_event_atlas",
    "scope_specification",
]
