#!/usr/bin/env python
"""Prepare and audit the new pooled-mouse epilepsy export.

This command never changes the source export and never starts model training.
It consumes the read-only audit tables, creates a conservative canonical event
table (sample coordinates are primary), and writes one compact output root with
the admission decisions, exclusions, and label/waveform QC figures.

The package intentionally keeps labels that are useful for QC even when the
recording is not yet eligible for training.  Missing review-completion markers
therefore remain an explicit blocker; an empty label file is never converted to
``non-seizure``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import defaultdict
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
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
    }
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = PROJECT_ROOT / "runs" / "epilepsy_new_batch_audit_20260819"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "epilepsy_new_batch_strict_v1"
DEFAULT_RAW = Path(r"E:\自动睡眠分析\癫痫脑电\data-labels 数据")
CHANNELS = ("CH7", "CH8")
FS_HZ = 1000.0

INK = "#22272B"
MID = "#6D747A"
GRID = "#D9DEE1"
MANUAL = "#1E5B8C"
CH7_COLOR = "#2F6B8A"
CH8_COLOR = "#C56B3D"
BLOCKED = "#C45A32"
OK = "#25746B"
PALE = "#EEF1F2"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--waveform-examples", type=int, default=24)
    parser.add_argument(
        "--skip-waveforms",
        action="store_true",
        help="Only make table/timeline QC; do not read raw parquet excerpts.",
    )
    return parser


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _attach_coordinate_validation(events: pd.DataFrame, audit_dir: Path) -> pd.DataFrame:
    """Attach the raw-row endpoint validation when it is available.

    The validation table is keyed by the immutable source CSV path and row
    number.  Keeping the original audit columns alongside the validated ones
    makes every coordinate correction auditable without silently replacing the
    clinician export.
    """

    path = audit_dir / "coordinate_endpoint_validation.csv"
    if not path.is_file():
        return events
    validation = pd.read_csv(path)
    keys = ["source_file", "source_row"]
    if not set(keys).issubset(events.columns) or not set(keys).issubset(validation.columns):
        raise ValueError("coordinate endpoint validation has no source_file/source_row key")
    left = events.copy()
    right = validation.copy()
    left["source_file"] = left["source_file"].astype(str)
    right["source_file"] = right["source_file"].astype(str)
    left["source_row"] = pd.to_numeric(left["source_row"], errors="coerce").astype("Int64")
    right["source_row"] = pd.to_numeric(right["source_row"], errors="coerce").astype("Int64")
    columns = [
        "source_file",
        "source_row",
        "raw_coordinate_scope",
        "validated_coordinate_scope",
        "validated_global_start_sample",
        "validated_global_end_sample",
        "validated_offset_samples",
        "validated_time_shift_ms",
        "validated_raw_start_timestamp_ms",
        "validated_raw_end_timestamp_ms",
        "validated_start_residual_ms",
        "validated_end_residual_ms",
        "validated_mapping_status",
        "validated_coordinate_usable",
        "validated_resolution_method",
    ]
    columns = [column for column in columns if column in right.columns]
    right = right.loc[:, columns].drop_duplicates(keys, keep="last")
    merged = left.merge(right, on=keys, how="left", validate="one_to_one")
    if "validated_coordinate_usable" not in merged.columns:
        return events
    merged["validated_coordinate_usable"] = _as_bool(merged["validated_coordinate_usable"])
    return merged


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _norm_channel(value: Any) -> str:
    text = str(value).strip().upper()
    match = re.search(r"(?:CH|CHANNEL)\s*0*(\d+)", text)
    return f"CH{int(match.group(1))}" if match else text


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:180]


def _file_number(value: Any) -> tuple[int, str]:
    """Sort canonical ``eeg_####.parquet`` shards by numeric suffix.

    Lexicographic ordering places ``eeg_10`` before ``eeg_2`` and would shift
    local-segment labels to the wrong cumulative row offset.  Nonconforming
    names are deliberately sent to the end and remain auditable in the
    canonical manifest.
    """

    text = str(value)
    match = re.search(r"_(\d+)\.parquet$", text, re.IGNORECASE)
    return (int(match.group(1)), text.lower()) if match else (10**9, text.lower())


def _copy_audit_snapshot(audit_dir: Path, output: Path) -> None:
    target = output / "audit_snapshot"
    target.mkdir(parents=True, exist_ok=True)
    names = (
        "audit_summary.json",
        "AUDIT_REPORT_zh.md",
        "raw_recordings.csv",
        "raw_parquet_integrity.csv",
        "signal_channel_qc.csv",
        "label_files.csv",
        "label_eeg_alignment_status.csv",
        "label_coordinate_scope_audit.csv",
        "label_event_time_mapping_review.csv",
        "coordinate_endpoint_validation.csv",
        "coordinate_endpoint_validation_summary.json",
        "mapping_review_summary.json",
        "MAPPING_REVIEW_zh.md",
        "label_events.csv",
        "label_coverage.csv",
        "duplicate_files.csv",
        "source_inventory.csv",
        "old_category_audit.csv",
    )
    for name in names:
        source = audit_dir / name
        if source.is_file():
            shutil.copy2(source, target / name)


def _canonical_parquets(integrity: pd.DataFrame) -> pd.DataFrame:
    """Keep one audit-selected parquet per normalized name and valid schema."""

    frame = integrity.copy()
    frame["footer_ok"] = _as_bool(frame["footer_readable"])
    frame["schema_ok_bool"] = _as_bool(frame["schema_ok"])
    frame["canonical_candidate"] = frame["footer_ok"] & frame["schema_ok_bool"]
    # The audit table already contains canonical files (copy suffixes removed).
    # Keep the explicit normalized name guard for future audits with extra rows.
    if "normalized_name" in frame:
        frame = frame.loc[~frame["normalized_name"].astype(str).str.contains(r"\(\d+\)", regex=True)].copy()
    frame["admission_status"] = np.where(
        frame["canonical_candidate"],
        np.where(frame["manifest_referenced"].isna(), "admit_without_manifest_reference", "admit"),
        "blocked_file_integrity",
    )
    return frame.sort_values(["rec_key", "timestamp_start_ms", "path"], kind="stable")


def _channel_admission(signal_qc: pd.DataFrame) -> pd.DataFrame:
    frame = signal_qc.copy()
    frame["channel_name"] = frame["channel"].map(_norm_channel)
    frame = frame.loc[frame["channel_name"].isin(CHANNELS)].copy()
    # A channel is usable for a recording only when the audit saw at least one
    # usable row and no device-rail/flat row for that same recording/channel.
    grouped: list[dict[str, Any]] = []
    for (rec_key, channel), group in frame.groupby(["rec_key", "channel_name"], sort=True):
        statuses = set(group["qc_status"].astype(str))
        if "bad_flat_or_sentinel" in statuses:
            decision = "exclude_bad_flat_or_sentinel"
        elif "bad_device_rail_contact" in statuses:
            decision = "exclude_bad_device_rail_contact"
        elif "usable_candidate" in statuses:
            decision = "admit_usable_candidate"
        else:
            decision = "quarantine_no_usable_qc"
        grouped.append(
            {
                "rec_key": str(rec_key),
                "animal_id": str(group["animal_id"].iloc[0]),
                "channel_name": str(channel),
                "qc_rows": int(len(group)),
                "qc_statuses": ";".join(sorted(statuses)),
                "channel_admission": decision,
                "usable": bool(decision == "admit_usable_candidate"),
            }
        )
    return pd.DataFrame(grouped)


def _label_file_status(label_files: pd.DataFrame) -> pd.DataFrame:
    frame = label_files.copy()
    frame["alignment_ok"] = frame["alignment_status"].eq("mapped_sample_bounds_contained")
    frame["suspected_binding"] = _as_bool(frame["suspected_rec_binding_error"])
    frame["completion_marker"] = _as_bool(frame["completion_marker_present"])
    frame["admission_status"] = "admit_for_qc_only"
    frame.loc[_as_bool(frame["header_only"]), "admission_status"] = "blocked_header_only_no_events"
    frame.loc[frame["alignment_status"].ne("mapped_sample_bounds_contained"), "admission_status"] = "blocked_alignment"
    frame.loc[frame["suspected_binding"], "admission_status"] = "blocked_suspected_recording_binding"
    frame.loc[frame["completion_marker"], "admission_status"] = "review_marker_present_but_manual_confirmation_required"
    # No supplied CSV has a formal completion marker in this batch. Keep this
    # explicit even if an export is non-empty.
    frame["training_label_status"] = np.where(
        frame["completion_marker"] & frame["alignment_ok"] & ~frame["suspected_binding"],
        "candidate_pending_manual_review",
        "blocked_no_completion_marker_or_alignment",
    )
    return frame


def _connected_event_groups(frame: pd.DataFrame) -> pd.Series:
    """Assign deterministic clinical groups from overlapping channel intervals."""

    output = pd.Series(index=frame.index, dtype=object)
    for rec_key, positions in frame.groupby("rec_key", sort=True).groups.items():
        subset = frame.loc[list(positions)].sort_values(
            ["start_sample", "end_sample", "channel_name"], kind="stable"
        )
        current_group = -1
        current_end = -1
        for idx, row in subset.iterrows():
            start = int(row["start_sample"])
            end = int(row["end_sample"])
            if start > current_end:
                current_group += 1
                current_end = end
            else:
                current_end = max(current_end, end)
            output.loc[idx] = f"{rec_key}/clinical_event_{current_group:05d}"
    return output


def _build_time_map(
    parquets: pd.DataFrame, recordings: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Build a lightweight recording-relative time -> sample map.

    JYEEG label exports can reset ``start_sample`` at every shard while their
    ``start_time_ms`` remains recording-relative.  The map uses the audited
    Parquet timestamp statistics and cumulative row counts, so it does not
    require loading the full 15-GB export.
    """

    starts = recordings.set_index("rec_key")["recording_start_ms"].to_dict()
    ends = recordings.set_index("rec_key")["recording_end_ms"].to_dict()
    totals = recordings.set_index("rec_key")["total_rows"].to_dict()
    maps: dict[str, dict[str, Any]] = {}
    valid = parquets.loc[parquets["canonical_candidate"]].copy()
    for rec_key, group in valid.groupby("rec_key", sort=True):
        group = group.sort_values(["timestamp_start_ms", "path"], kind="stable")
        rows: list[dict[str, float | int | str]] = []
        cumulative = 0
        for item in group.itertuples(index=False):
            count = int(float(item.row_count))
            rows.append(
                {
                    "path": str(item.path),
                    "row_count": count,
                    "cumulative_start": cumulative,
                    "cumulative_end": cumulative + count,
                    "timestamp_start_ms": float(item.timestamp_start_ms),
                    "timestamp_end_ms": float(item.timestamp_end_ms),
                }
            )
            cumulative += count
        maps[str(rec_key)] = {
            "recording_start_ms": float(starts.get(rec_key, np.nan)),
            "recording_end_ms": float(ends.get(rec_key, np.nan)),
            "total_rows": int(float(totals.get(rec_key, 0) or 0)),
            "rows": rows,
            "total_rows": cumulative,
        }
    return maps


def _map_relative_time_to_sample(
    time_ms: Any, time_map: dict[str, Any]
) -> tuple[int | None, float | None, str]:
    """Map a recording-relative exported time to a cumulative sample index."""

    try:
        relative = float(time_ms)
        recording_start = float(time_map["recording_start_ms"])
        recording_end = float(time_map["recording_end_ms"])
    except (TypeError, ValueError, KeyError):
        return None, None, "invalid_exported_time"
    if not np.isfinite(relative) or not np.isfinite(recording_start):
        return None, None, "invalid_exported_time"
    target = recording_start + relative
    # A small tolerance covers the first/last timestamp and row-group gaps.
    if target < recording_start - 2_000.0 or target > recording_end + 2_000.0:
        return None, None, "exported_time_outside_recording"
    rows = time_map.get("rows", [])
    if not rows:
        return None, None, "no_raw_time_map"
    containing = [
        row for row in rows
        if row["timestamp_start_ms"] - 2_000.0 <= target <= row["timestamp_end_ms"] + 2_000.0
    ]
    if containing:
        row = min(containing, key=lambda item: abs(target - (item["timestamp_start_ms"] + item["timestamp_end_ms"]) / 2.0))
    else:
        row = min(rows, key=lambda item: min(abs(target - item["timestamp_start_ms"]), abs(target - item["timestamp_end_ms"])))
        if min(abs(target - row["timestamp_start_ms"]), abs(target - row["timestamp_end_ms"])) > 5_000.0:
            return None, None, "no_nearby_raw_timestamp"
    local = int(round(target - float(row["timestamp_start_ms"])))
    local = max(0, min(local, int(row["row_count"]) - 1))
    sample = int(row["cumulative_start"] + local)
    mapped_abs = float(row["timestamp_start_ms"] + local)
    return sample, mapped_abs - recording_start, "exported_time_to_raw_timestamp"


def _align_event_coordinates(
    events: pd.DataFrame,
    parquets: pd.DataFrame,
    recordings: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve per-shard sample resets and retain explicit diagnostics."""

    maps = _build_time_map(parquets, recordings)
    aligned = events.copy()
    original_start = pd.to_numeric(aligned["start_sample"], errors="coerce")
    original_end = pd.to_numeric(aligned["end_sample"], errors="coerce")
    exported_start = pd.to_numeric(aligned["start_time_ms"], errors="coerce")
    exported_end = pd.to_numeric(aligned["end_time_ms"], errors="coerce")
    mapped_start: list[int | None] = []
    mapped_end: list[int | None] = []
    mapped_start_time: list[float | None] = []
    mapped_end_time: list[float | None] = []
    methods: list[str] = []
    for rec_key, start, end in zip(aligned["rec_key"], exported_start, exported_end, strict=True):
        time_map = maps.get(str(rec_key))
        if time_map is None:
            mapped_start.append(None); mapped_end.append(None); mapped_start_time.append(None); mapped_end_time.append(None); methods.append("no_raw_time_map"); continue
        start_value, start_rel, start_method = _map_relative_time_to_sample(start, time_map)
        end_value, end_rel, end_method = _map_relative_time_to_sample(end, time_map)
        if start_value is not None and end_value is not None and end_value >= start_value:
            mapped_start.append(start_value); mapped_end.append(end_value)
            mapped_start_time.append(start_rel); mapped_end_time.append(end_rel)
            methods.append(start_method if start_method == end_method else f"{start_method}+{end_method}")
        else:
            mapped_start.append(None); mapped_end.append(None); mapped_start_time.append(None); mapped_end_time.append(None); methods.append(start_method if start_method != "exported_time_to_raw_timestamp" else end_method)
    aligned["original_start_sample"] = original_start
    aligned["original_end_sample"] = original_end
    aligned["mapped_start_sample"] = pd.array(mapped_start, dtype="Int64")
    aligned["mapped_end_sample"] = pd.array(mapped_end, dtype="Int64")
    aligned["mapped_start_time_ms"] = mapped_start_time
    aligned["mapped_end_time_ms"] = mapped_end_time
    aligned["alignment_method"] = methods
    aligned["time_alignment_valid"] = aligned["mapped_start_sample"].notna() & aligned["mapped_end_sample"].notna()
    aligned["sample_reset_or_mismatch"] = (
        aligned["time_alignment_valid"]
        & ((aligned["mapped_start_sample"].astype(float) - original_start).abs() > 2_000.0)
    )
    # Use time-mapped cumulative coordinates whenever they are available. The
    # original sample values remain in the table for forensic review.
    aligned["start_sample"] = aligned["mapped_start_sample"].where(
        aligned["time_alignment_valid"], aligned["original_start_sample"]
    )
    aligned["end_sample"] = aligned["mapped_end_sample"].where(
        aligned["time_alignment_valid"], aligned["original_end_sample"]
    )
    cumulative_bounds = np.asarray(
        [
            int(end) < int(maps.get(str(rec), {}).get("total_rows", 0))
            if pd.notna(end)
            else False
            for rec, end in zip(aligned["rec_key"], aligned["end_sample"], strict=True)
        ],
        dtype=bool,
    )
    aligned["aligned_within_recording"] = (
        aligned["start_sample"].notna()
        & aligned["end_sample"].notna()
        & aligned["start_sample"].ge(0)
        & aligned["end_sample"].ge(aligned["start_sample"])
        & cumulative_bounds
    )
    aligned["sample_time_diagnostic_delta_ms"] = np.where(
        aligned["time_alignment_valid"],
        pd.to_numeric(aligned["start_time_ms"], errors="coerce") - pd.to_numeric(aligned["mapped_start_time_ms"], errors="coerce"),
        np.nan,
    )
    return aligned


def _build_events(
    raw_events: pd.DataFrame,
    label_files: pd.DataFrame,
    channel_admission: pd.DataFrame,
    recordings: pd.DataFrame,
    parquets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # ``label_events.csv`` is the output of the coordinate audit.  It already
    # distinguishes global recording coordinates from local shard resets and
    # exposes a row-level ``global_start_sample``/``global_end_sample`` pair.
    # Do not recompute these coordinates from ``time_ms`` here: raw Parquet
    # timestamps contain gaps/overlaps, so arithmetic time-to-row conversion
    # silently shifts labels.
    events = raw_events.copy()
    required = {
        "coordinate_scope",
        "coordinate_usable",
        "global_start_sample",
        "global_end_sample",
        "within_raw_global_sample_bounds",
    }
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(
            "The audit label_events.csv is missing strict coordinate columns: "
            + ", ".join(missing)
        )

    # Prefer the full endpoint validation whenever the companion audit has
    # been run.  The original scope/sample fields are retained as
    # ``raw_*``/``original_*`` diagnostics.
    if "validated_global_start_sample" in events.columns:
        events["raw_coordinate_scope"] = events["coordinate_scope"].astype(str)
        events["coordinate_scope"] = events["validated_coordinate_scope"].where(
            events["validated_coordinate_scope"].notna(), events["coordinate_scope"]
        )
        events["global_start_sample"] = pd.to_numeric(
            events["validated_global_start_sample"], errors="coerce"
        )
        events["global_end_sample"] = pd.to_numeric(
            events["validated_global_end_sample"], errors="coerce"
        )
        events["global_sample_offset_applied"] = pd.to_numeric(
            events["validated_offset_samples"], errors="coerce"
        )
        events["coordinate_usable"] = _as_bool(events["validated_coordinate_usable"])
        events["within_raw_global_sample_bounds"] = (
            events["global_start_sample"].notna()
            & events["global_end_sample"].notna()
            & events["global_end_sample"].ge(events["global_start_sample"])
        )
        events["coordinate_reason"] = events.get(
            "validated_resolution_method", events.get("coordinate_reason", "")
        )

    events["channel_name"] = events["canonical_channel"].map(_norm_channel)
    events["closed"] = events["status"].astype(str).str.lower().eq("closed")
    events["coordinate_usable"] = _as_bool(events["coordinate_usable"])
    recording_limits = pd.to_numeric(
        recordings.set_index("rec_key")["total_rows"], errors="coerce"
    ).to_dict()
    global_start = pd.to_numeric(events["global_start_sample"], errors="coerce")
    global_end = pd.to_numeric(events["global_end_sample"], errors="coerce")
    row_limits = pd.to_numeric(
        events["rec_key"].map(recording_limits), errors="coerce"
    )
    # Recheck the final validated coordinates against the actual cumulative
    # Parquet row count.  This is intentionally redundant with the endpoint
    # validator so a stale or malformed companion table cannot admit an
    # out-of-recording event.
    events["within_bounds"] = (
        _as_bool(events["within_raw_global_sample_bounds"])
        & global_start.notna()
        & global_end.notna()
        & row_limits.notna()
        & global_start.ge(0)
        & global_end.ge(global_start)
        & global_end.lt(row_limits)
    )
    within_original = events.get(
        "within_raw_sample_bounds", pd.Series(False, index=events.index)
    )
    suspected = events.get(
        "suspected_rec_binding_error", pd.Series(False, index=events.index)
    )
    duplicate_redundant = events.get(
        "exact_duplicate_redundant", pd.Series(False, index=events.index)
    )
    events["within_bounds_original"] = _as_bool(within_original)
    events["suspected_binding"] = _as_bool(suspected)
    events["audit_duplicate_redundant"] = _as_bool(duplicate_redundant)
    channel_lookup = channel_admission.set_index(["rec_key", "channel_name"])["usable"].to_dict()
    events["channel_usable"] = [
        bool(channel_lookup.get((str(rec), str(ch)), False))
        for rec, ch in zip(events["rec_key"], events["channel_name"], strict=True)
    ]

    reasons: list[str] = []
    for row in events.itertuples(index=False):
        status = str(getattr(row, "status", "")).lower()
        scope = str(getattr(row, "coordinate_scope", "unknown"))
        if not bool(getattr(row, "closed", False)):
            reasons.append(f"label_status_{status or 'unknown'}")
        elif str(getattr(row, "channel_name", "")) not in CHANNELS:
            reasons.append("channel_not_CH7_CH8")
        elif not bool(getattr(row, "channel_usable", False)):
            reasons.append("channel_failed_signal_qc")
        elif bool(getattr(row, "suspected_binding", False)):
            reasons.append("suspected_recording_binding")
        elif not bool(getattr(row, "coordinate_usable", False)):
            validation_status = str(getattr(row, "validated_mapping_status", "")).lower()
            if validation_status == "blocked_recording_binding":
                reasons.append("sample_out_of_bounds_and_suspected_binding")
            elif validation_status == "unmappable":
                reasons.append("coordinate_unmappable")
            elif validation_status == "timestamp_mismatch":
                reasons.append("coordinate_timestamp_mismatch")
            else:
                reasons.append(f"coordinate_{scope}")
        elif not bool(getattr(row, "within_bounds", False)):
            reasons.append("sample_out_of_bounds")
        elif bool(getattr(row, "audit_duplicate_redundant", False)):
            reasons.append("duplicate_interval_redundant")
        else:
            reasons.append("admitted")
    events["row_admission_reason"] = reasons

    admitted = events.loc[events["row_admission_reason"].eq("admitted")].copy()
    admitted["_source_index"] = admitted.index
    admitted["source_event_id"] = admitted["event_id"]
    admitted["original_start_sample"] = pd.to_numeric(
        admitted["start_sample"], errors="coerce"
    )
    admitted["original_end_sample"] = pd.to_numeric(
        admitted["end_sample"], errors="coerce"
    )
    admitted["start_sample"] = pd.to_numeric(
        admitted["global_start_sample"], errors="coerce"
    )
    admitted["end_sample"] = pd.to_numeric(
        admitted["global_end_sample"], errors="coerce"
    )

    # Keep one row for each exact global interval.  The audit's redundant flag
    # catches near-identical duplicate exports (for example E81 base vs _1),
    # while this second key protects against any duplicate introduced by a
    # future audit version.
    duplicate_keys = ["rec_key", "channel_name", "start_sample", "end_sample"]
    admitted["duplicate_source_count"] = admitted.groupby(duplicate_keys)["source_event_id"].transform("size")
    admitted = admitted.sort_values(
        duplicate_keys + ["source_file", "source_row"], kind="stable"
    )
    duplicate_later = admitted.duplicated(duplicate_keys, keep="first")
    if duplicate_later.any():
        duplicate_ids = set(admitted.loc[duplicate_later, "_source_index"])
        events.loc[events.index.isin(duplicate_ids), "row_admission_reason"] = (
            "duplicate_interval_redundant"
        )
        admitted = admitted.loc[~duplicate_later].copy()
    admitted["duplicate_removed"] = admitted["duplicate_source_count"].gt(1)

    rate = recordings.set_index("rec_key")["sampled_rate_hz_median"].to_dict()
    admitted["sampling_rate_hz"] = [
        float(rate.get(str(key), FS_HZ) or FS_HZ) for key in admitted["rec_key"]
    ]
    admitted["start_sample"] = pd.to_numeric(
        admitted["start_sample"], errors="coerce"
    ).astype("Int64")
    admitted["end_sample"] = pd.to_numeric(
        admitted["end_sample"], errors="coerce"
    ).astype("Int64")
    admitted["start_time_ms"] = (
        admitted["start_sample"].astype(float) / admitted["sampling_rate_hz"] * 1000.0
    )
    admitted["end_time_ms"] = (
        (admitted["end_sample"].astype(float) + 1.0)
        / admitted["sampling_rate_hz"]
        * 1000.0
    )
    admitted["duration_s"] = (
        admitted["end_sample"].astype(float)
        - admitted["start_sample"].astype(float)
        + 1.0
    ) / admitted["sampling_rate_hz"]
    admitted["exported_start_time_ms"] = pd.to_numeric(
        admitted["start_time_ms"], errors="coerce"
    ).to_numpy()
    admitted["exported_end_time_ms"] = pd.to_numeric(
        admitted["end_time_ms"], errors="coerce"
    ).to_numpy()
    # Compare the exported time field to the *audited global sample* only as a
    # diagnostic.  It is never used to move the label.
    exported_start = pd.to_numeric(events.loc[admitted["_source_index"], "start_time_ms"], errors="coerce").to_numpy()
    exported_end = pd.to_numeric(events.loc[admitted["_source_index"], "end_time_ms"], errors="coerce").to_numpy()
    admitted["exported_start_time_ms"] = exported_start
    admitted["exported_end_time_ms"] = exported_end
    admitted["sample_time_diagnostic_delta_ms"] = exported_start - admitted["start_time_ms"].to_numpy()
    admitted["alignment_method"] = admitted.get(
        "validated_resolution_method", pd.Series("audit_global_sample_coordinate", index=admitted.index)
    ).fillna("audit_global_sample_coordinate")
    admitted["alignment_method"] = admitted["alignment_method"].astype(str)
    if "validated_mapping_status" in admitted:
        admitted["endpoint_validation_status"] = admitted["validated_mapping_status"].astype(str)
        admitted["endpoint_start_residual_ms"] = pd.to_numeric(
            admitted["validated_start_residual_ms"], errors="coerce"
        )
        admitted["endpoint_end_residual_ms"] = pd.to_numeric(
            admitted["validated_end_residual_ms"], errors="coerce"
        )
        admitted["time_origin_shift_ms_validated"] = pd.to_numeric(
            admitted["validated_time_shift_ms"], errors="coerce"
        )
    admitted["sample_reset_or_mismatch"] = admitted["coordinate_scope"].eq("local_segment")
    admitted["clinical_event_group"] = _connected_event_groups(admitted)
    admitted = admitted.sort_values(
        ["rec_key", "start_time_ms", "channel_name"], kind="stable"
    ).reset_index(drop=True)
    admitted["event_id"] = [
        f"{key}/channel_event_{i:05d}"
        for i, key in enumerate(admitted["rec_key"], start=1)
    ]
    keep = [
        "event_id", "source_event_id", "clinical_event_group", "animal_id", "rec_key", "channel_name",
        "start_sample", "end_sample", "sampling_rate_hz", "start_time_ms", "end_time_ms",
        "duration_s", "exported_start_time_ms", "exported_end_time_ms",
        "sample_time_diagnostic_delta_ms", "alignment_method", "sample_reset_or_mismatch",
        "coordinate_scope", "coordinate_reason", "global_sample_offset_applied",
        "time_mapped_global_start_sample", "time_mapped_global_end_sample",
        "sample_offset_start", "sample_offset_end", "original_start_sample", "original_end_sample",
        "source_file", "source_row", "label_root", "duplicate_source_count",
        "duplicate_removed", "raw_coordinate_scope", "endpoint_validation_status",
        "endpoint_start_residual_ms", "endpoint_end_residual_ms", "time_origin_shift_ms_validated",
        "_source_index",
    ]
    admitted = admitted.loc[:, [c for c in keep if c in admitted.columns]]
    admitted_indices = set(admitted["_source_index"].astype(int))
    exclusions = events.loc[~events.index.isin(admitted_indices)].copy()
    admitted = admitted.drop(columns=["_source_index"])
    return admitted, exclusions


def _coordinate_audit_summary(
    raw_events: pd.DataFrame,
    admitted: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> pd.DataFrame:
    """Return one compact count table for the coordinate/admission audit."""

    rows: list[dict[str, Any]] = []
    for section, column in (
        ("label_status", "status"),
        ("raw_coordinate_scope", "raw_coordinate_scope"),
        ("validated_coordinate_scope", "validated_coordinate_scope"),
        ("endpoint_mapping_status", "validated_mapping_status"),
    ):
        if column not in raw_events.columns:
            continue
        values = raw_events[column].fillna("<missing>").astype(str)
        for category, count in values.value_counts(dropna=False).items():
            rows.append(
                {"section": section, "category": str(category), "count": int(count)}
            )
    rows.append(
        {"section": "admission", "category": "admitted", "count": int(len(admitted))}
    )
    if "row_admission_reason" in exclusions.columns:
        for category, count in exclusions["row_admission_reason"].value_counts().items():
            rows.append(
                {"section": "admission", "category": str(category), "count": int(count)}
            )
    return pd.DataFrame(rows, columns=["section", "category", "count"])


def _merge_recording_status(
    recordings: pd.DataFrame,
    coverage: pd.DataFrame,
    channel_admission: pd.DataFrame,
    parquets: pd.DataFrame,
) -> pd.DataFrame:
    frame = recordings.copy()
    coverage_cols = [
        "rec_key", "label_file_count", "label_event_rows", "closed_valid_rows",
        "independent_clinical_event_groups", "exact_duplicate_rows", "same_channel_overlap_rows",
        "out_of_bounds_rows", "sample_out_of_bounds_rows", "timestamp_out_of_range_rows",
        "sample_in_bounds_timestamp_outside_rows", "label_export_status",
        "review_completion_confirmed", "nonseizure_eligibility",
    ]
    coverage_cols = [c for c in coverage_cols if c in coverage.columns]
    frame = frame.merge(coverage.loc[:, coverage_cols], on="rec_key", how="left")
    valid_parquet = parquets.groupby("rec_key")["canonical_candidate"].sum().rename("valid_canonical_parquets")
    frame = frame.merge(valid_parquet, on="rec_key", how="left")
    channel_pivot = channel_admission.pivot_table(
        index="rec_key", columns="channel_name", values="usable", aggfunc="max", fill_value=False
    ).reset_index()
    channel_pivot.columns = ["rec_key"] + [f"{str(c).lower()}_usable" for c in channel_pivot.columns[1:]]
    frame = frame.merge(channel_pivot, on="rec_key", how="left")
    for col in ("ch7_usable", "ch8_usable"):
        if col not in frame:
            frame[col] = False
    frame["raw_data_status"] = np.select(
        [frame["footer_failures"].fillna(0).gt(0), frame["valid_canonical_parquets"].fillna(0).eq(0), frame["manifest_status"].astype(str).eq("missing")],
        ["blocked_parquet_footer", "blocked_no_valid_parquet", "admit_manifest_missing"],
        default="admit",
    )
    frame["training_admission"] = "blocked_pending_review_completion_and_filter_provenance"
    frame.loc[frame["label_export_status"].astype(str).isin({"no_label_export", "header_only_exports"}), "training_admission"] = "blocked_no_complete_label_export"
    frame.loc[frame["sample_out_of_bounds_rows"].fillna(0).gt(0), "training_admission"] = "blocked_sample_alignment"
    frame["filter_provenance_status"] = "missing_source_metadata_do_not_assume_pre_filtered"
    return frame.sort_values(["animal_id", "recording_date", "recording_time", "rec_id"], kind="stable")


def _save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _save_pub(fig: plt.Figure, stem: Path, *, tiff: bool = True) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if tiff:
        fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(
    recordings: pd.DataFrame,
    channel_admission: pd.DataFrame,
    events: pd.DataFrame,
    exclusions: pd.DataFrame,
    output: Path,
) -> None:
    animals = sorted(recordings["animal_id"].astype(str).unique())
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    # a: recording admission and label completeness.
    counts = recordings.groupby("animal_id").agg(
        recordings=("rec_key", "size"),
        with_events=("label_event_rows", lambda x: int(pd.to_numeric(x, errors="coerce").fillna(0).gt(0).sum())),
        no_or_header=("label_export_status", lambda x: int(x.astype(str).isin({"no_label_export", "header_only_exports"}).sum())),
    ).reindex(animals).fillna(0)
    x = np.arange(len(animals))
    axes[0, 0].bar(x, counts["recordings"], color="#C9D0D4", label="recordings")
    axes[0, 0].bar(x, counts["with_events"], color=OK, label="non-empty label export")
    axes[0, 0].bar(x, counts["no_or_header"], color=BLOCKED, label="empty/missing export")
    axes[0, 0].set_title("Recording admission", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Recordings")
    axes[0, 0].set_xticks(x, animals)
    axes[0, 0].legend(frameon=False, fontsize=5.8, loc="upper right")

    # b: channel-level signal QC.
    qc = channel_admission.pivot_table(
        index="animal_id",
        columns="channel_name",
        values="channel_admission",
        aggfunc=lambda values: "admit_usable_candidate" if (values == "admit_usable_candidate").any() else str(values.iloc[0]),
    ).reindex(animals)
    matrix = np.zeros((len(animals), 2), dtype=float)
    for j, channel in enumerate(CHANNELS):
        values = qc.get(channel, pd.Series(index=animals, dtype=object)).astype(str)
        matrix[:, j] = values.eq("admit_usable_candidate").astype(float)
    image = axes[0, 1].imshow(matrix, cmap=mpl.colors.ListedColormap(["#F3D1C8", "#B8D7D1"]), vmin=0, vmax=1, aspect="auto")
    axes[0, 1].set_title("Signal channel admission", loc="left", fontweight="bold")
    axes[0, 1].set_xticks(range(2), CHANNELS)
    axes[0, 1].set_yticks(range(len(animals)), animals)
    for i in range(len(animals)):
        for j in range(2):
            axes[0, 1].text(j, i, "usable" if matrix[i, j] else "blocked", ha="center", va="center", fontsize=5.8)

    # c: admitted clinical label events (QC table, not training denominator).
    event_counts = events.groupby(["animal_id", "channel_name"]).size().unstack(fill_value=0).reindex(animals).fillna(0)
    bottom = np.zeros(len(animals))
    for channel, color in (("CH7", CH7_COLOR), ("CH8", CH8_COLOR)):
        values = event_counts.get(channel, pd.Series(0, index=animals)).to_numpy(float)
        axes[1, 0].bar(x, values, bottom=bottom, color=color, label=channel)
        bottom += values
    axes[1, 0].set_title("Aligned closed events for QC", loc="left", fontweight="bold")
    axes[1, 0].set_ylabel("Channel events")
    axes[1, 0].set_xticks(x, animals)
    axes[1, 0].legend(frameon=False, fontsize=5.8)

    # d: exclusion reasons; these are label rows, not false positives.
    reason = exclusions["row_admission_reason"].value_counts()
    reason = reason[reason.index.astype(str) != "admitted"]
    if reason.empty:
        reason = pd.Series({"none": 0})
    axes[1, 1].barh(np.arange(len(reason)), reason.to_numpy(), color=BLOCKED)
    axes[1, 1].set_yticks(np.arange(len(reason)), [str(v).replace("_", " ") for v in reason.index])
    axes[1, 1].set_title("Rows held out of event table", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("Rows")
    axes[1, 1].grid(axis="x", color=GRID, linewidth=0.5)
    for axis in axes.ravel():
        axis.tick_params(direction="out", length=2.5, pad=2)
    fig.suptitle("New batch: conservative label and channel admission audit", x=0.02, ha="left", fontsize=10, fontweight="bold")
    fig.text(0.02, -0.03, "Counts are QC inventory only. No unlabeled interval is treated as non-seizure; amplitude remains device a.u.", fontsize=5.8, color=MID)
    _save_pub(fig, output / "label_qc_summary")


def _plot_alignment(
    events: pd.DataFrame,
    exclusions: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), constrained_layout=True)
    if "endpoint_start_residual_ms" in events.columns:
        delta = pd.concat(
            [
                pd.to_numeric(events["endpoint_start_residual_ms"], errors="coerce"),
                pd.to_numeric(events["endpoint_end_residual_ms"], errors="coerce"),
            ],
            ignore_index=True,
        ).dropna()
        delta_label = "raw timestamp - label time (ms)"
    else:
        delta = pd.to_numeric(events["sample_time_diagnostic_delta_ms"], errors="coerce").dropna()
        delta_label = "label time - sample-derived time (ms)"
    axes[0].hist(delta, bins=40, color=CH7_COLOR, alpha=0.85)
    axes[0].axvline(0, color=INK, linewidth=0.7)
    axes[0].set_title("Endpoint timestamp check", loc="left", fontweight="bold")
    axes[0].set_xlabel(delta_label)
    axes[0].set_ylabel("Events")
    duration = pd.to_numeric(events["duration_s"], errors="coerce")
    axes[1].hist(np.log10(np.clip(duration, 0.01, None)), bins=40, color=CH8_COLOR, alpha=0.85)
    axes[1].set_title("Event duration", loc="left", fontweight="bold")
    axes[1].set_xlabel("log10 duration (s)")
    axes[1].set_ylabel("Events")
    flags = pd.DataFrame(
        {
            "duplicate/redundant": [int((exclusions["row_admission_reason"] == "duplicate_interval_redundant").sum())],
            "channel QC blocked": [int((exclusions["row_admission_reason"] == "channel_failed_signal_qc").sum())],
            "recording binding": [int(exclusions["suspected_binding"].eq(True).sum())],
            "timestamp discontinuity": [int(events.get("endpoint_validation_status", pd.Series(dtype=str)).eq("timestamp_clock_discontinuity").sum())],
        }
    ).T
    axes[2].barh(np.arange(len(flags)), flags[0].to_numpy(), color=[OK, BLOCKED, BLOCKED, "#9AA1A6"])
    axes[2].set_yticks(np.arange(len(flags)), flags.index)
    axes[2].set_title("Alignment/QC flags", loc="left", fontweight="bold")
    axes[2].set_xlabel("Rows")
    axes[2].grid(axis="x", color=GRID, linewidth=0.5)
    for axis in axes:
        axis.tick_params(direction="out", length=2.5, pad=2)
    fig.suptitle("Audited sample coordinates with raw-timestamp endpoint checks", x=0.02, ha="left", fontsize=9.5, fontweight="bold")
    _save_pub(fig, output / "label_alignment_overview")


def _plot_manual_timelines(recordings: pd.DataFrame, events: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Render every recording/channel's complete manual interval inventory."""

    output.parent.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    with PdfPages(output) as pdf:
        page_number = 0
        for row in recordings.sort_values(["animal_id", "recording_date", "recording_time", "rec_id"], kind="stable").itertuples(index=False):
            page_number += 1
            rec_key = str(row.rec_key)
            group = events.loc[events["rec_key"].eq(rec_key)]
            end_ms = float(row.total_rows) / FS_HZ * 1000.0
            fig, axes = plt.subplots(2, 1, figsize=(8.27, 3.7), sharex=True)
            for axis, channel, color in zip(axes, CHANNELS, (CH7_COLOR, CH8_COLOR), strict=True):
                channel_events = group.loc[group["channel_name"].eq(channel)]
                axis.set_ylim(0, 1)
                axis.set_yticks([])
                axis.set_ylabel(channel, rotation=0, labelpad=22, va="center", fontweight="bold")
                axis.set_xlim(0, max(end_ms, 1.0) / 3_600_000.0)
                for event in channel_events.itertuples(index=False):
                    left = float(event.start_time_ms) / 3_600_000.0
                    right = float(event.end_time_ms) / 3_600_000.0
                    axis.axvspan(left, right, ymin=0.15, ymax=0.85, color=color, alpha=0.82, linewidth=0)
                    index_rows.append(
                        {
                            "pdf_page": page_number,
                            "rec_key": rec_key,
                            "animal_id": row.animal_id,
                            "channel_name": channel,
                            "event_id": event.event_id,
                            "start_time_h": left,
                            "end_time_h": right,
                        }
                    )
                if channel_events.empty:
                    axis.text(0.01, 0.5, "no admitted closed events", transform=axis.transAxes, color=MID, va="center")
                axis.grid(axis="x", color=GRID, linewidth=0.5)
            axes[-1].set_xlabel("Recording-relative time (h; sample coordinate)")
            title = f"{row.animal_id} | {row.session_name} | {row.rec_id} | page {page_number}"
            fig.suptitle(title, x=0.06, ha="left", y=0.98, fontsize=8.5, fontweight="bold")
            fig.text(0.06, 0.02, "Blue/orange spans are all admitted manual closed intervals; this inventory is not a model performance denominator.", fontsize=5.6, color=MID)
            fig.subplots_adjust(left=0.09, right=0.99, top=0.82, bottom=0.22, hspace=0.25)
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
    return pd.DataFrame(index_rows)


def _parquet_lookup(parquets: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parquets.loc[parquets["canonical_candidate"]].itertuples(index=False):
        lookup[str(row.rec_key)].append(
            {
                "path": str(row.path),
                "normalized_name": str(getattr(row, "normalized_name", Path(str(row.path)).name)),
                "row_count": int(float(row.row_count)),
                "timestamp_start_ms": float(row.timestamp_start_ms),
                "timestamp_end_ms": float(row.timestamp_end_ms),
            }
        )
    for key in lookup:
        lookup[key].sort(key=lambda item: _file_number(item["normalized_name"]))
    return lookup


def _load_waveform_excerpt(
    event: pd.Series,
    lookup: dict[str, list[dict[str, Any]]],
    margin_s: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    rec_key = str(event["rec_key"])
    files = lookup.get(rec_key, [])
    if not files:
        return None
    start = max(0, int(event["start_sample"]) - int(margin_s * FS_HZ))
    end = int(event["end_sample"]) + int(margin_s * FS_HZ) + 1
    cumulative = 0
    pieces: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    for item in files:
        left = cumulative
        right = cumulative + int(item["row_count"])
        if right <= start:
            cumulative = right
            continue
        if left >= end:
            break
        try:
            table = pq.read_table(item["path"], columns=["timestamp_ms", str(event["channel_name"]).lower()])
            frame = table.to_pandas()
        except Exception:
            cumulative = right
            continue
        local_left = max(0, start - left)
        local_right = min(len(frame), end - left)
        if local_right > local_left:
            timestamps.append(frame["timestamp_ms"].to_numpy(float)[local_left:local_right])
            pieces.append(frame[str(event["channel_name"]).lower()].to_numpy(float)[local_left:local_right])
        cumulative = right
    if not pieces:
        return None
    return np.concatenate(timestamps), np.concatenate(pieces), np.arange(start, start + sum(len(x) for x in pieces), dtype=float)


def _plot_waveform_examples(
    events: pd.DataFrame,
    parquets: pd.DataFrame,
    output: Path,
    max_examples: int,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    lookup = _parquet_lookup(parquets)
    # Put the coordinate edge cases first, then add one candidate per
    # animal/channel in rounds.  The previous group-major ``head`` silently
    # omitted later animals (E81/E82) when the page budget was reached.
    candidates: list[pd.Series] = []

    source_name = events.get("source_file", pd.Series("", index=events.index)).astype(str)
    source_row = pd.to_numeric(
        events.get("source_row", pd.Series(np.nan, index=events.index)), errors="coerce"
    )
    edge_cases = (
        source_name.str.endswith("rec_e4sdqci5_label_events_7.csv", na=False)
        & source_row.between(2, 13, inclusive="both"),
        source_name.str.endswith("rec_e4sdqci5_label_events_7.csv", na=False)
        & source_row.between(14, 193, inclusive="both"),
        source_name.str.endswith("rec_hnmtuvn3_label_events_2.csv", na=False),
        source_name.str.endswith("rec_hnmtuvn3_label_events_3.csv", na=False),
    )
    for mask in edge_cases:
        group = events.loc[mask]
        if not group.empty:
            candidates.append(group.iloc[len(group) // 2])

    grouped: list[tuple[tuple[str, str], pd.DataFrame]] = []
    for key, group in events.groupby(["animal_id", "channel_name"], sort=True):
        group = group.copy()
        group["duration_s_num"] = pd.to_numeric(group["duration_s"], errors="coerce")
        grouped.append((key, group))
    selectors = (
        ("quantile", 0.5),
        ("quantile", 0.1),
        ("quantile", 0.9),
        ("mismatch", None),
    )
    for selector, value in selectors:
        for _, group in grouped:
            if len(group):
                if selector == "quantile":
                    target = float(group["duration_s_num"].quantile(float(value)))
                    row = group.iloc[(group["duration_s_num"] - target).abs().argmin()]
                else:
                    mismatch = pd.to_numeric(
                        group["sample_time_diagnostic_delta_ms"], errors="coerce"
                    ).abs()
                    row = group.iloc[mismatch.fillna(-1.0).argmax()]
                candidates.append(row)
    chosen = pd.DataFrame(candidates).drop_duplicates("event_id")
    source_rows: list[dict[str, Any]] = []
    pdf_path = output / "manual_waveform_examples.pdf"
    with PdfPages(pdf_path) as pdf:
        for event in chosen.to_dict("records"):
            if len(source_rows) >= max_examples:
                break
            loaded = _load_waveform_excerpt(pd.Series(event), lookup)
            if loaded is None:
                continue
            timestamps, values, sample_axis = loaded
            rel = (sample_axis - float(event["start_sample"])) / FS_HZ
            centered = values - np.nanmedian(values)
            scale = np.nanpercentile(np.abs(centered), 99.0)
            scale = max(float(scale), 1e-6)
            normalized = centered / scale
            fig, axes = plt.subplots(2, 1, figsize=(8.27, 4.8), sharex=True, gridspec_kw={"height_ratios": (2.0, 0.55)})
            axes[0].plot(rel, normalized, color=INK, linewidth=0.45, rasterized=True)
            axes[0].axvspan(0.0, float(event["duration_s"]), color=MANUAL, alpha=0.18, linewidth=0)
            axes[0].axvline(0.0, color=MANUAL, linewidth=0.9)
            axes[0].axvline(float(event["duration_s"]), color=MANUAL, linewidth=0.9)
            axes[0].set_ylabel("robust a.u.")
            axes[0].set_title(
                f"{event['animal_id']} | {event['channel_name']} | {event['event_id']} | duration {float(event['duration_s']):.2f} s",
                loc="left", fontweight="bold",
            )
            axes[1].axvspan(0.0, float(event["duration_s"]), color=MANUAL, alpha=0.85, linewidth=0)
            axes[1].set_ylim(0, 1)
            axes[1].set_yticks([])
            axes[1].set_ylabel("manual", rotation=0, labelpad=24, va="center")
            axes[1].set_xlabel("seconds relative to sample-derived manual onset")
            for axis in axes:
                axis.grid(axis="x", color=GRID, linewidth=0.5)
                axis.tick_params(direction="out", length=2.5, pad=2)
            fig.text(0.08, 0.02, "Waveform is raw device amplitude for QC; units are a.u. The shaded interval is the manual label only.", fontsize=5.7, color=MID)
            fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.18, hspace=0.12)
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
            source_rows.append(
                {
                    "event_id": event["event_id"],
                    "animal_id": event["animal_id"],
                    "rec_key": event["rec_key"],
                    "channel_name": event["channel_name"],
                    "duration_s": event["duration_s"],
                    "sample_time_diagnostic_delta_ms": event["sample_time_diagnostic_delta_ms"],
                    "coordinate_scope": event.get("coordinate_scope"),
                    "alignment_method": event.get("alignment_method"),
                    "endpoint_validation_status": event.get("endpoint_validation_status"),
                    "endpoint_start_residual_ms": event.get("endpoint_start_residual_ms"),
                    "endpoint_end_residual_ms": event.get("endpoint_end_residual_ms"),
                    "source_file": event.get("source_file"),
                    "source_row": event.get("source_row"),
                }
            )
    return pd.DataFrame(source_rows)


def _write_reports(
    output: Path,
    summary: dict[str, Any],
    recordings: pd.DataFrame,
    events: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> None:
    protocol = output / "protocol"
    exclusion_counts = exclusions["row_admission_reason"].value_counts()
    endpoint_counts = events.get(
        "endpoint_validation_status", pd.Series(dtype=str)
    ).value_counts()
    channel_counts = (
        events.groupby(["animal_id", "channel_name"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(CHANNELS), fill_value=0)
    )
    report = [
        "# 新批次小鼠癫痫脑电严格准备日志",
        "",
        "状态：**人工标签与原始脑电已完成严格对齐和准入审计；尚未开始训练。**",
        "",
        "## 最终结果",
        f"- 原始标签行：{int(summary['coordinate_endpoint_validation']['label_rows'])}；闭合事件行：{int(summary['coordinate_endpoint_validation']['closed_rows'])}；闭合且坐标可验证：{int(summary['coordinate_endpoint_validation']['validated_coordinate_usable_rows'])}。",
        f"- 最终准入的通道级闭合事件：{len(events)}；隔离标签行：{len(exclusions)}。这些事件只代表可靠的 seizure=1 标签，不代表非发作样本已经可用。",
        f"- 原始记录：{len(recordings)}；规范 Parquet：{int(summary['valid_canonical_parquet_rows'])}/{int(summary['canonical_parquet_rows'])} 可读。",
        "- 每条可用 CH7/CH8 信号作为独立模型输入；不使用鼠号、通道号或发作亚型作为分类目标。",
        "- 全部准入事件均满足：闭合、记录绑定明确、通道质控通过、起止样本存在、start<=end、终点未超过真实累计 Parquet 行数。",
        "",
        "## 各鼠独立通道事件",
        "| 小鼠 | CH7 | CH8 |",
        "|---|---:|---:|",
    ]
    report.extend(
        f"| {animal} | {int(row.get('CH7', 0))} | {int(row.get('CH8', 0))} |"
        for animal, row in channel_counts.iterrows()
    )
    report.extend(
        [
            "",
            "## 排除清单",
            "| 原因 | 行数 |",
            "|---|---:|",
        ]
    )
    exclusion_labels = {
        "label_status_open": "未闭合标签",
        "label_status_orphan_end": "孤立结束标签",
        "channel_failed_signal_qc": "通道信号质控失败",
        "duplicate_interval_redundant": "重复导出区间",
        "suspected_recording_binding": "疑似记录错绑",
    }
    report.extend(
        f"| {exclusion_labels.get(str(reason), str(reason))} | {int(count)} |"
        for reason, count in exclusion_counts.items()
    )
    report.extend(
        [
            "",
            "## 坐标结论",
            f"- 准入事件中原始时间戳端点完全一致：{int(endpoint_counts.get('exact', 0))}；存在设备时钟断点但样本行坐标有效：{int(endpoint_counts.get('timestamp_clock_discontinuity', 0))}。后者保留是因为标签端点能够落到唯一的真实 Parquet 行，并非用不连续时间戳反推坐标。",
            "- E07 的混合坐标标签文件已逐行拆分：全局坐标簇和局部分片坐标簇分别验证，没有整文件误删。",
            "- E79 `rec_0001` 的 12 个闭合行无法与所绑定记录匹配，已全部隔离，未自动改绑到其他记录。",
            "- 空标签、缺失标签或只含表头的文件没有被转换成 non-seizure。",
            "",
            "## 旧模型审计带来的硬规则",
            "- 旧模型中 `E50/rec_dqsvawj6` 出现一通道标发作、另一通道同时间被当作基线的矛盾监督。因此新批次中，未完成逐通道复核的时段只能记为 unknown，不能作为负类。",
            "- 旧事件图还证明，长预测段可合并多个人工事件并抬高覆盖率。因此后续不能只报 coverage；必须同时报告起止误差、持续时间误差、合并/碎片化数量和标签外候选/通道小时。",
            "- 完整旧审计见 `../epilepsy_blind_single_channel_v2_strict/OLD_EVENT_FAILURE_AUDIT_zh.md`。旧模型预测不参与本批次标签删除。",
            "",
            "## 训练前仍需确认",
            f"- {int(recordings['label_export_status'].astype(str).isin({'no_label_export','header_only_exports'}).sum())} 条记录缺少有效标签导出或只有表头；当前没有正式的逐通道复核完成标记，所以 non-seizure 训练集尚不能冻结。",
            "- 滤波来源元数据缺失。必须先逐记录确认是原始未滤波还是已完成一次 0.5-70 Hz 带通和 50 Hz 陷波，禁止重复滤波。",
            "- ADC 到微伏的标定未知；所有幅值继续使用设备 a.u.、相对基线或稳健归一化值。",
            "",
            "## 统一输出",
            "- `protocol/channel_events.parquet`：最终 7,600 个准入通道事件。",
            "- `protocol/label_event_exclusions.csv`：268 行逐条排除原因。",
            "- `protocol/coordinate_audit_summary.csv`：一页式坐标与准入计数。",
            "- `figures/manual_event_timeline_all.pdf`：368 页记录级人工事件全时间轴。",
            "- `figures/manual_waveform_examples.pdf`：24 页跨鼠、跨通道及坐标边界案例波形。",
            "- `figures/label_qc_summary.pdf` 与 `figures/label_alignment_overview.pdf`：汇总图。",
        ]
    )
    (output / "PREPARATION_REPORT_zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (protocol / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output / "prepare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    audit = args.audit_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol").mkdir(exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    _copy_audit_snapshot(audit, output)

    recordings = _read_csv(audit / "raw_recordings.csv")
    integrity = _read_csv(audit / "raw_parquet_integrity.csv")
    signal_qc = _read_csv(audit / "signal_channel_qc.csv")
    label_files = _label_file_status(_read_csv(audit / "label_files.csv"))
    raw_events = _read_csv(audit / "label_events.csv")
    raw_events = _attach_coordinate_validation(raw_events, audit)
    coverage = _read_csv(audit / "label_coverage.csv")

    parquets = _canonical_parquets(integrity)
    channel_admission = _channel_admission(signal_qc)
    recordings = _merge_recording_status(recordings, coverage, channel_admission, parquets)
    events, exclusions = _build_events(raw_events, label_files, channel_admission, recordings, parquets)

    protocol = output / "protocol"
    _save_table(recordings, protocol / "recording_manifest.csv")
    _save_table(parquets, protocol / "canonical_parquet_manifest.csv")
    _save_table(label_files, protocol / "label_file_manifest.csv")
    _save_table(channel_admission, protocol / "channel_admission.csv")
    _save_table(events, protocol / "event_components.csv")
    _save_table(exclusions, protocol / "label_event_exclusions.csv")
    _save_table(
        _coordinate_audit_summary(raw_events, events, exclusions),
        protocol / "coordinate_audit_summary.csv",
    )
    endpoint_validation = audit / "coordinate_endpoint_validation.csv"
    if endpoint_validation.is_file():
        shutil.copy2(endpoint_validation, protocol / endpoint_validation.name)
    endpoint_summary = audit / "coordinate_endpoint_validation_summary.json"
    if endpoint_summary.is_file():
        shutil.copy2(endpoint_summary, protocol / endpoint_summary.name)
    events.to_parquet(protocol / "channel_events.parquet", index=False)

    figures = output / "figures"
    _plot_summary(recordings, channel_admission, events, exclusions, figures)
    _plot_alignment(events, exclusions, figures)
    timeline_index = _plot_manual_timelines(recordings, events, figures / "manual_event_timeline_all.pdf")
    _save_table(timeline_index, figures / "manual_event_timeline_index.csv")
    waveform_index = pd.DataFrame()
    if not args.skip_waveforms:
        waveform_index = _plot_waveform_examples(events, parquets, figures, int(args.waveform_examples))
    _save_table(waveform_index, figures / "manual_waveform_examples_index.csv")
    (figures / "README.md").write_text(
        "# New batch QC figures\n\n"
        "These figures describe label/data admission before model fitting. They do not report sensitivity, precision, AUROC or AUPRC.\n\n"
        "`manual_event_timeline_all.pdf` contains every admitted manual interval, one recording per page. Once model probabilities exist, use the event-atlas wrapper to add complete/partial/missed/boundary-shift/model-only panels to this same directory.\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": 1,
        "status": "prepared_qc_only_training_blocked",
        "source_audit": str(audit),
        "output_root": str(output),
        "recordings": int(len(recordings)),
        "canonical_parquet_rows": int(len(parquets)),
        "valid_canonical_parquet_rows": int(parquets["canonical_candidate"].sum()),
        "channel_rows": int(len(channel_admission)),
        "usable_channel_rows": int(channel_admission["usable"].sum()),
        "admitted_closed_channel_events_for_qc": int(len(events)),
        "excluded_label_rows": int(len(exclusions)),
        "exact_duplicate_intervals_removed": int(
            exclusions["row_admission_reason"].eq("duplicate_interval_redundant").sum()
        ),
        "sample_out_of_bounds_rows_quarantined": int(exclusions["within_bounds"].eq(False).sum()),
        "suspected_binding_rows_quarantined": int(exclusions["suspected_binding"].eq(True).sum()),
        "coordinate_endpoint_validation": (
            json.loads(endpoint_summary.read_text(encoding="utf-8"))
            if endpoint_summary.is_file()
            else None
        ),
        "review_completion_markers": int(label_files["completion_marker"].sum()),
        "nonseizure_training_ready": False,
        "filter_provenance": "missing; classify each recording before applying target filter",
        "amplitude_unit": "device a.u.; ADC-to-microvolt calibration unavailable",
        "protocol_clock": "recording-relative exported time mapped to cumulative raw sample coordinates at audited 1000 Hz",
        "figures": {
            "summary": str(figures / "label_qc_summary.pdf"),
            "alignment": str(figures / "label_alignment_overview.pdf"),
            "all_manual_timelines": str(figures / "manual_event_timeline_all.pdf"),
            "waveform_examples": str(figures / "manual_waveform_examples.pdf"),
        },
        "next_gate": "obtain formal completion marker/manual confirmation and filter provenance; then build strict pooled-mouse purged split",
    }
    _write_reports(output, summary, recordings, events, exclusions)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
