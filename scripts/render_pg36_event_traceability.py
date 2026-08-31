#!/usr/bin/env python
"""Render clinically traceable PG36 event evidence.

The formal PG36 label is an *event x independent channel* assignment.  This
renderer therefore never treats the first 4-s window as a representative of
an event.  It joins the frozen event label to every physical window, selects a
representative by robust distance to the event feature median, and writes a
traceability index containing the exact cache location for every displayed
window.  A page contains a continuous filtered waveform, the manual event
interval, and a compact strip of all windows in that event.

Long or overlapping annotations are retained in the index and marked for
review.  They are not silently removed from the scientific source tables.
Use ``--exclude-qc`` only for a review-only PDF; the index remains complete.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.6,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
# Results live in the sibling ``sleep_staging`` project, not inside the
# source repository.  Deriving this path from the repository location avoids
# hard-coded non-ASCII drive paths and prevents accidentally reading a stale
# output tree from another checkout.
DATA_ROOT = PROJECT_ROOT.parent / "sleep_staging"

from mouse_sleep.epilepsy.single_channel_visualization import (  # noqa: E402
    TraceStore,
    _waveform_display,
)


TAX_ROOT = (
    DATA_ROOT
    / "output"
    / "epilepsy_program_20260821"
    / "02_morphology_taxonomy"
    / "all_mice_old_plus_new"
    / "strict_all_channel_v1"
)
PROTOCOL_ROOT = (
    DATA_ROOT
    / "output"
    / "epilepsy_program_20260821"
    / "01_annotation_detection"
    / "03_combined_old_plus_new"
    / "protocol"
)
DEFAULT_OUT = (
    DATA_ROOT
    / "output"
    / "epilepsy_program_20260821"
    / "02_morphology_taxonomy"
    / "all_mice_old_plus_new"
    / "strict_all_channel_v1"
    / "07_clinical_traceability_pg36_v1"
)

KEYS = ["animal_id", "session_id", "event_id", "channel_name"]
FEATURES = [
    "log_robust_amplitude",
    "log_rms",
    "log_line_length",
    "hjorth_mobility",
    "hjorth_complexity",
    "peak_frequency_hz",
    "spectral_entropy",
    "relative_delta",
    "relative_theta",
    "relative_beta",
    "relative_low_gamma_clean",
    "relative_high_gamma",
    "baseline_z__log_robust_amplitude",
    "baseline_z__log_rms",
    "baseline_z__log_line_length",
    "baseline_z__hjorth_mobility",
    "baseline_z__hjorth_complexity",
    "baseline_z__peak_frequency_hz",
    "baseline_z__spectral_entropy",
    "baseline_z__relative_delta",
    "baseline_z__relative_theta",
    "baseline_z__relative_beta",
    "baseline_z__relative_low_gamma_clean",
    "baseline_z__relative_high_gamma",
]
CHANNELS = ("CH7", "CH8")
MANUAL = "#1769AA"
INK = "#23282D"
MUTED = "#66717A"
GRID = "#D9DEE3"
ACCENT = "#D55E00"
DISTANCE = "#6A3D9A"
# Explicit journal-width contract (single/double-column-safe).  Keeping the
# millimetre value as a named constant also lets the source preflight verify
# the intended final width instead of mistaking the conversion numerator for
# an inch value.
width_mm = 183.0
FIGURE_WIDTH_IN = width_mm / 25.4


def natural(value: object) -> tuple[int, str]:
    match = re.search(r"(\d+)", str(value))
    return (int(match.group(1)) if match else 10**9, str(value))


def pg_colour(label: str) -> str:
    """Stable muted categorical colour; PG identity is still text-labelled."""

    match = re.search(r"(\d+)", str(label))
    index = int(match.group(1)) - 1 if match else 0
    palette = list(mpl.colormaps["tab20"].colors)
    return mpl.colors.to_hex(palette[index % len(palette)])


def _numeric_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = frame[[column for column in FEATURES if column in frame]].replace(
        [np.inf, -np.inf], np.nan
    )
    values = values.apply(pd.to_numeric, errors="coerce")
    return values.fillna(values.median()).fillna(0.0).to_numpy(float)


def _representative(window_group: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    """Return a real physical window nearest the event feature median."""

    work = window_group.sort_values(
        ["start_time_ms", "cache_local_index"], kind="stable"
    ).reset_index(drop=True)
    matrix = _numeric_matrix(work)
    center = np.nanmedian(matrix, axis=0)
    scale = np.nanmedian(np.abs(matrix - center), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    distance = np.sqrt(np.nanmean(((matrix - center) / scale) ** 2, axis=1))
    distance = np.where(np.isfinite(distance), distance, np.inf)
    order = np.lexsort(
        (
            work["cache_local_index"].to_numpy(np.int64),
            work["start_time_ms"].to_numpy(float),
            distance,
        )
    )
    selected = work.iloc[int(order[0])].copy()
    selected["window_distance_to_event_median"] = float(distance[int(order[0])])
    return selected, distance


def _qc_events(protocol_events: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Audit duration, overlap chains, and physical-window support."""

    rows: list[dict[str, object]] = []
    grouped = protocol_events.copy()
    grouped["channel_name"] = grouped["channel_name"].astype(str).str.upper()
    grouped = grouped.loc[grouped.channel_name.isin(CHANNELS)].copy()
    window_key_frame = windows[KEYS].astype(str).copy()
    window_counts = window_key_frame.groupby(KEYS, sort=False).size().to_dict()
    for key, group in grouped.groupby(["animal_id", "session_id", "channel_name"], sort=True):
        group = group.sort_values(["start_time_ms", "end_time_ms"], kind="stable").reset_index(drop=True)
        # Linear sweep: an overlap chain is a maximal run whose next start is
        # before the current union end.  The previous implementation compared
        # every event with every other event, which made a 6,000-event stream
        # quadratic and could stall rendering before any figure was written.
        chain_ids: list[int] = []
        chain_id = -1
        chain_end = -np.inf
        for event in group.itertuples(index=False):
            start, end = float(event.start_time_ms), float(event.end_time_ms)
            if start >= chain_end:
                chain_id += 1
                chain_end = end
            else:
                chain_end = max(chain_end, end)
            chain_ids.append(chain_id)
        chain_counts = pd.Series(chain_ids).value_counts().to_dict()
        for row_index, event in enumerate(group.itertuples(index=False)):
            event_id = str(event.event_id)
            duration = max(0.0, float(event.end_time_ms) - float(event.start_time_ms)) / 1000.0
            chain_id_for_event = int(chain_ids[row_index])
            chain_size_final = int(chain_counts.get(chain_id_for_event, 1))
            overlap_before = chain_size_final > 1
            n_windows = int(window_counts.get((str(event.animal_id), str(event.session_id), event_id, str(event.channel_name).upper()), 0))
            reasons: list[str] = []
            if duration > 120.0:
                reasons.append("duration_gt_120s")
            if overlap_before:
                reasons.append("overlap_chain")
            if n_windows == 0:
                reasons.append("no_physical_windows")
            rows.append(
                {
                    "animal_id": str(event.animal_id),
                    "session_id": str(event.session_id),
                    "event_id": event_id,
                    "channel_name": str(event.channel_name).upper(),
                    "split": str(getattr(event, "split", "unknown")).lower(),
                    "event_start_ms": float(event.start_time_ms),
                    "event_end_ms": float(event.end_time_ms),
                    "duration_s": duration,
                    "n_windows": n_windows,
                    "overlap_chain_size": chain_size_final,
                    "qc_status": "review_required" if reasons else "eligible",
                    "qc_reasons": ";".join(reasons),
                }
            )
    return pd.DataFrame(rows)


def _load_inputs(tax_root: Path, protocol_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_features = pd.read_parquet(tax_root / "01_tables" / "channel_event_features.parquet")
    window_path = tax_root / "04_label_freeze_pooled" / "pooled_channel_window_labels.parquet"
    event_labels = pd.read_parquet(tax_root / "04_label_freeze_pooled" / "pooled_event_labels.parquet")
    windows = pd.read_parquet(window_path)
    for frame in (event_features, event_labels, windows):
        frame["channel_name"] = frame["channel_name"].astype(str).str.upper()
    event_labels = event_labels[KEYS + [
        "pooled_label", "pooled_probability", "pooled_global_label",
        "pooled_global_distance", "pooled_low_confidence", "global_label",
    ]].copy()
    events = event_features.merge(event_labels, on=KEYS, how="left", validate="one_to_one")
    if events["global_label"].isna().any():
        raise ValueError("some event-channel rows have no frozen PG label")
    windows = windows.copy()
    if windows.duplicated(KEYS + ["start_time_ms", "cache_local_index"]).any():
        raise ValueError("duplicate physical windows in pooled window table")
    windows["event_window_index"] = windows.sort_values(
        [*KEYS, "start_time_ms", "cache_local_index"], kind="stable"
    ).groupby(KEYS, sort=False).cumcount().astype(np.int32)
    windows["event_n_windows"] = windows.groupby(KEYS, sort=False)["event_id"].transform("size").astype(np.int32)
    # Materialize provenance at the event level from the complete window
    # table.  This is intentionally recomputed here so an older frozen table
    # whose event row points at the first physical window cannot contaminate
    # the clinical renderer.
    provenance_rows: list[dict[str, object]] = []
    windows["window_distance_to_event_median"] = np.nan
    for event_key, group in windows.groupby(KEYS, sort=True):
        positions = group.index.to_numpy()
        representative, distances = _representative(group)
        windows.loc[positions, "window_distance_to_event_median"] = distances
        key_values = dict(zip(KEYS, event_key, strict=True))
        provenance_rows.append(
            {
                **key_values,
                "event_n_windows": int(len(group)),
                "first_window_start_time_ms": float(group.start_time_ms.min()),
                "last_window_start_time_ms": float(group.start_time_ms.max()),
                "representative_cache_dir": str(representative.cache_dir),
                "representative_cache_local_index": int(representative.cache_local_index),
                "representative_start_time_ms": float(representative.start_time_ms),
                "representative_end_time_ms": float(representative.end_time_ms),
                "representative_window_index": int(representative.event_window_index),
                "representative_distance_robust": float(representative.window_distance_to_event_median),
                "representative_selection_rule": "nearest_event_feature_median_robust_MAD",
            }
        )
    provenance = pd.DataFrame(provenance_rows)
    protocol_events = pd.read_parquet(protocol_dir / "channel_events.parquet")
    protocol_key_frame = protocol_events[KEYS + ["split"]].astype(str).copy()
    protocol_key_frame["channel_name"] = protocol_key_frame["channel_name"].str.upper()
    protocol_key_frame["split"] = protocol_key_frame["split"].str.lower()
    protocol_keys = set(map(tuple, protocol_key_frame[KEYS].to_numpy().tolist()))
    protocol_split_keys = set(map(tuple, protocol_key_frame[KEYS + ["split"]].to_numpy().tolist()))
    event_keys = events[KEYS].astype(str).copy()
    event_keys["channel_name"] = event_keys["channel_name"].str.upper()
    events["trace_key_in_strict_protocol"] = [
        tuple(row) in protocol_keys for row in event_keys.to_numpy().tolist()
    ]
    event_split_keys = event_keys.copy()
    event_split_keys["split"] = events["split"].astype(str).str.lower().to_numpy()
    events["trace_key_and_split_in_strict_protocol"] = [
        tuple(row) in protocol_split_keys for row in event_split_keys.to_numpy().tolist()
    ]
    cache_roots: set[str] = set()
    for value in windows["cache_dir"].astype("string").unique().tolist():
        if value is pd.NA:
            continue
        root = str(value)
        if (Path(root) / "signals.npy").is_file():
            cache_roots.add(root)
    cache_key_frame = windows[KEYS].astype(str).copy()
    cache_key_frame["channel_name"] = cache_key_frame["channel_name"].str.upper()
    cache_key_frame["_cache_ok"] = windows["cache_dir"].astype(str).isin(cache_roots).to_numpy()
    cache_ok_by_key = cache_key_frame.groupby(KEYS, sort=False)["_cache_ok"].all().to_dict()
    events["waveform_cache_available"] = [
        bool(cache_ok_by_key.get(tuple(row), False)) for row in event_keys.to_numpy().tolist()
    ]
    qc = _qc_events(protocol_events, windows)
    events = events.merge(provenance, on=KEYS, how="left", validate="one_to_one")
    events = events.merge(qc, on=KEYS, how="left", validate="one_to_one", suffixes=("", "_qc"))
    if "split_qc" in events.columns:
        events = events.rename(columns={"split_qc": "protocol_split"})
    # The frozen PG36 tables and the current Task-1 protocol are not
    # interchangeable releases.  Any morphology key absent from the strict
    # protocol is retained but explicitly quarantined; a silent NaN status
    # would make it look eligible and would hide a real alignment problem.
    missing_protocol = events["qc_status"].isna()
    if bool(missing_protocol.any()):
        fallback = events.loc[missing_protocol, KEYS + ["event_duration_s", "event_n_windows"]].copy()
        fallback["split"] = events.loc[missing_protocol, "split"].astype(str).to_numpy()
        fallback["event_start_ms"] = events.loc[missing_protocol, "event_start_ms"].astype(float).to_numpy()
        fallback["event_end_ms"] = events.loc[missing_protocol, "event_end_ms"].astype(float).to_numpy()
        fallback["duration_s"] = fallback["event_duration_s"].astype(float)
        fallback["n_windows"] = fallback["event_n_windows"].astype(int)
        fallback["overlap_chain_size"] = np.nan
        fallback["qc_status"] = "review_required"
        fallback["qc_reasons"] = np.where(
            fallback["duration_s"].to_numpy(float) > 120.0,
            "protocol_key_missing;duration_gt_120s",
            "protocol_key_missing",
        )
        qc = pd.concat([qc, fallback[qc.columns]], ignore_index=True, sort=False)
        events.loc[missing_protocol, "qc_status"] = "review_required"
        events.loc[missing_protocol, "qc_reasons"] = fallback["qc_reasons"].to_numpy()
        events.loc[missing_protocol, "duration_s"] = fallback["duration_s"].to_numpy()
        events.loc[missing_protocol, "n_windows"] = fallback["n_windows"].to_numpy()
        events.loc[missing_protocol, "overlap_chain_size"] = np.nan
    events["qc_reasons"] = events["qc_reasons"].fillna("").astype(str)
    split_mismatch = (
        events["trace_key_in_strict_protocol"].astype(bool)
        & ~events["trace_key_and_split_in_strict_protocol"].astype(bool)
        & events["qc_status"].notna()
        & events.get("protocol_split", pd.Series(index=events.index, dtype="string")).astype(str).ne(events["split"].astype(str))
    )
    events.loc[split_mismatch, "qc_status"] = "review_required"
    events.loc[split_mismatch, "qc_reasons"] = events.loc[split_mismatch, "qc_reasons"].map(
        lambda value: ";".join([part for part in [value, "protocol_split_mismatch"] if part])
    )
    events.loc[
        ~events["waveform_cache_available"], "qc_reasons"
    ] = events.loc[~events["waveform_cache_available"], "qc_reasons"].map(
        lambda value: ";".join([part for part in [value, "waveform_cache_missing"] if part])
    )
    events.loc[~events["waveform_cache_available"], "qc_status"] = "review_required"
    windows = windows.merge(
        events[KEYS + ["global_label", "pooled_global_distance", "qc_status", "qc_reasons"]],
        on=KEYS, how="left", validate="many_to_one", suffixes=("", "_event"),
    )
    # Build the exported QC table from the *final* event state.  The protocol
    # join above can add split-mismatch, missing-key, and missing-cache reasons
    # after the initial duration/overlap audit; returning the earlier ``qc``
    # frame would therefore publish stale eligibility counts.
    qc_columns = [
        *KEYS,
        "split",
        "protocol_split",
        "event_start_ms",
        "event_end_ms",
        "event_duration_s",
        "event_n_windows",
        "overlap_chain_size",
        "trace_key_in_strict_protocol",
        "trace_key_and_split_in_strict_protocol",
        "waveform_cache_available",
        "qc_status",
        "qc_reasons",
    ]
    qc = events.reindex(columns=qc_columns).copy()
    qc = qc.rename(
        columns={
            "event_duration_s": "duration_s",
            "event_n_windows": "n_windows",
        }
    )
    return events, windows, qc


def _save_source_tables(events: pd.DataFrame, windows: pd.DataFrame, qc: pd.DataFrame, out: Path) -> None:
    source = out / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    events.to_parquet(source / "event_traceability.parquet", index=False)
    windows.to_parquet(source / "window_traceability.parquet", index=False)
    events.to_csv(source / "event_traceability.csv", index=False, encoding="utf-8-sig")
    qc.to_csv(source / "label_qc_audit.csv", index=False, encoding="utf-8-sig")


def _alignment_summary(events: pd.DataFrame, protocol_dir: Path, split: str = "all") -> pd.DataFrame:
    """Summarize PG36/protocol key agreement by mouse and channel."""

    protocol = pd.read_parquet(protocol_dir / "channel_events.parquet")
    if str(split).lower() != "all" and "split" in protocol:
        protocol = protocol.loc[protocol["split"].astype(str).str.lower().eq(str(split).lower())].copy()
    left = events[KEYS].astype(str).copy()
    right = protocol[KEYS].astype(str).copy()
    left["channel_name"] = left["channel_name"].str.upper()
    right["channel_name"] = right["channel_name"].str.upper()
    left["_morphology"] = True
    right["_protocol"] = True
    merged = left.merge(right, on=KEYS, how="outer")
    merged["_morphology"] = merged["_morphology"].astype("boolean").fillna(False).astype(bool)
    merged["_protocol"] = merged["_protocol"].astype("boolean").fillna(False).astype(bool)
    summary = (
        merged.groupby(["animal_id", "channel_name"], sort=True)
        .agg(
            morphology_event_channel_rows=("_morphology", "sum"),
            strict_protocol_event_channel_rows=("_protocol", "sum"),
            key_intersection=("_morphology", lambda value: int(((value) & merged.loc[value.index, "_protocol"]).sum())),
        )
        .reset_index()
    )
    summary["morphology_only_keys"] = summary.morphology_event_channel_rows - summary.key_intersection
    summary["protocol_only_keys"] = summary.strict_protocol_event_channel_rows - summary.key_intersection
    return summary


def _envelope(x: np.ndarray, y: np.ndarray, maximum_points: int = 2600) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    if len(x) <= maximum_points:
        return x, None, y
    bins = max(1, maximum_points // 2)
    edges = np.linspace(0, len(x), bins + 1, dtype=int)
    center = np.empty(bins, dtype=float)
    low = np.empty(bins, dtype=float)
    high = np.empty(bins, dtype=float)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        right = max(right, left + 1)
        block = y[left:right]
        center[index] = float(np.mean(x[left:right]))
        low[index] = float(np.nanmin(block))
        high[index] = float(np.nanmax(block))
    return center, low, high


def _draw_event_page(
    event: pd.Series,
    event_windows: pd.DataFrame,
    traces: TraceStore,
    *,
    margin_s: float,
    max_trace_s: float,
) -> plt.Figure:
    start = float(event.event_start_ms)
    end = float(event.event_end_ms)
    duration = max(0.001, (end - start) / 1000.0)
    # Very long annotations are shown as a declared overview plus three
    # clinically interpretable zooms; their complete physical window list is
    # still drawn in the strip and retained in source_data.
    if duration <= max_trace_s:
        intervals = [(start - margin_s * 1000.0, end + margin_s * 1000.0, "full")]
    else:
        zoom = min(20.0, max_trace_s)
        intervals = [
            (start, end, "overview"),
            (start - margin_s * 1000.0, start + zoom * 1000.0, "onset"),
            (max(start, (start + end) / 2.0 - zoom * 500.0), min(end, (start + end) / 2.0 + zoom * 500.0), "sustained"),
            (max(start, end - zoom * 1000.0), end + margin_s * 1000.0, "offset"),
        ]
    rep, distances = _representative(event_windows)
    fig, axes = plt.subplots(
        len(intervals), 2,
        figsize=(FIGURE_WIDTH_IN, max(55 / 25.4, len(intervals) * 38 / 25.4)),
        squeeze=False,
        gridspec_kw={"width_ratios": [3.25, 1.35], "hspace": 0.55, "wspace": 0.26},
    )
    label = str(event.global_label)
    for row_index, (crop_start, crop_end, phase) in enumerate(intervals):
        ax = axes[row_index, 0]
        if phase == "overview":
            # Do not load a 30-minute trace merely to draw an overview.  The
            # complete physical support is represented exactly by the raster
            # in the adjacent panel; the three zoom rows provide waveform
            # evidence at clinically readable scales.
            ax.text(0.5, 0.5, "long annotation overview\nraw waveform shown in onset / sustained / offset zooms", transform=ax.transAxes, ha="center", va="center", color=MUTED)
            ax.set_xlim(0.0, duration)
        else:
            try:
                time_ms, waveform = traces.load(
                    str(event.animal_id), str(event.session_id), str(event.channel_name), crop_start, crop_end
                )
                x = (time_ms - start) / 1000.0
                px, low, high = _envelope(x, waveform)
                if low is None:
                    ax.plot(px, high, color=INK, lw=0.45)
                else:
                    ax.fill_between(px, low, high, color="#4C5660", alpha=0.72, linewidth=0)
                ax.axvspan(
                    max(0.0, (crop_start - start) / 1000.0),
                    min(duration, (crop_end - start) / 1000.0),
                    color=MANUAL,
                    alpha=0.08 if duration > max_trace_s else 0.12,
                    linewidth=0,
                )
                rep_start = float(rep.start_time_ms)
                rep_end = float(rep.end_time_ms)
                if rep_end > crop_start and rep_start < crop_end:
                    ax.axvspan((rep_start - start) / 1000.0, (rep_end - start) / 1000.0, facecolor="none", edgecolor=ACCENT, lw=1.0)
                ax.set_xlim((crop_start - start) / 1000.0, (crop_end - start) / 1000.0)
            except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
                ax.text(0.5, 0.5, f"waveform unavailable\n{type(exc).__name__}", transform=ax.transAxes, ha="center", va="center", color=MUTED)
                ax.set_xlim((crop_start - start) / 1000.0, (crop_end - start) / 1000.0)
        ax.axvline(0.0, color=MANUAL, lw=0.8)
        ax.axvline(duration, color=MANUAL, lw=0.8)
        ax.set_ylabel("relative a.u.")
        if phase == "overview":
            ax.set_title("overview timeline | raw waveform is expanded in the three zoom rows below", loc="left", fontsize=7.0, fontweight="bold")
        else:
            ax.set_title(f"{phase} continuous filtered EEG | manual interval blue | representative window orange", loc="left", fontsize=7.0, fontweight="bold")
        ax.grid(axis="y", color=GRID, lw=0.35)

        strip = axes[row_index, 1]
        left = (crop_start - start) / 1000.0
        right = (crop_end - start) / 1000.0
        sub = event_windows.loc[
            event_windows.end_time_ms.astype(float).gt(crop_start)
            & event_windows.start_time_ms.astype(float).lt(crop_end)
        ].sort_values(["start_time_ms", "cache_local_index"], kind="stable")
        if phase == "overview":
            # A point per physical window keeps even very long annotations
            # lightweight while preserving the exact temporal order.
            centers = (
                (event_windows.start_time_ms.to_numpy(float) + event_windows.end_time_ms.to_numpy(float)) / 2000.0
                - start / 1000.0
            )
            labels = event_windows.get("global_label", pd.Series([label] * len(event_windows))).astype(str).to_numpy()
            strip.scatter(
                centers,
                np.full(len(centers), 0.5),
                s=8.0,
                marker="|",
                c=[pg_colour(value) for value in labels],
                linewidths=0.55,
                alpha=0.9,
                rasterized=True,
            )
            strip.set_xlim(0.0, duration)
            strip.set_ylim(0.0, 1.0)
            strip.set_yticks([])
            strip.set_xlabel("event time (s)")
            strip.set_title(f"all windows | n={len(event_windows):,} | complete map", loc="left", fontsize=6.8, fontweight="bold")
        elif sub.empty:
            strip.text(0.5, 0.5, "no windows", transform=strip.transAxes, ha="center", va="center", color=MUTED)
        else:
            for _, window in sub.iterrows():
                ws = (float(window.start_time_ms) - start) / 1000.0
                we = (float(window.end_time_ms) - start) / 1000.0
                pg = str(window.get("global_label", label))
                is_rep = int(window.event_window_index) == int(rep.event_window_index)
                strip.add_patch(Rectangle((ws, 0.38), max(0.02, we - ws), 0.26, facecolor=pg_colour(pg), edgecolor=ACCENT if is_rep else "white", linewidth=1.0 if is_rep else 0.2, alpha=0.88))
                if is_rep:
                    strip.text((ws + we) / 2.0, 0.76, f"rep #{int(window.event_window_index) + 1}", ha="center", va="bottom", fontsize=5.0, color=ACCENT)
                elif len(event_windows) <= 8:
                    strip.text((ws + we) / 2.0, 0.76, f"#{int(window.event_window_index) + 1}", ha="center", va="bottom", fontsize=5.0, color=INK)
            # Distance is a window QC distance, not a model probability.
            if "window_distance_to_event_median" in sub:
                dist = pd.to_numeric(sub["window_distance_to_event_median"], errors="coerce").to_numpy(float)
                centers = (sub.start_time_ms.to_numpy(float) + sub.end_time_ms.to_numpy(float)) / 2000.0 - start / 1000.0
                valid = np.isfinite(dist) & np.isfinite(centers)
                if valid.sum() >= 2 and float(np.nanmax(dist[valid]) - np.nanmin(dist[valid])) > 1e-6:
                    twin = strip.twinx()
                    twin.plot(centers[valid], dist[valid], color=DISTANCE, lw=0.7, marker=".", ms=2.0)
                    twin.set_ylabel("window-to-event median", color=DISTANCE, fontsize=5.4)
                    twin.tick_params(axis="y", colors=DISTANCE, labelsize=5.0, length=2)
                    twin.spines["top"].set_visible(False)
        if phase != "overview":
            strip.axvspan(0.0, duration, color=MANUAL, alpha=0.08, linewidth=0)
            strip.set_xlim(left, right)
            strip.set_ylim(0.0, 1.0)
            strip.set_yticks([])
            strip.set_xlabel("seconds from manual onset")
            strip.set_title(f"window support | n={len(event_windows):,} | event PG label", loc="left", fontsize=6.8, fontweight="bold")
        strip.grid(axis="x", color=GRID, lw=0.35)
    status = str(event.get("qc_status", "unknown"))
    status_colour = "#9C2C2C" if status != "eligible" else MUTED
    event_suffix = str(event.event_id).split(":")[-1]
    event_match = re.search(r"(?:clinical_)?event[_]?(\d+)$", event_suffix)
    short_event_id = f"event{event_match.group(1)}" if event_match else event_suffix[-24:]
    fig.suptitle(
        f"{event.animal_id} | {event.channel_name} independent | {label} | {short_event_id} | {event.split} | {status}",
        x=0.01, ha="left", y=0.995, fontsize=7.8, fontweight="bold", color=status_colour,
    )
    session_display = str(event.session_id)
    if len(session_display) > 34:
        session_display = session_display[:14] + "..." + session_display[-17:]
    info = (
        f"session={session_display}  duration={duration:.2f}s  windows={len(event_windows)}  "
        f"pooled assignment p={float(event.get('pooled_probability', np.nan)):.3f}  "
        f"global distance={float(event.get('pooled_global_distance', np.nan)):.3f}  "
        f"QC={event.get('qc_reasons', '') or 'pass'}"
    )
    fig.text(0.01, 0.008, info, fontsize=5.6, color=MUTED)
    return fig


def _render_pdfs(events: pd.DataFrame, windows: pd.DataFrame, protocol_dir: Path, out: Path, *, margin_s: float, max_trace_s: float, exclude_qc: bool) -> dict[str, int]:
    traces = TraceStore(protocol_dir)
    pdf_root = out / "pdf_by_mouse"
    pdf_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    selected = events.loc[events.qc_status.eq("eligible")] if exclude_qc else events
    preview_saved = False
    for animal, group in selected.groupby("animal_id", sort=True):
        path = pdf_root / f"{animal}_pg36_event_traceability.pdf"
        rendered = 0
        with PdfPages(path) as pdf:
            for _, event in group.sort_values(["session_id", "channel_name", "event_start_ms"], kind="stable").iterrows():
                key_mask = np.ones(len(windows), dtype=bool)
                for key in KEYS:
                    key_mask &= windows[key].astype(str).to_numpy() == str(event[key])
                event_windows = windows.loc[key_mask].copy()
                if event_windows.empty:
                    continue
                fig = _draw_event_page(event, event_windows, traces, margin_s=margin_s, max_trace_s=max_trace_s)
                pdf.savefig(fig, dpi=300, facecolor="white")
                if not preview_saved:
                    preview = out / "preview" / "first_event_traceability"
                    preview.parent.mkdir(parents=True, exist_ok=True)
                    # Keep a small, editable publication preview alongside the
                    # paginated clinical PDF.  The PDF remains the complete
                    # review product; this preview is only a visual QA handle.
                    fig.savefig(preview.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
                    fig.savefig(preview.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
                    fig.savefig(preview.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
                    fig.savefig(preview.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
                    preview_saved = True
                plt.close(fig)
                rendered += 1
        counts[str(animal)] = rendered
    return counts


def render(
    taxonomy_root: Path,
    protocol_dir: Path,
    output_dir: Path,
    *,
    split: str,
    max_events: int | None,
    margin_s: float,
    max_trace_s: float,
    exclude_qc: bool,
    index_only: bool = False,
) -> dict[str, object]:
    events, windows, qc = _load_inputs(taxonomy_root, protocol_dir)
    split = str(split).lower()
    if split != "all":
        events = events.loc[events.split.astype(str).eq(split)].copy()
        keep = set(map(tuple, events[KEYS].astype(str).to_numpy().tolist()))
        windows = windows.loc[windows[KEYS].astype(str).apply(tuple, axis=1).isin(keep)].copy()
        qc = qc.loc[qc[KEYS].astype(str).apply(tuple, axis=1).isin(keep)].copy()
    events = events.sort_values(["animal_id", "session_id", "channel_name", "event_start_ms"], kind="stable").reset_index(drop=True)
    if max_events is not None:
        events = events.iloc[: int(max_events)].copy()
        keep = set(map(tuple, events[KEYS].astype(str).to_numpy().tolist()))
        windows = windows.loc[windows[KEYS].astype(str).apply(tuple, axis=1).isin(keep)].copy()
        qc = qc.loc[qc[KEYS].astype(str).apply(tuple, axis=1).isin(keep)].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_source_tables(events, windows, qc, output_dir)
    alignment = _alignment_summary(events, protocol_dir, split=split)
    alignment.to_csv(output_dir / "source_data" / "protocol_alignment_summary.csv", index=False, encoding="utf-8-sig")
    pdf_counts = {} if index_only else _render_pdfs(
        events, windows, protocol_dir, output_dir,
        margin_s=margin_s, max_trace_s=max_trace_s, exclude_qc=exclude_qc,
    )
    strict_protocol = pd.read_parquet(protocol_dir / "channel_events.parquet")
    if split == "all":
        strict_protocol_selected = strict_protocol
    else:
        strict_protocol_selected = strict_protocol.loc[
            strict_protocol["split"].astype(str).str.lower().eq(split)
        ]
    manifest = {
        "schema_version": 1,
        "renderer": "render_pg36_event_traceability.py",
        "taxonomy_root": str(taxonomy_root.resolve()),
        "protocol_dir": str(protocol_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "observation_unit": "one clinical event x one valid channel",
        "window_unit": "one physical 4-s window from pooled_channel_window_labels",
        "channel_independence": True,
        "split_rendered": split,
        "event_channel_rows": int(len(events)),
        "window_rows": int(len(windows)),
        "qc_eligible_rows": int(events.qc_status.eq("eligible").sum()),
        "qc_review_required_rows": int(events.qc_status.ne("eligible").sum()),
        "strict_protocol_event_channel_rows": int(len(strict_protocol)),
        "strict_protocol_rows_in_selected_split": int(len(strict_protocol_selected)),
        "morphology_keys_absent_from_strict_protocol_or_split": int((~events.trace_key_and_split_in_strict_protocol.astype(bool)).sum()),
        "morphology_keys_present_but_split_mismatch": int((events.trace_key_in_strict_protocol.astype(bool) & ~events.trace_key_and_split_in_strict_protocol.astype(bool)).sum()),
        "waveform_cache_missing_rows": int((~events.waveform_cache_available.astype(bool)).sum()),
        "qc_rule": (
            "duration > 120 s, overlapping annotation chain, no physical windows, "
            "strict-protocol key missing, protocol split mismatch, or waveform cache "
            "missing => review_required; no silent deletion"
        ),
        "margin_seconds": float(margin_s),
        "max_trace_seconds_before_onset_sustained_offset_panels": float(max_trace_s),
        "representative_rule": "nearest event feature median using robust per-event MAD; exact cache directory/index recorded",
        "pooled_probability_semantics": "within-mouse GMM assignment probability; not Task-1 seizure probability",
        "pdf_pages_by_mouse": pdf_counts,
        "index_only": bool(index_only),
        "pdf_filter": "eligible_only" if exclude_qc else "all_rows_including_review_required",
        "source_tables": [
            "source_data/event_traceability.parquet",
            "source_data/window_traceability.parquet",
            "source_data/label_qc_audit.csv",
            "source_data/protocol_alignment_summary.csv",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README_zh.md").write_text(
        "\n".join(
            [
                "# PG36 临床波形溯源图集",
                "",
                "本目录把每个临床事件×独立有效通道回溯到连续滤波 EEG。事件标签来自冻结的 PG36 event-level taxonomy；事件内所有物理 4 秒滑窗都在右侧条带中保留，橙色边框是按 robust feature median 选出的真实代表窗。",
                "",
                "## 如何读图",
                "",
                "- 左侧：事件前后边界和连续滤波波形；蓝色是人工事件区间，橙色边框是代表窗。",
                "- 右侧：该事件的全部物理滑窗，而不是抽样 epoch；紫色曲线是窗口到该事件特征中位数的 QC 距离，不是 seizure probability。",
                "- 超过 120 秒、存在重叠链、没有物理窗、严格 protocol 键缺失、split 不一致或波形缓存缺失的事件显示为 `review_required`，仍保留在分母和源表中；这类事件在重新训练前应由临床复核后 quarantine，而不是由 renderer 静默删除。",
                "- 若 PG36 冻结表中的事件键或其 split 不在当前严格 Task-1 protocol 中，标记为 `protocol_key_missing` 或 `protocol_split_mismatch`；这说明两个发布版本未完全对齐，不能把它当作已通过质控的训练样本。",
                "- 当本目录由 `--exclude-qc` 生成时，PDF 只含通过门控的页面；同目录 source_data 仍保留所选 split 的完整事件和窗口索引。",
                "- PG 标签是探索性形态标签，不是临床发作亚型；CH7/CH8 始终是独立观察。当前 PG36 的 split 是形态学表自己的 split，必须先通过 alignment_summary 才能与 Task-1 protocol 合并分析。",
                "",
                "## 可追溯字段",
                "",
                "`event_traceability.parquet` 和 `window_traceability.parquet` 包含 animal/session/channel/event、split、事件边界、窗口序号、cache_dir、cache_local_index、PG 标签和 QC 状态。`protocol_alignment_summary.csv` 单独报告 PG36 冻结表与严格 Task-1 protocol 的键交集。任何一条图都能回到原始缓存窗口。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy-root", type=Path, default=TAX_ROOT)
    parser.add_argument("--protocol-dir", type=Path, default=PROTOCOL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="test")
    parser.add_argument("--max-events", type=int, default=None, help="smoke-test limit; omit for all events in the selected split")
    parser.add_argument("--margin-s", type=float, default=10.0)
    parser.add_argument("--max-trace-s", type=float, default=120.0)
    parser.add_argument("--exclude-qc", action="store_true", help="review-only PDF; source tables remain complete")
    parser.add_argument("--index-only", action="store_true", help="write complete source/QC tables without rendering PDFs")
    args = parser.parse_args()
    manifest = render(
        args.taxonomy_root.resolve(), args.protocol_dir.resolve(), args.output_dir.resolve(),
        split=args.split, max_events=args.max_events, margin_s=args.margin_s,
        max_trace_s=args.max_trace_s, exclude_qc=args.exclude_qc,
        index_only=args.index_only,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
