"""Build the channel-independent PG36 input tables from an audited protocol.

The historical morphology scripts consumed a project-specific table that was
created outside the public repository.  That made a clean checkout unable to
reproduce the PG36 workflow from raw EEG.  This module is the small bridge
between the public ``prepare``/``build-task1-protocol`` cache contract and the
PG36 scripts:

* one row is one event x valid channel x physical EEG window;
* the two channels are never concatenated at model input;
* event assignments are based on overlap with the clinical interval, while
  split labels come from the already audited event-level protocol;
* morphology features are computed from the filtered cache and baseline-z
  values use non-seizure windows from the same animal/channel when available.

The resulting tables deliberately use the file names expected by the pooled
taxonomy scripts.  No clinical subtype is inferred here; PG labels are frozen
only by the subsequent constrained taxonomy stage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from mouse_sleep.epilepsy.morphology import _feature_batch


FEATURE_SUFFIXES = (
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
)

KEYS = ["animal_id", "session_id", "event_id", "channel_name"]


def _read_table(value: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    path = Path(value)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _event_phase(midpoint: float, start: float, end: float) -> str:
    duration = max(1.0, end - start)
    position = float(np.clip((midpoint - start) / duration, 0.0, 1.0))
    if duration < 20_000.0:
        return "short_event"
    if position <= 0.10:
        return "onset_10s"
    if position >= 0.90:
        return "offset_10s"
    return "sustained"


def _valid_channel_rows(frame: pd.DataFrame, excluded: set[tuple[str, str]]) -> pd.DataFrame:
    required = {
        "animal_id",
        "session_id",
        "channel_name",
        "start_time_ms",
        "end_time_ms",
        "label",
        "channel_valid",
        "filtered_cache_dir",
        "filtered_local_index",
        "shard",
        "sampling_rate_hz",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"严格协议缺少形态学输入列: {missing}")
    result = frame.loc[frame["channel_valid"].astype(bool)].copy()
    keys = list(zip(result["animal_id"].astype(str).str.upper(), result["channel_name"].astype(str).str.upper()))
    result = result.loc[[key not in excluded for key in keys]].copy()
    result["channel_name"] = result["channel_name"].astype(str).str.upper()
    result = result.loc[result["channel_name"].isin(("CH7", "CH8"))]
    return result.reset_index(drop=True)


def _assign_event_windows(
    full_index: pd.DataFrame,
    events: pd.DataFrame,
    *,
    excluded: set[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign each overlapping physical window to its best clinical event."""

    windows = _valid_channel_rows(full_index, excluded)
    required_events = {
        "animal_id",
        "session_id",
        "event_id",
        "channel_name",
        "start_time_ms",
        "end_time_ms",
        "split",
    }
    missing = sorted(required_events - set(events.columns))
    if missing:
        raise ValueError(f"channel_events.parquet 缺少形态学事件列: {missing}")
    event_frame = events.copy()
    event_frame["channel_name"] = event_frame["channel_name"].astype(str).str.upper()
    event_frame = event_frame.loc[event_frame["channel_name"].isin(("CH7", "CH8"))].copy()
    event_excluded = np.asarray(
        [
            (str(a).upper(), str(c).upper()) in excluded
            for a, c in zip(event_frame["animal_id"], event_frame["channel_name"], strict=True)
        ],
        dtype=bool,
    )
    event_frame = event_frame.loc[~event_excluded].copy()
    event_frame["split"] = event_frame["split"].astype(str).str.lower()
    if event_frame.empty:
        raise ValueError("没有可用于 PG36 的有效通道事件")

    rows: list[dict[str, Any]] = []
    missing_events: list[dict[str, Any]] = []
    for key, group in windows.groupby(["animal_id", "session_id", "channel_name"], sort=True):
        animal, session, channel = map(str, key)
        group = group.sort_values("start_time_ms", kind="stable")
        local_events = event_frame.loc[
            event_frame["animal_id"].astype(str).eq(animal)
            & event_frame["session_id"].astype(str).eq(session)
            & event_frame["channel_name"].astype(str).eq(channel)
        ].sort_values("start_time_ms", kind="stable")
        if local_events.empty:
            continue
        starts = group["start_time_ms"].to_numpy(float)
        ends = group["end_time_ms"].to_numpy(float)
        for event in local_events.itertuples(index=False):
            event_start = float(event.start_time_ms)
            event_end = float(event.end_time_ms)
            overlaps = np.maximum(
                0.0,
                np.minimum(ends, event_end) - np.maximum(starts, event_start),
            )
            candidate_indices = np.flatnonzero(overlaps > 0.0)
            if candidate_indices.size == 0:
                missing_events.append(
                    {
                        "animal_id": animal,
                        "session_id": session,
                        "event_id": str(event.event_id),
                        "channel_name": channel,
                        "reason": "no_physical_window_overlap",
                    }
                )
                continue
            for local_index in candidate_indices.tolist():
                source = group.iloc[int(local_index)]
                # A window can touch two adjacent annotations.  Assign it to
                # the interval with the largest overlap; ties are deterministic
                # by event start/end order.
                competing = event_frame.loc[
                    event_frame["animal_id"].astype(str).eq(animal)
                    & event_frame["session_id"].astype(str).eq(session)
                    & event_frame["channel_name"].astype(str).eq(channel)
                ]
                overlap_all = np.maximum(
                    0.0,
                    np.minimum(float(source.end_time_ms), competing["end_time_ms"].to_numpy(float))
                    - np.maximum(float(source.start_time_ms), competing["start_time_ms"].to_numpy(float)),
                )
                winner = int(np.argmax(overlap_all))
                selected = competing.iloc[winner]
                if str(selected.event_id) != str(event.event_id):
                    continue
                midpoint = (float(source.start_time_ms) + float(source.end_time_ms)) / 2.0
                event_duration = max(1.0, event_end - event_start)
                overlap = float(overlaps[local_index])
                row = source.to_dict()
                row.update(
                    {
                        "event_id": str(event.event_id),
                        "event_start_ms": event_start,
                        "event_end_ms": event_end,
                        "event_duration_s": event_duration / 1000.0,
                        "event_overlap_s": overlap / 1000.0,
                        "event_overlap_fraction": overlap / max(1.0, float(source.end_time_ms) - float(source.start_time_ms)),
                        "event_phase": _event_phase(midpoint, event_start, event_end),
                        "split": str(event.split).lower(),
                    }
                )
                rows.append(row)
    if not rows:
        raise ValueError("没有临床事件与物理 EEG 窗口相交")
    assigned = pd.DataFrame(rows)
    # The same event must have one split across both independent channels.
    split_counts = assigned.groupby(["animal_id", "session_id", "event_id"])["split"].nunique()
    if bool(split_counts.gt(1).any()):
        raise ValueError("同一临床事件在形态学输入中出现多个 split")
    assigned = assigned.sort_values(
        ["animal_id", "session_id", "event_id", "channel_name", "start_time_ms"], kind="stable"
    ).reset_index(drop=True)
    missing_frame = pd.DataFrame(missing_events)
    return assigned, missing_frame


def _feature_rows(
    rows: pd.DataFrame,
    *,
    baseline_rows: pd.DataFrame,
    baseline_samples_per_stream: int,
) -> pd.DataFrame:
    """Compute features in cache batches and attach robust baseline z values."""

    if rows.empty:
        return rows.copy()
    work = rows.copy()
    # Baseline statistics are a train-only fit.  Filtering here as well as in
    # ``build_pg36_input_tables`` keeps direct callers from accidentally
    # passing validation/test background rows and leaking held-out scale
    # information into the taxonomy features.
    if "split" not in baseline_rows.columns:
        raise ValueError("baseline rows must contain an event-level split")
    baseline_rows = baseline_rows.loc[
        baseline_rows["split"].astype(str).str.lower().eq("train")
    ].copy()

    # Add a bounded background sample for within-stream baseline statistics.
    background_parts: list[pd.DataFrame] = []
    for key, group in baseline_rows.groupby(["animal_id", "session_id", "channel_name"], sort=True):
        group = group.sort_values("start_time_ms", kind="stable")
        if len(group) > baseline_samples_per_stream:
            positions = np.linspace(0, len(group) - 1, baseline_samples_per_stream, dtype=np.int64)
            group = group.iloc[np.unique(positions)]
        copy = group.copy()
        copy["_is_event_row"] = False
        background_parts.append(copy)
    event_copy = work.copy()
    event_copy["_is_event_row"] = True
    compute = pd.concat([event_copy, *background_parts], ignore_index=True, sort=False)
    feature_parts: list[pd.DataFrame] = []
    for cache_dir, group in compute.groupby("filtered_cache_dir", sort=False):
        root = Path(str(cache_dir))
        signal_path = root / "signals.npy"
        if not signal_path.is_file():
            raise FileNotFoundError(signal_path)
        signals = np.load(signal_path, mmap_mode="r")
        positions = group["filtered_local_index"].to_numpy(np.int64)
        rate = float(group["sampling_rate_hz"].iloc[0])
        # The raw adapter cache is already the canonical filtered signal.  We
        # deliberately pass it as both arguments to avoid a second filter pass.
        values = np.asarray(signals[positions], dtype=np.float32)
        feature = pd.DataFrame(_feature_batch(values, rate, filtered_signals=values))
        feature.index = group.index
        feature_parts.append(feature)
    feature_frame = pd.concat(feature_parts, axis=0).sort_index()
    merged = pd.concat([compute.reset_index(drop=True), feature_frame.reset_index(drop=True)], axis=1)

    # `_feature_batch` evaluates validity independently for CH7 and CH8.  A
    # channel can therefore be retained while the other channel is rejected
    # (for example the audited E77/CH7 case).  Require every feature used by
    # the taxonomy to be finite for the selected channel; this per-window gate
    # is intentionally stricter than the static channel-level flag.
    for channel in ("ch7", "ch8"):
        feature_columns = [f"{channel}_{suffix}" for suffix in FEATURE_SUFFIXES]
        missing_features = [column for column in feature_columns if column not in merged]
        if missing_features:
            raise ValueError(f"_feature_batch missing morphology columns: {missing_features}")
        finite_values = np.isfinite(
            merged[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        )
        finite_fraction = finite_values.mean(axis=1)
        merged[f"{channel}_feature_finite_fraction"] = finite_fraction
        merged[f"{channel}_feature_valid"] = (
            merged[f"{channel}_valid"].astype(bool).to_numpy()
            & (finite_fraction >= 1.0)
        )

    # Compute baseline z scores from non-seizure *training* windows in the same
    # animal and channel.  If no finite training background exists, use finite
    # training event rows as an explicitly recorded fallback.  We never use
    # validation/test rows for a baseline fit; if even training events are
    # unavailable, emit a neutral (0, 1) transform and mark the group.
    fallback_groups: set[tuple[str, str, str]] = set()
    no_reference_groups: set[tuple[str, str, str]] = set()
    for channel in ("ch7", "ch8"):
        for suffix in FEATURE_SUFFIXES:
            source = f"{channel}_{suffix}"
            target = f"baseline_z__{suffix}"
            values = pd.to_numeric(merged[source], errors="coerce")
            source_valid = merged[f"{channel}_feature_valid"].astype(bool).to_numpy()
            z = np.full(len(merged), np.nan, dtype=float)
            for (animal, _session, channel_name), group in merged.groupby(
                ["animal_id", "session_id", "channel_name"], sort=False
            ):
                # Each physical window appears once per independent channel.
                # Fit CH7 baselines from CH7 rows and CH8 baselines from CH8
                # rows; using the duplicate opposite-channel view would
                # double-count samples and inflate fallback diagnostics.
                if str(channel_name).upper() != channel.upper():
                    continue
                positions = group.index.to_numpy(np.int64)
                key = (str(animal), str(_session), str(channel_name))
                is_event = merged.loc[positions, "_is_event_row"].astype(bool).to_numpy()
                splits = merged.loc[positions, "split"].astype(str).str.lower().to_numpy()
                valid = source_valid[positions]
                reference_mask = (~is_event) & valid
                reference = values.iloc[positions][reference_mask].to_numpy(float)
                if not np.isfinite(reference).any():
                    # A stream may have no usable non-seizure training rows.
                    # Restrict the fallback to finite training event rows so
                    # validation/test distributions cannot affect centering.
                    fallback_mask = is_event & (splits == "train") & valid
                    reference = values.iloc[positions][fallback_mask].to_numpy(float)
                    if np.isfinite(reference).any():
                        fallback_groups.add(key)
                    else:
                        no_reference_groups.add(key)
                reference = reference[np.isfinite(reference)]
                if reference.size == 0:
                    center, scale = 0.0, 1.0
                else:
                    center = float(np.nanmedian(reference))
                    q10, q90 = np.nanquantile(reference, [0.10, 0.90])
                    scale = max(float(q90 - q10), 1e-6)
                z[positions] = (values.iloc[positions].fillna(center).to_numpy(float) - center) / scale
            # Keep the channel prefix until the two streams are unfolded.
            # This prevents CH7 and CH8 baseline statistics from being
            # accidentally copied into one another.
            merged[f"{channel}_{target}"] = z
    event_mask = merged["_is_event_row"].astype(bool).to_numpy()
    channel_name = merged["channel_name"].astype(str).str.upper().to_numpy()
    selected_valid = np.where(
        channel_name == "CH7",
        merged["ch7_feature_valid"].astype(bool).to_numpy(),
        merged["ch8_feature_valid"].astype(bool).to_numpy(),
    )
    selected_fraction = np.where(
        channel_name == "CH7",
        merged["ch7_feature_finite_fraction"].to_numpy(float),
        merged["ch8_feature_finite_fraction"].to_numpy(float),
    )
    merged["feature_valid"] = selected_valid
    merged["feature_finite_fraction"] = selected_fraction
    merged["feature_quality_reason"] = np.where(
        selected_valid,
        "valid",
        np.where(selected_fraction < 1.0, "nonfinite_feature", "channel_quality_gate"),
    )
    invalid_event = merged.loc[event_mask & ~selected_valid].copy()
    event_only = merged.loc[event_mask & selected_valid].copy()
    event_only.attrs["baseline_fallback_groups"] = int(len(fallback_groups))
    event_only.attrs["baseline_no_reference_groups"] = int(len(no_reference_groups))
    event_only.attrs["baseline_reference_scope"] = "train_background_only"
    event_only.attrs["invalid_event_windows"] = int(len(invalid_event))
    event_only.attrs["invalid_event_rows"] = invalid_event[
        [column for column in ["animal_id", "session_id", "event_id", "channel_name", "start_time_ms", "end_time_ms", "split", "feature_finite_fraction", "feature_quality_reason"] if column in invalid_event]
    ].to_dict("records")
    attributes = dict(event_only.attrs)
    result = event_only.reset_index(drop=True)
    result.attrs.update(attributes)
    return result


def _unfold_channel_features(feature_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return event-level and window-level channel-independent tables."""

    event_rows: list[pd.DataFrame] = []
    window_rows: list[pd.DataFrame] = []
    for channel_index, channel in enumerate(("CH7", "CH8")):
        prefix = channel.lower()
        selected = feature_rows.loc[feature_rows["channel_name"].astype(str).str.upper().eq(channel)].copy()
        if selected.empty:
            continue
        cols = [f"{prefix}_{suffix}" for suffix in FEATURE_SUFFIXES]
        missing = [column for column in cols if column not in selected]
        if missing:
            raise ValueError(f"特征表缺少 {channel} 列: {missing}")
        base_cols = [f"{prefix}_baseline_z__{suffix}" for suffix in FEATURE_SUFFIXES if f"{prefix}_baseline_z__{suffix}" in selected]
        metadata = [
            "animal_id", "session_id", "event_id", "channel_name", "split", "event_duration_s",
            "event_start_ms", "event_end_ms", "event_overlap_fraction",
            "filtered_cache_dir", "filtered_local_index", "shard", "start_time_ms",
            "end_time_ms", "sampling_rate_hz", "event_phase", "feature_valid",
            "feature_finite_fraction", "feature_quality_reason",
            "event_window_index", "event_n_windows", "window_midpoint_ms",
        ]
        selected = selected[[column for column in [*metadata, *cols, *base_cols] if column in selected]].copy()
        # Preserve the physical support of each clinical event explicitly.
        # The old event table kept the first row's cache index, which made a
        # later renderer mistake that row for a representative waveform.  A
        # stable ordinal lets clinical plots show every 4-s window in time and
        # makes the selected representative auditable.
        selected = selected.sort_values(
            ["animal_id", "session_id", "event_id", "channel_name", "start_time_ms", "filtered_local_index"],
            kind="stable",
        ).reset_index(drop=True)
        selected["event_window_index"] = selected.groupby(KEYS, sort=False).cumcount().astype(np.int32)
        selected["event_n_windows"] = selected.groupby(KEYS, sort=False)["event_id"].transform("size").astype(np.int32)
        selected["window_midpoint_ms"] = (
            pd.to_numeric(selected["start_time_ms"], errors="coerce")
            + pd.to_numeric(selected["end_time_ms"], errors="coerce")
        ) / 2.0
        renamed = {f"{prefix}_{suffix}": suffix for suffix in FEATURE_SUFFIXES}
        renamed.update({f"{prefix}_baseline_z__{suffix}": f"baseline_z__{suffix}" for suffix in FEATURE_SUFFIXES})
        selected = selected.rename(columns=renamed)
        selected["channel_name"] = channel
        selected["source_positive_channels"] = channel
        window_rows.append(selected)
        group_keys = KEYS
        numeric = [suffix for suffix in FEATURE_SUFFIXES if suffix in selected]
        numeric += [f"baseline_z__{suffix}" for suffix in FEATURE_SUFFIXES if f"baseline_z__{suffix}" in selected]
        distance_matrix = selected[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        distance_center = np.nanmedian(distance_matrix, axis=0)
        distance_scale = np.nanmedian(np.abs(distance_matrix - distance_center), axis=0)
        distance_scale = np.where(
            np.isfinite(distance_scale) & (distance_scale > 1e-8), distance_scale, 1.0
        )
        selected["window_distance_to_event_median"] = np.sqrt(
            np.nanmean(((distance_matrix - distance_center) / distance_scale) ** 2, axis=1)
        )
        # The list above is also the frame retained for downstream renderers;
        # append the derived distance before it is copied to ``window_rows``.
        window_rows[-1] = selected
        aggregates = selected.groupby(group_keys, sort=True)[numeric].median().reset_index()
        first = selected.groupby(group_keys, sort=True).first(numeric_only=False).reset_index()
        keep_meta = [
            "split", "event_duration_s", "event_start_ms", "event_end_ms",
            "event_overlap_fraction", "filtered_cache_dir", "filtered_local_index",
            "shard", "start_time_ms", "sampling_rate_hz", "source_positive_channels",
            "feature_valid", "feature_finite_fraction", "feature_quality_reason",
        ]
        result = aggregates.merge(first[group_keys + [column for column in keep_meta if column in first]], on=group_keys, how="left", validate="one_to_one")
        # Select the actual window closest to the event's feature median.  The
        # robust per-event scale prevents amplitude-heavy features from
        # dominating; ties are deterministic by physical window order.
        representative_rows: list[dict[str, Any]] = []
        for event_key, event_group in selected.groupby(group_keys, sort=True):
            feature_matrix = event_group[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            center = np.nanmedian(feature_matrix, axis=0)
            mad = np.nanmedian(np.abs(feature_matrix - center), axis=0)
            mad = np.where(np.isfinite(mad) & (mad > 1e-8), mad, 1.0)
            distance = np.sqrt(np.nanmean(((feature_matrix - center) / mad) ** 2, axis=1))
            distance = np.where(np.isfinite(distance), distance, np.inf)
            order = np.lexsort(
                (
                    event_group["filtered_local_index"].to_numpy(np.int64),
                    event_group["start_time_ms"].to_numpy(float),
                    distance,
                )
            )
            representative = event_group.iloc[int(order[0])]
            key_values = dict(zip(group_keys, event_key if isinstance(event_key, tuple) else (event_key,), strict=False))
            representative_rows.append(
                {
                    **key_values,
                    "representative_cache_dir": str(representative["filtered_cache_dir"]),
                    "representative_cache_local_index": int(representative["filtered_local_index"]),
                    "representative_start_time_ms": float(representative["start_time_ms"]),
                    "representative_end_time_ms": float(representative["end_time_ms"]),
                    "representative_window_index": int(representative["event_window_index"]),
                    "representative_distance_robust": float(distance[int(order[0])]),
                    "representative_selection_rule": "nearest_event_feature_median_robust_MAD",
                    "event_n_windows": int(len(event_group)),
                    "first_window_start_time_ms": float(event_group["start_time_ms"].iloc[0]),
                    "last_window_start_time_ms": float(event_group["start_time_ms"].iloc[-1]),
                }
            )
        representative_frame = pd.DataFrame(representative_rows)
        result = result.merge(representative_frame, on=group_keys, how="left", validate="one_to_one")
        result["channel_name"] = channel
        event_rows.append(result)
    if not event_rows:
        raise ValueError("没有可展开的 CH7/CH8 形态学事件")
    events = pd.concat(event_rows, ignore_index=True, sort=False)
    windows = pd.concat(window_rows, ignore_index=True, sort=False)
    # Keep the cache aliases on the event-level table as well; the paper
    # renderer uses one representative window from each event row.
    for frame in (events, windows):
        frame["cache_dir"] = frame["filtered_cache_dir"].astype(str)
        frame["cache_local_index"] = frame["filtered_local_index"].astype(np.int64)
    return events, windows


def build_pg36_input_tables(
    protocol_dir: str | Path,
    output_root: str | Path,
    *,
    excluded_animal_channels: Iterable[tuple[str, str]] = (("E07", "CH7"), ("E77", "CH7")),
    baseline_samples_per_stream: int = 256,
) -> dict[str, Any]:
    """Create the tables required by the pooled PG36 stages.

    Parameters are intentionally explicit so a rerun can be reproduced from
    the run manifest.  ``protocol_dir`` must be generated by the strict Task-1
    protocol builder (or an equivalent table with the same schema).
    """

    protocol = Path(protocol_dir).resolve()
    root = Path(output_root).resolve()
    table_root = root / "01_tables"
    freeze_root = root / "04_label_freeze"
    table_root.mkdir(parents=True, exist_ok=True)
    freeze_root.mkdir(parents=True, exist_ok=True)
    full_path = protocol / "full_index.parquet"
    event_path = protocol / "channel_events.parquet"
    if not full_path.is_file() or not event_path.is_file():
        raise FileNotFoundError("PG36 bridge requires protocol/full_index.parquet and channel_events.parquet")
    full = pd.read_parquet(full_path)
    events = pd.read_parquet(event_path)
    excluded = {(str(a).upper(), str(c).upper()) for a, c in excluded_animal_channels}
    assigned, missing = _assign_event_windows(full, events, excluded=excluded)
    if "split" not in full.columns:
        raise ValueError("strict protocol full_index.parquet must contain split")
    background_all = full.loc[
        full["label"].astype(int).eq(0)
        & full["channel_valid"].astype(bool)
    ].copy()
    background_all["channel_name"] = background_all["channel_name"].astype(str).str.upper()
    background_all = background_all.loc[background_all["channel_name"].isin(("CH7", "CH8"))]
    background_excluded_all = np.asarray(
        [
            (str(a).upper(), str(c).upper()) in excluded
            for a, c in zip(background_all["animal_id"], background_all["channel_name"], strict=True)
        ],
        dtype=bool,
    )
    background_all = background_all.loc[~background_excluded_all].copy()
    background = background_all.loc[
        background_all["split"].astype(str).str.lower().eq("train")
    ].copy()
    features = _feature_rows(
        assigned,
        baseline_rows=background,
        baseline_samples_per_stream=int(baseline_samples_per_stream),
    )
    feature_attrs = dict(features.attrs)
    event_features, window_features = _unfold_channel_features(features)
    # Canonical table consumed by fit_pooled_mouse_morphology.py and renderers.
    event_features = event_features.sort_values(KEYS, kind="stable").reset_index(drop=True)
    event_features.to_parquet(table_root / "channel_event_features.parquet", index=False)
    # Keep the historical source-table name as a transparent alias for users
    # who want to inspect the independent channel extraction stage.
    event_features.to_parquet(table_root / "combined_morphology_features_relative.parquet", index=False)

    window_features = window_features.sort_values(
        ["animal_id", "session_id", "channel_name", "start_time_ms"], kind="stable"
    ).reset_index(drop=True)
    # The public PG36 training/evaluation scripts use the historical
    # ``cache_dir``/``cache_local_index`` names.  Raw protocol tables expose
    # the more explicit ``filtered_*`` names; keep both aliases so a raw run
    # and a frozen-release rerun exercise the identical Dataset code path.
    window_features["cache_dir"] = window_features["filtered_cache_dir"].astype(str)
    window_features["cache_local_index"] = window_features["filtered_local_index"].astype(np.int64)
    window_features["feature_source"] = "raw_adapter_filtered_cache"
    window_features["eligible_formal"] = window_features["feature_valid"].astype(bool)
    if not bool(window_features["eligible_formal"].all()):
        raise AssertionError("invalid morphology windows survived the feature-quality gate")
    window_features.to_parquet(
        freeze_root / "frozen_channel_window_labels.parquet", index=False
    )
    missing.to_csv(table_root / "events_without_physical_windows.csv", index=False, encoding="utf-8-sig")
    invalid_rows = pd.DataFrame(feature_attrs.get("invalid_event_rows", []))
    invalid_rows.to_csv(
        table_root / "invalid_event_windows.csv", index=False, encoding="utf-8-sig"
    )
    assigned_keys = assigned[KEYS].drop_duplicates()
    retained_keys = event_features[KEYS].drop_duplicates()
    events_without_valid_features = assigned_keys.merge(
        retained_keys.assign(_retained=True), on=KEYS, how="left"
    )
    events_without_valid_features = events_without_valid_features.loc[
        events_without_valid_features["_retained"].isna(), KEYS
    ]
    events_without_valid_features.to_csv(
        table_root / "events_without_valid_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "schema_version": 1,
        "protocol_dir": str(protocol),
        "event_window_rows": int(len(window_features)),
        "event_channel_rows": int(len(event_features)),
        "animals": sorted(event_features["animal_id"].astype(str).unique()),
        "channels": sorted(event_features["channel_name"].astype(str).unique()),
        "excluded_animal_channels": [list(item) for item in sorted(excluded)],
        "baseline_samples_per_stream": int(baseline_samples_per_stream),
        "baseline_reference_scope": str(feature_attrs.get("baseline_reference_scope", "train_background_only")),
        "baseline_background_rows_all_splits": int(len(background_all)),
        "baseline_background_rows_train": int(len(background)),
        "baseline_fallback_groups": int(feature_attrs.get("baseline_fallback_groups", 0)),
        "baseline_no_reference_groups": int(feature_attrs.get("baseline_no_reference_groups", 0)),
        "unmatched_event_channel_rows": int(len(missing)),
        "invalid_event_windows_excluded": int(feature_attrs.get("invalid_event_windows", 0)),
        "event_channel_rows_without_valid_features": int(len(events_without_valid_features)),
        "feature_definition": list(FEATURE_SUFFIXES),
        "feature_quality_gate": "_feature_batch channel validity AND all taxonomy features finite",
        "amplitude_unit": "a.u.",
        "source_of_split": "strict Task-1 event-level protocol; no morphology-specific reshuffling",
        "channel_independence": True,
    }
    (table_root / "pg36_input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


__all__ = ["FEATURE_SUFFIXES", "build_pg36_input_tables"]
