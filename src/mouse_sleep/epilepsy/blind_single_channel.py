from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


CHANNEL_NAMES = ("CH7", "CH8")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class BlindSingleChannelConfig:
    """Frozen data contract for the channel-blind clinical comparison."""

    animals: tuple[str, ...] = ("E50", "E76", "E77")
    sequence_length: int = 21
    minimum_positive_overlap_seconds: float = 0.5
    continuity_tolerance_ms: float = 250.0
    excluded_animal_channels: tuple[tuple[str, str], ...] = (("E77", "CH7"),)

    def validated(self) -> "BlindSingleChannelConfig":
        if not self.animals or len(set(self.animals)) != len(self.animals):
            raise ValueError("animals must be a non-empty unique tuple")
        if self.sequence_length < 1 or self.sequence_length % 2 != 1:
            raise ValueError("sequence_length must be a positive odd integer")
        if self.minimum_positive_overlap_seconds <= 0:
            raise ValueError("minimum_positive_overlap_seconds must be positive")
        if self.continuity_tolerance_ms < 0:
            raise ValueError("continuity_tolerance_ms cannot be negative")
        unknown = {
            str(channel).upper()
            for _, channel in self.excluded_animal_channels
        } - set(CHANNEL_NAMES)
        if unknown:
            raise ValueError(f"unknown excluded channels: {sorted(unknown)}")
        return self


def build_filtered_single_channel_index(
    filtered_metadata: str | Path | pd.DataFrame,
    event_components: str | Path | pd.DataFrame,
    config: BlindSingleChannelConfig | None = None,
) -> pd.DataFrame:
    """Expand the frozen filtered train/val/test view into channel-specific rows.

    The original event/background group assignment is retained.  A window that
    belongs to a CH8-only event is a negative example for CH7, rather than being
    copied to CH7 as a positive example.  Partial channel-event overlaps below
    the cache's positive-overlap threshold are excluded from supervised use.
    """

    config = (config or BlindSingleChannelConfig()).validated()
    metadata = _read_table(filtered_metadata)
    events = _validated_events(event_components, config)
    _require_columns(
        metadata,
        {
            "animal_id",
            "session_id",
            "start_time_ms",
            "end_time_ms",
            "split",
            "group_id",
            "label",
            "sampling_rate_hz",
            "window_seconds",
            "stride_seconds",
            "filtered_cache_dir",
            "filtered_local_index",
            "ch7_valid",
            "ch8_valid",
        },
        "filtered metadata",
    )
    selected = metadata.loc[
        metadata["animal_id"].astype(str).isin(config.animals)
    ].copy()
    if selected.empty:
        raise ValueError("filtered metadata contains none of the configured animals")
    unknown_splits = sorted(set(selected["split"].astype(str)) - set(SPLITS))
    if unknown_splits:
        raise ValueError(f"filtered metadata has unknown splits: {unknown_splits}")

    rows: list[pd.DataFrame] = []
    excluded = _excluded_pairs(config)
    for (animal, session), session_rows in selected.groupby(
        [selected["animal_id"].astype(str), selected["session_id"].astype(str)],
        sort=True,
    ):
        ordered = session_rows.sort_values(
            ["start_time_ms", "filtered_cache_dir", "filtered_local_index"],
            kind="stable",
        ).copy()
        starts = pd.to_numeric(ordered["start_time_ms"], errors="raise").to_numpy(float)
        ends = pd.to_numeric(ordered["end_time_ms"], errors="raise").to_numpy(float)
        for channel_index, channel_name in enumerate(CHANNEL_NAMES):
            if (str(animal), channel_name) in excluded:
                continue
            valid_column = f"{channel_name.lower()}_valid"
            valid = ordered[valid_column].astype(bool).to_numpy()
            if not valid.any():
                continue
            labels, overlaps, event_ids = _channel_event_labels(
                starts,
                ends,
                events.loc[
                    events["animal_id"].eq(str(animal))
                    & events["session_id"].eq(str(session))
                ],
                channel_name,
                config.minimum_positive_overlap_seconds * 1000.0,
            )
            keep = valid & labels.ne(-1).to_numpy()
            if not keep.any():
                continue
            expanded = ordered.loc[keep].copy()
            positions = np.flatnonzero(keep)
            expanded["source_global_label"] = expanded["label"].astype(np.int8)
            expanded["label"] = labels.iloc[positions].to_numpy(np.int8)
            expanded["channel_event_overlap_ms"] = overlaps[positions]
            expanded["channel_event_ids"] = event_ids[positions]
            expanded["channel_name"] = channel_name
            expanded["channel_index"] = channel_index
            expanded["event_group_id"] = expanded["group_id"].astype(str)
            expanded["signal_cache_dir"] = expanded["filtered_cache_dir"].astype(str)
            expanded["signal_local_index"] = expanded["filtered_local_index"].astype(np.int64)
            expanded["signal_state"] = "filtered_0.5_70Hz_notch_50Hz"
            expanded["source_kind"] = "filtered_selected_view"
            rows.append(expanded)

    if not rows:
        raise ValueError("no valid single-channel windows remain after channel labeling")
    result = _finalize_index(pd.concat(rows, ignore_index=True), config)
    audit_blind_single_channel_index(result, require_supervised_splits=True)
    return result


def build_full_single_channel_index(
    window_manifest: str | Path | pd.DataFrame,
    event_components: str | Path | pd.DataFrame,
    config: BlindSingleChannelConfig | None = None,
) -> pd.DataFrame:
    """Build an audit/inference index from every retained full-cache window.

    Channel-specific transition windows are retained with ``label == -1`` so
    inference remains as complete as the source cache permits.  The legacy
    cache omitted globally ambiguous transition windows; timestamp gaps caused
    by those omissions become explicit continuity boundaries and are never
    bridged by sequence lookup.
    """

    config = (config or BlindSingleChannelConfig()).validated()
    manifest = _read_table(window_manifest)
    events = _validated_events(event_components, config)
    _require_columns(
        manifest,
        {
            "animal_id",
            "session_id",
            "shard",
            "cache_dir",
            "sampling_rate_hz",
            "window_seconds",
            "stride_seconds",
        },
        "window manifest",
    )
    selected = manifest.loc[
        manifest["animal_id"].astype(str).isin(config.animals)
    ].copy()
    if selected.empty:
        raise ValueError("window manifest contains none of the configured animals")

    rows: list[pd.DataFrame] = []
    excluded = _excluded_pairs(config)
    for manifest_row in selected.sort_values(
        ["animal_id", "session_id", "shard"], kind="stable"
    ).to_dict("records"):
        animal = str(manifest_row["animal_id"])
        session = str(manifest_row["session_id"])
        root = Path(str(manifest_row["cache_dir"]))
        starts = np.asarray(
            np.load(root / "start_time_ms.npy", mmap_mode="r"), dtype=np.float64
        )
        source_labels = np.asarray(
            np.load(root / "labels.npy", mmap_mode="r"), dtype=np.int8
        )
        signals = np.load(root / "signals.npy", mmap_mode="r")
        if signals.ndim != 3 or signals.shape[1] != len(CHANNEL_NAMES):
            raise ValueError(f"invalid source signal shape in {root}: {signals.shape}")
        if len(starts) != len(source_labels) or len(starts) != len(signals):
            raise ValueError(f"signal/label/time length mismatch in {root}")
        window_seconds = float(manifest_row["window_seconds"])
        ends = starts + window_seconds * 1000.0
        session_events = events.loc[
            events["animal_id"].eq(animal) & events["session_id"].eq(session)
        ]
        for channel_index, channel_name in enumerate(CHANNEL_NAMES):
            if (animal, channel_name) in excluded:
                continue
            labels, overlaps, event_ids = _channel_event_labels(
                starts,
                ends,
                session_events,
                channel_name,
                config.minimum_positive_overlap_seconds * 1000.0,
            )
            rows.append(
                pd.DataFrame(
                    {
                        "animal_id": animal,
                        "session_id": session,
                        "shard": int(manifest_row["shard"]),
                        "start_time_ms": starts,
                        "end_time_ms": ends,
                        "label": labels.to_numpy(np.int8),
                        "source_global_label": source_labels,
                        "channel_event_overlap_ms": overlaps,
                        "channel_event_ids": event_ids,
                        "channel_name": channel_name,
                        "channel_index": channel_index,
                        "event_group_id": event_ids,
                        "split": "inference",
                        "signal_cache_dir": str(root.resolve()),
                        "signal_local_index": np.arange(len(starts), dtype=np.int64),
                        "signal_state": "raw_resampled_unfiltered",
                        "source_kind": "full_retained_window_cache",
                        "sampling_rate_hz": float(manifest_row["sampling_rate_hz"]),
                        "window_seconds": window_seconds,
                        "stride_seconds": float(manifest_row["stride_seconds"]),
                    }
                )
            )

    if not rows:
        raise ValueError("no full-cache single-channel windows were found")
    result = _finalize_index(pd.concat(rows, ignore_index=True), config)
    audit_blind_single_channel_index(result, require_supervised_splits=False)
    return result


def audit_blind_single_channel_index(
    frame: pd.DataFrame,
    *,
    require_supervised_splits: bool,
) -> dict[str, Any]:
    """Validate channel blindness, bad-channel exclusion, and split integrity."""

    _require_columns(
        frame,
        {
            "sample_id",
            "animal_id",
            "session_id",
            "channel_name",
            "channel_index",
            "start_time_ms",
            "end_time_ms",
            "label",
            "event_group_id",
            "channel_event_ids",
            "split",
            "signal_cache_dir",
            "signal_local_index",
            "continuity_id",
            "continuity_position",
        },
        "single-channel index",
    )
    if frame.empty:
        raise ValueError("single-channel index is empty")
    if frame["sample_id"].duplicated().any():
        raise ValueError("sample_id values must be unique")
    if set(frame["channel_name"].astype(str)) - set(CHANNEL_NAMES):
        raise ValueError("single-channel index contains an unknown channel")
    channel_mapping = dict(zip(CHANNEL_NAMES, range(len(CHANNEL_NAMES)), strict=True))
    observed_mapping = frame[["channel_name", "channel_index"]].drop_duplicates()
    for row in observed_mapping.itertuples(index=False):
        if channel_mapping[str(row.channel_name)] != int(row.channel_index):
            raise ValueError("channel_name/channel_index mapping is inconsistent")
    if (
        frame["animal_id"].astype(str).str.upper().eq("E77")
        & frame["channel_name"].astype(str).str.upper().eq("CH7")
    ).any():
        raise ValueError("E77 CH7 is invalid and must be absent")
    labels = set(pd.to_numeric(frame["label"], errors="raise").astype(int))
    if labels - {-1, 0, 1}:
        raise ValueError(f"invalid channel labels: {sorted(labels)}")
    if require_supervised_splits:
        unknown = sorted(set(frame["split"].astype(str)) - set(SPLITS))
        if unknown:
            raise ValueError(f"invalid supervised split values: {unknown}")
        group_splits = (
            frame.loc[frame["event_group_id"].astype(str).ne("")]
            .groupby("event_group_id")["split"]
            .nunique()
        )
        leaked_groups = group_splits[group_splits.ne(1)].index.astype(str).tolist()
        if leaked_groups:
            raise ValueError(
                "event complexes cross splits after channel expansion: "
                f"{leaked_groups[:5]}"
            )
        event_rows = frame.loc[frame["channel_event_ids"].astype(str).ne("")]
        exploded = event_rows.assign(
            _event_id=event_rows["channel_event_ids"].astype(str).str.split(";")
        ).explode("_event_id")
        event_splits = exploded.groupby("_event_id")["split"].nunique()
        leaked_events = event_splits[event_splits.ne(1)].index.astype(str).tolist()
        if leaked_events:
            raise ValueError(
                "annotated events cross splits after channel expansion: "
                f"{leaked_events[:5]}"
            )
    continuity_splits = frame.groupby("continuity_id")["split"].nunique()
    if continuity_splits.gt(1).any():
        raise ValueError("a continuity segment crosses split boundaries")
    continuity_identity = frame.groupby("continuity_id").agg(
        animals=("animal_id", "nunique"),
        sessions=("session_id", "nunique"),
        channels=("channel_name", "nunique"),
    )
    if continuity_identity.ne(1).any(axis=None):
        raise ValueError("a continuity segment crosses animal/session/channel identity")
    duplicate_source_channel = frame.duplicated(
        ["signal_cache_dir", "signal_local_index", "channel_name"]
    )
    if duplicate_source_channel.any():
        raise ValueError("a source window/channel occurs more than once")
    return {
        "schema_version": 1,
        "model_input_channels": 1,
        "identity_exposed_as_model_input": False,
        "animals": sorted(frame["animal_id"].astype(str).unique().tolist()),
        "channels": sorted(frame["channel_name"].astype(str).unique().tolist()),
        "windows": int(len(frame)),
        "positive_windows": int(frame["label"].eq(1).sum()),
        "transition_windows": int(frame["label"].eq(-1).sum()),
        "continuity_segments": int(frame["continuity_id"].nunique()),
    }


class BlindSingleChannelSequenceDataset(torch.utils.data.Dataset):
    """Read centered sequences without exposing animal or channel identity.

    ``padding_mask`` follows the PyTorch Transformer convention: ``True`` marks
    padded positions.  Audit metadata remain available in ``audit_frame`` and
    are deliberately absent from the returned model batch.
    """

    def __init__(
        self,
        index: str | Path | pd.DataFrame,
        *,
        split: str | None = None,
        sequence_length: int = 21,
        include_transition_centers: bool = False,
    ) -> None:
        if sequence_length < 1 or sequence_length % 2 != 1:
            raise ValueError("sequence_length must be a positive odd integer")
        frame = _read_table(index)
        if split is not None:
            frame = frame.loc[frame["split"].astype(str).eq(str(split))].copy()
        if frame.empty:
            raise ValueError("no sequence context remains after split filtering")
        self._context_frame = frame.sort_values(
            [
                "animal_id",
                "session_id",
                "channel_name",
                "continuity_id",
                "continuity_position",
            ],
            kind="stable",
        ).reset_index(drop=True)
        center_mask = np.ones(len(self._context_frame), dtype=bool)
        if not include_transition_centers:
            center_mask &= self._context_frame["label"].ne(-1).to_numpy()
        self._center_context_rows = np.flatnonzero(center_mask).astype(np.int64)
        if not len(self._center_context_rows):
            raise ValueError("no sequence centers remain after filtering")
        self.audit_frame = self._context_frame.iloc[
            self._center_context_rows
        ].reset_index(drop=True)
        self.sequence_length = int(sequence_length)
        self._half_width = sequence_length // 2
        self._arrays: dict[str, np.ndarray] = {}
        self._segment_rows: dict[str, np.ndarray] = {
            str(segment): group.index.to_numpy(np.int64)
            for segment, group in self._context_frame.groupby("continuity_id", sort=False)
        }
        self._row_position = np.empty(len(self._context_frame), dtype=np.int64)
        for indices in self._segment_rows.values():
            self._row_position[indices] = np.arange(len(indices), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.audit_frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        center_context_row = int(self._center_context_rows[index])
        center = self._context_frame.iloc[center_context_row]
        segment_rows = self._segment_rows[str(center["continuity_id"])]
        center_position = int(self._row_position[center_context_row])
        center_values = self._read_channel(center)
        signals = np.zeros(
            (self.sequence_length, 1, center_values.shape[0]), dtype=np.float32
        )
        padding_mask = np.ones(self.sequence_length, dtype=bool)
        for output_position, offset in enumerate(
            range(-self._half_width, self._half_width + 1)
        ):
            neighbor_position = center_position + offset
            if not 0 <= neighbor_position < len(segment_rows):
                continue
            neighbor = self._context_frame.iloc[int(segment_rows[neighbor_position])]
            values = self._read_channel(neighbor)
            if values.shape != center_values.shape:
                raise ValueError("one continuity segment contains unequal window shapes")
            signals[output_position, 0] = values
            padding_mask[output_position] = False
        return {
            "signals": torch.from_numpy(signals),
            "padding_mask": torch.from_numpy(padding_mask),
            "label": int(center["label"]),
        }

    def _read_channel(self, row: pd.Series) -> np.ndarray:
        root = str(row["signal_cache_dir"])
        if root not in self._arrays:
            self._arrays[root] = np.load(Path(root) / "signals.npy", mmap_mode="r")
        source = self._arrays[root]
        values = np.asarray(
            source[int(row["signal_local_index"]), int(row["channel_index"])],
            dtype=np.float32,
        ).copy()
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("single-channel model input must be a finite 1D waveform")
        return values

    def labels(self) -> np.ndarray:
        return self.audit_frame["label"].to_numpy(np.int64)


def _validated_events(
    event_components: str | Path | pd.DataFrame,
    config: BlindSingleChannelConfig,
) -> pd.DataFrame:
    events = _read_table(event_components)
    _require_columns(
        events,
        {
            "animal_id",
            "session_id",
            "event_id",
            "start_time_ms",
            "end_time_ms",
            "channels",
        },
        "event components",
    )
    events = events.loc[events["animal_id"].astype(str).isin(config.animals)].copy()
    events["animal_id"] = events["animal_id"].astype(str)
    events["session_id"] = events["session_id"].astype(str)
    events["event_id"] = events["event_id"].astype(str)
    events["start_time_ms"] = pd.to_numeric(events["start_time_ms"], errors="raise")
    events["end_time_ms"] = pd.to_numeric(events["end_time_ms"], errors="raise")
    if events.empty:
        raise ValueError("event components contain none of the configured animals")
    if (~np.isfinite(events[["start_time_ms", "end_time_ms"]].to_numpy(float))).any():
        raise ValueError("event component bounds must be finite")
    if events["end_time_ms"].le(events["start_time_ms"]).any():
        raise ValueError("event components must have positive duration")
    parsed_channels = events["channels"].map(_parse_channels)
    if parsed_channels.map(len).eq(0).any():
        raise ValueError("every event component must name CH7 and/or CH8")
    events["_channel_set"] = parsed_channels
    return events.sort_values(
        ["animal_id", "session_id", "start_time_ms", "end_time_ms"], kind="stable"
    ).reset_index(drop=True)


def _channel_event_labels(
    starts: np.ndarray,
    ends: np.ndarray,
    session_events: pd.DataFrame,
    channel_name: str,
    minimum_overlap_ms: float,
) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    if starts.shape != ends.shape or np.any(ends <= starts):
        raise ValueError("window bounds must have equal shape and positive duration")
    order = np.argsort(starts, kind="stable")
    sorted_starts = starts[order]
    sorted_ends = ends[order]
    overlaps = np.zeros(len(starts), dtype=np.float64)
    matched_ids: list[list[str]] = [[] for _ in range(len(starts))]
    channel_events = session_events.loc[
        session_events["_channel_set"].map(lambda values: channel_name in values)
    ].sort_values(["start_time_ms", "end_time_ms"], kind="stable")
    window_duration = ends - starts
    for event in channel_events.itertuples(index=False):
        event_start = float(event.start_time_ms)
        event_end = float(event.end_time_ms)
        left = int(np.searchsorted(sorted_ends, event_start, side="right"))
        right = int(np.searchsorted(sorted_starts, event_end, side="left"))
        if right <= left:
            continue
        event_overlap = np.maximum(
            0.0,
            np.minimum(sorted_ends[left:right], event_end)
            - np.maximum(sorted_starts[left:right], event_start),
        )
        overlaps[left:right] += event_overlap
        for local_index in np.flatnonzero(event_overlap > 0):
            matched_ids[left + int(local_index)].append(str(event.event_id))
    overlaps = np.minimum(overlaps, window_duration[order])
    labels = np.zeros(len(starts), dtype=np.int8)
    labels[(overlaps > 0) & (overlaps < minimum_overlap_ms)] = -1
    labels[overlaps >= minimum_overlap_ms] = 1
    sorted_event_ids = np.asarray([";".join(ids) for ids in matched_ids], dtype=object)
    restored_labels = np.empty_like(labels)
    restored_overlaps = np.empty_like(overlaps)
    restored_event_ids = np.empty_like(sorted_event_ids)
    restored_labels[order] = labels
    restored_overlaps[order] = overlaps
    restored_event_ids[order] = sorted_event_ids
    return pd.Series(restored_labels), restored_overlaps, restored_event_ids


def _finalize_index(
    frame: pd.DataFrame,
    config: BlindSingleChannelConfig,
) -> pd.DataFrame:
    result = frame.copy()
    for column in ("animal_id", "session_id", "channel_name", "split"):
        result[column] = result[column].astype(str)
    result["start_time_ms"] = pd.to_numeric(result["start_time_ms"], errors="raise")
    result["end_time_ms"] = pd.to_numeric(result["end_time_ms"], errors="raise")
    result = result.sort_values(
        [
            "animal_id",
            "session_id",
            "channel_name",
            "start_time_ms",
            "signal_cache_dir",
            "signal_local_index",
        ],
        kind="stable",
    ).reset_index(drop=True)
    duplicates = result.duplicated(
        ["animal_id", "session_id", "channel_name", "start_time_ms"]
    )
    if duplicates.any():
        raise ValueError("duplicate time points exist within an animal/session/channel")
    result["sample_id"] = [
        f"{animal}:{session}:{channel}:{float(start):.6f}"
        for animal, session, channel, start in result[
            ["animal_id", "session_id", "channel_name", "start_time_ms"]
        ].itertuples(index=False, name=None)
    ]
    result = _annotate_continuity(result, config.continuity_tolerance_ms)
    preferred = [
        "sample_id",
        "animal_id",
        "session_id",
        "channel_name",
        "channel_index",
        "start_time_ms",
        "end_time_ms",
        "label",
        "source_global_label",
        "channel_event_overlap_ms",
        "channel_event_ids",
        "event_group_id",
        "split",
        "continuity_id",
        "continuity_position",
        "signal_cache_dir",
        "signal_local_index",
        "signal_state",
        "source_kind",
        "sampling_rate_hz",
        "window_seconds",
        "stride_seconds",
    ]
    return result[preferred + [column for column in result if column not in preferred]]


def _annotate_continuity(frame: pd.DataFrame, tolerance_ms: float) -> pd.DataFrame:
    result = frame.copy()
    continuity_ids = np.empty(len(result), dtype=object)
    continuity_positions = np.empty(len(result), dtype=np.int64)
    key_columns = ["animal_id", "session_id", "channel_name"]
    for (animal, session, channel), group in result.groupby(key_columns, sort=False):
        indices = group.index.to_numpy(np.int64)
        starts = group["start_time_ms"].to_numpy(float)
        strides = group["stride_seconds"].to_numpy(float) * 1000.0
        splits = group["split"].astype(str).to_numpy()
        segment_number = 0
        position = 0
        for local_position, row_index in enumerate(indices):
            if local_position == 0:
                is_break = True
            else:
                observed = starts[local_position] - starts[local_position - 1]
                expected = strides[local_position - 1]
                is_break = (
                    observed <= 0
                    or abs(observed - expected) > tolerance_ms
                    or splits[local_position] != splits[local_position - 1]
                )
            if is_break:
                segment_number += 1
                position = 0
            continuity_ids[row_index] = (
                f"{animal}:{session}:{channel}:{splits[local_position]}:"
                f"segment{segment_number:06d}"
            )
            continuity_positions[row_index] = position
            position += 1
    result["continuity_id"] = continuity_ids
    result["continuity_position"] = continuity_positions
    return result


def _parse_channels(value: Any) -> frozenset[str]:
    channels = {
        token.strip().upper()
        for token in str(value).replace(",", ";").split(";")
        if token.strip()
    }
    unknown = channels - set(CHANNEL_NAMES)
    if unknown:
        raise ValueError(f"event component has unknown channels: {sorted(unknown)}")
    return frozenset(channels)


def _excluded_pairs(config: BlindSingleChannelConfig) -> set[tuple[str, str]]:
    return {
        (str(animal).upper(), str(channel).upper())
        for animal, channel in config.excluded_animal_channels
    }


def _read_table(value: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    path = Path(value)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


__all__ = [
    "BlindSingleChannelConfig",
    "BlindSingleChannelSequenceDataset",
    "audit_blind_single_channel_index",
    "build_filtered_single_channel_index",
    "build_full_single_channel_index",
]
