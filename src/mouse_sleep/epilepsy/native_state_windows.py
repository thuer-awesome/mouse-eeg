from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mouse_sleep.epilepsy.signal_processing import (
    EEGFilterConfig,
    filter_continuous_eeg,
    reconstruct_continuous_cache,
)


CHANNEL_NAMES = ("CH7", "CH8")


@dataclass(frozen=True)
class NativeStateWindowConfig:
    """Deterministic extraction settings for native seizure-state windows."""

    window_seconds: float = 1.0
    stride_seconds: float = 1.0
    random_seed: int = 20260811
    minimum_finite_fraction: float = 0.999
    minimum_robust_amplitude: float = 1e-6
    maximum_flat_fraction: float = 0.995
    hardware_rail_absolute: float = 399_999.0
    maximum_hardware_rail_fraction: float = 0.5
    minimum_mouse_channel_valid_fraction: float = 0.5
    overlap_rtol: float = 1e-5
    overlap_atol: float = 1e-4
    filter_config: EEGFilterConfig = field(default_factory=EEGFilterConfig)

    def validated(self) -> "NativeStateWindowConfig":
        if self.window_seconds <= 0 or self.stride_seconds <= 0:
            raise ValueError("Native window and stride durations must be positive")
        if not 0 < self.minimum_finite_fraction <= 1:
            raise ValueError("minimum_finite_fraction must lie in (0, 1]")
        if self.minimum_robust_amplitude < 0:
            raise ValueError("minimum_robust_amplitude cannot be negative")
        if not 0 <= self.maximum_flat_fraction <= 1:
            raise ValueError("maximum_flat_fraction must lie in [0, 1]")
        if self.hardware_rail_absolute <= 0:
            raise ValueError("hardware_rail_absolute must be positive")
        if not 0 <= self.maximum_hardware_rail_fraction <= 1:
            raise ValueError("maximum_hardware_rail_fraction must lie in [0, 1]")
        if not 0 <= self.minimum_mouse_channel_valid_fraction <= 1:
            raise ValueError(
                "minimum_mouse_channel_valid_fraction must lie in [0, 1]"
            )
        if self.overlap_rtol < 0 or self.overlap_atol < 0:
            raise ValueError("Overlap tolerances cannot be negative")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_seconds": self.window_seconds,
            "stride_seconds": self.stride_seconds,
            "random_seed": self.random_seed,
            "minimum_finite_fraction": self.minimum_finite_fraction,
            "minimum_robust_amplitude": self.minimum_robust_amplitude,
            "maximum_flat_fraction": self.maximum_flat_fraction,
            "hardware_rail_absolute": self.hardware_rail_absolute,
            "maximum_hardware_rail_fraction": self.maximum_hardware_rail_fraction,
            "minimum_mouse_channel_valid_fraction": (
                self.minimum_mouse_channel_valid_fraction
            ),
            "overlap_rtol": self.overlap_rtol,
            "overlap_atol": self.overlap_atol,
            "filter_config": self.filter_config.to_dict(),
        }


@dataclass
class NativeStateWindowResult:
    """Native filtered signals and their auditable temporal mapping."""

    signals: np.ndarray
    window_metadata: pd.DataFrame
    event_coverage: pd.DataFrame
    mouse_channel_validity: pd.DataFrame
    config: NativeStateWindowConfig
    output_dir: Path | None = None


@dataclass(frozen=True)
class _SegmentPlan:
    segment_index: int
    row_indices: np.ndarray
    row_offsets: np.ndarray
    length_samples: int


def build_native_state_windows(
    cache_manifest: str | Path | pd.DataFrame,
    event_components: str | Path | pd.DataFrame,
    output_dir: str | Path | None = None,
    config: NativeStateWindowConfig | None = None,
) -> NativeStateWindowResult:
    """Build fully contained, native-duration windows from labeled events.

    Cached four-second windows are first stitched back into continuous shard
    segments.  Each continuous segment is filtered once, and only then are
    native windows cut.  No temporal resampling is performed.  Cache rows that
    were omitted by the old transition-label policy are recovered from the
    still-overlapping neighboring windows when possible.
    """

    config = (config or NativeStateWindowConfig()).validated()
    manifest = _read_table(cache_manifest)
    events = _read_table(event_components)
    manifest = _validate_manifest(manifest)
    events = _validate_events(events)
    event_lookup = _event_lookup(events)

    metadata_rows: list[dict[str, Any]] = []
    channel_counts: dict[tuple[str, str], list[int]] = {}
    cache_plans: dict[str, list[_SegmentPlan]] = {}
    cache_specs: dict[str, dict[str, Any]] = {}

    for row in manifest.to_dict("records"):
        root = Path(str(row["cache_dir"]))
        signals, starts, spec = _load_cache(root, row)
        rate = spec["sampling_rate_hz"]
        config.filter_config.validated(rate)
        native_samples = _duration_samples(config.window_seconds, rate, "window")
        native_stride = _duration_samples(config.stride_seconds, rate, "stride")
        if native_samples > signals.shape[-1]:
            raise ValueError(
                f"Native window ({native_samples} samples) exceeds cached window "
                f"length ({signals.shape[-1]}) in {root}"
            )
        valid_flags = _cached_window_channel_validity(signals, config)
        animal_id = str(row["animal_id"])
        for channel_index, channel_name in enumerate(CHANNEL_NAMES):
            key = (animal_id, channel_name)
            counts = channel_counts.setdefault(key, [0, 0])
            counts[0] += int(valid_flags[:, channel_index].sum())
            counts[1] += int(len(valid_flags))

        plans = _plan_cache_segments(
            signals,
            starts,
            spec["cache_stride_samples"],
            spec["cache_stride_seconds"],
            config,
        )
        cache_key = str(root.resolve())
        cache_plans[cache_key] = plans
        cache_specs[cache_key] = spec
        session_events = event_lookup.get(str(row["session_id"]))
        if session_events is None:
            continue
        if set(session_events["animal_id"].astype(str)) != {animal_id}:
            raise ValueError(
                f"Animal mismatch between cache and events for session {row['session_id']}"
            )
        for plan in plans:
            metadata_rows.extend(
                _segment_metadata_rows(
                    row,
                    cache_key,
                    starts,
                    plan,
                    session_events,
                    rate,
                    native_samples,
                    native_stride,
                    config,
                )
            )

    channel_validity = _channel_validity_table(channel_counts, config)
    mouse_masks = _mouse_channel_masks(channel_validity)
    metadata = _finalize_window_metadata(metadata_rows, mouse_masks, config)
    coverage = _event_coverage(events, metadata, mouse_masks, config)

    target_dir = Path(output_dir).resolve() if output_dir is not None else None
    signal_shape = _signal_shape(metadata, manifest, config)
    signal_path = target_dir / "signals.npy" if target_dir is not None else None
    if target_dir is not None:
        target_dir.mkdir(parents=True, exist_ok=True)
    signal_store = _allocate_signal_store(signal_shape, signal_path)

    if len(metadata):
        for cache_key, rows in metadata.groupby("cache_dir", sort=False):
            root = Path(str(cache_key))
            manifest_row = manifest.loc[
                manifest["cache_dir_resolved"].eq(str(root.resolve()))
            ].iloc[0]
            signals, _, spec = _load_cache(root, manifest_row.to_dict())
            plans = cache_plans[str(root.resolve())]
            rate = spec["sampling_rate_hz"]
            mouse_mask = mouse_masks[str(manifest_row["animal_id"])]
            for plan in plans:
                selected = rows.loc[rows["continuity_segment"].eq(plan.segment_index)]
                if selected.empty:
                    continue
                continuous = _materialize_segment(signals, plan)
                filtered = filter_continuous_eeg(
                    continuous,
                    rate,
                    config.filter_config,
                )
                filtered[~mouse_mask] = 0.0
                offsets = selected["sample_start_in_segment"].to_numpy(np.int64)
                window_samples = int(selected["window_samples"].iloc[0])
                extracted = np.stack(
                    [filtered[:, start : start + window_samples] for start in offsets]
                ).astype(np.float32, copy=False)
                signal_indices = selected["signal_index"].to_numpy(np.int64)
                signal_store[signal_indices] = extracted
    if isinstance(signal_store, np.memmap):
        signal_store.flush()

    result = NativeStateWindowResult(
        signals=signal_store,
        window_metadata=metadata,
        event_coverage=coverage,
        mouse_channel_validity=channel_validity,
        config=config,
        output_dir=target_dir,
    )
    if target_dir is not None:
        _save_result_tables(result)
    return result


def event_balanced_indices(
    window_metadata: pd.DataFrame,
    maximum_windows_per_event: int,
    random_seed: int = 20260811,
) -> np.ndarray:
    """Select at most an equal cap per event, distributed across its duration."""

    if maximum_windows_per_event < 1:
        raise ValueError("maximum_windows_per_event must be at least one")
    required = {"event_id", "start_time_ms"}
    missing = required - set(window_metadata)
    if missing:
        raise ValueError(f"Window metadata is missing columns: {sorted(missing)}")
    selected: list[int] = []
    group_columns = (
        ["session_id", "event_id"]
        if "session_id" in window_metadata
        else ["event_id"]
    )
    for event_key, group in window_metadata.groupby(group_columns, sort=True):
        ordered = group.sort_values("start_time_ms", kind="stable")
        values = (
            ordered["signal_index"].to_numpy(np.int64)
            if "signal_index" in ordered
            else ordered.index.to_numpy(np.int64)
        )
        if len(values) <= maximum_windows_per_event:
            selected.extend(values.tolist())
            continue
        seed = random_seed + _stable_text_seed(str(event_key))
        rng = np.random.default_rng(seed)
        edges = np.linspace(0, len(values), maximum_windows_per_event + 1)
        for left, right in zip(edges[:-1], edges[1:]):
            low = int(math.floor(left))
            high = max(low + 1, int(math.ceil(right)))
            high = min(high, len(values))
            selected.append(int(values[rng.integers(low, high)]))
    return np.asarray(sorted(selected), dtype=np.int64)


def _read_table(value: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    path = Path(value)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _validate_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"animal_id", "session_id", "shard", "cache_dir"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Cache manifest is missing columns: {sorted(missing)}")
    result = frame.copy().drop_duplicates("cache_dir")
    result["cache_dir_resolved"] = [
        str(Path(str(value)).resolve()) for value in result["cache_dir"]
    ]
    return result.sort_values(
        ["animal_id", "session_id", "shard"], kind="stable"
    ).reset_index(drop=True)


def _validate_events(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "animal_id",
        "session_id",
        "event_id",
        "start_time_ms",
        "end_time_ms",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Event table is missing columns: {sorted(missing)}")
    result = frame.copy()
    result["start_time_ms"] = pd.to_numeric(result["start_time_ms"], errors="coerce")
    result["end_time_ms"] = pd.to_numeric(result["end_time_ms"], errors="coerce")
    usable = (
        result["start_time_ms"].notna()
        & result["end_time_ms"].notna()
        & result["end_time_ms"].ge(result["start_time_ms"])
    )
    if not usable.all():
        raise ValueError(f"Event table contains {int((~usable).sum())} invalid intervals")
    if result.duplicated(["session_id", "event_id"]).any():
        raise ValueError("Event IDs must be unique within each session")
    return result.sort_values(
        ["animal_id", "session_id", "start_time_ms", "end_time_ms"],
        kind="stable",
    ).reset_index(drop=True)


def _event_lookup(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lookup: dict[str, pd.DataFrame] = {}
    for session_id, group in events.groupby("session_id", sort=False):
        ordered = group.sort_values("start_time_ms", kind="stable").reset_index(drop=True)
        starts = ordered["start_time_ms"].to_numpy(float)
        ends = ordered["end_time_ms"].to_numpy(float)
        if len(ordered) > 1 and np.any(starts[1:] < ends[:-1]):
            raise ValueError(f"Overlapping event components remain in session {session_id}")
        lookup[str(session_id)] = ordered
    return lookup


def _load_cache(
    root: Path, row: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    signal_path = root / "signals.npy"
    starts_path = root / "start_time_ms.npy"
    if not signal_path.exists() or not starts_path.exists():
        raise FileNotFoundError(f"Incomplete cache directory: {root}")
    signals = np.load(signal_path, mmap_mode="r")
    starts = np.load(starts_path, mmap_mode="r")
    if signals.ndim != 3 or signals.shape[1] != len(CHANNEL_NAMES):
        raise ValueError(f"Expected [windows, 2, samples] in {root}, got {signals.shape}")
    if signals.shape[0] != len(starts) or not len(starts):
        raise ValueError(f"Signal/time row mismatch or empty cache in {root}")
    metadata: dict[str, Any] = {}
    metadata_path = root / "meta.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rate = _first_finite(row.get("sampling_rate_hz"), metadata.get("sampling_rate_hz"))
    cache_window_seconds = _first_finite(
        row.get("window_seconds"), metadata.get("window_seconds"), signals.shape[-1] / rate
    )
    cache_stride_seconds = _first_finite(
        row.get("stride_seconds"), metadata.get("stride_seconds"), 1.0
    )
    cache_stride_samples = _duration_samples(
        cache_stride_seconds, rate, "cache stride"
    )
    expected_window_samples = _duration_samples(
        cache_window_seconds, rate, "cache window"
    )
    if expected_window_samples != signals.shape[-1]:
        raise ValueError(
            f"Cache metadata describes {expected_window_samples} samples but {root} "
            f"contains {signals.shape[-1]}"
        )
    return signals, starts, {
        "sampling_rate_hz": rate,
        "cache_window_seconds": cache_window_seconds,
        "cache_stride_seconds": cache_stride_seconds,
        "cache_stride_samples": cache_stride_samples,
    }


def _first_finite(*values: Any) -> float:
    for value in values:
        try:
            converted = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(converted) and converted > 0:
            return converted
    raise ValueError("A positive finite cache sampling value is required")


def _duration_samples(seconds: float, rate: float, name: str) -> int:
    samples = int(round(seconds * rate))
    if samples < 1 or not np.isclose(samples / rate, seconds, atol=1e-8):
        raise ValueError(f"{name} duration {seconds} s is not an integer sample count at {rate} Hz")
    return samples


def _cached_window_channel_validity(
    signals: np.ndarray,
    config: NativeStateWindowConfig,
    batch_size: int = 256,
) -> np.ndarray:
    result = np.empty((len(signals), signals.shape[1]), dtype=bool)
    for offset in range(0, len(signals), batch_size):
        raw = np.asarray(signals[offset : offset + batch_size], dtype=np.float64)
        finite = np.isfinite(raw)
        raw_nan = np.where(finite, raw, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(raw_nan, axis=2, keepdims=True)
            q01 = np.nanquantile(raw_nan, 0.01, axis=2)
            q99 = np.nanquantile(raw_nan, 0.99, axis=2)
        median = np.nan_to_num(median, nan=0.0)
        filled = np.where(finite, raw, median)
        differences = np.diff(filled, axis=2)
        flat_fraction = np.mean(np.isclose(differences, 0.0, atol=1e-7), axis=2)
        rail_fraction = np.mean(
            np.abs(filled) >= config.hardware_rail_absolute, axis=2
        )
        result[offset : offset + len(raw)] = (
            (finite.mean(axis=2) >= config.minimum_finite_fraction)
            & ((q99 - q01) > config.minimum_robust_amplitude)
            & (flat_fraction < config.maximum_flat_fraction)
            & (rail_fraction < config.maximum_hardware_rail_fraction)
        )
    return result


def _plan_cache_segments(
    signals: np.ndarray,
    starts: np.ndarray,
    base_stride_samples: int,
    base_stride_seconds: float,
    config: NativeStateWindowConfig,
) -> list[_SegmentPlan]:
    window_samples = signals.shape[-1]
    rows: list[int] = [0]
    offsets: list[int] = [0]
    plans: list[_SegmentPlan] = []
    for index in range(len(signals) - 1):
        delta_ms = float(starts[index + 1] - starts[index])
        lag = _infer_overlap_lag(
            signals[index],
            signals[index + 1],
            delta_ms,
            base_stride_samples,
            base_stride_seconds,
            config,
        )
        if lag is None:
            plans.append(
                _make_segment_plan(len(plans), rows, offsets, window_samples)
            )
            rows = [index + 1]
            offsets = [0]
        else:
            rows.append(index + 1)
            offsets.append(offsets[-1] + lag)
    plans.append(_make_segment_plan(len(plans), rows, offsets, window_samples))
    return plans


def _make_segment_plan(
    segment_index: int,
    row_indices: list[int],
    row_offsets: list[int],
    window_samples: int,
) -> _SegmentPlan:
    return _SegmentPlan(
        segment_index=segment_index,
        row_indices=np.asarray(row_indices, dtype=np.int64),
        row_offsets=np.asarray(row_offsets, dtype=np.int64),
        length_samples=int(row_offsets[-1] + window_samples),
    )


def _infer_overlap_lag(
    previous: np.ndarray,
    following: np.ndarray,
    delta_ms: float,
    base_stride_samples: int,
    base_stride_seconds: float,
    config: NativeStateWindowConfig,
) -> int | None:
    window_samples = previous.shape[-1]
    maximum_steps = (window_samples - 1) // base_stride_samples
    if maximum_steps < 1:
        return None
    nominal_ms = base_stride_seconds * 1000.0
    estimated_steps = 1 if delta_ms < nominal_ms / 2.0 else int(round(delta_ms / nominal_ms))
    candidates = list(range(1, maximum_steps + 1))
    candidates.sort(key=lambda value: (abs(value - estimated_steps), value))
    for steps in candidates:
        lag = steps * base_stride_samples
        if _sampled_overlap_matches(previous, following, lag, config):
            return lag
    return None


def _sampled_overlap_matches(
    previous: np.ndarray,
    following: np.ndarray,
    lag: int,
    config: NativeStateWindowConfig,
) -> bool:
    overlap = previous.shape[-1] - lag
    if overlap <= 0:
        return False
    points = np.linspace(0, overlap - 1, min(64, overlap), dtype=np.int64)
    left = np.asarray(previous[:, lag + points])
    right = np.asarray(following[:, points])
    return bool(
        np.allclose(
            left,
            right,
            rtol=config.overlap_rtol,
            atol=config.overlap_atol,
            equal_nan=True,
        )
    )


def _segment_metadata_rows(
    manifest_row: dict[str, Any],
    cache_key: str,
    cache_starts: np.ndarray,
    plan: _SegmentPlan,
    events: pd.DataFrame,
    rate: float,
    window_samples: int,
    stride_samples: int,
    config: NativeStateWindowConfig,
) -> list[dict[str, Any]]:
    candidate_offsets = np.arange(
        0,
        plan.length_samples - window_samples + 1,
        stride_samples,
        dtype=np.int64,
    )
    anchor_offsets = plan.row_offsets
    source_positions = np.searchsorted(anchor_offsets, candidate_offsets, side="right") - 1
    source_positions = np.maximum(source_positions, 0)
    source_rows = plan.row_indices[source_positions]
    anchor_times = np.asarray(cache_starts[plan.row_indices], dtype=np.float64)
    candidate_times = anchor_times[source_positions] + (
        (candidate_offsets - anchor_offsets[source_positions]) * 1000.0 / rate
    )
    candidate_ends = candidate_times + config.window_seconds * 1000.0

    event_starts = events["start_time_ms"].to_numpy(float)
    event_ends = events["end_time_ms"].to_numpy(float)
    event_positions = np.searchsorted(event_starts, candidate_times, side="right") - 1
    valid_position = event_positions >= 0
    clipped_positions = np.maximum(event_positions, 0)
    fully_inside = (
        valid_position
        & (candidate_times >= event_starts[clipped_positions] - 1e-6)
        & (candidate_ends <= event_ends[clipped_positions] + 1e-6)
    )
    rows: list[dict[str, Any]] = []
    event_records = events.to_dict("records")
    for candidate_index in np.flatnonzero(fully_inside):
        event = event_records[int(event_positions[candidate_index])]
        rows.append(
            {
                "animal_id": str(manifest_row["animal_id"]),
                "session_id": str(manifest_row["session_id"]),
                "event_id": str(event["event_id"]),
                "shard": int(manifest_row["shard"]),
                "cache_dir": cache_key,
                "cache_local_index": int(source_rows[candidate_index]),
                "continuity_segment": int(plan.segment_index),
                "sample_start_in_segment": int(candidate_offsets[candidate_index]),
                "sample_stop_in_segment": int(
                    candidate_offsets[candidate_index] + window_samples
                ),
                "window_samples": int(window_samples),
                "sampling_rate_hz": float(rate),
                "start_time_ms": float(candidate_times[candidate_index]),
                "end_time_ms": float(candidate_ends[candidate_index]),
                "event_start_ms": float(event["start_time_ms"]),
                "event_end_ms": float(event["end_time_ms"]),
                "event_duration_s": float(
                    (event["end_time_ms"] - event["start_time_ms"]) / 1000.0
                ),
                "native_window_seconds": float(config.window_seconds),
                "native_stride_seconds": float(config.stride_seconds),
                "temporal_resampling": False,
            }
        )
    return rows


def _channel_validity_table(
    counts: dict[tuple[str, str], list[int]],
    config: NativeStateWindowConfig,
) -> pd.DataFrame:
    rows = []
    for (animal_id, channel_name), (valid, total) in sorted(counts.items()):
        fraction = valid / total if total else 0.0
        rows.append(
            {
                "animal_id": animal_id,
                "channel_name": channel_name,
                "valid_cache_windows": int(valid),
                "total_cache_windows": int(total),
                "valid_fraction": float(fraction),
                "is_valid": bool(
                    fraction >= config.minimum_mouse_channel_valid_fraction
                ),
            }
        )
    return pd.DataFrame(rows)


def _mouse_channel_masks(channel_validity: pd.DataFrame) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for animal_id, group in channel_validity.groupby("animal_id", sort=True):
        mapping = dict(zip(group["channel_name"], group["is_valid"], strict=True))
        mask = np.asarray([bool(mapping.get(name, False)) for name in CHANNEL_NAMES])
        if not mask.any():
            raise ValueError(f"No valid EEG channel remains for mouse {animal_id}")
        masks[str(animal_id)] = mask
    return masks


def _finalize_window_metadata(
    rows: list[dict[str, Any]],
    mouse_masks: dict[str, np.ndarray],
    config: NativeStateWindowConfig,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_window_metadata_columns())
    frame = pd.DataFrame(rows).sort_values(
        ["animal_id", "session_id", "start_time_ms", "shard", "sample_start_in_segment"],
        kind="stable",
    )
    frame = frame.drop_duplicates(
        ["animal_id", "session_id", "event_id", "start_time_ms", "end_time_ms"],
        keep="first",
    ).reset_index(drop=True)
    frame.insert(0, "signal_index", np.arange(len(frame), dtype=np.int64))
    frame.insert(1, "window_id", [f"native_{index:09d}" for index in range(len(frame))])
    for channel_index, channel_name in enumerate(CHANNEL_NAMES):
        column = channel_name.lower() + "_valid"
        frame[column] = [
            bool(mouse_masks[str(animal)][channel_index]) for animal in frame["animal_id"]
        ]
    frame["valid_channel_count"] = frame[["ch7_valid", "ch8_valid"]].sum(axis=1)
    frame["valid_channels"] = [
        ";".join(
            name
            for name, valid in zip(CHANNEL_NAMES, mouse_masks[str(animal)], strict=True)
            if valid
        )
        for animal in frame["animal_id"]
    ]
    frame["event_window_index"] = frame.groupby("event_id", sort=False).cumcount()
    frame["event_native_window_count"] = frame.groupby("event_id", sort=False)[
        "event_id"
    ].transform("size")
    return frame


def _window_metadata_columns() -> list[str]:
    return [
        "signal_index",
        "window_id",
        "animal_id",
        "session_id",
        "event_id",
        "shard",
        "cache_dir",
        "cache_local_index",
        "continuity_segment",
        "sample_start_in_segment",
        "sample_stop_in_segment",
        "window_samples",
        "sampling_rate_hz",
        "start_time_ms",
        "end_time_ms",
        "event_start_ms",
        "event_end_ms",
        "event_duration_s",
        "native_window_seconds",
        "native_stride_seconds",
        "temporal_resampling",
        "ch7_valid",
        "ch8_valid",
        "valid_channel_count",
        "valid_channels",
        "event_window_index",
        "event_native_window_count",
    ]


def _event_coverage(
    events: pd.DataFrame,
    metadata: pd.DataFrame,
    mouse_masks: dict[str, np.ndarray],
    config: NativeStateWindowConfig,
) -> pd.DataFrame:
    result = events.copy()
    result["duration_s"] = (
        result["end_time_ms"] - result["start_time_ms"]
    ) / 1000.0
    grouped = {
        (str(session_id), str(event_id)): group
        for (session_id, event_id), group in metadata.groupby(
            ["session_id", "event_id"], sort=False
        )
    }
    counts: list[int] = []
    covered: list[float] = []
    first: list[float] = []
    last: list[float] = []
    channels: list[str] = []
    reasons: list[str] = []
    for row in result.itertuples(index=False):
        key = (str(row.session_id), str(row.event_id))
        group = grouped.get(key)
        count = 0 if group is None else len(group)
        counts.append(int(count))
        covered.append(0.0 if group is None else _interval_union_seconds(group))
        first.append(np.nan if group is None else float(group["start_time_ms"].min()))
        last.append(np.nan if group is None else float(group["end_time_ms"].max()))
        mask = mouse_masks.get(str(row.animal_id), np.zeros(len(CHANNEL_NAMES), bool))
        channels.append(
            ";".join(
                name
                for name, valid in zip(CHANNEL_NAMES, mask, strict=True)
                if valid
            )
        )
        duration = (float(row.end_time_ms) - float(row.start_time_ms)) / 1000.0
        if duration < config.window_seconds:
            reasons.append("event_shorter_than_native_window")
        elif count == 0:
            reasons.append("no_fully_contained_native_window")
        else:
            reasons.append("")
    result["native_window_count"] = counts
    result["covered_duration_s"] = covered
    result["coverage_fraction"] = np.minimum(
        1.0,
        result["covered_duration_s"] / np.maximum(result["duration_s"], 1e-12),
    )
    result["first_native_window_start_ms"] = first
    result["last_native_window_end_ms"] = last
    result["valid_channels"] = channels
    result["short_unresolved"] = result["native_window_count"].eq(0)
    result["unresolved_reason"] = reasons
    result["state_window_status"] = np.where(
        result["short_unresolved"], "short_unresolved", "resolved"
    )
    return result


def _interval_union_seconds(group: pd.DataFrame) -> float:
    intervals = group[["start_time_ms", "end_time_ms"]].sort_values(
        "start_time_ms", kind="stable"
    ).to_numpy(float)
    if not len(intervals):
        return 0.0
    total = 0.0
    start, stop = intervals[0]
    for next_start, next_stop in intervals[1:]:
        if next_start <= stop:
            stop = max(stop, next_stop)
        else:
            total += stop - start
            start, stop = next_start, next_stop
    total += stop - start
    return float(total / 1000.0)


def _signal_shape(
    metadata: pd.DataFrame,
    manifest: pd.DataFrame,
    config: NativeStateWindowConfig,
) -> tuple[int, int, int]:
    if len(metadata):
        sample_counts = metadata["window_samples"].unique()
        if len(sample_counts) != 1:
            raise ValueError("All caches must yield the same native sample count")
        samples = int(sample_counts[0])
    else:
        rates = pd.to_numeric(manifest.get("sampling_rate_hz"), errors="coerce")
        finite = rates[np.isfinite(rates)]
        if not len(finite):
            samples = 0
        else:
            samples = _duration_samples(config.window_seconds, float(finite.iloc[0]), "window")
    return len(metadata), len(CHANNEL_NAMES), samples


def _allocate_signal_store(
    shape: tuple[int, int, int], path: Path | None
) -> np.ndarray:
    if path is None or shape[0] == 0:
        values = np.empty(shape, dtype=np.float32)
        if path is not None:
            np.save(path, values)
        return values
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def _materialize_segment(signals: np.ndarray, plan: _SegmentPlan) -> np.ndarray:
    rows = plan.row_indices
    offsets = plan.row_offsets
    if len(rows) > 1 and np.all(np.diff(offsets) == np.diff(offsets)[0]):
        stride = int(np.diff(offsets)[0])
        return reconstruct_continuous_cache(np.asarray(signals[rows]), stride)
    first = np.asarray(signals[rows[0]], dtype=np.float32)
    parts = [first]
    for position in range(1, len(rows)):
        lag = int(offsets[position] - offsets[position - 1])
        parts.append(np.asarray(signals[rows[position], :, -lag:], dtype=np.float32))
    return np.concatenate(parts, axis=1)


def _save_result_tables(result: NativeStateWindowResult) -> None:
    assert result.output_dir is not None
    result.window_metadata.to_parquet(
        result.output_dir / "window_metadata.parquet", index=False
    )
    result.event_coverage.to_csv(
        result.output_dir / "event_coverage.csv", index=False, encoding="utf-8-sig"
    )
    result.mouse_channel_validity.to_csv(
        result.output_dir / "mouse_channel_validity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (result.output_dir / "config.json").write_text(
        json.dumps(result.config.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _stable_text_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


__all__ = [
    "NativeStateWindowConfig",
    "NativeStateWindowResult",
    "build_native_state_windows",
    "event_balanced_indices",
]
