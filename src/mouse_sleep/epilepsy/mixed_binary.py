from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from mouse_sleep.epilepsy.signal_processing import EEGFilterConfig, filter_continuous_eeg


SPLITS = ("train", "val", "test")
CHANNEL_NAMES = ("CH7", "CH8")


@dataclass(frozen=True)
class MixedBinaryConfig:
    """Frozen settings for the pooled-mouse binary seizure experiment."""

    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    event_group_gap_seconds: float = 8.0
    seizure_guard_seconds: float = 8.0
    background_block_seconds: float = 300.0
    maximum_windows_per_event_group: int = 64
    maximum_windows_per_background_group: int = 64
    random_seed: int = 20260814
    minimum_finite_fraction: float = 0.999
    minimum_robust_amplitude: float = 1e-6
    maximum_flat_fraction: float = 0.995
    hardware_rail_absolute: float = 399_999.0
    maximum_hardware_rail_fraction: float = 0.5
    normalization_clip: float = 20.0
    filter_config: EEGFilterConfig = field(default_factory=EEGFilterConfig)

    @property
    def fractions(self) -> np.ndarray:
        return np.asarray(
            [self.train_fraction, self.validation_fraction, self.test_fraction],
            dtype=float,
        )

    def validated(self) -> "MixedBinaryConfig":
        if not np.isclose(self.fractions.sum(), 1.0):
            raise ValueError("train/validation/test fractions must sum to one")
        if np.any(self.fractions <= 0):
            raise ValueError("all split fractions must be positive")
        if self.event_group_gap_seconds < 0 or self.seizure_guard_seconds < 0:
            raise ValueError("event grouping and guard durations cannot be negative")
        if self.background_block_seconds <= 0:
            raise ValueError("background blocks must have positive duration")
        if min(
            self.maximum_windows_per_event_group,
            self.maximum_windows_per_background_group,
        ) < 1:
            raise ValueError("per-group window caps must be positive")
        if not 0 < self.minimum_finite_fraction <= 1:
            raise ValueError("minimum_finite_fraction must lie in (0, 1]")
        if not 0 <= self.maximum_flat_fraction <= 1:
            raise ValueError("maximum_flat_fraction must lie in [0, 1]")
        if not 0 <= self.maximum_hardware_rail_fraction <= 1:
            raise ValueError("maximum_hardware_rail_fraction must lie in [0, 1]")
        if self.normalization_clip <= 0:
            raise ValueError("normalization_clip must be positive")
        return self

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["filter_config"] = self.filter_config.to_dict()
        return values


def build_mixed_binary_index(
    cache_manifest: str | Path | pd.DataFrame,
    event_components: str | Path | pd.DataFrame,
    output_dir: str | Path,
    config: MixedBinaryConfig | None = None,
) -> Path:
    """Pool all mice, then split independent seizure/background groups.

    Seizure subtypes are intentionally absent. Positive windows are grouped by
    event complexes; nearby events are kept together so overlapping four-second
    windows cannot cross splits. Negative windows are guarded from seizures and
    grouped into fixed continuous-time background blocks.
    """

    config = (config or MixedBinaryConfig()).validated()
    manifest = _read_table(cache_manifest)
    events = _read_table(event_components)
    _require_columns(
        manifest,
        {"animal_id", "session_id", "cache_dir", "window_seconds", "stride_seconds"},
        "cache manifest",
    )
    _require_columns(
        events,
        {"animal_id", "session_id", "event_id", "start_time_ms", "end_time_ms"},
        "event components",
    )
    events = events.copy()
    events["start_time_ms"] = pd.to_numeric(events["start_time_ms"], errors="raise")
    events["end_time_ms"] = pd.to_numeric(events["end_time_ms"], errors="raise")
    complexes = _build_event_complexes(events, config.event_group_gap_seconds * 1000.0)
    complex_lookup = {
        str(session): group.sort_values("start_time_ms", kind="stable").reset_index(drop=True)
        for session, group in complexes.groupby("session_id", sort=False)
    }

    rows: list[pd.DataFrame] = []
    for manifest_row in manifest.sort_values(
        ["animal_id", "session_id", "shard"], kind="stable"
    ).to_dict("records"):
        root = Path(str(manifest_row["cache_dir"]))
        labels = np.asarray(np.load(root / "labels.npy", mmap_mode="r"), dtype=np.int8)
        starts = np.asarray(
            np.load(root / "start_time_ms.npy", mmap_mode="r"), dtype=np.float64
        )
        if len(labels) != len(starts):
            raise ValueError(f"signal label/time mismatch in {root}")
        session_id = str(manifest_row["session_id"])
        session_complexes = complex_lookup.get(session_id)
        if session_complexes is None or session_complexes.empty:
            raise ValueError(f"no seizure events found for session {session_id}")
        window_ms = float(manifest_row["window_seconds"]) * 1000.0
        ends = starts + window_ms
        event_positions, event_overlap = _interval_lookup(
            starts,
            ends,
            session_complexes["start_time_ms"].to_numpy(float),
            session_complexes["end_time_ms"].to_numpy(float),
        )
        guard_positions, guard_overlap = _interval_lookup(
            starts,
            ends,
            session_complexes["start_time_ms"].to_numpy(float)
            - config.seizure_guard_seconds * 1000.0,
            session_complexes["end_time_ms"].to_numpy(float)
            + config.seizure_guard_seconds * 1000.0,
        )
        del guard_positions

        positive = labels == 1
        if np.any(positive & ~event_overlap):
            count = int(np.sum(positive & ~event_overlap))
            raise ValueError(
                f"{count} positive cache windows do not overlap an event complex in {root}"
            )
        block_ms = config.background_block_seconds * 1000.0
        block_number = np.floor(starts / block_ms).astype(np.int64)
        fully_inside_block = ends <= (block_number + 1) * block_ms + 1e-6
        negative_keep = (labels == 0) & ~guard_overlap & fully_inside_block
        keep = positive | negative_keep
        if not keep.any():
            continue

        local_indices = np.flatnonzero(keep)
        kept_labels = labels[local_indices]
        group_ids: list[str] = []
        event_ids: list[str] = []
        for source_index, label in zip(local_indices, kept_labels, strict=True):
            if label == 1:
                event_row = session_complexes.iloc[int(event_positions[source_index])]
                group_ids.append(str(event_row["group_id"]))
                event_ids.append(str(event_row["source_event_ids"]))
            else:
                block = int(block_number[source_index])
                group_ids.append(f"{session_id}:background:{block:05d}")
                event_ids.append("")
        rows.append(
            pd.DataFrame(
                {
                    "animal_id": str(manifest_row["animal_id"]),
                    "session_id": session_id,
                    "shard": int(manifest_row.get("shard", -1)),
                    "cache_dir": str(root.resolve()),
                    "cache_local_index": local_indices.astype(np.int64),
                    "start_time_ms": starts[local_indices],
                    "end_time_ms": ends[local_indices],
                    "label": kept_labels.astype(np.int8),
                    "group_id": group_ids,
                    "source_event_ids": event_ids,
                    "sampling_rate_hz": float(manifest_row["sampling_rate_hz"]),
                    "window_seconds": float(manifest_row["window_seconds"]),
                    "stride_seconds": float(manifest_row["stride_seconds"]),
                }
            )
        )
    if not rows:
        raise ValueError("no eligible binary windows were found")
    index = pd.concat(rows, ignore_index=True)
    index = _cap_group_windows(index, config)
    group_splits = _assign_group_splits(index, config)
    index["split"] = index["group_id"].map(group_splits)
    if index["split"].isna().any():
        raise AssertionError("some groups did not receive a split")
    audit = audit_mixed_binary_split(index)

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "window_index.parquet"
    index.to_parquet(index_path, index=False)
    group_summary = _group_summary(index)
    group_summary.to_csv(output / "group_summary.csv", index=False, encoding="utf-8-sig")
    split_summary = _split_summary(index)
    split_summary.to_csv(output / "split_summary.csv", index=False, encoding="utf-8-sig")
    (output / "config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    audit["window_index_sha256"] = _sha256(index_path)
    audit["config_sha256"] = _sha256(output / "config.json")
    (output / "split_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return index_path


def audit_mixed_binary_split(frame: pd.DataFrame) -> dict[str, Any]:
    _require_columns(
        frame,
        {
            "animal_id",
            "session_id",
            "start_time_ms",
            "end_time_ms",
            "label",
            "group_id",
            "split",
        },
        "mixed split",
    )
    unknown = sorted(set(frame["split"].astype(str)) - set(SPLITS))
    if unknown:
        raise ValueError(f"unknown split values: {unknown}")
    starts = pd.to_numeric(frame["start_time_ms"], errors="coerce").to_numpy(float)
    ends = pd.to_numeric(frame["end_time_ms"], errors="coerce").to_numpy(float)
    if not np.isfinite(starts).all() or not np.isfinite(ends).all():
        raise ValueError("window time bounds must be finite")
    if np.any(ends <= starts):
        raise ValueError("every window must have positive duration")
    group_integrity = frame.groupby("group_id").agg(
        animals=("animal_id", "nunique"),
        sessions=("session_id", "nunique"),
        labels=("label", "nunique"),
    )
    invalid_groups = group_integrity.loc[
        group_integrity[["animals", "sessions", "labels"]].ne(1).any(axis=1)
    ]
    if not invalid_groups.empty:
        raise ValueError(
            "each split group must contain one animal, session, and class: "
            f"{invalid_groups.index.astype(str).tolist()[:5]}"
        )
    group_counts = frame.groupby("group_id")["split"].nunique()
    leaked_groups = group_counts[group_counts.ne(1)].index.astype(str).tolist()
    if leaked_groups:
        raise ValueError(f"group leakage detected: {leaked_groups[:5]}")
    _audit_interval_overlap(frame)
    coverage = (
        frame.groupby(["animal_id", "label", "split"]).size().rename("windows").reset_index()
    )
    missing = []
    for animal in sorted(frame["animal_id"].astype(str).unique()):
        for label in (0, 1):
            observed = set(
                coverage.loc[
                    coverage["animal_id"].astype(str).eq(animal)
                    & coverage["label"].eq(label),
                    "split",
                ].astype(str)
            )
            if observed != set(SPLITS):
                missing.append({"animal_id": animal, "label": label, "splits": sorted(observed)})
    if missing:
        raise ValueError(f"each mouse/class must occur in every split: {missing}")
    split_counts = {
        split: int(frame["split"].eq(split).sum()) for split in SPLITS
    }
    group_split_counts = {
        split: int(frame.loc[frame["split"].eq(split), "group_id"].nunique())
        for split in SPLITS
    }
    total_windows = max(int(len(frame)), 1)
    return {
        "schema_version": 1,
        "protocol": "pooled_mice_grouped_event_background_70_15_15",
        "subtype_used_for_split": False,
        "event_groups_disjoint": True,
        "overlapping_raw_windows_cross_splits": 0,
        "animals": sorted(frame["animal_id"].astype(str).unique().tolist()),
        "sessions": int(frame["session_id"].nunique()),
        "groups": int(frame["group_id"].nunique()),
        "windows": int(len(frame)),
        "split_counts": split_counts,
        "split_fractions": {
            split: float(split_counts[split] / total_windows) for split in SPLITS
        },
        "group_split_counts": group_split_counts,
        "positive_counts": {
            split: int(
                (frame["split"].eq(split) & frame["label"].eq(1)).sum()
            )
            for split in SPLITS
        },
    }


def materialize_filtered_binary_view(
    window_index: str | Path | pd.DataFrame,
    output_dir: str | Path,
    config: MixedBinaryConfig | None = None,
) -> Path:
    """Filter reconstructed continuous segments and materialize selected windows."""

    config = (config or MixedBinaryConfig()).validated()
    frame = _read_table(window_index).copy()
    _require_columns(
        frame,
        {
            "cache_dir",
            "cache_local_index",
            "animal_id",
            "session_id",
            "split",
            "label",
        },
        "binary window index",
    )
    output = Path(output_dir).resolve()
    shard_root = output / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    metadata_frames: list[pd.DataFrame] = []

    grouped = frame.groupby("cache_dir", sort=False)
    for shard_number, (cache_dir, selected) in enumerate(grouped, start=1):
        root = Path(str(cache_dir))
        signals = np.load(root / "signals.npy", mmap_mode="r")
        starts = np.asarray(
            np.load(root / "start_time_ms.npy", mmap_mode="r"), dtype=np.float64
        )
        if signals.ndim != 3 or signals.shape[1] != 2 or len(signals) != len(starts):
            raise ValueError(f"invalid source cache shape in {root}: {signals.shape}")
        first = selected.iloc[0]
        rate = float(first["sampling_rate_hz"])
        stride_seconds = float(first["stride_seconds"])
        stride_samples = int(round(rate * stride_seconds))
        window_samples = signals.shape[-1]
        selected_by_index = {
            int(source_index): position
            for position, source_index in enumerate(
                selected["cache_local_index"].to_numpy(np.int64)
            )
        }
        extracted: dict[int, tuple[np.ndarray, np.ndarray, dict[str, float]]] = {}
        for segment in _continuity_segments(starts, stride_seconds * 1000.0):
            wanted = [index for index in segment if int(index) in selected_by_index]
            if not wanted:
                continue
            continuous = _reconstruct_segment(signals, segment, stride_samples)
            filtered = filter_continuous_eeg(continuous, rate, config.filter_config)
            segment_start = int(segment[0])
            raw_windows = np.asarray(signals[wanted], dtype=np.float32)
            validity, quality = assess_window_channels(raw_windows, config)
            animal_id = str(first["animal_id"])
            if animal_id.upper() == "E77":
                validity[:, 0] = False
            for row_position, source_index in enumerate(wanted):
                offset = (int(source_index) - segment_start) * stride_samples
                values = np.asarray(
                    filtered[:, offset : offset + window_samples], dtype=np.float32
                ).copy()
                mask = validity[row_position]
                values[~mask] = 0.0
                if not mask.any() or not np.isfinite(values).all():
                    continue
                selected_position = selected_by_index[int(source_index)]
                extracted[selected_position] = (
                    values,
                    mask.copy(),
                    {name: float(values_[row_position]) for name, values_ in quality.items()},
                )
        ordered_positions = sorted(extracted)
        if not ordered_positions:
            continue
        output_signals = np.stack([extracted[position][0] for position in ordered_positions])
        output_masks = np.stack([extracted[position][1] for position in ordered_positions])
        shard_dir = shard_root / f"shard_{shard_number:04d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        np.save(shard_dir / "signals.npy", output_signals.astype(np.float32, copy=False))
        kept = selected.iloc[ordered_positions].copy().reset_index(drop=True)
        kept["filtered_cache_dir"] = str(shard_dir)
        kept["filtered_local_index"] = np.arange(len(kept), dtype=np.int64)
        kept["ch7_valid"] = output_masks[:, 0]
        kept["ch8_valid"] = output_masks[:, 1]
        kept["valid_channel_count"] = output_masks.sum(axis=1)
        for name in next(iter(extracted.values()))[2]:
            kept[name] = [extracted[position][2][name] for position in ordered_positions]
        metadata_frames.append(kept)

    if not metadata_frames:
        raise ValueError("quality control removed every selected window")
    metadata = pd.concat(metadata_frames, ignore_index=True)
    audit_mixed_binary_split(metadata)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.parquet"
    metadata.to_parquet(metadata_path, index=False)
    baseline = fit_training_baseline(metadata)
    (output / "training_baseline.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    quality_summary = _quality_summary(frame, metadata)
    quality_summary.to_csv(
        output / "quality_summary.csv", index=False, encoding="utf-8-sig"
    )
    manifest = {
        "schema_version": 1,
        "method": "continuous_segment_zero_phase_filter_then_window_extraction",
        "filter": config.filter_config.to_dict(),
        "normalization": "per_mouse_channel_train_nonseizure_median_mad",
        "amplitude_units": "a.u._relative_to_training_nonseizure_baseline",
        "metadata_sha256": _sha256(metadata_path),
        "training_baseline_sha256": _sha256(output / "training_baseline.json"),
        "input_windows": int(len(frame)),
        "retained_windows": int(len(metadata)),
    }
    (output / "view_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata_path


def assess_window_channels(
    signals: np.ndarray,
    config: MixedBinaryConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values = np.asarray(signals, dtype=np.float64)
    finite = np.isfinite(values)
    nan_values = np.where(finite, values, np.nan)
    median = np.nanmedian(nan_values, axis=2, keepdims=True)
    median = np.nan_to_num(median, nan=0.0)
    filled = np.where(finite, values, median)
    robust_amplitude = np.nanquantile(nan_values, 0.99, axis=2) - np.nanquantile(
        nan_values, 0.01, axis=2
    )
    differences = np.diff(filled, axis=2)
    flat_fraction = np.mean(np.isclose(differences, 0.0, atol=1e-7), axis=2)
    rail_fraction = np.mean(
        np.abs(filled) >= config.hardware_rail_absolute, axis=2
    )
    derivative_scale = np.median(np.abs(differences), axis=2) * 1.4826
    maximum_jump_z = np.max(np.abs(differences), axis=2) / np.maximum(
        derivative_scale, 1e-6
    )
    valid = (
        (finite.mean(axis=2) >= config.minimum_finite_fraction)
        & (robust_amplitude > config.minimum_robust_amplitude)
        & (flat_fraction < config.maximum_flat_fraction)
        & (rail_fraction < config.maximum_hardware_rail_fraction)
    )
    quality = {
        "minimum_channel_finite_fraction": finite.mean(axis=2).min(axis=1),
        "maximum_channel_flat_fraction": flat_fraction.max(axis=1),
        "maximum_channel_rail_fraction": rail_fraction.max(axis=1),
        "maximum_channel_jump_z": maximum_jump_z.max(axis=1),
    }
    return valid, quality


def fit_training_baseline(metadata: pd.DataFrame, samples_per_window: int = 64) -> dict[str, Any]:
    train = metadata.loc[metadata["split"].eq("train") & metadata["label"].eq(0)].copy()
    if train.empty:
        raise ValueError("training split has no non-seizure baseline windows")
    samples: dict[tuple[str, int], list[np.ndarray]] = {}
    for cache_dir, group in train.groupby("filtered_cache_dir", sort=False):
        values = np.load(Path(str(cache_dir)) / "signals.npy", mmap_mode="r")
        indices = group["filtered_local_index"].to_numpy(np.int64)
        selected = np.asarray(values[indices], dtype=np.float32)
        sample_positions = np.linspace(
            0, selected.shape[-1] - 1, min(samples_per_window, selected.shape[-1]), dtype=np.int64
        )
        animal = group["animal_id"].astype(str).to_numpy()
        masks = group[["ch7_valid", "ch8_valid"]].to_numpy(bool)
        for row_index in range(len(group)):
            for channel in range(2):
                if masks[row_index, channel]:
                    samples.setdefault((animal[row_index], channel), []).append(
                        selected[row_index, channel, sample_positions]
                    )
    rows = []
    for animal in sorted(metadata["animal_id"].astype(str).unique()):
        for channel, channel_name in enumerate(CHANNEL_NAMES):
            blocks = samples.get((animal, channel), [])
            if not blocks:
                rows.append(
                    {
                        "animal_id": animal,
                        "channel_name": channel_name,
                        "valid": False,
                        "center": 0.0,
                        "scale": 1.0,
                        "sampled_values": 0,
                    }
                )
                continue
            values = np.concatenate(blocks).astype(np.float64, copy=False)
            center = float(np.median(values))
            scale = float(1.4826 * np.median(np.abs(values - center)))
            if not np.isfinite(scale) or scale <= 1e-6:
                scale = max(float(np.std(values)), 1e-6)
            rows.append(
                {
                    "animal_id": animal,
                    "channel_name": channel_name,
                    "valid": True,
                    "center": center,
                    "scale": scale,
                    "sampled_values": int(len(values)),
                }
            )
    return {
        "schema_version": 1,
        "fit_split": "train",
        "fit_class": "non-seizure",
        "unit": "a.u.",
        "method": "median_and_1.4826_MAD",
        "channels": rows,
    }


class MixedBinaryDataset(torch.utils.data.Dataset):
    """Lazy filtered windows with train-baseline-only robust scaling."""

    def __init__(
        self,
        metadata: str | Path | pd.DataFrame,
        baseline: str | Path | Mapping[str, Any],
        split: str,
        maximum_windows: int | None = None,
        normalization_clip: float = 20.0,
    ) -> None:
        frame = _read_table(metadata)
        self.frame = frame.loc[frame["split"].eq(split)].copy().reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"no windows found for split {split!r}")
        if maximum_windows is not None and len(self.frame) > maximum_windows:
            self.frame = _stratified_limit(self.frame, maximum_windows)
        baseline_data = (
            json.loads(Path(baseline).read_text(encoding="utf-8"))
            if isinstance(baseline, (str, Path))
            else dict(baseline)
        )
        self.stats = {
            (str(row["animal_id"]), str(row["channel_name"])): (
                float(row["center"]),
                float(row["scale"]),
            )
            for row in baseline_data["channels"]
        }
        self.normalization_clip = float(normalization_clip)
        self._arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        root = str(row["filtered_cache_dir"])
        if root not in self._arrays:
            self._arrays[root] = np.load(Path(root) / "signals.npy", mmap_mode="r")
        values = np.asarray(
            self._arrays[root][int(row["filtered_local_index"])], dtype=np.float32
        ).copy()
        mask = np.asarray([bool(row["ch7_valid"]), bool(row["ch8_valid"])])
        animal = str(row["animal_id"])
        for channel, name in enumerate(CHANNEL_NAMES):
            if mask[channel]:
                center, scale = self.stats[(animal, name)]
                values[channel] = np.clip(
                    (values[channel] - center) / max(scale, 1e-6),
                    -self.normalization_clip,
                    self.normalization_clip,
                )
            else:
                values[channel] = 0.0
        return {
            "signals": torch.from_numpy(values),
            "channel_mask": torch.from_numpy(mask),
            "label": int(row["label"]),
            "animal_id": animal,
            "session_id": str(row["session_id"]),
            "group_id": str(row["group_id"]),
            "start_time_ms": float(row["start_time_ms"]),
            "end_time_ms": float(row["end_time_ms"]),
        }

    def labels(self) -> np.ndarray:
        return self.frame["label"].to_numpy(np.int64)


def _build_event_complexes(events: pd.DataFrame, gap_ms: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (animal, session), group in events.groupby(
        ["animal_id", "session_id"], sort=True
    ):
        ordered = group.sort_values(["start_time_ms", "end_time_ms"], kind="stable")
        current: dict[str, Any] | None = None
        complex_number = 0
        for row in ordered.to_dict("records"):
            start, end = float(row["start_time_ms"]), float(row["end_time_ms"])
            if current is None or start > float(current["end_time_ms"]) + gap_ms:
                if current is not None:
                    rows.append(current)
                complex_number += 1
                current = {
                    "animal_id": str(animal),
                    "session_id": str(session),
                    "group_id": f"{session}:seizure:{complex_number:05d}",
                    "start_time_ms": start,
                    "end_time_ms": end,
                    "source_event_ids": str(row["event_id"]),
                    "source_event_count": 1,
                }
            else:
                current["end_time_ms"] = max(float(current["end_time_ms"]), end)
                current["source_event_ids"] += ";" + str(row["event_id"])
                current["source_event_count"] += 1
        if current is not None:
            rows.append(current)
    return pd.DataFrame(rows)


def _interval_lookup(
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    interval_starts: np.ndarray,
    interval_ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(interval_ends, window_starts, side="right")
    clipped = np.minimum(positions, max(len(interval_starts) - 1, 0))
    overlap = (positions < len(interval_starts)) & (
        interval_starts[clipped] < window_ends
    )
    return clipped, overlap


def _cap_group_windows(frame: pd.DataFrame, config: MixedBinaryConfig) -> pd.DataFrame:
    selected: list[np.ndarray] = []
    for _, group in frame.groupby("group_id", sort=True):
        ordered = group.sort_values("start_time_ms", kind="stable")
        label = int(ordered["label"].iloc[0])
        if ordered["label"].nunique() != 1:
            raise ValueError("a split group contains both seizure and non-seizure labels")
        cap = (
            config.maximum_windows_per_event_group
            if label == 1
            else config.maximum_windows_per_background_group
        )
        if len(ordered) <= cap:
            selected.append(ordered.index.to_numpy(np.int64))
        else:
            positions = np.linspace(0, len(ordered) - 1, cap, dtype=np.int64)
            selected.append(ordered.index.to_numpy(np.int64)[positions])
    result = frame.loc[np.sort(np.concatenate(selected))].copy()
    return result.sort_values(
        ["animal_id", "session_id", "start_time_ms", "cache_dir", "cache_local_index"],
        kind="stable",
    ).reset_index(drop=True)


def _assign_group_splits(frame: pd.DataFrame, config: MixedBinaryConfig) -> dict[str, str]:
    groups = (
        frame.groupby("group_id", as_index=False)
        .agg(animal_id=("animal_id", "first"), label=("label", "first"), windows=("label", "size"))
    )
    assignments: dict[str, str] = {}
    rng = np.random.default_rng(config.random_seed)
    fractions = config.fractions
    for (_, _), stratum in groups.groupby(["animal_id", "label"], sort=True):
        if len(stratum) < 3:
            raise ValueError("each mouse/class stratum needs at least three independent groups")
        records = stratum.to_dict("records")
        jitter = rng.random(len(records))
        order = sorted(
            range(len(records)),
            key=lambda index: (-int(records[index]["windows"]), float(jitter[index])),
        )
        total = float(stratum["windows"].sum())
        targets = fractions * total
        counts = np.zeros(3, dtype=float)
        for rank, record_index in enumerate(order):
            record = records[record_index]
            if rank < 3:
                split_index = rank
            else:
                relative_fill = counts / np.maximum(targets, 1.0)
                split_index = int(np.argmin(relative_fill + rng.random(3) * 1e-9))
            split = SPLITS[split_index]
            assignments[str(record["group_id"])] = split
            counts[split_index] += int(record["windows"])
    return assignments


def _audit_interval_overlap(frame: pd.DataFrame) -> None:
    for session, group in frame.groupby("session_id", sort=False):
        ordered = group.sort_values(["start_time_ms", "end_time_ms"], kind="stable")
        maximum_end = {split: -np.inf for split in SPLITS}
        for row in ordered.itertuples(index=False):
            start = float(row.start_time_ms)
            split = str(row.split)
            for other in SPLITS:
                if other != split and maximum_end[other] > start + 1e-6:
                    raise ValueError(
                        f"raw-window overlap crosses {other}/{split} in session {session}"
                    )
            maximum_end[split] = max(maximum_end[split], float(row.end_time_ms))


def _continuity_segments(starts: np.ndarray, stride_ms: float) -> list[np.ndarray]:
    if not len(starts):
        return []
    breaks = np.flatnonzero(~np.isclose(np.diff(starts), stride_ms, rtol=1e-5, atol=5.0)) + 1
    return [segment for segment in np.split(np.arange(len(starts), dtype=np.int64), breaks) if len(segment)]


def _reconstruct_segment(
    signals: np.ndarray, segment: np.ndarray, stride_samples: int
) -> np.ndarray:
    first = np.asarray(signals[int(segment[0])], dtype=np.float32)
    if len(segment) == 1:
        return first.copy()
    tails = np.asarray(signals[segment[1:], :, -stride_samples:], dtype=np.float32)
    appended = tails.transpose(1, 0, 2).reshape(signals.shape[1], -1)
    return np.concatenate([first, appended], axis=1)


def _stratified_limit(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if maximum < 2:
        raise ValueError("maximum_windows must be at least two")
    available = {label: int(frame["label"].eq(label).sum()) for label in (0, 1)}
    if not all(available.values()):
        raise ValueError("capped binary dataset needs both classes")
    counts = {label: min(available[label], maximum // 2) for label in (0, 1)}
    remainder = maximum - sum(counts.values())
    for label in sorted((0, 1), key=lambda value: -available[value]):
        extra = min(remainder, available[label] - counts[label])
        counts[label] += extra
        remainder -= extra

    selected_indices: list[Any] = []
    for label in (0, 1):
        group = frame.loc[frame["label"].eq(label)]
        positions = np.linspace(
            0, len(group) - 1, counts[label], dtype=np.int64
        )
        selected_indices.extend(group.index.to_numpy()[positions].tolist())
    if len(set(selected_indices)) != maximum:
        raise AssertionError("stratified cap selected duplicate or missing rows")
    return frame.loc[selected_indices].reset_index(drop=True)


def _group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["group_id", "animal_id", "session_id", "label", "split"], as_index=False)
        .agg(
            windows=("label", "size"),
            start_time_ms=("start_time_ms", "min"),
            end_time_ms=("end_time_ms", "max"),
        )
        .sort_values(["animal_id", "session_id", "start_time_ms"], kind="stable")
    )


def _split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, animal), group in frame.groupby(["split", "animal_id"], sort=True):
        rows.append(
            {
                "split": split,
                "animal_id": animal,
                "windows": int(len(group)),
                "seizure_windows": int(group["label"].sum()),
                "nonseizure_windows": int((group["label"] == 0).sum()),
                "positive_fraction": float(group["label"].mean()),
                "seizure_groups": int(group.loc[group["label"].eq(1), "group_id"].nunique()),
                "background_groups": int(group.loc[group["label"].eq(0), "group_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _quality_summary(input_frame: pd.DataFrame, retained: pd.DataFrame) -> pd.DataFrame:
    input_counts = input_frame.groupby(["animal_id", "split"]).size().rename("input_windows")
    kept_counts = retained.groupby(["animal_id", "split"]).size().rename("retained_windows")
    result = pd.concat([input_counts, kept_counts], axis=1).fillna(0).reset_index()
    result["removed_windows"] = result["input_windows"] - result["retained_windows"]
    result["retained_fraction"] = result["retained_windows"] / result["input_windows"].clip(lower=1)
    return result


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "MixedBinaryConfig",
    "MixedBinaryDataset",
    "assess_window_channels",
    "audit_mixed_binary_split",
    "build_mixed_binary_index",
    "fit_training_baseline",
    "materialize_filtered_binary_view",
]
