from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from mouse_sleep.epilepsy.blind_single_channel import _channel_event_labels, _parse_channels
from mouse_sleep.epilepsy.mixed_binary import MixedBinaryConfig, assess_window_channels
from mouse_sleep.epilepsy.signal_processing import EEGFilterConfig, filter_continuous_eeg


CHANNEL_NAMES = ("CH7", "CH8")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ContinuousSingleChannelConfig:
    """Frozen protocol for channel-blind continuous seizure annotation."""

    animals: tuple[str, ...] = ("E50", "E76", "E77")
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    sequence_length: int = 21
    window_seconds: float = 4.0
    stride_seconds: float = 1.0
    positive_overlap_fraction: float = 0.50
    event_context_margin_seconds: float = 12.0
    background_block_seconds: float = 300.0
    # ``None`` means that every eligible center is retained.  The previous
    # protocol used hard caps here, which silently discarded large portions of
    # both the seizure and background distributions before training.
    maximum_positive_centers_per_group_channel: int | None = None
    maximum_event_negative_centers_per_group_channel: int | None = None
    maximum_background_centers_per_group_channel: int | None = None
    random_seed: int = 20260814
    normalization_clip: float = 20.0
    excluded_animal_channels: tuple[tuple[str, str], ...] = (("E77", "CH7"),)
    # A retained center must not share raw samples with a center assigned to a
    # different split. This is stricter than masking another split's context
    # rows and is required for overlapping 4-s windows and 21-window Physio
    # sequences.
    purge_cross_split_windows: bool = True
    filter_config: EEGFilterConfig = field(default_factory=EEGFilterConfig)

    @property
    def fractions(self) -> np.ndarray:
        return np.asarray(
            [self.train_fraction, self.validation_fraction, self.test_fraction],
            dtype=float,
        )

    @property
    def context_half_width(self) -> int:
        return self.sequence_length // 2

    def validated(self) -> "ContinuousSingleChannelConfig":
        if not self.animals or len(set(self.animals)) != len(self.animals):
            raise ValueError("animals must be a non-empty unique tuple")
        if self.sequence_length < 1 or self.sequence_length % 2 == 0:
            raise ValueError("sequence_length must be a positive odd integer")
        if not np.isclose(self.fractions.sum(), 1.0) or np.any(self.fractions <= 0):
            raise ValueError("train/validation/test fractions must be positive and sum to one")
        if self.window_seconds <= 0 or self.stride_seconds <= 0:
            raise ValueError("window and stride durations must be positive")
        if not 0 < self.positive_overlap_fraction <= 1:
            raise ValueError("positive_overlap_fraction must lie in (0, 1]")
        required_margin = (
            self.context_half_width * self.stride_seconds + self.window_seconds / 2.0
        )
        if self.event_context_margin_seconds + 1e-9 < required_margin:
            raise ValueError(
                "event_context_margin_seconds must cover the complete raw sequence span "
                f"({required_margin:g} s)"
            )
        if self.background_block_seconds <= 2 * required_margin:
            raise ValueError("background blocks must be wider than two context margins")
        caps = (
            self.maximum_positive_centers_per_group_channel,
            self.maximum_event_negative_centers_per_group_channel,
            self.maximum_background_centers_per_group_channel,
        )
        for cap in caps:
            if cap is not None and cap < 1:
                raise ValueError("per-group center caps must be positive or None")
        if self.normalization_clip <= 0:
            raise ValueError("normalization_clip must be positive")
        unknown = {
            str(channel).upper() for _, channel in self.excluded_animal_channels
        } - set(CHANNEL_NAMES)
        if unknown:
            raise ValueError(f"unknown excluded channels: {sorted(unknown)}")
        return self

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["filter_config"] = self.filter_config.to_dict()
        return values


def recover_complete_window_grid(
    retained_windows: np.ndarray,
    retained_start_time_ms: np.ndarray,
    *,
    sampling_rate_hz: float,
    window_seconds: float,
    stride_seconds: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Recover label-removed windows from the exact overlap of retained windows.

    The legacy cache discarded boundary windows with a small manual-event overlap.
    At a one-second stride, adjacent retained four-second windows still cover those
    samples.  This function reconstructs the continuous samples first and then
    extracts the complete fixed-stride grid without consulting any labels.
    """

    values = np.asarray(retained_windows)
    starts = np.asarray(retained_start_time_ms, dtype=np.float64)
    if values.ndim != 3 or len(values) != len(starts) or len(values) < 1:
        raise ValueError("retained windows must be [windows, channels, samples]")
    if not np.isfinite(starts).all():
        raise ValueError("retained start times must be finite")
    stride_samples = int(round(sampling_rate_hz * stride_seconds))
    window_samples = int(round(sampling_rate_hz * window_seconds))
    if values.shape[-1] != window_samples or not 0 < stride_samples <= window_samples:
        raise ValueError("window shape does not match the sampling configuration")

    # Device timestamps contain rare local backward corrections.  Grid steps are
    # therefore inferred from exact waveform overlap, not from the annotations or
    # the timestamp difference.  Candidate k is valid when the last W-kS samples
    # of one retained window equal the first W-kS samples of the next.
    maximum_step = (window_samples - 1) // stride_samples
    rounded_steps = np.empty(max(len(values) - 1, 0), dtype=np.int64)
    maximum_step_match_error = 0.0
    for index in range(len(values) - 1):
        errors = []
        previous = np.asarray(values[index], dtype=np.float32)
        following = np.asarray(values[index + 1], dtype=np.float32)
        for step in range(1, maximum_step + 1):
            overlap_samples = window_samples - step * stride_samples
            difference = np.abs(
                previous[:, -overlap_samples:] - following[:, :overlap_samples]
            )
            errors.append(float(difference.max()) if difference.size else float("inf"))
        best = int(np.argmin(errors))
        rounded_steps[index] = best + 1
        maximum_step_match_error = max(maximum_step_match_error, errors[best])
    if maximum_step_match_error > 1e-5:
        raise ValueError(
            "adjacent retained windows do not share an exact fixed-stride overlap; "
            f"maximum best-match error={maximum_step_match_error:g}"
        )
    positions = np.r_[0, np.cumsum(rounded_steps)].astype(np.int64)
    complete_count = int(positions[-1]) + 1
    continuous_samples = (complete_count - 1) * stride_samples + window_samples
    continuous = np.zeros(
        (values.shape[1], continuous_samples), dtype=np.float32
    )
    coverage = np.zeros(continuous_samples, dtype=bool)
    maximum_overlap_error = 0.0
    for source_index, grid_position in enumerate(positions):
        left = int(grid_position) * stride_samples
        right = left + window_samples
        window = np.asarray(values[source_index], dtype=np.float32)
        overlap = coverage[left:right]
        if overlap.any():
            difference = np.abs(continuous[:, left:right][:, overlap] - window[:, overlap])
            if difference.size:
                maximum_overlap_error = max(maximum_overlap_error, float(difference.max()))
        continuous[:, left:right] = window
        coverage[left:right] = True
    if not coverage.all():
        missing_samples = int((~coverage).sum())
        raise ValueError(f"overlapping retained windows leave {missing_samples} samples uncovered")

    complete = np.empty(
        (complete_count, values.shape[1], window_samples), dtype=np.float32
    )
    for output_index in range(complete_count):
        left = output_index * stride_samples
        complete[output_index] = continuous[:, left : left + window_samples]
    grid_positions = np.arange(complete_count, dtype=np.float64)
    complete_starts, timestamp_diagnostics = _regularize_grid_timestamps(
        positions.astype(float), starts, grid_positions, stride_seconds
    )
    diagnostics = {
        "retained_windows": int(len(values)),
        "complete_windows": int(complete_count),
        "recovered_windows": int(complete_count - len(values)),
        "maximum_removed_consecutive_windows": int(
            max(int(rounded_steps.max()) - 1, 0) if len(rounded_steps) else 0
        ),
        "maximum_overlap_sample_error": float(maximum_overlap_error),
        "maximum_step_match_error": float(maximum_step_match_error),
        **timestamp_diagnostics,
    }
    return complete, complete_starts, diagnostics


def materialize_complete_filtered_cache(
    cache_manifest: str | Path | pd.DataFrame,
    output_dir: str | Path,
    config: ContinuousSingleChannelConfig | None = None,
) -> Path:
    """Create a label-independent, gap-free, continuously filtered stage-1 cache."""

    config = (config or ContinuousSingleChannelConfig()).validated()
    manifest = _read_table(cache_manifest)
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
        "cache manifest",
    )
    selected = manifest.loc[
        manifest["animal_id"].astype(str).isin(config.animals)
    ].copy()
    if selected.empty:
        raise ValueError("cache manifest contains none of the configured animals")
    output = Path(output_dir).resolve()
    shard_root = output / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    cache_root = _common_cache_root(cache_manifest)
    quality_config = MixedBinaryConfig(filter_config=config.filter_config).validated()
    rows: list[dict[str, Any]] = []
    for row in selected.sort_values(
        ["animal_id", "session_id", "shard"], kind="stable"
    ).to_dict("records"):
        source_root = _resolve_cache_dir(row["cache_dir"], cache_root)
        retained = np.load(source_root / "signals.npy", mmap_mode="r")
        retained_starts = np.load(source_root / "start_time_ms.npy", mmap_mode="r")
        rate = float(row["sampling_rate_hz"])
        complete, starts, recovery = recover_complete_window_grid(
            retained,
            retained_starts,
            sampling_rate_hz=rate,
            window_seconds=float(row["window_seconds"]),
            stride_seconds=float(row["stride_seconds"]),
        )
        source_meta_path = source_root / "meta.json"
        raw_edge_recovery_windows = 0
        if source_meta_path.is_file():
            source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
            expected_count = _expected_source_window_count(
                source_meta.get("source"),
                sampling_rate_hz=rate,
                window_samples=complete.shape[-1],
                stride_samples=int(round(rate * float(row["stride_seconds"]))),
            )
            if expected_count < len(complete):
                raise ValueError(
                    f"overlap recovery produced more windows than source parquet in {source_root}"
                )
            if expected_count > len(complete):
                raw_edge_recovery_windows = expected_count - len(complete)
                complete, starts, raw_diagnostics = _rebuild_grid_from_source_parquet(
                    source_meta["source"],
                    target_rate_hz=rate,
                    window_seconds=float(row["window_seconds"]),
                    stride_seconds=float(row["stride_seconds"]),
                )
                recovery.update(raw_diagnostics)
                recovery["complete_windows"] = int(len(complete))
                recovery["recovered_windows"] = int(len(complete) - len(retained))
        recovery["raw_edge_recovery_windows"] = int(raw_edge_recovery_windows)
        validity, quality = assess_window_channels(complete, quality_config)
        if str(row["animal_id"]).upper() == "E77":
            validity[:, 0] = False
        stride_samples = int(round(rate * float(row["stride_seconds"])))
        continuous = _windows_to_continuous(complete, stride_samples)
        filtered_continuous = filter_continuous_eeg(
            continuous, rate, config.filter_config
        )
        filtered = _continuous_to_windows(
            filtered_continuous, len(complete), stride_samples, complete.shape[-1]
        )
        filtered[~validity] = 0.0
        shard_name = source_root.name
        target = shard_root / shard_name
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / "signals.npy", filtered.astype(np.float32, copy=False))
        np.save(target / "start_time_ms.npy", starts.astype(np.float64, copy=False))
        np.save(target / "channel_valid.npy", validity.astype(bool, copy=False))
        np.savez_compressed(
            target / "quality.npz", **{name: value.astype(np.float32) for name, value in quality.items()}
        )
        rows.append(
            {
                "animal_id": str(row["animal_id"]),
                "session_id": str(row["session_id"]),
                "shard": int(row["shard"]),
                "source_cache_dir": str(source_root),
                "filtered_cache_dir": str(target),
                "sampling_rate_hz": rate,
                "window_seconds": float(row["window_seconds"]),
                "stride_seconds": float(row["stride_seconds"]),
                "n_windows": int(len(complete)),
                "valid_ch7_windows": int(validity[:, 0].sum()),
                "valid_ch8_windows": int(validity[:, 1].sum()),
                **recovery,
            }
        )
    result = pd.DataFrame(rows)
    manifest_path = output / "complete_filtered_manifest.csv"
    result.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    summary = {
        "schema_version": 1,
        "method": "exact_overlap_recovery_then_continuous_zero_phase_filter",
        "label_used_to_construct_signal_grid": False,
        "filter": config.filter_config.to_dict(),
        "amplitude_unit": "a.u.",
        "recovered_windows": int(result["recovered_windows"].sum()),
        "raw_edge_recovery_windows": int(result["raw_edge_recovery_windows"].sum()),
        "maximum_overlap_sample_error": float(result["maximum_overlap_sample_error"].max()),
        "manifest_sha256": _sha256(manifest_path),
    }
    (output / "cache_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_path


def build_continuous_single_channel_protocol(
    filtered_manifest: str | Path | pd.DataFrame,
    event_components: str | Path | pd.DataFrame,
    output_dir: str | Path,
    config: ContinuousSingleChannelConfig | None = None,
) -> Path:
    """Build purged train/validation/test regions and channel-specific labels."""

    config = (config or ContinuousSingleChannelConfig()).validated()
    manifest = _read_table(filtered_manifest)
    events = _validated_events(event_components, config.animals)
    _require_columns(
        manifest,
        {
            "animal_id",
            "session_id",
            "shard",
            "filtered_cache_dir",
            "sampling_rate_hz",
            "window_seconds",
            "stride_seconds",
        },
        "complete filtered manifest",
    )
    manifest = manifest.loc[
        manifest["animal_id"].astype(str).isin(config.animals)
    ].copy()
    timeline = _timeline_bounds(manifest)
    regions, event_group_lookup = _build_regions(timeline, events, config)
    regions = _assign_region_splits(regions, config)
    event_index = _expand_channel_events(events, regions, event_group_lookup, config)

    excluded = {
        (str(animal).upper(), str(channel).upper())
        for animal, channel in config.excluded_animal_channels
    }
    rows: list[pd.DataFrame] = []
    for manifest_row in manifest.sort_values(
        ["animal_id", "session_id", "shard"], kind="stable"
    ).to_dict("records"):
        animal = str(manifest_row["animal_id"])
        session = str(manifest_row["session_id"])
        root = Path(str(manifest_row["filtered_cache_dir"]))
        starts = np.asarray(np.load(root / "start_time_ms.npy", mmap_mode="r"), dtype=float)
        validity = np.asarray(np.load(root / "channel_valid.npy", mmap_mode="r"), dtype=bool)
        window_ms = float(manifest_row["window_seconds"]) * 1000.0
        ends = starts + window_ms
        centers = starts + window_ms / 2.0
        session_events = events.loc[
            events["animal_id"].eq(animal) & events["session_id"].eq(session)
        ]
        session_regions = regions.loc[
            regions["animal_id"].eq(animal) & regions["session_id"].eq(session)
        ].sort_values("start_time_ms", kind="stable")
        region_positions = _region_positions(centers, session_regions)
        assigned_regions = session_regions.iloc[region_positions].reset_index(drop=True)
        for channel_index, channel_name in enumerate(CHANNEL_NAMES):
            if (animal.upper(), channel_name) in excluded:
                continue
            labels, overlap_ms, event_ids = _channel_event_labels(
                starts,
                ends,
                session_events,
                channel_name,
                config.positive_overlap_fraction * window_ms,
            )
            count = len(starts)
            rows.append(
                pd.DataFrame(
                    {
                        "animal_id": animal,
                        "session_id": session,
                        "shard": int(manifest_row["shard"]),
                        "channel_name": channel_name,
                        "channel_index": channel_index,
                        "start_time_ms": starts,
                        "end_time_ms": ends,
                        "center_time_ms": centers,
                        "label": labels.to_numpy(np.int8),
                        "channel_event_overlap_ms": overlap_ms,
                        "channel_event_ids": event_ids,
                        "region_id": assigned_regions["region_id"].astype(str).to_numpy(),
                        "region_kind": assigned_regions["region_kind"].astype(str).to_numpy(),
                        "split": assigned_regions["split"].astype(str).to_numpy(),
                        "channel_valid": validity[:, channel_index],
                        "filtered_cache_dir": str(root),
                        "filtered_local_index": np.arange(count, dtype=np.int64),
                        "sampling_rate_hz": float(manifest_row["sampling_rate_hz"]),
                        "window_seconds": float(manifest_row["window_seconds"]),
                        "stride_seconds": float(manifest_row["stride_seconds"]),
                    }
                )
            )
    full_index = pd.concat(rows, ignore_index=True)
    full_index = _annotate_continuity_and_context(full_index, config)
    full_index = _annotate_cross_split_overlaps(full_index)
    centers = _select_supervised_centers(full_index, config)
    center_ids = set(centers["sample_id"].astype(str))
    full_index["supervised_center_eligible"] = full_index["sample_id"].astype(str).isin(
        center_ids
    )
    baseline = _fit_training_baseline(full_index, centers)
    audit = audit_continuous_single_channel_protocol(
        full_index, centers, regions, event_index, config
    )

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    full_path = output / "full_index.parquet"
    centers_path = output / "supervised_centers.parquet"
    regions_path = output / "split_regions.parquet"
    events_path = output / "channel_events.parquet"
    full_index.to_parquet(full_path, index=False)
    centers.to_parquet(centers_path, index=False)
    regions.to_parquet(regions_path, index=False)
    event_index.to_parquet(events_path, index=False)
    (output / "training_baseline.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    audit.update(
        {
            "full_index_sha256": _sha256(full_path),
            "supervised_centers_sha256": _sha256(centers_path),
            "split_regions_sha256": _sha256(regions_path),
            "channel_events_sha256": _sha256(events_path),
        }
    )
    (output / "protocol_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return full_path


class ContinuousSingleChannelDataset(torch.utils.data.Dataset):
    """Leakage-safe centered sequences with audit identity kept outside batches."""

    def __init__(
        self,
        full_index: str | Path | pd.DataFrame,
        centers: str | Path | pd.DataFrame | None,
        baseline: str | Path | Mapping[str, Any],
        *,
        split: str | None,
        sequence_length: int = 21,
        normalization_clip: float = 20.0,
        mask_other_splits: bool = True,
        include_transition_centers: bool = False,
        context_frame: pd.DataFrame | None = None,
    ) -> None:
        if sequence_length < 1 or sequence_length % 2 == 0:
            raise ValueError("sequence_length must be a positive odd integer")
        # Training creates train/validation/test datasets in one process.  A
        # large protocol can exceed several GB when each dataset independently
        # reads and sorts the full index.  ``context_frame`` lets the caller
        # share one immutable, already-sorted frame across those datasets;
        # standalone callers retain the original file-based behaviour.
        if context_frame is None:
            frame = _read_table(full_index)
            frame = frame.sort_values(
                ["continuity_id", "continuity_position"], kind="stable"
            ).reset_index(drop=True)
            frame["_context_row"] = np.arange(len(frame), dtype=np.int64)
        else:
            frame = context_frame
            if "_context_row" not in frame.columns:
                frame["_context_row"] = np.arange(len(frame), dtype=np.int64)
        if centers is None:
            audit = frame.loc[frame["channel_valid"]].copy()
            if not include_transition_centers:
                audit = audit.loc[audit["label"].ne(-1)]
        else:
            selected = _read_table(centers)
            key = [
                "animal_id",
                "session_id",
                "channel_name",
                "start_time_ms",
            ]
            audit = selected.merge(
                frame[key + ["_context_row"]], on=key, how="left", validate="one_to_one"
            )
            if audit["_context_row"].isna().any():
                raise ValueError("some supervised centers are absent from the full index")
        if split is not None:
            audit = audit.loc[audit["split"].astype(str).eq(str(split))]
        if audit.empty:
            raise ValueError("no centers remain for the requested dataset")
        self._context = frame
        self.audit_frame = audit.reset_index(drop=True)
        self.sequence_length = int(sequence_length)
        self.half_width = self.sequence_length // 2
        self.normalization_clip = float(normalization_clip)
        self.mask_other_splits = bool(mask_other_splits)
        baseline_data = (
            json.loads(Path(baseline).read_text(encoding="utf-8"))
            if isinstance(baseline, (str, Path))
            else dict(baseline)
        )
        self.stats = {
            (str(row["animal_id"]), str(row["channel_name"])): (
                float(row["center"]), float(row["scale"])
            )
            for row in baseline_data["channels"]
        }
        self._arrays: dict[str, np.ndarray] = {}
        # Compact immutable column arrays keep the hot ``__getitem__`` path
        # out of pandas.  The protocol identity and ordering remain exactly
        # those in ``full_index.parquet``; this is only a data-loader speedup.
        self._animal = frame["animal_id"].astype(str).to_numpy()
        self._channel = frame["channel_name"].astype(str).to_numpy()
        self._split = frame["split"].astype(str).to_numpy()
        self._valid = frame["channel_valid"].astype(bool).to_numpy()
        self._cache_dir = frame["filtered_cache_dir"].astype(str).to_numpy()
        self._local_index = frame["filtered_local_index"].to_numpy(np.int64)
        self._channel_index = frame["channel_index"].to_numpy(np.int64)
        self._sampling_rate = frame["sampling_rate_hz"].to_numpy(float)
        self._window_seconds = frame["window_seconds"].to_numpy(float)
        self._label = frame["label"].to_numpy(np.int8)
        self._continuity_id = frame["continuity_id"].astype(str).to_numpy()
        self._audit_context_rows = self.audit_frame["_context_row"].to_numpy(np.int64)
        self._audit_labels = self.audit_frame["label"].to_numpy(np.int8)
        self._segment_rows = {
            str(segment): group.index.to_numpy(np.int64)
            for segment, group in self._context.groupby("continuity_id", sort=False)
        }
        self._row_position = np.empty(len(self._context), dtype=np.int64)
        for indices in self._segment_rows.values():
            self._row_position[indices] = np.arange(len(indices), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.audit_frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        context_row = int(self._audit_context_rows[index])
        segment_rows = self._segment_rows[self._continuity_id[context_row]]
        position = int(self._row_position[context_row])
        samples = int(round(self._sampling_rate[context_row] * self._window_seconds[context_row]))
        signals = np.zeros((self.sequence_length, 1, samples), dtype=np.float32)
        padding = np.ones(self.sequence_length, dtype=bool)
        for output_position, offset in enumerate(range(-self.half_width, self.half_width + 1)):
            neighbor_position = position + offset
            if not 0 <= neighbor_position < len(segment_rows):
                continue
            neighbor_row = int(segment_rows[neighbor_position])
            if self.mask_other_splits and self._split[neighbor_row] != self._split[context_row]:
                continue
            if not self._valid[neighbor_row]:
                continue
            values = self._read_channel_index(neighbor_row)
            key = (self._animal[neighbor_row], self._channel[neighbor_row])
            center_value, scale = self.stats[key]
            signals[output_position, 0] = np.clip(
                (values - center_value) / max(scale, 1e-6),
                -self.normalization_clip,
                self.normalization_clip,
            )
            padding[output_position] = False
        if padding[self.half_width]:
            raise ValueError("the center window is invalid or unavailable")
        return {
            "signals": torch.from_numpy(signals),
            "padding_mask": torch.from_numpy(padding),
            "label": int(self._audit_labels[index]),
            "row_index": int(index),
        }

    def labels(self) -> np.ndarray:
        return self.audit_frame["label"].to_numpy(np.int64)

    def _read_channel(self, row: pd.Series) -> np.ndarray:
        root = str(row["filtered_cache_dir"])
        if root not in self._arrays:
            self._arrays[root] = np.load(Path(root) / "signals.npy", mmap_mode="r")
        return np.asarray(
            self._arrays[root][
                int(row["filtered_local_index"]), int(row["channel_index"])
            ],
            dtype=np.float32,
        ).copy()

    def _read_channel_index(self, row_index: int) -> np.ndarray:
        root = self._cache_dir[row_index]
        if root not in self._arrays:
            self._arrays[root] = np.load(Path(root) / "signals.npy", mmap_mode="r")
        return np.asarray(
            self._arrays[root][
                int(self._local_index[row_index]), int(self._channel_index[row_index])
            ],
            dtype=np.float32,
        ).copy()


def audit_continuous_single_channel_protocol(
    full_index: pd.DataFrame,
    centers: pd.DataFrame,
    regions: pd.DataFrame,
    event_index: pd.DataFrame,
    config: ContinuousSingleChannelConfig,
) -> dict[str, Any]:
    if (
        full_index["animal_id"].astype(str).str.upper().eq("E77")
        & full_index["channel_name"].astype(str).str.upper().eq("CH7")
    ).any():
        raise ValueError("E77 CH7 must be excluded")
    if set(full_index["split"].astype(str)) != set(SPLITS):
        raise ValueError("all three splits must be represented")
    if set(centers["label"].astype(int)) != {0, 1}:
        raise ValueError("supervised centers must contain both binary classes")
    region_split_counts = regions.groupby("region_id")["split"].nunique()
    if region_split_counts.ne(1).any():
        raise ValueError("a temporal region crosses splits")
    event_splits = event_index.groupby("event_id")["split"].nunique()
    if event_splits.ne(1).any():
        raise ValueError("an annotated event crosses splits")
    dual = event_index.groupby("event_id").filter(lambda group: len(group) > 1)
    if not dual.empty and dual.groupby("event_id")["split"].nunique().ne(1).any():
        raise ValueError("a dual-channel event crosses splits")
    # The protocol now purges both centers whose sequence context crosses a
    # split boundary and centers whose raw 4-s window overlaps a different
    # split. Dataset-side masking remains a defensive check, not the primary
    # leakage guarantee.
    positive_centers = centers.loc[centers["label"].eq(1)]
    context_masked = int((~centers["full_split_context"]).sum())
    positive_context_masked = int((~positive_centers["full_split_context"]).sum())
    eligible_all = full_index.loc[
        full_index["channel_valid"] & full_index["label"].isin([0, 1])
    ]
    eligible = eligible_all
    if config.purge_cross_split_windows:
        eligible = eligible.loc[
            eligible["full_split_context"]
            & ~eligible["cross_split_raw_overlap"]
        ]
    retained_cross_split_pairs = _count_cross_split_overlap_pairs(centers)
    if config.purge_cross_split_windows and retained_cross_split_pairs:
        raise ValueError(
            "strict split purge retained overlapping windows across splits: "
            f"{retained_cross_split_pairs} pairs"
        )
    transition_rows = int(full_index["label"].eq(-1).sum())
    split_counts = (
        centers.groupby(["split", "label"]).size().unstack(fill_value=0)
        .reindex(index=SPLITS, columns=[0, 1], fill_value=0)
    )
    split_totals = split_counts.sum(axis=1)
    total_centers = max(int(split_totals.sum()), 1)
    channel_summary = (
        centers.groupby(["animal_id", "channel_name", "split", "label"])
        .size().rename("centers").reset_index().to_dict("records")
    )
    event_split_summary = (
        event_index.groupby("split")
        .agg(
            independent_events=("event_id", "nunique"),
            channel_events=("event_id", "size"),
            annotated_duration_seconds=("duration_s", "sum"),
        )
        .reindex(SPLITS, fill_value=0)
    )
    return {
        "schema_version": 2,
        "protocol": "pooled_mice_channel_blind_purged_temporal_regions_70_15_15_strict_raw_window_purge",
        "model_input_channels": 1,
        "animal_or_channel_identity_is_model_input": False,
        "event_subtype_used": False,
        "event_groups_cross_splits": 0,
        "dual_channel_event_copies_cross_splits": 0,
        "positive_centers_without_full_split_context": positive_context_masked,
        "centers_with_masked_cross_split_context": context_masked,
        "context_masking_is_leakage_safe": True,
        "strict_split_window_purge_enabled": bool(config.purge_cross_split_windows),
        "cross_split_raw_overlap_pairs_full_index": int(
            _count_cross_split_overlap_pairs(full_index)
        ),
        "cross_split_raw_overlap_rows_full_index": int(
            full_index["cross_split_raw_overlap"].sum()
        ),
        "cross_split_raw_overlap_pairs_retained_centers": int(
            retained_cross_split_pairs
        ),
        "eligible_centers_before_split_purge": int(len(eligible_all)),
        "centers_removed_by_split_purge": int(len(eligible_all) - len(eligible)),
        "centers_removed_by_context_purge": int(
            eligible_all["full_split_context"].eq(False).sum()
        ) if config.purge_cross_split_windows else 0,
        "centers_removed_by_raw_overlap_purge": int(
            eligible_all["cross_split_raw_overlap"].sum()
        ) if config.purge_cross_split_windows else 0,
        "full_rows": int(len(full_index)),
        "eligible_centers": int(len(eligible)),
        "supervised_centers": int(len(centers)),
        "transition_rows_excluded": transition_rows,
        "all_eligible_centers_retained": int(len(eligible_all)) == int(len(centers)),
        "channel_events": int(len(event_index)),
        "unique_clinical_events": int(event_index["event_id"].nunique()),
        "event_split_summary": {
            split: {
                "independent_events": int(event_split_summary.loc[split, "independent_events"]),
                "channel_events": int(event_split_summary.loc[split, "channel_events"]),
                "annotated_duration_seconds": float(
                    event_split_summary.loc[split, "annotated_duration_seconds"]
                ),
            }
            for split in SPLITS
        },
        "split_center_counts": {
            split: {"non_seizure": int(split_counts.loc[split, 0]),
                    "seizure": int(split_counts.loc[split, 1])}
            for split in SPLITS
        },
        "split_center_fractions": {
            split: float(split_totals.loc[split] / total_centers) for split in SPLITS
        },
        "center_summary": channel_summary,
        "config": config.to_dict(),
    }


def _timeline_bounds(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (animal, session), group in manifest.groupby(["animal_id", "session_id"], sort=True):
        starts_all = []
        ends_all = []
        for row in group.itertuples(index=False):
            root = Path(str(row.filtered_cache_dir))
            starts = np.load(root / "start_time_ms.npy", mmap_mode="r")
            if len(starts):
                starts_all.append(float(starts[0]) + float(row.window_seconds) * 500.0)
                ends_all.append(float(starts[-1]) + float(row.window_seconds) * 500.0)
        rows.append(
            {
                "animal_id": str(animal),
                "session_id": str(session),
                "start_time_ms": min(starts_all),
                "end_time_ms": max(ends_all),
            }
        )
    return pd.DataFrame(rows)


def _build_regions(
    timeline: pd.DataFrame,
    events: pd.DataFrame,
    config: ContinuousSingleChannelConfig,
) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, Any]] = []
    event_lookup: dict[str, str] = {}
    margin_ms = config.event_context_margin_seconds * 1000.0
    for timeline_row in timeline.itertuples(index=False):
        animal = str(timeline_row.animal_id)
        session = str(timeline_row.session_id)
        session_events = events.loc[
            events["animal_id"].eq(animal) & events["session_id"].eq(session)
        ].sort_values(["start_time_ms", "end_time_ms"], kind="stable")
        event_regions: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for event in session_events.itertuples(index=False):
            expanded_start = max(float(timeline_row.start_time_ms), float(event.start_time_ms) - margin_ms)
            expanded_end = min(float(timeline_row.end_time_ms), float(event.end_time_ms) + margin_ms)
            if current is None or expanded_start > float(current["end_time_ms"]):
                if current is not None:
                    event_regions.append(current)
                current = {
                    "start_time_ms": expanded_start,
                    "end_time_ms": expanded_end,
                    "event_ids": [str(event.event_id)],
                }
            else:
                current["end_time_ms"] = max(float(current["end_time_ms"]), expanded_end)
                current["event_ids"].append(str(event.event_id))
        if current is not None:
            event_regions.append(current)

        cursor = float(timeline_row.start_time_ms)
        background_number = 0
        event_number = 0
        for event_region in event_regions:
            if float(event_region["start_time_ms"]) > cursor:
                background_number = _append_background_regions(
                    rows,
                    animal,
                    session,
                    cursor,
                    float(event_region["start_time_ms"]),
                    background_number,
                    config,
                )
            event_number += 1
            region_id = f"{session}:context_event:{event_number:05d}"
            event_ids = list(event_region["event_ids"])
            rows.append(
                {
                    "animal_id": animal,
                    "session_id": session,
                    "region_id": region_id,
                    "region_kind": "event_context",
                    "start_time_ms": float(event_region["start_time_ms"]),
                    "end_time_ms": float(event_region["end_time_ms"]),
                    "event_count": len(event_ids),
                    "duration_seconds": float(
                        (event_region["end_time_ms"] - event_region["start_time_ms"])
                        / 1000.0
                    ),
                    # Event count is the split target. Duration is retained for
                    # audit only because a few episodes are extremely long.
                    "weight": float(len(event_ids)),
                }
            )
            for event_id in event_ids:
                event_lookup[event_id] = region_id
            cursor = float(event_region["end_time_ms"])
        if cursor < float(timeline_row.end_time_ms):
            _append_background_regions(
                rows,
                animal,
                session,
                cursor,
                float(timeline_row.end_time_ms) + 1e-6,
                background_number,
                config,
            )
    regions = pd.DataFrame(rows).sort_values(
        ["animal_id", "session_id", "start_time_ms"], kind="stable"
    ).reset_index(drop=True)
    if not event_lookup:
        raise ValueError("no event regions were constructed")
    return regions, event_lookup


def _append_background_regions(
    rows: list[dict[str, Any]],
    animal: str,
    session: str,
    start_ms: float,
    end_ms: float,
    number: int,
    config: ContinuousSingleChannelConfig,
) -> int:
    block_ms = config.background_block_seconds * 1000.0
    cursor = start_ms
    while cursor < end_ms - 1e-6:
        stop = min(cursor + block_ms, end_ms)
        number += 1
        rows.append(
            {
                "animal_id": animal,
                "session_id": session,
                "region_id": f"{session}:background:{number:05d}",
                "region_kind": "background",
                "start_time_ms": float(cursor),
                "end_time_ms": float(stop),
                "event_count": 0,
                "duration_seconds": max(float((stop - cursor) / 1000.0), 1.0),
                "weight": max(float((stop - cursor) / 1000.0), 1.0),
            }
        )
        cursor = stop
    return number


def _assign_region_splits(
    regions: pd.DataFrame, config: ContinuousSingleChannelConfig
) -> pd.DataFrame:
    result = regions.copy()
    assignments: dict[str, str] = {}
    rng = np.random.default_rng(config.random_seed)
    for (_, _), group in result.groupby(["animal_id", "region_kind"], sort=True):
        if len(group) < 3:
            raise ValueError("each animal/region-kind stratum needs at least three groups")
        records = group.to_dict("records")
        jitter = rng.random(len(records))
        order = sorted(
            range(len(records)),
            key=lambda index: (-float(records[index]["weight"]), float(jitter[index])),
        )
        total = float(sum(float(record["weight"]) for record in records))
        targets = config.fractions * total
        counts = np.zeros(3, dtype=float)
        for rank, record_index in enumerate(order):
            record = records[record_index]
            split_index = rank if rank < 3 else int(
                np.argmin(counts / np.maximum(targets, 1.0) + rng.random(3) * 1e-9)
            )
            assignments[str(record["region_id"])] = SPLITS[split_index]
            counts[split_index] += float(record["weight"])
    result["split"] = result["region_id"].map(assignments)
    return result


def _region_positions(centers: np.ndarray, regions: pd.DataFrame) -> np.ndarray:
    starts = regions["start_time_ms"].to_numpy(float)
    ends = regions["end_time_ms"].to_numpy(float)
    positions = np.searchsorted(ends, centers, side="right")
    positions = np.minimum(positions, len(regions) - 1)
    valid = (starts[positions] <= centers + 1e-6) & (centers <= ends[positions] + 1e-6)
    if not valid.all():
        raise ValueError("temporal regions do not cover the complete session timeline")
    return positions


def _expand_channel_events(
    events: pd.DataFrame,
    regions: pd.DataFrame,
    event_lookup: Mapping[str, str],
    config: ContinuousSingleChannelConfig,
) -> pd.DataFrame:
    split_lookup = regions.set_index("region_id")["split"].astype(str).to_dict()
    rows = []
    excluded = {
        (str(animal).upper(), str(channel).upper())
        for animal, channel in config.excluded_animal_channels
    }
    for event in events.to_dict("records"):
        region_id = event_lookup[str(event["event_id"])]
        for channel in sorted(event["_channel_set"]):
            if (str(event["animal_id"]).upper(), str(channel).upper()) in excluded:
                continue
            rows.append(
                {
                    "animal_id": str(event["animal_id"]),
                    "session_id": str(event["session_id"]),
                    "channel_name": str(channel),
                    "event_id": str(event["event_id"]),
                    "start_time_ms": float(event["start_time_ms"]),
                    "end_time_ms": float(event["end_time_ms"]),
                    "duration_s": float(
                        (event["end_time_ms"] - event["start_time_ms"]) / 1000.0
                    ),
                    "region_id": region_id,
                    "split": split_lookup[region_id],
                }
            )
    return pd.DataFrame(rows)


def _annotate_continuity_and_context(
    frame: pd.DataFrame, config: ContinuousSingleChannelConfig
) -> pd.DataFrame:
    result = frame.sort_values(
        ["animal_id", "session_id", "channel_name", "start_time_ms"], kind="stable"
    ).reset_index(drop=True)
    continuity = np.empty(len(result), dtype=object)
    position = np.empty(len(result), dtype=np.int64)
    full_context = np.zeros(len(result), dtype=bool)
    for (animal, session, channel), group in result.groupby(
        ["animal_id", "session_id", "channel_name"], sort=False
    ):
        indices = group.index.to_numpy(np.int64)
        starts = group["start_time_ms"].to_numpy(float)
        expected = group["stride_seconds"].to_numpy(float)[:-1] * 1000.0
        breaks = np.r_[
            0,
            np.flatnonzero(~np.isclose(np.diff(starts), expected, rtol=0.05, atol=100.0)) + 1,
            len(group),
        ]
        for segment_number, (left, right) in enumerate(zip(breaks[:-1], breaks[1:], strict=True), start=1):
            segment_indices = indices[left:right]
            segment_splits = result.loc[segment_indices, "split"].astype(str).to_numpy()
            for local_position, row_index in enumerate(segment_indices):
                continuity[row_index] = f"{animal}:{session}:{channel}:segment{segment_number:04d}"
                position[row_index] = local_position
                context_left = max(0, local_position - config.context_half_width)
                context_right = min(len(segment_indices), local_position + config.context_half_width + 1)
                full_context[row_index] = bool(
                    np.all(segment_splits[context_left:context_right] == segment_splits[local_position])
                )
    result["continuity_id"] = continuity
    result["continuity_position"] = position
    result["full_split_context"] = full_context
    result["sample_id"] = [
        f"{animal}:{session}:{channel}:{float(start):.3f}"
        for animal, session, channel, start in result[
            ["animal_id", "session_id", "channel_name", "start_time_ms"]
        ].itertuples(index=False, name=None)
    ]
    return result


def _cross_split_overlap_flags(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, int]:
    """Return rows and pair count participating in cross-split raw overlap."""

    required = {"animal_id", "session_id", "channel_name", "start_time_ms", "end_time_ms", "split"}
    _require_columns(frame, required, "split-overlap audit frame")
    working = frame.reset_index(drop=True)
    flags = np.zeros(len(working), dtype=bool)
    pair_count = 0
    for _, group in working.groupby(
        ["animal_id", "session_id", "channel_name"], sort=False
    ):
        ordered = group.sort_values("start_time_ms", kind="stable")
        indices = ordered.index.to_numpy(np.int64)
        starts = ordered["start_time_ms"].to_numpy(float)
        ends = ordered["end_time_ms"].to_numpy(float)
        splits = ordered["split"].astype(str).to_numpy()
        active: list[int] = []
        for local_index, current_start in enumerate(starts):
            active = [
                previous
                for previous in active
                if ends[previous] > current_start + 1e-9
            ]
            for previous in active:
                if splits[previous] == splits[local_index]:
                    continue
                flags[indices[previous]] = True
                flags[indices[local_index]] = True
                pair_count += 1
            active.append(local_index)
    return flags, pair_count


def _annotate_cross_split_overlaps(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    flags, _ = _cross_split_overlap_flags(result)
    result["cross_split_raw_overlap"] = flags
    return result


def _count_cross_split_overlap_pairs(frame: pd.DataFrame) -> int:
    """Count unique overlapping window pairs assigned to different splits."""

    if frame.empty:
        return 0
    _, pairs = _cross_split_overlap_flags(frame)
    return int(pairs)


def _select_supervised_centers(
    full_index: pd.DataFrame, config: ContinuousSingleChannelConfig
) -> pd.DataFrame:
    eligible = full_index.loc[
        full_index["channel_valid"] & full_index["label"].isin([0, 1])
    ].copy()
    if config.purge_cross_split_windows:
        eligible = eligible.loc[
            eligible["full_split_context"]
            & ~eligible["cross_split_raw_overlap"]
        ]
    selected: list[np.ndarray] = []
    for (_, _, region_kind, label), group in eligible.groupby(
        ["region_id", "channel_name", "region_kind", "label"], sort=True
    ):
        ordered = group.sort_values("start_time_ms", kind="stable")
        if int(label) == 1:
            cap = config.maximum_positive_centers_per_group_channel
        elif str(region_kind) == "event_context":
            cap = config.maximum_event_negative_centers_per_group_channel
        else:
            cap = config.maximum_background_centers_per_group_channel
        if cap is None or len(ordered) <= cap:
            selected.append(ordered.index.to_numpy(np.int64))
        else:
            positions = np.linspace(0, len(ordered) - 1, cap, dtype=np.int64)
            selected.append(ordered.index.to_numpy(np.int64)[positions])
    if not selected:
        raise ValueError("no supervised centers were selected")
    result = full_index.loc[np.sort(np.concatenate(selected))].copy()
    return result.reset_index(drop=True)


def _fit_training_baseline(
    full_index: pd.DataFrame, centers: pd.DataFrame, samples_per_window: int = 64
) -> dict[str, Any]:
    train = centers.loc[
        centers["split"].eq("train")
        & centers["label"].eq(0)
        & centers["region_kind"].eq("background")
    ]
    if train.empty:
        raise ValueError("training split has no background centers")
    blocks: dict[tuple[str, str], list[np.ndarray]] = {}
    arrays: dict[str, np.ndarray] = {}
    for row in train.itertuples(index=False):
        root = str(row.filtered_cache_dir)
        if root not in arrays:
            arrays[root] = np.load(Path(root) / "signals.npy", mmap_mode="r")
        values = np.asarray(
            arrays[root][int(row.filtered_local_index), int(row.channel_index)],
            dtype=np.float32,
        )
        positions = np.linspace(0, len(values) - 1, min(samples_per_window, len(values)), dtype=np.int64)
        blocks.setdefault((str(row.animal_id), str(row.channel_name)), []).append(values[positions])
    rows = []
    for animal in sorted(full_index["animal_id"].astype(str).unique()):
        for channel in CHANNEL_NAMES:
            animal_channel = full_index.loc[
                full_index["animal_id"].eq(animal) & full_index["channel_name"].eq(channel)
            ]
            if animal_channel.empty:
                continue
            values = np.concatenate(blocks[(animal, channel)]).astype(np.float64, copy=False)
            center = float(np.median(values))
            scale = float(1.4826 * np.median(np.abs(values - center)))
            if not np.isfinite(scale) or scale <= 1e-6:
                scale = max(float(np.std(values)), 1e-6)
            rows.append(
                {
                    "animal_id": animal,
                    "channel_name": channel,
                    "center": center,
                    "scale": scale,
                    "sampled_values": int(len(values)),
                }
            )
    return {
        "schema_version": 1,
        "fit_split": "train",
        "fit_class": "non-seizure_background_only",
        "unit": "a.u.",
        "method": "per_recording_channel_median_and_1.4826_MAD",
        "identity_is_model_input": False,
        "channels": rows,
    }


def _validated_events(
    event_components: str | Path | pd.DataFrame, animals: Iterable[str]
) -> pd.DataFrame:
    events = _read_table(event_components)
    _require_columns(
        events,
        {"animal_id", "session_id", "event_id", "start_time_ms", "end_time_ms", "channels"},
        "event components",
    )
    events = events.loc[events["animal_id"].astype(str).isin(tuple(animals))].copy()
    for column in ("animal_id", "session_id", "event_id"):
        events[column] = events[column].astype(str)
    events["start_time_ms"] = pd.to_numeric(events["start_time_ms"], errors="raise")
    events["end_time_ms"] = pd.to_numeric(events["end_time_ms"], errors="raise")
    if events.empty or events["end_time_ms"].le(events["start_time_ms"]).any():
        raise ValueError("event components must be non-empty positive intervals")
    events["_channel_set"] = events["channels"].map(_parse_channels)
    return events.sort_values(
        ["animal_id", "session_id", "start_time_ms", "end_time_ms"], kind="stable"
    ).reset_index(drop=True)


def _windows_to_continuous(windows: np.ndarray, stride_samples: int) -> np.ndarray:
    first = np.asarray(windows[0], dtype=np.float32)
    if len(windows) == 1:
        return first.copy()
    tails = np.asarray(windows[1:, :, -stride_samples:], dtype=np.float32)
    return np.concatenate(
        [first, tails.transpose(1, 0, 2).reshape(windows.shape[1], -1)], axis=1
    )


def _continuous_to_windows(
    continuous: np.ndarray, count: int, stride_samples: int, window_samples: int
) -> np.ndarray:
    output = np.empty((count, continuous.shape[0], window_samples), dtype=np.float32)
    for index in range(count):
        left = index * stride_samples
        output[index] = continuous[:, left : left + window_samples]
    return output


def _regularize_grid_timestamps(
    known_positions: np.ndarray,
    known_timestamps_ms: np.ndarray,
    output_positions: np.ndarray,
    stride_seconds: float,
) -> tuple[np.ndarray, dict[str, float]]:
    positions = np.asarray(known_positions, dtype=float)
    timestamps = np.asarray(known_timestamps_ms, dtype=float)
    if len(positions) != len(timestamps) or not len(positions):
        raise ValueError("timestamp anchors must have equal non-zero length")
    if len(positions) == 1:
        slope = stride_seconds * 1000.0
        intercept = float(timestamps[0] - slope * positions[0])
    else:
        centered_positions = positions - float(np.mean(positions))
        centered_timestamps = timestamps - float(np.mean(timestamps))
        slope = float(
            np.dot(centered_positions, centered_timestamps)
            / np.dot(centered_positions, centered_positions)
        )
        intercept = float(np.median(timestamps - slope * positions))
    nominal_stride_ms = stride_seconds * 1000.0
    if not 0.8 * nominal_stride_ms <= slope <= 1.2 * nominal_stride_ms:
        raise ValueError(f"implausible fitted timestamp stride {slope:g} ms")
    residual = timestamps - (intercept + slope * positions)
    output = intercept + slope * np.asarray(output_positions, dtype=float)
    return output, {
        "fitted_stride_ms": float(slope),
        "timestamp_fit_rmse_ms": float(np.sqrt(np.mean(residual**2))),
        "maximum_absolute_timestamp_residual_ms": float(np.max(np.abs(residual))),
    }


def _expected_source_window_count(
    source: Any,
    *,
    sampling_rate_hz: float,
    window_samples: int,
    stride_samples: int,
) -> int:
    if source is None or not Path(str(source)).is_file():
        return 0
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(str(source))
    native_rows = int(parquet.metadata.num_rows)
    # The original cache builder used resample_poly and a near-1000 Hz source.
    # Its integer down factor can be recovered from the declared output rate.
    down = max(1, int(round(1000.0 / sampling_rate_hz)))
    resampled_rows = (native_rows + down - 1) // down
    return max(0, (resampled_rows - window_samples) // stride_samples + 1)


def _rebuild_grid_from_source_parquet(
    source: str | Path,
    *,
    target_rate_hz: float,
    window_seconds: float,
    stride_seconds: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import pyarrow.parquet as pq
    from scipy.signal import resample_poly

    table = pq.read_table(source, columns=["timestamp_ms", "ch7", "ch8"])
    values = table.to_pandas()
    timestamps = values["timestamp_ms"].to_numpy(np.float64)
    signals = values[["ch7", "ch8"]].to_numpy(np.float32).T
    native_rate = 1000.0 / max(float(np.median(np.diff(timestamps))), 1e-6)
    down = max(1, int(round(native_rate / target_rate_hz)))
    actual_rate = native_rate / down
    if not np.isclose(actual_rate, target_rate_hz, rtol=0.01, atol=0.1):
        raise ValueError(
            f"raw source rate {actual_rate:g} does not match target {target_rate_hz:g}"
        )
    if down > 1:
        signals = resample_poly(signals, up=1, down=down, axis=1).astype(np.float32)
        timestamps = timestamps[::down][: signals.shape[1]]
    window_samples = int(round(window_seconds * actual_rate))
    stride_samples = int(round(stride_seconds * actual_rate))
    count = max(0, (signals.shape[1] - window_samples) // stride_samples + 1)
    start_indices = np.arange(count, dtype=np.int64) * stride_samples
    windows = np.empty((count, 2, window_samples), dtype=np.float32)
    for index, start in enumerate(start_indices):
        windows[index] = signals[:, int(start) : int(start) + window_samples]
    regularized_starts, timestamp_diagnostics = _regularize_grid_timestamps(
        np.arange(count, dtype=float),
        timestamps[start_indices],
        np.arange(count, dtype=float),
        stride_seconds,
    )
    return windows, regularized_starts, {
        "source_parquet_edge_fallback": True,
        **timestamp_diagnostics,
    }


def _common_cache_root(value: str | Path | pd.DataFrame) -> Path | None:
    if isinstance(value, pd.DataFrame):
        return None
    path = Path(value)
    return path.parent.resolve() if path.is_file() or path.suffix else path.resolve()


def _resolve_cache_dir(value: Any, fallback_root: Path | None) -> Path:
    candidate = Path(str(value))
    if candidate.is_dir():
        return candidate.resolve()
    if fallback_root is not None:
        fallback = fallback_root / candidate.name
        if fallback.is_dir():
            return fallback.resolve()
    raise FileNotFoundError(f"cannot resolve cache directory {value!r}")


def _read_table(value: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    path = Path(value)
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


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
    "ContinuousSingleChannelConfig",
    "ContinuousSingleChannelDataset",
    "audit_continuous_single_channel_protocol",
    "build_continuous_single_channel_protocol",
    "materialize_complete_filtered_cache",
    "recover_complete_window_grid",
]
