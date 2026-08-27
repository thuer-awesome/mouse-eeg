"""MNE-backed CHB-MIT EDF inspection and fixed-window indexing."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from mouse_sleep.epilepsy.replication.chbmit import (
    CHBMIT_SAMPLING_RATE,
    DEFAULT_BIPOLAR_CHANNELS_18,
)


def _require_mne():
    cache = Path(tempfile.gettempdir()) / "mouse_sleep_matplotlib_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    try:
        import mne
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "CHB-MIT EDF support requires the 'chbmit' extra: "
            "pip install -e .[chbmit]"
        ) from error
    return mne


def normalize_chbmit_channel_name(name: str) -> str:
    """Normalize EDF labels while preserving the bipolar montage identity."""

    normalized = name.strip().upper().replace(" ", "")
    if normalized.startswith("EEG"):
        normalized = normalized[3:]
    normalized = re.sub(r"-REF$", "", normalized)
    # MNE appends -0/-1 when an EDF contains duplicate names (notably T8-P8).
    normalized = re.sub(r"-(?:[0-9]+)$", "", normalized)
    return normalized


def resolve_chbmit_channels(
    available: Sequence[str],
    required: Sequence[str] = DEFAULT_BIPOLAR_CHANNELS_18,
) -> tuple[list[int], list[str]]:
    """Resolve one deterministic EDF index per required canonical channel."""

    by_name: dict[str, list[int]] = {}
    for index, name in enumerate(available):
        by_name.setdefault(normalize_chbmit_channel_name(name), []).append(index)
    picks: list[int] = []
    missing: list[str] = []
    for channel in required:
        indices = by_name.get(normalize_chbmit_channel_name(channel), [])
        if not indices:
            missing.append(channel)
        else:
            picks.append(indices[0])
    return picks, missing


def inspect_chbmit_edf(
    path: str | Path,
    *,
    required_channels: Sequence[str] = DEFAULT_BIPOLAR_CHANNELS_18,
) -> dict[str, Any]:
    """Read an EDF header and report whether the canonical montage is usable."""

    mne = _require_mne()
    edf_path = Path(path).resolve()
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    picks, missing = resolve_chbmit_channels(raw.ch_names, required_channels)
    sampling_rate = float(raw.info["sfreq"])
    return {
        "path": str(edf_path),
        "sampling_rate_hz": sampling_rate,
        "n_times": int(raw.n_times),
        "duration_seconds": float(raw.n_times / sampling_rate),
        "channels": list(raw.ch_names),
        "required_channels": list(required_channels),
        "resolved_channel_indices": picks,
        "missing_required_channels": missing,
        "required_montage_available": not missing,
        "sampling_rate_matches_256_hz": bool(
            np.isclose(sampling_rate, CHBMIT_SAMPLING_RATE, rtol=0.0, atol=1e-9)
        ),
    }


def read_chbmit_window(
    path: str | Path,
    start_sample: int,
    stop_sample: int,
    *,
    required_channels: Sequence[str] = DEFAULT_BIPOLAR_CHANNELS_18,
    expected_sampling_rate_hz: float = CHBMIT_SAMPLING_RATE,
    output_microvolts: bool = True,
) -> np.ndarray:
    """Load one fixed EDF window as ``float32 [channels, samples]``."""

    if start_sample < 0 or stop_sample <= start_sample:
        raise ValueError("start_sample and stop_sample define an invalid window")
    mne = _require_mne()
    raw = mne.io.read_raw_edf(Path(path).resolve(), preload=False, verbose="ERROR")
    sampling_rate = float(raw.info["sfreq"])
    if not np.isclose(
        sampling_rate, expected_sampling_rate_hz, rtol=0.0, atol=1e-9
    ):
        raise ValueError(
            f"Expected {expected_sampling_rate_hz} Hz, received {sampling_rate} Hz"
        )
    if stop_sample > raw.n_times:
        raise ValueError(f"Window stop {stop_sample} exceeds EDF length {raw.n_times}")
    picks, missing = resolve_chbmit_channels(raw.ch_names, required_channels)
    if missing:
        raise ValueError(f"EDF is missing required channels: {', '.join(missing)}")
    values = raw.get_data(picks=picks, start=start_sample, stop=stop_sample)
    if output_microvolts:
        values = values * 1e6
    values = np.asarray(values, dtype=np.float32)
    expected_shape = (len(required_channels), stop_sample - start_sample)
    if values.shape != expected_shape or not np.isfinite(values).all():
        raise ValueError(
            f"Invalid EDF window: expected {expected_shape}, received {values.shape}"
        )
    return values


def seizure_overlap_fraction(
    intervals: Iterable[dict[str, float]], start_seconds: float, stop_seconds: float
) -> float:
    """Return the fraction of a window covered by annotated seizure time."""

    if stop_seconds <= start_seconds:
        raise ValueError("stop_seconds must be greater than start_seconds")
    overlap = 0.0
    for interval in intervals:
        seizure_start = float(interval["start_seconds"])
        seizure_stop = float(interval["end_seconds"])
        overlap += max(0.0, min(stop_seconds, seizure_stop) - max(start_seconds, seizure_start))
    return float(min(1.0, overlap / (stop_seconds - start_seconds)))


def label_chbmit_window(
    intervals: Iterable[dict[str, float]],
    start_seconds: float,
    stop_seconds: float,
    *,
    policy: str = "center",
) -> tuple[int, float]:
    """Label a window with an explicit, reproducible boundary policy."""

    intervals = list(intervals)
    fraction = seizure_overlap_fraction(intervals, start_seconds, stop_seconds)
    if policy == "any_overlap":
        label = int(fraction > 0.0)
    elif policy == "majority_overlap":
        label = int(fraction >= 0.5)
    elif policy == "center":
        center = (start_seconds + stop_seconds) / 2.0
        label = int(
            any(
                float(item["start_seconds"]) <= center < float(item["end_seconds"])
                for item in intervals
            )
        )
    else:
        raise ValueError("policy must be center, any_overlap, or majority_overlap")
    return label, fraction


def build_chbmit_window_manifest(
    file_manifest: str | Path | pd.DataFrame,
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    window_seconds: float = 4.0,
    stride_seconds: float = 4.0,
    label_policy: str = "center",
    required_channels: Sequence[str] = DEFAULT_BIPOLAR_CHANNELS_18,
    max_files: int | None = None,
) -> pd.DataFrame:
    """Index complete fixed windows without loading all EEG samples into RAM."""

    if window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("window_seconds and stride_seconds must be positive")
    frame = pd.read_csv(file_manifest) if not isinstance(file_manifest, pd.DataFrame) else file_manifest.copy()
    required = {"relative_path", "patient_id", "case_id", "seizure_intervals_json"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"CHB-MIT file manifest missing columns: {', '.join(sorted(missing_columns))}"
        )
    if max_files is not None:
        if max_files < 1:
            raise ValueError("max_files must be positive")
        frame = frame.head(max_files)
    root = Path(dataset_root).resolve()
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        edf_path = root / Path(str(record["relative_path"]))
        if not edf_path.is_file():
            raise FileNotFoundError(f"CHB-MIT EDF not found: {edf_path}")
        inspection = inspect_chbmit_edf(edf_path, required_channels=required_channels)
        if not inspection["sampling_rate_matches_256_hz"]:
            raise ValueError(f"Unexpected sampling rate in {edf_path}")
        if not inspection["required_montage_available"]:
            missing = ", ".join(inspection["missing_required_channels"])
            raise ValueError(f"Required montage is unavailable in {edf_path}: {missing}")
        sampling_rate = float(inspection["sampling_rate_hz"])
        window_samples = int(round(window_seconds * sampling_rate))
        stride_samples = int(round(stride_seconds * sampling_rate))
        intervals = json.loads(str(record["seizure_intervals_json"]))
        for start in range(0, int(inspection["n_times"]) - window_samples + 1, stride_samples):
            stop = start + window_samples
            start_seconds = start / sampling_rate
            stop_seconds = stop / sampling_rate
            label, overlap = label_chbmit_window(
                intervals, start_seconds, stop_seconds, policy=label_policy
            )
            row = {
                "relative_path": record["relative_path"],
                "case_id": record["case_id"],
                "patient_id": record["patient_id"],
                "start_sample": start,
                "stop_sample": stop,
                "start_seconds": start_seconds,
                "stop_seconds": stop_seconds,
                "sampling_rate_hz": sampling_rate,
                "label": label,
                "seizure_overlap_fraction": overlap,
                "label_policy": label_policy,
                "n_channels": len(required_channels),
            }
            if "split" in record:
                row["split"] = record["split"]
            if "fold" in record:
                row["fold"] = record["fold"]
            rows.append(row)
    result = pd.DataFrame(rows)
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".parquet":
        result.to_parquet(destination, index=False)
    elif destination.suffix.lower() == ".csv":
        result.to_csv(destination, index=False, encoding="utf-8-sig")
    else:
        raise ValueError("output_path must end in .csv or .parquet")
    return result


__all__ = [
    "build_chbmit_window_manifest",
    "inspect_chbmit_edf",
    "label_chbmit_window",
    "normalize_chbmit_channel_name",
    "read_chbmit_window",
    "resolve_chbmit_channels",
    "seizure_overlap_fraction",
]
