from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from mouse_sleep.epilepsy.morphology import _feature_batch
from mouse_sleep.epilepsy.temporal_state_taxonomy import EXPLAINABLE_FEATURES


def extract_native_state_features(
    signals: np.ndarray,
    window_metadata: pd.DataFrame,
    baseline_features: str | Path | pd.DataFrame,
) -> pd.DataFrame:
    """Extract physiology features from the same filtered native windows."""

    values = np.asarray(signals)
    metadata = window_metadata.reset_index(drop=True).copy()
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(f"Expected [windows, 2, samples], received {values.shape}")
    if len(values) != len(metadata):
        raise ValueError("Signal and metadata row counts do not match")
    if "sampling_rate_hz" not in metadata or "animal_id" not in metadata:
        raise ValueError("Window metadata needs sampling_rate_hz and animal_id")

    feature_frames: list[pd.DataFrame] = []
    for rate, group in metadata.groupby("sampling_rate_hz", sort=True):
        indices = group.index.to_numpy(np.int64)
        batch = np.asarray(values[indices], dtype=np.float32)
        features = _feature_batch(batch, float(rate), filtered_signals=batch)
        feature_frame = pd.DataFrame(features, index=indices)
        feature_frames.append(feature_frame)
    feature_values = pd.concat(feature_frames).sort_index()
    duplicate = set(metadata.columns) & set(feature_values.columns)
    for column in duplicate:
        metadata = metadata.drop(columns=column)
    result = pd.concat([metadata, feature_values], axis=1)

    baseline = _read_baseline_features(baseline_features)
    reference = (
        baseline.groupby("animal_id", as_index=False)[
            ["aggregate_log_robust_amplitude", "aggregate_log_line_length"]
        ]
        .median()
        .rename(
            columns={
                "aggregate_log_robust_amplitude": "reference_log_robust_amplitude",
                "aggregate_log_line_length": "reference_log_line_length",
            }
        )
    )
    result = result.merge(reference, on="animal_id", how="left", validate="many_to_one")
    if result[
        ["reference_log_robust_amplitude", "reference_log_line_length"]
    ].isna().any().any():
        missing = sorted(
            result.loc[
                result["reference_log_robust_amplitude"].isna(), "animal_id"
            ].astype(str).unique()
        )
        raise ValueError(f"Missing unlabelled reference features for mice {missing}")
    result["amplitude_vs_reference_log"] = (
        result["aggregate_log_robust_amplitude"]
        - result["reference_log_robust_amplitude"]
    )
    result["line_length_vs_reference_log"] = (
        result["aggregate_log_line_length"] - result["reference_log_line_length"]
    )
    result["amplitude_vs_reference_ratio"] = np.exp(
        np.clip(result["amplitude_vs_reference_log"], -20.0, 20.0)
    )
    result["line_length_vs_reference_ratio"] = np.exp(
        np.clip(result["line_length_vs_reference_log"], -20.0, 20.0)
    )
    return result


def _read_baseline_features(value: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    else:
        path = Path(value)
        if path.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
    required = {
        "animal_id",
        "aggregate_log_robust_amplitude",
        "aggregate_log_line_length",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Baseline feature table is missing {sorted(missing)}")
    return frame


def explainable_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in EXPLAINABLE_FEATURES:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().mean() >= 0.50:
            columns.append(column)
    return columns


def explainable_feature_blocks(
    columns: Iterable[str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for column in columns:
        lower = column.lower()
        if any(
            token in lower
            for token in (
                "channel_correlation",
                "spectral_cosine_similarity",
                "amplitude_log_ratio",
                "line_length_log_ratio",
            )
        ):
            block = "bilateral"
        elif any(
            token in lower
            for token in (
                "relative_",
                "frequency",
                "spectral_",
                "theta_delta",
                "high_low",
            )
        ):
            block = "frequency"
        elif any(
            token in lower
            for token in (
                "amplitude",
                "line_length",
                "rms",
                "peak_to_peak",
                "total_power",
            )
        ):
            block = "amplitude"
        else:
            block = "shape"
        mapping[str(column)] = block
    return mapping


__all__ = [
    "explainable_feature_blocks",
    "explainable_feature_columns",
    "extract_native_state_features",
]
