from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from mouse_sleep.epilepsy.morphology import _feature_batch
from mouse_sleep.epilepsy.signal_processing import (
    EEGFilterConfig,
    extract_filtered_windows,
    filter_continuous_eeg,
    reconstruct_continuous_cache,
)


def extract_baseline_features(
    cache_manifest: str | Path | pd.DataFrame,
    filter_config: EEGFilterConfig,
    maximum_windows_per_animal: int = 2000,
    maximum_caches_per_animal: int = 20,
    seizure_guard_seconds: float = 10.0,
    random_seed: int = 20260811,
) -> pd.DataFrame:
    """Sample non-overlapping baseline windows away from every seizure annotation."""

    manifest = (
        pd.read_csv(cache_manifest)
        if not isinstance(cache_manifest, pd.DataFrame)
        else cache_manifest.copy()
    )
    manifest = manifest.sort_values(
        ["animal_id", "session_id", "shard"], kind="stable"
    ).drop_duplicates("cache_dir")
    selected_rows: list[dict[str, Any]] = []
    for animal, animal_rows in manifest.groupby("animal_id", sort=True):
        candidates_by_cache: list[tuple[pd.Series, np.ndarray]] = []
        for _, row in animal_rows.iterrows():
            root = Path(str(row["cache_dir"]))
            labels = np.asarray(np.load(root / "labels.npy", mmap_mode="r"))
            stride_seconds = float(row.get("stride_seconds", 1.0))
            guard = int(math.ceil(seizure_guard_seconds / stride_seconds))
            positive = labels == 1
            excluded = binary_dilation(
                positive,
                structure=np.ones(2 * guard + 1, dtype=bool),
            )
            safe = np.flatnonzero((labels == 0) & ~excluded)
            nonoverlap_step = max(
                1, int(math.ceil(float(row["window_seconds"]) / stride_seconds))
            )
            safe = safe[safe % nonoverlap_step == 0]
            if len(safe):
                candidates_by_cache.append((row, safe))
        if not candidates_by_cache:
            continue
        cache_positions = np.linspace(
            0,
            len(candidates_by_cache) - 1,
            min(maximum_caches_per_animal, len(candidates_by_cache)),
            dtype=np.int64,
        )
        chosen_caches = [candidates_by_cache[index] for index in cache_positions]
        per_cache = int(math.ceil(maximum_windows_per_animal / len(chosen_caches)))
        rng = np.random.default_rng(random_seed + _stable_text_seed(str(animal)))
        animal_selections: list[dict[str, Any]] = []
        for row, safe in chosen_caches:
            count = min(per_cache, len(safe))
            chosen = np.sort(rng.choice(safe, count, replace=False))
            for local_index in chosen:
                animal_selections.append(
                    {
                        "animal_id": str(animal),
                        "session_id": str(row["session_id"]),
                        "shard": int(row["shard"]),
                        "cache_dir": str(row["cache_dir"]),
                        "cache_local_index": int(local_index),
                        "sampling_rate_hz": float(row["sampling_rate_hz"]),
                    }
                )
        if len(animal_selections) > maximum_windows_per_animal:
            keep = np.sort(
                rng.choice(
                    len(animal_selections),
                    maximum_windows_per_animal,
                    replace=False,
                )
            )
            animal_selections = [animal_selections[index] for index in keep]
        selected_rows.extend(animal_selections)
    selection = pd.DataFrame(selected_rows)
    if selection.empty:
        raise ValueError("No guarded baseline windows were available")

    feature_frames: list[pd.DataFrame] = []
    for cache_dir, group in selection.groupby("cache_dir", sort=False):
        root = Path(cache_dir)
        signals = np.load(root / "signals.npy", mmap_mode="r")
        metadata = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        rate = float(metadata["sampling_rate_hz"])
        stride_samples = int(round(rate * float(metadata["stride_seconds"])))
        continuous = reconstruct_continuous_cache(signals, stride_samples)
        filtered = filter_continuous_eeg(continuous, rate, filter_config)
        indices = group["cache_local_index"].to_numpy(np.int64)
        filtered_windows = extract_filtered_windows(
            filtered, indices, stride_samples, signals.shape[-1]
        )
        raw_windows = np.asarray(signals[indices], dtype=np.float32)
        features = pd.DataFrame(_feature_batch(raw_windows, rate, filtered_windows))
        features.index = group.index
        feature_frames.append(features)
    values = pd.concat(feature_frames).sort_index()
    result = pd.concat(
        [selection.reset_index(drop=True), values.reset_index(drop=True)], axis=1
    )
    result["state"] = "baseline"
    return result


def add_baseline_relative_features(
    seizure_features: pd.DataFrame,
    baseline_features: pd.DataFrame,
) -> pd.DataFrame:
    """Express amplitude and physiology features relative to each mouse baseline."""

    result = seizure_features.copy()
    candidates = [
        "aggregate_robust_amplitude",
        "aggregate_line_length",
        "aggregate_log_rms",
        "aggregate_hjorth_mobility",
        "aggregate_hjorth_complexity",
        "aggregate_excursion_rate_hz",
        "aggregate_peak_frequency_hz",
        "aggregate_spectral_entropy",
        "relative_delta",
        "relative_theta",
        "relative_beta",
        "relative_low_gamma_clean",
        "relative_high_gamma",
    ]
    for channel in ("ch7", "ch8"):
        candidates.extend(
            [
                f"{channel}_log_robust_amplitude",
                f"{channel}_log_line_length",
                f"{channel}_log_rms",
                f"{channel}_hjorth_mobility",
                f"{channel}_hjorth_complexity",
                f"{channel}_excursion_rate_hz",
                f"{channel}_peak_frequency_hz",
                f"{channel}_spectral_entropy",
                f"{channel}_relative_delta",
                f"{channel}_relative_theta",
                f"{channel}_relative_beta",
                f"{channel}_relative_low_gamma_clean",
                f"{channel}_relative_high_gamma",
            ]
        )
    available = [
        column
        for column in candidates
        if column in result and column in baseline_features
    ]
    for column in available:
        relative = np.full(len(result), np.nan, dtype=float)
        for animal, group in result.groupby("animal_id", sort=False):
            baseline = pd.to_numeric(
                baseline_features.loc[
                    baseline_features["animal_id"].astype(str).eq(str(animal)), column
                ],
                errors="coerce",
            ).to_numpy(float)
            baseline = baseline[np.isfinite(baseline)]
            if not len(baseline):
                continue
            median = float(np.median(baseline))
            q25, q75 = np.quantile(baseline, [0.25, 0.75])
            scale = max(float(q75 - q25), 1e-6)
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
            relative[group.index.to_numpy(np.int64)] = (values - median) / scale
        result[f"baseline_z__{column}"] = relative
    amplitude_baseline = (
        baseline_features.groupby("animal_id")["aggregate_robust_amplitude"]
        .median()
        .to_dict()
    )
    result["amplitude_vs_baseline"] = [
        float(value) / max(float(amplitude_baseline.get(animal, np.nan)), 1e-12)
        for animal, value in zip(
            result["animal_id"].astype(str), result["aggregate_robust_amplitude"]
        )
    ]
    return result


def summarize_baseline_seizure_amplitude(
    seizure_features: pd.DataFrame,
    baseline_features: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for animal in sorted(seizure_features["animal_id"].astype(str).unique()):
        seizure = seizure_features.loc[
            seizure_features["animal_id"].astype(str).eq(animal),
            "aggregate_robust_amplitude",
        ].to_numpy(float)
        baseline = baseline_features.loc[
            baseline_features["animal_id"].astype(str).eq(animal),
            "aggregate_robust_amplitude",
        ].to_numpy(float)
        rows.append(
            {
                "animal_id": animal,
                "baseline_windows": int(len(baseline)),
                "seizure_windows": int(len(seizure)),
                "baseline_amplitude_median": float(np.median(baseline)),
                "seizure_amplitude_median": float(np.median(seizure)),
                "median_seizure_baseline_ratio": float(
                    np.median(seizure) / max(np.median(baseline), 1e-12)
                ),
                "seizure_above_baseline_p95_fraction": float(
                    np.mean(seizure > np.quantile(baseline, 0.95))
                ),
            }
        )
    return pd.DataFrame(rows)


def _stable_text_seed(value: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(value))


__all__ = [
    "add_baseline_relative_features",
    "extract_baseline_features",
    "summarize_baseline_seizure_amplitude",
]
