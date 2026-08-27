from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from aeon.transformations.collection.convolution_based import MiniRocket
from scipy.signal import resample
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler

from mouse_sleep.epilepsy.per_mouse_morphology import PER_MOUSE_FEATURES
from mouse_sleep.epilepsy.signal_processing import (
    EEGFilterConfig,
    extract_filtered_windows,
    filter_continuous_eeg,
    reconstruct_continuous_cache,
)


def discover_waveform_shape_hierarchy(
    assignments: pd.DataFrame,
    output_dir: str | Path,
    filter_config: EEGFilterConfig,
    minimum_k: int = 3,
    maximum_k: int = 8,
    rocket_kernels: int = 5000,
    bootstrap_repeats: int = 8,
    fit_windows_per_event: int = 20,
    maximum_fit_windows: int = 2000,
    random_seed: int = 20260811,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a multi-resolution per-mouse hierarchy from actual filtered waveforms."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models" / "shape"
    model_dir.mkdir(parents=True, exist_ok=True)
    grid_rows: list[dict[str, Any]] = []
    assignment_frames: list[pd.DataFrame] = []
    for animal, group in assignments.groupby("animal_id", sort=True):
        frame = group.sort_values(
            ["session_id", "event_start_ms", "start_time_ms"], kind="stable"
        ).reset_index(drop=True)
        fit_indices = _shape_event_balanced_indices(
            frame,
            fit_windows_per_event,
            maximum_fit_windows,
            random_seed,
        )
        fit_rows = frame.iloc[fit_indices].reset_index(drop=True)
        filtered = load_filtered_windows(fit_rows, filter_config)
        shape_input = normalize_waveform_shape(filtered, fit_rows)
        rocket = MiniRocket(
            n_kernels=rocket_kernels,
            n_jobs=-1,
            random_state=random_seed,
        )
        rocket_features = np.asarray(rocket.fit_transform(shape_input), dtype=np.float32)
        rocket_scaler = StandardScaler().fit(rocket_features)
        rocket_scaled = np.clip(
            rocket_scaler.transform(rocket_features), -8.0, 8.0
        )
        rocket_components = min(30, rocket_scaled.shape[1], len(rocket_scaled) - 1)
        rocket_pca = PCA(
            n_components=rocket_components,
            whiten=False,
            random_state=random_seed,
        ).fit(rocket_scaled)
        shape_embedding = rocket_pca.transform(rocket_scaled)

        physiology_columns = _physiology_columns(fit_rows)
        physiology = fit_rows[physiology_columns].replace([np.inf, -np.inf], np.nan)
        physiology_medians = physiology.median(axis=0).fillna(0.0)
        physiology_values = physiology.fillna(physiology_medians).to_numpy(float)
        physiology_scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(
            physiology_values
        )
        physiology_scaled = np.clip(
            physiology_scaler.transform(physiology_values), -8.0, 8.0
        )
        physiology_components = min(12, physiology_scaled.shape[1], len(physiology_scaled) - 1)
        physiology_pca = PCA(
            n_components=physiology_components,
            whiten=False,
            random_state=random_seed,
        ).fit(physiology_scaled)
        physiology_embedding = physiology_pca.transform(physiology_scaled)

        # Preserve the PCA variance ordering.  Standardizing every PC here
        # would give low-variance tail components the same influence as the
        # leading waveform modes and turns noise into apparent clusters.
        # Normalize only the total energy of each branch, with waveform shape
        # carrying the larger share and physiology providing independent
        # amplitude/frequency/bilateral evidence.
        shape_branch, shape_center, shape_scale = _normalize_branch(shape_embedding)
        physiology_branch, physiology_center, physiology_scale = _normalize_branch(
            physiology_embedding
        )
        combined = np.concatenate(
            [np.sqrt(0.65) * shape_branch, np.sqrt(0.35) * physiology_branch],
            axis=1,
        )
        np.save(
            output_dir / f"{animal}_multiview_embedding.npy",
            combined.astype(np.float32),
        )
        event_ids = fit_rows["event_id"].astype(str).to_numpy()
        animal_assignments = fit_rows[
            [
                "animal_id",
                "session_id",
                "event_id",
                "cache_dir",
                "cache_local_index",
                "sampling_rate_hz",
                "start_time_ms",
                "end_time_ms",
                "event_duration_s",
                "amplitude_vs_baseline",
                "ch7_valid",
                "ch8_valid",
            ]
        ].copy()
        models: dict[int, KMeans] = {}
        for k in range(minimum_k, min(maximum_k, len(combined) - 1) + 1):
            model = KMeans(
                n_clusters=k,
                n_init=30,
                random_state=random_seed,
            ).fit(combined)
            mapping = _stable_map(model.labels_)
            labels = np.asarray([mapping[int(value)] for value in model.labels_])
            model_mapping = {int(raw): int(stable) for raw, stable in mapping.items()}
            counts = np.bincount(labels, minlength=k)
            component_events = [
                len(np.unique(event_ids[labels == cluster])) for cluster in range(k)
            ]
            single_event = []
            for cluster in range(k):
                values = pd.Series(event_ids[labels == cluster]).value_counts()
                single_event.append(float(values.max() / values.sum()))
            bootstrap_values = _event_bootstrap_kmeans(
                combined,
                event_ids,
                labels,
                k,
                bootstrap_repeats,
                random_seed,
            )
            support_ok = bool(
                counts.min() >= max(10, int(np.ceil(0.02 * len(labels))))
                and min(component_events) >= 3
                and max(single_event) <= 0.50
            )
            sample = np.linspace(
                0, len(combined) - 1, min(5000, len(combined)), dtype=np.int64
            )
            grid_rows.append(
                {
                    "animal_id": str(animal),
                    "k": k,
                    "fit_windows": int(len(combined)),
                    "fit_events": int(len(np.unique(event_ids))),
                    "minimum_cluster_windows": int(counts.min()),
                    "minimum_cluster_fraction": float(counts.min() / len(labels)),
                    "minimum_cluster_events": int(min(component_events)),
                    "maximum_single_event_fraction": float(max(single_event)),
                    "silhouette": float(
                        silhouette_score(combined[sample], labels[sample])
                    ),
                    "calinski_harabasz": float(
                        calinski_harabasz_score(combined, labels)
                    ),
                    "davies_bouldin": float(davies_bouldin_score(combined, labels)),
                    "event_bootstrap_ari_median": float(
                        np.median(bootstrap_values)
                    ),
                    "event_bootstrap_ari_q25": float(
                        np.quantile(bootstrap_values, 0.25)
                    ),
                    "support_ok": support_ok,
                }
            )
            animal_assignments[f"shape_k{k}_cluster"] = labels
            animal_assignments[f"shape_k{k}_label"] = [
                f"{animal}-K{k}-C{label + 1}" for label in labels
            ]
            distance = np.linalg.norm(
                combined - model.cluster_centers_[model.labels_], axis=1
            )
            animal_assignments[f"shape_k{k}_distance"] = distance
            models[k] = model
            models[f"mapping_{k}"] = model_mapping
        assignment_frames.append(animal_assignments)
        animal_assignments[
            [f"multiview_pc{index + 1}" for index in range(min(12, combined.shape[1]))]
        ] = combined[:, : min(12, combined.shape[1])]
        joblib.dump(
            {
                "animal_id": str(animal),
                "filter_config": filter_config.to_dict(),
                "fit_indices": fit_indices,
                "rocket": rocket,
                "rocket_scaler": rocket_scaler,
                "rocket_pca": rocket_pca,
                "physiology_columns": physiology_columns,
                "physiology_medians": physiology_medians.to_dict(),
                "physiology_scaler": physiology_scaler,
                "physiology_pca": physiology_pca,
                "shape_branch_center": shape_center,
                "shape_branch_scale": shape_scale,
                "physiology_branch_center": physiology_center,
                "physiology_branch_scale": physiology_scale,
                "branch_weights": {"waveform_shape": 0.65, "physiology": 0.35},
                "models": models,
            },
            model_dir / f"{animal}_shape_hierarchy.joblib",
        )
    grid = pd.DataFrame(grid_rows)
    grid["suggested_for_visual_review"] = False
    for animal, rows in grid.groupby("animal_id"):
        supported = rows.loc[rows["support_ok"]]
        candidates = supported if not supported.empty else rows
        chosen = candidates.sort_values(
            ["silhouette", "event_bootstrap_ari_median", "k"],
            ascending=[False, False, False],
        ).index[0]
        grid.loc[chosen, "suggested_for_visual_review"] = True
    shape_assignments = pd.concat(assignment_frames, ignore_index=True)
    grid.to_csv(
        output_dir / "waveform_shape_k3_to_k8_grid.csv",
        index=False,
        encoding="utf-8-sig",
    )
    shape_assignments.to_parquet(
        output_dir / "waveform_shape_fit_assignments.parquet", index=False
    )
    return shape_assignments, grid


def load_filtered_windows(
    rows: pd.DataFrame,
    filter_config: EEGFilterConfig,
) -> np.ndarray:
    result = np.empty((len(rows), 2, 1000), dtype=np.float32)
    for cache_dir, group in rows.groupby("cache_dir", sort=False):
        root = Path(str(cache_dir))
        signals = np.load(root / "signals.npy", mmap_mode="r")
        metadata = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        rate = float(metadata["sampling_rate_hz"])
        stride = int(round(rate * float(metadata["stride_seconds"])))
        continuous = reconstruct_continuous_cache(signals, stride)
        filtered = filter_continuous_eeg(continuous, rate, filter_config)
        local_indices = group["cache_local_index"].to_numpy(np.int64)
        windows = extract_filtered_windows(
            filtered, local_indices, stride, signals.shape[-1]
        )
        result[group.index.to_numpy(np.int64)] = windows
    return result


def normalize_waveform_shape(
    windows: np.ndarray,
    rows: pd.DataFrame,
) -> np.ndarray:
    windows = _crop_partial_events(windows, rows)
    centered = windows - np.median(windows, axis=2, keepdims=True)
    q01 = np.quantile(centered, 0.01, axis=2, keepdims=True)
    q99 = np.quantile(centered, 0.99, axis=2, keepdims=True)
    scale = np.maximum(q99 - q01, 1e-6)
    normalized = np.clip(centered / scale, -4.0, 4.0)
    valid = rows[["ch7_valid", "ch8_valid"]].to_numpy(bool)
    normalized = np.where(valid[:, :, None], normalized, 0.0)
    difference = np.diff(normalized, axis=2, prepend=normalized[:, :, :1])
    return np.concatenate([normalized, difference], axis=1).astype(np.float32)


def _shape_event_balanced_indices(
    frame: pd.DataFrame,
    per_event: int,
    maximum: int,
    seed: int,
) -> np.ndarray:
    """Sample every event, using exact-overlap crops for events under 4 seconds."""

    selected_by_event: list[list[int]] = []
    for _, event in frame.groupby("event_id", sort=True):
        valid = event.loc[event["valid_channel_count"].ge(1)]
        if valid.empty:
            continue
        contained = valid.loc[valid["event_overlap_fraction"].ge(0.99)]
        if contained.empty:
            # A short annotation has no fully contained four-second cache
            # window. Keep its best-overlap window and crop to the annotation
            # boundary before waveform normalization.
            selected_by_event.append(
                [int(valid["event_overlap_fraction"].idxmax())]
            )
            continue
        count = min(per_event, len(contained))
        positions = np.linspace(0, len(contained) - 1, count, dtype=np.int64)
        selected_by_event.append(
            contained.iloc[positions].index.astype(int).tolist()
        )
    if not selected_by_event:
        raise ValueError("No valid seizure windows are available for this mouse")
    first_pass = [values[0] for values in selected_by_event]
    remaining = [value for values in selected_by_event for value in values[1:]]
    rng = np.random.default_rng(seed)
    if len(first_pass) >= maximum:
        return np.sort(rng.choice(first_pass, maximum, replace=False)).astype(np.int64)
    capacity = maximum - len(first_pass)
    if len(remaining) > capacity:
        remaining = rng.choice(remaining, capacity, replace=False).tolist()
    return np.asarray(sorted(first_pass + remaining), dtype=np.int64)


def _crop_partial_events(windows: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    result = np.asarray(windows, dtype=np.float32).copy()
    for position, row in enumerate(rows.itertuples(index=False)):
        if float(row.event_overlap_fraction) >= 0.99:
            continue
        rate = float(row.sampling_rate_hz)
        start = max(
            0,
            int(round((float(row.event_start_ms) - float(row.start_time_ms)) * rate / 1000.0)),
        )
        stop = min(
            result.shape[-1],
            int(round((float(row.event_end_ms) - float(row.start_time_ms)) * rate / 1000.0)),
        )
        if stop - start < 4:
            continue
        result[position] = resample(
            result[position, :, start:stop], result.shape[-1], axis=-1
        ).astype(np.float32)
    return result


def _physiology_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in PER_MOUSE_FEATURES:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().mean() < 0.5:
            continue
        finite = values[np.isfinite(values)]
        if len(finite) >= 3 and float(finite.quantile(0.9) - finite.quantile(0.1)) > 1e-9:
            columns.append(column)
    return columns


def _event_bootstrap_kmeans(
    values: np.ndarray,
    event_ids: np.ndarray,
    reference: np.ndarray,
    k: int,
    repeats: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed + 100 * k)
    unique_events = np.unique(event_ids)
    scores: list[float] = []
    for repeat in range(repeats):
        sampled_events = rng.choice(unique_events, len(unique_events), replace=True)
        sampled_indices = np.concatenate(
            [np.flatnonzero(event_ids == event_id) for event_id in sampled_events]
        )
        model = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=seed + repeat,
        ).fit(values[sampled_indices])
        scores.append(float(adjusted_rand_score(reference, model.predict(values))))
    return scores


def _normalize_branch(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = np.median(values, axis=0)
    centered = values - center
    scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    scale = max(scale, 1e-12)
    return centered / scale, center, scale


def _stable_map(labels: np.ndarray) -> dict[int, int]:
    counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    return {int(old): int(new) for new, old in enumerate(counts.index)}


__all__ = [
    "discover_waveform_shape_hierarchy",
    "load_filtered_windows",
    "normalize_waveform_shape",
]
