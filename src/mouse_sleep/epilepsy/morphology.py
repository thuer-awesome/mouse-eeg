from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from mouse_sleep.epilepsy.data import read_events
from mouse_sleep.epilepsy.signal_processing import (
    EEGFilterConfig,
    extract_filtered_windows,
    filter_continuous_eeg,
    reconstruct_continuous_cache,
)


BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 55.0),
    "high_gamma": (55.0, 100.0),
}

IDENTITY_COLUMNS = {
    "animal_id",
    "session_id",
    "event_id",
    "cache_dir",
    "cache_local_index",
    "shard",
    "start_time_ms",
    "end_time_ms",
    "event_start_ms",
    "event_end_ms",
    "event_duration_s",
    "event_overlap_s",
    "event_overlap_fraction",
    "event_phase",
    "active_channels",
    "is_fit_window",
}

AMPLITUDE_FEATURE_SUFFIXES = (
    "log_robust_amplitude",
    "log_rms",
    "log_peak_to_peak",
    "log_line_length",
    "log_total_power",
)


@dataclass(frozen=True)
class MorphologyConfig:
    window_seconds: float = 4.0
    morphology_stride_seconds: float = 4.0
    minimum_overlap_fraction: float = 0.75
    fit_windows_per_event: int = 32
    max_fit_windows_per_animal: int = 6000
    minimum_cluster_fraction: float = 0.005
    maximum_cluster_fraction: float = 0.03
    target_noise_fraction: float = 0.15
    pca_components: int = 12
    gmm_max_components: int = 10
    gmm_cv_folds: int = 5
    gmm_bootstrap_repeats: int = 12
    minimum_gmm_component_fraction: float = 0.05
    minimum_gmm_component_animals: int = 3
    minimum_gmm_component_events: int = 20
    minimum_component_windows_per_animal: int = 3
    maximum_component_single_animal_fraction: float = 0.50
    maximum_component_single_event_fraction: float = 0.20
    minimum_bootstrap_median_ari: float = 0.60
    minimum_bootstrap_q25_ari: float = 0.50
    minimum_assignment_probability: float = 0.60
    filter_highpass_hz: float = 0.5
    filter_lowpass_hz: float = 70.0
    filter_notch_hz: float = 50.0
    filter_notch_quality_factor: float = 30.0
    filter_bandpass_order: int = 4
    random_seed: int = 20260811


def build_event_components(events: pd.DataFrame, session_id: str) -> pd.DataFrame:
    """Merge overlapping closed channel annotations into stable event components."""

    if events.empty:
        return _empty_event_frame()
    status = events.get("status", pd.Series(index=events.index, dtype=str))
    usable = events.loc[
        status.astype(str).str.lower().eq("closed")
        & events["start_time_ms"].notna()
        & events["end_time_ms"].notna()
        & events["end_time_ms"].ge(events["start_time_ms"])
    ].copy()
    if usable.empty:
        return _empty_event_frame()
    usable = usable.sort_values(["start_time_ms", "end_time_ms"], kind="stable")
    components: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in usable.to_dict("records"):
        start = float(row["start_time_ms"])
        end = float(row["end_time_ms"])
        channel = str(row.get("channel_name", "unknown"))
        source_id = str(row.get("event_id", ""))
        if current is None or start > float(current["end_time_ms"]):
            if current is not None:
                components.append(current)
            current = {
                "start_time_ms": start,
                "end_time_ms": end,
                "channels": {channel},
                "source_event_ids": {source_id},
                "source_interval_count": 1,
            }
        else:
            current["end_time_ms"] = max(float(current["end_time_ms"]), end)
            current["channels"].add(channel)
            current["source_event_ids"].add(source_id)
            current["source_interval_count"] += 1
    if current is not None:
        components.append(current)

    rows: list[dict[str, Any]] = []
    for index, component in enumerate(components, 1):
        start = float(component["start_time_ms"])
        end = float(component["end_time_ms"])
        rows.append(
            {
                "session_id": str(session_id),
                "event_id": f"{session_id}:event{index:04d}",
                "start_time_ms": start,
                "end_time_ms": end,
                "duration_s": (end - start) / 1000.0,
                "channels": ";".join(sorted(component["channels"])),
                "source_event_ids": ";".join(sorted(component["source_event_ids"])),
                "source_interval_count": int(component["source_interval_count"]),
            }
        )
    return pd.DataFrame(rows)


def build_morphology_index(
    cache_manifest: str | Path | pd.DataFrame,
    session_manifest: str | Path | pd.DataFrame,
    config: MorphologyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Index non-overlapping seizure windows while retaining every event component."""

    cache = (
        pd.read_csv(cache_manifest)
        if not isinstance(cache_manifest, pd.DataFrame)
        else cache_manifest.copy()
    )
    sessions = (
        pd.read_csv(session_manifest)
        if not isinstance(session_manifest, pd.DataFrame)
        else session_manifest.copy()
    )
    required_cache = {
        "animal_id",
        "session_id",
        "cache_dir",
        "shard",
        "sampling_rate_hz",
        "window_seconds",
    }
    missing = required_cache - set(cache)
    if missing:
        raise ValueError(f"Cache manifest is missing columns: {sorted(missing)}")

    all_components: list[pd.DataFrame] = []
    all_windows: list[dict[str, Any]] = []
    session_rows = {str(row["session_id"]): row for row in sessions.to_dict("records")}
    for session_id, cache_rows in cache.groupby(cache["session_id"].astype(str), sort=True):
        if session_id not in session_rows:
            raise ValueError(f"Session {session_id!r} is absent from the session manifest")
        session = session_rows[session_id]
        animal_id = str(session["animal_id"])
        components = build_event_components(
            read_events(session.get("label_files", "")), session_id
        )
        if components.empty:
            continue
        components.insert(0, "animal_id", animal_id)
        all_components.append(components)
        starts = components["start_time_ms"].to_numpy(np.float64)
        ends = components["end_time_ms"].to_numpy(np.float64)
        event_ids = components["event_id"].astype(str).to_numpy()
        event_channels = components["channels"].astype(str).to_numpy()

        session_windows: list[dict[str, Any]] = []
        for cache_row in cache_rows.to_dict("records"):
            root = Path(cache_row["cache_dir"])
            labels = np.load(root / "labels.npy", mmap_mode="r")
            window_starts = np.load(root / "start_time_ms.npy", mmap_mode="r")
            positive_indices = np.flatnonzero(np.asarray(labels) == 1)
            if positive_indices.size == 0:
                continue
            duration_ms = float(cache_row["window_seconds"]) * 1000.0
            for local_index in positive_indices.tolist():
                window_start = float(window_starts[local_index])
                window_end = window_start + duration_ms
                overlaps = np.maximum(
                    0.0, np.minimum(window_end, ends) - np.maximum(window_start, starts)
                )
                best = int(np.argmax(overlaps))
                overlap_ms = float(overlaps[best])
                if overlap_ms <= 0:
                    continue
                event_start = float(starts[best])
                event_end = float(ends[best])
                midpoint = window_start + duration_ms / 2.0
                phase = _event_phase(midpoint, event_start, event_end)
                session_windows.append(
                    {
                        "animal_id": animal_id,
                        "session_id": session_id,
                        "event_id": str(event_ids[best]),
                        "cache_dir": str(root.resolve()),
                        "cache_local_index": int(local_index),
                        "shard": int(cache_row["shard"]),
                        "sampling_rate_hz": float(cache_row["sampling_rate_hz"]),
                        "start_time_ms": window_start,
                        "end_time_ms": window_end,
                        "event_start_ms": event_start,
                        "event_end_ms": event_end,
                        "event_duration_s": (event_end - event_start) / 1000.0,
                        "event_overlap_s": overlap_ms / 1000.0,
                        "event_overlap_fraction": overlap_ms / duration_ms,
                        "event_phase": phase,
                        "active_channels": str(event_channels[best]),
                    }
                )
        if not session_windows:
            continue
        frame = pd.DataFrame(session_windows).sort_values(
            ["event_start_ms", "start_time_ms", "shard", "cache_local_index"],
            kind="stable",
        )
        frame = _select_morphology_stride(frame, config)
        all_windows.extend(frame.to_dict("records"))

    component_frame = (
        pd.concat(all_components, ignore_index=True)
        if all_components
        else _empty_event_frame()
    )
    window_frame = pd.DataFrame(all_windows)
    if window_frame.empty:
        raise ValueError("No seizure windows could be mapped to closed event components")
    window_frame = _ensure_every_event(window_frame, component_frame)
    window_frame["is_fit_window"] = _event_balanced_fit_mask(window_frame, config)
    return component_frame, window_frame.reset_index(drop=True)


def extract_morphology_features(
    morphology_index: pd.DataFrame,
    batch_size: int = 512,
    filter_config: EEGFilterConfig | None = None,
) -> pd.DataFrame:
    """Extract morphology features after continuous zero-phase EEG filtering."""

    identity = morphology_index.reset_index(drop=True).copy()
    feature_frames: list[pd.DataFrame] = []
    for cache_dir, group in identity.groupby("cache_dir", sort=False):
        root = Path(cache_dir)
        signals = np.load(root / "signals.npy", mmap_mode="r")
        cache_starts = np.load(root / "start_time_ms.npy", mmap_mode="r")
        positions = group["cache_local_index"].to_numpy(np.int64)
        rate = float(group["sampling_rate_hz"].iloc[0])
        metadata_path = root / "meta.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        if "stride_seconds" in metadata:
            stride_seconds = float(metadata["stride_seconds"])
        elif len(cache_starts) > 1:
            stride_seconds = float(
                np.median(np.diff(np.asarray(cache_starts, dtype=float))) / 1000.0
            )
        else:
            stride_seconds = 1.0
        stride_samples = int(round(rate * stride_seconds))
        if stride_samples <= 0 or stride_samples > signals.shape[-1]:
            raise ValueError(f"Invalid cache stride for {root}: {stride_samples} samples")
        if len(signals) > 1:
            overlap = signals.shape[-1] - stride_samples
            if overlap > 0 and not np.allclose(
                np.asarray(signals[0, :, stride_samples:]),
                np.asarray(signals[1, :, :overlap]),
                rtol=1e-5,
                atol=1e-4,
                equal_nan=True,
            ):
                raise ValueError(
                    f"Overlapping cache windows in {root} cannot be reconstructed continuously"
                )
        continuous = reconstruct_continuous_cache(signals, stride_samples)
        filtered_continuous = filter_continuous_eeg(
            continuous,
            rate,
            filter_config,
        )
        filtered_windows = extract_filtered_windows(
            filtered_continuous,
            positions,
            stride_samples,
            signals.shape[-1],
        )
        for offset in range(0, len(group), batch_size):
            selection = positions[offset : offset + batch_size]
            raw_values = np.asarray(signals[selection], dtype=np.float32)
            filtered_values = filtered_windows[offset : offset + len(selection)]
            batch_features = pd.DataFrame(
                _feature_batch(raw_values, rate, filtered_values)
            )
            batch_features.index = group.index[offset : offset + len(selection)]
            feature_frames.append(batch_features)
    features = pd.concat(feature_frames, axis=0).sort_index()
    merged = pd.concat([identity, features.reset_index(drop=True)], axis=1)
    return _within_animal_scale_amplitude_features(merged)


def fit_morphology_discovery(
    features: pd.DataFrame,
    output_dir: str | Path,
    config: MorphologyConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit a validated GMM morphology model with HDBSCAN as sensitivity analysis."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    features = _add_event_timing_columns(features)
    if "relative_low_gamma_clean" not in features:
        features["relative_low_gamma_clean"] = np.maximum(
            pd.to_numeric(features["relative_low_gamma"], errors="coerce")
            - pd.to_numeric(features["relative_line_noise_48_52"], errors="coerce"),
            0.0,
        )
    if (
        "ch8_relative_low_gamma_clean" not in features
        and "ch8_relative_low_gamma" in features
        and "ch8_relative_line_noise_48_52" in features
    ):
        features["ch8_relative_low_gamma_clean"] = np.maximum(
            pd.to_numeric(features["ch8_relative_low_gamma"], errors="coerce")
            - pd.to_numeric(
                features["ch8_relative_line_noise_48_52"], errors="coerce"
            ),
            0.0,
        )
    features = _within_animal_scale_amplitude_features(
        features,
        reference_mask=features["is_fit_window"].astype(bool).to_numpy(),
    )
    features = _within_animal_scale_channel_amplitudes(
        features,
        "ch8",
        features["is_fit_window"].astype(bool).to_numpy(),
    )
    columns = cluster_feature_columns(features)
    raw_matrix = features[columns].replace([np.inf, -np.inf], np.nan).copy()
    fit_mask = features["is_fit_window"].astype(bool).to_numpy()
    fit_rows = features.loc[fit_mask].reset_index(drop=True)
    raw_fit = raw_matrix.loc[fit_mask].reset_index(drop=True)
    medians = raw_fit.median(axis=0).fillna(0.0)
    matrix = raw_matrix.fillna(medians)
    fit_values = raw_fit.fillna(medians).to_numpy(np.float64)
    if len(fit_values) < 100:
        raise ValueError("At least 100 event-balanced windows are needed for discovery")

    scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(fit_values)
    scaled_fit = scaler.transform(fit_values)
    component_count = min(config.pca_components, scaled_fit.shape[1], len(scaled_fit) - 1)
    pca = PCA(n_components=component_count, whiten=True, random_state=config.random_seed)
    embedded_fit = pca.fit_transform(scaled_fit)
    all_scaled = scaler.transform(matrix.to_numpy(np.float64))
    all_embedded = pca.transform(all_scaled)

    gmm_grid, gmm_models, gmm_labels = _validated_gmm_grid(
        raw_fit,
        embedded_fit,
        fit_rows,
        config,
    )
    selected_gmm = _select_gmm_candidate(gmm_grid)
    selected_index = int(selected_gmm["candidate_index"])
    gmm = gmm_models[selected_index]
    raw_fit_labels = gmm_labels[selected_index]
    cluster_map = _stable_cluster_map(raw_fit_labels)
    raw_all_labels = gmm.predict(all_embedded)
    all_labels = np.asarray(
        [cluster_map[int(label)] for label in raw_all_labels], dtype=np.int32
    )
    raw_probabilities = gmm.predict_proba(all_embedded)
    all_probabilities = np.empty_like(raw_probabilities)
    for raw_label, stable_label in cluster_map.items():
        all_probabilities[:, stable_label] = raw_probabilities[:, raw_label]
    maximum_probability = np.max(all_probabilities, axis=1)
    posterior_entropy = -np.sum(
        all_probabilities * np.log(np.maximum(all_probabilities, 1e-12)), axis=1
    )
    uncertain = maximum_probability < config.minimum_assignment_probability

    review_gmm_row = _select_review_gmm_candidate(gmm_grid, selected_gmm)
    review_index = int(review_gmm_row["candidate_index"])
    review_gmm = gmm_models[review_index]
    review_fit_labels = gmm_labels[review_index]
    review_cluster_map = _stable_cluster_map(review_fit_labels)
    raw_review_labels = review_gmm.predict(all_embedded)
    review_labels = np.asarray(
        [review_cluster_map[int(label)] for label in raw_review_labels], dtype=np.int32
    )
    raw_review_probabilities = review_gmm.predict_proba(all_embedded)
    review_probabilities = np.empty_like(raw_review_probabilities)
    for raw_label, stable_label in review_cluster_map.items():
        review_probabilities[:, stable_label] = raw_review_probabilities[:, raw_label]
    review_maximum_probability = np.max(review_probabilities, axis=1)

    ch8_columns = ch8_feature_columns(features)
    ch8_raw_matrix = features[ch8_columns].replace([np.inf, -np.inf], np.nan)
    ch8_raw_fit = ch8_raw_matrix.loc[fit_mask].reset_index(drop=True)
    ch8_medians = ch8_raw_fit.median(axis=0).fillna(0.0)
    ch8_fit_values = ch8_raw_fit.fillna(ch8_medians).to_numpy(np.float64)
    ch8_scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(ch8_fit_values)
    ch8_scaled_fit = ch8_scaler.transform(ch8_fit_values)
    ch8_component_count = min(
        config.pca_components,
        ch8_scaled_fit.shape[1],
        len(ch8_scaled_fit) - 1,
    )
    ch8_pca = PCA(
        n_components=ch8_component_count,
        whiten=True,
        random_state=config.random_seed,
    ).fit(ch8_scaled_fit)
    ch8_embedded_fit = ch8_pca.transform(ch8_scaled_fit)
    ch8_grid, ch8_models, ch8_fit_labels_by_candidate = _validated_gmm_grid(
        ch8_raw_fit,
        ch8_embedded_fit,
        fit_rows,
        config,
    )
    ch8_selected = _select_gmm_candidate(ch8_grid)
    ch8_index = int(ch8_selected["candidate_index"])
    ch8_gmm = ch8_models[ch8_index]
    ch8_fit_labels = ch8_fit_labels_by_candidate[ch8_index]
    ch8_cluster_map = _stable_cluster_map(ch8_fit_labels)
    ch8_all_values = ch8_raw_matrix.fillna(ch8_medians).to_numpy(np.float64)
    ch8_all_embedded = ch8_pca.transform(ch8_scaler.transform(ch8_all_values))
    ch8_raw_all_labels = ch8_gmm.predict(ch8_all_embedded)
    ch8_all_labels = np.asarray(
        [ch8_cluster_map[int(label)] for label in ch8_raw_all_labels], dtype=np.int32
    )
    ch8_raw_probabilities = ch8_gmm.predict_proba(ch8_all_embedded)
    ch8_probabilities = np.empty_like(ch8_raw_probabilities)
    for raw_label, stable_label in ch8_cluster_map.items():
        ch8_probabilities[:, stable_label] = ch8_raw_probabilities[:, raw_label]
    ch8_maximum_probability = np.max(ch8_probabilities, axis=1)

    hdbscan_grid, hdbscan_labels = _hdbscan_grid(embedded_fit, config)
    selected_hdbscan = hdbscan_grid.sort_values(
        ["objective", "silhouette", "cluster_count"],
        ascending=[False, False, True],
    ).iloc[0]
    selected_hdbscan_labels = hdbscan_labels[int(selected_hdbscan["candidate_index"])]
    hdbscan_map = _stable_cluster_map(selected_hdbscan_labels)
    stable_hdbscan_fit = np.asarray(
        [hdbscan_map.get(int(label), -1) for label in selected_hdbscan_labels],
        dtype=np.int32,
    )
    hdbscan_all = np.full(len(features), -1, dtype=np.int32)
    hdbscan_distance = np.full(len(features), np.nan, dtype=float)
    hdbscan_medoids = np.empty((0, all_embedded.shape[1]), dtype=float)
    hdbscan_cutoffs = np.empty(0, dtype=float)
    hdbscan_fit_assignment_ari = float("nan")
    if np.any(stable_hdbscan_fit >= 0):
        hdbscan_medoids, hdbscan_cutoffs = _cluster_medoids(
            embedded_fit, stable_hdbscan_fit
        )
        hdbscan_all, hdbscan_distance = _assign_to_medoids(
            all_embedded, hdbscan_medoids, hdbscan_cutoffs
        )
        fit_indices = np.flatnonzero(fit_mask)
        hdbscan_fit_assignment_ari = float(
            adjusted_rand_score(stable_hdbscan_fit, hdbscan_all[fit_indices])
        )
        hdbscan_all[fit_indices] = stable_hdbscan_fit

    result = features.copy()
    result["morphology_cluster"] = all_labels
    result["morphology_label"] = [
        f"M{label + 1}" for label in all_labels
    ]
    result["morphology_probability"] = maximum_probability
    result["morphology_posterior_entropy"] = posterior_entropy
    result["morphology_uncertain"] = uncertain
    result["morphology_distance"] = 1.0 - maximum_probability
    result["review_morphology_cluster"] = review_labels
    result["review_morphology_label"] = [
        f"R{label + 1}" for label in review_labels
    ]
    result["review_morphology_probability"] = review_maximum_probability
    result["review_morphology_uncertain"] = (
        review_maximum_probability < config.minimum_assignment_probability
    )
    result["review_morphology_distance"] = 1.0 - review_maximum_probability
    result["ch8_morphology_cluster"] = ch8_all_labels
    result["ch8_morphology_label"] = [
        f"C8M{label + 1}" for label in ch8_all_labels
    ]
    result["ch8_morphology_probability"] = ch8_maximum_probability
    result["ch8_morphology_uncertain"] = (
        ch8_maximum_probability < config.minimum_assignment_probability
    )
    result["hdbscan_cluster"] = hdbscan_all
    result["hdbscan_distance"] = hdbscan_distance
    result["pc1"] = all_embedded[:, 0]
    result["pc2"] = all_embedded[:, 1] if all_embedded.shape[1] > 1 else 0.0

    local_counts, local_diagnostics = _per_mouse_local_counts(result, columns, config)
    pairwise_ari = _grid_pairwise_ari(hdbscan_labels)
    all_phases = features["absolute_event_phase"].astype(str).to_numpy()
    all_quality = (
        features["ch7_valid"].astype(int).to_numpy() * 2
        + features["ch8_valid"].astype(int).to_numpy()
    )
    phase_nmi = float(normalized_mutual_info_score(all_phases, all_labels))
    quality_nmi = float(normalized_mutual_info_score(all_quality, all_labels))
    hdbscan_parameters = {
        "min_cluster_size": int(selected_hdbscan["min_cluster_size"]),
        "min_samples": int(selected_hdbscan["min_samples"]),
        "cluster_selection_method": "eom",
        "allow_single_cluster": bool(selected_hdbscan["allow_single_cluster"]),
    }
    hdbscan_event_bootstrap = _event_block_bootstrap_stability(
        embedded_fit, fit_rows, stable_hdbscan_fit, hdbscan_parameters, config.random_seed
    )
    gmm_leave_one_mouse = _gmm_leave_one_mouse_out(
        raw_fit,
        fit_rows,
        raw_fit_labels,
        int(selected_gmm["components"]),
        config,
    )
    selected_k = int(selected_gmm["components"])
    review_k = int(review_gmm_row["components"])
    ch8_k = int(ch8_selected["components"])
    gmm_grid["selected_primary"] = gmm_grid["candidate_index"].eq(selected_index)
    gmm_grid["selected_review_resolution"] = gmm_grid["candidate_index"].eq(
        review_index
    )
    ch8_grid["selected_primary"] = ch8_grid["candidate_index"].eq(ch8_index)
    diagnostics = {
        "schema_version": 2,
        "config": asdict(config),
        "primary_method": "event-block-validated diagonal-covariance GMM",
        "feature_columns": columns,
        "feature_medians": {key: float(value) for key, value in medians.items()},
        "fit_windows": int(fit_mask.sum()),
        "all_windows": int(len(result)),
        "selected_gmm": selected_gmm.to_dict(),
        "selected_review_gmm": review_gmm_row.to_dict(),
        "global_cluster_count": selected_k,
        "review_cluster_count": review_k,
        "uncertain_fraction": float(np.mean(uncertain)),
        "unknown_fraction": float(np.mean(uncertain)),
        "animal_cluster_nmi": float(
            normalized_mutual_info_score(features["animal_id"].astype(str), all_labels)
        ),
        "phase_cluster_nmi": phase_nmi,
        "channel_quality_cluster_nmi": quality_nmi,
        "gmm_bic_selected_components": int(
            gmm_grid.sort_values("bic").iloc[0]["components"]
        ),
        "gmm_selected_components": selected_k,
        "gmm_leave_one_mouse_out": gmm_leave_one_mouse,
        "ch8_only_selected_gmm": ch8_selected.to_dict(),
        "ch8_only_cluster_count": ch8_k,
        "ch8_only_agreement_with_primary_fit_ari": float(
            adjusted_rand_score(raw_fit_labels, ch8_fit_labels)
        ),
        "selected_hdbscan": selected_hdbscan.to_dict(),
        "hdbscan_cluster_count": int(len(hdbscan_medoids)),
        "hdbscan_unknown_fraction": float(np.mean(hdbscan_all < 0)),
        "hdbscan_fit_assignment_ari": hdbscan_fit_assignment_ari,
        "hdbscan_grid_pairwise_ari_median": pairwise_ari,
        "hdbscan_event_block_bootstrap_ari": hdbscan_event_bootstrap,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "local_mouse_counts": local_counts.to_dict("records"),
        "local_mouse_diagnostics": local_diagnostics,
    }

    hdbscan_grid.to_csv(output_dir / "hdbscan_parameter_grid.csv", index=False)
    gmm_grid.to_csv(output_dir / "gmm_parameter_grid.csv", index=False)
    ch8_grid.to_csv(output_dir / "ch8_only_gmm_parameter_grid.csv", index=False)
    local_counts.to_csv(output_dir / "per_mouse_local_type_counts.csv", index=False)
    joblib.dump(
        {
            "scaler": scaler,
            "pca": pca,
            "gmm": gmm,
            "cluster_map": cluster_map,
            "review_gmm": review_gmm,
            "review_cluster_map": review_cluster_map,
            "ch8_scaler": ch8_scaler,
            "ch8_pca": ch8_pca,
            "ch8_gmm": ch8_gmm,
            "ch8_cluster_map": ch8_cluster_map,
            "ch8_feature_columns": ch8_columns,
            "ch8_feature_medians": ch8_medians.to_dict(),
            "feature_columns": columns,
            "feature_medians": medians.to_dict(),
            "minimum_assignment_probability": config.minimum_assignment_probability,
            "hdbscan_medoids": hdbscan_medoids,
            "hdbscan_distance_cutoffs": hdbscan_cutoffs,
        },
        model_dir / "morphology_discovery.joblib",
    )
    (output_dir / "clustering_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    return result, diagnostics


def summarize_morphologies(
    assignments: pd.DataFrame,
    events: pd.DataFrame,
    cluster_column: str = "morphology_cluster",
    label_column: str = "morphology_label",
    probability_column: str = "morphology_probability",
    uncertain_column: str = "morphology_uncertain",
) -> dict[str, pd.DataFrame]:
    """Produce event-balanced cluster, mouse, and event tables for review."""

    required = {cluster_column, label_column, probability_column, uncertain_column}
    missing = required - set(assignments)
    if missing:
        raise ValueError(f"Morphology assignments are missing columns: {sorted(missing)}")
    known = assignments.loc[assignments[cluster_column].ge(0)].copy()
    global_line_median = float(known["relative_line_noise_48_52"].median())
    global_line_p95 = float(known["relative_line_noise_48_52"].quantile(0.95))
    phase_column = (
        "absolute_event_phase" if "absolute_event_phase" in known else "event_phase"
    )
    global_phases = known[phase_column].astype(str).value_counts(normalize=True)
    cluster_rows: list[dict[str, Any]] = []
    for (cluster, label), group in known.groupby(
        [cluster_column, label_column], sort=True
    ):
        animal_counts = group["animal_id"].astype(str).value_counts()
        event_counts = group["event_id"].astype(str).value_counts()
        phase_counts = group[phase_column].astype(str).value_counts(normalize=True)
        phase_ratio = max(
            (
                float(fraction / max(global_phases.get(phase, 0.0), 1e-12))
                for phase, fraction in phase_counts.items()
            ),
            default=0.0,
        )
        maximum_phase_fraction = float(phase_counts.max()) if len(phase_counts) else 0.0
        line_noise = float(group["relative_line_noise_48_52"].median())
        artifact_candidate = bool(
            line_noise > global_line_p95
            or line_noise > max(global_line_median * 10.0, 1e-12)
        )
        phase_candidate = bool(maximum_phase_fraction > 0.40 and phase_ratio > 2.0)
        animal_limited = bool(
            len(animal_counts) < 3
            or float(animal_counts.max() / len(group)) > 0.50
        )
        flags = []
        if artifact_candidate:
            flags.append("line_noise_or_artifact")
        if phase_candidate:
            flags.append("phase_associated")
        if animal_limited:
            flags.append("animal_limited")
        cluster_rows.append(
            {
                "morphology_cluster": int(cluster),
                "morphology_label": str(label),
                "windows": int(len(group)),
                "animals": int(group["animal_id"].nunique()),
                "events": int(group["event_id"].nunique()),
                "maximum_single_animal_fraction": float(animal_counts.max() / len(group)),
                "maximum_single_event_fraction": float(event_counts.max() / len(group)),
                "median_assignment_probability": float(group[probability_column].median()),
                "uncertain_fraction": float(group[uncertain_column].mean()),
                "median_event_duration_s": float(group["event_duration_s"].median()),
                "median_robust_amplitude": float(group["aggregate_robust_amplitude"].median()),
                "median_line_length": float(group["aggregate_line_length"].median()),
                "median_peak_frequency_hz": float(group["aggregate_peak_frequency_hz"].median()),
                "median_spectral_entropy": float(group["aggregate_spectral_entropy"].median()),
                "median_line_noise_48_52": line_noise,
                "valid_ch7_fraction": float(group["ch7_valid"].mean()),
                "valid_ch8_fraction": float(group["ch8_valid"].mean()),
                "maximum_phase_fraction": maximum_phase_fraction,
                "maximum_phase_enrichment": phase_ratio,
                "shared_across_animals": not animal_limited,
                "clinical_review_status": "review" if not flags else ";".join(flags),
            }
        )
    cluster_summary = pd.DataFrame(cluster_rows)

    total_windows = known.groupby("animal_id").size()
    total_events = known.groupby("animal_id")["event_id"].nunique()
    event_totals = known.groupby(["animal_id", "event_id"]).size().rename("event_windows")
    event_counts_by_type = (
        known.groupby(["animal_id", "event_id", cluster_column, label_column])
        .size()
        .rename("type_windows")
        .reset_index()
        .join(event_totals, on=["animal_id", "event_id"])
    )
    event_counts_by_type["within_event_fraction"] = (
        event_counts_by_type["type_windows"] / event_counts_by_type["event_windows"]
    )
    mouse_rows: list[dict[str, Any]] = []
    for (animal, cluster, label), group in known.groupby(
        ["animal_id", cluster_column, label_column], sort=True
    ):
        event_type = event_counts_by_type.loc[
            event_counts_by_type["animal_id"].astype(str).eq(str(animal))
            & event_counts_by_type[cluster_column].eq(cluster)
        ]
        animal_event_count = int(total_events.loc[animal])
        event_balanced_fraction = float(
            event_type["within_event_fraction"].sum() / max(animal_event_count, 1)
        )
        mouse_rows.append(
            {
                "animal_id": str(animal),
                "morphology_cluster": int(cluster),
                "morphology_label": str(label),
                "windows": int(len(group)),
                "events": int(group["event_id"].nunique()),
                "animal_windows": int(total_windows.loc[animal]),
                "window_fraction": float(len(group) / total_windows.loc[animal]),
                "event_presence_fraction": float(
                    group["event_id"].nunique() / max(animal_event_count, 1)
                ),
                "event_balanced_fraction": event_balanced_fraction,
                "median_within_event_fraction": float(
                    event_type["within_event_fraction"].median()
                ),
                "present_for_review": bool(
                    len(group) >= 10
                    and event_balanced_fraction >= 0.01
                    and group["event_id"].nunique() >= 2
                ),
            }
        )
    mouse_cluster = pd.DataFrame(mouse_rows)

    event_rows: list[dict[str, Any]] = []
    for event_id, group in assignments.groupby("event_id", sort=True):
        ordered = group.sort_values("start_time_ms")
        labels = ordered[label_column].astype(str)
        confident = labels.loc[~ordered[uncertain_column].astype(bool)]
        counts = confident.value_counts()
        sequence = labels.to_numpy()
        transitions = int((sequence[1:] != sequence[:-1]).sum()) if len(sequence) > 1 else 0
        probabilities = counts.to_numpy(float) / max(counts.sum(), 1)
        entropy = float(
            -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))
        )
        event_rows.append(
            {
                "event_id": str(event_id),
                "animal_id": str(group["animal_id"].iloc[0]),
                "session_id": str(group["session_id"].iloc[0]),
                "start_time_ms": float(group["event_start_ms"].iloc[0]),
                "end_time_ms": float(group["event_end_ms"].iloc[0]),
                "duration_s": float(group["event_duration_s"].iloc[0]),
                "sampled_windows": int(len(group)),
                "candidate_state_count": int(len(counts)),
                "dominant_morphology": str(counts.index[0]) if len(counts) else "uncertain",
                "dominant_fraction": float(counts.iloc[0] / len(group)) if len(counts) else 0.0,
                "uncertain_fraction": float(group[uncertain_column].mean()),
                "morphology_entropy": entropy,
                "window_transitions": transitions,
                "morphology_sequence": ";".join(labels.tolist()),
            }
        )
    event_summary = pd.DataFrame(event_rows)
    represented = set(event_summary["event_id"].astype(str))
    missing_events = events.loc[~events["event_id"].astype(str).isin(represented)].copy()

    shared_labels = set(
        cluster_summary.loc[cluster_summary["shared_across_animals"], "morphology_label"]
    )
    accepted = mouse_cluster.loc[
        mouse_cluster["present_for_review"]
        & mouse_cluster["morphology_label"].isin(shared_labels)
    ]
    type_counts = (
        accepted.groupby("animal_id")
        .agg(shared_global_types=("morphology_label", "nunique"))
        .reset_index()
    )
    all_animals = pd.DataFrame(
        {"animal_id": sorted(assignments["animal_id"].astype(str).unique())}
    )
    type_counts = all_animals.merge(type_counts, on="animal_id", how="left").fillna(0)
    type_counts["shared_global_types"] = type_counts["shared_global_types"].astype(int)
    type_counts["represented_events"] = type_counts["animal_id"].map(
        assignments.groupby("animal_id")["event_id"].nunique()
    ).astype(int)

    return {
        "global_cluster_summary": cluster_summary,
        "mouse_cluster_summary": mouse_cluster,
        "event_morphology_summary": event_summary,
        "per_mouse_global_type_counts": type_counts,
        "events_without_morphology_windows": missing_events,
    }


def cluster_feature_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "scaled_aggregate_log_robust_amplitude",
        "scaled_aggregate_log_rms",
        "scaled_aggregate_log_peak_to_peak",
        "scaled_aggregate_log_line_length",
        "scaled_aggregate_log_total_power",
        "aggregate_hjorth_mobility",
        "aggregate_hjorth_complexity",
        "aggregate_zero_crossing_rate",
        "aggregate_excursion_rate_hz",
        "aggregate_crest_factor",
        "aggregate_peak_frequency_hz",
        "aggregate_median_frequency_hz",
        "aggregate_spectral_entropy",
        "aggregate_spectral_concentration",
        "relative_delta",
        "relative_theta",
        "relative_alpha",
        "relative_beta",
        "relative_low_gamma_clean",
        "relative_high_gamma",
        "theta_delta_ratio",
        "high_low_ratio",
        "channel_correlation",
        "spectral_cosine_similarity",
        "amplitude_log_ratio",
        "line_length_log_ratio",
    ]
    columns = [column for column in preferred if column in frame]
    if len(columns) < 10:
        raise ValueError(f"Only {len(columns)} morphology features are available")
    return columns


def ch8_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Single-channel sensitivity features that cannot encode CH7 hardware quality."""

    preferred = [
        "scaled_ch8_log_robust_amplitude",
        "scaled_ch8_log_rms",
        "scaled_ch8_log_peak_to_peak",
        "scaled_ch8_log_line_length",
        "scaled_ch8_log_total_power",
        "ch8_hjorth_mobility",
        "ch8_hjorth_complexity",
        "ch8_zero_crossing_rate",
        "ch8_excursion_rate_hz",
        "ch8_crest_factor",
        "ch8_peak_frequency_hz",
        "ch8_median_frequency_hz",
        "ch8_spectral_entropy",
        "ch8_spectral_concentration",
        "ch8_relative_delta",
        "ch8_relative_theta",
        "ch8_relative_alpha",
        "ch8_relative_beta",
        "ch8_relative_low_gamma_clean",
        "ch8_relative_high_gamma",
        "ch8_theta_delta_ratio",
        "ch8_high_low_ratio",
    ]
    columns = [column for column in preferred if column in frame]
    if len(columns) < 10:
        raise ValueError(f"Only {len(columns)} CH8 sensitivity features are available")
    return columns


def _feature_batch(
    signals: np.ndarray,
    rate: float,
    filtered_signals: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if signals.ndim != 3 or signals.shape[1] != 2:
        raise ValueError(f"Expected [windows, 2, samples], received {signals.shape}")
    raw = np.asarray(signals, dtype=np.float64)
    finite = np.isfinite(raw)
    raw_nan = np.where(finite, raw, np.nan)
    raw_median = np.nanmedian(raw_nan, axis=2)
    raw_filled = np.where(finite, raw, raw_median[:, :, None])
    raw_centered = raw_filled - raw_median[:, :, None]
    raw_mad = 1.4826 * np.median(np.abs(raw_centered), axis=2)
    raw_q01 = np.quantile(raw_filled, 0.01, axis=2)
    raw_q99 = np.quantile(raw_filled, 0.99, axis=2)
    raw_robust_amplitude = raw_q99 - raw_q01
    raw_differences = np.diff(raw_filled, axis=2)
    flat_fraction = np.mean(np.isclose(raw_differences, 0.0, atol=1e-7), axis=2)
    rail_fraction = np.mean(np.abs(raw_filled) >= 399_999.0, axis=2)
    valid = (
        finite.mean(axis=2) >= 0.999
    ) & (raw_robust_amplitude > 1e-6) & (flat_fraction < 0.995) & (rail_fraction < 0.5)

    if filtered_signals is None:
        filtered_signals = filter_continuous_eeg(raw_filled, rate)
    x = np.asarray(filtered_signals, dtype=np.float64)
    if x.shape != raw.shape:
        raise ValueError(
            f"Filtered EEG shape {x.shape} does not match raw EEG shape {raw.shape}"
        )
    median = np.median(x, axis=2)
    centered = x - median[:, :, None]
    mad = 1.4826 * np.median(np.abs(centered), axis=2)
    q01 = np.quantile(x, 0.01, axis=2)
    q99 = np.quantile(x, 0.99, axis=2)
    robust_amplitude = q99 - q01
    rms = np.sqrt(np.mean(centered**2, axis=2))
    peak_to_peak = np.ptp(x, axis=2)
    differences = np.diff(x, axis=2)
    line_length = np.mean(np.abs(differences), axis=2) * rate
    variance = np.var(centered, axis=2)
    diff_variance = np.var(differences, axis=2)
    second_variance = np.var(np.diff(differences, axis=2), axis=2)
    mobility = np.sqrt(diff_variance / np.maximum(variance, 1e-12))
    diff_mobility = np.sqrt(second_variance / np.maximum(diff_variance, 1e-12))
    complexity = diff_mobility / np.maximum(mobility, 1e-12)
    zero_crossing = np.mean(centered[:, :, 1:] * centered[:, :, :-1] < 0, axis=2) * rate
    threshold = 3.0 * np.maximum(mad, 1e-12)
    excursions = np.abs(centered) >= threshold[:, :, None]
    excursion_onsets = excursions[:, :, 1:] & ~excursions[:, :, :-1]
    excursion_rate = excursion_onsets.sum(axis=2) / (signals.shape[2] / rate)
    crest = np.max(np.abs(centered), axis=2) / np.maximum(rms, 1e-12)

    taper = np.hanning(signals.shape[2])
    spectrum = np.abs(np.fft.rfft(centered * taper, axis=2)) ** 2
    raw_spectrum = np.abs(np.fft.rfft(raw_centered * taper, axis=2)) ** 2
    frequencies = np.fft.rfftfreq(signals.shape[2], d=1.0 / rate)
    total_mask = (frequencies >= 0.5) & (frequencies <= min(100.0, rate / 2.0))
    selected_power = spectrum[:, :, total_mask]
    raw_selected_power = raw_spectrum[:, :, total_mask]
    total_power = selected_power.sum(axis=2) + 1e-12
    raw_total_power = raw_selected_power.sum(axis=2) + 1e-12
    normalized_power = selected_power / total_power[:, :, None]
    selected_frequencies = frequencies[total_mask]
    peak_frequency = selected_frequencies[np.argmax(selected_power, axis=2)]
    cumulative = np.cumsum(normalized_power, axis=2)
    median_frequency = selected_frequencies[np.argmax(cumulative >= 0.5, axis=2)]
    spectral_entropy = -np.sum(normalized_power * np.log(normalized_power + 1e-12), axis=2)
    spectral_entropy /= math.log(max(2, normalized_power.shape[2]))
    spectral_concentration = np.max(normalized_power, axis=2)

    channel_features = {
        "log_robust_amplitude": np.log1p(np.maximum(robust_amplitude, 0.0)),
        "log_rms": np.log1p(np.maximum(rms, 0.0)),
        "log_peak_to_peak": np.log1p(np.maximum(peak_to_peak, 0.0)),
        "log_line_length": np.log1p(np.maximum(line_length, 0.0)),
        "log_total_power": np.log1p(np.maximum(total_power, 0.0)),
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
        "zero_crossing_rate": zero_crossing,
        "excursion_rate_hz": excursion_rate,
        "crest_factor": crest,
        "peak_frequency_hz": peak_frequency,
        "median_frequency_hz": median_frequency,
        "spectral_entropy": spectral_entropy,
        "spectral_concentration": spectral_concentration,
    }
    output: dict[str, np.ndarray] = {}
    for channel_index, channel in enumerate(("ch7", "ch8")):
        output[f"{channel}_valid"] = valid[:, channel_index]
        output[f"{channel}_raw_median"] = raw_median[:, channel_index]
        output[f"{channel}_raw_mad"] = raw_mad[:, channel_index]
        output[f"{channel}_flat_fraction"] = flat_fraction[:, channel_index]
        output[f"{channel}_hardware_rail_fraction"] = rail_fraction[:, channel_index]
        for name, values in channel_features.items():
            output[f"{channel}_{name}"] = values[:, channel_index]

    weights = valid.astype(np.float64)
    denominator = np.maximum(weights.sum(axis=1), 1.0)
    for name, values in channel_features.items():
        safe = np.where(valid, values, 0.0)
        output[f"aggregate_{name}"] = safe.sum(axis=1) / denominator
    output["aggregate_robust_amplitude"] = (
        np.where(valid, robust_amplitude, 0.0).sum(axis=1) / denominator
    )
    output["aggregate_line_length"] = (
        np.where(valid, line_length, 0.0).sum(axis=1) / denominator
    )
    output["valid_channel_count"] = valid.sum(axis=1)

    both = valid[:, 0] & valid[:, 1]
    ch7 = centered[:, 0]
    ch8 = centered[:, 1]
    corr_denominator = np.sqrt(np.sum(ch7**2, axis=1) * np.sum(ch8**2, axis=1))
    correlation = np.sum(ch7 * ch8, axis=1) / np.maximum(corr_denominator, 1e-12)
    output["channel_correlation"] = np.where(both, correlation, np.nan)
    spectral_denominator = np.sqrt(
        np.sum(selected_power[:, 0] ** 2, axis=1)
        * np.sum(selected_power[:, 1] ** 2, axis=1)
    )
    spectral_similarity = np.sum(selected_power[:, 0] * selected_power[:, 1], axis=1)
    spectral_similarity /= np.maximum(spectral_denominator, 1e-12)
    output["spectral_cosine_similarity"] = np.where(both, spectral_similarity, np.nan)
    output["amplitude_log_ratio"] = np.where(
        both,
        np.abs(channel_features["log_robust_amplitude"][:, 0] - channel_features["log_robust_amplitude"][:, 1]),
        np.nan,
    )
    output["line_length_log_ratio"] = np.where(
        both,
        np.abs(channel_features["log_line_length"][:, 0] - channel_features["log_line_length"][:, 1]),
        np.nan,
    )

    band_values: dict[str, np.ndarray] = {}
    for name, (low, high) in BANDS.items():
        mask = (frequencies >= low) & (frequencies < min(high, rate / 2.0))
        power = spectrum[:, :, mask].sum(axis=2) / total_power
        band_values[name] = power
        output[f"relative_{name}"] = np.where(valid, power, 0.0).sum(axis=1) / denominator
        for channel_index, channel in enumerate(("ch7", "ch8")):
            output[f"{channel}_relative_{name}"] = power[:, channel_index]
    line_noise_mask = (frequencies >= 48.0) & (frequencies <= 52.0)
    line_noise = spectrum[:, :, line_noise_mask].sum(axis=2) / total_power
    raw_line_noise = (
        raw_spectrum[:, :, line_noise_mask].sum(axis=2) / raw_total_power
    )
    output["relative_line_noise_48_52"] = (
        np.where(valid, line_noise, 0.0).sum(axis=1) / denominator
    )
    output["raw_relative_line_noise_48_52"] = (
        np.where(valid, raw_line_noise, 0.0).sum(axis=1) / denominator
    )
    for channel_index, channel in enumerate(("ch7", "ch8")):
        output[f"{channel}_relative_line_noise_48_52"] = line_noise[:, channel_index]
        output[f"{channel}_relative_low_gamma_clean"] = np.maximum(
            band_values["low_gamma"][:, channel_index]
            - line_noise[:, channel_index],
            0.0,
        )
    output["relative_low_gamma_clean"] = np.maximum(
        output["relative_low_gamma"] - output["relative_line_noise_48_52"], 0.0
    )
    output["theta_delta_ratio"] = np.log1p(
        output["relative_theta"] / np.maximum(output["relative_delta"], 1e-12)
    )
    high = (
        output["relative_beta"]
        + output["relative_low_gamma_clean"]
        + output["relative_high_gamma"]
    )
    low = output["relative_delta"] + output["relative_theta"]
    output["high_low_ratio"] = np.log1p(high / np.maximum(low, 1e-12))
    for channel_index, channel in enumerate(("ch7", "ch8")):
        output[f"{channel}_theta_delta_ratio"] = np.log1p(
            band_values["theta"][:, channel_index]
            / np.maximum(band_values["delta"][:, channel_index], 1e-12)
        )
        channel_high = (
            band_values["beta"][:, channel_index]
            + np.maximum(
                band_values["low_gamma"][:, channel_index]
                - line_noise[:, channel_index],
                0.0,
            )
            + band_values["high_gamma"][:, channel_index]
        )
        channel_low = (
            band_values["delta"][:, channel_index]
            + band_values["theta"][:, channel_index]
        )
        output[f"{channel}_high_low_ratio"] = np.log1p(
            channel_high / np.maximum(channel_low, 1e-12)
        )
    output["has_nan_or_inf"] = ~finite.all(axis=(1, 2))
    return output


def _add_event_timing_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    required = {"start_time_ms", "end_time_ms", "event_start_ms", "event_end_ms"}
    if not required.issubset(result):
        result["seconds_from_onset"] = np.nan
        result["seconds_to_offset"] = np.nan
        result["event_position_fraction"] = 0.5
        result["event_decile"] = 5
        result["absolute_event_phase"] = result.get(
            "event_phase", pd.Series("unknown", index=result.index)
        ).astype(str)
        return result
    midpoint = (
        pd.to_numeric(result["start_time_ms"], errors="coerce")
        + pd.to_numeric(result["end_time_ms"], errors="coerce")
    ) / 2.0
    event_start = pd.to_numeric(result["event_start_ms"], errors="coerce")
    event_end = pd.to_numeric(result["event_end_ms"], errors="coerce")
    duration_ms = (event_end - event_start).clip(lower=1.0)
    from_onset = (midpoint - event_start).clip(lower=0.0) / 1000.0
    to_offset = (event_end - midpoint).clip(lower=0.0) / 1000.0
    position = ((midpoint - event_start) / duration_ms).clip(lower=0.0, upper=1.0)
    result["seconds_from_onset"] = from_onset
    result["seconds_to_offset"] = to_offset
    result["event_position_fraction"] = position
    result["event_decile"] = np.minimum((position * 10).astype(int), 9)
    short = duration_ms < 20_000.0
    phase = np.full(len(result), "sustained", dtype=object)
    phase[short.to_numpy()] = "short_event"
    phase[(~short & from_onset.le(10.0)).to_numpy()] = "onset_10s"
    phase[(~short & to_offset.le(10.0)).to_numpy()] = "offset_10s"
    result["absolute_event_phase"] = phase
    return result


def _within_animal_scale_amplitude_features(
    frame: pd.DataFrame,
    reference_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    if reference_mask is None:
        reference_mask = np.ones(len(result), dtype=bool)
    reference_mask = np.asarray(reference_mask, dtype=bool)
    if len(reference_mask) != len(result):
        raise ValueError("Amplitude-scaling reference mask has the wrong length")
    amplitude_columns = [
        column
        for column in result
        if column.startswith("aggregate_")
        and column.endswith(AMPLITUDE_FEATURE_SUFFIXES)
    ]
    for column in amplitude_columns:
        scaled = np.zeros(len(result), dtype=np.float64)
        for _, group in result.groupby("animal_id", sort=False):
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
            group_positions = group.index.to_numpy(np.int64)
            reference = values[reference_mask[group_positions]]
            if not np.isfinite(reference).any():
                reference = values
            median = np.nanmedian(reference)
            q10, q90 = np.nanquantile(reference, [0.1, 0.9])
            scale = max(float(q90 - q10), 1e-6)
            scaled[group_positions] = (
                np.nan_to_num(values, nan=median) - median
            ) / scale
        result[f"scaled_{column}"] = scaled
    return result


def _within_animal_scale_channel_amplitudes(
    frame: pd.DataFrame,
    channel: str,
    reference_mask: np.ndarray,
) -> pd.DataFrame:
    result = frame.copy()
    reference_mask = np.asarray(reference_mask, dtype=bool)
    columns = [
        f"{channel}_{suffix}"
        for suffix in AMPLITUDE_FEATURE_SUFFIXES
        if f"{channel}_{suffix}" in result
    ]
    for column in columns:
        scaled = np.zeros(len(result), dtype=np.float64)
        for _, group in result.groupby("animal_id", sort=False):
            positions = group.index.to_numpy(np.int64)
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
            reference = values[reference_mask[positions]]
            if not np.isfinite(reference).any():
                reference = values
            median = np.nanmedian(reference)
            q10, q90 = np.nanquantile(reference, [0.1, 0.9])
            scale = max(float(q90 - q10), 1e-6)
            scaled[positions] = (np.nan_to_num(values, nan=median) - median) / scale
        result[f"scaled_{column}"] = scaled
    return result


def _hdbscan_grid(
    embedded: np.ndarray, config: MorphologyConfig
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    fractional_sizes = [
        int(round(len(embedded) * fraction))
        for fraction in np.geomspace(
            config.minimum_cluster_fraction, config.maximum_cluster_fraction, 4
        )
    ]
    cluster_sizes = sorted(
        {
            value
            for value in [5, 10, 20, 30, 50, 80, *fractional_sizes]
            if 2 <= value < max(3, len(embedded) // 2)
        }
    )
    rows: list[dict[str, Any]] = []
    labels_by_candidate: list[np.ndarray] = []
    rng = np.random.default_rng(config.random_seed)
    sample = rng.choice(len(embedded), min(5000, len(embedded)), replace=False)
    candidate_index = 0
    for minimum in cluster_sizes:
        for min_samples in sorted({3, 5, min(10, minimum)}):
            for allow_single in (False, True):
                model = HDBSCAN(
                    min_cluster_size=minimum,
                    min_samples=min_samples,
                    cluster_selection_method="eom",
                    allow_single_cluster=allow_single,
                    copy=False,
                )
                labels = model.fit_predict(embedded)
                labels_by_candidate.append(labels)
                valid = labels >= 0
                cluster_count = len(set(labels[valid]))
                noise_fraction = float(np.mean(~valid))
                sampled_valid = valid[sample]
                silhouette = float("nan")
                if cluster_count >= 2 and sampled_valid.sum() > cluster_count:
                    silhouette = float(
                        silhouette_score(
                            embedded[sample][sampled_valid], labels[sample][sampled_valid]
                        )
                    )
                if cluster_count == 1:
                    score = 0.0
                else:
                    score = -1.0 if not np.isfinite(silhouette) else silhouette
                objective = score - 0.10 * noise_fraction
                if cluster_count == 0:
                    objective -= 1.0
                if cluster_count > 12:
                    objective -= 0.05 * (cluster_count - 12)
                rows.append(
                    {
                        "candidate_index": candidate_index,
                        "minimum_cluster_fraction": minimum / len(embedded),
                        "min_cluster_size": minimum,
                        "min_samples": min_samples,
                        "allow_single_cluster": allow_single,
                        "cluster_count": cluster_count,
                        "noise_fraction": noise_fraction,
                        "silhouette": silhouette,
                        "objective": objective,
                    }
                )
                candidate_index += 1
    return pd.DataFrame(rows), labels_by_candidate


def _validated_gmm_grid(
    raw_fit: pd.DataFrame,
    embedded_fit: np.ndarray,
    fit_rows: pd.DataFrame,
    config: MorphologyConfig,
) -> tuple[pd.DataFrame, list[GaussianMixture], list[np.ndarray]]:
    """Compare GMM K values without splitting windows from the same event."""

    if not (len(raw_fit) == len(embedded_fit) == len(fit_rows)):
        raise ValueError("GMM inputs must contain the same number of fitting windows")
    folds = _event_stratified_fold_ids(
        fit_rows, min(config.gmm_cv_folds, fit_rows["event_id"].nunique()), config.random_seed
    )
    rows: list[dict[str, Any]] = []
    models: list[GaussianMixture] = []
    labels_by_candidate: list[np.ndarray] = []
    rng = np.random.default_rng(config.random_seed)
    silhouette_selection = rng.choice(
        len(embedded_fit), min(5000, len(embedded_fit)), replace=False
    )
    maximum = min(config.gmm_max_components, len(embedded_fit) - 1)
    animals = fit_rows["animal_id"].astype(str).to_numpy()
    events = fit_rows["event_id"].astype(str).to_numpy()
    quality = (
        fit_rows["ch7_valid"].astype(int).to_numpy() * 2
        + fit_rows["ch8_valid"].astype(int).to_numpy()
    )
    for candidate_index, components in enumerate(range(1, maximum + 1)):
        model = GaussianMixture(
            n_components=components,
            covariance_type="diag",
            reg_covar=1e-5,
            n_init=10,
            random_state=config.random_seed,
        ).fit(embedded_fit)
        labels = model.predict(embedded_fit)
        models.append(model)
        labels_by_candidate.append(labels)
        counts = np.bincount(labels, minlength=components)
        fractions = counts / len(labels)
        component_animals: list[int] = []
        component_events: list[int] = []
        maximum_animal_fraction: list[float] = []
        maximum_event_fraction: list[float] = []
        minimum_per_animal_windows: list[int] = []
        qualified_animals: list[int] = []
        for component in range(components):
            selected = labels == component
            animal_counts = pd.Series(animals[selected]).value_counts()
            event_counts = pd.Series(events[selected]).value_counts()
            component_animals.append(int(len(animal_counts)))
            component_events.append(int(len(event_counts)))
            maximum_animal_fraction.append(float(animal_counts.max() / selected.sum()))
            maximum_event_fraction.append(float(event_counts.max() / selected.sum()))
            minimum_per_animal_windows.append(
                int(animal_counts.min()) if len(animal_counts) else 0
            )
            qualified_animals.append(
                int(
                    animal_counts.ge(
                        config.minimum_component_windows_per_animal
                    ).sum()
                )
            )
        cv_scores = _gmm_event_block_cv_scores(
            raw_fit, fit_rows, folds, components, config
        )
        bootstrap = _gmm_event_bootstrap_stability(
            embedded_fit, fit_rows, labels, components, config
        )
        silhouette = (
            float(
                silhouette_score(
                    embedded_fit[silhouette_selection], labels[silhouette_selection]
                )
            )
            if components > 1
            else float("nan")
        )
        support_ok = bool(
            np.min(fractions) >= config.minimum_gmm_component_fraction
            and min(qualified_animals) >= config.minimum_gmm_component_animals
            and min(component_events) >= config.minimum_gmm_component_events
            and max(maximum_animal_fraction)
            <= config.maximum_component_single_animal_fraction
            and max(maximum_event_fraction)
            <= config.maximum_component_single_event_fraction
        )
        stability_ok = bool(
            bootstrap["median_ari"] >= config.minimum_bootstrap_median_ari
            and bootstrap["q25_ari"] >= config.minimum_bootstrap_q25_ari
        )
        cv_array = np.asarray(cv_scores, dtype=float)
        finite_cv = cv_array[np.isfinite(cv_array)]
        cv_mean = float(np.mean(finite_cv)) if len(finite_cv) else float("nan")
        cv_se = (
            float(np.std(finite_cv, ddof=1) / math.sqrt(len(finite_cv)))
            if len(finite_cv) > 1
            else float("nan")
        )
        rows.append(
            {
                "candidate_index": candidate_index,
                "components": components,
                "bic": float(model.bic(embedded_fit)),
                "aic": float(model.aic(embedded_fit)),
                "silhouette": silhouette,
                "cv_mean_log_likelihood": cv_mean,
                "cv_se_log_likelihood": cv_se,
                "cv_fold_scores": json.dumps(cv_scores),
                "bootstrap_median_ari": bootstrap["median_ari"],
                "bootstrap_q25_ari": bootstrap["q25_ari"],
                "bootstrap_q75_ari": bootstrap["q75_ari"],
                "bootstrap_ari_values": json.dumps(bootstrap["ari_values"]),
                "minimum_component_fraction": float(np.min(fractions)),
                "minimum_component_animals": int(min(component_animals)),
                "minimum_component_qualified_animals": int(min(qualified_animals)),
                "minimum_component_events": int(min(component_events)),
                "minimum_component_windows_per_animal": int(
                    min(minimum_per_animal_windows)
                ),
                "maximum_component_single_animal_fraction": float(
                    max(maximum_animal_fraction)
                ),
                "maximum_component_single_event_fraction": float(
                    max(maximum_event_fraction)
                ),
                "animal_cluster_nmi": float(
                    normalized_mutual_info_score(animals, labels)
                ),
                "channel_quality_cluster_nmi": float(
                    normalized_mutual_info_score(quality, labels)
                ),
                "support_ok": support_ok,
                "stability_ok": stability_ok,
                "accepted_for_primary": support_ok and stability_ok,
            }
        )
    return pd.DataFrame(rows), models, labels_by_candidate


def _select_gmm_candidate(grid: pd.DataFrame) -> pd.Series:
    """Use event-CV one-SE selection, preferring a stable parsimonious model."""

    valid = grid.loc[
        grid["accepted_for_primary"].astype(bool)
        & np.isfinite(grid["cv_mean_log_likelihood"])
    ].copy()
    selection_basis = "support_and_stability"
    if valid.empty:
        valid = grid.loc[
            grid["support_ok"].astype(bool)
            & np.isfinite(grid["cv_mean_log_likelihood"])
        ].copy()
        selection_basis = "support_only_low_confidence"
    if valid.empty:
        valid = grid.loc[np.isfinite(grid["cv_mean_log_likelihood"])].copy()
        selection_basis = "cv_only_exploratory"
    best = valid.sort_values("cv_mean_log_likelihood", ascending=False).iloc[0]
    best_se = float(best["cv_se_log_likelihood"])
    if not np.isfinite(best_se):
        best_se = 0.0
    threshold = float(best["cv_mean_log_likelihood"]) - best_se
    parsimonious = valid.loc[valid["cv_mean_log_likelihood"].ge(threshold)]
    selected = parsimonious.sort_values("components").iloc[0].copy()
    selected["selection_basis"] = selection_basis
    selected["one_se_threshold"] = threshold
    return selected


def _select_review_gmm_candidate(
    grid: pd.DataFrame, primary: pd.Series
) -> pd.Series:
    """Return the finest supportable K for clinical waveform review, not diagnosis."""

    supported = grid.loc[grid["support_ok"].astype(bool)].copy()
    if supported.empty:
        selected = primary.copy()
        selected["selection_basis"] = "same_as_primary_no_finer_supported_model"
        return selected
    selected = supported.sort_values("components", ascending=False).iloc[0].copy()
    selected["selection_basis"] = "maximum_supported_exploratory_resolution"
    return selected


def _event_stratified_fold_ids(
    fit_rows: pd.DataFrame, folds: int, seed: int
) -> np.ndarray:
    if folds < 2:
        raise ValueError("Event-block cross-validation needs at least two folds")
    result = np.full(len(fit_rows), -1, dtype=np.int32)
    rng = np.random.default_rng(seed)
    for _, group in fit_rows.groupby("animal_id", sort=True):
        events = group["event_id"].astype(str).unique()
        events = events[rng.permutation(len(events))]
        for position, event_id in enumerate(events):
            indices = group.index[
                group["event_id"].astype(str).eq(event_id)
            ].to_numpy(np.int64)
            result[indices] = position % folds
    if np.any(result < 0):
        raise RuntimeError("Some fitting windows were not assigned to an event CV fold")
    return result


def _preprocess_train_test(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    medians = raw_train.median(axis=0).fillna(0.0)
    train = raw_train.fillna(medians).to_numpy(np.float64)
    test = raw_test.fillna(medians).to_numpy(np.float64)
    scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    pca_components = min(components, train_scaled.shape[1], len(train_scaled) - 1)
    pca = PCA(n_components=pca_components, whiten=True, random_state=seed).fit(
        train_scaled
    )
    return pca.transform(train_scaled), pca.transform(test_scaled)


def _gmm_event_block_cv_scores(
    raw_fit: pd.DataFrame,
    fit_rows: pd.DataFrame,
    fold_ids: np.ndarray,
    components: int,
    config: MorphologyConfig,
) -> list[float]:
    scores: list[float] = []
    for fold in sorted(np.unique(fold_ids)):
        train = fold_ids != fold
        test = ~train
        train_values, test_values = _preprocess_train_test(
            raw_fit.loc[train],
            raw_fit.loc[test],
            config.pca_components,
            config.random_seed + int(fold),
        )
        if len(train_values) <= components:
            scores.append(float("nan"))
            continue
        model = GaussianMixture(
            n_components=components,
            covariance_type="diag",
            reg_covar=1e-5,
            n_init=5,
            random_state=config.random_seed + int(fold),
        ).fit(train_values)
        scores.append(float(model.score(test_values)))
    return scores


def _gmm_event_bootstrap_stability(
    embedded: np.ndarray,
    fit_rows: pd.DataFrame,
    reference_labels: np.ndarray,
    components: int,
    config: MorphologyConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(config.random_seed + 1000 * components)
    animals = fit_rows["animal_id"].astype(str)
    event_ids = fit_rows["event_id"].astype(str)
    values: list[float] = []
    for repeat in range(config.gmm_bootstrap_repeats):
        sampled: list[int] = []
        for animal in sorted(animals.unique()):
            animal_events = event_ids.loc[animals.eq(animal)].unique()
            drawn = rng.choice(animal_events, len(animal_events), replace=True)
            for event_id in drawn:
                sampled.extend(np.flatnonzero(event_ids.to_numpy() == event_id).tolist())
        model = GaussianMixture(
            n_components=components,
            covariance_type="diag",
            reg_covar=1e-5,
            n_init=5,
            random_state=config.random_seed + repeat,
        ).fit(embedded[np.asarray(sampled, dtype=np.int64)])
        values.append(float(adjusted_rand_score(reference_labels, model.predict(embedded))))
    array = np.asarray(values, dtype=float)
    return {
        "ari_values": values,
        "median_ari": float(np.median(array)),
        "q25_ari": float(np.quantile(array, 0.25)),
        "q75_ari": float(np.quantile(array, 0.75)),
    }


def _gmm_leave_one_mouse_out(
    raw_fit: pd.DataFrame,
    fit_rows: pd.DataFrame,
    reference_labels: np.ndarray,
    components: int,
    config: MorphologyConfig,
) -> list[dict[str, Any]]:
    animals = fit_rows["animal_id"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for fold_index, animal in enumerate(sorted(np.unique(animals))):
        train = animals != animal
        test = ~train
        train_values, test_values = _preprocess_train_test(
            raw_fit.loc[train],
            raw_fit.loc[test],
            config.pca_components,
            config.random_seed + 200 + fold_index,
        )
        model = GaussianMixture(
            n_components=components,
            covariance_type="diag",
            reg_covar=1e-5,
            n_init=10,
            random_state=config.random_seed + 200 + fold_index,
        ).fit(train_values)
        predicted = model.predict(test_values)
        rows.append(
            {
                "held_out_animal": animal,
                "test_windows": int(test.sum()),
                "mean_log_likelihood": float(model.score(test_values)),
                "agreement_with_global_ari": float(
                    adjusted_rand_score(reference_labels[test], predicted)
                ),
            }
        )
    return rows


def _cluster_medoids(
    embedded: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    medoids: list[np.ndarray] = []
    cutoffs: list[float] = []
    for label in sorted(set(labels[labels >= 0])):
        values = embedded[labels == label]
        centroid = values.mean(axis=0)
        distances = np.linalg.norm(values - centroid, axis=1)
        medoid = values[int(np.argmin(distances))]
        member_distances = np.linalg.norm(values - medoid, axis=1)
        medoids.append(medoid)
        cutoffs.append(float(np.quantile(member_distances, 0.99) * 1.25 + 1e-6))
    if not medoids:
        raise ValueError("HDBSCAN did not identify a usable morphology cluster")
    return np.stack(medoids), np.asarray(cutoffs)


def _assign_to_medoids(
    embedded: np.ndarray, medoids: np.ndarray, cutoffs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    distances = cdist(embedded, medoids, metric="euclidean")
    labels = np.argmin(distances, axis=1)
    nearest = distances[np.arange(len(distances)), labels]
    labels = labels.astype(np.int32)
    labels[nearest > cutoffs[labels]] = -1
    return labels, nearest


def _stable_cluster_map(labels: np.ndarray) -> dict[int, int]:
    valid = labels[labels >= 0]
    if not len(valid):
        return {}
    counts = pd.Series(valid).value_counts().sort_values(ascending=False)
    return {int(old): int(new) for new, old in enumerate(counts.index.tolist())}


def _grid_pairwise_ari(labels: list[np.ndarray]) -> float:
    values: list[float] = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            values.append(float(adjusted_rand_score(labels[left], labels[right])))
    return float(np.median(values)) if values else float("nan")


def _per_mouse_local_counts(
    result: pd.DataFrame,
    columns: list[str],
    config: MorphologyConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for animal, group in result.groupby("animal_id", sort=True):
        group = _local_event_balanced_rows(group, config)
        values = group[columns].replace([np.inf, -np.inf], np.nan)
        values = values.fillna(values.median(axis=0).fillna(0.0)).to_numpy(float)
        if len(values) > 6000:
            positions = np.linspace(0, len(values) - 1, 6000, dtype=np.int64)
            values = values[positions]
        scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(values)
        scaled = scaler.transform(values)
        components = min(config.pca_components, scaled.shape[1], len(scaled) - 1)
        embedded = PCA(n_components=components, whiten=True, random_state=config.random_seed).fit_transform(scaled)
        local_grid, _ = _hdbscan_grid(embedded, config)
        selected = local_grid.sort_values(["objective", "silhouette"], ascending=False).iloc[0]
        rows.append(
            {
                "animal_id": str(animal),
                "local_candidate_types": int(selected["cluster_count"]),
                "local_noise_fraction": float(selected["noise_fraction"]),
                "local_silhouette": float(selected["silhouette"]),
                "windows_used": int(len(values)),
            }
        )
        diagnostics.append({"animal_id": str(animal), "grid": local_grid.to_dict("records")})
    return pd.DataFrame(rows), diagnostics


def _select_morphology_stride(frame: pd.DataFrame, config: MorphologyConfig) -> pd.DataFrame:
    selected: list[int] = []
    minimum_gap = config.morphology_stride_seconds * 1000.0 - 1e-6
    for _, group in frame.groupby("event_id", sort=False):
        pure = group.loc[group["event_overlap_fraction"].ge(config.minimum_overlap_fraction)]
        candidates = pure if not pure.empty else group.nlargest(1, "event_overlap_fraction")
        last = -float("inf")
        for index, row in candidates.sort_values("start_time_ms").iterrows():
            start = float(row["start_time_ms"])
            if start - last >= minimum_gap:
                selected.append(int(index))
                last = start
        if not any(index in selected for index in candidates.index):
            selected.append(int(candidates.index[0]))
    return frame.loc[sorted(set(selected))].copy()


def _event_balanced_fit_mask(frame: pd.DataFrame, config: MorphologyConfig) -> np.ndarray:
    selected_by_animal: dict[str, list[int]] = {}
    for _, animal_group in frame.groupby("animal_id", sort=True):
        animal_selected: list[int] = []
        for _, event_group in animal_group.groupby("event_id", sort=True):
            middle = event_group.loc[event_group["event_phase"].eq("middle")]
            candidates = middle if not middle.empty else event_group.iloc[[len(event_group) // 2]]
            count = min(config.fit_windows_per_event, len(candidates))
            positions = np.linspace(0, len(candidates) - 1, count, dtype=np.int64)
            animal_selected.extend(candidates.iloc[positions].index.astype(int).tolist())
        selected_by_animal[str(animal_group["animal_id"].iloc[0])] = animal_selected
    target = min(
        config.max_fit_windows_per_animal,
        min(len(values) for values in selected_by_animal.values()),
    )
    if target < 20:
        raise ValueError(f"Animal-equal morphology fitting has only {target} windows per animal")
    selected: list[int] = []
    for values in selected_by_animal.values():
        positions = np.linspace(0, len(values) - 1, target, dtype=np.int64)
        selected.extend(values[position] for position in positions)
    mask = np.zeros(len(frame), dtype=bool)
    mask[np.asarray(sorted(set(selected)), dtype=np.int64)] = True
    return mask


def _local_event_balanced_rows(
    group: pd.DataFrame, config: MorphologyConfig
) -> pd.DataFrame:
    selected: list[int] = []
    for _, event_group in group.groupby("event_id", sort=True):
        middle = event_group.loc[event_group["event_phase"].eq("middle")]
        candidates = middle if not middle.empty else event_group.iloc[[len(event_group) // 2]]
        count = min(config.fit_windows_per_event, len(candidates))
        positions = np.linspace(0, len(candidates) - 1, count, dtype=np.int64)
        selected.extend(candidates.iloc[positions].index.astype(int).tolist())
    if len(selected) > config.max_fit_windows_per_animal:
        positions = np.linspace(0, len(selected) - 1, config.max_fit_windows_per_animal, dtype=np.int64)
        selected = [selected[position] for position in positions]
    return group.loc[sorted(set(selected))].copy()


def _event_block_bootstrap_stability(
    embedded: np.ndarray,
    fit_rows: pd.DataFrame,
    reference_labels: np.ndarray,
    parameters: dict[str, Any],
    seed: int,
    repeats: int = 5,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    event_ids = fit_rows["event_id"].astype(str).to_numpy()
    unique_events = np.unique(event_ids)
    scores: list[float] = []
    cluster_counts: list[int] = []
    unknown_fractions: list[float] = []
    for _ in range(repeats):
        sampled_events = rng.choice(unique_events, len(unique_events), replace=True)
        sampled_indices = np.concatenate(
            [np.flatnonzero(event_ids == event_id) for event_id in sampled_events]
        )
        bootstrap_values = embedded[sampled_indices]
        labels = _fit_hdbscan_labels(bootstrap_values, parameters)
        if not np.any(labels >= 0):
            scores.append(float("nan"))
            cluster_counts.append(0)
            unknown_fractions.append(1.0)
            continue
        medoids, cutoffs = _cluster_medoids(bootstrap_values, labels)
        predicted, _ = _assign_to_medoids(embedded, medoids, cutoffs)
        comparable = (reference_labels >= 0) & (predicted >= 0)
        score = (
            float(adjusted_rand_score(reference_labels[comparable], predicted[comparable]))
            if comparable.sum() >= 2
            else float("nan")
        )
        scores.append(score)
        cluster_counts.append(int(len(set(labels[labels >= 0]))))
        unknown_fractions.append(float(np.mean(predicted < 0)))
    finite = np.asarray(scores, dtype=float)
    return {
        "repeats": repeats,
        "ari_values": scores,
        "median_ari": float(np.nanmedian(finite)) if np.isfinite(finite).any() else float("nan"),
        "cluster_counts": cluster_counts,
        "unknown_fractions": unknown_fractions,
    }


def _leave_one_mouse_out_stability(
    embedded: np.ndarray,
    fit_rows: pd.DataFrame,
    reference_labels: np.ndarray,
    parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    animals = fit_rows["animal_id"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for animal in sorted(np.unique(animals)):
        train = animals != animal
        test = ~train
        scaled_parameters = dict(parameters)
        scaled_parameters["min_cluster_size"] = max(
            20,
            int(round(parameters["min_cluster_size"] * train.sum() / len(train))),
        )
        scaled_parameters["min_samples"] = min(
            int(parameters["min_samples"]), scaled_parameters["min_cluster_size"]
        )
        labels = _fit_hdbscan_labels(embedded[train], scaled_parameters)
        if not np.any(labels >= 0):
            rows.append(
                {
                    "held_out_animal": animal,
                    "training_cluster_count": 0,
                    "held_out_unknown_fraction": 1.0,
                    "agreement_with_global_ari": float("nan"),
                }
            )
            continue
        medoids, cutoffs = _cluster_medoids(embedded[train], labels)
        predicted, _ = _assign_to_medoids(embedded[test], medoids, cutoffs)
        reference = reference_labels[test]
        comparable = (reference >= 0) & (predicted >= 0)
        ari = (
            float(adjusted_rand_score(reference[comparable], predicted[comparable]))
            if comparable.sum() >= 2
            else float("nan")
        )
        rows.append(
            {
                "held_out_animal": animal,
                "training_cluster_count": int(len(set(labels[labels >= 0]))),
                "held_out_unknown_fraction": float(np.mean(predicted < 0)),
                "agreement_with_global_ari": ari,
            }
        )
    return rows


def _fit_hdbscan_labels(
    embedded: np.ndarray, parameters: dict[str, Any]
) -> np.ndarray:
    model = HDBSCAN(
        min_cluster_size=min(int(parameters["min_cluster_size"]), max(2, len(embedded) - 1)),
        min_samples=min(int(parameters["min_samples"]), max(1, len(embedded) - 1)),
        cluster_selection_method=str(parameters.get("cluster_selection_method", "eom")),
        allow_single_cluster=bool(parameters.get("allow_single_cluster", False)),
        copy=False,
    )
    labels = model.fit_predict(embedded)
    mapping = _stable_cluster_map(labels)
    return np.asarray([mapping.get(int(label), -1) for label in labels], dtype=np.int32)


def _ensure_every_event(windows: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    represented = set(windows["event_id"].astype(str))
    missing = events.loc[~events["event_id"].astype(str).isin(represented)]
    if not missing.empty:
        missing_ids = ", ".join(missing["event_id"].astype(str).head(10))
        raise ValueError(f"Closed events are absent from the morphology index: {missing_ids}")
    return windows


def _event_phase(midpoint: float, start: float, end: float) -> str:
    duration = max(end - start, 1.0)
    fraction = np.clip((midpoint - start) / duration, 0.0, 1.0)
    if fraction < 0.15:
        return "early"
    if fraction > 0.85:
        return "late"
    return "middle"


def _empty_event_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "animal_id",
            "session_id",
            "event_id",
            "start_time_ms",
            "end_time_ms",
            "duration_s",
            "channels",
            "source_event_ids",
            "source_interval_count",
        ]
    )


__all__ = [
    "BANDS",
    "MorphologyConfig",
    "build_event_components",
    "build_morphology_index",
    "cluster_feature_columns",
    "extract_morphology_features",
    "fit_morphology_discovery",
    "summarize_morphologies",
]
