from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from mouse_sleep.epilepsy.baseline import add_baseline_relative_features
from mouse_sleep.epilepsy.morphology import (
    MorphologyConfig,
    _select_gmm_candidate,
    _select_review_gmm_candidate,
    _stable_cluster_map,
    _validated_gmm_grid,
)


PER_MOUSE_FEATURES = [
    "baseline_z__aggregate_robust_amplitude",
    "baseline_z__aggregate_line_length",
    "baseline_z__aggregate_log_rms",
    "baseline_z__aggregate_hjorth_mobility",
    "baseline_z__aggregate_hjorth_complexity",
    "baseline_z__aggregate_excursion_rate_hz",
    "baseline_z__aggregate_peak_frequency_hz",
    "baseline_z__aggregate_spectral_entropy",
    "baseline_z__relative_delta",
    "baseline_z__relative_theta",
    "baseline_z__relative_beta",
    "baseline_z__relative_low_gamma_clean",
    "baseline_z__relative_high_gamma",
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
for _channel in ("ch7", "ch8"):
    PER_MOUSE_FEATURES.extend(
        [
            f"baseline_z__{_channel}_log_robust_amplitude",
            f"baseline_z__{_channel}_log_line_length",
            f"baseline_z__{_channel}_log_rms",
            f"baseline_z__{_channel}_hjorth_mobility",
            f"baseline_z__{_channel}_hjorth_complexity",
            f"baseline_z__{_channel}_excursion_rate_hz",
            f"baseline_z__{_channel}_peak_frequency_hz",
            f"baseline_z__{_channel}_spectral_entropy",
            f"baseline_z__{_channel}_relative_delta",
            f"baseline_z__{_channel}_relative_theta",
            f"baseline_z__{_channel}_relative_beta",
            f"baseline_z__{_channel}_relative_low_gamma_clean",
            f"baseline_z__{_channel}_relative_high_gamma",
            f"{_channel}_log_robust_amplitude",
            f"{_channel}_log_line_length",
            f"{_channel}_hjorth_mobility",
            f"{_channel}_hjorth_complexity",
            f"{_channel}_peak_frequency_hz",
            f"{_channel}_spectral_entropy",
            f"{_channel}_relative_delta",
            f"{_channel}_relative_theta",
            f"{_channel}_relative_beta",
            f"{_channel}_relative_low_gamma_clean",
            f"{_channel}_relative_high_gamma",
        ]
    )


def fit_per_mouse_morphologies(
    seizure_features: pd.DataFrame,
    baseline_features: pd.DataFrame,
    output_dir: str | Path,
    base_config: MorphologyConfig | None = None,
    maximum_components: int = 8,
    fit_windows_per_event: int = 20,
    maximum_fit_windows: int = 5000,
    switch_penalty: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Discover morphology states within each mouse before any cross-mouse matching."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models" / "per_mouse"
    model_dir.mkdir(parents=True, exist_ok=True)
    enriched = add_baseline_relative_features(seizure_features, baseline_features)
    all_assignments: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []
    segment_frames: list[pd.DataFrame] = []
    type_frames: list[pd.DataFrame] = []
    base_config = base_config or MorphologyConfig()

    for animal, animal_rows in enriched.groupby("animal_id", sort=True):
        frame = animal_rows.sort_values(
            ["session_id", "event_start_ms", "start_time_ms"], kind="stable"
        ).reset_index(drop=True)
        columns = _usable_feature_columns(frame)
        raw = frame[columns].replace([np.inf, -np.inf], np.nan)
        fit_indices = _event_balanced_indices(
            frame,
            fit_windows_per_event,
            maximum_fit_windows,
            base_config.random_seed,
        )
        fit_rows = frame.iloc[fit_indices].reset_index(drop=True)
        raw_fit = raw.iloc[fit_indices].reset_index(drop=True)
        medians = raw_fit.median(axis=0).fillna(0.0)
        fit_values = raw_fit.fillna(medians).to_numpy(float)
        scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(fit_values)
        scaled_fit = scaler.transform(fit_values)
        components = min(
            base_config.pca_components,
            scaled_fit.shape[1],
            len(scaled_fit) - 1,
        )
        pca = PCA(
            n_components=components,
            whiten=True,
            random_state=base_config.random_seed,
        ).fit(scaled_fit)
        embedded_fit = pca.transform(scaled_fit)
        local_config = replace(
            base_config,
            gmm_max_components=maximum_components,
            gmm_bootstrap_repeats=max(base_config.gmm_bootstrap_repeats, 12),
            minimum_gmm_component_fraction=0.03,
            minimum_gmm_component_animals=1,
            minimum_gmm_component_events=3,
            minimum_component_windows_per_animal=1,
            maximum_component_single_animal_fraction=1.0,
            maximum_component_single_event_fraction=0.50,
            minimum_bootstrap_median_ari=0.60,
            minimum_bootstrap_q25_ari=0.40,
        )
        grid, models, fit_labels_by_candidate = _validated_gmm_grid(
            raw_fit, embedded_fit, fit_rows, local_config
        )
        selected = _select_gmm_candidate(grid)
        review = _select_review_gmm_candidate(grid, selected)
        primary_index = int(selected["candidate_index"])
        review_index = int(review["candidate_index"])
        grid["selected_primary"] = grid["candidate_index"].eq(primary_index)
        grid["selected_review_resolution"] = grid["candidate_index"].eq(review_index)
        grid.to_csv(
            output_dir / f"per_mouse_gmm_grid_{animal}.csv", index=False
        )
        all_values = pca.transform(
            scaler.transform(raw.fillna(medians).to_numpy(float))
        )
        primary = _assign_model(
            models[primary_index],
            fit_labels_by_candidate[primary_index],
            all_values,
        )
        exploratory = _assign_model(
            models[review_index],
            fit_labels_by_candidate[review_index],
            all_values,
        )
        primary_smoothed = _smooth_by_event(
            frame, primary["probabilities"], switch_penalty
        )
        review_smoothed = _smooth_by_event(
            frame, exploratory["probabilities"], switch_penalty
        )
        frame["individual_cluster"] = primary["labels"]
        frame["individual_label"] = [
            f"{animal}-T{label + 1}" for label in primary["labels"]
        ]
        frame["individual_probability"] = primary["maximum_probability"]
        frame["individual_smoothed_cluster"] = primary_smoothed
        frame["individual_smoothed_label"] = [
            f"{animal}-T{label + 1}" for label in primary_smoothed
        ]
        frame["individual_review_cluster"] = exploratory["labels"]
        frame["individual_review_label"] = [
            f"{animal}-C{label + 1}" for label in exploratory["labels"]
        ]
        frame["individual_review_probability"] = exploratory[
            "maximum_probability"
        ]
        frame["individual_review_smoothed_cluster"] = review_smoothed
        frame["individual_review_smoothed_label"] = [
            f"{animal}-C{label + 1}" for label in review_smoothed
        ]
        all_assignments.append(frame)
        segments = _build_segments(frame, "individual_review_smoothed_cluster")
        segment_frames.append(segments)
        type_frames.append(_summarize_types(frame, segments))
        model_rows.append(
            {
                "animal_id": str(animal),
                "all_windows": int(len(frame)),
                "all_events": int(frame["event_id"].nunique()),
                "fit_windows": int(len(fit_indices)),
                "fit_events": int(fit_rows["event_id"].nunique()),
                "usable_features": int(len(columns)),
                "stable_type_count": int(selected["components"]),
                "review_type_count": int(review["components"]),
                "stable_selection_basis": str(selected["selection_basis"]),
                "stable_cv_log_likelihood": float(
                    selected["cv_mean_log_likelihood"]
                ),
                "stable_bootstrap_median_ari": float(
                    selected["bootstrap_median_ari"]
                ),
                "stable_bootstrap_q25_ari": float(selected["bootstrap_q25_ari"]),
                "review_bootstrap_median_ari": float(
                    review["bootstrap_median_ari"]
                ),
                "review_support_ok": bool(review["support_ok"]),
                "review_stability_ok": bool(review["stability_ok"]),
            }
        )
        joblib.dump(
            {
                "animal_id": str(animal),
                "feature_columns": columns,
                "feature_medians": medians.to_dict(),
                "scaler": scaler,
                "pca": pca,
                "primary_gmm": models[primary_index],
                "primary_cluster_map": primary["cluster_map"],
                "review_gmm": models[review_index],
                "review_cluster_map": exploratory["cluster_map"],
                "switch_penalty": switch_penalty,
            },
            model_dir / f"{animal}_morphology.joblib",
        )

    assignments = pd.concat(all_assignments, ignore_index=True)
    model_summary = pd.DataFrame(model_rows)
    segments = pd.concat(segment_frames, ignore_index=True)
    type_summary = pd.concat(type_frames, ignore_index=True)
    assignments.to_parquet(
        output_dir / "per_mouse_morphology_assignments.parquet", index=False
    )
    model_summary.to_csv(
        output_dir / "per_mouse_model_selection.csv", index=False, encoding="utf-8-sig"
    )
    segments.to_csv(
        output_dir / "per_mouse_morphology_segments.csv", index=False, encoding="utf-8-sig"
    )
    type_summary.to_csv(
        output_dir / "per_mouse_type_summary.csv", index=False, encoding="utf-8-sig"
    )
    return assignments, model_summary, segments, type_summary


def _usable_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in PER_MOUSE_FEATURES:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().mean() < 0.50:
            continue
        finite = values[np.isfinite(values)]
        if len(finite) < 3 or float(finite.quantile(0.90) - finite.quantile(0.10)) <= 1e-9:
            continue
        columns.append(column)
    if len(columns) < 8:
        raise ValueError(f"Only {len(columns)} usable within-mouse features remain")
    return columns


def _event_balanced_indices(
    frame: pd.DataFrame,
    per_event: int,
    maximum: int,
    seed: int,
) -> np.ndarray:
    selected_by_event: list[list[int]] = []
    for _, event in frame.groupby("event_id", sort=True):
        eligible = event.loc[
            event["event_overlap_fraction"].ge(0.99)
            & event["valid_channel_count"].ge(1)
            & event["event_duration_s"].ge(4.0)
        ]
        if eligible.empty:
            continue
        count = min(per_event, len(eligible))
        positions = np.linspace(0, len(eligible) - 1, count, dtype=np.int64)
        selected_by_event.append(eligible.iloc[positions].index.astype(int).tolist())
    if not selected_by_event:
        raise ValueError("No fully contained seizure windows are available for this mouse")
    first_pass = [values[0] for values in selected_by_event]
    remaining = [value for values in selected_by_event for value in values[1:]]
    if len(first_pass) >= maximum:
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(first_pass, maximum, replace=False)).astype(np.int64)
    capacity = maximum - len(first_pass)
    if len(remaining) > capacity:
        rng = np.random.default_rng(seed)
        remaining = rng.choice(remaining, capacity, replace=False).tolist()
    return np.asarray(sorted(first_pass + remaining), dtype=np.int64)


def _assign_model(
    model: Any,
    fit_labels: np.ndarray,
    embedded: np.ndarray,
) -> dict[str, Any]:
    cluster_map = _stable_cluster_map(fit_labels)
    raw_labels = model.predict(embedded)
    labels = np.asarray([cluster_map[int(label)] for label in raw_labels], dtype=np.int32)
    raw_probabilities = model.predict_proba(embedded)
    probabilities = np.empty_like(raw_probabilities)
    for raw_label, stable_label in cluster_map.items():
        probabilities[:, stable_label] = raw_probabilities[:, raw_label]
    return {
        "labels": labels,
        "probabilities": probabilities,
        "maximum_probability": probabilities.max(axis=1),
        "cluster_map": cluster_map,
    }


def _smooth_by_event(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    switch_penalty: float,
) -> np.ndarray:
    result = np.zeros(len(frame), dtype=np.int32)
    for _, event in frame.groupby("event_id", sort=False):
        indices = event.sort_values("start_time_ms").index.to_numpy(np.int64)
        result[indices] = _viterbi_smooth(probabilities[indices], switch_penalty)
    return result


def _viterbi_smooth(probabilities: np.ndarray, switch_penalty: float) -> np.ndarray:
    if probabilities.shape[1] == 1:
        return np.zeros(len(probabilities), dtype=np.int32)
    emission = np.log(np.maximum(probabilities, 1e-12))
    score = np.empty_like(emission)
    back = np.zeros_like(emission, dtype=np.int32)
    score[0] = emission[0]
    states = np.arange(probabilities.shape[1])
    for time in range(1, len(probabilities)):
        transitions = score[time - 1][:, None] - switch_penalty * (
            states[:, None] != states[None, :]
        )
        back[time] = np.argmax(transitions, axis=0)
        score[time] = emission[time] + transitions[back[time], states]
    path = np.empty(len(probabilities), dtype=np.int32)
    path[-1] = int(np.argmax(score[-1]))
    for time in range(len(probabilities) - 2, -1, -1):
        path[time] = back[time + 1, path[time + 1]]
    return path


def _build_segments(frame: pd.DataFrame, cluster_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event_id, event in frame.groupby("event_id", sort=True):
        ordered = event.sort_values("start_time_ms")
        labels = ordered[cluster_column].astype(int).to_numpy()
        start = 0
        segment_index = 1
        for position in range(1, len(ordered) + 1):
            boundary = position == len(ordered) or labels[position] != labels[start]
            if not boundary:
                continue
            segment = ordered.iloc[start:position]
            label = int(labels[start])
            rows.append(
                {
                    "animal_id": str(segment["animal_id"].iloc[0]),
                    "session_id": str(segment["session_id"].iloc[0]),
                    "event_id": str(event_id),
                    "segment_id": f"{event_id}:segment{segment_index:03d}",
                    "candidate_cluster": label,
                    "candidate_label": f"{segment['animal_id'].iloc[0]}-C{label + 1}",
                    "start_time_ms": float(segment["start_time_ms"].iloc[0]),
                    "end_time_ms": float(segment["end_time_ms"].iloc[-1]),
                    "duration_s": float(
                        (segment["end_time_ms"].iloc[-1] - segment["start_time_ms"].iloc[0])
                        / 1000.0
                    ),
                    "windows": int(len(segment)),
                    "mean_probability": float(
                        segment["individual_review_probability"].mean()
                    ),
                    "amplitude_vs_baseline_median": float(
                        segment["amplitude_vs_baseline"].median()
                    ),
                }
            )
            start = position
            segment_index += 1
    return pd.DataFrame(rows)


def _summarize_types(frame: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (cluster, label), group in frame.groupby(
        ["individual_review_smoothed_cluster", "individual_review_smoothed_label"],
        sort=True,
    ):
        type_segments = segments.loc[segments["candidate_cluster"].eq(cluster)]
        rows.append(
            {
                "animal_id": str(group["animal_id"].iloc[0]),
                "candidate_cluster": int(cluster),
                "candidate_label": str(label),
                "windows": int(len(group)),
                "events": int(group["event_id"].nunique()),
                "segments": int(len(type_segments)),
                "window_fraction": float(len(group) / len(frame)),
                "median_segment_duration_s": float(type_segments["duration_s"].median()),
                "median_probability": float(group["individual_review_probability"].median()),
                "median_amplitude_vs_baseline": float(group["amplitude_vs_baseline"].median()),
                "median_peak_frequency_hz": float(group["aggregate_peak_frequency_hz"].median()),
                "median_spectral_entropy": float(group["aggregate_spectral_entropy"].median()),
                "median_channel_correlation": float(group["channel_correlation"].median()),
            }
        )
    return pd.DataFrame(rows)


__all__ = ["fit_per_mouse_morphologies"]
