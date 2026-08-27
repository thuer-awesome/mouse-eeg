from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from aeon.transformations.collection.convolution_based import MiniRocket
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import RobustScaler, StandardScaler


AUXILIARY_FEATURES = (
    "amplitude_vs_reference_log",
    "line_length_vs_reference_log",
    "aggregate_peak_frequency_hz",
    "aggregate_spectral_entropy",
    "relative_delta",
    "relative_theta",
    "relative_beta",
    "relative_low_gamma_clean",
    "theta_delta_ratio",
    "channel_correlation",
    "spectral_cosine_similarity",
    "amplitude_log_ratio",
)

EXPLAINABLE_FEATURES = (
    "amplitude_vs_reference_log",
    "line_length_vs_reference_log",
    "aggregate_peak_frequency_hz",
    "aggregate_median_frequency_hz",
    "aggregate_spectral_entropy",
    "aggregate_crest_factor",
    "aggregate_hjorth_mobility",
    "aggregate_hjorth_complexity",
    "relative_delta",
    "relative_theta",
    "relative_beta",
    "relative_low_gamma_clean",
    "theta_delta_ratio",
    "high_low_ratio",
    "channel_correlation",
    "spectral_cosine_similarity",
    "amplitude_log_ratio",
    "line_length_log_ratio",
)


@dataclass(frozen=True)
class TemporalTaxonomyConfig:
    window_seconds: float = 1.0
    stride_seconds: float = 1.0
    minimum_k: int = 1
    maximum_k: int = 8
    rocket_kernels: int = 5000
    rocket_pca_variance: float = 0.90
    rocket_pca_max_components: int = 20
    auxiliary_pca_variance: float = 0.90
    auxiliary_pca_max_components: int = 8
    shape_branch_weight: float = 0.65
    auxiliary_branch_weight: float = 0.35
    fit_windows_per_event: int = 20
    maximum_fit_windows: int = 5000
    cv_folds: int = 5
    bootstrap_repeats: int = 50
    minimum_state_windows: int = 10
    minimum_state_events: int = 3
    maximum_single_event_fraction: float = 0.50
    bootstrap_ari_median: float = 0.60
    bootstrap_ari_q25: float = 0.40
    channel_valid_fraction: float = 0.50
    gmm_reg_covar: float = 1e-4
    event_trajectory_bins: int = 20
    maximum_event_types: int = 6
    minimum_events_per_type: int = 3
    random_seed: int = 20260812

    def validated(self) -> "TemporalTaxonomyConfig":
        if self.minimum_k != 1:
            raise ValueError("K=1 must remain available as the null state model")
        if not 1 <= self.maximum_k <= 20:
            raise ValueError("maximum_k must be between 1 and 20")
        if not np.isclose(
            self.shape_branch_weight + self.auxiliary_branch_weight, 1.0
        ):
            raise ValueError("The two branch weights must sum to one")
        if self.window_seconds <= 0 or self.stride_seconds <= 0:
            raise ValueError("Native window and stride durations must be positive")
        if self.fit_windows_per_event < 1 or self.maximum_fit_windows < 10:
            raise ValueError("Fit sampling limits are too small")
        if self.bootstrap_repeats < 1:
            raise ValueError("At least one event bootstrap repeat is required")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepresentationArtifacts:
    animal_id: str
    active_channel_indices: tuple[int, ...]
    active_channel_names: tuple[str, ...]
    auxiliary_columns: tuple[str, ...]
    rocket: MiniRocket
    rocket_scaler: StandardScaler
    rocket_pca: PCA
    auxiliary_medians: dict[str, float]
    auxiliary_scaler: RobustScaler
    auxiliary_pca: PCA
    shape_center: np.ndarray
    shape_scale: float
    auxiliary_center: np.ndarray
    auxiliary_scale: float
    shape_weight: float
    auxiliary_weight: float


@dataclass
class StateTaxonomyResult:
    animal_id: str
    selected_k: int
    grid: pd.DataFrame
    assignments: pd.DataFrame
    segments: pd.DataFrame
    state_summary: pd.DataFrame
    fit_indices: np.ndarray
    full_embedding: np.ndarray
    model_bundle: dict[str, Any]


def event_balanced_indices(
    rows: pd.DataFrame,
    per_event: int,
    maximum: int,
    seed: int,
) -> np.ndarray:
    """Return an exactly event-balanced sample, resampling short events."""

    if rows.empty:
        raise ValueError("No native seizure windows were supplied")
    events = [group.index.to_numpy(np.int64) for _, group in rows.groupby("event_id")]
    events = [indices for indices in events if len(indices)]
    if not events:
        raise ValueError("No resolved seizure events were supplied")
    target = min(per_event, max(1, maximum // len(events)))
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for indices in events:
        if len(indices) >= target:
            positions = np.linspace(0, len(indices) - 1, target, dtype=np.int64)
            chosen = indices[positions]
        else:
            chosen = rng.choice(indices, target, replace=True)
        selected.append(np.asarray(chosen, dtype=np.int64))
    return np.concatenate(selected)


def active_channels_for_mouse(
    rows: pd.DataFrame,
    minimum_valid_fraction: float = 0.50,
) -> tuple[int, ...]:
    fractions = []
    for column in ("ch7_valid", "ch8_valid"):
        if column not in rows:
            fractions.append(0.0)
        else:
            fractions.append(float(pd.to_numeric(rows[column], errors="coerce").mean()))
    active = tuple(index for index, value in enumerate(fractions) if value >= minimum_valid_fraction)
    if not active:
        active = (int(np.argmax(fractions)),)
    return active


def normalize_native_shapes(
    windows: np.ndarray,
    rows: pd.DataFrame,
    active_channels: Iterable[int],
) -> np.ndarray:
    """Normalize amplitude per native window while preserving temporal shape."""

    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(f"Expected [windows, 2, samples], received {values.shape}")
    if len(rows) != len(values):
        raise ValueError("Signal and metadata row counts do not match")
    channel_indices = tuple(int(index) for index in active_channels)
    selected = values[:, channel_indices].copy()
    centered = selected - np.median(selected, axis=2, keepdims=True)
    q01 = np.quantile(centered, 0.01, axis=2, keepdims=True)
    q99 = np.quantile(centered, 0.99, axis=2, keepdims=True)
    scale = np.maximum(q99 - q01, 1e-6)
    normalized = np.clip(centered / scale, -4.0, 4.0)
    for local_index, channel_index in enumerate(channel_indices):
        valid = rows[f"ch{channel_index + 7}_valid"].to_numpy(bool)
        normalized[~valid, local_index] = 0.0
    differences = np.diff(normalized, axis=2, prepend=normalized[:, :, :1])
    return np.concatenate([normalized, differences], axis=1).astype(np.float32)


def _fit_variance_pca(
    values: np.ndarray,
    variance_target: float,
    maximum_components: int,
    seed: int,
) -> tuple[PCA, np.ndarray]:
    maximum = min(maximum_components, values.shape[1], len(values) - 1)
    if maximum < 1:
        raise ValueError("PCA needs at least two rows and one varying feature")
    probe = PCA(n_components=maximum, whiten=False, random_state=seed).fit(values)
    cumulative = np.cumsum(probe.explained_variance_ratio_)
    components = min(maximum, int(np.searchsorted(cumulative, variance_target) + 1))
    if components == maximum:
        return probe, probe.transform(values)
    model = PCA(n_components=components, whiten=False, random_state=seed).fit(values)
    return model, model.transform(values)


def _normalize_branch(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = np.median(values, axis=0)
    centered = values - center
    scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    scale = max(scale, 1e-12)
    return centered / scale, center, scale


def _usable_auxiliary_columns(
    rows: pd.DataFrame,
    requested: Iterable[str] = AUXILIARY_FEATURES,
) -> tuple[str, ...]:
    columns: list[str] = []
    for column in requested:
        if column not in rows:
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if values.notna().mean() < 0.50 or len(finite) < 3:
            continue
        if float(finite.quantile(0.90) - finite.quantile(0.10)) <= 1e-9:
            continue
        columns.append(column)
    if not columns:
        raise ValueError("No usable physiology features are available")
    return tuple(columns)


def fit_multiview_representation(
    animal_id: str,
    windows: np.ndarray,
    rows: pd.DataFrame,
    fit_indices: np.ndarray,
    config: TemporalTaxonomyConfig,
) -> tuple[RepresentationArtifacts, np.ndarray]:
    config.validated()
    active = active_channels_for_mouse(rows, config.channel_valid_fraction)
    fit_rows = rows.iloc[fit_indices].reset_index(drop=True)
    fit_windows = np.asarray(windows[fit_indices], dtype=np.float32)
    shape_input = normalize_native_shapes(fit_windows, fit_rows, active)
    rocket = MiniRocket(
        n_kernels=config.rocket_kernels,
        n_jobs=-1,
        random_state=config.random_seed,
    )
    rocket_features = np.asarray(rocket.fit_transform(shape_input), dtype=np.float32)
    rocket_scaler = StandardScaler().fit(rocket_features)
    rocket_scaled = np.clip(rocket_scaler.transform(rocket_features), -8.0, 8.0)
    rocket_pca, shape_embedding = _fit_variance_pca(
        rocket_scaled,
        config.rocket_pca_variance,
        config.rocket_pca_max_components,
        config.random_seed,
    )

    auxiliary_columns = _usable_auxiliary_columns(fit_rows)
    auxiliary = fit_rows.loc[:, auxiliary_columns].apply(pd.to_numeric, errors="coerce")
    auxiliary_medians = auxiliary.median(axis=0).fillna(0.0)
    auxiliary_values = auxiliary.fillna(auxiliary_medians).to_numpy(float)
    auxiliary_scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(auxiliary_values)
    auxiliary_scaled = np.clip(auxiliary_scaler.transform(auxiliary_values), -8.0, 8.0)
    auxiliary_pca, auxiliary_embedding = _fit_variance_pca(
        auxiliary_scaled,
        config.auxiliary_pca_variance,
        config.auxiliary_pca_max_components,
        config.random_seed,
    )
    shape_branch, shape_center, shape_scale = _normalize_branch(shape_embedding)
    auxiliary_branch, auxiliary_center, auxiliary_scale = _normalize_branch(
        auxiliary_embedding
    )
    combined = np.concatenate(
        [
            np.sqrt(config.shape_branch_weight) * shape_branch,
            np.sqrt(config.auxiliary_branch_weight) * auxiliary_branch,
        ],
        axis=1,
    ).astype(np.float32)
    artifacts = RepresentationArtifacts(
        animal_id=str(animal_id),
        active_channel_indices=active,
        active_channel_names=tuple(f"CH{index + 7}" for index in active),
        auxiliary_columns=auxiliary_columns,
        rocket=rocket,
        rocket_scaler=rocket_scaler,
        rocket_pca=rocket_pca,
        auxiliary_medians={key: float(value) for key, value in auxiliary_medians.items()},
        auxiliary_scaler=auxiliary_scaler,
        auxiliary_pca=auxiliary_pca,
        shape_center=np.asarray(shape_center),
        shape_scale=float(shape_scale),
        auxiliary_center=np.asarray(auxiliary_center),
        auxiliary_scale=float(auxiliary_scale),
        shape_weight=float(config.shape_branch_weight),
        auxiliary_weight=float(config.auxiliary_branch_weight),
    )
    return artifacts, combined


def transform_multiview_representation(
    artifacts: RepresentationArtifacts,
    windows: np.ndarray,
    rows: pd.DataFrame,
    batch_size: int = 1000,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        stop = min(len(rows), start + batch_size)
        batch_rows = rows.iloc[start:stop].reset_index(drop=True)
        shape_input = normalize_native_shapes(
            np.asarray(windows[start:stop]),
            batch_rows,
            artifacts.active_channel_indices,
        )
        rocket_values = np.asarray(artifacts.rocket.transform(shape_input), dtype=np.float32)
        rocket_values = np.clip(
            artifacts.rocket_scaler.transform(rocket_values), -8.0, 8.0
        )
        shape = artifacts.rocket_pca.transform(rocket_values)
        shape = (shape - artifacts.shape_center) / artifacts.shape_scale

        auxiliary = batch_rows.loc[:, artifacts.auxiliary_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        auxiliary = auxiliary.fillna(artifacts.auxiliary_medians).to_numpy(float)
        auxiliary = np.clip(artifacts.auxiliary_scaler.transform(auxiliary), -8.0, 8.0)
        auxiliary = artifacts.auxiliary_pca.transform(auxiliary)
        auxiliary = (
            auxiliary - artifacts.auxiliary_center
        ) / artifacts.auxiliary_scale
        blocks.append(
            np.concatenate(
                [
                    np.sqrt(artifacts.shape_weight) * shape,
                    np.sqrt(artifacts.auxiliary_weight) * auxiliary,
                ],
                axis=1,
            ).astype(np.float32)
        )
    return np.concatenate(blocks, axis=0)


def _fit_gmm(values: np.ndarray, k: int, seed: int, reg_covar: float, n_init: int) -> GaussianMixture:
    return GaussianMixture(
        n_components=k,
        covariance_type="diag",
        reg_covar=reg_covar,
        n_init=n_init,
        max_iter=300,
        random_state=seed,
    ).fit(values)


def _event_group_cv(
    values: np.ndarray,
    event_ids: np.ndarray,
    k: int,
    config: TemporalTaxonomyConfig,
) -> tuple[float, float, int]:
    unique_events = np.unique(event_ids)
    if len(unique_events) < 2:
        return float("nan"), float("nan"), 0
    folds = min(config.cv_folds, len(unique_events))
    splitter = GroupKFold(n_splits=folds)
    event_scores: list[float] = []
    for fold, (train, test) in enumerate(splitter.split(values, groups=event_ids)):
        if len(train) <= k:
            continue
        try:
            model = _fit_gmm(
                values[train],
                k,
                config.random_seed + 1000 * k + fold,
                config.gmm_reg_covar,
                n_init=3,
            )
        except ValueError:
            continue
        scores = model.score_samples(values[test])
        test_events = event_ids[test]
        for event_id in np.unique(test_events):
            event_scores.append(float(np.mean(scores[test_events == event_id])))
    if not event_scores:
        return float("nan"), float("nan"), 0
    values_array = np.asarray(event_scores, dtype=float)
    standard_error = (
        float(np.std(values_array, ddof=1) / np.sqrt(len(values_array)))
        if len(values_array) > 1
        else 0.0
    )
    return float(np.mean(values_array)), standard_error, len(values_array)


def _event_bootstrap_stability(
    values: np.ndarray,
    event_ids: np.ndarray,
    reference_labels: np.ndarray,
    k: int,
    config: TemporalTaxonomyConfig,
) -> np.ndarray:
    if k == 1:
        return np.ones(config.bootstrap_repeats, dtype=float)
    unique_events = np.unique(event_ids)
    rng = np.random.default_rng(config.random_seed + 10000 * k)
    scores: list[float] = []
    for repeat in range(config.bootstrap_repeats):
        sampled = rng.choice(unique_events, len(unique_events), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(event_ids == event_id) for event_id in sampled]
        )
        try:
            model = _fit_gmm(
                values[indices],
                k,
                config.random_seed + 20000 * k + repeat,
                config.gmm_reg_covar,
                n_init=2,
            )
            scores.append(
                float(adjusted_rand_score(reference_labels, model.predict(values)))
            )
        except ValueError:
            scores.append(0.0)
    return np.asarray(scores, dtype=float)


def evaluate_state_grid(
    values: np.ndarray,
    event_ids: np.ndarray,
    config: TemporalTaxonomyConfig,
) -> tuple[pd.DataFrame, dict[int, GaussianMixture]]:
    event_ids = np.asarray(event_ids, dtype=str)
    maximum_k = min(config.maximum_k, len(values) - 1)
    rows: list[dict[str, Any]] = []
    models: dict[int, GaussianMixture] = {}
    for k in range(config.minimum_k, maximum_k + 1):
        model = _fit_gmm(
            values,
            k,
            config.random_seed + k,
            config.gmm_reg_covar,
            n_init=10,
        )
        labels = model.predict(values)
        models[k] = model
        counts = np.bincount(labels, minlength=k)
        state_event_counts: list[int] = []
        single_event_fractions: list[float] = []
        for state in range(k):
            state_events = pd.Series(event_ids[labels == state]).value_counts()
            state_event_counts.append(int(len(state_events)))
            single_event_fractions.append(
                float(state_events.max() / state_events.sum()) if len(state_events) else 1.0
            )
        bootstrap = _event_bootstrap_stability(
            values, event_ids, labels, k, config
        )
        cv_mean, cv_se, cv_events = _event_group_cv(
            values, event_ids, k, config
        )
        if k == 1:
            support_ok = True
            silhouette = float("nan")
        else:
            support_ok = bool(
                counts.min() >= config.minimum_state_windows
                and min(state_event_counts) >= config.minimum_state_events
                and max(single_event_fractions) <= config.maximum_single_event_fraction
                and np.median(bootstrap) >= config.bootstrap_ari_median
                and np.quantile(bootstrap, 0.25) >= config.bootstrap_ari_q25
            )
            sample = np.linspace(0, len(values) - 1, min(5000, len(values)), dtype=int)
            silhouette = float(silhouette_score(values[sample], labels[sample]))
        rows.append(
            {
                "k": k,
                "fit_windows": int(len(values)),
                "fit_events": int(len(np.unique(event_ids))),
                "minimum_state_windows": int(counts.min()),
                "minimum_state_fraction": float(counts.min() / len(labels)),
                "minimum_state_events": int(min(state_event_counts)),
                "maximum_single_event_fraction": float(max(single_event_fractions)),
                "event_cv_log_likelihood": cv_mean,
                "event_cv_standard_error": cv_se,
                "event_cv_events": int(cv_events),
                "event_bootstrap_ari_median": float(np.median(bootstrap)),
                "event_bootstrap_ari_q25": float(np.quantile(bootstrap, 0.25)),
                "silhouette": silhouette,
                "support_ok": support_ok,
            }
        )
    grid = pd.DataFrame(rows)
    grid["selected"] = False
    supported = grid.loc[grid["support_ok"] & grid["event_cv_log_likelihood"].notna()]
    if supported.empty:
        selected_k = 1
    else:
        best = supported.loc[supported["event_cv_log_likelihood"].idxmax()]
        threshold = float(best["event_cv_log_likelihood"] - best["event_cv_standard_error"])
        eligible = supported.loc[supported["event_cv_log_likelihood"] >= threshold]
        selected_k = int(eligible["k"].min())
    grid.loc[grid["k"].eq(selected_k), "selected"] = True
    return grid, models


def _stable_component_order(
    model: GaussianMixture,
    values: np.ndarray,
    rows: pd.DataFrame,
) -> tuple[int, ...]:
    labels = model.predict(values)
    amplitude = pd.to_numeric(
        rows.get("amplitude_vs_reference_log", pd.Series(np.zeros(len(rows)))),
        errors="coerce",
    ).to_numpy(float)
    frequency = pd.to_numeric(
        rows.get("aggregate_peak_frequency_hz", pd.Series(np.zeros(len(rows)))),
        errors="coerce",
    ).to_numpy(float)
    keys: list[tuple[float, float, int]] = []
    for state in range(model.n_components):
        mask = labels == state
        keys.append(
            (
                float(np.nanmedian(amplitude[mask])) if np.any(mask) else np.inf,
                float(np.nanmedian(frequency[mask])) if np.any(mask) else np.inf,
                state,
            )
        )
    return tuple(item[2] for item in sorted(keys))


def _sequence_groups(rows: pd.DataFrame, expected_stride_s: float) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    for _, event in rows.groupby("event_id", sort=False):
        event = event.sort_values("start_time_ms", kind="stable")
        indices = event.index.to_numpy(np.int64)
        times = event["start_time_ms"].to_numpy(float)
        breaks = np.flatnonzero(np.diff(times) > 1.5 * expected_stride_s * 1000.0) + 1
        groups.extend(np.split(indices, breaks))
    return [group for group in groups if len(group)]


def _estimate_event_equal_markov(
    posterior: np.ndarray,
    rows: pd.DataFrame,
    stride_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    k = posterior.shape[1]
    starts: list[np.ndarray] = []
    transitions: list[np.ndarray] = []
    for indices in _sequence_groups(rows, stride_seconds):
        labels = np.argmax(posterior[indices], axis=1)
        start = np.ones(k, dtype=float)
        start[labels[0]] += 1.0
        starts.append(start / start.sum())
        matrix = np.ones((k, k), dtype=float)
        matrix[np.diag_indices(k)] += 2.0
        for previous, current in zip(labels[:-1], labels[1:]):
            matrix[int(previous), int(current)] += 1.0
        matrix /= matrix.sum(axis=1, keepdims=True)
        transitions.append(matrix)
    start_probability = np.mean(starts, axis=0) if starts else np.full(k, 1.0 / k)
    transition_probability = (
        np.mean(transitions, axis=0)
        if transitions
        else np.full((k, k), 1.0 / k)
    )
    start_probability /= start_probability.sum()
    transition_probability /= transition_probability.sum(axis=1, keepdims=True)
    return start_probability, transition_probability


def viterbi_decode(
    log_emission: np.ndarray,
    start_probability: np.ndarray,
    transition_probability: np.ndarray,
) -> np.ndarray:
    emissions = np.asarray(log_emission, dtype=float)
    if emissions.ndim != 2:
        raise ValueError("Viterbi emissions must have shape [time, states]")
    timepoints, states = emissions.shape
    if timepoints == 0:
        return np.empty(0, dtype=np.int32)
    log_start = np.log(np.maximum(start_probability, 1e-12))
    log_transition = np.log(np.maximum(transition_probability, 1e-12))
    score = np.empty((timepoints, states), dtype=float)
    back = np.zeros((timepoints, states), dtype=np.int32)
    score[0] = log_start + emissions[0]
    for time in range(1, timepoints):
        candidate = score[time - 1, :, None] + log_transition
        back[time] = np.argmax(candidate, axis=0)
        score[time] = emissions[time] + candidate[back[time], np.arange(states)]
    path = np.empty(timepoints, dtype=np.int32)
    path[-1] = int(np.argmax(score[-1]))
    for time in range(timepoints - 2, -1, -1):
        path[time] = back[time + 1, path[time + 1]]
    return path


def _build_state_segments(
    assignments: pd.DataFrame,
    stride_seconds: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    segment_number = 0
    for sequence_number, indices in enumerate(
        _sequence_groups(assignments, stride_seconds), start=1
    ):
        sequence = assignments.loc[indices].sort_values("start_time_ms", kind="stable")
        labels = sequence["state"].to_numpy(int)
        starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
        stops = np.r_[starts[1:], len(sequence)]
        for start, stop in zip(starts, stops):
            segment_number += 1
            block = sequence.iloc[start:stop]
            records.append(
                {
                    "animal_id": str(block["animal_id"].iloc[0]),
                    "session_id": str(block["session_id"].iloc[0]),
                    "event_id": str(block["event_id"].iloc[0]),
                    "sequence_id": f"seq{sequence_number:05d}",
                    "segment_id": f"seg{segment_number:06d}",
                    "state": int(labels[start]),
                    "state_label": str(block["state_label"].iloc[0]),
                    "start_time_ms": float(block["start_time_ms"].iloc[0]),
                    "end_time_ms": float(block["end_time_ms"].iloc[-1]),
                    "duration_s": float(
                        (block["end_time_ms"].iloc[-1] - block["start_time_ms"].iloc[0])
                        / 1000.0
                    ),
                    "window_count": int(len(block)),
                    "mean_posterior": float(block["state_posterior"].mean()),
                }
            )
    return pd.DataFrame(records)


def _state_summary(
    assignments: pd.DataFrame,
    full_embedding: np.ndarray,
    explainable_features: Iterable[str] = EXPLAINABLE_FEATURES,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state, group in assignments.groupby("state", sort=True):
        indices = group.index.to_numpy(np.int64)
        center = np.mean(full_embedding[indices], axis=0)
        distances = np.linalg.norm(full_embedding[indices] - center, axis=1)
        representative = int(indices[int(np.argmin(distances))])
        record: dict[str, Any] = {
            "animal_id": str(group["animal_id"].iloc[0]),
            "state": int(state),
            "state_label": str(group["state_label"].iloc[0]),
            "windows": int(len(group)),
            "events": int(group["event_id"].nunique()),
            "fraction": float(len(group) / len(assignments)),
            "median_posterior": float(group["state_posterior"].median()),
            "representative_row_index": representative,
        }
        for feature in explainable_features:
            if feature in group:
                values = pd.to_numeric(group[feature], errors="coerce")
                record[f"median__{feature}"] = float(values.median())
                record[f"q25__{feature}"] = float(values.quantile(0.25))
                record[f"q75__{feature}"] = float(values.quantile(0.75))
        rows.append(record)
    return pd.DataFrame(rows)


def fit_mouse_state_taxonomy(
    animal_id: str,
    windows: np.ndarray,
    rows: pd.DataFrame,
    config: TemporalTaxonomyConfig | None = None,
) -> StateTaxonomyResult:
    config = (config or TemporalTaxonomyConfig()).validated()
    frame = rows.reset_index(drop=True).copy()
    if len(frame) != len(windows):
        raise ValueError("Native windows and metadata rows do not match")
    if frame["animal_id"].astype(str).nunique() != 1:
        raise ValueError("State taxonomy must be fit independently for one mouse")
    fit_indices = event_balanced_indices(
        frame,
        config.fit_windows_per_event,
        config.maximum_fit_windows,
        config.random_seed,
    )
    artifacts, fit_embedding = fit_multiview_representation(
        animal_id,
        windows,
        frame,
        fit_indices,
        config,
    )
    fit_rows = frame.iloc[fit_indices].reset_index(drop=True)
    event_ids = fit_rows["event_id"].astype(str).to_numpy()
    grid, models = evaluate_state_grid(fit_embedding, event_ids, config)
    selected_k = int(grid.loc[grid["selected"], "k"].iloc[0])
    selected_model = models[selected_k]
    order = _stable_component_order(selected_model, fit_embedding, fit_rows)

    full_embedding = transform_multiview_representation(
        artifacts, windows, frame
    )
    raw_posterior = selected_model.predict_proba(full_embedding)
    posterior = raw_posterior[:, order]
    start_probability, transition_probability = _estimate_event_equal_markov(
        posterior, frame, config.stride_seconds
    )
    raw_state = np.argmax(posterior, axis=1).astype(np.int32)
    state = np.empty(len(frame), dtype=np.int32)
    for indices in _sequence_groups(frame, config.stride_seconds):
        state[indices] = viterbi_decode(
            np.log(np.maximum(posterior[indices], 1e-12)),
            start_probability,
            transition_probability,
        )
    assignments = frame.copy()
    assignments["raw_state"] = raw_state
    assignments["state"] = state
    assignments["state_label"] = [f"{animal_id}-S{value + 1}" for value in state]
    assignments["state_posterior"] = posterior[np.arange(len(state)), state]
    assignments["posterior_entropy"] = -np.sum(
        posterior * np.log(np.maximum(posterior, 1e-12)), axis=1
    )
    for state_index in range(selected_k):
        assignments[f"state_probability_{state_index + 1}"] = posterior[:, state_index]
    segments = _build_state_segments(assignments, config.stride_seconds)
    summary = _state_summary(assignments, full_embedding)
    bundle = {
        "animal_id": str(animal_id),
        "config": config.to_dict(),
        "representation": artifacts,
        "gmm": selected_model,
        "component_order": order,
        "start_probability": start_probability,
        "transition_probability": transition_probability,
    }
    return StateTaxonomyResult(
        animal_id=str(animal_id),
        selected_k=selected_k,
        grid=grid.assign(animal_id=str(animal_id)),
        assignments=assignments,
        segments=segments,
        state_summary=summary,
        fit_indices=fit_indices,
        full_embedding=full_embedding,
        model_bundle=bundle,
    )


def _event_feature_matrix(
    assignments: pd.DataFrame,
    selected_k: int,
    bins: int,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    names = [
        f"trajectory_t{time + 1:02d}_s{state + 1}"
        for time in range(bins)
        for state in range(selected_k)
    ]
    names += [f"occupancy_s{state + 1}" for state in range(selected_k)]
    names += ["transition_count", "log_duration_s"]
    target = np.linspace(0.0, 1.0, bins)
    for event_id, event in assignments.groupby("event_id", sort=True):
        event = event.sort_values("start_time_ms", kind="stable")
        start = float(event["event_start_ms"].iloc[0])
        end = float(event["event_end_ms"].iloc[0])
        denominator = max(end - start, 1.0)
        position = np.clip(
            (event["start_time_ms"].to_numpy(float) - start) / denominator,
            0.0,
            1.0,
        )
        trajectory: list[np.ndarray] = []
        for state in range(selected_k):
            probability = event[f"state_probability_{state + 1}"].to_numpy(float)
            trajectory.append(np.interp(target, position, probability))
        trajectory_values = np.column_stack(trajectory).reshape(-1)
        state_path = event["state"].to_numpy(int)
        occupancy = np.bincount(state_path, minlength=selected_k) / len(state_path)
        transitions = float(np.sum(state_path[1:] != state_path[:-1]))
        duration_s = max((end - start) / 1000.0, 1e-3)
        vector = np.concatenate(
            [trajectory_values, occupancy, [transitions, np.log(duration_s)]]
        )
        vectors.append(vector)
        record: dict[str, Any] = {
            "animal_id": str(event["animal_id"].iloc[0]),
            "session_id": str(event["session_id"].iloc[0]),
            "event_id": str(event_id),
            "event_start_ms": start,
            "event_end_ms": end,
            "duration_s": duration_s,
            "window_count": int(len(event)),
            "transition_count": int(transitions),
        }
        for feature in EXPLAINABLE_FEATURES:
            if feature in event:
                values = pd.to_numeric(event[feature], errors="coerce")
                record[f"median__{feature}"] = float(values.median())
                record[f"iqr__{feature}"] = float(
                    values.quantile(0.75) - values.quantile(0.25)
                )
        records.append(record)
    return pd.DataFrame(records), np.asarray(vectors, dtype=float), names


def _event_type_grid(
    values: np.ndarray,
    config: TemporalTaxonomyConfig,
) -> tuple[pd.DataFrame, dict[int, GaussianMixture]]:
    events = len(values)
    maximum = min(config.maximum_event_types, max(1, events // 3))
    rows: list[dict[str, Any]] = []
    models: dict[int, GaussianMixture] = {}
    folds = min(config.cv_folds, events)
    for count in range(1, maximum + 1):
        model = _fit_gmm(
            values,
            count,
            config.random_seed + 50000 + count,
            config.gmm_reg_covar,
            n_init=20,
        )
        labels = model.predict(values)
        models[count] = model
        sizes = np.bincount(labels, minlength=count)
        cv_scores: list[float] = []
        if folds >= 2:
            splitter = KFold(
                n_splits=folds, shuffle=True, random_state=config.random_seed
            )
            for fold, (train, test) in enumerate(splitter.split(values)):
                if len(train) <= count:
                    continue
                try:
                    fold_model = _fit_gmm(
                        values[train],
                        count,
                        config.random_seed + 51000 + count * 100 + fold,
                        config.gmm_reg_covar,
                        n_init=3,
                    )
                    cv_scores.extend(fold_model.score_samples(values[test]).tolist())
                except ValueError:
                    continue
        if count == 1:
            stability = np.ones(config.bootstrap_repeats)
        else:
            stability_values: list[float] = []
            rng = np.random.default_rng(config.random_seed + 52000 + count)
            for repeat in range(config.bootstrap_repeats):
                indices = rng.choice(events, events, replace=True)
                try:
                    bootstrap_model = _fit_gmm(
                        values[indices],
                        count,
                        config.random_seed + 53000 + count * 100 + repeat,
                        config.gmm_reg_covar,
                        n_init=2,
                    )
                    stability_values.append(
                        float(adjusted_rand_score(labels, bootstrap_model.predict(values)))
                    )
                except ValueError:
                    stability_values.append(0.0)
            stability = np.asarray(stability_values)
        cv = np.asarray(cv_scores, dtype=float)
        cv_mean = float(np.mean(cv)) if len(cv) else float("nan")
        cv_se = (
            float(np.std(cv, ddof=1) / np.sqrt(len(cv))) if len(cv) > 1 else 0.0
        )
        support = bool(
            count == 1
            or (
                sizes.min() >= config.minimum_events_per_type
                and np.median(stability) >= config.bootstrap_ari_median
                and np.quantile(stability, 0.25) >= config.bootstrap_ari_q25
            )
        )
        rows.append(
            {
                "event_type_count": count,
                "events": events,
                "minimum_type_events": int(sizes.min()),
                "event_cv_log_likelihood": cv_mean,
                "event_cv_standard_error": cv_se,
                "bootstrap_ari_median": float(np.median(stability)),
                "bootstrap_ari_q25": float(np.quantile(stability, 0.25)),
                "support_ok": support,
            }
        )
    grid = pd.DataFrame(rows)
    grid["selected"] = False
    supported = grid.loc[grid["support_ok"] & grid["event_cv_log_likelihood"].notna()]
    if supported.empty:
        selected = 1
    else:
        best = supported.loc[supported["event_cv_log_likelihood"].idxmax()]
        threshold = float(best["event_cv_log_likelihood"] - best["event_cv_standard_error"])
        eligible = supported.loc[supported["event_cv_log_likelihood"] >= threshold]
        selected = int(eligible["event_type_count"].min())
    grid.loc[grid["event_type_count"].eq(selected), "selected"] = True
    return grid, models


def fit_mouse_event_types(
    state_result: StateTaxonomyResult,
    config: TemporalTaxonomyConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config = (config or TemporalTaxonomyConfig()).validated()
    events, values, feature_names = _event_feature_matrix(
        state_result.assignments,
        state_result.selected_k,
        config.event_trajectory_bins,
    )
    scaler = StandardScaler().fit(values)
    scaled = np.clip(scaler.transform(values), -8.0, 8.0)
    grid, models = _event_type_grid(scaled, config)
    selected = int(grid.loc[grid["selected"], "event_type_count"].iloc[0])
    model = models[selected]
    raw_labels = model.predict(scaled)
    counts = pd.Series(raw_labels).value_counts().sort_values(ascending=False)
    mapping = {int(old): int(new) for new, old in enumerate(counts.index)}
    labels = np.asarray([mapping[int(value)] for value in raw_labels], dtype=np.int32)
    events["event_type"] = labels
    events["event_type_label"] = [
        f"{state_result.animal_id}-T{value + 1}" for value in labels
    ]
    events["event_type_probability"] = np.max(model.predict_proba(scaled), axis=1)
    grid["animal_id"] = state_result.animal_id
    bundle = {
        "animal_id": state_result.animal_id,
        "selected_event_types": selected,
        "feature_names": feature_names,
        "scaler": scaler,
        "model": model,
        "label_mapping": mapping,
    }
    return events, grid, bundle


def summarize_event_types(
    state_assignments: pd.DataFrame,
    event_type_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize each local whole-seizure type using complete state paths."""

    required_states = {"animal_id", "event_id", "state"}
    required_events = {
        "animal_id",
        "event_id",
        "event_type",
        "event_type_label",
        "duration_s",
        "transition_count",
    }
    if required_states - set(state_assignments.columns):
        raise ValueError("State assignments are missing event/state identifiers")
    if required_events - set(event_type_assignments.columns):
        raise ValueError("Event assignments are missing local type identifiers")
    states = sorted(state_assignments["state"].astype(int).unique())
    counts = (
        state_assignments.groupby(["event_id", "state"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=states, fill_value=0)
    )
    occupancy = counts.div(counts.sum(axis=1), axis=0)
    occupancy.columns = [f"occupancy_s{state + 1}" for state in states]
    events = event_type_assignments.set_index("event_id").join(
        occupancy, how="left"
    )
    occupancy_columns = list(occupancy.columns)
    rows: list[dict[str, Any]] = []
    animal_total = int(events.index.nunique())
    for local_type, group in events.groupby("event_type", sort=True):
        medians = group[occupancy_columns].median(axis=0)
        dominant_column = str(medians.idxmax())
        record: dict[str, Any] = {
            "animal_id": str(group["animal_id"].iloc[0]),
            "event_type": int(local_type),
            "event_type_label": str(group["event_type_label"].iloc[0]),
            "events": int(len(group)),
            "event_fraction": float(len(group) / max(animal_total, 1)),
            "duration_median_s": float(group["duration_s"].median()),
            "duration_q25_s": float(group["duration_s"].quantile(0.25)),
            "duration_q75_s": float(group["duration_s"].quantile(0.75)),
            "transition_median": float(group["transition_count"].median()),
            "dominant_state": int(dominant_column.removeprefix("occupancy_s")),
            "median_type_probability": float(
                group["event_type_probability"].median()
            ),
        }
        for column in occupancy_columns:
            record[f"median_{column}"] = float(group[column].median())
            record[f"q25_{column}"] = float(group[column].quantile(0.25))
            record[f"q75_{column}"] = float(group[column].quantile(0.75))
        rows.append(record)
    return pd.DataFrame(rows)


def save_mouse_taxonomy(
    result: StateTaxonomyResult,
    output_dir: str | Path,
    event_assignments: pd.DataFrame | None = None,
    event_grid: pd.DataFrame | None = None,
    event_bundle: dict[str, Any] | None = None,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result.grid.to_csv(root / "state_model_grid.csv", index=False, encoding="utf-8-sig")
    result.assignments.to_parquet(root / "state_assignments.parquet", index=False)
    result.segments.to_csv(root / "state_segments.csv", index=False, encoding="utf-8-sig")
    result.state_summary.to_csv(root / "state_summary.csv", index=False, encoding="utf-8-sig")
    np.save(root / "multiview_embedding.npy", result.full_embedding.astype(np.float32))
    joblib.dump(result.model_bundle, root / "state_model.joblib")
    if event_assignments is not None:
        event_assignments.to_csv(
            root / "event_type_assignments.csv", index=False, encoding="utf-8-sig"
        )
    if event_grid is not None:
        event_grid.to_csv(
            root / "event_type_model_grid.csv", index=False, encoding="utf-8-sig"
        )
    if event_bundle is not None:
        joblib.dump(event_bundle, root / "event_type_model.joblib")


__all__ = [
    "AUXILIARY_FEATURES",
    "EXPLAINABLE_FEATURES",
    "RepresentationArtifacts",
    "StateTaxonomyResult",
    "TemporalTaxonomyConfig",
    "active_channels_for_mouse",
    "evaluate_state_grid",
    "event_balanced_indices",
    "fit_mouse_event_types",
    "fit_mouse_state_taxonomy",
    "fit_multiview_representation",
    "normalize_native_shapes",
    "save_mouse_taxonomy",
    "summarize_event_types",
    "transform_multiview_representation",
    "viterbi_decode",
]
