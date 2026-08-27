from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler


EVENT_PHYSIOLOGY_FEATURES = [
    "scaled_aggregate_log_robust_amplitude",
    "scaled_aggregate_log_line_length",
    "aggregate_hjorth_mobility",
    "aggregate_hjorth_complexity",
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
    "relative_line_noise_48_52",
]


@dataclass(frozen=True)
class EventMorphologyConfig:
    minimum_training_windows: int = 2
    minimum_event_duration_s: float = 4.0
    maximum_components: int = 6
    balanced_repeats: int = 12
    pca_components: int = 8
    minimum_component_fraction: float = 0.05
    minimum_component_animals: int = 3
    minimum_component_events: int = 20
    maximum_single_animal_fraction: float = 0.70
    minimum_stability_ari: float = 0.60
    minimum_assignment_probability: float = 0.60
    pure_event_fraction: float = 0.80
    random_seed: int = 20260811


def build_event_signatures(
    assignments: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse each complete annotated event to one composition/physiology vector."""

    required = {
        "event_id",
        "animal_id",
        "session_id",
        "morphology_label",
        "review_morphology_label",
        "start_time_ms",
        "event_duration_s",
    }
    missing = required - set(assignments)
    if missing:
        raise ValueError(f"Event signatures need assignment columns: {sorted(missing)}")
    physiology = [column for column in EVENT_PHYSIOLOGY_FEATURES if column in assignments]
    stable_labels = sorted(assignments["morphology_label"].astype(str).unique())
    review_labels = sorted(assignments["review_morphology_label"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    for event_id, group in assignments.groupby("event_id", sort=True):
        ordered = group.sort_values("start_time_ms")
        stable = ordered["morphology_label"].astype(str)
        review = ordered["review_morphology_label"].astype(str)
        review_counts = review.value_counts(normalize=True)
        dominant_review = str(review_counts.index[0])
        dominant_fraction = float(review_counts.iloc[0])
        review_probabilities = review_counts.to_numpy(float)
        review_entropy = float(
            -np.sum(
                review_probabilities
                * np.log(np.maximum(review_probabilities, 1e-12))
            )
        )
        review_values = review.to_numpy()
        transitions = (
            int((review_values[1:] != review_values[:-1]).sum())
            if len(review_values) > 1
            else 0
        )
        row: dict[str, Any] = {
            "event_id": str(event_id),
            "animal_id": str(group["animal_id"].iloc[0]),
            "session_id": str(group["session_id"].iloc[0]),
            "event_start_ms": float(group["event_start_ms"].iloc[0]),
            "event_end_ms": float(group["event_end_ms"].iloc[0]),
            "duration_s": float(group["event_duration_s"].iloc[0]),
            "sampled_windows": int(len(group)),
            "minimum_overlap_fraction": float(group["event_overlap_fraction"].min()),
            "dominant_review_state": dominant_review,
            "dominant_review_fraction": dominant_fraction,
            "review_state_entropy": review_entropy,
            "review_transition_count": transitions,
            "review_transition_rate": float(transitions / max(len(group) - 1, 1)),
            "window_uncertain_fraction": float(group["morphology_uncertain"].mean()),
            "valid_ch7_fraction": float(group["ch7_valid"].mean()),
            "valid_ch8_fraction": float(group["ch8_valid"].mean()),
            "event_composition": (
                "single_dominant_state" if dominant_fraction >= 0.80 else "mixed_states"
            ),
        }
        for label in stable_labels:
            row[f"stable_prop_{label}"] = float(stable.eq(label).mean())
        for label in review_labels:
            row[f"review_prop_{label}"] = float(review.eq(label).mean())
        for column in physiology:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"median__{column}"] = float(values.median())
            row[f"iqr__{column}"] = float(values.quantile(0.75) - values.quantile(0.25))
        rows.append(row)
    signatures = pd.DataFrame(rows)
    represented = set(signatures["event_id"].astype(str))
    missing_events = events.loc[~events["event_id"].astype(str).isin(represented)]
    if not missing_events.empty:
        raise ValueError(
            f"{len(missing_events)} complete events have no event-level signature"
        )
    signatures["short_unresolved"] = signatures["duration_s"].lt(4.0)
    signatures["limited_context"] = signatures["sampled_windows"].lt(2)
    return signatures


def fit_event_morphology(
    signatures: pd.DataFrame,
    output_dir: str | Path,
    config: EventMorphologyConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit event-level candidate types using repeated mouse-balanced event samples."""

    config = config or EventMorphologyConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = _event_feature_columns(signatures)
    raw = signatures[feature_columns].replace([np.inf, -np.inf], np.nan)
    eligible = (
        signatures["sampled_windows"].ge(config.minimum_training_windows)
        & signatures["duration_s"].ge(config.minimum_event_duration_s)
        & signatures["valid_ch8_fraction"].gt(0.5)
    )
    eligible_rows = signatures.loc[eligible].reset_index(drop=True)
    eligible_raw = raw.loc[eligible].reset_index(drop=True)
    animal_counts = eligible_rows.groupby("animal_id").size()
    if len(animal_counts) < 2 or animal_counts.min() < 3:
        raise ValueError("Event-level discovery needs at least three eligible events per animal")
    balanced_events_per_animal = int(animal_counts.min())
    maximum_components = min(
        config.maximum_components,
        balanced_events_per_animal * len(animal_counts) - 1,
    )

    rows: list[dict[str, Any]] = []
    candidate_packs: list[dict[str, Any]] = []
    for components in range(1, maximum_components + 1):
        predictions: list[np.ndarray] = []
        held_out_scores: list[float] = []
        packs: list[dict[str, Any]] = []
        for repeat in range(config.balanced_repeats):
            fit_indices = _balanced_event_sample(
                eligible_rows,
                balanced_events_per_animal,
                config.random_seed + repeat,
            )
            held_out = np.ones(len(eligible_rows), dtype=bool)
            held_out[fit_indices] = False
            pack = _fit_event_gmm(
                eligible_raw.iloc[fit_indices],
                components,
                config,
                config.random_seed + repeat,
            )
            all_values = _transform_event_values(eligible_raw, pack)
            labels = pack["gmm"].predict(all_values)
            predictions.append(labels)
            packs.append(pack)
            score_rows = all_values[held_out] if held_out.any() else all_values
            held_out_scores.append(float(pack["gmm"].score(score_rows)))
        pairwise = _pairwise_ari(predictions)
        agreement = np.asarray(
            [
                np.mean(
                    [
                        adjusted_rand_score(predictions[index], predictions[other])
                        for other in range(len(predictions))
                        if other != index
                    ]
                )
                for index in range(len(predictions))
            ]
        )
        medoid_index = int(np.argmax(agreement))
        medoid_labels = predictions[medoid_index]
        medoid_pack = packs[medoid_index]
        stable_map = _stable_map(medoid_labels)
        stable_labels = np.asarray(
            [stable_map[int(label)] for label in medoid_labels], dtype=np.int32
        )
        support = _event_component_support(eligible_rows, stable_labels, components)
        score_array = np.asarray(held_out_scores, dtype=float)
        support_ok = bool(
            support["minimum_component_fraction"] >= config.minimum_component_fraction
            and support["minimum_component_animals"] >= config.minimum_component_animals
            and support["minimum_component_events"] >= config.minimum_component_events
            and support["maximum_single_animal_fraction"]
            <= config.maximum_single_animal_fraction
        )
        stability = float(np.median(pairwise)) if len(pairwise) else 1.0
        rows.append(
            {
                "candidate_index": components - 1,
                "components": components,
                "held_out_mean_log_likelihood": float(score_array.mean()),
                "held_out_se_log_likelihood": float(
                    score_array.std(ddof=1) / math.sqrt(len(score_array))
                ),
                "balanced_repeat_pairwise_ari_median": stability,
                "balanced_repeat_pairwise_ari_q25": float(
                    np.quantile(pairwise, 0.25) if len(pairwise) else 1.0
                ),
                "medoid_repeat": medoid_index,
                **support,
                "animal_cluster_nmi": float(
                    normalized_mutual_info_score(
                        eligible_rows["animal_id"].astype(str), stable_labels
                    )
                ),
                "support_ok": support_ok,
                "stability_ok": stability >= config.minimum_stability_ari,
                "accepted_for_primary": support_ok
                and stability >= config.minimum_stability_ari,
            }
        )
        medoid_pack["cluster_map"] = stable_map
        candidate_packs.append(medoid_pack)

    grid = pd.DataFrame(rows)
    selected = _select_event_candidate(grid)
    selected_index = int(selected["candidate_index"])
    pack = candidate_packs[selected_index]
    review_selected = _select_event_review_candidate(grid, selected)
    review_index = int(review_selected["candidate_index"])
    review_pack = candidate_packs[review_index]
    assignable = ~signatures["short_unresolved"].astype(bool)
    assignment_values = _transform_event_values(raw.loc[assignable], pack)
    raw_labels = pack["gmm"].predict(assignment_values)
    raw_probabilities = pack["gmm"].predict_proba(assignment_values)
    mapping = pack["cluster_map"]
    labels = np.asarray([mapping[int(label)] for label in raw_labels], dtype=np.int32)
    probabilities = np.empty_like(raw_probabilities)
    for raw_label, stable_label in mapping.items():
        probabilities[:, stable_label] = raw_probabilities[:, raw_label]
    maximum_probability = probabilities.max(axis=1)
    review_values = _transform_event_values(raw.loc[assignable], review_pack)
    raw_review_labels = review_pack["gmm"].predict(review_values)
    raw_review_probabilities = review_pack["gmm"].predict_proba(review_values)
    review_mapping = review_pack["cluster_map"]
    review_labels = np.asarray(
        [review_mapping[int(label)] for label in raw_review_labels], dtype=np.int32
    )
    review_probabilities = np.empty_like(raw_review_probabilities)
    for raw_label, stable_label in review_mapping.items():
        review_probabilities[:, stable_label] = raw_review_probabilities[:, raw_label]
    review_maximum_probability = review_probabilities.max(axis=1)

    result = signatures.copy()
    result["event_type_cluster"] = -1
    result["event_type_label"] = "short_unresolved"
    result["event_type_probability"] = np.nan
    result.loc[assignable, "event_type_cluster"] = labels
    result.loc[assignable, "event_type_label"] = [
        f"E{label + 1}" for label in labels
    ]
    result.loc[assignable, "event_type_probability"] = maximum_probability
    result["event_type_uncertain"] = (
        result["short_unresolved"].astype(bool)
        | result["limited_context"].astype(bool)
        | result["event_type_probability"].fillna(0.0).lt(
            config.minimum_assignment_probability
        )
    )
    result["review_event_type_cluster"] = -1
    result["review_event_type_label"] = "short_unresolved"
    result["review_event_type_probability"] = np.nan
    result.loc[assignable, "review_event_type_cluster"] = review_labels
    result.loc[assignable, "review_event_type_label"] = [
        f"ER{label + 1}" for label in review_labels
    ]
    result.loc[assignable, "review_event_type_probability"] = (
        review_maximum_probability
    )
    result["review_event_type_uncertain"] = (
        result["short_unresolved"].astype(bool)
        | result["limited_context"].astype(bool)
        | result["review_event_type_probability"].fillna(0.0).lt(
            config.minimum_assignment_probability
        )
    )
    grid["selected_primary"] = grid["candidate_index"].eq(selected_index)
    grid["selected_review_resolution"] = grid["candidate_index"].eq(review_index)
    summary = _summarize_event_types(result, "event_type")
    review_summary = _summarize_event_types(result, "review_event_type")
    diagnostics = {
        "schema_version": 1,
        "config": asdict(config),
        "feature_columns": feature_columns,
        "events": int(len(result)),
        "eligible_training_events": int(eligible.sum()),
        "balanced_events_per_animal": balanced_events_per_animal,
        "selected_components": int(selected["components"]),
        "selected_candidate": selected.to_dict(),
        "review_components": int(review_selected["components"]),
        "review_candidate": review_selected.to_dict(),
        "short_unresolved_events": int(result["short_unresolved"].sum()),
        "limited_context_events": int(result["limited_context"].sum()),
        "uncertain_events": int(result["event_type_uncertain"].sum()),
    }
    grid.to_csv(output_dir / "event_gmm_parameter_grid.csv", index=False)
    summary.to_csv(output_dir / "event_type_summary.csv", index=False, encoding="utf-8-sig")
    review_summary.to_csv(
        output_dir / "review_event_type_summary.csv", index=False, encoding="utf-8-sig"
    )
    result.to_parquet(output_dir / "event_morphology_assignments.parquet", index=False)
    joblib.dump(
        {"primary": pack, "review_resolution": review_pack},
        model_dir / "event_morphology_gmm.joblib",
    )
    (output_dir / "event_morphology_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    return result, summary, diagnostics


def _event_feature_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = ("stable_prop_", "review_prop_", "median__", "iqr__")
    columns = [column for column in frame if column.startswith(prefixes)]
    columns.extend(
        column
        for column in ("review_state_entropy", "review_transition_rate")
        if column in frame
    )
    if len(columns) < 5:
        raise ValueError("Too few event composition/physiology features")
    return columns


def _balanced_event_sample(frame: pd.DataFrame, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for _, group in frame.groupby("animal_id", sort=True):
        indices = group.index.to_numpy(np.int64)
        selected.extend(rng.choice(indices, count, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _fit_event_gmm(
    raw: pd.DataFrame,
    components: int,
    config: EventMorphologyConfig,
    seed: int,
) -> dict[str, Any]:
    medians = raw.median(axis=0).fillna(0.0)
    values = raw.fillna(medians).to_numpy(np.float64)
    scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(values)
    scaled = scaler.transform(values)
    dimensions = min(config.pca_components, scaled.shape[1], len(scaled) - 1)
    pca = PCA(n_components=dimensions, whiten=True, random_state=seed).fit(scaled)
    embedded = pca.transform(scaled)
    gmm = GaussianMixture(
        n_components=components,
        covariance_type="diag",
        reg_covar=1e-5,
        n_init=10,
        random_state=seed,
    ).fit(embedded)
    return {
        "medians": medians.to_dict(),
        "feature_columns": raw.columns.tolist(),
        "scaler": scaler,
        "pca": pca,
        "gmm": gmm,
    }


def _transform_event_values(raw: pd.DataFrame, pack: dict[str, Any]) -> np.ndarray:
    medians = pd.Series(pack["medians"])
    ordered = raw.reindex(columns=pack["feature_columns"]).fillna(medians)
    return pack["pca"].transform(pack["scaler"].transform(ordered.to_numpy(float)))


def _pairwise_ari(predictions: list[np.ndarray]) -> np.ndarray:
    values: list[float] = []
    for left in range(len(predictions)):
        for right in range(left + 1, len(predictions)):
            values.append(float(adjusted_rand_score(predictions[left], predictions[right])))
    return np.asarray(values, dtype=float)


def _stable_map(labels: np.ndarray) -> dict[int, int]:
    counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    return {int(old): int(new) for new, old in enumerate(counts.index)}


def _event_component_support(
    frame: pd.DataFrame, labels: np.ndarray, components: int
) -> dict[str, Any]:
    fractions: list[float] = []
    animals: list[int] = []
    events: list[int] = []
    maximum_animal: list[float] = []
    for component in range(components):
        selected = labels == component
        animal_counts = frame.loc[selected, "animal_id"].astype(str).value_counts()
        fractions.append(float(selected.mean()))
        animals.append(int(len(animal_counts)))
        events.append(int(selected.sum()))
        maximum_animal.append(float(animal_counts.max() / selected.sum()))
    return {
        "minimum_component_fraction": min(fractions),
        "minimum_component_animals": min(animals),
        "minimum_component_events": min(events),
        "maximum_single_animal_fraction": max(maximum_animal),
    }


def _select_event_candidate(grid: pd.DataFrame) -> pd.Series:
    valid = grid.loc[grid["accepted_for_primary"].astype(bool)].copy()
    basis = "support_and_balanced_repeat_stability"
    if valid.empty:
        valid = grid.loc[grid["support_ok"].astype(bool)].copy()
        basis = "support_only_low_confidence"
    if valid.empty:
        valid = grid.copy()
        basis = "exploratory_no_supported_multicomponent_model"
    best = valid.sort_values("held_out_mean_log_likelihood", ascending=False).iloc[0]
    threshold = float(best["held_out_mean_log_likelihood"]) - float(
        best["held_out_se_log_likelihood"]
    )
    within_one_se = valid.loc[
        valid["held_out_mean_log_likelihood"].ge(threshold)
    ]
    selected = within_one_se.sort_values("components").iloc[0].copy()
    selected["selection_basis"] = basis
    selected["one_se_threshold"] = threshold
    return selected


def _select_event_review_candidate(
    grid: pd.DataFrame, primary: pd.Series
) -> pd.Series:
    candidates = grid.loc[
        grid["stability_ok"].astype(bool)
        & grid["minimum_component_fraction"].ge(0.05)
        & grid["minimum_component_animals"].ge(3)
        & grid["minimum_component_events"].ge(20)
    ].copy()
    if candidates.empty:
        selected = primary.copy()
        selected["selection_basis"] = "same_as_primary_no_finer_stable_resolution"
        return selected
    selected = candidates.sort_values("components", ascending=False).iloc[0].copy()
    selected["selection_basis"] = (
        "maximum_stable_exploratory_resolution_support_bias_may_remain"
    )
    return selected


def _summarize_event_types(result: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cluster_column = f"{prefix}_cluster"
    label_column = f"{prefix}_label"
    probability_column = f"{prefix}_probability"
    uncertain_column = f"{prefix}_uncertain"
    classified = result.loc[result[cluster_column].ge(0)]
    rows: list[dict[str, Any]] = []
    for (animal, cluster, label), group in classified.groupby(
        ["animal_id", cluster_column, label_column], sort=True
    ):
        rows.append(
            {
                "animal_id": str(animal),
                cluster_column: int(cluster),
                label_column: str(label),
                "events": int(len(group)),
                "animal_event_fraction": float(
                    len(group) / max((classified["animal_id"].astype(str) == str(animal)).sum(), 1)
                ),
                "median_duration_s": float(group["duration_s"].median()),
                "median_assignment_probability": float(
                    group[probability_column].median()
                ),
                "uncertain_fraction": float(group[uncertain_column].mean()),
                "mixed_state_fraction": float(group["event_composition"].eq("mixed_states").mean()),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "EventMorphologyConfig",
    "build_event_signatures",
    "fit_event_morphology",
]
