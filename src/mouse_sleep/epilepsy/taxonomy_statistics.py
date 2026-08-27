from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


FEATURE_BLOCKS = ("shape", "amplitude", "frequency", "bilateral")

_NON_FEATURE_COLUMNS = {
    "animal_id",
    "session_id",
    "event_id",
    "state",
    "state_label",
    "start_time_ms",
    "end_time_ms",
    "event_start_ms",
    "event_end_ms",
    "event_duration_s",
    "sampling_rate_hz",
    "cache_local_index",
    "shard",
    "window_index",
    "state_probability",
    "posterior_probability",
    "is_fit_window",
    "ch7_valid",
    "ch8_valid",
    "valid_channel_count",
}

_EFFECT_COLUMNS = [
    "animal_id",
    "contrast_type",
    "state_a",
    "state_b",
    "feature",
    "feature_block",
    "n_events_a",
    "n_events_b",
    "n_shared_events",
    "events_resampled",
    "state_a_event_median",
    "state_b_event_median",
    "median_difference",
    "pooled_robust_scale",
    "effect_size",
    "abs_effect_size",
    "effect_ci_low",
    "effect_ci_high",
    "effect_ci_lower_abs",
    "direction",
    "bootstrap_sign_p_value",
    "fdr_q_value",
    "bootstrap_repeats",
    "valid_bootstrap_repeats",
    "status",
]

_SELECTED_COLUMNS = [
    "animal_id",
    "rank",
    "feature",
    "feature_block",
    "contrast_type",
    "state_a",
    "state_b",
    "effect_size",
    "abs_effect_size",
    "effect_ci_low",
    "effect_ci_high",
    "effect_ci_lower_abs",
    "direction",
    "fdr_q_value",
    "maximum_abs_spearman_with_selected",
]

_DESCRIPTIVE_COLUMNS = [
    "animal_id",
    "state",
    "feature",
    "feature_block",
    "events",
    "event_median",
    "event_q25",
    "event_q75",
    "robust_scale",
]

_BLOCK_COLUMNS = [
    "animal_id",
    "feature_block",
    "features_available",
    "features_tested",
    "separation_score",
    "conservative_separation_score",
    "maximum_abs_effect",
    "fdr_significant_features",
    "fdr_significant_fraction",
    "status",
]


@dataclass(frozen=True)
class TaxonomyStatisticsConfig:
    """Configuration for event-weighted interpretation of discovered states.

    Effects are computed after reducing every ``event x state`` cell to one
    median.  Bootstrap samples are complete events, never individual windows.
    ``contrast_mode`` may be ``"pairwise"``, ``"one_vs_rest"``, or ``"both"``.
    """

    animal_column: str = "animal_id"
    event_column: str = "event_id"
    state_column: str = "state"
    contrast_mode: str = "pairwise"
    bootstrap_repeats: int = 1000
    confidence_level: float = 0.95
    minimum_events_per_group: int = 3
    correlation_threshold: float = 0.80
    minimum_representative_dimensions: int = 3
    maximum_representative_dimensions: int = 5
    random_seed: int = 20260811

    def __post_init__(self) -> None:
        if self.contrast_mode not in {"pairwise", "one_vs_rest", "both"}:
            raise ValueError(
                "contrast_mode must be 'pairwise', 'one_vs_rest', or 'both'"
            )
        if self.bootstrap_repeats < 20:
            raise ValueError("bootstrap_repeats must be at least 20")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between zero and one")
        if self.minimum_events_per_group < 2:
            raise ValueError("minimum_events_per_group must be at least two")
        if not 0.0 < self.correlation_threshold <= 1.0:
            raise ValueError("correlation_threshold must be in (0, 1]")
        if not 3 <= self.minimum_representative_dimensions <= 5:
            raise ValueError("minimum_representative_dimensions must be 3--5")
        if not (
            self.minimum_representative_dimensions
            <= self.maximum_representative_dimensions
            <= 5
        ):
            raise ValueError(
                "maximum_representative_dimensions must be between the minimum and 5"
            )


@dataclass(frozen=True)
class TaxonomyStatisticsResult:
    """All audit tables produced by :func:`analyze_state_dimensions`."""

    event_state_summary: pd.DataFrame
    state_descriptives: pd.DataFrame
    dimension_effects: pd.DataFrame
    selected_dimensions: pd.DataFrame
    block_separation: pd.DataFrame
    diagnostics: pd.DataFrame


def analyze_state_dimensions(
    assignments: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    feature_block_map: Mapping[str, str] | None = None,
    config: TaxonomyStatisticsConfig | None = None,
) -> TaxonomyStatisticsResult:
    """Quantify and select interpretable dimensions for per-mouse EEG states.

    Parameters
    ----------
    assignments:
        One row per state-assigned window.  It must contain animal, event, and
        state identifiers plus raw interpretable feature columns.
    feature_columns:
        Features to audit.  Supplying this explicitly is recommended for a
        preregistered analysis.  If omitted, numeric non-metadata columns are
        used.
    feature_block_map:
        Optional mapping from feature name to one of ``shape``, ``amplitude``,
        ``frequency``, or ``bilateral``.  Unmapped features use name-based
        rules aligned with the project's morphology feature names.
    config:
        Column names, event-bootstrap settings, and representative-dimension
        constraints.

    Notes
    -----
    The signed robust effect is the difference between state event medians,
    divided by a pooled within-state robust scale.  Its CI resamples event IDs
    with replacement and retains every sampled event's state summaries.  The
    bootstrap sign p-value and Benjamini-Hochberg q-value are exploratory
    auxiliaries; the effect magnitude and event-bootstrap CI are primary.
    """

    config = config or TaxonomyStatisticsConfig()
    _validate_required_columns(assignments, config)
    features = _resolve_feature_columns(assignments, feature_columns, config)
    blocks = _resolve_feature_blocks(features, feature_block_map)
    prepared = _prepare_assignments(assignments, features, config)
    event_state = _event_state_summary(prepared, features)
    state_descriptives = _state_descriptives(event_state, features, blocks)

    effects, state_counts = _dimension_effects(
        event_state, features, blocks, config
    )
    effects = _add_fdr_q_values(effects)
    selected = _select_representative_dimensions(
        event_state, effects, features, blocks, config
    )
    block_separation = _summarize_feature_blocks(
        event_state, effects, features, blocks
    )
    diagnostics = _build_diagnostics(
        event_state, effects, selected, features, state_counts, config
    )
    return TaxonomyStatisticsResult(
        event_state_summary=event_state,
        state_descriptives=state_descriptives,
        dimension_effects=effects,
        selected_dimensions=selected,
        block_separation=block_separation,
        diagnostics=diagnostics,
    )


def _validate_required_columns(
    assignments: pd.DataFrame, config: TaxonomyStatisticsConfig
) -> None:
    required = {
        config.animal_column,
        config.event_column,
        config.state_column,
    }
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"assignments is missing columns: {sorted(missing)}")
    if assignments.empty:
        raise ValueError("assignments is empty")
    if assignments[list(required)].isna().any(axis=None):
        raise ValueError("animal, event, and state identifiers cannot be missing")


def _resolve_feature_columns(
    assignments: pd.DataFrame,
    feature_columns: Sequence[str] | None,
    config: TaxonomyStatisticsConfig,
) -> list[str]:
    if feature_columns is not None:
        features = list(dict.fromkeys(str(column) for column in feature_columns))
        missing = set(features) - set(assignments.columns)
        if missing:
            raise ValueError(f"feature columns are missing: {sorted(missing)}")
    else:
        identifiers = {
            config.animal_column,
            config.event_column,
            config.state_column,
            *_NON_FEATURE_COLUMNS,
        }
        features = [
            str(column)
            for column in assignments.columns
            if column not in identifiers
            and pd.api.types.is_numeric_dtype(assignments[column])
        ]
    if not features:
        raise ValueError("no numeric feature columns are available")
    forbidden = {
        config.animal_column,
        config.event_column,
        config.state_column,
    }
    overlap = forbidden.intersection(features)
    if overlap:
        raise ValueError(f"identifier columns cannot be features: {sorted(overlap)}")
    return features


def _resolve_feature_blocks(
    features: Sequence[str], supplied: Mapping[str, str] | None
) -> dict[str, str]:
    supplied = supplied or {}
    unknown_features = set(supplied) - set(features)
    if unknown_features:
        raise ValueError(
            f"feature_block_map contains unknown features: {sorted(unknown_features)}"
        )
    invalid_blocks = set(supplied.values()) - set(FEATURE_BLOCKS)
    if invalid_blocks:
        raise ValueError(
            f"unknown feature blocks {sorted(invalid_blocks)}; use {FEATURE_BLOCKS}"
        )
    return {
        feature: supplied.get(feature, _infer_feature_block(feature))
        for feature in features
    }


def _infer_feature_block(feature: str) -> str:
    name = feature.lower()
    bilateral_terms = (
        "bilateral",
        "interchannel",
        "channel_correlation",
        "coherence",
        "cosine_similarity",
        "spectral_similarity",
        "amplitude_log_ratio",
        "line_length_log_ratio",
        "asymmetry",
    )
    frequency_terms = (
        "frequency",
        "spectral_entropy",
        "spectral_concentration",
        "relative_delta",
        "relative_theta",
        "relative_alpha",
        "relative_beta",
        "gamma",
        "theta_delta",
        "high_low_ratio",
        "band_power",
    )
    amplitude_terms = (
        "amplitude",
        "peak_to_peak",
        "line_length",
        "total_power",
        "log_rms",
        "_rms",
    )
    if any(term in name for term in bilateral_terms):
        return "bilateral"
    if any(term in name for term in frequency_terms):
        return "frequency"
    if any(term in name for term in amplitude_terms):
        return "amplitude"
    return "shape"


def _prepare_assignments(
    assignments: pd.DataFrame,
    features: Sequence[str],
    config: TaxonomyStatisticsConfig,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "animal_id": assignments[config.animal_column].astype(str),
            "event_id": assignments[config.event_column].astype(str),
            "state": assignments[config.state_column].astype(str),
        }
    )
    for feature in features:
        frame[feature] = pd.to_numeric(assignments[feature], errors="coerce")
        frame.loc[~np.isfinite(frame[feature]), feature] = np.nan
    return frame


def _event_state_summary(
    assignments: pd.DataFrame, features: Sequence[str]
) -> pd.DataFrame:
    grouped = assignments.groupby(
        ["animal_id", "event_id", "state"], sort=True, observed=True
    )
    summary = grouped[list(features)].median().reset_index()
    counts = grouped.size().rename("window_count").reset_index()
    return summary.merge(
        counts, on=["animal_id", "event_id", "state"], validate="one_to_one"
    )


def _state_descriptives(
    event_state: pd.DataFrame,
    features: Sequence[str],
    blocks: Mapping[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (animal, state), group in event_state.groupby(
        ["animal_id", "state"], sort=True, observed=True
    ):
        for feature in features:
            values = _finite_values(group[feature])
            rows.append(
                {
                    "animal_id": str(animal),
                    "state": str(state),
                    "feature": feature,
                    "feature_block": blocks[feature],
                    "events": int(len(values)),
                    "event_median": _quantile_or_nan(values, 0.50),
                    "event_q25": _quantile_or_nan(values, 0.25),
                    "event_q75": _quantile_or_nan(values, 0.75),
                    "robust_scale": _robust_scale(values),
                }
            )
    return pd.DataFrame(rows, columns=_DESCRIPTIVE_COLUMNS)


def _dimension_effects(
    event_state: pd.DataFrame,
    features: Sequence[str],
    blocks: Mapping[str, str],
    config: TaxonomyStatisticsConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    state_counts: dict[str, int] = {}
    for animal, animal_rows in event_state.groupby("animal_id", sort=True):
        states = sorted(animal_rows["state"].astype(str).unique(), key=str)
        state_counts[str(animal)] = len(states)
        if len(states) < 2:
            continue
        contrasts: list[tuple[str, str, str]] = []
        if config.contrast_mode in {"pairwise", "both"}:
            contrasts.extend(
                ("pairwise", state_a, state_b)
                for state_a, state_b in itertools.combinations(states, 2)
            )
        if config.contrast_mode in {"one_vs_rest", "both"}:
            contrasts.extend(
                ("one_vs_rest", state, "__rest__") for state in states
            )
        for contrast_type, state_a, state_b in contrasts:
            rows_a = animal_rows.loc[animal_rows["state"].eq(state_a)]
            rows_b = (
                animal_rows.loc[animal_rows["state"].eq(state_b)]
                if contrast_type == "pairwise"
                else animal_rows.loc[~animal_rows["state"].eq(state_a)]
                .groupby("event_id", sort=True, observed=True)[list(features)]
                .median()
                .reset_index()
            )
            for feature in features:
                values_a = _event_value_table(rows_a, feature)
                values_b = _event_value_table(rows_b, feature)
                rows.append(
                    _effect_row(
                        str(animal),
                        contrast_type,
                        state_a,
                        state_b,
                        feature,
                        blocks[feature],
                        values_a,
                        values_b,
                        config,
                    )
                )
    return pd.DataFrame(rows, columns=_EFFECT_COLUMNS), state_counts


def _event_value_table(rows: pd.DataFrame, feature: str) -> pd.DataFrame:
    values = rows[["event_id", feature]].copy()
    values[feature] = pd.to_numeric(values[feature], errors="coerce")
    values = values.loc[np.isfinite(values[feature])]
    if values["event_id"].duplicated().any():
        values = (
            values.groupby("event_id", sort=True, observed=True)[feature]
            .median()
            .reset_index()
        )
    return values.rename(columns={feature: "value"})


def _effect_row(
    animal: str,
    contrast_type: str,
    state_a: str,
    state_b: str,
    feature: str,
    feature_block: str,
    values_a: pd.DataFrame,
    values_b: pd.DataFrame,
    config: TaxonomyStatisticsConfig,
) -> dict[str, Any]:
    a = values_a["value"].to_numpy(float)
    b = values_b["value"].to_numpy(float)
    events_a = set(values_a["event_id"].astype(str))
    events_b = set(values_b["event_id"].astype(str))
    common = sorted(events_a | events_b)
    base: dict[str, Any] = {
        "animal_id": animal,
        "contrast_type": contrast_type,
        "state_a": str(state_a),
        "state_b": str(state_b),
        "feature": feature,
        "feature_block": feature_block,
        "n_events_a": int(len(a)),
        "n_events_b": int(len(b)),
        "n_shared_events": int(len(events_a & events_b)),
        "events_resampled": int(len(common)),
        "state_a_event_median": _quantile_or_nan(a, 0.50),
        "state_b_event_median": _quantile_or_nan(b, 0.50),
        "median_difference": float("nan"),
        "pooled_robust_scale": float("nan"),
        "effect_size": float("nan"),
        "abs_effect_size": float("nan"),
        "effect_ci_low": float("nan"),
        "effect_ci_high": float("nan"),
        "effect_ci_lower_abs": float("nan"),
        "direction": "not_estimable",
        "bootstrap_sign_p_value": float("nan"),
        "fdr_q_value": float("nan"),
        "bootstrap_repeats": int(config.bootstrap_repeats),
        "valid_bootstrap_repeats": 0,
        "status": "insufficient_events",
    }
    if (
        len(a) < config.minimum_events_per_group
        or len(b) < config.minimum_events_per_group
    ):
        return base
    effect, difference, scale = _robust_standardized_effect(a, b)
    seed = _stable_seed(
        config.random_seed, animal, contrast_type, state_a, state_b, feature
    )
    bootstrap = _event_cluster_bootstrap_effects(
        values_a, values_b, config.bootstrap_repeats, seed
    )
    valid = bootstrap[np.isfinite(bootstrap)]
    minimum_valid = max(20, int(np.ceil(config.bootstrap_repeats * 0.50)))
    base.update(
        {
            "median_difference": difference,
            "pooled_robust_scale": scale,
            "effect_size": effect,
            "abs_effect_size": abs(effect),
            "direction": _effect_direction(effect, state_a, state_b),
            "valid_bootstrap_repeats": int(len(valid)),
        }
    )
    if len(valid) < minimum_valid:
        base["status"] = "insufficient_bootstrap_support"
        return base
    alpha = (1.0 - config.confidence_level) / 2.0
    ci_low, ci_high = np.quantile(valid, [alpha, 1.0 - alpha])
    sign_p = 2.0 * min(
        (np.count_nonzero(valid <= 0.0) + 1.0) / (len(valid) + 1.0),
        (np.count_nonzero(valid >= 0.0) + 1.0) / (len(valid) + 1.0),
    )
    base.update(
        {
            "effect_ci_low": float(ci_low),
            "effect_ci_high": float(ci_high),
            "effect_ci_lower_abs": _signed_ci_lower_abs(ci_low, ci_high),
            "bootstrap_sign_p_value": float(min(1.0, sign_p)),
            "status": "ok",
        }
    )
    return base


def _event_cluster_bootstrap_effects(
    values_a: pd.DataFrame,
    values_b: pd.DataFrame,
    repeats: int,
    seed: int,
) -> np.ndarray:
    events = sorted(
        set(values_a["event_id"].astype(str))
        | set(values_b["event_id"].astype(str))
    )
    event_index = {event: index for index, event in enumerate(events)}
    a_by_event = np.full(len(events), np.nan, dtype=float)
    b_by_event = np.full(len(events), np.nan, dtype=float)
    for row in values_a.itertuples(index=False):
        a_by_event[event_index[str(row.event_id)]] = float(row.value)
    for row in values_b.itertuples(index=False):
        b_by_event[event_index[str(row.event_id)]] = float(row.value)
    rng = np.random.default_rng(seed)
    effects = np.full(repeats, np.nan, dtype=float)
    for repeat in range(repeats):
        sampled = rng.integers(0, len(events), size=len(events))
        a = a_by_event[sampled]
        b = b_by_event[sampled]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            continue
        effects[repeat] = _robust_standardized_effect(a, b)[0]
    return effects


def _robust_standardized_effect(
    values_a: np.ndarray, values_b: np.ndarray
) -> tuple[float, float, float]:
    a = _finite_values(values_a)
    b = _finite_values(values_b)
    difference = float(np.median(a) - np.median(b))
    scale_a = _robust_scale(a)
    scale_b = _robust_scale(b)
    positive = [scale for scale in (scale_a, scale_b) if np.isfinite(scale) and scale > 1e-12]
    if len(positive) == 2:
        scale = float(np.sqrt((positive[0] ** 2 + positive[1] ** 2) / 2.0))
    elif len(positive) == 1:
        scale = float(positive[0])
    else:
        scale = _robust_scale(np.concatenate([a, b]))
    if not np.isfinite(scale) or scale <= 1e-12:
        return 0.0, difference, 0.0
    return float(difference / scale), difference, float(scale)


def _robust_scale(values: np.ndarray | Sequence[float]) -> float:
    values = _finite_values(values)
    if len(values) < 2:
        return float("nan")
    median = float(np.median(values))
    mad_scale = 1.4826 * float(np.median(np.abs(values - median)))
    if mad_scale > 1e-12:
        return mad_scale
    iqr_scale = float(np.quantile(values, 0.75) - np.quantile(values, 0.25)) / 1.349
    if iqr_scale > 1e-12:
        return iqr_scale
    standard = float(np.std(values, ddof=1))
    return standard if standard > 1e-12 else 0.0


def _effect_direction(effect: float, state_a: str, state_b: str) -> str:
    if effect > 0.0:
        return f"{state_a}>{state_b}"
    if effect < 0.0:
        return f"{state_a}<{state_b}"
    return "no_difference"


def _signed_ci_lower_abs(ci_low: float, ci_high: float) -> float:
    if ci_low <= 0.0 <= ci_high:
        return 0.0
    return float(min(abs(ci_low), abs(ci_high)))


def _add_fdr_q_values(effects: pd.DataFrame) -> pd.DataFrame:
    if effects.empty:
        return pd.DataFrame(columns=_EFFECT_COLUMNS)
    result = effects.copy()
    for _, indices in result.groupby("animal_id", sort=True).groups.items():
        index = np.asarray(list(indices), dtype=int)
        p_values = result.loc[index, "bootstrap_sign_p_value"].to_numpy(float)
        finite = np.isfinite(p_values)
        if finite.any():
            result.loc[index[finite], "fdr_q_value"] = _benjamini_hochberg(
                p_values[finite]
            )
    return result.reindex(columns=_EFFECT_COLUMNS)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def _select_representative_dimensions(
    event_state: pd.DataFrame,
    effects: pd.DataFrame,
    features: Sequence[str],
    blocks: Mapping[str, str],
    config: TaxonomyStatisticsConfig,
) -> pd.DataFrame:
    selected_rows: list[dict[str, Any]] = []
    for animal, animal_events in event_state.groupby("animal_id", sort=True):
        animal_effects = effects.loc[
            effects["animal_id"].eq(str(animal)) & effects["status"].eq("ok")
        ]
        if animal_effects.empty:
            continue
        candidates = (
            animal_effects.sort_values(
                ["effect_ci_lower_abs", "abs_effect_size", "feature"],
                ascending=[False, False, True],
                kind="stable",
            )
            .drop_duplicates("feature", keep="first")
            .reset_index(drop=True)
        )
        event_features = (
            animal_events.groupby("event_id", sort=True, observed=True)[list(features)]
            .median()
            .replace([np.inf, -np.inf], np.nan)
        )
        correlations = event_features.corr(method="spearman").abs()
        accepted: list[str] = []
        for row in candidates.to_dict("records"):
            feature = str(row["feature"])
            pair_correlations = [
                float(correlations.loc[feature, previous])
                for previous in accepted
                if feature in correlations.index
                and previous in correlations.columns
                and np.isfinite(correlations.loc[feature, previous])
            ]
            maximum_correlation = max(pair_correlations, default=0.0)
            if maximum_correlation >= config.correlation_threshold:
                continue
            accepted.append(feature)
            selected_rows.append(
                {
                    "animal_id": str(animal),
                    "rank": int(len(accepted)),
                    "feature": feature,
                    "feature_block": blocks[feature],
                    "contrast_type": row["contrast_type"],
                    "state_a": row["state_a"],
                    "state_b": row["state_b"],
                    "effect_size": row["effect_size"],
                    "abs_effect_size": row["abs_effect_size"],
                    "effect_ci_low": row["effect_ci_low"],
                    "effect_ci_high": row["effect_ci_high"],
                    "effect_ci_lower_abs": row["effect_ci_lower_abs"],
                    "direction": row["direction"],
                    "fdr_q_value": row["fdr_q_value"],
                    "maximum_abs_spearman_with_selected": maximum_correlation,
                }
            )
            if len(accepted) >= config.maximum_representative_dimensions:
                break
    return pd.DataFrame(selected_rows, columns=_SELECTED_COLUMNS)


def _summarize_feature_blocks(
    event_state: pd.DataFrame,
    effects: pd.DataFrame,
    features: Sequence[str],
    blocks: Mapping[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for animal, animal_rows in event_state.groupby("animal_id", sort=True):
        state_count = int(animal_rows["state"].nunique())
        animal_effects = effects.loc[
            effects["animal_id"].eq(str(animal)) & effects["status"].eq("ok")
        ]
        for block in FEATURE_BLOCKS:
            available = [feature for feature in features if blocks[feature] == block]
            block_effects = animal_effects.loc[
                animal_effects["feature_block"].eq(block)
            ]
            if block_effects.empty:
                status = (
                    "single_state"
                    if state_count < 2
                    else "no_features"
                    if not available
                    else "no_valid_contrasts"
                )
                rows.append(
                    {
                        "animal_id": str(animal),
                        "feature_block": block,
                        "features_available": int(len(available)),
                        "features_tested": 0,
                        "separation_score": float("nan"),
                        "conservative_separation_score": float("nan"),
                        "maximum_abs_effect": float("nan"),
                        "fdr_significant_features": 0,
                        "fdr_significant_fraction": float("nan"),
                        "status": status,
                    }
                )
                continue
            per_feature_rows: list[dict[str, float | str]] = []
            for feature, feature_rows in block_effects.groupby("feature", sort=True):
                strongest = feature_rows.sort_values(
                    ["abs_effect_size", "effect_ci_lower_abs"],
                    ascending=[False, False],
                    kind="stable",
                ).iloc[0]
                per_feature_rows.append(
                    {
                        "feature": str(feature),
                        "abs_effect": float(strongest["abs_effect_size"]),
                        "ci_lower": float(
                            feature_rows["effect_ci_lower_abs"].max()
                        ),
                        "minimum_q": float(feature_rows["fdr_q_value"].min()),
                    }
                )
            per_feature = pd.DataFrame(per_feature_rows)
            significant = int(np.count_nonzero(per_feature["minimum_q"] <= 0.05))
            rows.append(
                {
                    "animal_id": str(animal),
                    "feature_block": block,
                    "features_available": int(len(available)),
                    "features_tested": int(len(per_feature)),
                    "separation_score": float(per_feature["abs_effect"].median()),
                    "conservative_separation_score": float(
                        per_feature["ci_lower"].median()
                    ),
                    "maximum_abs_effect": float(per_feature["abs_effect"].max()),
                    "fdr_significant_features": significant,
                    "fdr_significant_fraction": float(significant / len(per_feature)),
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows, columns=_BLOCK_COLUMNS)


def _build_diagnostics(
    event_state: pd.DataFrame,
    effects: pd.DataFrame,
    selected: pd.DataFrame,
    features: Sequence[str],
    state_counts: Mapping[str, int],
    config: TaxonomyStatisticsConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for animal, animal_rows in event_state.groupby("animal_id", sort=True):
        animal = str(animal)
        state_count = int(state_counts.get(animal, animal_rows["state"].nunique()))
        effects_count = int(
            np.count_nonzero(
                effects.loc[effects["animal_id"].eq(animal), "status"].eq("ok")
            )
        )
        selected_count = int(
            np.count_nonzero(selected["animal_id"].eq(animal))
            if not selected.empty
            else 0
        )
        if state_count < 2:
            status = "single_state"
        elif effects_count == 0:
            status = "no_valid_contrasts"
        elif selected_count < config.minimum_representative_dimensions:
            status = "insufficient_nonredundant_features"
        else:
            status = "ok"
        rows.append(
            {
                "animal_id": animal,
                "states": state_count,
                "events": int(animal_rows["event_id"].nunique()),
                "event_state_cells": int(len(animal_rows)),
                "features": int(len(features)),
                "valid_feature_contrasts": effects_count,
                "selected_dimensions": selected_count,
                "correlation_threshold": float(config.correlation_threshold),
                "bootstrap_unit": "event_id",
                "bootstrap_repeats": int(config.bootstrap_repeats),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def _finite_values(values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    return result[np.isfinite(result)]


def _quantile_or_nan(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile)) if len(values) else float("nan")


def _stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in (base_seed, *parts)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % np.iinfo(np.uint32).max


__all__ = [
    "FEATURE_BLOCKS",
    "TaxonomyStatisticsConfig",
    "TaxonomyStatisticsResult",
    "analyze_state_dimensions",
]
