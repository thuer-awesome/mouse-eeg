from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler


@dataclass(frozen=True)
class CrossMouseTaxonomyConfig:
    """Settings for matching local, interpretable prototypes across mice."""

    minimum_global_types: int = 1
    maximum_global_types: int = 8
    minimum_feature_finite_fraction: float = 0.50
    covariance_type: str = "diag"
    regularization: float = 1e-4
    n_init: int = 30
    clip_scaled_values: float = 8.0
    random_seed: int = 20260812

    def __post_init__(self) -> None:
        if self.minimum_global_types != 1:
            raise ValueError("K=1 must remain available as the global null model")
        if not 1 <= self.maximum_global_types <= 20:
            raise ValueError("maximum_global_types must be between 1 and 20")
        if not 0.0 < self.minimum_feature_finite_fraction <= 1.0:
            raise ValueError("minimum_feature_finite_fraction must be in (0, 1]")
        if self.covariance_type not in {"diag", "full", "tied", "spherical"}:
            raise ValueError("unsupported GaussianMixture covariance_type")
        if self.regularization <= 0.0 or self.n_init < 1:
            raise ValueError("regularization and n_init must be positive")
        if self.clip_scaled_values <= 0.0:
            raise ValueError("clip_scaled_values must be positive")


@dataclass(frozen=True)
class PrototypeMatchResult:
    """Cross-mouse model grid, local mappings, and dense presence table."""

    model_grid: pd.DataFrame
    prototype_assignments: pd.DataFrame
    presence_matrix: pd.DataFrame
    global_summary: pd.DataFrame
    feature_columns: tuple[str, ...]
    model_bundle: dict[str, Any]


@dataclass(frozen=True)
class CrossMouseTaxonomyResult:
    event_types: PrototypeMatchResult
    states: PrototypeMatchResult


def match_cross_mouse_taxonomy(
    event_type_assignments: pd.DataFrame | Sequence[pd.DataFrame],
    state_summaries: pd.DataFrame | Sequence[pd.DataFrame],
    event_feature_columns: Sequence[str] | None = None,
    state_feature_columns: Sequence[str] | None = None,
    config: CrossMouseTaxonomyConfig | None = None,
) -> CrossMouseTaxonomyResult:
    """Match local event types and local states in two independent models."""

    config = config or CrossMouseTaxonomyConfig()
    return CrossMouseTaxonomyResult(
        event_types=match_event_type_prototypes(
            event_type_assignments, event_feature_columns, config
        ),
        states=match_state_prototypes(
            state_summaries, state_feature_columns, config
        ),
    )


def match_event_type_prototypes(
    event_type_assignments: pd.DataFrame | Sequence[pd.DataFrame],
    feature_columns: Sequence[str] | None = None,
    config: CrossMouseTaxonomyConfig | None = None,
) -> PrototypeMatchResult:
    """Match per-mouse event types using event-equal interpretable summaries.

    The input is the concatenation of the per-mouse
    ``event_type_assignments.csv`` tables.  Every event contributes once to
    its local prototype, and every local prototype contributes once to the
    cross-mouse GMM regardless of its event count.
    """

    config = config or CrossMouseTaxonomyConfig()
    events = _concatenate_tables(event_type_assignments, "event type")
    _require_columns(events, {"animal_id", "event_id", "event_type"})
    _reject_missing_identifiers(events, ["animal_id", "event_id", "event_type"])
    duplicate = events.duplicated(["animal_id", "event_id"], keep=False)
    if duplicate.any():
        raise ValueError("event assignments must contain one row per animal/event")
    features = _resolve_event_features(events, feature_columns)
    prototypes = _build_event_prototypes(events, features)
    matched = _match_prototypes(prototypes, features, config, kind="event")
    presence = _event_presence_matrix(matched["assignments"])
    summary = _event_global_summary(presence)
    return PrototypeMatchResult(
        model_grid=matched["grid"],
        prototype_assignments=matched["assignments"],
        presence_matrix=presence,
        global_summary=summary,
        feature_columns=matched["feature_columns"],
        model_bundle=matched["bundle"],
    )


def match_state_prototypes(
    state_summaries: pd.DataFrame | Sequence[pd.DataFrame],
    feature_columns: Sequence[str] | None = None,
    config: CrossMouseTaxonomyConfig | None = None,
) -> PrototypeMatchResult:
    """Match local states from concatenated per-mouse ``state_summary`` tables."""

    config = config or CrossMouseTaxonomyConfig()
    states = _concatenate_tables(state_summaries, "state summary")
    _require_columns(states, {"animal_id", "state"})
    _reject_missing_identifiers(states, ["animal_id", "state"])
    if states.duplicated(["animal_id", "state"], keep=False).any():
        raise ValueError("state summaries must contain one row per animal/state")
    features = _resolve_state_features(states, feature_columns)
    prototypes = _build_state_prototypes(states, features)
    matched = _match_prototypes(prototypes, features, config, kind="state")
    presence = _state_presence_matrix(matched["assignments"])
    summary = _state_global_summary(presence)
    return PrototypeMatchResult(
        model_grid=matched["grid"],
        prototype_assignments=matched["assignments"],
        presence_matrix=presence,
        global_summary=summary,
        feature_columns=matched["feature_columns"],
        model_bundle=matched["bundle"],
    )


def save_cross_mouse_taxonomy(
    result: CrossMouseTaxonomyResult, output_dir: str | Path
) -> None:
    """Persist all cross-mouse audit tables and fitted preprocessing objects."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, matched in (("event", result.event_types), ("state", result.states)):
        matched.model_grid.to_csv(
            root / f"{name}_global_model_grid.csv", index=False, encoding="utf-8-sig"
        )
        matched.prototype_assignments.to_csv(
            root / f"{name}_local_prototype_assignments.csv",
            index=False,
            encoding="utf-8-sig",
        )
        matched.presence_matrix.to_csv(
            root / f"{name}_global_presence_matrix.csv",
            index=False,
            encoding="utf-8-sig",
        )
        matched.global_summary.to_csv(
            root / f"{name}_global_summary.csv", index=False, encoding="utf-8-sig"
        )
    joblib.dump(
        {
            "event": result.event_types.model_bundle,
            "state": result.states.model_bundle,
        },
        root / "cross_mouse_prototype_models.joblib",
    )


def _concatenate_tables(
    tables: pd.DataFrame | Sequence[pd.DataFrame], name: str
) -> pd.DataFrame:
    if isinstance(tables, pd.DataFrame):
        result = tables.copy()
    else:
        frames = list(tables)
        if not frames:
            raise ValueError(f"no {name} tables were supplied")
        if not all(isinstance(frame, pd.DataFrame) for frame in frames):
            raise TypeError(f"all {name} inputs must be pandas DataFrames")
        result = pd.concat(frames, ignore_index=True, sort=False)
    if result.empty:
        raise ValueError(f"the {name} table is empty")
    return result


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"table is missing columns: {sorted(missing)}")


def _reject_missing_identifiers(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    if frame[list(columns)].isna().any(axis=None):
        raise ValueError(f"identifier columns cannot be missing: {list(columns)}")


def _resolve_event_features(
    frame: pd.DataFrame, requested: Sequence[str] | None
) -> list[str]:
    if requested is None:
        features = [
            str(column)
            for column in frame.columns
            if str(column).startswith(("median__", "iqr__"))
        ]
        features.extend(
            column
            for column in ("duration_s", "transition_count")
            if column in frame and column not in features
        )
    else:
        features = list(dict.fromkeys(str(column) for column in requested))
    return _validate_features(frame, features)


def _resolve_state_features(
    frame: pd.DataFrame, requested: Sequence[str] | None
) -> list[str]:
    if requested is None:
        features = [
            str(column)
            for column in frame.columns
            if str(column).startswith("median__")
        ]
    else:
        features = list(dict.fromkeys(str(column) for column in requested))
    return _validate_features(frame, features)


def _validate_features(frame: pd.DataFrame, features: Sequence[str]) -> list[str]:
    if not features:
        raise ValueError("no interpretable prototype features are available")
    missing = set(features) - set(frame.columns)
    if missing:
        raise ValueError(f"prototype features are missing: {sorted(missing)}")
    return list(features)


def _build_event_prototypes(
    events: pd.DataFrame, features: Sequence[str]
) -> pd.DataFrame:
    frame = events.copy()
    frame["animal_id"] = frame["animal_id"].astype(str)
    frame["local_event_type"] = frame["event_type"].astype(str)
    for feature in features:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    animal_totals = frame.groupby("animal_id", sort=True)["event_id"].nunique()
    rows: list[dict[str, Any]] = []
    for (animal, local_type), group in frame.groupby(
        ["animal_id", "local_event_type"], sort=True, observed=True
    ):
        event_count = int(group["event_id"].nunique())
        label = (
            str(group["event_type_label"].iloc[0])
            if "event_type_label" in group
            else f"{animal}-T{local_type}"
        )
        record: dict[str, Any] = {
            "animal_id": str(animal),
            "local_event_type": str(local_type),
            "local_event_type_label": label,
            "event_count": event_count,
            "event_fraction": float(event_count / animal_totals.loc[animal]),
        }
        for feature in features:
            record[feature] = _finite_median(group[feature])
        rows.append(record)
    return pd.DataFrame(rows)


def _build_state_prototypes(
    states: pd.DataFrame, features: Sequence[str]
) -> pd.DataFrame:
    frame = states.copy()
    frame["animal_id"] = frame["animal_id"].astype(str)
    frame["local_state"] = frame["state"].astype(str)
    for feature in features:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        animal = str(row["animal_id"])
        local_state = str(row["local_state"])
        record: dict[str, Any] = {
            "animal_id": animal,
            "local_state": local_state,
            "local_state_label": str(
                row.get("state_label", f"{animal}-S{local_state}")
            ),
            "window_count": _safe_int(row.get("windows", 0)),
            "event_count": _safe_int(row.get("events", 0)),
            "state_fraction": _safe_float(row.get("fraction", np.nan)),
        }
        for feature in features:
            record[feature] = _safe_float(row.get(feature, np.nan))
        rows.append(record)
    return pd.DataFrame(rows)


def _match_prototypes(
    prototypes: pd.DataFrame,
    requested_features: Sequence[str],
    config: CrossMouseTaxonomyConfig,
    kind: str,
) -> dict[str, Any]:
    matrix, usable_features, preprocessing = _prototype_matrix(
        prototypes, requested_features, config
    )
    allow_multiple = bool(usable_features) and len(prototypes) >= 2
    grid, models, selected = _gmm_bic_grid(
        matrix, config, allow_multiple=allow_multiple
    )
    model = models.get(selected)
    if model is None:
        raw_labels = np.zeros(len(prototypes), dtype=np.int32)
        probabilities = np.ones(len(prototypes), dtype=float)
    else:
        raw_labels = model.predict(matrix).astype(np.int32)
        probabilities = np.max(model.predict_proba(matrix), axis=1)
    mapping = _stable_component_mapping(raw_labels, matrix, prototypes["animal_id"])
    labels = np.asarray([mapping[int(value)] for value in raw_labels], dtype=np.int32)
    assignments = prototypes.copy()
    if kind == "event":
        assignments["global_event_type"] = labels
        assignments["global_event_type_label"] = [f"GE{value}" for value in labels]
        assignments["global_match_probability"] = probabilities
    else:
        assignments["global_state"] = labels
        assignments["global_state_label"] = [f"GS{value}" for value in labels]
        assignments["global_match_probability"] = probabilities
    bundle = {
        "kind": kind,
        "config": config,
        "requested_feature_columns": tuple(requested_features),
        "feature_columns": tuple(usable_features),
        **preprocessing,
        "model": model,
        "selected_global_types": int(selected),
        "component_mapping": mapping,
    }
    return {
        "grid": grid,
        "assignments": assignments,
        "feature_columns": tuple(usable_features),
        "bundle": bundle,
    }


def _prototype_matrix(
    prototypes: pd.DataFrame,
    features: Sequence[str],
    config: CrossMouseTaxonomyConfig,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    raw = prototypes[list(features)].apply(pd.to_numeric, errors="coerce")
    finite_fraction = np.isfinite(raw).mean(axis=0)
    sufficiently_observed = [
        feature
        for feature in features
        if finite_fraction[feature] >= config.minimum_feature_finite_fraction
    ]
    varying: list[str] = []
    for feature in sufficiently_observed:
        values = raw[feature].to_numpy(float)
        values = values[np.isfinite(values)]
        if len(values) >= 2 and float(np.quantile(values, 0.90) - np.quantile(values, 0.10)) > 1e-9:
            varying.append(feature)
    if not varying:
        return (
            np.zeros((len(prototypes), 1), dtype=float),
            [],
            {
                "feature_medians": {},
                "scaler": None,
                "preprocessing_status": "no_varying_features",
            },
        )
    selected = raw[varying].replace([np.inf, -np.inf], np.nan)
    medians = selected.median(axis=0).fillna(0.0)
    values = selected.fillna(medians).to_numpy(float)
    scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(values)
    scaled = np.clip(
        scaler.transform(values),
        -config.clip_scaled_values,
        config.clip_scaled_values,
    )
    return (
        scaled,
        varying,
        {
            "feature_medians": {key: float(value) for key, value in medians.items()},
            "scaler": scaler,
            "preprocessing_status": "ok",
        },
    )


def _gmm_bic_grid(
    values: np.ndarray,
    config: CrossMouseTaxonomyConfig,
    allow_multiple: bool,
) -> tuple[pd.DataFrame, dict[int, GaussianMixture], int]:
    samples = len(values)
    distinct_samples = len(np.unique(np.round(values, decimals=10), axis=0))
    maximum = (
        min(config.maximum_global_types, samples, max(1, distinct_samples))
        if allow_multiple
        else 1
    )
    rows: list[dict[str, Any]] = []
    models: dict[int, GaussianMixture] = {}
    if samples == 1:
        grid = pd.DataFrame(
            [
                {
                    "global_type_count": 1,
                    "local_prototypes": 1,
                    "bic": float("nan"),
                    "bic_delta": 0.0,
                    "converged": True,
                    "selected": True,
                    "selection_basis": "single_local_prototype",
                }
            ]
        )
        return grid, models, 1
    for count in range(1, maximum + 1):
        try:
            model = GaussianMixture(
                n_components=count,
                covariance_type=config.covariance_type,
                reg_covar=config.regularization,
                n_init=config.n_init,
                random_state=config.random_seed + 101 * count,
            ).fit(values)
            bic = float(model.bic(values))
            models[count] = model
            rows.append(
                {
                    "global_type_count": count,
                    "local_prototypes": samples,
                    "bic": bic,
                    "converged": bool(model.converged_),
                    "selection_basis": "minimum_gmm_bic",
                }
            )
        except (ValueError, np.linalg.LinAlgError):
            rows.append(
                {
                    "global_type_count": count,
                    "local_prototypes": samples,
                    "bic": float("nan"),
                    "converged": False,
                    "selection_basis": "fit_failed",
                }
            )
    grid = pd.DataFrame(rows)
    finite = grid.loc[grid["bic"].notna()]
    if finite.empty or 1 not in models:
        raise RuntimeError("the mandatory K=1 global GMM could not be fit")
    selected = int(
        finite.sort_values(
            ["bic", "global_type_count"], ascending=[True, True], kind="stable"
        )["global_type_count"].iloc[0]
    )
    minimum_bic = float(finite["bic"].min())
    grid["bic_delta"] = grid["bic"] - minimum_bic
    grid["selected"] = grid["global_type_count"].eq(selected)
    return grid[
        [
            "global_type_count",
            "local_prototypes",
            "bic",
            "bic_delta",
            "converged",
            "selected",
            "selection_basis",
        ]
    ], models, selected


def _stable_component_mapping(
    raw_labels: np.ndarray,
    values: np.ndarray,
    animals: pd.Series,
) -> dict[int, int]:
    records: list[tuple[int, int, int, tuple[float, ...]]] = []
    animal_values = animals.astype(str).to_numpy()
    for label in sorted(np.unique(raw_labels)):
        mask = raw_labels == label
        center = tuple(np.mean(values[mask], axis=0).round(12).tolist())
        records.append(
            (
                int(label),
                int(len(np.unique(animal_values[mask]))),
                int(np.count_nonzero(mask)),
                center,
            )
        )
    order = sorted(records, key=lambda row: (-row[1], -row[2], row[3], row[0]))
    return {raw: index + 1 for index, (raw, _, _, _) in enumerate(order)}


def _event_presence_matrix(assignments: pd.DataFrame) -> pd.DataFrame:
    animals = sorted(assignments["animal_id"].astype(str).unique())
    global_types = sorted(assignments["global_event_type"].astype(int).unique())
    rows: list[dict[str, Any]] = []
    for global_type in global_types:
        for animal in animals:
            matches = assignments.loc[
                assignments["global_event_type"].eq(global_type)
                & assignments["animal_id"].astype(str).eq(animal)
            ]
            if matches.empty:
                rows.append(
                    {
                        "global_event_type": global_type,
                        "global_event_type_label": f"GE{global_type}",
                        "animal_id": animal,
                        "local_event_type": pd.NA,
                        "local_event_type_label": pd.NA,
                        "event_count": 0,
                        "event_fraction": 0.0,
                        "global_match_probability": float("nan"),
                        "present": False,
                    }
                )
                continue
            for row in matches.to_dict("records"):
                rows.append(
                    {
                        "global_event_type": global_type,
                        "global_event_type_label": f"GE{global_type}",
                        "animal_id": animal,
                        "local_event_type": row["local_event_type"],
                        "local_event_type_label": row["local_event_type_label"],
                        "event_count": int(row["event_count"]),
                        "event_fraction": float(row["event_fraction"]),
                        "global_match_probability": float(
                            row["global_match_probability"]
                        ),
                        "present": True,
                    }
                )
    return pd.DataFrame(rows)


def _state_presence_matrix(assignments: pd.DataFrame) -> pd.DataFrame:
    animals = sorted(assignments["animal_id"].astype(str).unique())
    global_states = sorted(assignments["global_state"].astype(int).unique())
    rows: list[dict[str, Any]] = []
    for global_state in global_states:
        for animal in animals:
            matches = assignments.loc[
                assignments["global_state"].eq(global_state)
                & assignments["animal_id"].astype(str).eq(animal)
            ]
            if matches.empty:
                rows.append(
                    {
                        "global_state": global_state,
                        "global_state_label": f"GS{global_state}",
                        "animal_id": animal,
                        "local_state": pd.NA,
                        "local_state_label": pd.NA,
                        "window_count": 0,
                        "event_count": 0,
                        "state_fraction": 0.0,
                        "global_match_probability": float("nan"),
                        "present": False,
                    }
                )
                continue
            for row in matches.to_dict("records"):
                rows.append(
                    {
                        "global_state": global_state,
                        "global_state_label": f"GS{global_state}",
                        "animal_id": animal,
                        "local_state": row["local_state"],
                        "local_state_label": row["local_state_label"],
                        "window_count": int(row["window_count"]),
                        "event_count": int(row["event_count"]),
                        "state_fraction": float(row["state_fraction"]),
                        "global_match_probability": float(
                            row["global_match_probability"]
                        ),
                        "present": True,
                    }
                )
    return pd.DataFrame(rows)


def _event_global_summary(presence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for global_type, group in presence.groupby("global_event_type", sort=True):
        observed = group.loc[group["present"]]
        rows.append(
            {
                "global_event_type": int(global_type),
                "global_event_type_label": f"GE{int(global_type)}",
                "mice_present": int(observed["animal_id"].nunique()),
                "mouse_ids": ";".join(sorted(observed["animal_id"].astype(str).unique())),
                "local_types": int(len(observed)),
                "total_event_count": int(observed["event_count"].sum()),
                "median_mouse_event_fraction": float(
                    observed["event_fraction"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def _state_global_summary(presence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for global_state, group in presence.groupby("global_state", sort=True):
        observed = group.loc[group["present"]]
        rows.append(
            {
                "global_state": int(global_state),
                "global_state_label": f"GS{int(global_state)}",
                "mice_present": int(observed["animal_id"].nunique()),
                "mouse_ids": ";".join(sorted(observed["animal_id"].astype(str).unique())),
                "local_states": int(len(observed)),
                "total_windows": int(observed["window_count"].sum()),
                "total_event_memberships": int(observed["event_count"].sum()),
                "median_mouse_state_fraction": float(
                    observed["state_fraction"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def _finite_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    numeric = numeric[np.isfinite(numeric)]
    return float(np.median(numeric)) if len(numeric) else float("nan")


def _safe_int(value: Any) -> int:
    return int(value) if pd.notna(value) else 0


def _safe_float(value: Any) -> float:
    return float(value) if pd.notna(value) else float("nan")


__all__ = [
    "CrossMouseTaxonomyConfig",
    "CrossMouseTaxonomyResult",
    "PrototypeMatchResult",
    "match_cross_mouse_taxonomy",
    "match_event_type_prototypes",
    "match_state_prototypes",
    "save_cross_mouse_taxonomy",
]
