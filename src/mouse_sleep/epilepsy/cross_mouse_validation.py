from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from mouse_sleep.epilepsy.cross_mouse_taxonomy import (
    CrossMouseTaxonomyConfig,
    match_cross_mouse_taxonomy,
)


@dataclass(frozen=True)
class CrossMouseValidationResult:
    sensitivity: pd.DataFrame
    local_event_inventory: pd.DataFrame
    local_state_inventory: pd.DataFrame
    status: dict[str, Any]


def _selected_count(grid: pd.DataFrame) -> int:
    return int(grid.loc[grid["selected"].astype(bool), "global_type_count"].iloc[0])


def _local_event_inventory(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (animal, local_type), group in events.groupby(
        ["animal_id", "event_type"], sort=True
    ):
        count = int(group["event_id"].nunique())
        total = int(
            events.loc[events["animal_id"].astype(str).eq(str(animal)), "event_id"].nunique()
        )
        duration = pd.to_numeric(group["duration_s"], errors="coerce")
        transitions = pd.to_numeric(group["transition_count"], errors="coerce")
        rows.append(
            {
                "animal_id": str(animal),
                "local_event_type": int(local_type),
                "local_event_type_label": (
                    str(group["event_type_label"].iloc[0])
                    if "event_type_label" in group
                    else f"{animal}-T{int(local_type) + 1}"
                ),
                "event_count": count,
                "event_fraction": float(count / max(total, 1)),
                "duration_median_s": float(duration.median()),
                "duration_q25_s": float(duration.quantile(0.25)),
                "duration_q75_s": float(duration.quantile(0.75)),
                "transition_median": float(transitions.median()),
            }
        )
    return pd.DataFrame(rows)


def _local_state_inventory(states: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in (
            "animal_id",
            "state",
            "state_label",
            "windows",
            "events",
            "fraction",
            "median_posterior",
        )
        if column in states
    ]
    return states.loc[:, columns].sort_values(
        ["animal_id", "state"], kind="stable"
    ).reset_index(drop=True)


def audit_cross_mouse_matching(
    event_type_assignments: pd.DataFrame,
    state_summaries: pd.DataFrame,
    event_feature_columns: Sequence[str],
    state_feature_columns: Sequence[str],
    regularizations: Sequence[float] = (1e-4, 0.01, 0.05, 0.10, 0.25),
    random_seed: int = 20260812,
) -> CrossMouseValidationResult:
    """Test whether cross-mouse prototype counts survive plausible GMM scales."""

    events = event_type_assignments.copy()
    states = state_summaries.copy()
    rows: list[dict[str, Any]] = []
    for regularization in regularizations:
        config = CrossMouseTaxonomyConfig(
            regularization=float(regularization),
            n_init=50,
            random_seed=random_seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = match_cross_mouse_taxonomy(
                events,
                states,
                event_feature_columns=event_feature_columns,
                state_feature_columns=state_feature_columns,
                config=config,
            )
        rows.append(
            {
                "regularization": float(regularization),
                "minimum_component_sd_scaled": float(np.sqrt(regularization)),
                "event_global_type_count": _selected_count(
                    result.event_types.model_grid
                ),
                "state_global_type_count": _selected_count(
                    result.states.model_grid
                ),
                "event_maximum_mouse_coverage": int(
                    result.event_types.global_summary["mice_present"].max()
                ),
                "state_maximum_mouse_coverage": int(
                    result.states.global_summary["mice_present"].max()
                ),
            }
        )
    sensitivity = pd.DataFrame(rows)
    event_local_prototypes = int(
        events.groupby(["animal_id", "event_type"], sort=True).ngroups
    )
    state_local_prototypes = int(len(states))
    event_multi_type_mice = int(
        (
            events.groupby("animal_id", sort=True)["event_type"].nunique()
            > 1
        ).sum()
    )
    state_multi_state_mice = int(
        (states.groupby("animal_id", sort=True)["state"].nunique() > 1).sum()
    )
    event_counts = sensitivity["event_global_type_count"].unique()
    state_counts = sensitivity["state_global_type_count"].unique()
    base = sensitivity.iloc[0]
    event_identifiable = bool(
        event_multi_type_mice >= 2
        and len(event_counts) == 1
        and int(base["event_global_type_count"]) < event_local_prototypes
    )
    state_identifiable = bool(
        state_multi_state_mice >= 2
        and len(state_counts) == 1
        and int(base["state_global_type_count"]) < state_local_prototypes
    )
    status = {
        "schema_version": 1,
        "event_matching_status": (
            "identifiable" if event_identifiable else "not_identifiable"
        ),
        "state_matching_status": (
            "identifiable" if state_identifiable else "not_identifiable"
        ),
        "event_local_prototypes": event_local_prototypes,
        "state_local_prototypes": state_local_prototypes,
        "mice_with_multiple_local_event_types": event_multi_type_mice,
        "mice_with_multiple_local_states": state_multi_state_mice,
        "event_selected_counts_across_regularization": sorted(
            int(value) for value in event_counts
        ),
        "state_selected_counts_across_regularization": sorted(
            int(value) for value in state_counts
        ),
        "primary_failure_reasons": [
            reason
            for reason, failed in (
                (
                    "Only one mouse currently has more than one validated whole-seizure type; cross-mouse type identity cannot be replicated across multi-type mice.",
                    event_multi_type_mice < 2,
                ),
                (
                    "The selected global event-type count changes across plausible scaled covariance floors.",
                    len(event_counts) > 1,
                ),
                (
                    "The selected global state count changes across plausible scaled covariance floors.",
                    len(state_counts) > 1,
                ),
                (
                    "The base event model assigns one global component per local prototype.",
                    int(base["event_global_type_count"]) >= event_local_prototypes,
                ),
                (
                    "The base state model is near the configured component ceiling rather than a stable cross-mouse optimum.",
                    int(base["state_global_type_count"])
                    >= min(8, state_local_prototypes),
                ),
            )
            if failed
        ],
        "interpretation": (
            "Per-mouse local states and local whole-seizure types remain valid. "
            "Current data do not support a unique global cross-mouse taxonomy; "
            "global GMM assignments must be treated as sensitivity output only."
        ),
    }
    return CrossMouseValidationResult(
        sensitivity=sensitivity,
        local_event_inventory=_local_event_inventory(events),
        local_state_inventory=_local_state_inventory(states),
        status=status,
    )


def save_cross_mouse_validation(
    result: CrossMouseValidationResult,
    output_dir: str | Path,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result.sensitivity.to_csv(
        root / "matching_regularization_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result.local_event_inventory.to_csv(
        root / "validated_local_event_type_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result.local_state_inventory.to_csv(
        root / "validated_local_state_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (root / "cross_mouse_identifiability.json").write_text(
        json.dumps(result.status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


__all__ = [
    "CrossMouseValidationResult",
    "audit_cross_mouse_matching",
    "save_cross_mouse_validation",
]
