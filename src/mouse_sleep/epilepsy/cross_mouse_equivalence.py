from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_EQUIVALENCE_FEATURE_BLOCKS: dict[str, tuple[str, ...]] = {
    "amplitude": (
        "median__amplitude_vs_reference_log",
        "median__line_length_vs_reference_log",
    ),
    "shape": (
        "median__aggregate_crest_factor",
        "median__aggregate_hjorth_mobility",
        "median__aggregate_hjorth_complexity",
    ),
    "frequency": (
        "median__aggregate_median_frequency_hz",
        "median__aggregate_spectral_entropy",
        "median__theta_delta_ratio",
        "median__high_low_ratio",
    ),
}

REQUIRED_FEATURE_BLOCKS = ("amplitude", "shape", "frequency")


@dataclass(frozen=True)
class CrossMouseEquivalenceConfig:
    """Operational rules for conservative cross-mouse type matching.

    ``operational_effect_threshold`` is a transparent analysis convention in
    units of pooled robust within-type variation.  It is not a biological or
    internationally standardized seizure threshold.  A pair is linked only
    when both its point estimate and event-bootstrap confidence upper bound
    are no larger than this margin in every required feature block.
    """

    operational_effect_threshold: float = 0.75
    minimum_events_per_local_type: int = 20
    bootstrap_repeats: int = 1000
    confidence_level: float = 0.95
    minimum_valid_bootstrap_fraction: float = 0.80
    random_seed: int = 20260812

    def __post_init__(self) -> None:
        if self.operational_effect_threshold <= 0.0:
            raise ValueError("operational_effect_threshold must be positive")
        if self.minimum_events_per_local_type < 2:
            raise ValueError("minimum_events_per_local_type must be at least two")
        if self.bootstrap_repeats < 20:
            raise ValueError("bootstrap_repeats must be at least 20")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between zero and one")
        if not 0.0 < self.minimum_valid_bootstrap_fraction <= 1.0:
            raise ValueError(
                "minimum_valid_bootstrap_fraction must be in (0, 1]"
            )


@dataclass(frozen=True)
class CrossMouseEquivalenceResult:
    """Formal tables from conservative equivalence-gated matching."""

    local_inventory: pd.DataFrame
    pairwise_evidence: pd.DataFrame
    global_type_table: pd.DataFrame
    presence_matrix: pd.DataFrame


def match_event_types_by_equivalence(
    event_type_assignments: pd.DataFrame | Sequence[pd.DataFrame],
    feature_blocks: Mapping[str, Sequence[str]] | None = None,
    config: CrossMouseEquivalenceConfig | None = None,
) -> CrossMouseEquivalenceResult:
    """Conservatively match per-mouse whole-seizure types.

    Parameters
    ----------
    event_type_assignments:
        One row per complete annotated seizure, containing ``animal_id``,
        ``event_id``, and ``event_type`` plus interpretable event summaries.
        Window-level rows are deliberately rejected when they duplicate an
        animal/event identifier.
    feature_blocks:
        Exactly three blocks named ``amplitude``, ``shape``, and ``frequency``.
        Missing feature columns are ignored within a block, but every pair
        needs at least one estimable feature in every block before it can be
        linked automatically.
    config:
        Operational equivalence margin, event-count requirement, and event
        bootstrap settings.

    Returns
    -------
    CrossMouseEquivalenceResult
        Local inventory, one-row-per-pair evidence, formal global type table,
        and a dense ``global type x mouse`` presence table.

    Notes
    -----
    The algorithm does not estimate a global component count.  It first tests
    local prototype pairs, then applies deterministic complete-link merging.
    Two clusters merge only if every cross-cluster local pair passed all three
    operational equivalence gates.  Local types from the same mouse are a
    permanent cannot-link constraint.
    """

    config = config or CrossMouseEquivalenceConfig()
    blocks = _validate_feature_blocks(
        feature_blocks or DEFAULT_EQUIVALENCE_FEATURE_BLOCKS
    )
    events = _concatenate_event_tables(event_type_assignments)
    _validate_event_rows(events)
    inventory = _build_local_inventory(events, blocks, config)
    evidence = _build_pairwise_evidence(events, inventory, blocks, config)
    memberships = _complete_link_memberships(inventory, evidence)
    inventory, global_types, presence = _build_formal_tables(
        inventory, memberships, evidence, config
    )
    return CrossMouseEquivalenceResult(
        local_inventory=inventory,
        pairwise_evidence=evidence,
        global_type_table=global_types,
        presence_matrix=presence,
    )


def save_cross_mouse_equivalence(
    result: CrossMouseEquivalenceResult, output_dir: str | Path
) -> None:
    """Save the four formal evidence tables; no figures or model files are made."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result.local_inventory.to_csv(
        root / "local_type_inventory.csv", index=False, encoding="utf-8-sig"
    )
    result.pairwise_evidence.to_csv(
        root / "pairwise_equivalence_evidence.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result.global_type_table.to_csv(
        root / "formal_global_type_table.csv", index=False, encoding="utf-8-sig"
    )
    result.presence_matrix.to_csv(
        root / "formal_global_type_presence.csv", index=False, encoding="utf-8-sig"
    )


def _concatenate_event_tables(
    tables: pd.DataFrame | Sequence[pd.DataFrame],
) -> pd.DataFrame:
    if isinstance(tables, pd.DataFrame):
        frame = tables.copy()
    else:
        parts = list(tables)
        if not parts:
            raise ValueError("no event type assignment tables were supplied")
        if not all(isinstance(part, pd.DataFrame) for part in parts):
            raise TypeError("all event type assignment inputs must be DataFrames")
        frame = pd.concat(parts, ignore_index=True, sort=False)
    if frame.empty:
        raise ValueError("event type assignments are empty")
    return frame


def _validate_event_rows(events: pd.DataFrame) -> None:
    required = {"animal_id", "event_id", "event_type"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"event type assignments are missing: {sorted(missing)}")
    if events[list(required)].isna().any(axis=None):
        raise ValueError("animal, event, and local type identifiers cannot be missing")
    if events.duplicated(["animal_id", "event_id"], keep=False).any():
        raise ValueError(
            "expected one row per complete seizure; duplicate animal/event rows "
            "look like window-level pseudoreplication"
        )


def _validate_feature_blocks(
    feature_blocks: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    if set(feature_blocks) != set(REQUIRED_FEATURE_BLOCKS):
        raise ValueError(
            "feature_blocks must contain exactly amplitude, shape, and frequency"
        )
    result: dict[str, tuple[str, ...]] = {}
    for block in REQUIRED_FEATURE_BLOCKS:
        columns = tuple(dict.fromkeys(str(value) for value in feature_blocks[block]))
        if not columns:
            raise ValueError(f"feature block {block!r} is empty")
        result[block] = columns
    return result


def _build_local_inventory(
    events: pd.DataFrame,
    blocks: Mapping[str, Sequence[str]],
    config: CrossMouseEquivalenceConfig,
) -> pd.DataFrame:
    frame = events.copy()
    frame["animal_id"] = frame["animal_id"].astype(str)
    frame["local_event_type"] = frame["event_type"].astype(str)
    frame["local_prototype_id"] = (
        frame["animal_id"] + "::" + frame["local_event_type"]
    )
    animal_totals = frame.groupby("animal_id", sort=True)["event_id"].nunique()
    rows: list[dict[str, Any]] = []
    requested = [feature for columns in blocks.values() for feature in columns]
    for prototype, group in frame.groupby("local_prototype_id", sort=True):
        animal = str(group["animal_id"].iloc[0])
        local_type = str(group["local_event_type"].iloc[0])
        event_count = int(group["event_id"].nunique())
        label = (
            str(group["event_type_label"].iloc[0])
            if "event_type_label" in group
            else f"{animal}-T{local_type}"
        )
        if "session_id" in group:
            session_counts = group.groupby(group["session_id"].astype(str)).size()
            session_count = int(len(session_counts))
            maximum_session_fraction = float(session_counts.max() / len(group))
        else:
            session_count = 0
            maximum_session_fraction = float("nan")
        estimable = {
            block: sum(
                feature in group
                and pd.to_numeric(group[feature], errors="coerce").notna().sum()
                >= 2
                for feature in columns
            )
            for block, columns in blocks.items()
        }
        event_status = (
            "eligible_by_event_count"
            if event_count >= config.minimum_events_per_local_type
            else "data_insufficient_event_count"
        )
        feature_status = (
            "all_required_blocks_estimable"
            if all(estimable.values())
            else "data_insufficient_feature_block"
        )
        record: dict[str, Any] = {
            "local_prototype_id": str(prototype),
            "animal_id": animal,
            "local_event_type": local_type,
            "local_event_type_label": label,
            "event_count": event_count,
            "event_fraction": float(event_count / animal_totals.loc[animal]),
            "session_count": session_count,
            "maximum_session_fraction": maximum_session_fraction,
            "session_evidence_status": (
                "multi_session_observed" if session_count >= 2 else "single_session_only"
            ),
            "event_count_status": event_status,
            "feature_evidence_status": feature_status,
            "automatic_matching_eligible": bool(
                event_status == "eligible_by_event_count"
                and feature_status == "all_required_blocks_estimable"
            ),
            "minimum_required_events": int(config.minimum_events_per_local_type),
            "operational_effect_threshold": float(
                config.operational_effect_threshold
            ),
            "threshold_basis": "operational_not_biological_standard",
        }
        for block in REQUIRED_FEATURE_BLOCKS:
            record[f"{block}_features_estimable"] = int(estimable[block])
        for feature in requested:
            if feature in group:
                record[f"prototype_median__{feature}"] = _finite_median(
                    group[feature]
                )
        rows.append(record)
    return pd.DataFrame(rows).sort_values(
        ["animal_id", "local_event_type_label"], kind="stable"
    ).reset_index(drop=True)


def _build_pairwise_evidence(
    events: pd.DataFrame,
    inventory: pd.DataFrame,
    blocks: Mapping[str, Sequence[str]],
    config: CrossMouseEquivalenceConfig,
) -> pd.DataFrame:
    prepared = events.copy()
    prepared["animal_id"] = prepared["animal_id"].astype(str)
    prepared["local_event_type"] = prepared["event_type"].astype(str)
    prepared["local_prototype_id"] = (
        prepared["animal_id"] + "::" + prepared["local_event_type"]
    )
    all_features = [feature for columns in blocks.values() for feature in columns]
    for feature in all_features:
        if feature in prepared:
            prepared[feature] = pd.to_numeric(prepared[feature], errors="coerce")
    groups = {
        key: group.reset_index(drop=True)
        for key, group in prepared.groupby("local_prototype_id", sort=True)
    }
    inventory_map = inventory.set_index("local_prototype_id")
    rows: list[dict[str, Any]] = []
    for prototype_a, prototype_b in itertools.combinations(sorted(groups), 2):
        group_a = groups[prototype_a]
        group_b = groups[prototype_b]
        info_a = inventory_map.loc[prototype_a]
        info_b = inventory_map.loc[prototype_b]
        cannot_link = str(info_a["animal_id"]) == str(info_b["animal_id"])
        event_count_sufficient = bool(
            int(info_a["event_count"]) >= config.minimum_events_per_local_type
            and int(info_b["event_count"]) >= config.minimum_events_per_local_type
        )
        record: dict[str, Any] = {
            "prototype_a": prototype_a,
            "prototype_b": prototype_b,
            "animal_a": str(info_a["animal_id"]),
            "animal_b": str(info_b["animal_id"]),
            "local_type_a": str(info_a["local_event_type"]),
            "local_type_b": str(info_b["local_event_type"]),
            "local_label_a": str(info_a["local_event_type_label"]),
            "local_label_b": str(info_b["local_event_type_label"]),
            "events_a": int(info_a["event_count"]),
            "events_b": int(info_b["event_count"]),
            "cannot_link_same_mouse": cannot_link,
            "event_count_sufficient": event_count_sufficient,
            "minimum_required_events": int(config.minimum_events_per_local_type),
            "operational_effect_threshold": float(
                config.operational_effect_threshold
            ),
            "threshold_basis": "operational_not_biological_standard",
            "bootstrap_unit": "complete_event",
            "bootstrap_repeats": int(config.bootstrap_repeats),
        }
        block_results: dict[str, dict[str, Any]] = {}
        for block, features in blocks.items():
            block_results[block] = _block_equivalence_evidence(
                group_a,
                group_b,
                features,
                config,
                prototype_a,
                prototype_b,
                block,
            )
            for key, value in block_results[block].items():
                record[f"{block}_{key}"] = value
        all_estimable = all(
            result["status"] == "ok" for result in block_results.values()
        )
        all_points_pass = bool(
            all(
                float(result["effect_max"])
                <= config.operational_effect_threshold
                for result in block_results.values()
                if result["status"] == "ok"
            )
        ) if all_estimable else False
        all_cis_pass = bool(
            all(
                float(result["ci_high"])
                <= config.operational_effect_threshold
                for result in block_results.values()
                if result["status"] == "ok"
            )
        ) if all_estimable else False
        any_ci_above_margin = bool(
            any(
                float(result["ci_low"])
                > config.operational_effect_threshold
                for result in block_results.values()
                if result["status"] == "ok"
            )
        ) if all_estimable else False
        maximum_effect = (
            max(float(result["effect_max"]) for result in block_results.values())
            if all_estimable
            else float("nan")
        )
        maximum_ci_high = (
            max(float(result["ci_high"]) for result in block_results.values())
            if all_estimable
            else float("nan")
        )
        if cannot_link:
            decision = "cannot_link_same_mouse"
        elif not event_count_sufficient:
            decision = "data_insufficient_event_count"
        elif not all_estimable:
            decision = "data_insufficient_feature_or_bootstrap"
        elif all_points_pass and all_cis_pass:
            decision = "confirmed_operational_equivalence"
        elif any_ci_above_margin:
            decision = "evidence_exceeds_operational_margin"
        else:
            decision = "inconclusive_equivalence_evidence"
        automatic_link = decision == "confirmed_operational_equivalence"
        record.update(
            {
                "all_required_blocks_estimable": all_estimable,
                "all_block_points_within_margin": all_points_pass,
                "all_block_ci_high_within_margin": all_cis_pass,
                "maximum_block_effect": maximum_effect,
                "maximum_block_ci_high": maximum_ci_high,
                "equivalence_decision": decision,
                "automatic_link": automatic_link,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _block_equivalence_evidence(
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
    requested_features: Sequence[str],
    config: CrossMouseEquivalenceConfig,
    prototype_a: str,
    prototype_b: str,
    block: str,
) -> dict[str, Any]:
    features = [
        feature
        for feature in requested_features
        if feature in group_a
        and feature in group_b
        and pd.to_numeric(group_a[feature], errors="coerce").notna().sum() >= 2
        and pd.to_numeric(group_b[feature], errors="coerce").notna().sum() >= 2
    ]
    base: dict[str, Any] = {
        "features_requested": ";".join(requested_features),
        "features_estimable": ";".join(features),
        "estimable_feature_count": int(len(features)),
        "driver_feature": pd.NA,
        "effect_max": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "valid_bootstrap_repeats": 0,
        "status": "no_estimable_features",
    }
    if not features:
        return base
    effects = {
        feature: _absolute_robust_effect(
            pd.to_numeric(group_a[feature], errors="coerce").to_numpy(float),
            pd.to_numeric(group_b[feature], errors="coerce").to_numpy(float),
        )
        for feature in features
    }
    driver = max(features, key=lambda feature: (effects[feature], feature))
    point = float(max(effects.values()))
    seed = _stable_seed(
        config.random_seed, prototype_a, prototype_b, block
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.full(config.bootstrap_repeats, np.nan, dtype=float)
    for repeat in range(config.bootstrap_repeats):
        sampled_a = group_a.iloc[
            rng.integers(0, len(group_a), size=len(group_a))
        ]
        sampled_b = group_b.iloc[
            rng.integers(0, len(group_b), size=len(group_b))
        ]
        replicate_effects = [
            _absolute_robust_effect(
                pd.to_numeric(sampled_a[feature], errors="coerce").to_numpy(float),
                pd.to_numeric(sampled_b[feature], errors="coerce").to_numpy(float),
            )
            for feature in features
        ]
        finite_or_infinite = [value for value in replicate_effects if not np.isnan(value)]
        if finite_or_infinite:
            bootstrap[repeat] = max(finite_or_infinite)
    valid = bootstrap[~np.isnan(bootstrap)]
    required = int(
        np.ceil(
            config.bootstrap_repeats * config.minimum_valid_bootstrap_fraction
        )
    )
    base.update(
        {
            "driver_feature": driver,
            "effect_max": point,
            "valid_bootstrap_repeats": int(len(valid)),
        }
    )
    if len(valid) < required:
        base["status"] = "insufficient_bootstrap_support"
        return base
    alpha = (1.0 - config.confidence_level) / 2.0
    base.update(
        {
            "ci_low": float(np.quantile(valid, alpha)),
            "ci_high": float(np.quantile(valid, 1.0 - alpha)),
            "status": "ok",
        }
    )
    return base


def _absolute_robust_effect(values_a: np.ndarray, values_b: np.ndarray) -> float:
    a = values_a[np.isfinite(values_a)]
    b = values_b[np.isfinite(values_b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    difference = abs(float(np.median(a) - np.median(b)))
    scale_a = _robust_scale(a)
    scale_b = _robust_scale(b)
    positive = [
        scale for scale in (scale_a, scale_b) if np.isfinite(scale) and scale > 1e-12
    ]
    if len(positive) == 2:
        scale = float(np.sqrt((positive[0] ** 2 + positive[1] ** 2) / 2.0))
    elif len(positive) == 1:
        scale = float(positive[0])
    else:
        return 0.0 if difference <= 1e-12 else float("inf")
    return float(difference / scale)


def _robust_scale(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    median = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - median)))
    if mad > 1e-12:
        return mad
    iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25)) / 1.349
    if iqr > 1e-12:
        return iqr
    standard = float(np.std(values, ddof=1))
    return standard if standard > 1e-12 else 0.0


def _complete_link_memberships(
    inventory: pd.DataFrame, evidence: pd.DataFrame
) -> list[list[str]]:
    clusters = [
        [str(value)] for value in inventory["local_prototype_id"].sort_values()
    ]
    automatic: dict[tuple[str, str], bool] = {}
    scores: dict[tuple[str, str], float] = {}
    for row in evidence.to_dict("records"):
        pair = tuple(sorted((str(row["prototype_a"]), str(row["prototype_b"]))))
        automatic[pair] = bool(row["automatic_link"])
        scores[pair] = float(row["maximum_block_ci_high"])
    animal_by_prototype = inventory.set_index("local_prototype_id")[
        "animal_id"
    ].astype(str).to_dict()
    while True:
        candidates: list[tuple[float, tuple[str, ...], int, int]] = []
        for left, right in itertools.combinations(range(len(clusters)), 2):
            first = clusters[left]
            second = clusters[right]
            animals = [animal_by_prototype[value] for value in first + second]
            if len(animals) != len(set(animals)):
                continue
            pairs = [
                tuple(sorted((a, b))) for a in first for b in second
            ]
            if not all(automatic.get(pair, False) for pair in pairs):
                continue
            score = max(scores[pair] for pair in pairs)
            members = tuple(sorted(first + second))
            candidates.append((score, members, left, right))
        if not candidates:
            break
        _, members, left, right = min(candidates, key=lambda item: (item[0], item[1]))
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {left, right}
        ]
        clusters.append(list(members))
        clusters.sort(key=lambda values: tuple(values))
    return sorted((sorted(cluster) for cluster in clusters), key=lambda values: tuple(values))


def _build_formal_tables(
    inventory: pd.DataFrame,
    memberships: Sequence[Sequence[str]],
    evidence: pd.DataFrame,
    config: CrossMouseEquivalenceConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_inventory = inventory.copy()
    inventory_map = output_inventory.set_index("local_prototype_id")
    membership_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    ordered_memberships = sorted(
        memberships,
        key=lambda members: (
            -len(members),
            tuple(
                inventory_map.loc[member, "local_event_type_label"]
                for member in members
            ),
        ),
    )
    for index, members in enumerate(ordered_memberships, start=1):
        member_rows = inventory_map.loc[list(members)].reset_index()
        global_id = f"GT{index:03d}"
        if len(members) > 1:
            global_label = f"XM-GT{index:02d}"
            evidence_status = "confirmed_operational_cross_mouse_equivalence"
            unmatched_reason = ""
        else:
            global_label = str(member_rows["local_event_type_label"].iloc[0])
            if not bool(member_rows["automatic_matching_eligible"].iloc[0]):
                evidence_status = "local_type_retained_data_insufficient"
                unmatched_reason = str(
                    member_rows["event_count_status"].iloc[0]
                    if member_rows["event_count_status"].iloc[0]
                    != "eligible_by_event_count"
                    else member_rows["feature_evidence_status"].iloc[0]
                )
            else:
                evidence_status = "local_type_retained_no_confirmed_match"
                unmatched_reason = "no_pair_passed_all_operational_equivalence_gates"
        mouse_ids = sorted(member_rows["animal_id"].astype(str).unique())
        local_labels = member_rows["local_event_type_label"].astype(str).tolist()
        global_rows.append(
            {
                "global_type_id": global_id,
                "global_type_label": global_label,
                "global_type_status": evidence_status,
                "unmatched_reason": unmatched_reason,
                "mouse_count": int(len(mouse_ids)),
                "mouse_ids": ";".join(mouse_ids),
                "local_type_count": int(len(members)),
                "local_type_labels": ";".join(local_labels),
                "total_event_count": int(member_rows["event_count"].sum()),
                "matching_weight": "one_equal_weight_per_local_prototype",
                "linkage_rule": "conservative_complete_link",
                "operational_effect_threshold": float(
                    config.operational_effect_threshold
                ),
                "threshold_basis": "operational_not_biological_standard",
            }
        )
        for member in members:
            membership_rows.append(
                {
                    "local_prototype_id": member,
                    "global_type_id": global_id,
                    "global_type_label": global_label,
                    "global_type_status": evidence_status,
                }
            )
    output_inventory = output_inventory.merge(
        pd.DataFrame(membership_rows), on="local_prototype_id", validate="one_to_one"
    )
    global_table = pd.DataFrame(global_rows)
    animals = sorted(output_inventory["animal_id"].astype(str).unique())
    presence_rows: list[dict[str, Any]] = []
    for global_row in global_table.to_dict("records"):
        members = output_inventory.loc[
            output_inventory["global_type_id"].eq(global_row["global_type_id"])
        ]
        for animal in animals:
            present = members.loc[members["animal_id"].astype(str).eq(animal)]
            if present.empty:
                presence_rows.append(
                    {
                        "global_type_id": global_row["global_type_id"],
                        "global_type_label": global_row["global_type_label"],
                        "global_type_status": global_row["global_type_status"],
                        "animal_id": animal,
                        "local_prototype_id": pd.NA,
                        "local_event_type": pd.NA,
                        "local_event_type_label": pd.NA,
                        "event_count": 0,
                        "event_fraction": 0.0,
                        "local_data_status": "not_present",
                        "present": False,
                    }
                )
            else:
                row = present.iloc[0]
                presence_rows.append(
                    {
                        "global_type_id": global_row["global_type_id"],
                        "global_type_label": global_row["global_type_label"],
                        "global_type_status": global_row["global_type_status"],
                        "animal_id": animal,
                        "local_prototype_id": row["local_prototype_id"],
                        "local_event_type": row["local_event_type"],
                        "local_event_type_label": row["local_event_type_label"],
                        "event_count": int(row["event_count"]),
                        "event_fraction": float(row["event_fraction"]),
                        "local_data_status": row["event_count_status"],
                        "present": True,
                    }
                )
    return output_inventory, global_table, pd.DataFrame(presence_rows)


def _finite_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    numeric = numeric[np.isfinite(numeric)]
    return float(np.median(numeric)) if len(numeric) else float("nan")


def _stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in (base_seed, *parts)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % np.iinfo(np.uint32).max


__all__ = [
    "CrossMouseEquivalenceConfig",
    "CrossMouseEquivalenceResult",
    "DEFAULT_EQUIVALENCE_FEATURE_BLOCKS",
    "REQUIRED_FEATURE_BLOCKS",
    "match_event_types_by_equivalence",
    "save_cross_mouse_equivalence",
]
