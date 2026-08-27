from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score


FEATURE_BLOCKS: dict[str, tuple[str, ...]] = {
    "amplitude": (
        "log_amplitude_vs_baseline",
        "baseline_z__aggregate_line_length",
        "baseline_z__aggregate_log_rms",
    ),
    "shape": (
        "aggregate_hjorth_mobility",
        "aggregate_hjorth_complexity",
        "aggregate_crest_factor",
        "aggregate_excursion_rate_hz",
    ),
    "frequency": (
        "aggregate_peak_frequency_hz",
        "aggregate_median_frequency_hz",
        "aggregate_spectral_entropy",
        "aggregate_spectral_concentration",
        "relative_delta",
        "relative_theta",
        "relative_alpha",
        "relative_beta",
        "relative_low_gamma_clean",
        "theta_delta_ratio",
        "high_low_ratio",
    ),
    "bilateral": (
        "channel_correlation",
        "spectral_cosine_similarity",
        "abs_amplitude_log_ratio",
        "abs_line_length_log_ratio",
    ),
}


@dataclass(frozen=True)
class CrossMouseWaveformConfig:
    """Operational rules for matching existing per-mouse waveform categories.

    The main thresholds are empirical 95th percentiles of within-category
    split-half distances. They define repeatability on this dataset and are
    not biological or international seizure-type thresholds.
    """

    label_column: str = "individual_review_smoothed_label"
    minimum_events_for_matching: int = 20
    split_half_repeats: int = 400
    pair_bootstrap_repeats: int = 500
    threshold_quantile: float = 0.95
    confidence_level: float = 0.95
    random_seed: int = 20260812

    def __post_init__(self) -> None:
        if self.minimum_events_for_matching < 4:
            raise ValueError("minimum_events_for_matching must be at least four")
        if self.split_half_repeats < 50 or self.pair_bootstrap_repeats < 50:
            raise ValueError("bootstrap repeat counts must be at least 50")
        if not 0.5 < self.threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be between 0.5 and 1")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1")


@dataclass(frozen=True)
class CrossMouseWaveformResult:
    event_category_features: pd.DataFrame
    local_inventory: pd.DataFrame
    category_prototypes: pd.DataFrame
    local_scatter_coordinates: pd.DataFrame
    common_space_coordinates: pd.DataFrame
    threshold_calibration: pd.DataFrame
    pairwise_evidence: pd.DataFrame
    type_registry: pd.DataFrame
    presence: pd.DataFrame
    representative_dimensions: pd.DataFrame
    local_position_audit: pd.DataFrame
    within_mouse_pair_audit: pd.DataFrame


def analyze_cross_mouse_waveform_types(
    assignments: pd.DataFrame,
    model_summary: pd.DataFrame,
    local_models: Mapping[str, Mapping[str, Any]],
    config: CrossMouseWaveformConfig | None = None,
    feature_blocks: Mapping[str, Sequence[str]] | None = None,
) -> CrossMouseWaveformResult:
    """Compare the existing 21 within-seizure waveform categories across mice.

    Every ``animal x event x local category`` cell contributes one median
    feature vector. A long seizure can therefore contain several categories,
    but many overlapping windows from that seizure cannot dominate a local
    category prototype or its uncertainty estimate.
    """

    config = config or CrossMouseWaveformConfig()
    blocks = _validate_feature_blocks(feature_blocks or FEATURE_BLOCKS)
    prepared = _prepare_assignments(assignments, config.label_column, blocks)
    event_category = _event_category_features(prepared, blocks)
    inventory = _local_inventory(prepared, event_category, model_summary, config)
    prototypes = _category_prototypes(event_category, blocks)
    scaled_events, scaled_prototypes, scaling = _common_scaling(
        event_category, prototypes, blocks
    )
    calibration, thresholds = _calibrate_thresholds(
        scaled_events, blocks, config
    )
    evidence = _pairwise_evidence(
        scaled_events,
        scaled_prototypes,
        inventory,
        thresholds,
        blocks,
        prepared,
        config,
    )
    registry, presence, inventory = _build_registry(
        inventory, evidence, prepared
    )
    prototypes = prototypes.merge(
        inventory[
            [
                "local_type",
                "analysis_type_id",
                "analysis_type_label",
                "analysis_type_status",
            ]
        ],
        on="local_type",
        validate="one_to_one",
    )
    local_coordinates = _local_scatter_coordinates(prepared, local_models)
    common_coordinates = _common_space_coordinates(
        scaled_prototypes, blocks, inventory
    )
    dimensions = _representative_dimensions(
        scaled_events, scaled_prototypes, blocks, maximum=5
    )
    position_audit, within_mouse_pairs = _within_mouse_duplicate_audit(
        prepared, scaled_events, scaled_prototypes, blocks, thresholds, config
    )
    calibration = calibration.assign(
        main_threshold_quantile=config.threshold_quantile,
        threshold_basis="within_local_type_event_split_half",
        interpretation="operational_repeatability_not_biological_standard",
    )
    prototypes.attrs["feature_scaling"] = scaling
    return CrossMouseWaveformResult(
        event_category_features=event_category,
        local_inventory=inventory,
        category_prototypes=prototypes,
        local_scatter_coordinates=local_coordinates,
        common_space_coordinates=common_coordinates,
        threshold_calibration=calibration,
        pairwise_evidence=evidence,
        type_registry=registry,
        presence=presence,
        representative_dimensions=dimensions,
        local_position_audit=position_audit,
        within_mouse_pair_audit=within_mouse_pairs,
    )


def _validate_feature_blocks(
    blocks: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    required = set(FEATURE_BLOCKS)
    if not required.issubset(blocks):
        raise ValueError(f"feature blocks must include {sorted(required)}")
    result: dict[str, tuple[str, ...]] = {}
    for block in blocks:
        values = tuple(dict.fromkeys(str(value) for value in blocks[block]))
        if len(values) < 2:
            raise ValueError(f"feature block {block} needs at least two features")
        result[block] = values
    return result


def _prepare_assignments(
    assignments: pd.DataFrame,
    label_column: str,
    blocks: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    required = {
        "animal_id",
        "session_id",
        "event_id",
        label_column,
        "amplitude_vs_baseline",
        "amplitude_log_ratio",
        "line_length_log_ratio",
        "ch7_valid",
        "ch8_valid",
    }
    features = {feature for values in blocks.values() for feature in values}
    derived = {
        "log_amplitude_vs_baseline",
        "abs_amplitude_log_ratio",
        "abs_line_length_log_ratio",
    }
    missing = (required | (features - derived)) - set(assignments.columns)
    if missing:
        raise ValueError(f"waveform assignments are missing {sorted(missing)}")
    frame = assignments.copy()
    frame["animal_id"] = frame["animal_id"].astype(str)
    frame["session_id"] = frame["session_id"].astype(str)
    frame["event_id"] = frame["event_id"].astype(str)
    frame["local_type"] = frame[label_column].astype(str)
    if "event_position_fraction" in assignments:
        frame["event_position_fraction"] = pd.to_numeric(
            assignments["event_position_fraction"], errors="coerce"
        )
    amplitude = pd.to_numeric(frame["amplitude_vs_baseline"], errors="coerce")
    frame["log_amplitude_vs_baseline"] = np.log(
        np.maximum(amplitude, 1e-8)
    )
    frame["abs_amplitude_log_ratio"] = np.abs(
        pd.to_numeric(frame["amplitude_log_ratio"], errors="coerce")
    )
    frame["abs_line_length_log_ratio"] = np.abs(
        pd.to_numeric(frame["line_length_log_ratio"], errors="coerce")
    )
    for feature in features:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
        frame.loc[~np.isfinite(frame[feature]), feature] = np.nan
    # E77 CH7 is physiologically invalid.  Do not let finite values derived
    # from that channel leak into common PCA coordinates or matching.
    e77 = frame["animal_id"].eq("E77")
    for feature in blocks.get("bilateral", ()):
        frame.loc[e77, feature] = np.nan
    if frame[["animal_id", "event_id", "local_type"]].isna().any(axis=None):
        raise ValueError("animal, event, and local category identifiers cannot be missing")
    return frame


def _event_category_features(
    assignments: pd.DataFrame, blocks: Mapping[str, Sequence[str]]
) -> pd.DataFrame:
    features = [feature for values in blocks.values() for feature in values]
    keys = ["animal_id", "session_id", "event_id", "local_type"]
    grouped = assignments.groupby(keys, sort=True, observed=True)
    medians = grouped[features].median().reset_index()
    counts = grouped.size().rename("window_count").reset_index()
    return medians.merge(counts, on=keys, validate="one_to_one")


def _local_inventory(
    assignments: pd.DataFrame,
    event_category: pd.DataFrame,
    model_summary: pd.DataFrame,
    config: CrossMouseWaveformConfig,
) -> pd.DataFrame:
    windows = (
        assignments.groupby(["animal_id", "local_type"], sort=True)
        .size()
        .rename("window_count")
        .reset_index()
    )
    events = (
        event_category.groupby(["animal_id", "local_type"], sort=True)
        .agg(
            independent_events=("event_id", "nunique"),
            sessions=("session_id", "nunique"),
            median_windows_per_event=("window_count", "median"),
        )
        .reset_index()
    )
    total_events = assignments.groupby("animal_id")["event_id"].nunique()
    inventory = windows.merge(
        events, on=["animal_id", "local_type"], validate="one_to_one"
    )
    inventory["events_with_category_fraction"] = [
        count / total_events.loc[animal]
        for animal, count in zip(
            inventory["animal_id"], inventory["independent_events"]
        )
    ]
    model = model_summary.copy()
    model["animal_id"] = model["animal_id"].astype(str)
    inventory = inventory.merge(
        model[
            [
                "animal_id",
                "stable_type_count",
                "review_type_count",
                "review_bootstrap_median_ari",
                "review_stability_ok",
            ]
        ],
        on="animal_id",
        validate="many_to_one",
    )
    inventory["local_resolution_status"] = np.where(
        inventory["review_stability_ok"].astype(bool),
        "validated_stable",
        "exploratory_subdivision",
    )
    inventory["event_support_status"] = np.select(
        [
            inventory["independent_events"].ge(
                config.minimum_events_for_matching
            ),
            inventory["independent_events"].ge(10),
        ],
        ["adequate_events", "limited_events"],
        default="insufficient_events",
    )
    inventory["minimum_events_for_matching"] = (
        config.minimum_events_for_matching
    )
    inventory["_local_type_order"] = inventory["local_type"].map(
        lambda value: _natural_label_key(value)[1]
    )
    inventory = inventory.sort_values(
        ["animal_id", "_local_type_order", "local_type"], kind="stable"
    ).drop(columns="_local_type_order")
    return inventory.reset_index(drop=True)


def _category_prototypes(
    event_category: pd.DataFrame, blocks: Mapping[str, Sequence[str]]
) -> pd.DataFrame:
    features = [feature for values in blocks.values() for feature in values]
    prototypes = (
        event_category.groupby(["animal_id", "local_type"], sort=True)[features]
        .median()
        .reset_index()
    )
    counts = (
        event_category.groupby(["animal_id", "local_type"], sort=True)
        .agg(independent_events=("event_id", "nunique"))
        .reset_index()
    )
    return prototypes.merge(
        counts, on=["animal_id", "local_type"], validate="one_to_one"
    )


def _common_scaling(
    event_category: pd.DataFrame,
    prototypes: pd.DataFrame,
    blocks: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = [feature for values in blocks.values() for feature in values]
    scaled_events = event_category.copy()
    scaled_prototypes = prototypes.copy()
    rows: list[dict[str, Any]] = []
    for feature in features:
        values = pd.to_numeric(prototypes[feature], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        is_shared_shape_pc = feature.startswith("raw_shape_pc")
        if is_shared_shape_pc:
            # The shared MiniRocket branch was already centered and normalized
            # as one branch.  Per-PC robust scaling here would give noisy tail
            # PCs the same influence as leading waveform modes.
            center, scale = 0.0, 1.0
        else:
            center = float(np.median(finite)) if len(finite) else 0.0
            if len(finite):
                q25, q75 = np.quantile(finite, [0.25, 0.75])
                scale = float(q75 - q25)
            else:
                scale = 0.0
            if scale <= 1e-8:
                scale = float(np.std(finite, ddof=1)) if len(finite) >= 2 else 1.0
            scale = max(scale, 1e-8)
        scaled_events[feature] = (
            pd.to_numeric(scaled_events[feature], errors="coerce") - center
        ) / scale
        scaled_prototypes[feature] = (
            pd.to_numeric(scaled_prototypes[feature], errors="coerce") - center
        ) / scale
        rows.append(
            {
                "feature": feature,
                "center_across_local_prototypes": center,
                "iqr_scale_across_local_prototypes": scale,
            }
        )
    return scaled_events, scaled_prototypes, pd.DataFrame(rows)


def _calibrate_thresholds(
    scaled_events: pd.DataFrame,
    blocks: Mapping[str, Sequence[str]],
    config: CrossMouseWaveformConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(config.random_seed)
    distributions: dict[str, list[float]] = {block: [] for block in blocks}
    contributing_types: dict[str, set[str]] = {block: set() for block in blocks}
    eligible_types = 0
    for local_type, group in scaled_events.groupby("local_type", sort=True):
        if any(
            _block_valid_row_mask(group, features).sum()
            < config.minimum_events_for_matching
            for features in blocks.values()
            if not all(feature in FEATURE_BLOCKS["bilateral"] for feature in features)
        ):
            continue
        eligible_types += 1
        for _ in range(config.split_half_repeats):
            order = rng.permutation(len(group))
            half = len(order) // 2
            first = _prototype_vector(group.iloc[order[:half]], blocks)
            second = _prototype_vector(group.iloc[order[half : 2 * half]], blocks)
            for block, features in blocks.items():
                distance = _block_distance(first, second, features)
                if np.isfinite(distance):
                    distributions[block].append(distance)
                    contributing_types[block].add(str(local_type))
    if eligible_types < 2:
        raise ValueError("too few adequately supported local categories for calibration")
    rows: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    for block, values in distributions.items():
        array = np.asarray(values, dtype=float)
        if len(array) < 50:
            raise ValueError(f"insufficient split-half distances for {block}")
        for quantile in (0.90, 0.95, 0.975):
            value = float(np.quantile(array, quantile))
            rows.append(
                {
                    "feature_block": block,
                    "quantile": quantile,
                    "distance_threshold": value,
                    "split_half_distances": int(len(array)),
                    "eligible_local_types": int(len(contributing_types[block])),
                    "eligible_local_types_all_core_blocks": int(eligible_types),
                    "split_half_repeats_per_type": config.split_half_repeats,
                }
            )
            if np.isclose(quantile, config.threshold_quantile):
                thresholds[block] = value
    if set(thresholds) != set(blocks):
        raise RuntimeError("main threshold quantile must be one of the reported values")
    return pd.DataFrame(rows), thresholds


def _pairwise_evidence(
    scaled_events: pd.DataFrame,
    scaled_prototypes: pd.DataFrame,
    inventory: pd.DataFrame,
    thresholds: Mapping[str, float],
    blocks: Mapping[str, Sequence[str]],
    prepared_assignments: pd.DataFrame,
    config: CrossMouseWaveformConfig,
) -> pd.DataFrame:
    event_groups = {
        str(label): group.reset_index(drop=True)
        for label, group in scaled_events.groupby("local_type", sort=True)
    }
    prototype_map = scaled_prototypes.set_index("local_type")
    inventory_map = inventory.set_index("local_type")
    bilateral_valid = (
        prepared_assignments.groupby("animal_id")[["ch7_valid", "ch8_valid"]]
        .mean()
        .min(axis=1)
        .ge(0.50)
        .to_dict()
    )
    labels = sorted(event_groups, key=_natural_label_key)
    alpha = 1.0 - config.confidence_level
    rows: list[dict[str, Any]] = []
    for label_a, label_b in itertools.combinations(labels, 2):
        info_a = inventory_map.loc[label_a]
        info_b = inventory_map.loc[label_b]
        animal_a = str(info_a["animal_id"])
        animal_b = str(info_b["animal_id"])
        if animal_a == animal_b:
            continue
        group_a = event_groups[label_a]
        group_b = event_groups[label_b]
        point_a = prototype_map.loc[label_a]
        point_b = prototype_map.loc[label_b]
        required_blocks = [
            block
            for block in ("raw_shape", "amplitude", "shape", "frequency")
            if block in blocks
        ]
        bilateral_available = bool(
            bilateral_valid.get(animal_a, False)
            and bilateral_valid.get(animal_b, False)
        )
        if bilateral_available:
            required_blocks.append("bilateral")
        point_distances = {
            block: _block_distance(point_a, point_b, features)
            for block, features in blocks.items()
        }
        rng = np.random.default_rng(
            _stable_seed(config.random_seed, label_a, label_b)
        )
        bootstrap: dict[str, list[float]] = {block: [] for block in blocks}
        joint_passes = 0
        for _ in range(config.pair_bootstrap_repeats):
            sample_a = group_a.iloc[
                rng.integers(0, len(group_a), size=len(group_a))
            ]
            sample_b = group_b.iloc[
                rng.integers(0, len(group_b), size=len(group_b))
            ]
            vector_a = _prototype_vector(sample_a, blocks)
            vector_b = _prototype_vector(sample_b, blocks)
            replicate: dict[str, float] = {}
            for block, features in blocks.items():
                value = _block_distance(vector_a, vector_b, features)
                if np.isfinite(value):
                    bootstrap[block].append(value)
                replicate[block] = value
            if all(
                np.isfinite(replicate[block])
                and replicate[block] <= thresholds[block]
                for block in required_blocks
            ):
                joint_passes += 1
        ci_low: dict[str, float] = {}
        ci_high: dict[str, float] = {}
        for block in blocks:
            values = np.asarray(bootstrap[block], dtype=float)
            ci_low[block] = (
                float(np.quantile(values, alpha / 2.0)) if len(values) else np.nan
            )
            ci_high[block] = (
                float(np.quantile(values, 1.0 - alpha / 2.0))
                if len(values)
                else np.nan
            )
        block_event_counts_a = {
            block: int(_block_valid_row_mask(group_a, blocks[block]).sum())
            for block in required_blocks
        }
        block_event_counts_b = {
            block: int(_block_valid_row_mask(group_b, blocks[block]).sum())
            for block in required_blocks
        }
        event_support = bool(
            all(
                block_event_counts_a[block] >= config.minimum_events_for_matching
                and block_event_counts_b[block] >= config.minimum_events_for_matching
                for block in required_blocks
            )
        )
        point_pass = all(
            np.isfinite(point_distances[block])
            and point_distances[block] <= thresholds[block]
            for block in required_blocks
        )
        ci_pass = all(
            np.isfinite(ci_high[block])
            and ci_high[block] <= thresholds[block]
            for block in required_blocks
        )
        clearly_distinct = any(
            np.isfinite(ci_low[block]) and ci_low[block] > thresholds[block]
            for block in required_blocks
        )
        both_stable = bool(
            info_a["local_resolution_status"] == "validated_stable"
            and info_b["local_resolution_status"] == "validated_stable"
        )
        if point_pass and ci_pass and event_support and both_stable:
            decision = "confirmed_shared_candidate"
        elif point_pass and ci_pass and event_support:
            decision = "provisional_shared_candidate"
        elif point_pass and not event_support:
            decision = "compatible_low_event_support"
        elif point_pass:
            decision = "compatible_uncertain"
        elif clearly_distinct:
            decision = "distinct_evidence"
        else:
            decision = "not_merged_mixed_evidence"
        record: dict[str, Any] = {
            "local_type_a": label_a,
            "local_type_b": label_b,
            "animal_a": animal_a,
            "animal_b": animal_b,
            "events_a": int(len(group_a)),
            "events_b": int(len(group_b)),
            "local_resolution_a": info_a["local_resolution_status"],
            "local_resolution_b": info_b["local_resolution_status"],
            "bilateral_required": bilateral_available,
            "required_blocks": ";".join(required_blocks),
            "point_all_required_blocks_pass": point_pass,
            "ci_high_all_required_blocks_pass": ci_pass,
            "bootstrap_joint_pass_fraction": joint_passes
            / config.pair_bootstrap_repeats,
            "pair_decision": decision,
            # The formal registry merges only stable, event-supported pairs.
            # Provisional pairs remain visible in the evidence table and map,
            # but retain their animal-qualified local labels.
            "registry_link": decision == "confirmed_shared_candidate",
            "threshold_basis": "within_local_type_event_split_half_q95",
            "bootstrap_unit": "animal_event_local_category",
            "bootstrap_repeats": config.pair_bootstrap_repeats,
        }
        required_ratios = []
        for block in blocks:
            record[f"{block}_distance"] = point_distances[block]
            record[f"{block}_threshold"] = thresholds[block]
            record[f"{block}_ci_low"] = ci_low[block]
            record[f"{block}_ci_high"] = ci_high[block]
            if block in required_blocks and np.isfinite(point_distances[block]):
                required_ratios.append(point_distances[block] / thresholds[block])
            if block in required_blocks:
                record[f"{block}_events_a"] = block_event_counts_a[block]
                record[f"{block}_events_b"] = block_event_counts_b[block]
        record["maximum_required_distance_ratio"] = (
            max(required_ratios) if required_ratios else np.nan
        )
        rows.append(record)
    return pd.DataFrame(rows).sort_values(
        ["registry_link", "maximum_required_distance_ratio"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def _build_registry(
    inventory: pd.DataFrame,
    evidence: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = inventory["local_type"].astype(str).tolist()
    animal_by_label = inventory.set_index("local_type")["animal_id"].to_dict()
    links = {
        tuple(sorted((str(row.local_type_a), str(row.local_type_b)))): str(
            row.pair_decision
        )
        for row in evidence.itertuples(index=False)
        if bool(row.registry_link)
    }
    clusters: list[list[str]] = [[label] for label in labels]
    while True:
        candidates: list[tuple[int, tuple[str, ...], int, int]] = []
        for left, right in itertools.combinations(range(len(clusters)), 2):
            merged = clusters[left] + clusters[right]
            animals = [animal_by_label[label] for label in merged]
            if len(animals) != len(set(animals)):
                continue
            cross_pairs = [
                tuple(sorted((a, b)))
                for a in clusters[left]
                for b in clusters[right]
            ]
            if all(pair in links for pair in cross_pairs):
                priority = sum(
                    links[pair] == "provisional_shared_candidate"
                    for pair in cross_pairs
                )
                candidates.append((priority, tuple(sorted(merged)), left, right))
        if not candidates:
            break
        _, members, left, right = min(candidates)
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {left, right}
        ] + [list(members)]
    clusters = sorted(
        clusters,
        key=lambda members: (-len(members), tuple(_natural_label_key(x) for x in members)),
    )
    inventory_map = inventory.set_index("local_type")
    registry_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    for index, members in enumerate(clusters, start=1):
        member_rows = inventory_map.loc[members]
        mouse_ids = sorted(member_rows["animal_id"].astype(str).unique())
        if len(members) > 1:
            pair_statuses = [
                links[tuple(sorted(pair))]
                for pair in itertools.combinations(members, 2)
            ]
            if all(value == "confirmed_shared_candidate" for value in pair_statuses):
                status = "confirmed_shared_candidate"
                prefix = "S"
            else:
                status = "provisional_shared_candidate"
                prefix = "P"
            label = f"{prefix}{index:02d}"
        else:
            row = member_rows.iloc[0]
            status = (
                "animal_specific_insufficient_evidence"
                if row["event_support_status"] != "adequate_events"
                else "animal_specific_no_supported_match"
            )
            label = str(members[0])
        registry_rows.append(
            {
                "analysis_type_id": f"WT{index:02d}",
                "analysis_type_label": label,
                "analysis_type_status": status,
                "mouse_count": len(mouse_ids),
                "mouse_coverage": f"{len(mouse_ids)}/{inventory['animal_id'].nunique()}",
                "mouse_ids": ";".join(mouse_ids),
                "local_type_count": len(members),
                "local_types": ";".join(members),
                "independent_event_category_cells": int(
                    member_rows["independent_events"].sum()
                ),
                "merge_rule": "complete_link_all_cross_mouse_pairs_pass",
            }
        )
        for member in members:
            membership_rows.append(
                {
                    "local_type": member,
                    "analysis_type_id": f"WT{index:02d}",
                    "analysis_type_label": label,
                    "analysis_type_status": status,
                }
            )
    registry = pd.DataFrame(registry_rows)
    output_inventory = inventory.merge(
        pd.DataFrame(membership_rows), on="local_type", validate="one_to_one"
    )
    animals = sorted(output_inventory["animal_id"].astype(str).unique())
    total_events = assignments.groupby("animal_id")["event_id"].nunique().to_dict()
    presence_rows: list[dict[str, Any]] = []
    for type_row in registry.to_dict("records"):
        members = output_inventory.loc[
            output_inventory["analysis_type_id"].eq(type_row["analysis_type_id"])
        ]
        for animal in animals:
            observed = members.loc[members["animal_id"].eq(animal)]
            if observed.empty:
                presence_rows.append(
                    {
                        "analysis_type_id": type_row["analysis_type_id"],
                        "analysis_type_label": type_row["analysis_type_label"],
                        "analysis_type_status": type_row["analysis_type_status"],
                        "mouse_count": type_row["mouse_count"],
                        "animal_id": animal,
                        "local_type": pd.NA,
                        "independent_events": 0,
                        "events_with_category_fraction": 0.0,
                        "present": False,
                    }
                )
            else:
                row = observed.iloc[0]
                presence_rows.append(
                    {
                        "analysis_type_id": type_row["analysis_type_id"],
                        "analysis_type_label": type_row["analysis_type_label"],
                        "analysis_type_status": type_row["analysis_type_status"],
                        "mouse_count": type_row["mouse_count"],
                        "animal_id": animal,
                        "local_type": row["local_type"],
                        "independent_events": int(row["independent_events"]),
                        "events_with_category_fraction": float(
                            row["independent_events"] / total_events[animal]
                        ),
                        "present": True,
                    }
                )
    return registry, pd.DataFrame(presence_rows), output_inventory


def _local_scatter_coordinates(
    assignments: pd.DataFrame,
    local_models: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for animal, group in assignments.groupby("animal_id", sort=True):
        if animal not in local_models:
            raise ValueError(f"missing saved local model for {animal}")
        model = local_models[animal]
        features = list(model["feature_columns"])
        event_group = (
            group.groupby(
                ["animal_id", "session_id", "event_id", "local_type"],
                sort=True,
                observed=True,
            )[features]
            .median()
            .reset_index()
        )
        window_counts = (
            group.groupby(
                ["animal_id", "session_id", "event_id", "local_type"],
                sort=True,
                observed=True,
            )
            .size()
            .rename("window_count")
            .reset_index()
        )
        event_group = event_group.merge(
            window_counts,
            on=["animal_id", "session_id", "event_id", "local_type"],
            validate="one_to_one",
        )
        raw = event_group[features].replace([np.inf, -np.inf], np.nan)
        medians = pd.Series(model["feature_medians"])
        values = raw.fillna(medians).to_numpy(float)
        embedded = model["pca"].transform(model["scaler"].transform(values))
        frame = event_group[
            ["animal_id", "session_id", "event_id", "local_type", "window_count"]
        ].copy()
        frame["local_pc1"] = embedded[:, 0]
        frame["local_pc2"] = embedded[:, 1] if embedded.shape[1] > 1 else 0.0
        frame["coordinate_scope"] = "within_mouse_only_not_cross_mouse_comparable"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _common_space_coordinates(
    scaled_prototypes: pd.DataFrame,
    blocks: Mapping[str, Sequence[str]],
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    branches: list[np.ndarray] = []
    for block, features in blocks.items():
        values = scaled_prototypes[list(features)].to_numpy(float)
        medians = np.nanmedian(values, axis=0)
        medians = np.nan_to_num(medians, nan=0.0)
        missing = ~np.isfinite(values)
        values[missing] = np.take(medians, np.where(missing)[1])
        # Each evidence block has equal total weight; leading variance within
        # the shared raw-waveform branch remains preserved.
        if block == "raw_shape":
            energy = float(np.sqrt(np.mean(np.sum(values**2, axis=1))))
            branches.append(values / max(energy, 1e-12))
        else:
            branches.append(values / np.sqrt(len(features)))
    matrix = np.concatenate(branches, axis=1)
    pca = PCA(n_components=2, random_state=20260812).fit(matrix)
    coordinates = pca.transform(matrix)
    result = scaled_prototypes[["animal_id", "local_type", "independent_events"]].copy()
    result["common_pc1"] = coordinates[:, 0]
    result["common_pc2"] = coordinates[:, 1]
    result["common_pc1_variance_fraction"] = float(pca.explained_variance_ratio_[0])
    result["common_pc2_variance_fraction"] = float(pca.explained_variance_ratio_[1])
    result["coordinate_scope"] = "common_event_balanced_raw_shape_plus_physiology_space"
    return result.merge(
        inventory[
            [
                "local_type",
                "local_resolution_status",
                "event_support_status",
                "analysis_type_id",
                "analysis_type_label",
                "analysis_type_status",
            ]
        ],
        on="local_type",
        validate="one_to_one",
    )


def _representative_dimensions(
    scaled_events: pd.DataFrame,
    scaled_prototypes: pd.DataFrame,
    blocks: Mapping[str, Sequence[str]],
    maximum: int,
) -> pd.DataFrame:
    candidates: list[dict[str, Any]] = []
    for block, features in blocks.items():
        if block == "raw_shape":
            # MiniRocket PCs are independent shape evidence, but are not
            # clinically named physiological dimensions.
            continue
        for feature in features:
            prototype_values = scaled_prototypes[feature].to_numpy(float)
            between = _robust_scale(prototype_values)
            within_values = []
            for _, group in scaled_events.groupby("local_type", sort=True):
                within_values.append(_robust_scale(group[feature].to_numpy(float)))
            finite_within = [value for value in within_values if np.isfinite(value)]
            within = float(np.median(finite_within)) if finite_within else np.nan
            score = between / max(within, 1e-8) if np.isfinite(within) else 0.0
            candidates.append(
                {
                    "feature": feature,
                    "feature_block": block,
                    "between_local_type_robust_scale": between,
                    "median_within_local_type_robust_scale": within,
                    "separation_ratio": score,
                    "finite_local_type_fraction": float(
                        np.isfinite(prototype_values).mean()
                    ),
                }
            )
    ranking = pd.DataFrame(candidates).sort_values(
        ["separation_ratio", "feature"], ascending=[False, True], kind="stable"
    )
    prototype_matrix = scaled_prototypes.set_index("local_type")
    selected: list[dict[str, Any]] = []
    selected_features: list[str] = []

    def add_candidate(row: pd.Series) -> bool:
        feature = str(row["feature"])
        values = prototype_matrix[feature]
        maximum_correlation = 0.0
        for existing in selected_features:
            pair = pd.concat([values, prototype_matrix[existing]], axis=1).dropna()
            if len(pair) >= 5:
                correlation = abs(float(pair.corr(method="spearman").iloc[0, 1]))
                maximum_correlation = max(maximum_correlation, correlation)
        if selected_features and maximum_correlation >= 0.80:
            return False
        record = row.to_dict()
        record["maximum_abs_spearman_with_selected"] = maximum_correlation
        selected.append(record)
        selected_features.append(feature)
        return True

    for block in blocks:
        rows = ranking.loc[
            ranking["feature_block"].eq(block)
            & ranking["finite_local_type_fraction"].ge(0.70)
        ]
        for _, row in rows.iterrows():
            if add_candidate(row):
                break
    for _, row in ranking.iterrows():
        if len(selected) >= maximum:
            break
        if row["feature"] in selected_features:
            continue
        add_candidate(row)
    result = pd.DataFrame(selected[:maximum])
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


def _within_mouse_duplicate_audit(
    assignments: pd.DataFrame,
    scaled_events: pd.DataFrame,
    scaled_prototypes: pd.DataFrame,
    blocks: Mapping[str, Sequence[str]],
    thresholds: Mapping[str, float],
    config: CrossMouseWaveformConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    position_rows: list[dict[str, Any]] = []
    if "event_position_fraction" not in assignments:
        assignments = assignments.copy()
        assignments["event_position_fraction"] = np.nan
    event_positions = (
        assignments.groupby(
            ["animal_id", "event_id", "local_type"],
            sort=True,
            observed=True,
        )["event_position_fraction"]
        .median()
        .reset_index()
    )
    animal_nmi = {}
    animal_ami = {}
    for animal, group in event_positions.groupby("animal_id", sort=True):
        finite = group.loc[np.isfinite(group["event_position_fraction"])].copy()
        deciles = np.minimum(9, np.floor(finite["event_position_fraction"] * 10)).astype(int)
        animal_nmi[str(animal)] = (
            float(normalized_mutual_info_score(finite["local_type"], deciles))
            if len(finite)
            else np.nan
        )
        animal_ami[str(animal)] = (
            float(adjusted_mutual_info_score(finite["local_type"], deciles))
            if len(finite)
            else np.nan
        )
    for (animal, local_type), group in event_positions.groupby(
        ["animal_id", "local_type"], sort=True
    ):
        values = pd.to_numeric(group["event_position_fraction"], errors="coerce")
        finite = values[np.isfinite(values)]
        q25, median, q75 = (
            np.quantile(finite, [0.25, 0.50, 0.75])
            if len(finite)
            else (np.nan, np.nan, np.nan)
        )
        position_rows.append(
            {
                "animal_id": animal,
                "local_type": local_type,
                "event_category_units": len(group),
                "independent_events": group["event_id"].nunique(),
                "event_position_q25": q25,
                "event_position_median": median,
                "event_position_q75": q75,
                "near_onset_fraction": float((finite < 0.20).mean())
                if len(finite)
                else np.nan,
                "near_offset_fraction": float((finite > 0.80).mean())
                if len(finite)
                else np.nan,
                "animal_position_decile_nmi_event_balanced": animal_nmi[str(animal)],
                "animal_position_decile_ami_event_balanced": animal_ami[str(animal)],
                "position_summary_unit": "event_x_local_category_median",
            }
        )
    position = pd.DataFrame(position_rows)
    prototype_map = scaled_prototypes.set_index("local_type")
    position_map = position.set_index("local_type")
    pair_rows: list[dict[str, Any]] = []
    for animal, animal_types in scaled_prototypes.groupby("animal_id", sort=True):
        labels = sorted(animal_types["local_type"].astype(str), key=_natural_label_key)
        for label_a, label_b in itertools.combinations(labels, 2):
            first = prototype_map.loc[label_a]
            second = prototype_map.loc[label_b]
            distances = {
                block: _block_distance(first, second, features)
                for block, features in blocks.items()
            }
            support = {
                label: int(
                    _block_valid_row_mask(
                        scaled_events.loc[scaled_events["local_type"].eq(label)],
                        blocks.get("raw_shape", blocks["shape"]),
                    ).sum()
                )
                for label in (label_a, label_b)
            }
            group_a = scaled_events.loc[
                scaled_events["local_type"].eq(label_a)
            ].reset_index(drop=True)
            group_b = scaled_events.loc[
                scaled_events["local_type"].eq(label_b)
            ].reset_index(drop=True)
            rng = np.random.default_rng(
                _stable_seed(config.random_seed, "within", label_a, label_b)
            )
            bootstrap = {block: [] for block in blocks}
            for _ in range(config.pair_bootstrap_repeats):
                sample_a = group_a.iloc[
                    rng.integers(0, len(group_a), size=len(group_a))
                ]
                sample_b = group_b.iloc[
                    rng.integers(0, len(group_b), size=len(group_b))
                ]
                vector_a = _prototype_vector(sample_a, blocks)
                vector_b = _prototype_vector(sample_b, blocks)
                for block, features in blocks.items():
                    value = _block_distance(vector_a, vector_b, features)
                    if np.isfinite(value):
                        bootstrap[block].append(value)
            alpha = 1.0 - config.confidence_level
            ci_low = {
                block: (
                    float(np.quantile(values, alpha / 2.0))
                    if values
                    else np.nan
                )
                for block, values in bootstrap.items()
            }
            ci_high = {
                block: (
                    float(np.quantile(values, 1.0 - alpha / 2.0))
                    if values
                    else np.nan
                )
                for block, values in bootstrap.items()
            }
            position_gap = abs(
                float(position_map.loc[label_a, "event_position_median"])
                - float(position_map.loc[label_b, "event_position_median"])
            )
            q25_overlap = max(
                float(position_map.loc[label_a, "event_position_q25"]),
                float(position_map.loc[label_b, "event_position_q25"]),
            )
            q75_overlap = min(
                float(position_map.loc[label_a, "event_position_q75"]),
                float(position_map.loc[label_b, "event_position_q75"]),
            )
            raw_shape_distance = distances.get("raw_shape", np.nan)
            raw_shape_threshold = thresholds.get("raw_shape", np.nan)
            shape_same = bool(
                np.isfinite(ci_high.get("raw_shape", np.nan))
                and np.isfinite(raw_shape_threshold)
                and ci_high["raw_shape"] <= raw_shape_threshold
            )
            physiology_blocks = [
                block for block in ("amplitude", "frequency", "bilateral")
                if block in distances and np.isfinite(distances[block])
            ]
            physiology_distinct = any(
                np.isfinite(ci_low[block]) and ci_low[block] > thresholds[block]
                for block in physiology_blocks
            )
            physiology_same = bool(
                physiology_blocks
                and all(
                    np.isfinite(ci_high[block])
                    and ci_high[block] <= thresholds[block]
                    for block in physiology_blocks
                )
            )
            low_support = min(support.values()) < 10
            boundary_transient = bool(
                position_gap >= 0.20
                and (
                    float(position_map.loc[label_a, "near_onset_fraction"]) >= 0.50
                    or float(position_map.loc[label_b, "near_onset_fraction"]) >= 0.50
                    or float(position_map.loc[label_a, "near_offset_fraction"]) >= 0.50
                    or float(position_map.loc[label_b, "near_offset_fraction"]) >= 0.50
                )
            )
            if (
                np.isfinite(ci_low.get("raw_shape", np.nan))
                and ci_low["raw_shape"] > raw_shape_threshold
            ):
                duplicate_status = "shape_distinct"
            elif shape_same and physiology_distinct:
                duplicate_status = "same_shape_family_physiology_distinct"
            elif shape_same and low_support and boundary_transient:
                duplicate_status = "low_sample_boundary_transient"
            elif shape_same and physiology_same and not low_support:
                duplicate_status = "possible_duplicate_subdivision"
            else:
                duplicate_status = "insufficient_or_mixed_evidence"
            pair_rows.append(
                {
                    "animal_id": animal,
                    "local_type_a": label_a,
                    "local_type_b": label_b,
                    "events_a": int(
                        scaled_events.loc[
                            scaled_events["local_type"].eq(label_a), "event_id"
                        ].nunique()
                    ),
                    "events_b": int(
                        scaled_events.loc[
                            scaled_events["local_type"].eq(label_b), "event_id"
                        ].nunique()
                    ),
                    **{f"{block}_distance": value for block, value in distances.items()},
                    **{f"{block}_threshold": thresholds[block] for block in distances},
                    **{f"{block}_ci_low": ci_low[block] for block in distances},
                    **{f"{block}_ci_high": ci_high[block] for block in distances},
                    "event_position_median_gap": position_gap,
                    "event_position_iqr_overlap": max(0.0, q75_overlap - q25_overlap),
                    "raw_shape_events_a": support[label_a],
                    "raw_shape_events_b": support[label_b],
                    "within_mouse_relation": duplicate_status,
                    "position_is_interpretation_not_clustering_input": True,
                    "bootstrap_unit": "independent_event_x_local_category",
                    "bootstrap_repeats": config.pair_bootstrap_repeats,
                }
            )
    return position, pd.DataFrame(pair_rows).sort_values(
        ["animal_id", "shape_distance"], kind="stable"
    ).reset_index(drop=True)


def _prototype_vector(
    rows: pd.DataFrame, blocks: Mapping[str, Sequence[str]]
) -> pd.Series:
    features = [feature for values in blocks.values() for feature in values]
    return rows[features].median(axis=0)


def _block_distance(
    first: pd.Series, second: pd.Series, features: Sequence[str]
) -> float:
    a = pd.to_numeric(first[list(features)], errors="coerce").to_numpy(float)
    b = pd.to_numeric(second[list(features)], errors="coerce").to_numpy(float)
    finite = np.isfinite(a) & np.isfinite(b)
    minimum = max(2, int(np.ceil(len(features) / 2.0)))
    if np.count_nonzero(finite) < minimum:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(a[finite] - b[finite]))))


def _block_valid_row_mask(rows: pd.DataFrame, features: Sequence[str]) -> pd.Series:
    values = rows.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(values.to_numpy(float))
    minimum = max(2, int(np.ceil(len(features) / 2.0)))
    return pd.Series(finite.sum(axis=1) >= minimum, index=rows.index)


def _robust_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return float("nan")
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    iqr = float(q75 - q25) / 1.349
    if iqr > 1e-8:
        return iqr
    return float(np.std(finite, ddof=1))


def _stable_seed(seed: int, *parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in (seed, *parts)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % np.iinfo(np.uint32).max


def _natural_label_key(value: Any) -> tuple[str, int, str]:
    text = str(value)
    animal, _, suffix = text.partition("-C")
    try:
        number = int(suffix)
    except ValueError:
        number = 999999
    return animal, number, text


__all__ = [
    "CrossMouseWaveformConfig",
    "CrossMouseWaveformResult",
    "FEATURE_BLOCKS",
    "analyze_cross_mouse_waveform_types",
]
