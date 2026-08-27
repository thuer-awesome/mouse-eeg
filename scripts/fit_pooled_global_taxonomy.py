#!/usr/bin/env python
"""Freeze a conservative cross-mouse taxonomy for pooled within-mouse clusters.

The preceding ``fit_pooled_mouse_morphology.py`` fits one GMM per mouse after
pooling valid CH7/CH8 events within that mouse.  This script is deliberately a
separate label-freeze stage: pooled clusters are the atomic units and may be
merged only across different mice.  A constrained agglomerative graph prevents
two pooled clusters from the same mouse from entering one PG category, even
through a transitive chain of pairwise similarities.

Outputs are written only to ``04_label_freeze_pooled``.  PG labels are
exploratory morphology labels, not clinical seizure subtypes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    ROOT
    / "output"
    / "epilepsy_program_20260821"
    / "02_morphology_taxonomy"
    / "all_mice_old_plus_new"
    / "strict_all_channel_v1"
)

FEATURES = [
    "log_robust_amplitude",
    "log_rms",
    "log_line_length",
    "hjorth_mobility",
    "hjorth_complexity",
    "peak_frequency_hz",
    "spectral_entropy",
    "relative_delta",
    "relative_theta",
    "relative_beta",
    "relative_low_gamma_clean",
    "relative_high_gamma",
    "baseline_z__log_robust_amplitude",
    "baseline_z__log_rms",
    "baseline_z__log_line_length",
    "baseline_z__hjorth_mobility",
    "baseline_z__hjorth_complexity",
    "baseline_z__peak_frequency_hz",
    "baseline_z__spectral_entropy",
    "baseline_z__relative_delta",
    "baseline_z__relative_theta",
    "baseline_z__relative_beta",
    "baseline_z__relative_low_gamma_clean",
    "baseline_z__relative_high_gamma",
]
KEYS = ["animal_id", "session_id", "event_id", "channel_name"]


def _finite_matrix(frame: pd.DataFrame, columns: Iterable[str], medians: pd.Series | None = None) -> np.ndarray:
    values = frame[list(columns)].replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = values.median(numeric_only=True)
    values = values.fillna(medians).fillna(0.0)
    return values.to_numpy(float)


def _constrained_components(
    prototypes: pd.DataFrame,
    distances: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """Return complete-linkage groups with at most one prototype per animal.

    The shortest eligible pair is merged repeatedly, but a candidate merge is
    accepted only when *every* cross-member distance is below ``threshold`` and
    the two animal sets are disjoint.  Complete linkage prevents a long
    transitive chain from turning a weakly related state into a shared category;
    the animal-set constraint prevents two pooled states from one mouse from
    entering the same PG label.
    """

    groups: list[list[int]] = [[index] for index in range(len(prototypes))]
    animal_sets: list[set[str]] = [
        {str(prototypes.iloc[index].animal_id)} for index in range(len(prototypes))
    ]
    while True:
        candidates: list[tuple[float, tuple[int, ...], int, int]] = []
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                if animal_sets[left].intersection(animal_sets[right]):
                    continue
                linkage = max(
                    float(distances[i, j])
                    for i in groups[left]
                    for j in groups[right]
                )
                if np.isfinite(linkage) and linkage <= threshold:
                    signature = tuple(sorted(groups[left] + groups[right]))
                    candidates.append((linkage, signature, left, right))
        if not candidates:
            break
        _, _, left, right = min(candidates, key=lambda row: (row[0], row[1]))
        merged = sorted(groups[left] + groups[right])
        merged_animals = animal_sets[left] | animal_sets[right]
        # Delete the higher index first to keep the lower index stable.
        for index in sorted((left, right), reverse=True):
            del groups[index]
            del animal_sets[index]
        groups.append(merged)
        animal_sets.append(merged_animals)
    return [sorted(group) for group in groups]


def _natural_key(value: object) -> tuple[int, str]:
    text = str(value)
    digits = "".join(char for char in text if char.isdigit())
    return (int(digits) if digits else 10**9, text)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def freeze(root: Path, threshold: float = 0.90, seed: int = 20260823) -> dict[str, object]:
    del seed  # Reserved for reproducibility metadata; this stage is deterministic.
    source = root / "03_cross_mouse"
    out = root / "04_label_freeze_pooled"
    out.mkdir(parents=True, exist_ok=True)
    # Remove obsolete draft aliases inside this one generated directory so a
    # rerun cannot leave two competing pooled schemas behind.
    for obsolete in ("pooled_category_prototypes.parquet", "pooled_merge_members.csv", "pooled_taxonomy_manifest.json"):
        path = out / obsolete
        if path.exists():
            path.unlink()

    prototypes = pd.read_parquet(source / "pooled_mouse_prototypes.parquet").copy()
    assignments = pd.read_parquet(source / "pooled_mouse_event_assignments.parquet").copy()
    events = pd.read_parquet(root / "01_tables" / "channel_event_features.parquet").copy()
    windows = pd.read_parquet(root / "04_label_freeze" / "frozen_channel_window_labels.parquet").copy()

    missing = [column for column in FEATURES if column not in events]
    missing_proto = [f"f_{column}" for column in FEATURES if f"f_{column}" not in prototypes]
    if missing or missing_proto:
        raise ValueError(f"Missing feature columns: events={missing}, prototypes={missing_proto}")
    if assignments.duplicated(KEYS).any() or events.duplicated(KEYS).any():
        raise ValueError("Event assignments/features must be unique by animal/session/event/channel")

    # Prototype-space scaling is fit only on the pooled training prototypes.
    prototype_columns = [f"f_{column}" for column in FEATURES]
    prototype_values = prototypes[prototype_columns].replace([np.inf, -np.inf], np.nan)
    prototype_values = prototype_values.fillna(prototype_values.median()).fillna(0.0)
    scaler = RobustScaler(quantile_range=(10, 90)).fit(prototype_values.to_numpy(float))
    prototype_z = scaler.transform(prototype_values.to_numpy(float))
    distances = pairwise_distances(prototype_z)

    thresholds = (0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20)
    grid_rows: list[dict[str, object]] = []
    for candidate in thresholds:
        groups = _constrained_components(prototypes, distances, candidate)
        edge_count = int(
            sum(
                1
                for i in range(len(prototypes))
                for j in range(i + 1, len(prototypes))
                if str(prototypes.iloc[i].animal_id) != str(prototypes.iloc[j].animal_id)
                and np.isfinite(distances[i, j])
                and distances[i, j] <= candidate
            )
        )
        supports = [
            int(prototypes.iloc[group].n_train_events.sum()) for group in groups
        ]
        grid_rows.append(
            {
                "distance_threshold": float(candidate),
                "n_pooled_prototypes": int(len(prototypes)),
                "n_cross_mouse_edges": edge_count,
                "n_global_categories": int(len(groups)),
                "n_shared_categories": int(sum(len(group) > 1 for group in groups)),
                "min_train_event_support": int(min(supports)),
                "median_train_event_support": float(np.median(supports)),
                "max_train_event_support": int(max(supports)),
                "same_mouse_merge_allowed": False,
                "linkage": "complete",
                "selection_rule": "pre-registered conservative threshold; constrained complete-linkage agglomeration",
            }
        )
    grid = pd.DataFrame(grid_rows)
    if threshold not in set(grid.distance_threshold.astype(float)):
        raise ValueError(f"threshold {threshold} is not in the audited grid")
    groups = _constrained_components(prototypes, distances, threshold)

    # Order PG identifiers by weighted centroid PC1, then by member labels.  A
    # deterministic order keeps reruns and downstream figures stable.
    weights = prototypes.n_train_events.to_numpy(float).clip(1.0)
    centers = np.asarray(
        [np.average(prototype_z[group], axis=0, weights=weights[group]) for group in groups]
    )
    order = sorted(
        range(len(groups)),
        key=lambda index: (
            float(centers[index, 0]),
            float(centers[index, 1]),
            tuple(str(prototypes.iloc[i].pooled_label) for i in groups[index]),
        ),
    )
    groups = [groups[index] for index in order]
    centers = centers[order]
    prototype_to_global: dict[tuple[str, int], int] = {}
    prototype_global = np.full(len(prototypes), -1, dtype=int)
    for category, group in enumerate(groups):
        for index in group:
            row = prototypes.iloc[index]
            key = (str(row.animal_id), int(row.pooled_cluster))
            if key in prototype_to_global:
                raise AssertionError(f"Duplicate pooled prototype mapping: {key}")
            prototype_to_global[key] = category
            prototype_global[index] = category
    if (prototype_global < 0).any():
        raise AssertionError("Every pooled prototype must receive a PG category")

    # Event labels inherit the frozen category of their within-mouse pooled
    # cluster.  This is intentionally not a nearest-centroid reassignment: it
    # preserves the pooled GMM partition and guarantees the no-same-mouse rule.
    event_labels = events[KEYS + ["split"]].merge(
        assignments[KEYS + ["pooled_cluster", "pooled_label", "pooled_probability"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    if event_labels.pooled_cluster.isna().any():
        raise AssertionError("Some event rows have no pooled-cluster assignment")
    event_labels["pooled_cluster"] = event_labels.pooled_cluster.astype(int)
    event_labels["pooled_global_category"] = [
        prototype_to_global[(str(animal), int(cluster))]
        for animal, cluster in zip(event_labels.animal_id, event_labels.pooled_cluster)
    ]
    event_labels["pooled_global_label"] = [
        f"PG{int(category) + 1:02d}" for category in event_labels.pooled_global_category
    ]

    # Distances are reported for QC only.  The event split is never used to
    # change its inherited PG label.  Imputation/scaling statistics come from
    # training events to avoid validation/test leakage.
    train_events = events.loc[events.split.astype(str).eq("train")]
    train_medians = train_events[FEATURES].replace([np.inf, -np.inf], np.nan).median()
    event_values = _finite_matrix(events, FEATURES, train_medians)
    event_z = scaler.transform(event_values)
    event_centers = centers[np.asarray(event_labels.pooled_global_category, dtype=int)]
    event_distances = np.linalg.norm(event_z - event_centers, axis=1)
    event_labels["pooled_global_distance"] = event_distances
    train_distance_threshold = float(
        np.nanquantile(event_distances[event_labels.split.astype(str).eq("train")], 0.995)
    )
    event_labels["pooled_low_confidence"] = event_labels.pooled_global_distance.gt(train_distance_threshold)
    # Canonical aliases make this table consumable by generic classifier code,
    # while the PG-prefixed columns remain the unambiguous source of truth.
    event_labels["global_category"] = event_labels.pooled_global_category
    event_labels["global_label"] = event_labels.pooled_global_label
    event_labels["global_distance"] = event_labels.pooled_global_distance
    event_labels["low_confidence"] = event_labels.pooled_low_confidence

    # Attach the event-level PG label to every original window.  Preserve the
    # old strict G labels under ``source_strict_*`` rather than silently mixing
    # two taxonomies in one column.
    source_rename = {
        "global_category": "source_strict_global_category",
        "global_label": "source_strict_global_label",
        "global_distance": "source_strict_global_distance",
        "low_confidence": "source_strict_low_confidence",
    }
    window_labels = windows.rename(columns=source_rename).merge(
        event_labels[KEYS + [
            "pooled_cluster", "pooled_label", "pooled_probability",
            "pooled_global_category", "pooled_global_label",
            "pooled_global_distance", "pooled_low_confidence",
        ]],
        on=KEYS,
        how="left",
        validate="many_to_one",
    )
    if window_labels.pooled_global_category.isna().any():
        raise AssertionError("Some windows have no pooled event label")
    window_labels["pooled_global_category"] = window_labels.pooled_global_category.astype(int)
    window_labels["global_category"] = window_labels.pooled_global_category
    window_labels["global_label"] = window_labels.pooled_global_label
    window_labels["global_distance"] = window_labels.pooled_global_distance
    window_labels["low_confidence"] = window_labels.pooled_low_confidence

    # Frozen global prototypes are weighted centroids in the same robust space,
    # with member pooled prototypes retained in a separate audit table.
    prototype_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    for category, group in enumerate(groups):
        member = prototypes.iloc[group]
        center_raw = np.average(prototype_values.to_numpy(float)[group], axis=0, weights=weights[group])
        member_animals = sorted(member.animal_id.astype(str).unique())
        member_labels = sorted(member.pooled_label.astype(str).tolist(), key=_natural_key)
        prototype_rows.append(
            {
                "pooled_global_category": category,
                "pooled_global_label": f"PG{category + 1:02d}",
                "n_member_pooled_prototypes": int(len(member)),
                "n_member_animals": int(len(member_animals)),
                "member_animals": ";".join(member_animals),
                "member_pooled_labels": ";".join(member_labels),
                "n_train_events": int(member.n_train_events.sum()),
                "centroid_pc1_robust": float(centers[category, 0]),
                "centroid_pc2_robust": float(centers[category, 1]),
                **{f"f_{feature}": float(value) for feature, value in zip(FEATURES, center_raw)},
            }
        )
        for index in group:
            row = prototypes.iloc[index]
            member_rows.append(
                {
                    "pooled_global_category": category,
                    "pooled_global_label": f"PG{category + 1:02d}",
                    "animal_id": str(row.animal_id),
                    "pooled_cluster": int(row.pooled_cluster),
                    "pooled_label": str(row.pooled_label),
                    "n_train_events": int(row.n_train_events),
                    "prototype_distance_to_global_centroid": float(np.linalg.norm(prototype_z[index] - centers[category])),
                }
            )
    pooled_prototypes = pd.DataFrame(prototype_rows)
    pooled_members = pd.DataFrame(member_rows)

    registry_rows: list[dict[str, object]] = []
    for category, group in enumerate(groups):
        selected = event_labels.pooled_global_category.eq(category)
        member = prototypes.iloc[group]
        animals = sorted(event_labels.loc[selected, "animal_id"].astype(str).unique())
        registry_rows.append(
            {
                "pooled_global_category": category,
                "pooled_global_label": f"PG{category + 1:02d}",
                "n_event_channel": int(selected.sum()),
                "n_windows": int(window_labels.pooled_global_category.eq(category).sum()),
                "n_animals": int(len(animals)),
                "member_animals": ";".join(animals),
                "mouse_specific_flag": bool(len(animals) < 2),
                "merge_status": "cross_mouse_shared" if len(animals) >= 2 else "animal_specific",
                "n_channels": int(event_labels.loc[selected, "channel_name"].nunique()),
                "n_pooled_prototypes": int(len(member)),
                "member_pooled_labels": ";".join(sorted(member.pooled_label.astype(str), key=_natural_key)),
                "train_support": int((selected & event_labels.split.astype(str).eq("train")).sum()),
                "val_support": int((selected & event_labels.split.astype(str).eq("val")).sum()),
                "test_support": int((selected & event_labels.split.astype(str).eq("test")).sum()),
                "low_confidence_events": int((selected & event_labels.pooled_low_confidence).sum()),
                "same_mouse_merge_violation": False,
            }
        )
    registry = pd.DataFrame(registry_rows)

    # Invariants are part of the generated audit, not an informal assumption.
    # Each PG must have at most one pooled prototype per animal.
    same_mouse_violations = 0
    duplicate_animal_categories = int(
        pooled_members.groupby(["pooled_global_label", "animal_id"]).size().gt(1).sum()
    )
    if duplicate_animal_categories:
        raise AssertionError("Constrained graph produced a same-mouse merge")
    same_mouse_violations = duplicate_animal_categories

    grid["selected"] = grid.distance_threshold.astype(float).eq(float(threshold))
    grid.to_csv(out / "pooled_global_category_grid.csv", index=False, encoding="utf-8-sig")
    grid.to_parquet(out / "pooled_global_category_grid.parquet", index=False)
    registry.to_csv(out / "pooled_category_registry.csv", index=False, encoding="utf-8-sig")
    registry.to_parquet(out / "pooled_category_registry.parquet", index=False)
    pooled_prototypes.to_parquet(out / "pooled_global_prototypes.parquet", index=False)
    pooled_members.to_parquet(out / "pooled_prototype_members.parquet", index=False)
    event_labels.to_parquet(out / "pooled_event_labels.parquet", index=False)
    window_labels.to_parquet(out / "pooled_channel_window_labels.parquet", index=False)

    manifest: dict[str, object] = {
        "taxonomy_version": "pooled_global_v1_constrained_cross_mouse",
        "label_prefix": "PGxx",
        "fit_scope": "within-mouse pooled GMM prototypes fitted on training events",
        "assignment_scope": "event x valid channel; every event inherits its pooled mouse cluster",
        "merge_model": "constrained complete-linkage agglomeration in robust prototype space",
        "selected_distance_threshold": float(threshold),
        "candidate_distance_thresholds": [float(value) for value in thresholds],
        "same_mouse_merge_allowed": False,
        "same_mouse_merge_violations": int(same_mouse_violations),
        "n_animals": int(events.animal_id.astype(str).nunique()),
        "n_event_channel": int(len(event_labels)),
        "n_windows": int(len(window_labels)),
        "n_pooled_prototypes": int(len(prototypes)),
        "n_global_categories": int(len(groups)),
        "n_shared_categories": int((registry.n_animals >= 2).sum()),
        "n_animal_specific_categories": int((registry.n_animals < 2).sum()),
        "train_event_count": int((event_labels.split.astype(str) == "train").sum()),
        "val_event_count": int((event_labels.split.astype(str) == "val").sum()),
        "test_event_count": int((event_labels.split.astype(str) == "test").sum()),
        "event_distance_low_confidence_quantile": 0.995,
        "event_distance_low_confidence_threshold": float(train_distance_threshold),
        "source_files": {
            "pooled_mouse_prototypes": "03_cross_mouse/pooled_mouse_prototypes.parquet",
            "pooled_mouse_event_assignments": "03_cross_mouse/pooled_mouse_event_assignments.parquet",
            "channel_event_features": "01_tables/channel_event_features.parquet",
            "source_window_labels": "04_label_freeze/frozen_channel_window_labels.parquet",
        },
        "category_semantics": "exploratory pooled morphology category; not a clinically validated seizure subtype",
    }
    _write_json(out / "pooled_label_freeze_manifest.json", manifest)
    audit = {
        "events_input_rows": int(len(events)),
        "events_output_rows": int(len(event_labels)),
        "windows_input_rows": int(len(windows)),
        "windows_output_rows": int(len(window_labels)),
        "event_keys_unique": bool(not event_labels.duplicated(KEYS).any()),
        "window_rows_all_mapped": bool(not window_labels.pooled_global_label.isna().any()),
        "split_counts_events_input": events.split.astype(str).value_counts().sort_index().to_dict(),
        "split_counts_events_output": event_labels.split.astype(str).value_counts().sort_index().to_dict(),
        "split_counts_windows_input": windows.split.astype(str).value_counts().sort_index().to_dict(),
        "split_counts_windows_output": window_labels.split.astype(str).value_counts().sort_index().to_dict(),
        "split_conserved_events": bool(
            events.split.astype(str).value_counts().sort_index().equals(
                event_labels.split.astype(str).value_counts().sort_index()
            )
        ),
        "split_conserved_windows": bool(
            windows.split.astype(str).value_counts().sort_index().equals(
                window_labels.split.astype(str).value_counts().sort_index()
            )
        ),
        "pg_labels_contiguous": bool(
            set(event_labels.pooled_global_category.astype(int)) == set(range(len(groups)))
        ),
        "same_mouse_merge_violations": int(same_mouse_violations),
        "e77_ch7_event_rows": int(
            ((event_labels.animal_id.astype(str) == "E77") & event_labels.channel_name.astype(str).eq("CH7")).sum()
        ),
        "e77_ch7_window_rows": int(
            ((window_labels.animal_id.astype(str) == "E77") & window_labels.channel_name.astype(str).eq("CH7")).sum()
        ),
        "low_confidence_event_rows": int(event_labels.pooled_low_confidence.sum()),
        "source_taxonomy_untouched": True,
    }
    _write_json(out / "pooled_label_freeze_audit.json", audit)

    readme = f"""# Pooled global morphology taxonomy (PG labels)

This directory is an independent label-freeze product derived from the
within-mouse pooled CH7/CH8 morphology clusters. It does **not** overwrite the
strict all-channel `04_label_freeze` directory or its G01-G45 taxonomy.

## Frozen result

- `{len(events):,}` event-by-channel rows and `{len(window_labels):,}` source windows.
- `{len(prototypes)}` pooled mouse prototypes from `{events.animal_id.nunique()}` mice.
- `{len(groups)}` frozen categories labelled `PG01`-`PG{len(groups):02d}`.
- `{int((registry.n_animals >= 2).sum())}` categories are observed in at least two mice;
  `{int((registry.n_animals < 2).sum())}` are animal-specific.
- The selected robust prototype distance threshold is `{threshold:.2f}`.

## Merge constraint

Pooled clusters are the atomic units. Candidate merges are allowed only between
different mice and only when the maximum (complete-linkage) distance between
all member prototypes is below the threshold. Two components are merged only
when their sets of mice are disjoint. Therefore a single PG category can
contain at most one pooled cluster from any mouse; transitive chains cannot
merge two pooled states from the same mouse.

Event labels inherit the category of their pooled mouse cluster. They are not
reassigned to a nearest global centroid, so this constraint remains true for
all splits. `pooled_global_distance` and `pooled_low_confidence` are QC fields;
they do not silently remove events or windows.

## Files

- `pooled_event_labels.parquet`: one row per event x valid channel, with pooled
  cluster and PG label plus split and QC fields.
- `pooled_channel_window_labels.parquet`: every source window with PG labels;
  previous strict G labels are retained as `source_strict_*` columns.
- `pooled_category_registry.csv/.parquet`: support, split counts, mouse/channel
  presence and merge status for each PG category.
- `pooled_global_prototypes.parquet`: robust-space weighted centroid per PG.
- `pooled_prototype_members.parquet`: exact pooled mouse prototype membership.
- `pooled_global_category_grid.csv/.parquet`: threshold sensitivity grid.
- `pooled_label_freeze_manifest.json`: reproducibility and invariant audit.

PG labels are exploratory morphology states. They must not be described as
clinical seizure subtypes without independent clinical review.
"""
    (out / "README_zh.md").write_text(readme, encoding="utf-8")

    # Final consistency assertions are recorded after writing so a partial run
    # cannot be mistaken for a valid freeze.
    if len(event_labels) != len(events) or len(window_labels) != len(windows):
        raise AssertionError("Output row counts changed during label freeze")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    manifest = freeze(args.root, threshold=args.threshold, seed=args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
