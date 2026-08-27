from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SharedRawShapeConfig:
    """Configuration for one cross-mouse raw-waveform representation."""

    label_column: str = "individual_review_smoothed_label"
    channel_index: int = 1
    sequence_seconds: int = 4
    rocket_kernels: int = 5000
    fit_sequences_per_event_category: int = 4
    maximum_fit_sequences: int = 8000
    pca_variance: float = 0.90
    pca_max_components: int = 64
    transform_chunk_size: int = 1024
    random_seed: int = 20260813

    def validated(self) -> "SharedRawShapeConfig":
        if self.channel_index != 1:
            raise ValueError("The shared cross-mouse branch must use valid CH8 (index 1)")
        if self.sequence_seconds != 4:
            raise ValueError("Existing local labels are defined on four-second windows")
        if self.rocket_kernels < 100:
            raise ValueError("rocket_kernels must be at least 100")
        if self.fit_sequences_per_event_category < 1:
            raise ValueError("fit_sequences_per_event_category must be positive")
        if self.maximum_fit_sequences < 100:
            raise ValueError("maximum_fit_sequences must be at least 100")
        if not 0.5 <= self.pca_variance <= 1.0:
            raise ValueError("pca_variance must be in [0.5, 1.0]")
        if self.pca_max_components < 2:
            raise ValueError("pca_max_components must be at least two")
        return self


@dataclass(frozen=True)
class SharedRawShapeResult:
    sequence_assignments: pd.DataFrame
    event_category_embeddings: pd.DataFrame
    category_prototypes: pd.DataFrame
    coverage: pd.DataFrame
    model_metadata: dict[str, Any]


def fit_shared_raw_shape_representation(
    assignments: pd.DataFrame,
    native_metadata: pd.DataFrame,
    signals_path: str | Path,
    output_dir: str | Path | None = None,
    config: SharedRawShapeConfig | None = None,
) -> SharedRawShapeResult:
    """Fit one CH8 MiniRocket space shared by all mice and all 21 labels.

    Each retained row is an existing labelled four-second cache window that can
    be reconstructed from four consecutive, already filtered native one-second
    windows in the same event and continuity segment.  The representation is
    fitted with event-category-balanced sampling.  Event-category medians are
    the downstream statistical units, so long events cannot dominate matching.
    """

    cfg = (config or SharedRawShapeConfig()).validated()
    sequences = map_existing_labels_to_native_sequences(
        assignments, native_metadata, cfg
    )
    if sequences.empty:
        raise ValueError("No existing label window could be reconstructed natively")
    signals = np.load(Path(signals_path), mmap_mode="r")
    if signals.ndim != 3 or signals.shape[1] != 2:
        raise ValueError(f"Expected native signals [windows,2,samples], got {signals.shape}")
    samples_per_window = int(signals.shape[2])
    expected_samples = int(round(
        float(native_metadata["sampling_rate_hz"].median()) * cfg.sequence_seconds
    ))
    if expected_samples != samples_per_window * cfg.sequence_seconds:
        raise ValueError("Native windows do not concatenate to the requested duration")

    fit_indices = _event_category_balanced_indices(sequences, cfg)
    fit_input = _load_normalized_sequences(
        signals, sequences.iloc[fit_indices], cfg.channel_index
    )
    rocket = MiniRocket(
        n_kernels=cfg.rocket_kernels,
        n_jobs=-1,
        random_state=cfg.random_seed,
    )
    fit_rocket = np.asarray(rocket.fit_transform(fit_input), dtype=np.float32)
    scaler = StandardScaler().fit(fit_rocket)
    fit_scaled = np.clip(scaler.transform(fit_rocket), -8.0, 8.0)
    pca, fit_embedding = _fit_variance_pca(
        fit_scaled, cfg.pca_variance, cfg.pca_max_components, cfg.random_seed
    )
    branch_center = np.median(fit_embedding, axis=0)
    centered_fit = fit_embedding - branch_center
    branch_scale = float(np.sqrt(np.mean(np.sum(centered_fit**2, axis=1))))
    branch_scale = max(branch_scale, 1e-12)

    embedding = np.empty((len(sequences), pca.n_components_), dtype=np.float32)
    for start in range(0, len(sequences), cfg.transform_chunk_size):
        stop = min(len(sequences), start + cfg.transform_chunk_size)
        shape_input = _load_normalized_sequences(
            signals, sequences.iloc[start:stop], cfg.channel_index
        )
        rocket_values = np.asarray(rocket.transform(shape_input), dtype=np.float32)
        scaled = np.clip(scaler.transform(rocket_values), -8.0, 8.0)
        projected = pca.transform(scaled)
        embedding[start:stop] = ((projected - branch_center) / branch_scale).astype(
            np.float32
        )

    pc_columns = [f"raw_shape_pc{index + 1}" for index in range(embedding.shape[1])]
    sequence_assignments = sequences.copy()
    sequence_assignments[pc_columns] = embedding
    keys = ["animal_id", "session_id", "event_id", "local_type"]
    grouped = sequence_assignments.groupby(keys, sort=True, observed=True)
    event_category = grouped[pc_columns].median().reset_index()
    sequence_counts = grouped.size().rename("raw_shape_sequence_count").reset_index()
    event_category = event_category.merge(
        sequence_counts, on=keys, validate="one_to_one"
    )
    prototypes = (
        event_category.groupby(["animal_id", "local_type"], sort=True)[pc_columns]
        .median()
        .reset_index()
    )
    event_counts = (
        event_category.groupby(["animal_id", "local_type"], sort=True)
        .agg(raw_shape_independent_events=("event_id", "nunique"))
        .reset_index()
    )
    prototypes = prototypes.merge(
        event_counts, on=["animal_id", "local_type"], validate="one_to_one"
    )
    coverage = _coverage_table(assignments, sequences, cfg.label_column)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "representation": "one_shared_CH8_MiniRocket_PCA",
        "shape_only_normalization": "per_sequence_median_and_q01_q99_range",
        "amplitude_preserved_elsewhere": True,
        "input_sequence_samples": expected_samples,
        "input_channels": ["CH8_normalized", "CH8_first_difference"],
        "fit_sequences": int(len(fit_indices)),
        "all_reconstructed_sequences": int(len(sequences)),
        "event_category_units": int(len(event_category)),
        "pca_components": int(pca.n_components_),
        "pca_explained_variance_fraction": float(
            np.sum(pca.explained_variance_ratio_)
        ),
        "config": asdict(cfg),
    }
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        sequence_assignments.to_parquet(
            root / "shared_raw_shape_sequence_assignments.parquet", index=False
        )
        event_category.to_parquet(
            root / "shared_raw_shape_event_category_embeddings.parquet", index=False
        )
        prototypes.to_csv(
            root / "shared_raw_shape_category_prototypes.csv",
            index=False,
            encoding="utf-8-sig",
        )
        coverage.to_csv(
            root / "shared_raw_shape_coverage.csv", index=False, encoding="utf-8-sig"
        )
        joblib.dump(
            {
                "rocket": rocket,
                "scaler": scaler,
                "pca": pca,
                "branch_center": branch_center,
                "branch_scale": branch_scale,
                "fit_indices": fit_indices,
                "metadata": metadata,
            },
            root / "shared_raw_shape_model.joblib",
        )
        (root / "shared_raw_shape_manifest.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return SharedRawShapeResult(
        sequence_assignments=sequence_assignments,
        event_category_embeddings=event_category,
        category_prototypes=prototypes,
        coverage=coverage,
        model_metadata=metadata,
    )


def map_existing_labels_to_native_sequences(
    assignments: pd.DataFrame,
    native_metadata: pd.DataFrame,
    config: SharedRawShapeConfig | None = None,
) -> pd.DataFrame:
    """Map exact existing 4-s labels to continuous native 1-s sequences."""

    cfg = (config or SharedRawShapeConfig()).validated()
    required_old = {
        "animal_id",
        "session_id",
        "event_id",
        "cache_dir",
        "cache_local_index",
        "start_time_ms",
        "end_time_ms",
        "event_overlap_fraction",
        cfg.label_column,
    }
    required_native = {
        "signal_index",
        "animal_id",
        "session_id",
        "event_id",
        "cache_dir",
        "cache_local_index",
        "continuity_segment",
        "sample_start_in_segment",
        "window_samples",
        "sampling_rate_hz",
        "ch8_valid",
    }
    missing_old = required_old - set(assignments)
    missing_native = required_native - set(native_metadata)
    if missing_old or missing_native:
        raise ValueError(
            f"Missing old columns {sorted(missing_old)}; native columns {sorted(missing_native)}"
        )
    native = native_metadata.copy()
    if not native["ch8_valid"].astype(bool).all():
        native = native.loc[native["ch8_valid"].astype(bool)].copy()
    start_lookup = {
        (str(row.cache_dir), int(row.cache_local_index)): row
        for row in native.itertuples(index=False)
    }
    position_lookup = {
        (
            str(row.event_id),
            int(row.continuity_segment),
            int(row.sample_start_in_segment),
        ): int(row.signal_index)
        for row in native.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for old in assignments.itertuples(index=False):
        if float(getattr(old, "event_overlap_fraction")) < 0.99:
            continue
        first = start_lookup.get((str(old.cache_dir), int(old.cache_local_index)))
        if first is None or str(first.event_id) != str(old.event_id):
            continue
        stride = int(first.window_samples)
        indices = []
        for offset in range(cfg.sequence_seconds):
            key = (
                str(old.event_id),
                int(first.continuity_segment),
                int(first.sample_start_in_segment) + offset * stride,
            )
            index = position_lookup.get(key)
            if index is None:
                indices = []
                break
            indices.append(index)
        if len(indices) != cfg.sequence_seconds:
            continue
        label = str(getattr(old, cfg.label_column))
        record = {
            "animal_id": str(old.animal_id),
            "session_id": str(old.session_id),
            "event_id": str(old.event_id),
            "local_type": label,
            "start_time_ms": float(old.start_time_ms),
            "end_time_ms": float(old.end_time_ms),
            "continuity_segment": int(first.continuity_segment),
            "native_signal_indices": ";".join(str(value) for value in indices),
            "label_mapping_method": "exact_existing_4s_window_start",
            "label_mapping_ambiguous": False,
            "label_mapping_support": 1.0,
        }
        if hasattr(old, "event_position_fraction"):
            record["event_position_fraction"] = float(old.event_position_fraction)
        rows.append(record)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result.insert(0, "shape_sequence_index", np.arange(len(result), dtype=np.int64))
    return result


def _load_normalized_sequences(
    signals: np.ndarray, rows: pd.DataFrame, channel_index: int
) -> np.ndarray:
    sequences = []
    for value in rows["native_signal_indices"].astype(str):
        indices = np.fromstring(value, sep=";", dtype=np.int64)
        sequences.append(np.asarray(signals[indices, channel_index], dtype=np.float32).reshape(-1))
    values = np.stack(sequences).astype(np.float32, copy=False)
    centered = values - np.median(values, axis=1, keepdims=True)
    q01 = np.quantile(centered, 0.01, axis=1, keepdims=True)
    q99 = np.quantile(centered, 0.99, axis=1, keepdims=True)
    scale = np.maximum(q99 - q01, 1e-6)
    normalized = np.clip(centered / scale, -4.0, 4.0).astype(np.float32)
    difference = np.diff(normalized, axis=1, prepend=normalized[:, :1])
    return np.stack([normalized, difference], axis=1).astype(np.float32)


def _event_category_balanced_indices(
    sequences: pd.DataFrame, config: SharedRawShapeConfig
) -> np.ndarray:
    groups = [
        group.index.to_numpy(np.int64)
        for _, group in sequences.groupby(
            ["animal_id", "event_id", "local_type"],
            sort=True,
            observed=True,
        )
    ]
    if not groups:
        raise ValueError("No event-category groups are available for shape fitting")
    target = min(
        config.fit_sequences_per_event_category,
        max(1, config.maximum_fit_sequences // len(groups)),
    )
    rng = np.random.default_rng(config.random_seed)
    selected: list[int] = []
    for indices in groups:
        if len(indices) >= target:
            positions = np.linspace(0, len(indices) - 1, target, dtype=np.int64)
            chosen = indices[positions]
        else:
            chosen = rng.choice(indices, target, replace=True)
        selected.extend(int(value) for value in chosen)
    # Duplicated indices intentionally make short event-category cells carry
    # the same fit weight as long cells.  Do not deduplicate this array.
    return np.asarray(selected, dtype=np.int64)


def _fit_variance_pca(
    values: np.ndarray, variance: float, maximum_components: int, seed: int
) -> tuple[PCA, np.ndarray]:
    maximum = min(maximum_components, values.shape[1], len(values) - 1)
    if maximum < 2:
        raise ValueError("Shared raw-shape PCA needs at least three fit sequences")
    probe = PCA(n_components=maximum, random_state=seed).fit(values)
    cumulative = np.cumsum(probe.explained_variance_ratio_)
    components = min(maximum, max(2, int(np.searchsorted(cumulative, variance) + 1)))
    if components == maximum:
        return probe, probe.transform(values)
    model = PCA(n_components=components, random_state=seed).fit(values)
    return model, model.transform(values)


def _coverage_table(
    assignments: pd.DataFrame, sequences: pd.DataFrame, label_column: str
) -> pd.DataFrame:
    old = (
        assignments.groupby(["animal_id", label_column], sort=True)
        .agg(
            labelled_windows=("event_id", "size"),
            labelled_independent_events=("event_id", "nunique"),
        )
        .reset_index()
        .rename(columns={label_column: "local_type"})
    )
    recovered = (
        sequences.groupby(["animal_id", "local_type"], sort=True)
        .agg(
            reconstructed_sequences=("event_id", "size"),
            raw_shape_independent_events=("event_id", "nunique"),
        )
        .reset_index()
    )
    result = old.merge(
        recovered, on=["animal_id", "local_type"], how="left", validate="one_to_one"
    ).fillna({"reconstructed_sequences": 0, "raw_shape_independent_events": 0})
    result["sequence_coverage_fraction"] = (
        result["reconstructed_sequences"] / result["labelled_windows"]
    )
    result["event_coverage_fraction"] = (
        result["raw_shape_independent_events"] / result["labelled_independent_events"]
    )
    return result


__all__ = [
    "SharedRawShapeConfig",
    "SharedRawShapeResult",
    "fit_shared_raw_shape_representation",
    "map_existing_labels_to_native_sequences",
]
