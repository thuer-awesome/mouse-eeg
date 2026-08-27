"""Minimal, read-only reanalysis of the legacy six-mouse morphology cohort.

This stage consumes historical feature tables only.  It never refilters raw
signals, rebuilds windows, rewrites annotations, or modifies ``runs/``.  The
historical 21 within-mouse candidate labels remain a reference overlay, while
the new calculations use CH8, independent clinical events, and mouse-balanced
cross-mouse aggregation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Must be set before NumPy/scikit-learn import so the bootstrap audit has a
# stable, bounded CPU footprint when formal training is running concurrently.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler


ANIMALS = ("E50", "E76", "E77", "E79", "E81", "E82")
IDENTITY = ("animal_id", "session_id", "event_id", "cache_dir", "cache_local_index")
EVENT = ("animal_id", "session_id", "event_id")
OLD_LABEL = "individual_review_smoothed_label"
FEATURES = (
    "ch8_log_robust_amplitude",
    "ch8_log_rms",
    "ch8_log_peak_to_peak",
    "ch8_log_line_length",
    "ch8_log_total_power",
    "ch8_hjorth_mobility",
    "ch8_hjorth_complexity",
    "ch8_zero_crossing_rate",
    "ch8_excursion_rate_hz",
    "ch8_crest_factor",
    "ch8_peak_frequency_hz",
    "ch8_median_frequency_hz",
    "ch8_spectral_entropy",
    "ch8_spectral_concentration",
    "ch8_relative_delta",
    "ch8_relative_theta",
    "ch8_relative_alpha",
    "ch8_relative_beta",
    "ch8_relative_low_gamma_clean",
    "ch8_relative_high_gamma",
    "ch8_theta_delta_ratio",
    "ch8_high_low_ratio",
    "ch8_relative_line_noise_48_52",
)


@dataclass(frozen=True)
class Stage1Config:
    fully_contained_fraction: float = 0.99
    max_components: int = 5
    min_events_multistate: int = 20
    min_events_per_state: int = 8
    bootstrap_repeats: int = 24
    min_median_ari: float = 0.60
    min_q25_ari: float = 0.50
    min_bic_improvement: float = 10.0
    pca_components: int = 8
    seed: int = 20260821


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _need(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def source_paths(root: str | Path) -> dict[str, Path]:
    root = Path(root).resolve()
    return {
        "features": root / "tables" / "morphology_features.parquet",
        "baseline": root / "per_mouse" / "baseline_features.parquet",
        "assignments": root / "per_mouse" / "per_mouse_morphology_assignments.parquet",
        "feature_filter": root / "tables" / "feature_preprocessing.json",
        "baseline_filter": root / "per_mouse" / "baseline_preprocessing.json",
        "manifest": root / "morphology_manifest.json",
        "event_components": root / "tables" / "event_components.csv",
        "shared_shape": root / "per_mouse" / "cross_mouse_waveform_types" / "analysis_manifest.json",
    }


def load_sources(root: str | Path) -> dict[str, Any]:
    paths = source_paths(root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing legacy sources: " + "; ".join(missing))
    return {
        "paths": paths,
        "features": pd.read_parquet(paths["features"]),
        "baseline": pd.read_parquet(paths["baseline"]),
        "assignments": pd.read_parquet(paths["assignments"]),
        "event_components": pd.read_csv(paths["event_components"]),
        "feature_filter": json.loads(paths["feature_filter"].read_text(encoding="utf-8")),
        "baseline_filter": json.loads(paths["baseline_filter"].read_text(encoding="utf-8")),
        "manifest": json.loads(paths["manifest"].read_text(encoding="utf-8")),
        "shared_shape": json.loads(paths["shared_shape"].read_text(encoding="utf-8")),
    }


def audit_sources(data: dict[str, Any]) -> dict[str, Any]:
    features, baseline, assignments = data["features"], data["baseline"], data["assignments"]
    event_components = data["event_components"]
    _need(features, (*IDENTITY, "event_overlap_fraction", "ch7_valid", "ch8_valid"), "features")
    _need(baseline, ("animal_id", "session_id", "ch7_valid", "ch8_valid"), "baseline")
    _need(assignments, (*IDENTITY, OLD_LABEL), "assignments")
    _need(event_components, EVENT, "event_components")
    key_merge = features[list(IDENTITY)].merge(
        assignments[list(IDENTITY)].drop_duplicates(), on=list(IDENTITY), how="outer", indicator=True
    )
    event_merge = features[list(EVENT)].drop_duplicates().merge(
        event_components[list(EVENT)].drop_duplicates(),
        on=list(EVENT),
        how="outer",
        indicator=True,
    )
    validity = features.groupby("animal_id", sort=True)[["ch7_valid", "ch8_valid"]].mean().reset_index()
    old_counts = assignments.groupby("animal_id", sort=True)[OLD_LABEL].nunique().astype(int).to_dict()
    event_counts = features.groupby("animal_id", sort=True)["event_id"].nunique().astype(int).to_dict()
    full_counts = (
        features.loc[features["event_overlap_fraction"].ge(0.99)]
        .groupby("animal_id", sort=True)["event_id"].nunique().astype(int).to_dict()
    )
    identity = {
        "feature_duplicate_rows": int(features.duplicated(list(IDENTITY)).sum()),
        "assignment_duplicate_rows": int(assignments.duplicated(list(IDENTITY)).sum()),
        "feature_rows_missing_assignment": int(key_merge["_merge"].eq("left_only").sum()),
        "assignment_rows_missing_feature": int(key_merge["_merge"].eq("right_only").sum()),
        "event_component_duplicate_rows": int(event_components.duplicated(list(EVENT)).sum()),
        "feature_events_missing_component": int(event_merge["_merge"].eq("left_only").sum()),
        "component_events_missing_feature": int(event_merge["_merge"].eq("right_only").sum()),
    }
    expected_filter = {
        "highpass_hz": 0.5,
        "lowpass_hz": 70.0,
        "notch_hz": 50.0,
        "notch_quality_factor": 30.0,
        "bandpass_order": 4,
    }
    feature_filter = data["feature_filter"].get("filter", {})
    baseline_filter = data["baseline_filter"].get("filter", {})
    filter_check = {
        "expected": expected_filter,
        "feature_table_matches_expected": all(feature_filter.get(key) == value for key, value in expected_filter.items()),
        "baseline_table_matches_expected": all(baseline_filter.get(key) == value for key, value in expected_filter.items()),
        "feature_and_baseline_match": feature_filter == baseline_filter,
    }
    status = (
        "pass"
        if not any(identity.values())
        and all(value for key, value in filter_check.items() if key != "expected")
        else "attention_required"
    )
    return {
        "status": status,
        "source_hashes": {key: {"path": str(path), "sha256": _hash(path)} for key, path in data["paths"].items()},
        "source_rows": {"features": int(len(features)), "baseline": int(len(baseline)), "assignments": int(len(assignments))},
        "identity": identity,
        "event_component_rows": int(len(event_components)),
        "event_counts_by_mouse": event_counts,
        "fully_contained_event_counts_by_mouse": full_counts,
        "legacy_local_candidate_counts_by_mouse": old_counts,
        "legacy_local_candidate_total": int(sum(old_counts.values())),
        "channel_validity_by_mouse": validity.to_dict(orient="records"),
        "filter_contract": feature_filter,
        "baseline_filter_contract": baseline_filter,
        "filter_contract_check": filter_check,
        "signal_unit_verified": bool(data["manifest"].get("signal_unit_verified", False)),
        "shared_ch8_shape_reference": data["shared_shape"].get("shared_raw_shape", {}),
        "rule": "E77 CH7 excluded from primary analysis; CH8 required for every included event",
    }


def _robust_stats(values: pd.Series) -> tuple[float, float, str, int]:
    values = pd.to_numeric(values, errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan, "no_finite_baseline", 0
    center = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - center)))
    method = "MAD"
    if not np.isfinite(scale) or scale <= 1e-8:
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale, method = float(q75 - q25) / 1.349, "IQR"
    if not np.isfinite(scale) or scale <= 1e-8:
        return center, math.nan, "constant_baseline", int(len(values))
    return center, scale, method, int(len(values))


def normalize_ch8(features: pd.DataFrame, baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    _need(features, (*FEATURES, "animal_id", "session_id", "ch8_valid"), "features")
    _need(baseline, (*FEATURES, "animal_id", "session_id", "ch8_valid"), "baseline")
    normalized = features.copy()
    rows: list[dict[str, Any]] = []
    stats: dict[tuple[str, str, str], tuple[float, float]] = {}
    for (animal, session), group in baseline.loc[baseline["ch8_valid"].astype(bool)].groupby(["animal_id", "session_id"], sort=True):
        for feature in FEATURES:
            center, scale, method, n = _robust_stats(group[feature])
            stats[(str(animal), str(session), feature)] = (center, scale)
            rows.append({"animal_id": str(animal), "session_id": str(session), "source_feature": feature, "center": center, "scale": scale, "baseline_rows_finite": n, "method": method, "unit": "a.u. or relative baseline"})
    parameters = pd.DataFrame(rows)
    usable: list[str] = []
    for feature in FEATURES:
        target = f"norm__{feature}"
        values = np.full(len(normalized), np.nan, dtype=float)
        for index, row in enumerate(normalized[["animal_id", "session_id", feature]].itertuples(index=False)):
            center, scale = stats.get((str(row[0]), str(row[1]), feature), (math.nan, math.nan))
            value = pd.to_numeric(pd.Series([row[2]]), errors="coerce").iloc[0]
            if np.isfinite(value) and np.isfinite(scale):
                values[index] = (float(value) - center) / scale
        normalized[target] = values
        check = parameters.loc[parameters["source_feature"].eq(feature), "scale"]
        if len(check) and check.notna().all():
            usable.append(target)
    parameters["cross_mouse_usable"] = parameters["source_feature"].map(lambda x: f"norm__{x}" in usable)
    return normalized, parameters, usable


def _mode(values: pd.Series) -> str | None:
    values = values.dropna().astype(str)
    if values.empty:
        return None
    counts = values.value_counts()
    return sorted(counts.loc[counts.eq(counts.max())].index.astype(str))[0]


def event_summary(normalized: pd.DataFrame, assignments: pd.DataFrame, usable: list[str], cfg: Stage1Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = assignments[list(IDENTITY) + [OLD_LABEL]]
    frame = normalized.merge(labels, on=list(IDENTITY), how="left", validate="one_to_one")
    finite = frame[usable].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    frame["_eligible"] = frame["ch8_valid"].astype(bool) & frame["event_overlap_fraction"].ge(cfg.fully_contained_fraction) & finite
    audit_rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(EVENT), sort=True):
        eligible = group.loc[group["_eligible"]]
        full = group.loc[group["event_overlap_fraction"].ge(cfg.fully_contained_fraction)]
        ch8 = group.loc[group["ch8_valid"].astype(bool)]
        reason = "eligible" if len(eligible) else ("no_valid_CH8" if not len(ch8) else "no_fully_contained_4s_window" if not len(full) else "nonfinite_feature")
        audit_rows.append({"animal_id": str(key[0]), "session_id": str(key[1]), "event_id": str(key[2]), "all_windows": int(len(group)), "valid_ch8_windows": int(len(ch8)), "fully_contained_windows": int(len(full)), "eligible_windows": int(len(eligible)), "event_duration_s": float(pd.to_numeric(group["event_duration_s"], errors="coerce").median()), "eligibility": reason})
    audit = pd.DataFrame(audit_rows)
    selected = frame.loc[frame["_eligible"]].copy()
    if selected.empty:
        raise ValueError("No eligible CH8 fully-contained event windows")
    keys = list(EVENT)
    events = selected.groupby(keys, sort=True)[usable].median().reset_index()
    metadata = selected.groupby(keys, sort=True).agg(
        eligible_windows=("event_id", "size"),
        event_duration_s=("event_duration_s", "median"),
        legacy_local_state=(OLD_LABEL, _mode),
        legacy_local_state_count=(OLD_LABEL, "nunique"),
    ).reset_index()
    return events.merge(metadata, on=keys, validate="one_to_one"), audit, selected


def _gmm(values: np.ndarray, k: int, seed: int) -> GaussianMixture:
    model = GaussianMixture(n_components=k, covariance_type="diag", n_init=4, max_iter=400, reg_covar=1e-4, random_state=seed)
    model.fit(values)
    return model


def _stability(values: np.ndarray, reference: np.ndarray, k: int, repeats: int, seed: int) -> tuple[float, float]:
    if k == 1:
        return 1.0, 1.0
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(repeats):
        sample = rng.integers(0, len(values), len(values))
        try:
            predicted = _gmm(values[sample], k, int(rng.integers(1, 2**31 - 1))).predict(values)
        except ValueError:
            continue
        scores.append(float(adjusted_rand_score(reference, predicted)))
    if not scores:
        return math.nan, math.nan
    return float(np.median(scores)), float(np.quantile(scores, 0.25))


def local_event_states(events: pd.DataFrame, usable: list[str], cfg: Stage1Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments: list[pd.DataFrame] = []
    candidate_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for mouse, group in events.groupby("animal_id", sort=True):
        group = group.reset_index(drop=True)
        matrix = RobustScaler(quantile_range=(10, 90)).fit_transform(group[usable].to_numpy(float))
        maximum = 1 if len(group) < cfg.min_events_multistate else min(cfg.max_components, max(1, len(group) // cfg.min_events_per_state))
        rows: list[dict[str, Any]] = []
        models: dict[int, tuple[GaussianMixture, np.ndarray]] = {}
        for k in range(1, maximum + 1):
            model = _gmm(matrix, k, cfg.seed + 101 * k + len(group))
            labels = model.predict(matrix)
            median_ari, q25_ari = _stability(matrix, labels, k, cfg.bootstrap_repeats, cfg.seed + 1009 * k + len(group))
            support = np.bincount(labels, minlength=k)
            rows.append({"animal_id": mouse, "n_events": int(len(group)), "components": k, "bic": float(model.bic(matrix)), "aic": float(model.aic(matrix)), "min_state_events": int(support.min()), "bootstrap_median_ari": median_ari, "bootstrap_q25_ari": q25_ari})
            models[k] = (model, labels)
        grid = pd.DataFrame(rows)
        bic_one = float(grid.loc[grid["components"].eq(1), "bic"].iloc[0])
        grid["bic_improvement_vs_one"] = bic_one - grid["bic"]
        grid["stability_pass"] = grid["components"].eq(1) | (grid["min_state_events"].ge(cfg.min_events_per_state) & grid["bootstrap_median_ari"].ge(cfg.min_median_ari) & grid["bootstrap_q25_ari"].ge(cfg.min_q25_ari) & grid["bic_improvement_vs_one"].ge(cfg.min_bic_improvement))
        stable = grid.loc[grid["components"].gt(1) & grid["stability_pass"]]
        selected = 1 if stable.empty else int(stable.sort_values(["bic", "components"]).iloc[0]["components"])
        reason = "single_state_reference_no_stable_multistate_resolution" if selected == 1 else "lowest_BIC_stable_event_bootstrap_resolution"
        grid["selected"] = grid["components"].eq(selected)
        grid["selection_reason"] = reason
        model, labels = models[selected]
        order = np.argsort(model.means_[:, 0], kind="stable")
        remap = {int(old): int(new) for new, old in enumerate(order)}
        labels = np.asarray([remap[int(value)] for value in labels], dtype=int)
        probabilities = model.predict_proba(matrix)[:, order].max(axis=1)
        assigned = group[["animal_id", "session_id", "event_id", "event_duration_s", "eligible_windows", "legacy_local_state", "legacy_local_state_count"]].copy()
        assigned["exploratory_event_state"] = [f"{mouse}-R{value + 1}" for value in labels]
        assigned["exploratory_event_state_probability"] = probabilities
        assigned["selected_components"] = selected
        assigned["selection_reason"] = reason
        assignments.append(assigned)
        candidate_rows.extend(grid.to_dict(orient="records"))
        for state, state_rows in assigned.groupby("exploratory_event_state", sort=True):
            summary_rows.append({"animal_id": mouse, "exploratory_event_state": state, "independent_events": int(len(state_rows)), "event_fraction": float(len(state_rows) / len(assigned)), "median_duration_s": float(state_rows["event_duration_s"].median()), "median_probability": float(state_rows["exploratory_event_state_probability"].median()), "status": "exploratory_not_clinical_subtype"})
    return pd.concat(assignments, ignore_index=True), pd.DataFrame(candidate_rows), pd.DataFrame(summary_rows)


def balanced_pca(events: pd.DataFrame, usable: list[str], cfg: Stage1Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if len(events) < 3:
        raise ValueError("Mouse-balanced PCA requires at least three independent events")
    values = events[usable].to_numpy(float)
    counts = events.groupby("animal_id").size().astype(float)
    weights = events["animal_id"].map((1.0 / counts).to_dict()).to_numpy(float)
    weights = weights / weights.sum() * len(weights)
    center = np.average(values, axis=0, weights=weights)
    scale = np.sqrt(np.average(np.square(values - center), axis=0, weights=weights))
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    standardized = (values - center) / scale
    n_components = min(cfg.pca_components, len(usable), len(events) - 1)
    # The rows have a zero weighted mean by construction.  Direct SVD of the
    # weighted, centered matrix therefore gives the actual equal-mouse PCA;
    # multiplying rows and then using sklearn.PCA would center that matrix a
    # second time and distort the intended weighting.
    _, singular_values, right_vectors = np.linalg.svd(
        standardized * np.sqrt(weights)[:, None], full_matrices=False
    )
    components = right_vectors[:n_components].copy()
    eigenvalues = np.square(singular_values)
    for component in components:
        anchor = int(np.argmax(np.abs(component)))
        if component[anchor] < 0:
            component *= -1.0
    coordinates = standardized @ components.T
    pc_columns = [f"weighted_pc{i + 1}" for i in range(coordinates.shape[1])]
    result = events[["animal_id", "session_id", "event_id", "event_duration_s", "eligible_windows", "legacy_local_state", "legacy_local_state_count", "exploratory_event_state"]].copy()
    result["mouse_balanced_weight"] = weights
    result[pc_columns] = coordinates
    loadings = pd.DataFrame({"feature": usable, **{f"pc{i + 1}_loading": components[i] for i in range(n_components)}})
    umap = events[["animal_id", "session_id", "event_id"]].copy()
    umap["mouse_balanced_weight"] = weights
    umap[[f"umap_feature_{i + 1:02d}" for i in range(len(usable))]] = standardized
    metadata = {
        "representation": "CH8_primary_event_median_mouse_balanced_PCA",
        "analysis_unit": "one independent clinical event",
        "features": usable,
        "pca_components": int(n_components),
        "explained_variance_fraction": [float(value / eigenvalues.sum()) for value in eigenvalues[:n_components]],
        "mouse_weight_rule": "each mouse has equal total weight",
        "pca_implementation": "direct_SVD_of_weighted_centered_matrix",
        "umap_status": "not run; umap-learn unavailable",
    }
    result.attrs["pca_center"] = center
    result.attrs["pca_scale"] = scale
    result.attrs["pca_components_matrix"] = components
    return result, loadings, umap, metadata


def legacy_reference(normalized: pd.DataFrame, assignments: pd.DataFrame, usable: list[str], pca_events: pd.DataFrame, cfg: Stage1Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = assignments[list(IDENTITY) + [OLD_LABEL]]
    frame = normalized.merge(labels, on=list(IDENTITY), how="left", validate="one_to_one")
    frame = frame.loc[frame["ch8_valid"].astype(bool) & frame["event_overlap_fraction"].ge(cfg.fully_contained_fraction)].copy()
    frame = frame.loc[frame[usable].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)]
    cell_keys = [*EVENT, OLD_LABEL]
    cells = frame.groupby(cell_keys, sort=True)[usable].median().reset_index()
    support = cells.groupby(["animal_id", OLD_LABEL], sort=True).agg(independent_events=("event_id", "nunique"), event_state_cells=("event_id", "size")).reset_index()
    prototype = cells.groupby(["animal_id", OLD_LABEL], sort=True)[usable].median().reset_index().merge(support, on=["animal_id", OLD_LABEL], validate="one_to_one")
    center = pca_events.attrs["pca_center"]
    scale = pca_events.attrs["pca_scale"]
    components = pca_events.attrs["pca_components_matrix"]
    coordinates = ((prototype[usable].to_numpy(float) - center) / scale) @ components.T
    prototype[[f"weighted_pc{i + 1}" for i in range(coordinates.shape[1])]] = coordinates
    prototype["reference_status"] = "legacy_21_read_only_overlay_not_formal_global_label"
    return prototype, cells


def write_scope(root: Path, audit: dict[str, Any], meta: dict[str, Any], events: int, states: int) -> None:
    protocol = root / "01_protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    old = audit["legacy_local_candidate_counts_by_mouse"]
    count_text = "+".join(str(old.get(mouse, 0)) for mouse in ANIMALS)
    (protocol / "legacy_reuse_and_scope.md").write_text(
        f"""# 旧六鼠形态结果复用与重算范围

## 直接复用

- 38,133 个已过滤四秒窗和 1,142 个合并临床事件；不重滤波、不重切窗、不重标注。
- 过滤合同：0.5–70 Hz 零相位带通、50 Hz 陷波（Q=30，四阶）。
- E77 CH7 永久排除；CH8 为六鼠共同主通道。
- 历史逐鼠候选保留：E50/E76/E77/E79/E81/E82 = {count_text}，总计 21；旧原型和 CH8 shared MiniRocket PCA 只读复用为对照。

## 最小重算

- 对既有 CH8 标量特征按记录做稳健相对基线标准化。
- 以完全包含的四秒窗求每个独立事件中位数，避免长事件和重复窗主导。
- 逐鼠进行事件级候选分辨率和 bootstrap 稳定性审查。
- 以每鼠总权重相等做事件级跨鼠 PCA；UMAP 输入已导出，环境缺少 `umap-learn`，不伪造 UMAP 图。

## 不可作正式结论

- 历史 21 个局部候选不是全局临床 21 类，不能直接训练多分类器。
- 当前事件状态只是探索性复核，不能命名为发作亚型。
- ADC 到微伏未标定，幅值只解释为 a.u. 或相对基线。
- 本轮未接入新批次，未进行多分类训练；`06_multiclass_after_label_freeze` 保持阻塞。

## 本次产物

- CH8 事件级主分析：{events} 个独立事件。
- 历史标签可投影参考状态：{states} 个。
- PCA 前两成分解释方差：{100 * sum(meta['explained_variance_fraction'][:2]):.1f}%。
""",
        encoding="utf-8",
    )
    (root / "06_multiclass_after_label_freeze" / "STATUS_zh.md").write_text(
        "# 多分类状态：阻塞\n\n全局类别未完成跨鼠稳定性与临床冻结。禁止把旧 21 个逐鼠候选直接作为正式多分类标签。\n",
        encoding="utf-8",
    )
    (root / "STATUS_zh.md").write_text(
        "# 任务 2 状态\n\n仅完成旧六鼠的最小统一复核。旧 `runs/` 只读；新批次未纳入；多分类训练保持阻塞。\n",
        encoding="utf-8",
    )


def run(source_root: str | Path, output_root: str | Path, cfg: Stage1Config | None = None) -> dict[str, Any]:
    cfg = cfg or Stage1Config()
    root = Path(output_root).resolve()
    for name in ("01_protocol", "02_feature_inventory", "03_per_mouse", "04_cross_mouse", "05_figures", "06_multiclass_after_label_freeze"):
        (root / name).mkdir(parents=True, exist_ok=True)
    data = load_sources(source_root)
    audit = audit_sources(data)
    if audit["status"] != "pass":
        raise ValueError(f"Source identity audit failed: {audit['identity']}")
    normalized, parameters, usable = normalize_ch8(data["features"], data["baseline"])
    if len(usable) < 5:
        raise ValueError(
            "Fewer than five CH8 features have finite session-baseline normalization; "
            "cross-mouse morphology reassessment is not interpretable."
        )
    events, eligibility, eligible_windows = event_summary(normalized, data["assignments"], usable, cfg)
    state_assignments, candidates, state_summary = local_event_states(events, usable, cfg)
    events = events.merge(state_assignments[[*EVENT, "exploratory_event_state"]], on=list(EVENT), validate="one_to_one")
    selected_resolution = (
        candidates.loc[candidates["selected"].astype(bool), ["animal_id", "components", "selection_reason"]]
        .drop_duplicates("animal_id")
        .sort_values("animal_id")
    )
    selected_resolution_by_mouse = {
        str(row.animal_id): int(row.components)
        for row in selected_resolution.itertuples(index=False)
    }
    pca_events, loadings, umap, metadata = balanced_pca(events, usable, cfg)
    prototypes, cells = legacy_reference(normalized, data["assignments"], usable, pca_events, cfg)
    feature_dir, mouse_dir, cross_dir = root / "02_feature_inventory", root / "03_per_mouse", root / "04_cross_mouse"
    parameters.to_csv(feature_dir / "session_baseline_normalization.csv", index=False, encoding="utf-8-sig")
    eligibility.to_csv(feature_dir / "event_eligibility_audit.csv", index=False, encoding="utf-8-sig")
    events.to_parquet(feature_dir / "ch8_event_feature_summary.parquet", index=False)
    eligible_windows[[*EVENT, OLD_LABEL, *usable]].to_parquet(feature_dir / "ch8_eligible_window_reference.parquet", index=False)
    state_assignments.to_parquet(mouse_dir / "exploratory_event_state_assignments.parquet", index=False)
    candidates.to_csv(mouse_dir / "event_state_resolution_candidates.csv", index=False, encoding="utf-8-sig")
    state_summary.to_csv(mouse_dir / "exploratory_event_state_summary.csv", index=False, encoding="utf-8-sig")
    # Transform parameters are intentionally retained only in memory for the
    # immediately following legacy-label projection.  Parquet metadata accepts
    # JSON only, so write a clean tabular copy after that projection is done.
    pca_events_disk = pca_events.copy()
    pca_events_disk.attrs = {}
    pca_events_disk.to_parquet(cross_dir / "mouse_balanced_event_pca.parquet", index=False)
    loadings.to_csv(cross_dir / "mouse_balanced_pca_loadings.csv", index=False, encoding="utf-8-sig")
    umap.to_parquet(cross_dir / "umap_ready_event_matrix.parquet", index=False)
    prototypes.to_csv(cross_dir / "legacy_21_read_only_pca_centroids.csv", index=False, encoding="utf-8-sig")
    cells.to_parquet(cross_dir / "legacy_21_event_state_cells.parquet", index=False)
    _write_json(cross_dir / "umap_status.json", {"status": "blocked", "reason": "umap-learn unavailable", "input": "umap_ready_event_matrix.parquet"})
    manifest = {
        "stage": "legacy_morphology_stage1",
        "config": asdict(cfg),
        "audit": audit,
        "usable_ch8_features": usable,
        "eligible_independent_events": int(len(events)),
        "legacy_21_reference_states": int(len(prototypes)),
        "event_level_exploratory_resolution_by_mouse": selected_resolution_by_mouse,
        "event_level_exploratory_state_total": int(sum(selected_resolution_by_mouse.values())),
        "global_category_count_established": False,
        "pca": metadata,
        "historical_runs_modified": False,
        "new_batch_included": False,
        "multiclass_training_performed": False,
    }
    _write_json(root / "01_protocol" / "source_audit.json", audit)
    _write_json(root / "01_protocol" / "stage1_manifest.json", manifest)
    write_scope(root, audit, metadata, len(events), len(prototypes))
    return manifest


__all__ = ["ANIMALS", "FEATURES", "OLD_LABEL", "Stage1Config", "load_sources", "audit_sources", "normalize_ch8", "event_summary", "local_event_states", "balanced_pca", "legacy_reference", "run"]
