from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score


DEFAULT_RUNS = (
    "eegnet82",
    "eegwavenet",
    "physio_no_aux",
    "physio_stats_aux",
)
EXTENDED_RUNS = DEFAULT_RUNS + (
    "physio_no_aux_mouse_balanced",
    "physio_stats_aux_mouse_balanced",
)
KEY_COLUMNS = (
    "animal_id",
    "session_id",
    "group_id",
    "start_time_ms",
    "end_time_ms",
    "label",
    "split",
)
BOOTSTRAP_METRICS = (
    "auprc",
    "auroc",
    "balanced_accuracy",
    "mcc",
    "macro_mouse_auprc",
    "macro_mouse_balanced_accuracy",
)


def summarize_mixed_binary_runs(
    run_root: str | Path,
    output_dir: str | Path | None = None,
    run_names: tuple[str, ...] = DEFAULT_RUNS,
    bootstrap_repetitions: int = 1000,
    random_seed: int = 20260814,
) -> Path:
    """Summarize frozen runs and run a paired animal/class-stratified group bootstrap."""

    if bootstrap_repetitions < 100:
        raise ValueError("bootstrap_repetitions must be at least 100")
    root = Path(run_root).resolve()
    output = Path(output_dir).resolve() if output_dir else root / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    runs = {name: _read_run(root / name) for name in run_names}
    aligned = _align_predictions(runs, split="test")

    overall_rows: list[dict[str, Any]] = []
    per_mouse_rows: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    for name, run in runs.items():
        metrics = run["metrics"]
        manifest = run["manifest"]
        thresholds[name] = float(metrics["threshold"])
        for split in ("val", "test"):
            window = metrics[split]["window"]
            mouse_values = list(metrics[split]["per_mouse_window"].values())
            overall_rows.append(
                {
                    "model": name,
                    "split": split,
                    "parameters": int(manifest["parameters"]),
                    "best_epoch": int(metrics["best_epoch"]),
                    "threshold": float(metrics["threshold"]),
                    **{metric: window[metric] for metric in (
                        "n_windows",
                        "positive_windows",
                        "accuracy",
                        "sensitivity",
                        "specificity",
                        "f1",
                        "balanced_accuracy",
                        "mcc",
                        "auroc",
                        "auprc",
                    )},
                    **{
                        f"macro_mouse_{metric}": float(
                            np.mean([float(values[metric]) for values in mouse_values])
                        )
                        for metric in (
                            "sensitivity",
                            "specificity",
                            "f1",
                            "balanced_accuracy",
                            "mcc",
                            "auroc",
                            "auprc",
                        )
                    },
                    "positive_group_sensitivity": metrics[split]["group"][
                        "positive_group_sensitivity"
                    ],
                    "false_positive_background_group_fraction": metrics[split][
                        "group"
                    ]["false_positive_background_group_fraction"],
                }
            )
            for animal, values in metrics[split]["per_mouse_window"].items():
                per_mouse_rows.append(
                    {
                        "model": name,
                        "split": split,
                        "animal_id": animal,
                        **values,
                    }
                )
    overall = pd.DataFrame(overall_rows)
    per_mouse = pd.DataFrame(per_mouse_rows)
    overall.to_csv(output / "overall_metrics.csv", index=False, encoding="utf-8-sig")
    per_mouse.to_csv(output / "per_mouse_metrics.csv", index=False, encoding="utf-8-sig")

    bootstrap = _paired_group_bootstrap(
        aligned,
        thresholds,
        repetitions=bootstrap_repetitions,
        random_seed=random_seed,
    )
    bootstrap_path = output / "paired_group_bootstrap.json"
    bootstrap_path.write_text(
        json.dumps(bootstrap, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    report = _render_report(overall, per_mouse, bootstrap)
    report_path = output / "REPORT_zh.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def summarize_physio_seed_repeats(
    run_roots: tuple[str | Path, ...], output_dir: str | Path
) -> Path:
    """Describe paired no-aux/aux results across a small set of training seeds."""

    if len(run_roots) < 2:
        raise ValueError("multi-seed reporting requires at least two run roots")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    mouse_rows: list[dict[str, Any]] = []
    metadata_hashes = set()
    for root_value in run_roots:
        root = Path(root_value).resolve()
        for model in ("physio_no_aux", "physio_stats_aux"):
            run = _read_run(root / model)
            metrics = run["metrics"]
            manifest = run["manifest"]
            seed = int(manifest["config"]["seed"])
            metadata_hashes.add(str(manifest["metadata_sha256"]))
            mouse_values = list(metrics["test"]["per_mouse_window"].values())
            rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "best_epoch": int(metrics["best_epoch"]),
                    "validation_auprc": float(metrics["val"]["window"]["auprc"]),
                    **{
                        f"test_{metric}": float(metrics["test"]["window"][metric])
                        for metric in (
                            "auprc",
                            "auroc",
                            "balanced_accuracy",
                            "mcc",
                            "sensitivity",
                            "specificity",
                        )
                    },
                    "macro_mouse_auprc": float(
                        np.mean([values["auprc"] for values in mouse_values])
                    ),
                    "macro_mouse_balanced_accuracy": float(
                        np.mean(
                            [values["balanced_accuracy"] for values in mouse_values]
                        )
                    ),
                }
            )
            for animal, values in metrics["test"]["per_mouse_window"].items():
                mouse_rows.append(
                    {
                        "seed": seed,
                        "model": model,
                        "animal_id": animal,
                        **values,
                    }
                )
    if len(metadata_hashes) != 1:
        raise ValueError("seed repeats do not share one frozen metadata hash")
    frame = pd.DataFrame(rows).sort_values(["seed", "model"], kind="stable")
    per_mouse = pd.DataFrame(mouse_rows).sort_values(
        ["seed", "model", "animal_id"], kind="stable"
    )
    expected = len(run_roots) * 2
    if len(frame) != expected or frame.groupby("seed")["model"].nunique().ne(2).any():
        raise ValueError("every seed must contain the paired Physio runs")
    frame.to_csv(output / "seed_metrics.csv", index=False, encoding="utf-8-sig")
    per_mouse.to_csv(
        output / "seed_per_mouse_metrics.csv", index=False, encoding="utf-8-sig"
    )

    value_columns = [
        column
        for column in frame.columns
        if column not in {"seed", "model", "best_epoch"}
    ]
    aggregate_rows = []
    for model, group in frame.groupby("model", sort=True):
        for metric in value_columns:
            values = group[metric].to_numpy(float)
            aggregate_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sample_sd": float(values.std(ddof=1)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "seeds": int(len(values)),
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output / "seed_aggregate.csv", index=False, encoding="utf-8-sig")

    wide = frame.pivot(index="seed", columns="model", values=value_columns)
    difference_rows = []
    for metric in value_columns:
        differences = (
            wide[(metric, "physio_stats_aux")]
            - wide[(metric, "physio_no_aux")]
        ).to_numpy(float)
        difference_rows.append(
            {
                "metric": metric,
                "mean_aux_minus_no_aux": float(differences.mean()),
                "sample_sd": float(differences.std(ddof=1)),
                "minimum": float(differences.min()),
                "maximum": float(differences.max()),
                "positive_seeds": int(np.sum(differences > 0)),
                "negative_seeds": int(np.sum(differences < 0)),
                "zero_seeds": int(np.sum(differences == 0)),
                "seeds": int(len(differences)),
            }
        )
    differences = pd.DataFrame(difference_rows)
    differences.to_csv(
        output / "paired_seed_differences.csv", index=False, encoding="utf-8-sig"
    )

    report_path = output / "PHYSIO_MULTI_SEED_REPORT_zh.md"
    report_path.write_text(
        _render_seed_report(frame, aggregate, differences), encoding="utf-8"
    )
    return report_path


def _read_run(path: Path) -> dict[str, Any]:
    required = (
        path / "completed.json",
        path / "metrics.json",
        path / "run_manifest.json",
        path / "predictions.parquet",
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete run {path}: {missing}")
    return {
        "metrics": json.loads((path / "metrics.json").read_text(encoding="utf-8")),
        "manifest": json.loads(
            (path / "run_manifest.json").read_text(encoding="utf-8")
        ),
        "predictions": pd.read_parquet(path / "predictions.parquet"),
    }


def _align_predictions(
    runs: Mapping[str, Mapping[str, Any]], split: str
) -> pd.DataFrame:
    aligned: pd.DataFrame | None = None
    for name, run in runs.items():
        frame = run["predictions"]
        frame = frame.loc[frame["split"].eq(split), [*KEY_COLUMNS, "seizure_probability"]]
        frame = frame.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)
        if frame.duplicated(list(KEY_COLUMNS)).any():
            raise ValueError(f"duplicate prediction keys in {name}")
        frame = frame.rename(columns={"seizure_probability": f"probability__{name}"})
        if aligned is None:
            aligned = frame
        else:
            before = len(aligned)
            aligned = aligned.merge(
                frame,
                on=list(KEY_COLUMNS),
                how="inner",
                validate="one_to_one",
            )
            if len(aligned) != before or len(frame) != before:
                raise ValueError(f"prediction rows are not identical across runs: {name}")
    if aligned is None or aligned.empty:
        raise ValueError("no predictions were found")
    return aligned


def _paired_group_bootstrap(
    frame: pd.DataFrame,
    thresholds: Mapping[str, float],
    repetitions: int,
    random_seed: int,
) -> dict[str, Any]:
    model_names = tuple(thresholds)
    labels = frame["label"].to_numpy(np.int8)
    animals = frame["animal_id"].astype(str).to_numpy()
    probabilities = {
        name: frame[f"probability__{name}"].to_numpy(float) for name in model_names
    }
    group_table = frame[["animal_id", "label", "group_id"]].drop_duplicates()
    group_rows = {
        str(group_id): indices.to_numpy(np.int64)
        for group_id, indices in frame.groupby("group_id", sort=False).groups.items()
    }
    strata = [
        group["group_id"].astype(str).to_numpy()
        for _, group in group_table.groupby(["animal_id", "label"], sort=True)
    ]
    if any(len(groups) == 0 for groups in strata):
        raise ValueError("every bootstrap stratum must contain an independent group")

    observed = {
        name: _metric_values(
            labels, probabilities[name], thresholds[name], animals
        )
        for name in model_names
    }
    distributions = {
        name: {metric: np.empty(repetitions, dtype=np.float64) for metric in BOOTSTRAP_METRICS}
        for name in model_names
    }
    rng = np.random.default_rng(random_seed)
    for repetition in range(repetitions):
        sampled_groups = [
            rng.choice(groups, size=len(groups), replace=True) for groups in strata
        ]
        indices = np.concatenate(
            [group_rows[str(group_id)] for groups in sampled_groups for group_id in groups]
        )
        sampled_labels = labels[indices]
        sampled_animals = animals[indices]
        for name in model_names:
            values = _metric_values(
                sampled_labels,
                probabilities[name][indices],
                thresholds[name],
                sampled_animals,
            )
            for metric in BOOTSTRAP_METRICS:
                distributions[name][metric][repetition] = values[metric]

    performance = {
        name: {
            metric: _distribution_summary(
                observed[name][metric], distributions[name][metric]
            )
            for metric in BOOTSTRAP_METRICS
        }
        for name in model_names
    }
    requested_contrasts = {
        "physio_no_aux_minus_eegwavenet": ("physio_no_aux", "eegwavenet"),
        "physio_stats_aux_minus_physio_no_aux": (
            "physio_stats_aux",
            "physio_no_aux",
        ),
        "physio_no_aux_mouse_balanced_minus_physio_no_aux": (
            "physio_no_aux_mouse_balanced",
            "physio_no_aux",
        ),
        "physio_stats_aux_mouse_balanced_minus_physio_no_aux_mouse_balanced": (
            "physio_stats_aux_mouse_balanced",
            "physio_no_aux_mouse_balanced",
        ),
    }
    contrasts = {}
    for contrast_name, (left, right) in requested_contrasts.items():
        if left not in distributions or right not in distributions:
            continue
        contrasts[contrast_name] = {}
        for metric in BOOTSTRAP_METRICS:
            difference = distributions[left][metric] - distributions[right][metric]
            observed_difference = observed[left][metric] - observed[right][metric]
            summary = _distribution_summary(observed_difference, difference)
            summary["bootstrap_probability_greater_than_zero"] = float(
                np.mean(difference > 0.0)
            )
            contrasts[contrast_name][metric] = summary
    return {
        "schema_version": 1,
        "method": "paired_cluster_bootstrap_stratified_by_mouse_and_binary_label",
        "resampling_unit": "independent_event_complex_or_background_block",
        "repetitions": repetitions,
        "random_seed": random_seed,
        "test_groups": int(frame["group_id"].nunique()),
        "test_windows": int(len(frame)),
        "conditioned_on": (
            "the six observed mice, the frozen split, one trained seed per model, "
            "and validation-selected checkpoints/thresholds"
        ),
        "performance": performance,
        "contrasts": contrasts,
    }


def _metric_values(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    animals: np.ndarray,
) -> dict[str, float]:
    result = _base_metric_values(labels, probabilities, threshold)
    per_mouse = [
        _base_metric_values(
            labels[animals == animal], probabilities[animals == animal], threshold
        )
        for animal in sorted(np.unique(animals))
    ]
    result["macro_mouse_auprc"] = float(
        np.mean([values["auprc"] for values in per_mouse])
    )
    result["macro_mouse_balanced_accuracy"] = float(
        np.mean([values["balanced_accuracy"] for values in per_mouse])
    )
    return result


def _base_metric_values(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = probabilities >= threshold
    positive = labels == 1
    negative = ~positive
    tp = int(np.sum(predictions & positive))
    tn = int(np.sum(~predictions & negative))
    fp = int(np.sum(predictions & negative))
    fn = int(np.sum(~predictions & positive))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "auprc": float(average_precision_score(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }


def _distribution_summary(observed: float, distribution: np.ndarray) -> dict[str, float]:
    lower, upper = np.quantile(distribution, [0.025, 0.975])
    return {
        "observed": float(observed),
        "bootstrap_median": float(np.median(distribution)),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def _render_report(
    overall: pd.DataFrame, per_mouse: pd.DataFrame, bootstrap: Mapping[str, Any]
) -> str:
    test = overall.loc[overall["split"].eq("test")].set_index("model")
    lines = [
        "# 六鼠混合癫痫二分类正式结果",
        "",
        "## 总体测试结果",
        "",
        "| 模型 | AUPRC | AUROC | Balanced accuracy | Sensitivity | Specificity | F1 | MCC | 鼠宏平均AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, row in test.iterrows():
        lines.append(
            f"| {model} | {row.auprc:.4f} | {row.auroc:.4f} | "
            f"{row.balanced_accuracy:.4f} | {row.sensitivity:.4f} | "
            f"{row.specificity:.4f} | {row.f1:.4f} | {row.mcc:.4f} | "
            f"{row.macro_mouse_auprc:.4f} |"
        )
    lines.extend(
        [
            "",
            "这些指标来自每个事件/背景组最多64窗的组均衡视图，不是完整连续监测。",
            "因此不报告false alarms/hour或检测延迟。",
            "",
            "## 预设配对比较",
            "",
            "95%区间来自按鼠和二分类标签分层、以独立事件复合体/背景块为单位的配对bootstrap。",
            "区间只反映当前六鼠测试组的抽样不确定性，不包含重新训练种子的变异。",
            "",
            "| 比较 | 指标 | 观察差值 | 95% bootstrap区间 | P(差值>0) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for contrast, metrics in bootstrap["contrasts"].items():
        for metric, values in metrics.items():
            lines.append(
                f"| {contrast} | {metric} | {values['observed']:.4f} | "
                f"[{values['ci95_lower']:.4f}, {values['ci95_upper']:.4f}] | "
                f"{values['bootstrap_probability_greater_than_zero']:.3f} |"
            )
    test_mouse = per_mouse.loc[per_mouse["split"].eq("test")]
    lines.extend(
        [
            "",
            "## 逐鼠限制",
            "",
            "E81和E82测试集各只有一个阳性事件复合组（各64个阳性窗），其指标方差很大。",
            "总体结果主要由窗口和事件更多的E50、E76、E77贡献；必须同时查看",
            "`per_mouse_metrics.csv`，不能把总体AUPRC解释为每只鼠均达到同一水平。",
            "",
            "Physio两模型逐鼠测试AUPRC：",
            "",
            "| 鼠 | 无统计辅助 | 统计辅助0.15 | 差值 |",
            "|---|---:|---:|---:|",
        ]
    )
    pivot = test_mouse.loc[
        test_mouse["model"].isin(["physio_no_aux", "physio_stats_aux"])
    ].pivot(index="animal_id", columns="model", values="auprc")
    for animal, row in pivot.iterrows():
        difference = row["physio_stats_aux"] - row["physio_no_aux"]
        lines.append(
            f"| {animal} | {row['physio_no_aux']:.4f} | "
            f"{row['physio_stats_aux']:.4f} | {difference:+.4f} |"
        )
    if {
        "physio_no_aux_mouse_balanced",
        "physio_stats_aux_mouse_balanced",
    }.issubset(set(test.index)):
        lines.extend(
            [
                "",
                "## 鼠贡献敏感性分析",
                "",
                "该分析把正负总损失保持相等，并在每一类内部按鼠×类别样本数的平方根反比调整。",
                "它是单一复合损失权重，不叠加WeightedRandomSampler。",
                "",
                "| 模型 | Pooled AUPRC | 鼠宏平均AUPRC | Pooled BA | 鼠宏平均BA |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for model in (
            "physio_no_aux",
            "physio_no_aux_mouse_balanced",
            "physio_stats_aux_mouse_balanced",
        ):
            row = test.loc[model]
            lines.append(
                f"| {model} | {row.auprc:.4f} | {row.macro_mouse_auprc:.4f} | "
                f"{row.balanced_accuracy:.4f} | "
                f"{row.macro_mouse_balanced_accuracy:.4f} |"
            )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- `physio_no_aux`与`physio_stats_aux`只有辅助权重不同，但当前只有一个训练随机种子。",
            "- pooled划分包含已见鼠，不能声称对新鼠泛化。",
            "- 21类发作形态未参与主标签、划分或这些结果。",
            "- AITRICS需要完整26窗状态序列，未混入独立窗口排名。",
            "",
        ]
    )
    return "\n".join(lines)


def _render_seed_report(
    frame: pd.DataFrame, aggregate: pd.DataFrame, differences: pd.DataFrame
) -> str:
    lines = [
        "# Physio统计辅助三种子配对结果",
        "",
        "每个种子使用同一冻结划分、相同初始化种子配对、相同batch顺序和验证规则；",
        "两模型唯一计划差异是统计辅助权重0与0.15。测试集不参与checkpoint或阈值选择。",
        "",
        "## 每种子结果",
        "",
        "| Seed | 模型 | Best epoch | Test AUPRC | AUROC | BA | MCC | 鼠宏平均AUPRC | 鼠宏平均BA |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.seed} | {row.model} | {row.best_epoch} | "
            f"{row.test_auprc:.4f} | {row.test_auroc:.4f} | "
            f"{row.test_balanced_accuracy:.4f} | {row.test_mcc:.4f} | "
            f"{row.macro_mouse_auprc:.4f} | "
            f"{row.macro_mouse_balanced_accuracy:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 三种子描述性汇总",
            "",
            "| 模型 | 指标 | 均值 | 样本SD | 范围 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    selected = aggregate.loc[
        aggregate["metric"].isin(
            [
                "test_auprc",
                "test_auroc",
                "test_balanced_accuracy",
                "test_mcc",
                "macro_mouse_auprc",
                "macro_mouse_balanced_accuracy",
            ]
        )
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.metric} | {row.mean:.4f} | "
            f"{row.sample_sd:.4f} | [{row.minimum:.4f}, {row.maximum:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## 辅助减无辅助的配对差值",
            "",
            "| 指标 | 平均差值 | 样本SD | 范围 | 正/负种子 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    selected_differences = differences.loc[
        differences["metric"].isin(
            [
                "test_auprc",
                "test_auroc",
                "test_balanced_accuracy",
                "test_mcc",
                "macro_mouse_auprc",
                "macro_mouse_balanced_accuracy",
            ]
        )
    ]
    for row in selected_differences.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.mean_aux_minus_no_aux:+.4f} | "
            f"{row.sample_sd:.4f} | [{row.minimum:+.4f}, {row.maximum:+.4f}] | "
            f"{row.positive_seeds}/{row.negative_seeds} |"
        )
    lines.extend(
        [
            "",
            "只有3个训练种子，因此不做正态性假设、t检验或显著性声明。",
            "这些重复衡量训练随机性；独立测试组抽样不确定性另见组级paired bootstrap报告。",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_RUNS",
    "EXTENDED_RUNS",
    "summarize_mixed_binary_runs",
    "summarize_physio_seed_repeats",
]
