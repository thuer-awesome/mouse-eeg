from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from mouse_sleep.epilepsy.signal_processing import EEGFilterConfig


FONTS = ["Microsoft YaHei", "Noto Sans SC", "SimHei", "DejaVu Sans"]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def run_statistical_audit(
    labeled_windows: pd.DataFrame,
    reference_windows: pd.DataFrame,
    event_components: pd.DataFrame,
    output_dir: str | Path,
    filter_config: EEGFilterConfig,
    random_seed: int = 20260811,
    dpi: int = 300,
) -> dict[str, pd.DataFrame]:
    """Create a cluster-free, event-block-aware statistical audit."""

    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    labeled = _add_physical_feature_aliases(labeled_windows)
    reference = _add_physical_feature_aliases(reference_windows)
    event_statistics = _event_statistics(labeled, reference, event_components)
    phase_statistics = _event_phase_statistics(labeled, reference)
    reference_statistics = _reference_block_statistics(reference)
    comparison = _reference_event_comparison(
        event_statistics, reference_statistics, random_seed
    )
    filter_qc = _filter_qc(labeled)
    channel_qc = _channel_qc(labeled, event_components)
    duration = _duration_summary(event_components)
    anomaly = _event_anomaly_flags(event_statistics)
    outputs = {
        "event_statistics": event_statistics,
        "event_phase_statistics": phase_statistics,
        "reference_block_statistics": reference_statistics,
        "reference_vs_labeled_summary": comparison,
        "filter_qc": filter_qc,
        "channel_qc": channel_qc,
        "duration_summary": duration,
        "event_anomaly_flags": anomaly,
    }
    for name, frame in outputs.items():
        frame.to_csv(tables / f"{name}.csv", index=False, encoding="utf-8-sig")
    previous = plt.rcParams.copy()
    plt.rcParams.update(
        {
            "font.sans-serif": FONTS,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "font.size": 9,
        }
    )
    try:
        _plot_filter_qc(filter_qc, figures, dpi)
        _plot_reference_amplitude(event_statistics, figures, dpi)
        _plot_duration(event_components, figures, dpi)
        _plot_bilateral(event_statistics, figures, dpi)
        _plot_effect_heatmap(comparison, figures, dpi)
        _plot_all_event_maps(event_statistics, figures, dpi)
        _plot_phase_evolution(phase_statistics, figures, dpi)
    finally:
        plt.rcParams.update(previous)
    _write_report(
        output_dir,
        outputs,
        event_components,
        labeled,
        reference,
        filter_config,
    )
    return outputs


def _add_physical_feature_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for channel in ("ch7", "ch8"):
        for suffix in (
            "robust_amplitude",
            "rms",
            "peak_to_peak",
            "line_length",
            "total_power",
        ):
            source = f"{channel}_log_{suffix}"
            if source in result:
                result[f"{channel}_{suffix}"] = np.expm1(
                    pd.to_numeric(result[source], errors="coerce")
                )
    return result


def _reference_median_lookup(
    reference: pd.DataFrame,
    column: str,
) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    session = (
        reference.groupby(["animal_id", "session_id"])[column].median().to_dict()
    )
    animal = reference.groupby("animal_id")[column].median().to_dict()
    return session, animal


def _event_statistics(
    labeled: pd.DataFrame,
    reference: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    ratio_features = [
        "aggregate_robust_amplitude",
        "aggregate_line_length",
        "ch7_robust_amplitude",
        "ch8_robust_amplitude",
        "ch7_line_length",
        "ch8_line_length",
    ]
    lookups = {
        column: _reference_median_lookup(reference, column)
        for column in ratio_features
        if column in labeled and column in reference
    }
    event_channels = events.set_index("event_id")["channels"].astype(str).to_dict()
    rows: list[dict[str, Any]] = []
    for event_id, group in labeled.groupby("event_id", sort=True):
        animal = str(group["animal_id"].iloc[0])
        session = str(group["session_id"].iloc[0])
        row: dict[str, Any] = {
            "animal_id": animal,
            "session_id": session,
            "event_id": str(event_id),
            "annotation_channels": event_channels.get(str(event_id), "unknown"),
            "start_time_ms": float(group["event_start_ms"].iloc[0]),
            "end_time_ms": float(group["event_end_ms"].iloc[0]),
            "duration_s": float(group["event_duration_s"].iloc[0]),
            "sampled_windows": int(len(group)),
            "ch7_valid_fraction": float(group["ch7_valid"].mean()),
            "ch8_valid_fraction": float(group["ch8_valid"].mean()),
            "median_peak_frequency_hz": float(
                group["aggregate_peak_frequency_hz"].median()
            ),
            "median_spectral_entropy": float(
                group["aggregate_spectral_entropy"].median()
            ),
            "median_relative_delta": float(group["relative_delta"].median()),
            "median_relative_theta": float(group["relative_theta"].median()),
            "median_relative_beta": float(group["relative_beta"].median()),
            "median_channel_correlation": float(group["channel_correlation"].median()),
            "median_spectral_similarity": float(
                group["spectral_cosine_similarity"].median()
            ),
        }
        for column, (session_lookup, animal_lookup) in lookups.items():
            value = float(group[column].median())
            baseline = float(
                session_lookup.get((animal, session), animal_lookup.get(animal, np.nan))
            )
            row[f"median_{column}"] = value
            row[f"reference_{column}"] = baseline
            row[f"ratio_{column}"] = value / max(baseline, 1e-12)
            row[f"high_{column}_window_fraction"] = float(
                (group[column] > 1.5 * baseline).mean()
            )
        # A hardware rail can contain finite numbers while still being wholly
        # non-physiological. Remove channel summaries whenever fewer than half
        # of the source windows pass channel QC (notably E77 CH7).
        for channel in ("ch7", "ch8"):
            if row[f"{channel}_valid_fraction"] < 0.5:
                for key in list(row):
                    if (
                        key.startswith(f"median_{channel}_")
                        or key.startswith(f"reference_{channel}_")
                        or key.startswith(f"ratio_{channel}_")
                        or key.startswith(f"high_{channel}_")
                    ):
                        row[key] = np.nan
        if min(row["ch7_valid_fraction"], row["ch8_valid_fraction"]) < 0.5:
            row["median_channel_correlation"] = np.nan
            row["median_spectral_similarity"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _event_phase_statistics(
    labeled: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    session_lookup, animal_lookup = _reference_median_lookup(
        reference, "aggregate_robust_amplitude"
    )
    rows: list[dict[str, Any]] = []
    for (animal, session, event_id, decile), group in labeled.groupby(
        ["animal_id", "session_id", "event_id", "event_decile"], sort=True
    ):
        baseline = float(
            session_lookup.get(
                (str(animal), str(session)), animal_lookup.get(str(animal), np.nan)
            )
        )
        amplitude = float(group["aggregate_robust_amplitude"].median())
        rows.append(
            {
                "animal_id": str(animal),
                "session_id": str(session),
                "event_id": str(event_id),
                "event_decile": int(decile),
                "windows": int(len(group)),
                "median_amplitude_vs_reference": amplitude / max(baseline, 1e-12),
                "median_peak_frequency_hz": float(
                    group["aggregate_peak_frequency_hz"].median()
                ),
                "median_line_length": float(group["aggregate_line_length"].median()),
                "median_channel_correlation": float(
                    group["channel_correlation"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def _reference_block_statistics(reference: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "aggregate_robust_amplitude",
        "aggregate_line_length",
        "ch7_robust_amplitude",
        "ch8_robust_amplitude",
        "ch7_line_length",
        "ch8_line_length",
        "aggregate_peak_frequency_hz",
        "aggregate_spectral_entropy",
        "relative_delta",
        "relative_theta",
        "relative_beta",
        "channel_correlation",
    ]
    available = [column for column in columns if column in reference]
    return (
        reference.groupby(["animal_id", "session_id", "cache_dir"], sort=True)[
            available
        ]
        .median()
        .reset_index()
    )


def _reference_event_comparison(
    event_statistics: pd.DataFrame,
    reference_blocks: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    features = {
        "aggregate_robust_amplitude": "median_aggregate_robust_amplitude",
        "aggregate_line_length": "median_aggregate_line_length",
        "ch7_robust_amplitude": "median_ch7_robust_amplitude",
        "ch8_robust_amplitude": "median_ch8_robust_amplitude",
        "aggregate_peak_frequency_hz": "median_peak_frequency_hz",
        "aggregate_spectral_entropy": "median_spectral_entropy",
        "relative_delta": "median_relative_delta",
        "relative_theta": "median_relative_theta",
        "relative_beta": "median_relative_beta",
        "channel_correlation": "median_channel_correlation",
    }
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for animal in sorted(event_statistics["animal_id"].astype(str).unique()):
        event_group = event_statistics.loc[
            event_statistics["animal_id"].astype(str).eq(animal)
        ]
        reference_group = reference_blocks.loc[
            reference_blocks["animal_id"].astype(str).eq(animal)
        ]
        for reference_column, event_column in features.items():
            if reference_column not in reference_group or event_column not in event_group:
                continue
            event_values = pd.to_numeric(event_group[event_column], errors="coerce").dropna().to_numpy(float)
            reference_values = pd.to_numeric(reference_group[reference_column], errors="coerce").dropna().to_numpy(float)
            if len(event_values) < 2 or len(reference_values) < 2:
                continue
            u = float(mannwhitneyu(event_values, reference_values, alternative="two-sided").statistic)
            cliffs_delta = 2.0 * u / (len(event_values) * len(reference_values)) - 1.0
            ratios = []
            for _ in range(1000):
                event_sample = rng.choice(event_values, len(event_values), replace=True)
                reference_sample = rng.choice(
                    reference_values, len(reference_values), replace=True
                )
                ratios.append(
                    float(
                        np.median(event_sample)
                        / max(np.median(reference_sample), 1e-12)
                    )
                )
            rows.append(
                {
                    "animal_id": animal,
                    "feature": reference_column,
                    "event_blocks": int(len(event_values)),
                    "reference_blocks": int(len(reference_values)),
                    "event_median": float(np.median(event_values)),
                    "reference_median": float(np.median(reference_values)),
                    "median_ratio": float(
                        np.median(event_values)
                        / max(np.median(reference_values), 1e-12)
                    ),
                    "bootstrap_ratio_ci_low": float(np.quantile(ratios, 0.025)),
                    "bootstrap_ratio_ci_high": float(np.quantile(ratios, 0.975)),
                    "cliffs_delta": cliffs_delta,
                }
            )
    return pd.DataFrame(rows)


def _filter_qc(labeled: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for animal, group in labeled.groupby("animal_id", sort=True):
        raw = float(group["raw_relative_line_noise_48_52"].median())
        filtered = float(group["relative_line_noise_48_52"].median())
        rows.append(
            {
                "animal_id": str(animal),
                "raw_48_52_relative_power_median": raw,
                "filtered_48_52_relative_power_median": filtered,
                "attenuation_ratio": filtered / max(raw, 1e-12),
                "attenuation_db": 10.0 * np.log10(max(filtered, 1e-12) / max(raw, 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def _channel_qc(labeled: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    channel_counts = pd.crosstab(events["animal_id"], events["channels"])
    rows: list[dict[str, Any]] = []
    for animal, group in labeled.groupby("animal_id", sort=True):
        row = {
            "animal_id": str(animal),
            "ch7_valid_fraction": float(group["ch7_valid"].mean()),
            "ch8_valid_fraction": float(group["ch8_valid"].mean()),
            "median_channel_correlation": float(group["channel_correlation"].median()),
            "median_spectral_similarity": float(
                group["spectral_cosine_similarity"].median()
            ),
        }
        for state in ("CH7", "CH8", "CH7;CH8"):
            row[f"annotation_events_{state.replace(';', '_')}"] = int(
                channel_counts.loc[animal, state]
                if animal in channel_counts.index and state in channel_counts
                else 0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _duration_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for animal, group in events.groupby("animal_id", sort=True):
        duration = group["duration_s"].to_numpy(float)
        rows.append(
            {
                "animal_id": str(animal),
                "events": int(len(group)),
                "median_duration_s": float(np.median(duration)),
                "q25_duration_s": float(np.quantile(duration, 0.25)),
                "q75_duration_s": float(np.quantile(duration, 0.75)),
                "maximum_duration_s": float(np.max(duration)),
                "events_under_4s": int(np.sum(duration < 4.0)),
                "events_over_300s": int(np.sum(duration > 300.0)),
                "events_over_1h": int(np.sum(duration > 3600.0)),
                "total_labeled_hours": float(duration.sum() / 3600.0),
            }
        )
    return pd.DataFrame(rows)


def _event_anomaly_flags(events: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()
    result["short_under_4s"] = result["duration_s"].lt(4.0)
    result["long_over_300s"] = result["duration_s"].gt(300.0)
    result["extreme_over_1h"] = result["duration_s"].gt(3600.0)
    result["low_amplitude_vs_reference"] = result[
        "ratio_aggregate_robust_amplitude"
    ].lt(1.0)
    result["high_amplitude_vs_reference"] = result[
        "ratio_aggregate_robust_amplitude"
    ].ge(1.5)
    result["single_valid_channel"] = (
        result[["ch7_valid_fraction", "ch8_valid_fraction"]].gt(0.5).sum(axis=1)
        < 2
    )
    return result


def _plot_filter_qc(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    positions = np.arange(len(frame))
    width = 0.36
    ax.bar(
        positions - width / 2,
        frame["raw_48_52_relative_power_median"],
        width,
        color="#777777",
        label="滤波前 48–52 Hz 相对功率",
    )
    ax.bar(
        positions + width / 2,
        frame["filtered_48_52_relative_power_median"],
        width,
        color="#0072B2",
        label="50 Hz 陷波后残余",
    )
    ax.set_yscale("log")
    ax.set_xticks(positions, frame["animal_id"])
    ax.set_ylabel("48–52 Hz 相对功率（对数坐标）")
    ax.set_title("连续重建后滤波与 50 Hz 陷波质控", loc="left", fontsize=14, fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, output / "01_filter_qc", dpi)


def _plot_reference_amplitude(events: pd.DataFrame, output: Path, dpi: int) -> None:
    animals = sorted(events["animal_id"].astype(str).unique())
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for ax, animal, color in zip(axes.ravel(), animals, COLORS):
        group = events.loc[events["animal_id"].astype(str).eq(animal)]
        values = group["ratio_aggregate_robust_amplitude"].clip(upper=10)
        ax.hist(values, bins=30, color=color, alpha=0.85)
        ax.axvline(1.0, color="black", ls="--", lw=1, label="未标注参考中位数")
        ax.axvline(1.5, color="#D55E00", ls=":", lw=1, label="参考的 1.5 倍")
        ax.set_title(f"{animal}（{len(group)} 个完整事件）", fontweight="bold")
        ax.set_xlabel("事件中位幅值 / 同 session 未标注参考中位幅值")
        ax.set_ylabel("事件数")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "每个完整标注事件相对于未标注参考段的滤波后幅值",
        fontsize=15,
        fontweight="bold",
    )
    _save(fig, output / "02_event_amplitude_vs_reference", dpi)


def _plot_duration(events: pd.DataFrame, output: Path, dpi: int) -> None:
    animals = sorted(events["animal_id"].astype(str).unique())
    values = [
        events.loc[events["animal_id"].astype(str).eq(animal), "duration_s"].clip(lower=0.1)
        for animal in animals
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    boxes = ax.boxplot(values, tick_labels=animals, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_yscale("log")
    ax.axhline(300, color="#D55E00", ls="--", lw=1, label="5 分钟")
    ax.axhline(3600, color="#CC79A7", ls=":", lw=1, label="1 小时")
    ax.set_ylabel("完整标注区间持续时间（秒，对数坐标）")
    ax.set_xlabel("小鼠")
    ax.set_title("完整标注区间持续时间分布", loc="left", fontsize=14, fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, output / "03_event_duration", dpi)


def _plot_bilateral(events: pd.DataFrame, output: Path, dpi: int) -> None:
    animals = sorted(events["animal_id"].astype(str).unique())
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    scatter = None
    for ax, animal, color in zip(axes.ravel(), animals, COLORS):
        group = events.loc[
            events["animal_id"].astype(str).eq(animal)
            & events["ratio_ch7_robust_amplitude"].notna()
            & events["ratio_ch8_robust_amplitude"].notna()
        ]
        if group.empty:
            ax.text(
                0.5,
                0.5,
                "No valid bilateral events",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#666666",
            )
            ax.set_title(animal, fontweight="bold")
            ax.set_axis_off()
            continue
        scatter = ax.scatter(
            group["ratio_ch7_robust_amplitude"].clip(upper=8),
            group["ratio_ch8_robust_amplitude"].clip(upper=8),
            c=group["median_channel_correlation"],
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            s=12,
            alpha=0.65,
            linewidths=0,
        )
        ax.axvline(1, color="#777777", lw=0.8, ls="--")
        ax.axhline(1, color="#777777", lw=0.8, ls="--")
        ax.set_title(animal, fontweight="bold")
        ax.set_xlabel("CH7 幅值 / 参考")
        ax.set_ylabel("CH8 幅值 / 参考")
        ax.spines[["top", "right"]].set_visible(False)
    if scatter is None:
        _save(fig, output / "04_bilateral_amplitude_sync", dpi)
        return
    colorbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.75)
    colorbar.set_label("双通道相关系数")
    fig.suptitle("每个完整事件的双通道幅值与同步性", fontsize=15, fontweight="bold")
    _save(fig, output / "04_bilateral_amplitude_sync", dpi)


def _plot_effect_heatmap(comparison: pd.DataFrame, output: Path, dpi: int) -> None:
    features = [
        "aggregate_robust_amplitude",
        "aggregate_line_length",
        "aggregate_peak_frequency_hz",
        "aggregate_spectral_entropy",
        "relative_delta",
        "relative_theta",
        "relative_beta",
        "channel_correlation",
    ]
    animals = sorted(comparison["animal_id"].astype(str).unique())
    table = (
        comparison.pivot(index="animal_id", columns="feature", values="cliffs_delta")
        .reindex(index=animals, columns=features)
    )
    labels = ["幅值", "线长", "主频", "谱熵", "δ", "θ", "β", "双侧相关"]
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    image = ax.imshow(table.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(animals)), animals)
    for row in range(len(animals)):
        for column in range(len(labels)):
            value = table.iloc[row, column]
            if np.isfinite(value):
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
    colorbar.set_label("Cliff's δ：正值表示标注事件更高")
    ax.set_title("标注事件与未标注参考块的效应方向", loc="left", fontsize=14, fontweight="bold")
    _save(fig, output / "05_reference_effect_heatmap", dpi)


def _plot_all_event_maps(events: pd.DataFrame, output: Path, dpi: int) -> None:
    animals = sorted(events["animal_id"].astype(str).unique())
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2), constrained_layout=True)
    for ax, animal in zip(axes.ravel(), animals):
        group = events.loc[events["animal_id"].astype(str).eq(animal)]
        sizes = 8 + 10 * np.log10(group["duration_s"].clip(lower=1))
        scatter = ax.scatter(
            group["ratio_aggregate_robust_amplitude"].clip(upper=10),
            group["median_peak_frequency_hz"],
            c=group["median_channel_correlation"],
            s=sizes,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            alpha=0.55,
            linewidths=0,
        )
        ax.axvline(1, color="#777777", lw=0.8, ls="--")
        ax.set_title(f"{animal}：全部 {len(group)} 个事件", fontweight="bold")
        ax.set_xlabel("幅值 / 未标注参考")
        ax.set_ylabel("主频（Hz）")
        ax.spines[["top", "right"]].set_visible(False)
    colorbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.75)
    colorbar.set_label("双通道相关系数；点大小表示持续时间")
    fig.suptitle(
        "所有完整标注事件的连续特征分布（不做聚类、不指定类别数）",
        fontsize=15,
        fontweight="bold",
    )
    _save(fig, output / "06_all_event_continuous_map", dpi)


def _plot_phase_evolution(phase: pd.DataFrame, output: Path, dpi: int) -> None:
    animals = sorted(phase["animal_id"].astype(str).unique())
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for ax, animal, color in zip(axes.ravel(), animals, COLORS):
        group = phase.loc[phase["animal_id"].astype(str).eq(animal)]
        summary = group.groupby("event_decile")["median_amplitude_vs_reference"].agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        x = (summary.index.to_numpy() + 0.5) * 10
        ax.plot(x, summary["median"], color=color, lw=1.6)
        ax.fill_between(x, summary["q25"], summary["q75"], color=color, alpha=0.22)
        ax.axhline(1, color="#777777", ls="--", lw=0.8)
        ax.set_title(animal, fontweight="bold")
        ax.set_xlabel("事件内相对进程（%）")
        ax.set_ylabel("幅值 / 未标注参考")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "完整标注事件内部的幅值演化：中位数与四分位范围",
        fontsize=15,
        fontweight="bold",
    )
    _save(fig, output / "07_event_phase_evolution", dpi)


def _write_report(
    output: Path,
    tables: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    labeled: pd.DataFrame,
    reference: pd.DataFrame,
    config: EEGFilterConfig,
) -> None:
    duration = tables["duration_summary"]
    channel = tables["channel_qc"]
    amplitude = tables["reference_vs_labeled_summary"]
    amplitude = amplitude.loc[
        amplitude["feature"].eq("aggregate_robust_amplitude")
    ]
    lines = [
        "# 癫痫脑电第一阶段统计审计",
        "",
        "## 边界",
        "",
        "本报告不做聚类、不指定发作种类数，也不训练识别模型。所有未标注窗口仅称为参考/候选间期，未经临床确认不能称为正常基线。",
        "",
        "## 预处理",
        "",
        f"- 连续重建后零相位 {config.highpass_hz:g}-{config.lowpass_hz:g} Hz 带通。",
        f"- {config.notch_hz:g} Hz 陷波，Q={config.notch_quality_factor:g}。",
        "- 先滤波连续 shard，再切 4 秒窗，避免逐窗滤波边缘伪影。",
        "- CH7/CH8 均作为 EEG；E77 的 CH7 为硬件轨故障，只能使用 CH8。",
        "",
        "## 数据",
        "",
        f"- 6 只小鼠，{len(events):,} 个跨通道重叠并集事件，{len(labeled):,} 个不重叠标注窗。",
        f"- 抽取 {len(reference):,} 个距标注至少 10 秒、互不重叠的未标注参考窗。",
        "",
        "## 逐鼠持续时间",
        "",
        _markdown_table(duration),
        "",
        "## 通道质量与标注通道",
        "",
        _markdown_table(channel),
        "",
        "## 标注事件幅值相对未标注参考",
        "",
        _markdown_table(amplitude),
        "",
        "## 必须临床确认",
        "",
        "- E82 的多数标注事件滤波后幅值低于所抽取的未标注参考段，需要复核标注时间、参考段是否真为间期以及临床显示设置。",
        "- 超过 5 分钟或 1 小时的区间可能是持续状态、连续分段标注或长时间异常背景，不能等同于普通单次发作。",
        "- 标签文件中的 CH7/CH8 表示在哪一路画了标记，不证明另一侧没有放电。",
        "- 取得设备 ADC-to-µV 换算前，幅值只能报告为原始采集单位及同 session 相对倍数。",
    ]
    (output / "FIRST_STAGE_STATISTICAL_AUDIT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_无数据_"
    display = frame.copy()
    for column in display:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4g}"
            )
    header = "| " + " | ".join(map(str, display.columns)) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _save(fig: plt.Figure, root: Path, dpi: int) -> None:
    for suffix, options in ((".png", {"dpi": dpi}), (".pdf", {})):
        fig.savefig(root.with_suffix(suffix), bbox_inches="tight", facecolor="white", **options)
    plt.close(fig)


__all__ = ["run_statistical_audit"]
