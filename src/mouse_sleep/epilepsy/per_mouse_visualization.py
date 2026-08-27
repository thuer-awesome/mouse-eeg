from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mouse_sleep.epilepsy.signal_processing import (
    EEGFilterConfig,
    filter_continuous_eeg,
    reconstruct_continuous_cache,
)


FONTS = ["Microsoft YaHei", "Noto Sans SC", "SimHei", "DejaVu Sans"]


def generate_per_mouse_clinical_figures(
    assignments: pd.DataFrame,
    baseline: pd.DataFrame,
    model_summary: pd.DataFrame,
    type_summary: pd.DataFrame,
    output_dir: str | Path,
    filter_config: EEGFilterConfig,
    dpi: int = 300,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
        outputs = _baseline_seizure_comparison(
            assignments, baseline, output_dir, filter_config, dpi
        )
        outputs.extend(
            _candidate_count_summary(model_summary, type_summary, output_dir, dpi)
        )
        for animal in sorted(assignments["animal_id"].astype(str).unique()):
            outputs.extend(
                _animal_candidate_atlas(
                    assignments,
                    model_summary,
                    type_summary,
                    animal,
                    output_dir,
                    filter_config,
                    dpi,
                )
            )
        return outputs
    finally:
        plt.rcParams.update(previous)


def _baseline_seizure_comparison(
    assignments: pd.DataFrame,
    baseline: pd.DataFrame,
    output_dir: Path,
    config: EEGFilterConfig,
    dpi: int,
) -> list[Path]:
    animals = sorted(assignments["animal_id"].astype(str).unique())
    fig, axes = plt.subplots(
        len(animals), 3, figsize=(12, 2.1 * len(animals) + 1.5), constrained_layout=False
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.06, hspace=0.65, wspace=0.22)
    for row_index, animal in enumerate(animals):
        seizure_rows = assignments.loc[assignments["animal_id"].astype(str).eq(animal)]
        baseline_rows = baseline.loc[baseline["animal_id"].astype(str).eq(animal)]
        shared_sessions = sorted(
            set(seizure_rows["session_id"].astype(str))
            & set(baseline_rows["session_id"].astype(str))
        )
        if shared_sessions:
            session = max(
                shared_sessions,
                key=lambda value: int(
                    seizure_rows["session_id"].astype(str).eq(value).sum()
                ),
            )
            seizure_rows = seizure_rows.loc[
                seizure_rows["session_id"].astype(str).eq(session)
            ]
            baseline_rows = baseline_rows.loc[
                baseline_rows["session_id"].astype(str).eq(session)
            ]
        baseline_row = _nearest_quantile_row(
            baseline_rows, "aggregate_robust_amplitude", 0.50
        )
        seizure_median = _nearest_quantile_row(
            seizure_rows, "aggregate_robust_amplitude", 0.50
        )
        seizure_high = _nearest_quantile_row(
            seizure_rows, "aggregate_robust_amplitude", 0.90
        )
        selected = [baseline_row, seizure_median, seizure_high]
        signals = [_load_filtered_window(row, config) for row in selected]
        span = max(_signal_span(signal, row) for signal, row in zip(signals, selected))
        titles = [
            "同 session 未标注参考段（距标注发作至少 10 秒）",
            "标注发作（中位幅值）",
            "标注发作（较高幅值 P90）",
        ]
        baseline_amplitude = max(float(baseline_row["aggregate_robust_amplitude"]), 1e-12)
        for column_index, (ax, row, signal, title) in enumerate(
            zip(axes[row_index], selected, signals, titles)
        ):
            _plot_fixed_trace(ax, signal, row, span)
            ratio = float(row["aggregate_robust_amplitude"]) / baseline_amplitude
            ax.set_title(f"{title}\n幅值为该基线的 {ratio:.2f} 倍", fontsize=8)
            if column_index == 0:
                ax.set_ylabel(f"{animal}\nCH7 / CH8", fontweight="bold")
    fig.suptitle(
        "同一只鼠、同一 session：未标注参考段与标注发作固定纵轴对照",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        f"连续信号零相位 {config.highpass_hz:g}-{config.lowpass_hz:g} Hz 带通 + "
        f"{config.notch_hz:g} Hz 陷波；每一行三个小图使用完全相同纵轴。",
        ha="center",
        fontsize=9,
    )
    return _save(fig, output_dir / "baseline_vs_seizure_fixed_scale", dpi)


def _candidate_count_summary(
    model_summary: pd.DataFrame,
    type_summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    table = model_summary.copy().sort_values("animal_id")
    fig, (ax, note) = plt.subplots(
        1, 2, figsize=(11, 5.5), gridspec_kw={"width_ratios": [1.55, 1.0]}, constrained_layout=True
    )
    positions = np.arange(len(table))
    ax.bar(
        positions - 0.18,
        table["stable_type_count"],
        width=0.36,
        color="#0072B2",
        label="保守统计粗分",
    )
    ax.bar(
        positions + 0.18,
        table["review_type_count"],
        width=0.36,
        color="#E69F00",
        label="全部保留供临床复核",
    )
    ax.set_xticks(positions, table["animal_id"])
    ax.set_ylabel("候选发作内波形形态数")
    ax.set_xlabel("小鼠")
    ax.set_ylim(0, max(8, int(table["review_type_count"].max()) + 1))
    ax.set_title("先逐鼠分析得到的形态数", loc="left", fontsize=14, fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    note.axis("off")
    note.text(0.0, 0.98, "如何读这张图", fontsize=12, fontweight="bold", va="top")
    lines = []
    for row in table.to_dict("records"):
        reliability = (
            "稳定"
            if row["review_stability_ok"]
            else "复核细分未通过稳定性门槛"
        )
        lines.append(
            f"{row['animal_id']}：保守粗分 {int(row['stable_type_count'])} 组；"
            f"保留复核 {int(row['review_type_count'])} 种候选波形（{reliability}）"
        )
    note.text(0.0, 0.84, "\n\n".join(lines), fontsize=9.2, va="top", linespacing=1.2)
    note.text(
        0.0,
        0.08,
        "这些是“发作内波形形态”，尚未经过跨鼠匹配和临床命名，不能直接称为癫痫亚型。",
        fontsize=9,
        color="#7A1F1F",
        va="bottom",
    )
    return _save(fig, output_dir / "per_mouse_candidate_counts", dpi)


def _animal_candidate_atlas(
    assignments: pd.DataFrame,
    model_summary: pd.DataFrame,
    type_summary: pd.DataFrame,
    animal: str,
    output_dir: Path,
    config: EEGFilterConfig,
    dpi: int,
) -> list[Path]:
    animal_rows = assignments.loc[assignments["animal_id"].astype(str).eq(animal)].copy()
    types = type_summary.loc[type_summary["animal_id"].astype(str).eq(animal)].sort_values(
        "candidate_cluster"
    )
    if types.empty:
        return []
    representatives: dict[str, pd.DataFrame] = {}
    signals: list[tuple[np.ndarray, pd.Series]] = []
    for label in types["candidate_label"].astype(str):
        candidates = animal_rows.loc[
            animal_rows["individual_review_smoothed_label"].astype(str).eq(label)
        ].copy()
        median_ratio = float(candidates["amplitude_vs_baseline"].median())
        candidates["_typicality"] = np.abs(
            np.log(
                np.maximum(candidates["amplitude_vs_baseline"], 1e-6)
                / max(median_ratio, 1e-6)
            )
        )
        candidates = candidates.sort_values(
            ["_typicality", "individual_review_probability"],
            ascending=[True, False],
            kind="stable",
        ).drop_duplicates("event_id")
        selected = candidates.head(3)
        representatives[label] = selected
        signals.extend((_load_filtered_window(row, config), row) for _, row in selected.iterrows())
    span = float(np.quantile([_signal_span(signal, row) for signal, row in signals], 0.90))
    span = max(span, 1.0)
    fig, axes = plt.subplots(
        len(types), 3, figsize=(12, max(4.5, 2.1 * len(types) + 1.4)), constrained_layout=False
    )
    if len(types) == 1:
        axes = np.asarray([axes])
    fig.subplots_adjust(left=0.15, right=0.98, top=0.88, bottom=0.07, hspace=0.75, wspace=0.25)
    model = model_summary.loc[model_summary["animal_id"].astype(str).eq(animal)].iloc[0]
    stable_count = int(model["stable_type_count"])
    review_is_stable = bool(model["review_stability_ok"])
    for row_index, type_row in enumerate(types.to_dict("records")):
        label = str(type_row["candidate_label"])
        selected = representatives[label]
        for column_index, ax in enumerate(axes[row_index]):
            if column_index >= len(selected):
                ax.axis("off")
                continue
            representative = selected.iloc[column_index]
            signal = _load_filtered_window(representative, config)
            _plot_fixed_trace(ax, signal, representative, span)
            ax.set_title(
                f"{_short_event(representative['event_id'])}\n"
                f"幅值/基线 {float(representative['amplitude_vs_baseline']):.2f} 倍",
                fontsize=7.5,
            )
            if column_index == 0:
                status = (
                    "该复核细分整体通过稳定性门槛"
                    if review_is_stable
                    else "探索候选，细分稳定性不足"
                )
                ax.set_ylabel(
                    f"{label}\n{status}\n"
                    f"{int(type_row['events'])} 个事件",
                    fontsize=8,
                    fontweight="bold",
                )
    fig.suptitle(
        f"{animal}：逐鼠独立发现的候选发作内波形形态",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.93,
        f"保守统计粗分为 {stable_count} 组；本图不删除候选，完整展示 {len(types)} 种待复核波形。"
        "每行来自三个不同事件，整张图使用同一纵轴。",
        ha="center",
        fontsize=9,
    )
    return _save(fig, output_dir / f"{animal}_candidate_waveform_atlas", dpi)


def _nearest_quantile_row(frame: pd.DataFrame, column: str, quantile: float) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    target = float(values.quantile(quantile))
    return frame.loc[(values - target).abs().idxmin()]


def _load_filtered_window(row: pd.Series, config: EEGFilterConfig) -> np.ndarray:
    continuous, stride_samples = _filtered_cache(
        str(row["cache_dir"]),
        config.highpass_hz,
        config.lowpass_hz,
        config.notch_hz,
        config.notch_quality_factor,
        config.bandpass_order,
    )
    start = int(row["cache_local_index"]) * stride_samples
    window_samples = int(round(float(row["sampling_rate_hz"]) * 4.0))
    return np.asarray(continuous[:, start : start + window_samples], dtype=np.float32)


@lru_cache(maxsize=6)
def _filtered_cache(
    cache_dir: str,
    highpass: float,
    lowpass: float,
    notch: float,
    notch_q: float,
    order: int,
) -> tuple[np.ndarray, int]:
    root = Path(cache_dir)
    metadata = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    rate = float(metadata["sampling_rate_hz"])
    stride_samples = int(round(rate * float(metadata["stride_seconds"])))
    signals = np.load(root / "signals.npy", mmap_mode="r")
    continuous = reconstruct_continuous_cache(signals, stride_samples)
    filtered = filter_continuous_eeg(
        continuous,
        rate,
        EEGFilterConfig(highpass, lowpass, notch, notch_q, order),
    )
    return filtered, stride_samples


def _signal_span(signal: np.ndarray, row: pd.Series) -> float:
    valid = np.asarray(
        [bool(row.get("ch7_valid", True)), bool(row.get("ch8_valid", True))]
    )
    values = signal[valid]
    return float(np.quantile(np.abs(values), 0.995)) if values.size else 1.0


def _plot_fixed_trace(
    ax: plt.Axes,
    signal: np.ndarray,
    row: pd.Series,
    span: float,
) -> None:
    rate = float(row["sampling_rate_hz"])
    time = np.arange(signal.shape[-1]) / rate
    valid = [bool(row.get("ch7_valid", True)), bool(row.get("ch8_valid", True))]
    offset = 2.6 * span
    for channel, (name, color) in enumerate((("CH7", "#0072B2"), ("CH8", "#D55E00"))):
        base = (1 - channel) * offset
        if valid[channel]:
            ax.plot(time, signal[channel] + base, color=color, lw=0.65)
        else:
            ax.plot(time, np.full_like(time, base), color="#AAAAAA", lw=0.7)
            ax.text(time[-1], base, "CH7 无效", ha="right", va="center", fontsize=6, color="#777777")
        ax.text(time[0], base + 0.72 * span, name, fontsize=6, color=color if valid[channel] else "#777777")
    ax.set_xlim(0, time[-1])
    ax.set_ylim(-1.25 * span, offset + 1.25 * span)
    ax.set_yticks([])
    ax.set_xlabel("时间（秒）", fontsize=7)
    ax.spines[["top", "right", "left"]].set_visible(False)


def _short_event(value: Any) -> str:
    return str(value).split(":")[-1]


def _save(fig: plt.Figure, root: Path, dpi: int) -> list[Path]:
    outputs: list[Path] = []
    for suffix, options in ((".png", {"dpi": dpi}), (".pdf", {})):
        path = root.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="white", **options)
        outputs.append(path)
    plt.close(fig)
    return outputs


__all__ = ["generate_per_mouse_clinical_figures"]
