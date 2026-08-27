from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


CHINESE_FONTS = ["Microsoft YaHei", "Noto Sans SC", "SimHei", "DejaVu Sans"]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
REVIEW_NAMES = {"R1": "候选波形 A", "R2": "候选波形 B", "R3": "候选波形 C", "R4": "候选波形 D"}
EVENT_NAMES = {"ER1": "探索组 I", "ER2": "探索组 II"}


def generate_clinical_review_figures(
    assignments: pd.DataFrame,
    review_summaries: dict[str, pd.DataFrame],
    event_assignments: pd.DataFrame,
    diagnostics: dict[str, Any],
    event_diagnostics: dict[str, Any],
    output_dir: str | Path,
    dpi: int = 300,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    previous = plt.rcParams.copy()
    plt.rcParams.update(
        {
            "font.sans-serif": CHINESE_FONTS,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 9,
        }
    )
    try:
        outputs: list[Path] = []
        outputs.extend(
            _clinical_summary(
                assignments,
                event_assignments,
                diagnostics,
                event_diagnostics,
                output_dir,
                dpi,
            )
        )
        outputs.extend(
            _cross_mouse_waveform_atlas(
                assignments, review_summaries, output_dir, dpi
            )
        )
        outputs.extend(_mouse_composition(assignments, output_dir, dpi))
        outputs.extend(
            _complete_event_groups(
                event_assignments, event_diagnostics, output_dir, dpi
            )
        )
        return outputs
    finally:
        plt.rcParams.update(previous)


def _clinical_summary(
    assignments: pd.DataFrame,
    events: pd.DataFrame,
    diagnostics: dict[str, Any],
    event_diagnostics: dict[str, Any],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=[0.34, 1.0, 1.15])
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.78,
        "六只小鼠癫痫波形分析：临床阅读总览",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )
    title_ax.text(
        0.5,
        0.22,
        "先区分“发作内波形状态”和“整次发作种类”，两者不能混为一谈",
        ha="center",
        va="center",
        fontsize=11,
        color="#444444",
    )

    _statement_panel(
        fig.add_subplot(grid[1, 0]),
        "数据",
        [
            f"6 只小鼠",
            f"{events['event_id'].nunique():,} 个完整标注发作区间",
            f"{len(assignments):,} 个不重叠 4 秒波形窗",
        ],
        "#4C78A8",
    )
    _statement_panel(
        fig.add_subplot(grid[1, 1]),
        "发作内波形状态",
        [
            f"统计稳定粗分：{diagnostics['global_cluster_count']} 类",
            f"临床待复核细分：{diagnostics['review_cluster_count']} 类（A-D）",
            "表示一次发作过程中出现的不同波形，不等于发作种类",
        ],
        "#59A14F",
    )
    _statement_panel(
        fig.add_subplot(grid[1, 2]),
        "整次发作种类",
        [
            f"严格主结果：{event_diagnostics['selected_components']} 类",
            f"探索性分组：{event_diagnostics['review_components']} 组",
            "探索性 2 组存在动物比例偏倚，暂不能命名为两种癫痫",
        ],
        "#E15759",
    )

    count_ax = fig.add_subplot(grid[2, :2])
    counts = events.groupby("animal_id").agg(
        完整发作区间=("event_id", "size"),
        短于4秒=("short_unresolved", "sum"),
        仅一个波形窗=("limited_context", "sum"),
    )
    count_ax.axis("off")
    table = count_ax.table(
        cellText=counts.reset_index().values,
        colLabels=["小鼠", *counts.columns.tolist()],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    for column in range(len(counts.columns) + 1):
        table[(0, column)].set_facecolor("#EAF0F6")
        table[(0, column)].set_text_props(fontweight="bold")
    count_ax.set_title("每只小鼠的完整发作区间", loc="left", fontsize=12, fontweight="bold")

    conclusion_ax = fig.add_subplot(grid[2, 2])
    conclusion_ax.axis("off")
    conclusion_ax.text(
        0.0,
        1.0,
        "目前可以告诉临床",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    conclusion_ax.text(
        0.0,
        0.83,
        "1. 六只小鼠均可见多种发作内波形形态。\n\n"
        "2. 这些形态可稳定粗分为 2 类，并可展开为 A-D 四个待复核形态。\n\n"
        "3. 把每个标注区间作为完整事件后，目前没有可靠证据证明存在多个独立发作种类。\n\n"
        "4. 下一步由临床对 A-D 代表波形命名，并确认长区间是否属于持续发作或连续状态。",
        va="top",
        fontsize=10,
        linespacing=1.25,
    )
    fig.text(
        0.5,
        0.01,
        "电压换算系数尚未取得；所有波形仍使用去直流后的原始采集单位（a.u.），不能标为 µV。",
        ha="center",
        fontsize=8.5,
        color="#7A1F1F",
    )
    return _save(fig, output_dir / "clinical_summary", dpi)


def _statement_panel(ax: plt.Axes, title: str, lines: list[str], color: str) -> None:
    ax.set_facecolor("#F7F8FA")
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.05, 0.88, title, fontsize=13, fontweight="bold", color=color, va="top")
    ax.text(0.05, 0.68, "\n\n".join(lines), fontsize=10, va="top", linespacing=1.15)


def _cross_mouse_waveform_atlas(
    assignments: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    review = assignments.copy()
    review["cluster"] = review["review_morphology_cluster"]
    review["label"] = review["review_morphology_label"].astype(str)
    review["probability"] = review["review_morphology_probability"]
    review["uncertain"] = review["review_morphology_uncertain"].astype(bool)
    summary = summaries["global_cluster_summary"].set_index("morphology_label")
    labels = sorted(review["label"].unique(), key=_numeric_label)
    fig = plt.figure(figsize=(13, 10.5), constrained_layout=False)
    grid = fig.add_gridspec(len(labels), 4, width_ratios=[1.15, 2.2, 2.2, 2.2])
    fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.06, hspace=0.75, wspace=0.32)
    for row_index, label in enumerate(labels):
        candidates = review.loc[review["label"].eq(label)].copy()
        representatives = _distinct_mouse_representatives(candidates, 3)
        info_ax = fig.add_subplot(grid[row_index, 0])
        info_ax.axis("off")
        row = summary.loc[label]
        status = "需排查动物偏倚" if "animal_limited" in str(row["clinical_review_status"]) else "通过自动支持筛查"
        info_ax.text(
            0.0,
            0.95,
            f"{REVIEW_NAMES.get(label, label)}\n（内部编号 {label}）",
            va="top",
            fontsize=12,
            fontweight="bold",
            color=COLORS[row_index % len(COLORS)],
        )
        info_ax.text(
            0.0,
            0.62,
            f"覆盖 {int(row['animals'])} 只小鼠\n"
            f"覆盖 {int(row['events'])} 个事件\n"
            f"主频中位数 {float(row['median_peak_frequency_hz']):.2f} Hz\n"
            f"{status}",
            va="top",
            fontsize=8.5,
            linespacing=1.35,
        )
        for column_index, (_, representative) in enumerate(representatives.iterrows(), 1):
            ax = fig.add_subplot(grid[row_index, column_index])
            signal = _load_signal(representative)
            _plot_clinical_trace(ax, signal, representative)
            ax.set_title(
                f"{representative['animal_id']} | {_short_event(representative['event_id'])}\n"
                f"置信度 {float(representative['probability']):.2f}",
                fontsize=8,
            )
    fig.suptitle(
        "A-D 四种待临床复核的发作内波形形态：跨小鼠代表波形",
        fontsize=16,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.925,
        "每行取自不同小鼠/不同事件；各小图纵向自适应，只比较形状，不比较绝对电压。A-D 不是已确认的发作种类。",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    return _save(fig, output_dir / "candidate_waveforms_A_to_D", dpi)


def _distinct_mouse_representatives(candidates: pd.DataFrame, count: int) -> pd.DataFrame:
    ranked = candidates.sort_values(
        ["uncertain", "probability", "review_morphology_distance"],
        ascending=[True, False, True],
        kind="stable",
    )
    per_mouse = ranked.drop_duplicates("animal_id")
    selected = per_mouse.head(count)
    if len(selected) < count:
        remaining = ranked.loc[~ranked.index.isin(selected.index)].drop_duplicates("event_id")
        selected = pd.concat([selected, remaining.head(count - len(selected))])
    return selected.head(count)


def _plot_clinical_trace(ax: plt.Axes, signal: np.ndarray, row: pd.Series) -> None:
    centered = signal - np.nanmedian(signal, axis=1, keepdims=True)
    valid = [bool(row.get("ch7_valid", True)), bool(row.get("ch8_valid", True))]
    valid_values = centered[np.asarray(valid, dtype=bool)]
    span = float(np.nanquantile(np.abs(valid_values), 0.995)) if valid_values.size else 1.0
    span = max(span, 1.0)
    offset = 2.6 * span
    rate = float(row["sampling_rate_hz"])
    time = np.arange(centered.shape[1]) / rate
    for index, (name, color) in enumerate((("CH7", "#0072B2"), ("CH8", "#D55E00"))):
        base = (1 - index) * offset
        if valid[index]:
            ax.plot(time, centered[index] + base, color=color, lw=0.65)
        else:
            ax.plot(time, np.full_like(time, base), color="#AAAAAA", lw=0.7)
            ax.text(time[-1], base, " CH7 无效", fontsize=6.5, color="#777777", va="center", ha="right")
        ax.text(time[0], base + 0.7 * span, name, fontsize=6.5, color=color if valid[index] else "#777777")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(-1.25 * span, offset + 1.25 * span)
    ax.set_yticks([])
    ax.set_xlabel("时间（秒）", fontsize=7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    scale = _nice_raw_scale(span)
    x0 = max(time[-1] - 1.15, time[0])
    y0 = -0.9 * span
    ax.plot([x0, min(x0 + 1.0, time[-1])], [y0, y0], color="black", lw=1.0)
    ax.plot([x0, x0], [y0, y0 + scale], color="black", lw=1.0)
    ax.text(x0 + 0.5, y0 - 0.13 * span, "1 秒", fontsize=6, ha="center", va="top")
    ax.text(x0 - 0.02, y0 + scale / 2, f"{scale:g} a.u.", fontsize=6, ha="right", va="center")


def _mouse_composition(assignments: pd.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
    labels = sorted(assignments["review_morphology_label"].astype(str).unique(), key=_numeric_label)
    animals = sorted(assignments["animal_id"].astype(str).unique())
    event_counts = (
        assignments.groupby(["animal_id", "event_id", "review_morphology_label"])
        .size()
        .rename("state_windows")
        .reset_index()
    )
    totals = assignments.groupby(["animal_id", "event_id"]).size().rename("event_windows")
    event_counts = event_counts.join(totals, on=["animal_id", "event_id"])
    event_counts["fraction"] = event_counts["state_windows"] / event_counts["event_windows"]
    prevalence = (
        event_counts.pivot_table(
            index=["animal_id", "event_id"],
            columns="review_morphology_label",
            values="fraction",
            fill_value=0.0,
        )
        .reindex(columns=labels, fill_value=0.0)
        .groupby(level="animal_id")
        .mean()
        .reindex(animals)
    )
    fig, ax = plt.subplots(figsize=(10, 5.7), constrained_layout=True)
    left = np.zeros(len(animals))
    for index, label in enumerate(labels):
        values = prevalence[label].to_numpy(float)
        ax.barh(animals, values, left=left, color=COLORS[index], label=f"{REVIEW_NAMES.get(label, label)}（{label}）")
        for row, (start, value) in enumerate(zip(left, values)):
            if value >= 0.06:
                ax.text(start + value / 2, row, f"{100 * value:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        left += values
    ax.set_xlim(0, 1)
    ax.set_xlabel("该小鼠每个完整事件内的平均波形组成（事件等权）")
    ax.set_ylabel("小鼠")
    ax.set_title(
        "每只小鼠都出现多种发作内波形形态",
        loc="left",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0.0,
        1.04,
        "这里显示 A-D 波形在完整事件内部的组成，不表示每只鼠有四种独立癫痫。",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
    )
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, output_dir / "mouse_candidate_waveform_composition", dpi)


def _complete_event_groups(
    events: pd.DataFrame,
    diagnostics: dict[str, Any],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    classified = events.loc[events["review_event_type_cluster"].ge(0)].copy()
    animals = sorted(events["animal_id"].astype(str).unique())
    labels = sorted(classified["review_event_type_label"].astype(str).unique(), key=_numeric_label)
    counts = (
        classified.groupby(["animal_id", "review_event_type_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=animals, columns=labels, fill_value=0)
    )
    fig = plt.figure(figsize=(11.5, 6.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.48, 1.0], width_ratios=[1.55, 1.0])
    conclusion = fig.add_subplot(grid[0, :])
    conclusion.axis("off")
    conclusion.text(
        0.0,
        0.90,
        "整次发作分类结论",
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    conclusion.text(
        0.0,
        0.54,
        f"严格主模型只支持 {diagnostics['selected_components']} 类完整发作。"
        f"下方 {diagnostics['review_components']} 个探索组仅用于挑选临床复核样本，不能命名为两种癫痫。",
        fontsize=11,
        va="top",
        color="#7A1F1F",
    )
    conclusion.text(
        0.0,
        0.20,
        "原因：探索组 II 的单鼠占比达到 76.4%，未通过预设的跨鼠比例门槛（≤70%）。",
        fontsize=9.5,
        va="top",
    )

    bar_ax = fig.add_subplot(grid[1, 0])
    bottom = np.zeros(len(animals))
    for index, label in enumerate(labels):
        values = counts[label].to_numpy(int)
        bar_ax.bar(animals, values, bottom=bottom, color=COLORS[index], label=f"{EVENT_NAMES.get(label, label)}（{label}）")
        for position, (base, value) in enumerate(zip(bottom, values)):
            if value > 0:
                bar_ax.text(position, base + value / 2, str(value), ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        bottom += values
    bar_ax.set_ylabel("完整标注发作区间数量")
    bar_ax.set_xlabel("小鼠")
    bar_ax.set_title("探索性两组在各小鼠中的数量", loc="left", fontsize=12, fontweight="bold")
    bar_ax.legend(frameon=False)
    bar_ax.spines[["top", "right"]].set_visible(False)

    note_ax = fig.add_subplot(grid[1, 1])
    note_ax.axis("off")
    note_ax.text(0.0, 0.96, "事件级质量标记", fontsize=12, fontweight="bold", va="top")
    note_ax.text(
        0.0,
        0.82,
        f"短于 4 秒、暂不分类：{diagnostics['short_unresolved_events']} 个\n\n"
        f"仅有一个 4 秒窗、信息有限：{diagnostics['limited_context_events']} 个\n\n"
        f"可进入平衡事件发现：{diagnostics['eligible_training_events']} 个\n\n"
        "临床复核重点：\n"
        "• 探索组 II 中来自不同小鼠的典型完整波形\n"
        "• 数小时长区间是否为持续状态而非单次发作\n"
        "• A-D 波形是否只是同一发作的阶段演化",
        va="top",
        fontsize=9.5,
        linespacing=1.25,
    )
    return _save(fig, output_dir / "complete_event_clinical_interpretation", dpi)


def _load_signal(row: pd.Series) -> np.ndarray:
    signals = np.load(Path(str(row["cache_dir"])) / "signals.npy", mmap_mode="r")
    return np.asarray(signals[int(row["cache_local_index"])], dtype=np.float32)


def _short_event(value: Any) -> str:
    text = str(value)
    return text.split(":")[-1]


def _numeric_label(value: Any) -> int:
    digits = "".join(character for character in str(value) if character.isdigit())
    return int(digits) if digits else 10_000


def _nice_raw_scale(span: float) -> float:
    exponent = 10 ** np.floor(np.log10(max(span, 1e-12)))
    candidates = exponent * np.asarray([0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    return float(candidates[np.argmin(np.abs(candidates - 0.6 * span))])


def _save(fig: plt.Figure, root: Path, dpi: int) -> list[Path]:
    outputs: list[Path] = []
    for suffix, options in ((".png", {"dpi": dpi}), (".pdf", {})):
        path = root.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="white", **options)
        outputs.append(path)
    plt.close(fig)
    return outputs


__all__ = ["generate_clinical_review_figures"]
