from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from mouse_sleep.epilepsy.cross_mouse_waveform_types import FEATURE_BLOCKS


OKABE_ITO = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)

MOUSE_MARKERS = {
    "E50": "o",
    "E76": "s",
    "E77": "^",
    "E79": "D",
    "E81": "P",
    "E82": "X",
}

BLOCK_LABELS = {
    "raw_shape": "完整波形形状",
    "amplitude": "相对幅值",
    "shape": "时域统计",
    "frequency": "频域",
    "bilateral": "双侧一致性",
}

FEATURE_LABELS = {
    "log_amplitude_vs_baseline": "相对基线幅值",
    "aggregate_hjorth_mobility": "Hjorth 移动性",
    "theta_delta_ratio": "theta/delta 比值",
    "abs_line_length_log_ratio": "双侧线长差",
    "channel_correlation": "双侧相关性",
}


def generate_clinical_cross_mouse_figures(
    result: Any,
    output_dir: str | Path,
    dpi: int = 300,
) -> list[Path]:
    """Redraw figures 2/4/5/7/8/9 from an existing formal result."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _apply_style()
    outputs: list[Path] = []
    outputs += plot_common_category_space(
        result.common_space_coordinates,
        result.pairwise_evidence,
        result.category_prototypes,
        root / "figure_2_cross_mouse_21_category_map",
        dpi,
    )
    outputs += plot_interpretable_category_profile(
        result.category_prototypes,
        result.representative_dimensions,
        root / "figure_4_interpretable_category_dimensions",
        dpi,
    )
    outputs += plot_category_count_and_evidence(
        result.local_inventory,
        result.type_registry,
        result.pairwise_evidence,
        root / "figure_5_category_count_and_evidence",
        dpi,
    )
    outputs += plot_candidate_evidence_matrix(
        result.pairwise_evidence,
        root / "figure_7_strongest_cross_mouse_candidate_evidence",
        dpi,
    )
    outputs += plot_within_mouse_pair_summary(
        result.within_mouse_pair_audit,
        result.local_inventory,
        root / "figure_8_within_mouse_possible_duplicate_audit",
        dpi,
    )
    outputs += plot_exploratory_candidate_table(
        result.pairwise_evidence,
        root / "figure_9_exploratory_candidate_mouse_presence",
        dpi,
    )
    return outputs


def plot_common_category_space(
    coordinates: pd.DataFrame,
    evidence: pd.DataFrame,
    prototypes: pd.DataFrame,
    output_stem: str | Path,
    dpi: int = 300,
) -> list[Path]:
    """Render the two-PC overview and state its dimensionality limits."""

    metrics = _projection_metrics(prototypes, coordinates)
    fig = plt.figure(figsize=(12.2, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(2.45, 1.0))
    axis = fig.add_subplot(grid[0, 0])
    summary_axis = fig.add_subplot(grid[0, 1])

    animals = sorted(coordinates["animal_id"].astype(str).unique())
    mouse_colors = {
        animal: OKABE_ITO[index % len(OKABE_ITO)]
        for index, animal in enumerate(animals)
    }
    x_low, x_high = _robust_limits(coordinates["common_pc1"].to_numpy(float), 0.0)
    y_low, y_high = _robust_limits(coordinates["common_pc2"].to_numpy(float), 0.0)
    x_margin = 0.08 * (x_high - x_low)
    y_margin = 0.08 * (y_high - y_low)
    x_low, x_high = x_low - x_margin, x_high + x_margin
    y_low, y_high = y_low - y_margin, y_high + y_margin
    display = coordinates.copy()
    display["display_pc1"] = np.clip(display["common_pc1"], x_low, x_high)
    display["display_pc2"] = np.clip(display["common_pc2"], y_low, y_high)
    coordinate_map = display.set_index("local_type")

    candidates = _exploratory_candidate_pairs(evidence)
    for row in candidates.itertuples(index=False):
        first = coordinate_map.loc[str(row.local_type_a)]
        second = coordinate_map.loc[str(row.local_type_b)]
        axis.plot(
            [first.display_pc1, second.display_pc1],
            [first.display_pc2, second.display_pc2],
            color="#E69F00",
            linewidth=1.2,
            linestyle="--",
            alpha=0.75,
            zorder=1,
        )

    custom_offsets = {
        "E77-C1": (8, -16),
        "E77-C2": (-48, 10),
        "E77-C3": (8, 12),
        "E79-C1": (8, -17),
        "E79-C2": (-50, 10),
        "E50-C7": (8, 11),
        "E76-C2": (-48, 10),
    }
    default_offsets = ((6, 7), (6, -13), (-44, 7), (-44, -13), (8, 16), (-44, 16))
    for row_index, row in enumerate(display.itertuples(index=False)):
        animal = str(row.animal_id)
        local_type = str(row.local_type)
        edge = "#111111" if row.local_resolution_status == "validated_stable" else "#777777"
        axis.scatter(
            row.display_pc1,
            row.display_pc2,
            marker=MOUSE_MARKERS.get(animal, "o"),
            s=90 if row.event_support_status == "adequate_events" else 65,
            facecolor=mouse_colors[animal],
            edgecolor=edge,
            linewidth=1.1,
            alpha=1.0 if row.event_support_status == "adequate_events" else 0.65,
            zorder=3,
        )
        offset = custom_offsets.get(local_type, default_offsets[row_index % len(default_offsets)])
        axis.annotate(
            local_type,
            (row.display_pc1, row.display_pc2),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.0,
            fontweight="bold",
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.4},
        )

    handles = [
        mpl.lines.Line2D(
            [],
            [],
            marker=MOUSE_MARKERS.get(animal, "o"),
            linestyle="none",
            markerfacecolor=mouse_colors[animal],
            markeredgecolor="#222222",
            markersize=7,
            label=animal,
        )
        for animal in animals
    ]
    axis.legend(handles=handles, title="小鼠", frameon=False, ncol=3, loc="best")
    variance_1 = 100 * float(coordinates["common_pc1_variance_fraction"].iloc[0])
    variance_2 = 100 * float(coordinates["common_pc2_variance_fraction"].iloc[0])
    axis.set_xlabel(f"共同特征主成分 PC1（解释 {variance_1:.2f}% 变异）")
    axis.set_ylabel(f"共同特征主成分 PC2（解释 {variance_2:.2f}% 变异）")
    axis.set_title(
        "图 2  21 个发作内局部波形类别的共同空间概览\n"
        f"PC1 + PC2 共解释 {variance_1 + variance_2:.2f}% 变异；虚线仅标出待复核配对"
    )
    axis.set_xlim(x_low, x_high)
    axis.set_ylim(y_low, y_high)
    _despine(axis)

    summary_axis.axis("off")
    summary_axis.text(0.0, 0.98, "如何读这张图", fontsize=12, fontweight="bold", va="top")
    summary_axis.text(
        0.0,
        0.88,
        "这是 79 维共同表征的二维 PCA 投影。\n"
        "整体远近关系保留较好，但局部邻居会丢失。",
        fontsize=9,
        va="top",
        linespacing=1.55,
    )
    metric_rows = [
        ("二维解释变异", f"{variance_1 + variance_2:.1f}%"),
        ("高维/二维距离相关", f"rho = {metrics['distance_spearman']:.3f}"),
        ("最近邻保留率", f"{100 * metrics['top1_preservation']:.1f}%"),
    ]
    y = 0.70
    for label, value in metric_rows:
        summary_axis.add_patch(
            mpl.patches.FancyBboxPatch(
                (0.0, y - 0.055),
                0.96,
                0.10,
                boxstyle="round,pad=0.01,rounding_size=0.01",
                facecolor="#F3F5F7",
                edgecolor="#D4D8DC",
                linewidth=0.7,
                transform=summary_axis.transAxes,
            )
        )
        summary_axis.text(0.035, y, label, va="center", fontsize=8.3)
        summary_axis.text(0.92, y, value, va="center", ha="right", fontsize=9.2, fontweight="bold")
        y -= 0.13
    summary_axis.text(0.0, 0.27, "结论", fontsize=10, fontweight="bold", color="#D55E00", va="top")
    summary_axis.text(
        0.0,
        0.21,
        "点靠近不等于同一类型。\n"
        "类别合并必须依据完整 79 维证据和\n"
        "事件级 bootstrap，而不是这两个坐标轴。",
        fontsize=9,
        va="top",
        linespacing=1.55,
    )
    summary_axis.text(
        0.0,
        0.03,
        "仅作全局概览，不用于合并判定",
        fontsize=9.5,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#0072B2", "edgecolor": "none"},
        va="bottom",
    )
    return _save(fig, output_stem, dpi)


def plot_interpretable_category_profile(
    prototypes: pd.DataFrame,
    representative_dimensions: pd.DataFrame,
    output_stem: str | Path,
    dpi: int = 300,
) -> list[Path]:
    """Show interpretable directions and their category-separation strength."""

    selected = representative_dimensions.sort_values("rank")
    features = selected["feature"].astype(str).tolist()
    indexed = prototypes.set_index("local_type")
    matrix_columns = []
    for feature in features:
        values = pd.to_numeric(indexed[feature], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        center = float(np.median(finite)) if len(finite) else 0.0
        q25, q75 = np.quantile(finite, [0.25, 0.75]) if len(finite) else (0.0, 1.0)
        scale = max(float(q75 - q25), 1e-8)
        matrix_columns.append((values - center) / scale)
    matrix = np.column_stack(matrix_columns)
    order = sorted(indexed.index.astype(str), key=_natural_label_sort_key)
    positions = [indexed.index.get_loc(label) for label in order]
    matrix = np.clip(matrix[positions], -3.0, 3.0)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 7.2),
        gridspec_kw={"width_ratios": (2.25, 1.0)},
        constrained_layout=True,
    )
    image = axes[0].imshow(matrix, aspect="auto", cmap="PuOr", vmin=-3.0, vmax=3.0)
    axes[0].set_xticks(
        np.arange(len(features)),
        [FEATURE_LABELS.get(feature, feature) for feature in features],
        rotation=22,
        ha="right",
    )
    axes[0].set_yticks(np.arange(len(order)), order)
    axes[0].set_xlabel("五个互不高度冗余、且具有生理含义的代表维度")
    axes[0].set_ylabel("局部波形类别")
    axes[0].set_title("A  每个类别相对 21 类总体中位数的高低")
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.035, pad=0.02)
    colorbar.set_label("稳健 z 分数：正值=较高，负值=较低")
    for boundary in (7, 10, 13, 17, 19):
        axes[0].axhline(boundary - 0.5, color="white", linewidth=1.4)

    ratios = pd.to_numeric(selected["separation_ratio"], errors="coerce").to_numpy(float)
    labels = [FEATURE_LABELS.get(feature, feature) for feature in features]
    bars = axes[1].barh(
        np.arange(len(features)),
        ratios,
        color=["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#56B4E9"][: len(features)],
    )
    axes[1].bar_label(bars, labels=[f"{value:.2f}" for value in ratios], padding=3, fontsize=8)
    axes[1].set_yticks(np.arange(len(features)), labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("类间差异 / 类内波动")
    axes[1].set_title("B  哪些维度更能区分类别")
    axes[1].text(
        0.0,
        -0.18,
        "数值越大，说明该维度的类间差异相对类内波动越明显。\n"
        "它表示区分能力，不表示生理重要性或因果性。",
        transform=axes[1].transAxes,
        fontsize=8.5,
        va="top",
        linespacing=1.45,
    )
    _despine(axes[1])
    fig.suptitle(
        "图 4  21 个局部波形类别在哪些可解释维度上不同\n"
        "E77 的 CH7 无效，因此其双侧两列留空；匹配阈值不再画成易误解的‘重要性柱状图’",
        fontsize=11,
    )
    return _save(fig, output_stem, dpi)


def plot_category_count_and_evidence(
    inventory: pd.DataFrame,
    registry: pd.DataFrame,
    evidence: pd.DataFrame,
    output_stem: str | Path,
    dpi: int = 300,
) -> list[Path]:
    animals = sorted(inventory["animal_id"].astype(str).unique())
    counts = inventory.groupby("animal_id")["local_type"].nunique().reindex(animals)
    stable = inventory.groupby("animal_id")["review_stability_ok"].first().reindex(animals)
    candidates = _exploratory_candidate_pairs(evidence)
    confirmed = int(
        registry["analysis_type_status"].isin(
            ["confirmed_shared_candidate", "provisional_shared_candidate"]
        ).sum()
    )
    retained = int(registry["local_type_count"].sum()) if "local_type_count" in registry else len(registry)

    fig = plt.figure(figsize=(11.8, 4.6), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.7, 1.15))
    axis = fig.add_subplot(grid[0, 0])
    summary_axis = fig.add_subplot(grid[0, 1])
    colors = ["#009E73" if bool(value) else "#E69F00" for value in stable]
    bars = axis.bar(animals, counts, color=colors, edgecolor="#333333", linewidth=0.6)
    axis.bar_label(bars, labels=[str(int(value)) for value in counts], padding=3, fontsize=9)
    axis.set_ylim(0, max(counts) + 1.5)
    axis.set_ylabel("发作内局部波形类别数")
    axis.set_xlabel("小鼠")
    axis.set_title("A  每只鼠保留的局部类别")
    axis.legend(
        handles=[
            mpl.patches.Patch(color="#009E73", label="稳定分辨率：E50、E77"),
            mpl.patches.Patch(color="#E69F00", label="探索性分辨率：其余小鼠"),
        ],
        frameon=False,
        fontsize=8,
    )
    _despine(axis)

    summary_axis.axis("off")
    summary_axis.set_title("B  跨鼠匹配的直接结论")
    cards = [
        (str(confirmed), "正式共有类型", "#0072B2"),
        (str(len(candidates)), "探索性候选配对", "#E69F00"),
        (str(retained), "正式保留局部类别", "#009E73"),
    ]
    x_positions = (0.02, 0.35, 0.68)
    for x, (value, label, color) in zip(x_positions, cards):
        summary_axis.add_patch(
            mpl.patches.FancyBboxPatch(
                (x, 0.56),
                0.28,
                0.29,
                boxstyle="round,pad=0.018,rounding_size=0.02",
                facecolor="#F7F8FA",
                edgecolor=color,
                linewidth=1.5,
                transform=summary_axis.transAxes,
            )
        )
        summary_axis.text(x + 0.14, 0.72, value, ha="center", va="center", fontsize=22, fontweight="bold", color=color)
        summary_axis.text(x + 0.14, 0.61, label, ha="center", va="center", fontsize=8.5)
    summary_axis.text(
        0.02,
        0.42,
        "解释：3 个探索性配对全部只涉及 E77 与 E79，且证据不足或冲突。\n"
        "因此不能把它们登记为跨鼠共有类型，也不能说其余类别一定没有生物学共同性。",
        fontsize=8.5,
        va="top",
        linespacing=1.55,
    )
    summary_axis.text(
        0.02,
        0.10,
        "本图统计的是发作内部的局部波形状态，\n不是每只鼠完整发作类型的数量。",
        fontsize=9,
        fontweight="bold",
        color="#D55E00",
        va="bottom",
    )
    fig.suptitle("图 5  鼠内类别发现与跨鼠匹配是两个独立统计步骤", fontsize=11)
    return _save(fig, output_stem, dpi)


def plot_candidate_evidence_matrix(
    evidence: pd.DataFrame,
    output_stem: str | Path,
    dpi: int = 300,
) -> list[Path]:
    rows = _exploratory_candidate_pairs(evidence)
    blocks = ["raw_shape", "amplitude", "shape", "frequency", "bilateral"]
    values = np.full((len(rows), len(blocks)), np.nan, dtype=float)
    for row_index, row in enumerate(rows.itertuples(index=False)):
        for column_index, block in enumerate(blocks):
            high = _as_float(getattr(row, f"{block}_ci_high", np.nan))
            threshold = _as_float(getattr(row, f"{block}_threshold", np.nan))
            if np.isfinite(high) and np.isfinite(threshold) and threshold > 0:
                values[row_index, column_index] = high / threshold

    fig = plt.figure(figsize=(12.0, 4.9), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(2.55, 1.2))
    axis = fig.add_subplot(grid[0, 0])
    conclusion_axis = fig.add_subplot(grid[0, 1])
    states = np.full(values.shape, 2, dtype=int)
    states[np.isfinite(values) & (values <= 1.0)] = 0
    states[np.isfinite(values) & (values > 1.0)] = 1
    cmap = mpl.colors.ListedColormap(["#CDEBDD", "#F4C7C3", "#E0E0E0"])
    axis.imshow(states, aspect="auto", cmap=cmap, vmin=0, vmax=2)
    axis.set_xticks(np.arange(len(blocks)), [BLOCK_LABELS[block] for block in blocks])
    axis.set_yticks(
        np.arange(len(rows)),
        [
            f"CP{index + 1:02d}  {row.local_type_a} - {row.local_type_b}"
            for index, row in enumerate(rows.itertuples(index=False))
        ],
    )
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            label = "不可用" if not np.isfinite(value) else f"{value:.2f}\n{'通过' if value <= 1.0 else '未通过'}"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8.0,
                fontweight="bold" if np.isfinite(value) else "normal",
                color="#202020",
            )
    axis.set_xlabel("95% bootstrap 上界 / 同类重复性阈值（<= 1 才通过）")
    axis.set_ylabel("探索性跨鼠配对")
    axis.set_title("A  每个证据块是否通过")

    conclusion_axis.axis("off")
    conclusion_axis.set_title("B  配对结论")
    conclusions = [
        ("CP01", "形状接近阈值，但幅值明显不同", "不合并"),
        ("CP02", "E77-C3 仅 18 次事件，且四个维度未通过", "低样本，不合并"),
        ("CP03", "完整形状、幅值、时域和频域均未通过", "不合并"),
    ]
    y = 0.85
    for candidate, reason, decision in conclusions[: len(rows)]:
        conclusion_axis.text(0.0, y, candidate, fontsize=9, fontweight="bold", color="#0072B2")
        conclusion_axis.text(0.0, y - 0.07, reason, fontsize=8.2, va="top", wrap=True)
        conclusion_axis.text(0.0, y - 0.18, decision, fontsize=8.5, fontweight="bold", color="#D55E00")
        y -= 0.27
    conclusion_axis.text(
        0.0,
        0.02,
        "正式跨鼠共有类型：0",
        fontsize=10,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#0072B2", "edgecolor": "none"},
    )
    fig.suptitle(
        "图 7  三个探索性候选配对的直接证据\n"
        "绿色=通过，红色=未通过，灰色=E77 双侧证据不可用",
        fontsize=11,
    )
    return _save(fig, output_stem, dpi)


def plot_within_mouse_pair_summary(
    audit: pd.DataFrame,
    inventory: pd.DataFrame,
    output_stem: str | Path,
    dpi: int = 300,
) -> list[Path]:
    animals = sorted(inventory["animal_id"].astype(str).unique())
    relation_order = [
        "shape_distinct",
        "same_shape_family_physiology_distinct",
        "insufficient_or_mixed_evidence",
    ]
    labels = ["完整波形形状不同", "形状相近但生理证据不同", "证据混合或不足"]
    colors = ["#0072B2", "#E69F00", "#A7A9AC"]
    counts = pd.crosstab(audit["animal_id"], audit["within_mouse_relation"]).reindex(
        index=animals, columns=relation_order, fill_value=0
    )
    fig = plt.figure(figsize=(11.4, 5.4), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(2.7, 1.0))
    axis = fig.add_subplot(grid[0, 0])
    summary_axis = fig.add_subplot(grid[0, 1])
    left = np.zeros(len(animals), dtype=float)
    for relation, label, color in zip(relation_order, labels, colors):
        values = counts[relation].to_numpy(int)
        bars = axis.barh(
            animals,
            values,
            left=left,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=label,
        )
        for bar, value in zip(bars, values):
            if value:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white" if color != "#E69F00" else "#202020",
                )
        left += values
    for index, total in enumerate(left.astype(int)):
        axis.text(total + 0.25, index, f"共 {total} 对", va="center", fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("同一只鼠内的局部类别配对数")
    axis.set_ylabel("小鼠")
    axis.set_title(
        "图 8  同鼠内 21 个标签的重复细分审计\n"
        "没有发现可确认的重复细分：0 对"
    )
    axis.legend(
        frameon=False,
        loc="center right",
        bbox_to_anchor=(0.98, 0.28),
        fontsize=7.5,
    )
    _despine(axis)

    summary_axis.axis("off")
    summary_axis.text(0.0, 0.96, "直接结论", fontsize=11, fontweight="bold", va="top")
    summary_axis.text(
        0.0,
        0.82,
        "可确认的重复细分",
        fontsize=8.5,
        color="#555555",
    )
    summary_axis.text(0.0, 0.69, "0 对", fontsize=24, fontweight="bold", color="#0072B2")
    summary_axis.text(
        0.0,
        0.50,
        "E50 有 10 对类别：\n"
        "完整波形轮廓相近，\n"
        "但幅值、频域或双侧\n"
        "生理证据不同。",
        fontsize=9,
        va="top",
        linespacing=1.45,
    )
    summary_axis.text(
        0.0,
        0.22,
        "因此：保留这些标签，\n不是把它们合并。",
        fontsize=9.5,
        fontweight="bold",
        color="#D55E00",
        va="top",
        linespacing=1.45,
    )
    summary_axis.text(
        0.0,
        0.04,
        "灰色表示现有证据不足以作确定判断",
        fontsize=8,
        color="#666666",
        va="bottom",
    )
    return _save(fig, output_stem, dpi)


def plot_exploratory_candidate_table(
    evidence: pd.DataFrame,
    output_stem: str | Path,
    dpi: int = 300,
) -> list[Path]:
    candidates = _exploratory_candidate_pairs(evidence)
    if candidates.empty:
        return []
    rows = []
    for index, row in enumerate(candidates.itertuples(index=False)):
        if str(row.pair_decision) == "compatible_low_event_support":
            issue = f"{row.local_type_a} 仅 {int(row.events_a)} 次事件；\n置信区间较宽"
            conclusion = "低样本候选，\n不合并"
        elif index == 0:
            issue = "完整形状临界，\n但幅值差异明显"
            conclusion = "混合证据，\n不合并"
        else:
            issue = "完整形状、幅值、时域\n和频域均有差异"
            conclusion = "多维差异，\n不合并"
        rows.append(
            [
                f"CP{index + 1:02d}",
                f"{row.local_type_a} - {row.local_type_b}",
                "E77、E79\n（2/6）",
                issue,
                conclusion,
            ]
        )
    fig, axis = plt.subplots(figsize=(12.0, 4.5), constrained_layout=True)
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["候选", "局部类别配对", "涉及小鼠", "主要证据问题", "正式结论"],
        colWidths=[0.08, 0.19, 0.16, 0.35, 0.20],
        cellLoc="left",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 2.35)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D4D8")
        cell.set_linewidth(0.7)
        if row_index == 0:
            cell.set_facecolor("#0072B2")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#F7F8FA" if row_index % 2 else "white")
            if column_index == 4:
                cell.get_text().set_color("#D55E00")
                cell.get_text().set_fontweight("bold")
    axis.set_title(
        "图 9  探索性跨鼠相似配对总表\n"
        "三行全部只涉及 E77 与 E79，且配对互相重叠，不能计作 3 种跨鼠共有发作类型",
        fontsize=11,
        pad=15,
    )
    axis.text(
        0.5,
        0.02,
        "正式跨鼠共有类型 = 0；正式登记仍保留 21 个带小鼠编号的局部类别",
        transform=axis.transAxes,
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color="#0072B2",
    )
    return _save(fig, output_stem, dpi)


def _projection_metrics(
    prototypes: pd.DataFrame,
    coordinates: pd.DataFrame,
) -> dict[str, float]:
    raw_columns = sorted(
        [column for column in prototypes if column.startswith("raw_shape_pc")],
        key=lambda value: int(value.rsplit("pc", 1)[1]),
    )
    blocks: dict[str, tuple[str, ...]] = {
        "raw_shape": tuple(raw_columns),
        **FEATURE_BLOCKS,
    }
    ordered = prototypes.set_index("local_type").loc[coordinates["local_type"].astype(str)]
    branches: list[np.ndarray] = []
    for block, columns in blocks.items():
        values = ordered[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if block != "raw_shape":
            for column in range(values.shape[1]):
                finite = values[:, column][np.isfinite(values[:, column])]
                center = float(np.median(finite)) if len(finite) else 0.0
                q25, q75 = np.quantile(finite, [0.25, 0.75]) if len(finite) else (0.0, 1.0)
                scale = float(q75 - q25)
                if scale <= 1e-8:
                    scale = float(np.std(finite, ddof=1)) if len(finite) >= 2 else 1.0
                values[:, column] = (values[:, column] - center) / max(scale, 1e-8)
        medians = np.nanmedian(values, axis=0)
        medians = np.nan_to_num(medians, nan=0.0)
        missing = ~np.isfinite(values)
        values[missing] = np.take(medians, np.where(missing)[1])
        if block == "raw_shape":
            energy = float(np.sqrt(np.mean(np.sum(values**2, axis=1))))
            branches.append(values / max(energy, 1e-12))
        else:
            branches.append(values / np.sqrt(len(columns)))
    full = np.concatenate(branches, axis=1)
    projected = coordinates[["common_pc1", "common_pc2"]].to_numpy(float)
    full_distances = pdist(full)
    projected_distances = pdist(projected)
    correlation = float(spearmanr(full_distances, projected_distances).statistic)
    stress = float(
        np.sqrt(
            np.sum((full_distances - projected_distances) ** 2)
            / np.sum(full_distances**2)
        )
    )
    full_square = squareform(full_distances)
    projected_square = squareform(projected_distances)
    np.fill_diagonal(full_square, np.inf)
    np.fill_diagonal(projected_square, np.inf)
    nearest = np.mean(
        [
            int(np.argmin(full_square[index]) == np.argmin(projected_square[index]))
            for index in range(len(full_square))
        ]
    )
    return {
        "distance_spearman": correlation,
        "normalized_stress": stress,
        "top1_preservation": float(nearest),
    }


def _exploratory_candidate_pairs(evidence: pd.DataFrame) -> pd.DataFrame:
    decisions = {"compatible_low_event_support", "not_merged_mixed_evidence"}
    rows = evidence.loc[evidence["pair_decision"].isin(decisions)].copy()
    if rows.empty:
        return rows
    ratios = []
    for row in rows.itertuples(index=False):
        values = []
        for block in ("raw_shape", "amplitude", "shape", "frequency"):
            high = _as_float(getattr(row, f"{block}_ci_high", np.nan))
            threshold = _as_float(getattr(row, f"{block}_threshold", np.nan))
            if np.isfinite(high) and np.isfinite(threshold) and threshold > 0:
                values.append(high / threshold)
        ratios.append(max(values) if values else np.inf)
    rows["_candidate_ratio"] = ratios
    return rows.sort_values("_candidate_ratio", kind="stable").reset_index(drop=True)


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "axes.unicode_minus": False,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save(fig: plt.Figure, stem: str | Path, dpi: int) -> list[Path]:
    root = Path(stem)
    outputs: list[Path] = []
    for suffix, options in (
        (".png", {"dpi": dpi}),
        (".pdf", {}),
        (".svg", {}),
    ):
        path = root.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="white", **options)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _despine(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _natural_label_sort_key(value: Any) -> tuple[str, int]:
    text = str(value)
    try:
        number = int(text.rsplit("C", 1)[1])
    except (IndexError, ValueError):
        number = 999999
    return text.split("-", 1)[0], number


def _robust_limits(values: np.ndarray, tail: float = 0.01) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 3:
        low, high = (float(np.min(finite)), float(np.max(finite))) if len(finite) else (-1.0, 1.0)
    else:
        low, high = np.quantile(finite, [tail, 1.0 - tail])
    if high - low < 1e-8:
        low, high = low - 1.0, high + 1.0
    margin = 0.08 * (high - low)
    return float(low - margin), float(high + margin)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


__all__ = ["generate_clinical_cross_mouse_figures"]
