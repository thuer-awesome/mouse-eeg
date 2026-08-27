from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Arial", "SimHei", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "legend.fontsize": 6.3,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, Rectangle

from mouse_sleep.epilepsy.single_channel_visualization import TraceStore

# Importing the reusable trace store loads the legacy figure module, which has
# its own English-only font preference. Restore this report's CJK-capable stack.
mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "Arial",
    "SimHei",
    "DejaVu Sans",
]


MM_PER_INCH = 25.4
MANUSCRIPT_WIDTH_IN = 7.204724409448819  # 183 mm, Nature double-column width.
A4_PORTRAIT = (8.27, 11.69)

GROUP_KEYS = ["animal_id", "session_id", "channel_name"]
MANUAL_KEYS = GROUP_KEYS + ["event_id"]
PREDICTION_KEYS = GROUP_KEYS + ["predicted_event_index"]

MATCHED = "matched_one_to_one"
MANUAL_MERGED = "manual_overlap_unmatched_merged_prediction"
MANUAL_MISS = "manual_zero_overlap_miss"
PRED_FRAGMENT = "prediction_overlap_unmatched_redundant_segment"
PRED_CANDIDATE = "prediction_zero_manual_overlap_candidate"

MANUAL_COLOR = "#1E5B8C"
PREDICTION_COLOR = "#C45A32"
PROBABILITY_COLOR = "#25746B"
MATCH_COLOR = "#3D7F78"
MERGE_COLOR = "#76528B"
MISS_COLOR = "#B64342"
FRAGMENT_COLOR = "#D08A2E"
CANDIDATE_COLOR = "#7B858E"
INK = "#252A2E"
MID_GREY = "#6C7378"
LIGHT_GREY = "#E6E9EB"
VERY_LIGHT_GREY = "#F5F6F7"

CATEGORY_ORDER = [MATCHED, MANUAL_MISS, MANUAL_MERGED, PRED_FRAGMENT, PRED_CANDIDATE]
CATEGORY_META: dict[str, dict[str, str]] = {
    MATCHED: {
        "short_zh": "一对一配对",
        "title_zh": "人工标签与模型区间获得一对一配对",
        "definition_zh": (
            "同一鼠、同一记录、同一通道内存在正时间重叠，并在最大总 IoU 匹配中获得唯一配对。"
            "这表示检测到，不等于完整覆盖。"
        ),
        "color": MATCH_COLOR,
    },
    MANUAL_MISS: {
        "short_zh": "人工零重叠漏检",
        "title_zh": "人工标签内没有任何模型区间重叠",
        "definition_zh": "人工区间与所有模型预测均无正时间交集，是严格意义上的零重叠漏检。",
        "color": MISS_COLOR,
    },
    MANUAL_MERGED: {
        "short_zh": "人工有重叠但未配对",
        "title_zh": "人工标签有模型重叠，但未获得一对一配对",
        "definition_zh": (
            "常见原因是一个较长模型区间跨越多个相邻人工标签，只能与其中一个标签形成唯一配对。"
        ),
        "color": MERGE_COLOR,
    },
    PRED_FRAGMENT: {
        "short_zh": "预测有重叠但未配对",
        "title_zh": "模型区间落在人工标签内，但未获得一对一配对",
        "definition_zh": (
            "常见原因是同一人工发作被切成多个预测片段；额外片段仍有人工重叠，因此不能与标签外候选混为一类。"
        ),
        "color": FRAGMENT_COLOR,
    },
    PRED_CANDIDATE: {
        "short_zh": "预测零人工重叠候选",
        "title_zh": "模型区间与任何人工标签均无重叠",
        "definition_zh": (
            "这是临床优先复核的标签外候选；它在当前标签口径下计为误报，但仍可能包含漏标发作或伪迹。"
        ),
        "color": CANDIDATE_COLOR,
    },
}


@dataclass(frozen=True)
class AccountingTables:
    manual: pd.DataFrame
    prediction: pd.DataFrame
    matched_pairs: pd.DataFrame
    summary: pd.DataFrame
    per_group: pd.DataFrame
    split_summary: pd.DataFrame
    test_channel_expansion: pd.DataFrame
    quality_ladder: pd.DataFrame


@dataclass(frozen=True)
class ReviewBundle:
    output_dir: Path
    summary_figure_pdf: Path
    quick_review_pdf: Path
    complete_atlas_pdf: Path
    quick_pages: int
    complete_pages: int
    counts: Mapping[str, int]


def _require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _group_tuple(row: Any) -> tuple[str, str, str]:
    return str(row.animal_id), str(row.session_id), str(row.channel_name)


def _interval_union_ms(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((float(start), float(end)) for start, end in intervals if end > start)
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _overlap_graph(
    truth: pd.DataFrame, predicted: pd.DataFrame
) -> tuple[dict[tuple[str, str, str, str], list[int]], dict[tuple[str, str, str, int], list[str]]]:
    manual_to_prediction: dict[tuple[str, str, str, str], list[int]] = {}
    prediction_to_manual: dict[tuple[str, str, str, int], list[str]] = {}
    group_keys = set(map(tuple, truth[GROUP_KEYS].drop_duplicates().itertuples(index=False, name=None)))
    group_keys.update(
        map(tuple, predicted[GROUP_KEYS].drop_duplicates().itertuples(index=False, name=None))
    )
    for group_key in sorted(group_keys):
        truth_group = truth
        prediction_group = predicted
        for column, value in zip(GROUP_KEYS, group_key, strict=True):
            truth_group = truth_group.loc[truth_group[column].astype(str).eq(str(value))]
            prediction_group = prediction_group.loc[
                prediction_group[column].astype(str).eq(str(value))
            ]
        for manual_row in truth_group.itertuples(index=False):
            overlaps = prediction_group.loc[
                prediction_group["end_time_ms"].gt(float(manual_row.start_time_ms))
                & prediction_group["start_time_ms"].lt(float(manual_row.end_time_ms))
            ]
            key = (*map(str, group_key), str(manual_row.event_id))
            manual_to_prediction[key] = sorted(
                overlaps["predicted_event_index"].astype(int).tolist()
            )
        for prediction_row in prediction_group.itertuples(index=False):
            overlaps = truth_group.loc[
                truth_group["end_time_ms"].gt(float(prediction_row.start_time_ms))
                & truth_group["start_time_ms"].lt(float(prediction_row.end_time_ms))
            ]
            key = (*map(str, group_key), int(prediction_row.predicted_event_index))
            prediction_to_manual[key] = sorted(overlaps["event_id"].astype(str).tolist())
    return manual_to_prediction, prediction_to_manual


def _probability_groups(
    probabilities: pd.DataFrame | None,
) -> dict[tuple[str, str, str], pd.DataFrame]:
    if probabilities is None or probabilities.empty:
        return {}
    _require_columns(
        probabilities,
        GROUP_KEYS + ["center_time_ms", "seizure_probability"],
        "continuous probabilities",
    )
    return {
        tuple(map(str, key)): group.sort_values("center_time_ms", kind="stable")
        for key, group in probabilities.groupby(GROUP_KEYS, sort=False)
    }


def _probability_stats(
    groups: Mapping[tuple[str, str, str], pd.DataFrame],
    group_key: tuple[str, str, str],
    start_ms: float,
    end_ms: float,
) -> tuple[float, float]:
    group = groups.get(tuple(map(str, group_key)))
    if group is None or group.empty:
        return math.nan, math.nan
    selected = group.loc[
        group["center_time_ms"].ge(float(start_ms))
        & group["center_time_ms"].le(float(end_ms))
    ]
    if selected.empty:
        selected = group.loc[
            group["end_time_ms"].gt(float(start_ms))
            & group["start_time_ms"].lt(float(end_ms))
        ]
    if selected.empty:
        return math.nan, math.nan
    values = selected["seizure_probability"].to_numpy(float)
    return float(np.max(values)), float(np.mean(values))


def build_event_accounting_tables(
    truth: pd.DataFrame,
    predicted: pd.DataFrame,
    matches: pd.DataFrame,
    assessments: pd.DataFrame | None = None,
    probabilities: pd.DataFrame | None = None,
    *,
    all_channel_events: pd.DataFrame | None = None,
) -> AccountingTables:
    """Create mutually exclusive manual-side and prediction-side event ledgers."""

    _require_columns(
        truth,
        MANUAL_KEYS + ["split", "start_time_ms", "end_time_ms", "duration_s"],
        "manual events",
    )
    _require_columns(
        predicted,
        PREDICTION_KEYS + ["split", "start_time_ms", "end_time_ms", "duration_s"],
        "predicted events",
    )
    _require_columns(
        matches,
        MANUAL_KEYS
        + [
            "predicted_event_index",
            "overlap_ms",
            "iou",
            "annotated_coverage",
            "onset_error_seconds",
            "offset_error_seconds",
        ],
        "event matches",
    )
    truth = truth.copy().reset_index(drop=True)
    predicted = predicted.copy().reset_index(drop=True)
    matches = matches.copy().reset_index(drop=True)
    truth["event_id"] = truth["event_id"].astype(str)
    predicted["predicted_event_index"] = predicted["predicted_event_index"].astype(int)
    matches["event_id"] = matches["event_id"].astype(str)
    matches["predicted_event_index"] = matches["predicted_event_index"].astype(int)
    if truth.duplicated(MANUAL_KEYS).any():
        raise ValueError("manual event keys are not unique")
    if predicted.duplicated(PREDICTION_KEYS).any():
        raise ValueError("predicted event keys are not unique")
    if matches.duplicated(MANUAL_KEYS).any() or matches.duplicated(PREDICTION_KEYS).any():
        raise ValueError("event matches are not one-to-one")
    if (matches["overlap_ms"] <= 0).any() or (matches["iou"] <= 0).any():
        raise ValueError("matched events must have positive overlap and IoU")

    manual_to_prediction, prediction_to_manual = _overlap_graph(truth, predicted)
    probability_groups = _probability_groups(probabilities)

    match_manual = matches.set_index(MANUAL_KEYS, drop=False)
    match_prediction = matches.set_index(PREDICTION_KEYS, drop=False)
    assessment_lookup: pd.DataFrame | None = None
    if assessments is not None and not assessments.empty:
        _require_columns(
            assessments,
            MANUAL_KEYS
            + [
                "annotated_duration_ms",
                "covered_duration_ms",
                "coverage",
                "overlapping_predicted_segments",
            ],
            "manual event assessments",
        )
        if assessments.duplicated(MANUAL_KEYS).any():
            raise ValueError("manual assessment keys are not unique")
        assessment_lookup = assessments.set_index(MANUAL_KEYS, drop=False)

    prediction_lookup = predicted.set_index(PREDICTION_KEYS, drop=False)
    truth_lookup = truth.set_index(MANUAL_KEYS, drop=False)
    manual_rows: list[dict[str, Any]] = []
    for row in truth.itertuples(index=False):
        group_key = _group_tuple(row)
        manual_key = (*group_key, str(row.event_id))
        overlap_indices = manual_to_prediction.get(manual_key, [])
        overlap_intervals = []
        for prediction_index in overlap_indices:
            prediction_row = prediction_lookup.loc[(*group_key, int(prediction_index))]
            overlap_intervals.append(
                (
                    max(float(row.start_time_ms), float(prediction_row.start_time_ms)),
                    min(float(row.end_time_ms), float(prediction_row.end_time_ms)),
                )
            )
        covered_ms = _interval_union_ms(overlap_intervals)
        duration_ms = float(row.end_time_ms) - float(row.start_time_ms)
        union_coverage = covered_ms / duration_ms if duration_ms > 0 else math.nan
        match_row = match_manual.loc[manual_key] if manual_key in match_manual.index else None
        if match_row is not None:
            category = MATCHED
            matched_prediction_index: int | pd._libs.missing.NAType = int(
                match_row["predicted_event_index"]
            )
        elif overlap_indices:
            category = MANUAL_MERGED
            matched_prediction_index = pd.NA
        else:
            category = MANUAL_MISS
            matched_prediction_index = pd.NA
        if assessment_lookup is not None and manual_key in assessment_lookup.index:
            assessment = assessment_lookup.loc[manual_key]
            assessment_coverage = float(assessment["coverage"])
            assessment_segments = int(assessment["overlapping_predicted_segments"])
            if not np.isclose(assessment_coverage, union_coverage, atol=1e-7):
                raise ValueError(f"coverage mismatch for manual event {manual_key}")
            if assessment_segments != len(overlap_indices):
                raise ValueError(f"overlap-count mismatch for manual event {manual_key}")
        peak_probability, mean_probability = _probability_stats(
            probability_groups,
            group_key,
            float(row.start_time_ms),
            float(row.end_time_ms),
        )
        manual_rows.append(
            {
                "manual_uid": "|".join(manual_key),
                **{column: value for column, value in zip(GROUP_KEYS, group_key, strict=True)},
                "split": str(row.split),
                "event_id": str(row.event_id),
                "start_time_ms": float(row.start_time_ms),
                "end_time_ms": float(row.end_time_ms),
                "duration_s": duration_ms / 1000.0,
                "category": category,
                "overlapping_predicted_count": len(overlap_indices),
                "overlapping_predicted_indices": ";".join(map(str, overlap_indices)),
                "matched_predicted_event_index": matched_prediction_index,
                "union_covered_duration_ms": covered_ms,
                "union_coverage": union_coverage,
                "matched_iou": float(match_row["iou"]) if match_row is not None else math.nan,
                "matched_annotated_coverage": (
                    float(match_row["annotated_coverage"]) if match_row is not None else math.nan
                ),
                "onset_error_seconds": (
                    float(match_row["onset_error_seconds"]) if match_row is not None else math.nan
                ),
                "offset_error_seconds": (
                    float(match_row["offset_error_seconds"]) if match_row is not None else math.nan
                ),
                "peak_seizure_probability": peak_probability,
                "mean_seizure_probability": mean_probability,
            }
        )
    manual = pd.DataFrame(manual_rows)
    manual["matched_predicted_event_index"] = manual[
        "matched_predicted_event_index"
    ].astype("Int64")

    prediction_rows: list[dict[str, Any]] = []
    for row in predicted.itertuples(index=False):
        group_key = _group_tuple(row)
        prediction_key = (*group_key, int(row.predicted_event_index))
        overlap_ids = prediction_to_manual.get(prediction_key, [])
        match_row = (
            match_prediction.loc[prediction_key] if prediction_key in match_prediction.index else None
        )
        if match_row is not None:
            category = MATCHED
            matched_event_id: str | pd._libs.missing.NAType = str(match_row["event_id"])
        elif overlap_ids:
            category = PRED_FRAGMENT
            matched_event_id = pd.NA
        else:
            category = PRED_CANDIDATE
            matched_event_id = pd.NA
        peak_probability, mean_probability = _probability_stats(
            probability_groups,
            group_key,
            float(row.start_time_ms),
            float(row.end_time_ms),
        )
        prediction_rows.append(
            {
                "prediction_uid": "|".join(map(str, prediction_key)),
                **{column: value for column, value in zip(GROUP_KEYS, group_key, strict=True)},
                "split": str(row.split),
                "predicted_event_index": int(row.predicted_event_index),
                "start_time_ms": float(row.start_time_ms),
                "end_time_ms": float(row.end_time_ms),
                "duration_s": float(row.duration_s),
                "category": category,
                "overlapping_manual_count": len(overlap_ids),
                "overlapping_manual_event_ids": ";".join(overlap_ids),
                "matched_event_id": matched_event_id,
                "matched_iou": float(match_row["iou"]) if match_row is not None else math.nan,
                "peak_seizure_probability": peak_probability,
                "mean_seizure_probability": mean_probability,
            }
        )
    prediction = pd.DataFrame(prediction_rows)
    prediction["matched_event_id"] = prediction["matched_event_id"].astype("string")

    pair_rows: list[dict[str, Any]] = []
    for row in matches.itertuples(index=False):
        group_key = _group_tuple(row)
        manual_key = (*group_key, str(row.event_id))
        prediction_key = (*group_key, int(row.predicted_event_index))
        manual_row = truth_lookup.loc[manual_key]
        prediction_row = prediction_lookup.loc[prediction_key]
        manual_account = manual.set_index(MANUAL_KEYS).loc[manual_key]
        prediction_account = prediction.set_index(PREDICTION_KEYS).loc[prediction_key]
        pair_rows.append(
            {
                "pair_uid": f"{'|'.join(manual_key)}<->P{int(row.predicted_event_index)}",
                **{column: value for column, value in zip(GROUP_KEYS, group_key, strict=True)},
                "split": str(row.split),
                "event_id": str(row.event_id),
                "predicted_event_index": int(row.predicted_event_index),
                "manual_start_time_ms": float(manual_row.start_time_ms),
                "manual_end_time_ms": float(manual_row.end_time_ms),
                "manual_duration_s": (
                    float(manual_row.end_time_ms) - float(manual_row.start_time_ms)
                )
                / 1000.0,
                "predicted_start_time_ms": float(prediction_row.start_time_ms),
                "predicted_end_time_ms": float(prediction_row.end_time_ms),
                "predicted_duration_s": float(prediction_row.duration_s),
                "overlap_ms": float(row.overlap_ms),
                "iou": float(row.iou),
                "dice": float(getattr(row, "dice", math.nan)),
                "annotated_coverage": float(row.annotated_coverage),
                "union_coverage": float(manual_account.union_coverage),
                "overlapping_predicted_count": int(
                    manual_account.overlapping_predicted_count
                ),
                "overlapping_manual_count": int(
                    prediction_account.overlapping_manual_count
                ),
                "onset_error_seconds": float(row.onset_error_seconds),
                "offset_error_seconds": float(row.offset_error_seconds),
                "duration_error_seconds": float(
                    getattr(row, "duration_error_seconds", math.nan)
                ),
                "category": MATCHED,
            }
        )
    matched_pairs = pd.DataFrame(pair_rows)

    summary_rows = [
        {
            "entity": "manual",
            "category": category,
            "count": int((manual["category"] == category).sum()),
            "denominator": int(len(manual)),
            "fraction": float((manual["category"] == category).mean()),
            "definition_zh": CATEGORY_META[category]["definition_zh"],
        }
        for category in [MATCHED, MANUAL_MERGED, MANUAL_MISS]
    ]
    summary_rows.extend(
        {
            "entity": "prediction",
            "category": category,
            "count": int((prediction["category"] == category).sum()),
            "denominator": int(len(prediction)),
            "fraction": float((prediction["category"] == category).mean()),
            "definition_zh": CATEGORY_META[category]["definition_zh"],
        }
        for category in [MATCHED, PRED_FRAGMENT, PRED_CANDIDATE]
    )
    summary = pd.DataFrame(summary_rows)

    manual_group = (
        manual.groupby(GROUP_KEYS + ["category"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    prediction_group = (
        prediction.groupby(GROUP_KEYS + ["category"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    manual_group = manual_group.rename(
        columns={
            MATCHED: "manual_matched",
            MANUAL_MERGED: "manual_overlap_unmatched",
            MANUAL_MISS: "manual_zero_overlap",
        }
    )
    prediction_group = prediction_group.rename(
        columns={
            MATCHED: "prediction_matched",
            PRED_FRAGMENT: "prediction_overlap_unmatched",
            PRED_CANDIDATE: "prediction_zero_overlap",
        }
    )
    per_group = manual_group.merge(prediction_group, on=GROUP_KEYS, how="outer").fillna(0)
    count_columns = [column for column in per_group.columns if column not in GROUP_KEYS]
    per_group[count_columns] = per_group[count_columns].astype(int)
    per_group["manual_total"] = per_group[
        ["manual_matched", "manual_overlap_unmatched", "manual_zero_overlap"]
    ].sum(axis=1)
    per_group["prediction_total"] = per_group[
        ["prediction_matched", "prediction_overlap_unmatched", "prediction_zero_overlap"]
    ].sum(axis=1)
    per_group = per_group.sort_values(GROUP_KEYS, kind="stable").reset_index(drop=True)

    complete_events = all_channel_events if all_channel_events is not None else truth
    _require_columns(complete_events, MANUAL_KEYS + ["split"], "all channel events")
    split_rows = []
    for split, group in complete_events.groupby("split", sort=False):
        clinical_count = int(group[["animal_id", "session_id", "event_id"]].drop_duplicates().shape[0])
        split_rows.append(
            {
                "split": str(split),
                "channel_event_count": int(len(group)),
                "unique_clinical_event_count": clinical_count,
                "extra_channel_copies": int(len(group) - clinical_count),
            }
        )
    split_summary = pd.DataFrame(split_rows)
    split_order = pd.Categorical(
        split_summary["split"], categories=["train", "val", "test"], ordered=True
    )
    split_summary = split_summary.assign(_order=split_order).sort_values("_order").drop(
        columns="_order"
    )

    test_events = complete_events.loc[complete_events["split"].astype(str).eq("test")]
    expansion = (
        test_events.groupby(["animal_id", "session_id", "event_id"], observed=True)[
            "channel_name"
        ]
        .nunique()
        .rename("channel_count")
        .reset_index()
    )
    test_channel_expansion = (
        expansion.groupby("channel_count", observed=True)
        .size()
        .rename("clinical_event_count")
        .reset_index()
    )
    test_channel_expansion["channel_event_rows"] = (
        test_channel_expansion["channel_count"]
        * test_channel_expansion["clinical_event_count"]
    )

    ladder_rows = [
        ("any_positive_overlap", int((manual["overlapping_predicted_count"] > 0).sum())),
        ("one_to_one_match_iou_gt_0", int((manual["category"] == MATCHED).sum())),
        ("matched_iou_ge_0_5", int((manual["matched_iou"] >= 0.5).sum())),
        ("matched_iou_ge_0_7", int((manual["matched_iou"] >= 0.7).sum())),
        ("matched_iou_ge_0_9", int((manual["matched_iou"] >= 0.9).sum())),
        ("union_coverage_ge_0_8", int((manual["union_coverage"] >= 0.8).sum())),
    ]
    quality_ladder = pd.DataFrame(ladder_rows, columns=["criterion", "manual_event_count"])
    quality_ladder["denominator"] = len(manual)
    quality_ladder["fraction"] = quality_ladder["manual_event_count"] / len(manual)

    if len(matched_pairs) != int((manual["category"] == MATCHED).sum()):
        raise ValueError("manual matched count does not equal matched-pair count")
    if len(matched_pairs) != int((prediction["category"] == MATCHED).sum()):
        raise ValueError("prediction matched count does not equal matched-pair count")
    if len(manual) != int(summary.loc[summary["entity"].eq("manual"), "count"].sum()):
        raise ValueError("manual accounting is not exhaustive")
    if len(prediction) != int(
        summary.loc[summary["entity"].eq("prediction"), "count"].sum()
    ):
        raise ValueError("prediction accounting is not exhaustive")
    return AccountingTables(
        manual=manual.sort_values(GROUP_KEYS + ["start_time_ms"], kind="stable").reset_index(
            drop=True
        ),
        prediction=prediction.sort_values(
            GROUP_KEYS + ["start_time_ms"], kind="stable"
        ).reset_index(drop=True),
        matched_pairs=matched_pairs.sort_values(
            GROUP_KEYS + ["manual_start_time_ms"], kind="stable"
        ).reset_index(drop=True),
        summary=summary,
        per_group=per_group,
        split_summary=split_summary.reset_index(drop=True),
        test_channel_expansion=test_channel_expansion.sort_values("channel_count").reset_index(
            drop=True
        ),
        quality_ladder=quality_ladder,
    )


def _rectangle(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    *,
    edgecolor: str | None = None,
    linewidth: float = 0.7,
    alpha: float = 1.0,
) -> Rectangle:
    patch = Rectangle(
        (x, y),
        width,
        height,
        facecolor=color,
        edgecolor=edgecolor or color,
        linewidth=linewidth,
        alpha=alpha,
    )
    axis.add_patch(patch)
    return patch


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.035,
        1.03,
        label,
        transform=axis.transAxes,
        fontsize=8.2,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
    )


def _clean_axis(axis: plt.Axes) -> None:
    axis.spines["right"].set_visible(False)
    axis.spines["top"].set_visible(False)
    axis.tick_params(length=2.5, pad=2)


def _plot_split_logic(axis: plt.Axes, tables: AccountingTables) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    _panel_label(axis, "a")
    axis.text(0, 0.99, "188 个测试事件是怎样进入评估的", fontsize=8.2, fontweight="bold", va="top")
    axis.text(
        0,
        0.84,
        "先按鼠将发作上下文块与背景块做 70/15/15 平衡分配；同一块及其 12 s 保护上下文只属于一个集合。",
        fontsize=6.5,
        color=MID_GREY,
        va="top",
    )
    boxes = [
        (0.02, "1,061\n独立临床事件", "人工事件层"),
        (0.30, "1,233\n通道事件行", "按有效通道展开"),
        (0.66, "855 / 190 / 188", "训练 / 验证 / 测试"),
    ]
    for x, main, sub in boxes:
        _rectangle(axis, x, 0.29, 0.22 if x < 0.6 else 0.31, 0.33, VERY_LIGHT_GREY, edgecolor="#BFC5C9")
        axis.text(x + (0.11 if x < 0.6 else 0.155), 0.48, main, ha="center", va="center", fontsize=8.1, fontweight="bold", color=INK)
        axis.text(x + (0.11 if x < 0.6 else 0.155), 0.34, sub, ha="center", va="center", fontsize=6.0, color=MID_GREY)
    for start, end in [(0.24, 0.30), (0.52, 0.66)]:
        axis.add_patch(FancyArrowPatch((start, 0.455), (end, 0.455), arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color=MID_GREY))
    axis.text(0.41, 0.22, "+172 个第二通道副本", ha="center", fontsize=6.1, color=MANUAL_COLOR)
    test_expansion = tables.test_channel_expansion.set_index("channel_count")
    single = int(test_expansion.loc[1, "clinical_event_count"]) if 1 in test_expansion.index else 0
    dual = int(test_expansion.loc[2, "clinical_event_count"]) if 2 in test_expansion.index else 0
    axis.text(
        0.815,
        0.18,
        f"测试集：160 个临床事件 = {single} 个单通道 + {dual} 个双通道\n通道展开：{single} + 2×{dual} = 188",
        ha="center",
        va="top",
        fontsize=6.4,
        color=INK,
    )
    axis.text(0.02, 0.08, "E77 CH7 已排除；测试标签没有从 1,233 行中再次随机抽样。", fontsize=6.2, color=MISS_COLOR)


def _plot_accounting_hero(axis: plt.Axes, tables: AccountingTables) -> None:
    axis.set_xlim(-70, 370)
    axis.set_ylim(-0.65, 2.3)
    axis.axis("off")
    _panel_label(axis, "b")
    axis.text(-65, 2.23, "人工标签与模型输出不是同一个计数过程", fontsize=8.2, fontweight="bold", va="top")
    axis.text(
        -65,
        2.00,
        "模型在完整测试时间上连续扫描，因此预测段数可以少于、等于或多于人工标签数。",
        fontsize=6.5,
        color=MID_GREY,
        va="top",
    )
    rows = [
        (
            "人工标签",
            188,
            [(143, MATCH_COLOR, "143\n一对一配对"), (10, MERGE_COLOR, "10"), (35, MISS_COLOR, "35\n零重叠")],
            1.22,
        ),
        (
            "模型输出",
            351,
            [(143, MATCH_COLOR, "143\n一对一配对"), (34, FRAGMENT_COLOR, "34\n重叠冗余"), (174, CANDIDATE_COLOR, "174\n零人工重叠")],
            0.22,
        ),
    ]
    for label, total, segments, y in rows:
        axis.text(-65, y + 0.22, f"{label}\n{total}", ha="left", va="center", fontsize=7.1, fontweight="bold", color=INK)
        cursor = 0.0
        for count, color, text_value in segments:
            _rectangle(axis, cursor, y, count, 0.45, color, edgecolor="white", linewidth=0.7)
            if count >= 8:
                axis.text(cursor + count / 2, y + 0.225, text_value, ha="center", va="center", fontsize=5.7 if count < 30 else 6.3, color="white", fontweight="bold")
            else:
                axis.text(cursor + count / 2, y + 0.56, text_value, ha="center", va="bottom", fontsize=6.0, color=color, fontweight="bold")
            cursor += count
    axis.plot([143, 143], [0.67, 1.22], color=MATCH_COLOR, linewidth=1.0)
    axis.text(151, 0.94, "同一批 143 对", fontsize=6.1, color=MATCH_COLOR, va="center")
    axis.text(0, -0.20, "配对条件：同鼠 + 同记录 + 同通道；IoU > 0；匈牙利算法最大化总 IoU。", fontsize=6.2, color=INK)
    axis.text(0, -0.45, "因此 143 表示“有正重叠且获得唯一配对”，不表示 143 个事件都被完整覆盖。", fontsize=6.2, color=MISS_COLOR)


def _timeline_row(
    axis: plt.Axes,
    y: float,
    label: str,
    manual_spans: list[tuple[float, float]],
    predicted_spans: list[tuple[float, float]],
    note: str,
) -> None:
    axis.text(-0.02, y, label, transform=axis.transAxes, ha="right", va="center", fontsize=6.1, fontweight="bold", color=INK)
    x0, x1 = 0.04, 0.72
    axis.plot([x0, x1], [y, y], transform=axis.transAxes, color=LIGHT_GREY, linewidth=2.0, solid_capstyle="butt")
    for start, end in manual_spans:
        _rectangle(axis, x0 + (x1 - x0) * start, y + 0.025, (x1 - x0) * (end - start), 0.055, MANUAL_COLOR, edgecolor=MANUAL_COLOR, alpha=0.45)
    for start, end in predicted_spans:
        _rectangle(axis, x0 + (x1 - x0) * start, y - 0.080, (x1 - x0) * (end - start), 0.055, PREDICTION_COLOR, edgecolor=PREDICTION_COLOR, alpha=0.75)
    axis.text(0.76, y, note, transform=axis.transAxes, ha="left", va="center", fontsize=5.9, color=MID_GREY)


def _plot_why_more(axis: plt.Axes) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    _panel_label(axis, "c")
    axis.text(0, 0.98, "351 为什么会多于 188", fontsize=8.0, fontweight="bold", va="top")
    _timeline_row(axis, 0.75, "切碎", [(0.12, 0.88)], [(0.10, 0.31), (0.38, 0.60), (0.66, 0.91)], "7 个标签被多段预测覆盖")
    _timeline_row(axis, 0.53, "合并", [(0.10, 0.34), (0.57, 0.84)], [(0.06, 0.90)], "11 个预测跨越多个标签")
    _timeline_row(axis, 0.31, "标签外", [], [(0.32, 0.65)], "174 个零人工重叠候选")
    _timeline_row(axis, 0.09, "漏检", [(0.30, 0.68)], [], "35 个人工标签零预测重叠")
    axis.text(0.98, 0.91, "蓝 = 人工  |  橙 = 模型", transform=axis.transAxes, fontsize=5.6, color=MID_GREY, ha="right")


def _plot_quality_ladder(axis: plt.Axes, tables: AccountingTables) -> None:
    _panel_label(axis, "d")
    axis.set_title("“检测到”随严格标准逐级收紧", loc="left", fontsize=8.0, fontweight="bold", pad=8)
    ladder = tables.quality_ladder.set_index("criterion")["manual_event_count"]
    labels = ["任意正重叠", "一对一配对", "IoU ≥ 0.5", "IoU ≥ 0.7"]
    criteria = ["any_positive_overlap", "one_to_one_match_iou_gt_0", "matched_iou_ge_0_5", "matched_iou_ge_0_7"]
    values = [int(ladder[item]) for item in criteria]
    y = np.arange(len(labels))[::-1]
    colors = [MANUAL_COLOR, MATCH_COLOR, "#5B7488", "#7A8791"]
    axis.barh(y, values, height=0.52, color=colors, edgecolor="none")
    for yi, value in zip(y, values, strict=True):
        axis.text(value + 3, yi, f"{value}/188", va="center", fontsize=6.4, color=INK, fontweight="bold")
    axis.set_yticks(y, labels)
    axis.set_xlim(0, 200)
    axis.set_xlabel("人工通道事件数")
    axis.set_xticks([0, 50, 100, 150, 188])
    axis.axvline(188, color=LIGHT_GREY, linewidth=0.7)
    _clean_axis(axis)
    axis.text(0, -0.36, "正式报告必须同时写明匹配门槛、IoU、覆盖率和边界误差。", transform=axis.transAxes, fontsize=5.9, color=MISS_COLOR)


def plot_event_accounting_summary(tables: AccountingTables, output_stem: Path) -> None:
    fig = plt.figure(figsize=(7.204724409448819, 7.35))  # 183 mm wide.
    grid = fig.add_gridspec(3, 2, height_ratios=(1.45, 1.78, 1.55), hspace=0.50, wspace=0.42)
    _plot_split_logic(fig.add_subplot(grid[0, :]), tables)
    _plot_accounting_hero(fig.add_subplot(grid[1, :]), tables)
    _plot_why_more(fig.add_subplot(grid[2, 0]))
    _plot_quality_ladder(fig.add_subplot(grid[2, 1]), tables)
    fig.suptitle("测试事件划分与模型区间账目", x=0.06, y=0.997, ha="left", fontsize=10.2, fontweight="bold", color=INK)
    fig.text(0.06, 0.969, "Physio-Sequence（无辅助统计头）| 冻结测试集 | 通道独立评估", ha="left", fontsize=6.5, color=MID_GREY)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.935, bottom=0.075)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".svg"))
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig)


def _entries_from_tables(tables: AccountingTables) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in tables.matched_pairs.itertuples(index=False):
        rows.append(
            {
                "entry_uid": str(row.pair_uid),
                "entity_kind": "pair",
                "category": MATCHED,
                **{column: str(getattr(row, column)) for column in GROUP_KEYS},
                "event_id": str(row.event_id),
                "predicted_event_index": int(row.predicted_event_index),
                "primary_start_time_ms": min(float(row.manual_start_time_ms), float(row.predicted_start_time_ms)),
                "primary_end_time_ms": max(float(row.manual_end_time_ms), float(row.predicted_end_time_ms)),
                "duration_s": float(row.manual_duration_s),
                "iou": float(row.iou),
                "union_coverage": float(row.union_coverage),
                "onset_error_seconds": float(row.onset_error_seconds),
                "offset_error_seconds": float(row.offset_error_seconds),
                "overlap_count": int(row.overlapping_predicted_count),
                "peak_seizure_probability": math.nan,
            }
        )
    for row in tables.manual.loc[tables.manual["category"].ne(MATCHED)].itertuples(index=False):
        rows.append(
            {
                "entry_uid": str(row.manual_uid),
                "entity_kind": "manual",
                "category": str(row.category),
                **{column: str(getattr(row, column)) for column in GROUP_KEYS},
                "event_id": str(row.event_id),
                "predicted_event_index": pd.NA,
                "primary_start_time_ms": float(row.start_time_ms),
                "primary_end_time_ms": float(row.end_time_ms),
                "duration_s": float(row.duration_s),
                "iou": math.nan,
                "union_coverage": float(row.union_coverage),
                "onset_error_seconds": math.nan,
                "offset_error_seconds": math.nan,
                "overlap_count": int(row.overlapping_predicted_count),
                "peak_seizure_probability": float(row.peak_seizure_probability),
            }
        )
    for row in tables.prediction.loc[tables.prediction["category"].ne(MATCHED)].itertuples(index=False):
        rows.append(
            {
                "entry_uid": str(row.prediction_uid),
                "entity_kind": "prediction",
                "category": str(row.category),
                **{column: str(getattr(row, column)) for column in GROUP_KEYS},
                "event_id": "",
                "predicted_event_index": int(row.predicted_event_index),
                "primary_start_time_ms": float(row.start_time_ms),
                "primary_end_time_ms": float(row.end_time_ms),
                "duration_s": float(row.duration_s),
                "iou": math.nan,
                "union_coverage": math.nan,
                "onset_error_seconds": math.nan,
                "offset_error_seconds": math.nan,
                "overlap_count": int(row.overlapping_manual_count),
                "peak_seizure_probability": float(row.peak_seizure_probability),
            }
        )
    entries = pd.DataFrame(rows)
    entries["predicted_event_index"] = entries["predicted_event_index"].astype("Int64")
    entries["category_order"] = pd.Categorical(
        entries["category"], categories=CATEGORY_ORDER, ordered=True
    )
    return entries.sort_values(
        ["category_order"] + GROUP_KEYS + ["primary_start_time_ms"], kind="stable"
    ).drop(columns="category_order").reset_index(drop=True)


def _select_evenly(frame: pd.DataFrame, count: int, sort_column: str) -> pd.DataFrame:
    if len(frame) <= count:
        return frame.copy()
    ordered = frame.sort_values(sort_column, kind="stable").reset_index(drop=True)
    positions = np.unique(np.linspace(0, len(ordered) - 1, count, dtype=int))
    return ordered.iloc[positions].copy()


def _quick_entries(entries: pd.DataFrame) -> pd.DataFrame:
    matched = _select_evenly(entries.loc[entries["category"].eq(MATCHED)], 8, "iou")
    misses = entries.loc[entries["category"].eq(MANUAL_MISS)]
    merged = entries.loc[entries["category"].eq(MANUAL_MERGED)]
    fragments = entries.loc[entries["category"].eq(PRED_FRAGMENT)]
    candidates = entries.loc[entries["category"].eq(PRED_CANDIDATE)].copy()
    candidates["_peak"] = candidates["peak_seizure_probability"].fillna(-1.0)
    by_peak = candidates.sort_values(["_peak", "duration_s"], ascending=[False, False]).head(6)
    by_duration = candidates.sort_values("duration_s", ascending=False).head(8)
    candidate_review = (
        pd.concat([by_peak, by_duration], ignore_index=True)
        .drop_duplicates("entry_uid")
        .head(12)
        .drop(columns="_peak")
    )
    quick = pd.concat(
        [matched, misses, merged, fragments, candidate_review], ignore_index=True
    )
    quick["category_order"] = pd.Categorical(
        quick["category"], categories=CATEGORY_ORDER, ordered=True
    )
    return quick.sort_values(
        ["category_order"] + GROUP_KEYS + ["primary_start_time_ms"], kind="stable"
    ).drop(columns="category_order").reset_index(drop=True)


def _downsample_envelope(
    x: np.ndarray, y: np.ndarray, maximum_points: int = 2200
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    if len(x) <= maximum_points:
        return x, None, y, None
    bins = max(maximum_points // 2, 1)
    edges = np.linspace(0, len(x), bins + 1, dtype=int)
    centers = np.empty(bins, dtype=float)
    lower = np.empty(bins, dtype=float)
    median = np.empty(bins, dtype=float)
    upper = np.empty(bins, dtype=float)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        right = max(right, left + 1)
        block = y[left:right]
        centers[index] = float(np.mean(x[left:right]))
        lower[index] = float(np.min(block))
        median[index] = float(np.median(block))
        upper[index] = float(np.max(block))
    return centers, lower, median, upper


def _format_seconds(value: float) -> str:
    sign = "-" if value < 0 else ""
    value = abs(float(value))
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = value % 60
    if hours:
        return f"{sign}{hours:d}:{minutes:02d}:{seconds:04.1f}"
    return f"{sign}{minutes:02d}:{seconds:04.1f}"


def _entry_metadata(entry: Any) -> tuple[str, str]:
    identity = f"{entry.animal_id} | {entry.session_id} | {entry.channel_name}"
    duration = f"时长 {float(entry.duration_s):.1f} s"
    if entry.category == MATCHED:
        title = f"{identity} | {entry.event_id} 与 P{int(entry.predicted_event_index)}"
        detail = (
            f"{duration} | IoU {float(entry.iou):.3f} | 人工联合覆盖 {float(entry.union_coverage):.1%} | "
            f"起点误差 {float(entry.onset_error_seconds):+.1f} s | 终点误差 {float(entry.offset_error_seconds):+.1f} s"
        )
    elif entry.entity_kind == "manual":
        title = f"{identity} | 人工 {entry.event_id}"
        detail = (
            f"{duration} | 重叠模型段 {int(entry.overlap_count)} | 人工联合覆盖 {float(entry.union_coverage):.1%}"
        )
    else:
        title = f"{identity} | 模型 P{int(entry.predicted_event_index)}"
        peak = float(entry.peak_seizure_probability)
        peak_text = f"{peak:.3f}" if np.isfinite(peak) else "NA"
        detail = f"{duration} | 重叠人工标签 {int(entry.overlap_count)} | 区间内峰值概率 {peak_text}"
    return title, detail


def _entry_crop(entry: Any) -> tuple[float, float]:
    start = float(entry.primary_start_time_ms)
    end = float(entry.primary_end_time_ms)
    duration = max(end - start, 1000.0)
    context = min(max(8_000.0, duration * 0.05), 30_000.0)
    return start - context, end + context


def _group_filter(frame: pd.DataFrame, entry: Any) -> pd.DataFrame:
    result = frame
    for column in GROUP_KEYS:
        result = result.loc[result[column].astype(str).eq(str(getattr(entry, column)))]
    return result


def _draw_review_panel(
    axis: plt.Axes,
    entry: Any,
    truth: pd.DataFrame,
    predicted: pd.DataFrame,
    probabilities: pd.DataFrame,
    traces: TraceStore,
    *,
    show_xlabel: bool,
) -> None:
    crop_start, crop_end = _entry_crop(entry)
    truth_group = _group_filter(truth, entry)
    predicted_group = _group_filter(predicted, entry)
    manual_spans = truth_group.loc[
        truth_group["end_time_ms"].gt(crop_start)
        & truth_group["start_time_ms"].lt(crop_end)
    ]
    prediction_spans = predicted_group.loc[
        predicted_group["end_time_ms"].gt(crop_start)
        & predicted_group["start_time_ms"].lt(crop_end)
    ]
    time_ms, waveform = traces.load(
        str(entry.animal_id),
        str(entry.session_id),
        str(entry.channel_name),
        crop_start,
        crop_end,
    )
    x = (time_ms - crop_start) / 1000.0
    plot_x, lower, median, upper = _downsample_envelope(x, waveform)
    if lower is None or upper is None:
        axis.plot(plot_x, median, color=INK, linewidth=0.45, zorder=2)
    else:
        axis.fill_between(plot_x, lower, upper, color="#4B5054", alpha=0.30, linewidth=0, zorder=1)
        axis.plot(plot_x, median, color=INK, linewidth=0.35, zorder=2)

    primary_manual_id = str(entry.event_id) if entry.entity_kind in {"pair", "manual"} else None
    primary_prediction_index = (
        int(entry.predicted_event_index)
        if entry.entity_kind in {"pair", "prediction"} and not pd.isna(entry.predicted_event_index)
        else None
    )
    for span in manual_spans.itertuples(index=False):
        left = (float(span.start_time_ms) - crop_start) / 1000.0
        right = (float(span.end_time_ms) - crop_start) / 1000.0
        primary = primary_manual_id is not None and str(span.event_id) == primary_manual_id
        axis.axvspan(left, right, color=MANUAL_COLOR, alpha=0.13 if primary else 0.07, linewidth=0, zorder=0)
        axis.axvline(left, color=MANUAL_COLOR, linewidth=1.0 if primary else 0.55, linestyle="--")
        axis.axvline(right, color=MANUAL_COLOR, linewidth=1.0 if primary else 0.55, linestyle="--")
    for span in prediction_spans.itertuples(index=False):
        left = (float(span.start_time_ms) - crop_start) / 1000.0
        right = (float(span.end_time_ms) - crop_start) / 1000.0
        primary = (
            primary_prediction_index is not None
            and int(span.predicted_event_index) == primary_prediction_index
        )
        axis.axvspan(left, right, color=PREDICTION_COLOR, alpha=0.14 if primary else 0.07, linewidth=0, zorder=0)
        axis.axvline(left, color=PREDICTION_COLOR, linewidth=1.0 if primary else 0.55)
        axis.axvline(right, color=PREDICTION_COLOR, linewidth=1.0 if primary else 0.55)

    probability = _group_filter(probabilities, entry)
    probability = probability.loc[
        probability["center_time_ms"].ge(crop_start)
        & probability["center_time_ms"].le(crop_end)
    ]
    twin = axis.twinx()
    if not probability.empty:
        twin.plot(
            (probability["center_time_ms"].to_numpy(float) - crop_start) / 1000.0,
            probability["seizure_probability"].to_numpy(float),
            color=PROBABILITY_COLOR,
            linewidth=0.75,
            alpha=0.95,
            zorder=3,
        )
    twin.set_ylim(0, 1.02)
    twin.set_yticks([0, 0.5, 1.0])
    twin.set_ylabel("P", color=PROBABILITY_COLOR, labelpad=1)
    twin.tick_params(axis="y", colors=PROBABILITY_COLOR, length=2, pad=1)
    twin.spines["top"].set_visible(False)
    twin.spines["right"].set_color(PROBABILITY_COLOR)

    title, detail = _entry_metadata(entry)
    category = CATEGORY_META[str(entry.category)]
    axis.set_title(title, loc="left", fontsize=7.1, fontweight="bold", color=INK, pad=10)
    axis.text(0, 1.025, detail, transform=axis.transAxes, ha="left", va="bottom", fontsize=5.8, color=MID_GREY)
    axis.text(1.0, 1.11, category["short_zh"], transform=axis.transAxes, ha="right", va="bottom", fontsize=6.0, color=category["color"], fontweight="bold")
    axis.set_ylabel("相对幅值 (a.u.)")
    axis.set_xlabel("相对本页裁剪起点的时间 (s)" if show_xlabel else "")
    if not show_xlabel:
        axis.tick_params(labelbottom=False)
    axis.set_xlim(0, max((crop_end - crop_start) / 1000.0, 1.0))
    _clean_axis(axis)


def _title_page(pdf: PdfPages, kind: str, entries: pd.DataFrame, total_pages: int) -> None:
    fig = plt.figure(figsize=A4_PORTRAIT)
    axis = fig.add_axes([0.08, 0.08, 0.84, 0.84])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(0, 0.96, "发作事件账目临床审阅册", fontsize=20, fontweight="bold", color=INK, va="top")
    subtitle = "快速审阅版" if kind == "quick" else "完整逐条审计版"
    axis.text(0, 0.88, subtitle, fontsize=12, fontweight="bold", color=MATCH_COLOR, va="top")
    axis.text(0, 0.80, "冻结测试集 | Physio-Sequence（无辅助统计头）| 单通道盲法连续扫描", fontsize=8.5, color=MID_GREY, va="top")
    axis.plot([0, 1], [0.755, 0.755], color=LIGHT_GREY, linewidth=1.0)
    counts = entries["category"].value_counts()
    y = 0.68
    for category in CATEGORY_ORDER:
        count = int(counts.get(category, 0))
        meta = CATEGORY_META[category]
        _rectangle(axis, 0.0, y - 0.018, 0.018, 0.036, meta["color"], edgecolor=meta["color"])
        axis.text(0.035, y, f"{meta['short_zh']}  {count}", fontsize=9.0, fontweight="bold", color=INK, va="center")
        axis.text(0.35, y, meta["definition_zh"], fontsize=7.0, color=MID_GREY, va="center", wrap=True)
        y -= 0.095
    axis.text(0, 0.16, f"本册共 {total_pages} 页。每条记录均显示滤波后波形、人工区间、模型区间和连续概率。", fontsize=8.0, color=INK)
    axis.text(0, 0.11, "颜色：蓝色 = 人工标签；橙色 = 模型区间；青绿色 = 发作概率。幅值为相对基线 a.u.，不作绝对微伏解释。", fontsize=7.2, color=MID_GREY)
    axis.text(0, 0.04, "研究用途：零人工重叠候选需临床复核，不能仅凭当前标签自动认定为生理学假阳性。", fontsize=7.2, color=MISS_COLOR)
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def _toc_page(
    pdf: PdfPages,
    sections: list[tuple[str, pd.DataFrame]],
    section_starts: Mapping[str, int],
    total_pages: int,
) -> None:
    fig = plt.figure(figsize=A4_PORTRAIT)
    axis = fig.add_axes([0.09, 0.08, 0.82, 0.84])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(0, 0.98, "目录与读图规则", fontsize=16, fontweight="bold", color=INK, va="top")
    axis.text(0, 0.90, "每一节先给出类别定义，再逐条展示。页码也写入配套 CSV 索引，便于按事件 ID 定位。", fontsize=8.0, color=MID_GREY, va="top")
    y = 0.80
    for index, (category, frame) in enumerate(sections, start=1):
        meta = CATEGORY_META[category]
        _rectangle(axis, 0.0, y - 0.025, 0.020, 0.050, meta["color"], edgecolor=meta["color"])
        axis.text(0.04, y, f"{index}. {meta['title_zh']}", fontsize=9.0, fontweight="bold", color=INK, va="center")
        axis.text(0.78, y, f"{len(frame)} 条", fontsize=8.2, color=MID_GREY, ha="right", va="center")
        axis.text(0.98, y, f"p. {section_starts[category]}", fontsize=8.2, color=meta["color"], ha="right", va="center", fontweight="bold")
        y -= 0.105
    axis.plot([0, 1], [0.25, 0.25], color=LIGHT_GREY, linewidth=1.0)
    axis.text(0, 0.20, "边界线：人工起止为蓝色虚线；模型起止为橙色实线。浅色区间表示持续区间。", fontsize=7.4, color=INK)
    axis.text(0, 0.15, "匹配页会显示 IoU、人工联合覆盖率以及模型起止相对人工边界的误差。", fontsize=7.4, color=INK)
    axis.text(0, 0.10, "完整册中的每条记录只归入一个账目类别；总条目数 = 143 + 35 + 10 + 34 + 174 = 396。", fontsize=7.4, color=INK)
    axis.text(1, 0.02, f"2 / {total_pages}", fontsize=7.0, color=MID_GREY, ha="right")
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def _section_page(
    pdf: PdfPages,
    category: str,
    count: int,
    page_number: int,
    total_pages: int,
) -> None:
    meta = CATEGORY_META[category]
    fig = plt.figure(figsize=A4_PORTRAIT)
    axis = fig.add_axes([0.10, 0.12, 0.80, 0.76])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    _rectangle(axis, 0.0, 0.70, 0.035, 0.17, meta["color"], edgecolor=meta["color"])
    axis.text(0.07, 0.84, meta["title_zh"], fontsize=17, fontweight="bold", color=INK, va="top", wrap=True)
    axis.text(0.07, 0.64, f"本节 {count} 条", fontsize=11, fontweight="bold", color=meta["color"], va="top")
    axis.text(0.07, 0.52, meta["definition_zh"], fontsize=10, color=MID_GREY, va="top", wrap=True, linespacing=1.5)
    if category == MATCHED:
        caution = "注意：当前门槛只要求 IoU > 0。必须结合 IoU、覆盖率和边界误差判断是否接近完整打标。"
    elif category == PRED_CANDIDATE:
        caution = "注意：这里的“零人工重叠”是与现有标签比较的结果，不是已经经过临床复核的最终假阳性。"
    else:
        caution = "本节条目与其他类别互斥；附近但不相交的区间仍会作为上下文显示。"
    axis.text(0.07, 0.34, caution, fontsize=8.2, color=MISS_COLOR, va="top", wrap=True)
    axis.text(1, 0.02, f"{page_number} / {total_pages}", fontsize=7.0, color=MID_GREY, ha="right")
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def _atlas_plan(
    entries: pd.DataFrame, panels_per_page: int = 4
) -> tuple[list[tuple[str, pd.DataFrame]], dict[str, int], int]:
    sections = [
        (category, entries.loc[entries["category"].eq(category)].reset_index(drop=True))
        for category in CATEGORY_ORDER
        if entries["category"].eq(category).any()
    ]
    page = 3
    starts: dict[str, int] = {}
    for category, frame in sections:
        starts[category] = page
        page += 1 + math.ceil(len(frame) / panels_per_page)
    return sections, starts, page - 1


def _write_review_atlas(
    path: Path,
    entries: pd.DataFrame,
    truth: pd.DataFrame,
    predicted: pd.DataFrame,
    probabilities: pd.DataFrame,
    traces: TraceStore,
    *,
    kind: str,
    index_csv: Path,
    panels_per_page: int = 4,
) -> int:
    sections, section_starts, total_pages = _atlas_plan(entries, panels_per_page)
    index_rows: list[dict[str, Any]] = []
    with PdfPages(path) as pdf:
        _title_page(pdf, kind, entries, total_pages)
        _toc_page(pdf, sections, section_starts, total_pages)
        current_page = 3
        for category, frame in sections:
            _section_page(pdf, category, len(frame), current_page, total_pages)
            current_page += 1
            for page_start in range(0, len(frame), panels_per_page):
                page = frame.iloc[page_start : page_start + panels_per_page]
                fig, axes = plt.subplots(
                    panels_per_page,
                    1,
                    figsize=A4_PORTRAIT,
                    squeeze=False,
                    gridspec_kw={"hspace": 0.62},
                )
                for panel_index, (axis, entry) in enumerate(
                    zip(axes[:, 0], page.itertuples(index=False), strict=False)
                ):
                    _draw_review_panel(
                        axis,
                        entry,
                        truth,
                        predicted,
                        probabilities,
                        traces,
                        show_xlabel=True,
                    )
                    index_rows.append(
                        {
                            "atlas": kind,
                            "category": category,
                            "entry_uid": str(entry.entry_uid),
                            "entity_kind": str(entry.entity_kind),
                            **{column: str(getattr(entry, column)) for column in GROUP_KEYS},
                            "event_id": str(entry.event_id),
                            "predicted_event_index": (
                                int(entry.predicted_event_index)
                                if not pd.isna(entry.predicted_event_index)
                                else pd.NA
                            ),
                            "pdf_page": current_page,
                            "panel": "abcd"[panel_index],
                        }
                    )
                for axis in axes[len(page) :, 0]:
                    axis.axis("off")
                meta = CATEGORY_META[category]
                fig.suptitle(
                    f"{meta['short_zh']} | {page_start + 1}-{page_start + len(page)} / {len(frame)}",
                    x=0.07,
                    y=0.992,
                    ha="left",
                    fontsize=9.0,
                    fontweight="bold",
                    color=meta["color"],
                )
                fig.text(
                    0.93,
                    0.987,
                    f"{current_page} / {total_pages}",
                    ha="right",
                    va="top",
                    fontsize=6.5,
                    color=MID_GREY,
                )
                fig.text(
                    0.07,
                    0.012,
                    "蓝色虚线/浅蓝区 = 人工；橙色实线/浅橙区 = 模型；青绿色 = 连续发作概率。",
                    fontsize=5.8,
                    color=MID_GREY,
                )
                fig.subplots_adjust(left=0.085, right=0.915, top=0.955, bottom=0.055)
                pdf.savefig(fig, dpi=300)
                plt.close(fig)
                current_page += 1
    index = pd.DataFrame(index_rows)
    if not index.empty:
        index["predicted_event_index"] = index["predicted_event_index"].astype("Int64")
    index.to_csv(index_csv, index=False, encoding="utf-8-sig")
    if current_page - 1 != total_pages:
        raise ValueError(
            f"atlas page accounting failed for {kind}: {current_page - 1} != {total_pages}"
        )
    return total_pages


def _write_tables(tables: AccountingTables, source_dir: Path) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "manual_event_accounting.csv": tables.manual,
        "predicted_event_accounting.csv": tables.prediction,
        "matched_pair_accounting.csv": tables.matched_pairs,
        "event_accounting_summary.csv": tables.summary,
        "per_recording_channel_accounting.csv": tables.per_group,
        "split_event_summary.csv": tables.split_summary,
        "test_channel_expansion_summary.csv": tables.test_channel_expansion,
        "match_quality_ladder.csv": tables.quality_ladder,
    }
    for filename, frame in frames.items():
        frame.to_csv(source_dir / filename, index=False, encoding="utf-8-sig")


def _write_readme(
    output_dir: Path,
    tables: AccountingTables,
    quick_pages: int,
    complete_pages: int,
) -> None:
    counts = tables.summary.set_index(["entity", "category"])["count"]
    text = f"""# 测试事件账目说明

## 188 是怎么来的

- 全部 1,061 个独立临床事件按有效通道展开为 1,233 个通道事件行。
- 划分对象是每只鼠内部的发作上下文块和背景块，比例为 70/15/15；同一事件块及 12 秒上下文不会跨集合。
- 测试时间块内保留全部事件，共 160 个独立临床事件；其中 132 个单通道、28 个双通道，所以通道事件数为 `132 + 2×28 = 188`。
- 188 不是从 1,233 行中另行随机抽取的事件样本。
- 70/15/15 是事件数或背景时长的权重目标，不代表每个记录段都恰好拥有三种 split；例如 E76 CH8 的测试人工发作数为 0。
- 当前完整通道时间实际约为 69.04% / 17.41% / 13.55%，测试暴露时间为 25.93 channel-hours。

## 351 是什么

351 是模型对完整测试时间连续扫描后独立输出的区间数，不受人工标签数约束：

- 一对一正重叠配对：{int(counts.loc[("manual", MATCHED)])}
- 人工有重叠但未配对：{int(counts.loc[("manual", MANUAL_MERGED)])}
- 人工零重叠漏检：{int(counts.loc[("manual", MANUAL_MISS)])}
- 预测有人工重叠但未配对：{int(counts.loc[("prediction", PRED_FRAGMENT)])}
- 预测零人工重叠候选：{int(counts.loc[("prediction", PRED_CANDIDATE)])}

匹配限定在同一鼠、同一记录、同一通道，当前门槛为 `IoU > 0`，再用匈牙利算法最大化总 IoU。因此“一对一配对”不等于“完整覆盖”。

## 文件怎么读

- `event_accounting_summary.pdf/png/svg/tiff`：一张主图解释划分、互斥账目和严格度阶梯。
- `event_accounting_quick_review.pdf`：{quick_pages} 页，含代表性配对、全部人工零重叠漏检、全部合并冲突、全部重叠冗余段及高优先级标签外候选。
- `event_accounting_complete_atlas.pdf`：{complete_pages} 页，完整展示 396 条互斥账目记录。
- `source_data/*accounting.csv`：逐行账本；两个 `atlas_index.csv` 给出 PDF 页码和面板位置。

颜色固定为：蓝色人工标签、橙色模型区间、青绿色连续概率。波形幅值使用相对基线 a.u.，不解释为绝对微伏。
"""
    (output_dir / "README_zh.md").write_text(text, encoding="utf-8")


def generate_event_accounting_review(
    run_root: str | Path,
    protocol_dir: str | Path,
    output_dir: str | Path,
) -> ReviewBundle:
    run_root = Path(run_root).resolve()
    protocol_dir = Path(protocol_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_data"
    model_root = run_root / "physio_sequence_no_aux"
    all_channel_events = pd.read_parquet(protocol_dir / "channel_events.parquet")
    truth = all_channel_events.loc[all_channel_events["split"].astype(str).eq("test")].copy()
    predicted = pd.read_parquet(model_root / "events" / "test_predicted_events.parquet")
    matches = pd.read_parquet(model_root / "events" / "test_event_matches.parquet")
    assessments = pd.read_parquet(
        model_root / "events" / "test_true_event_assessments.parquet"
    )
    probabilities = pd.read_parquet(
        model_root / "continuous" / "continuous_predictions.parquet"
    )
    probabilities = probabilities.loc[probabilities["split"].astype(str).eq("test")].copy()
    tables = build_event_accounting_tables(
        truth,
        predicted,
        matches,
        assessments,
        probabilities,
        all_channel_events=all_channel_events,
    )
    _write_tables(tables, source_dir)
    summary_stem = output_dir / "event_accounting_summary"
    plot_event_accounting_summary(tables, summary_stem)

    entries = _entries_from_tables(tables)
    if len(entries) != 396:
        raise ValueError(f"expected 396 mutually exclusive review entries, found {len(entries)}")
    quick_entries = _quick_entries(entries)
    entries.to_csv(source_dir / "complete_atlas_entries.csv", index=False, encoding="utf-8-sig")
    quick_entries.to_csv(source_dir / "quick_review_entries.csv", index=False, encoding="utf-8-sig")
    trace_store = TraceStore(protocol_dir)
    quick_pdf = output_dir / "event_accounting_quick_review.pdf"
    complete_pdf = output_dir / "event_accounting_complete_atlas.pdf"
    quick_pages = _write_review_atlas(
        quick_pdf,
        quick_entries,
        truth,
        predicted,
        probabilities,
        trace_store,
        kind="quick",
        index_csv=source_dir / "quick_review_atlas_index.csv",
    )
    complete_pages = _write_review_atlas(
        complete_pdf,
        entries,
        truth,
        predicted,
        probabilities,
        trace_store,
        kind="complete",
        index_csv=source_dir / "complete_atlas_index.csv",
    )
    _write_readme(output_dir, tables, quick_pages, complete_pages)
    counts = {
        "manual_events": int(len(tables.manual)),
        "predicted_intervals": int(len(tables.prediction)),
        "matched_pairs": int(len(tables.matched_pairs)),
        "manual_zero_overlap": int((tables.manual["category"] == MANUAL_MISS).sum()),
        "manual_overlap_unmatched": int(
            (tables.manual["category"] == MANUAL_MERGED).sum()
        ),
        "prediction_overlap_unmatched": int(
            (tables.prediction["category"] == PRED_FRAGMENT).sum()
        ),
        "prediction_zero_overlap": int(
            (tables.prediction["category"] == PRED_CANDIDATE).sum()
        ),
    }
    manifest = {
        "schema_version": 1,
        "figure_contract": {
            "core_conclusion": (
                "The 188 held-out channel labels are all events in frozen test regions, "
                "whereas 351 intervals arise from independent continuous model scanning."
            ),
            "archetype": "schematic-led clinical composite",
            "backend": "python/matplotlib",
            "manuscript_width_mm": 183,
            "atlas_page_size": "A4 portrait",
            "matching_rule": "same animal/session/channel; IoU > 0; maximum-total-IoU one-to-one assignment",
            "amplitude_unit": "relative baseline a.u.; absolute ADC-to-microvolt calibration unavailable",
        },
        "counts": counts,
        "quick_review_pages": quick_pages,
        "complete_atlas_pages": complete_pages,
        "quick_review_entries": int(len(quick_entries)),
        "complete_atlas_entries": int(len(entries)),
        "outputs": {
            "summary_pdf": str(summary_stem.with_suffix(".pdf")),
            "quick_pdf": str(quick_pdf),
            "complete_pdf": str(complete_pdf),
        },
    }
    (output_dir / "event_accounting_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ReviewBundle(
        output_dir=output_dir,
        summary_figure_pdf=summary_stem.with_suffix(".pdf"),
        quick_review_pdf=quick_pdf,
        complete_atlas_pdf=complete_pdf,
        quick_pages=quick_pages,
        complete_pages=complete_pages,
        counts=counts,
    )


__all__ = [
    "AccountingTables",
    "ReviewBundle",
    "build_event_accounting_tables",
    "generate_event_accounting_review",
    "plot_event_accounting_summary",
]
