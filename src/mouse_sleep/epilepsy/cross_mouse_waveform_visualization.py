from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mouse_sleep.epilepsy.cross_mouse_waveform_types import (
    CrossMouseWaveformResult,
)
from mouse_sleep.epilepsy.cross_mouse_clinical_visualization import (
    generate_clinical_cross_mouse_figures,
)


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
    "E07": "v",
    "E50": "o",
    "E76": "s",
    "E77": "^",
    "E79": "D",
    "E81": "P",
    "E82": "X",
}


def generate_cross_mouse_waveform_figures(
    result: CrossMouseWaveformResult,
    assignments: pd.DataFrame,
    output_dir: str | Path,
    existing_atlas_dir: str | Path | None = None,
    dpi: int = 300,
) -> list[Path]:
    """Render the 21 local categories and their conservative cross-mouse registry."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _apply_style()
    outputs: list[Path] = []
    outputs += plot_per_mouse_local_clusters(
        result.local_scatter_coordinates,
        result.local_inventory,
        root / "figure_1_per_mouse_21_category_clusters",
        dpi,
    )
    outputs += plot_cross_mouse_presence(
        result.presence,
        root / "figure_3_cross_mouse_type_presence",
        dpi,
    )
    outputs += generate_clinical_cross_mouse_figures(result, root, dpi)
    if existing_atlas_dir is not None:
        outputs += build_atlas_contact_sheet(
            existing_atlas_dir,
            root / "figure_6_filtered_waveform_atlas_contact_sheet",
            dpi,
        )
    return outputs


def plot_per_mouse_local_clusters(
    coordinates: pd.DataFrame,
    inventory: pd.DataFrame,
    output_stem: str | Path,
    dpi: int = 300,
) -> list[Path]:
    animals = sorted(coordinates["animal_id"].astype(str).unique())
    ncols = 3
    nrows = max(1, int(np.ceil(len(animals) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.4, 3.25 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for axis, animal in zip(axes.ravel(), animals):
        rows = coordinates.loc[coordinates["animal_id"].eq(animal)]
        x_low, x_high = _robust_limits(rows["local_pc1"].to_numpy(float), 0.03)
        y_low, y_high = _robust_limits(rows["local_pc2"].to_numpy(float), 0.03)
        clipped_count = int(
            (
                (rows["local_pc1"] < x_low)
                | (rows["local_pc1"] > x_high)
                | (rows["local_pc2"] < y_low)
                | (rows["local_pc2"] > y_high)
            ).sum()
        )
        types = inventory.loc[inventory["animal_id"].eq(animal)].sort_values(
            "local_type", key=lambda values: values.map(_label_number)
        )
        legend_handles = []
        for index, type_row in enumerate(types.itertuples(index=False)):
            label = str(type_row.local_type)
            display_label = str(getattr(type_row, "display_label", label.split("-")[-1]))
            group = rows.loc[rows["local_type"].eq(label)]
            color = OKABE_ITO[index % len(OKABE_ITO)]
            display_x = np.clip(group["local_pc1"], x_low, x_high)
            display_y = np.clip(group["local_pc2"], y_low, y_high)
            axis.scatter(
                display_x,
                display_y,
                s=10,
                alpha=0.22,
                color=color,
                linewidths=0,
            )
            center = group[["local_pc1", "local_pc2"]].median()
            center_x = float(np.clip(center.iloc[0], x_low, x_high))
            center_y = float(np.clip(center.iloc[1], y_low, y_high))
            axis.scatter(
                center_x,
                center_y,
                s=80,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            axis.text(
                center_x, center_y, display_label,
                ha="center", va="center", fontsize=5.8, fontweight="bold",
                color="black", zorder=4,
            )
            legend_handles.append(
                mpl.lines.Line2D(
                    [], [], marker="o", linestyle="none", color=color,
                    markersize=5, label=f"{display_label}  n={int(type_row.independent_events)} events"
                )
            )
        stable = bool(types["review_stability_ok"].iloc[0])
        status = "validated resolution" if stable else "exploratory resolution"
        axis.set_title(f"{animal} · {len(types)} categories · {status}")
        axis.set_xlim(x_low, x_high)
        axis.set_ylim(y_low, y_high)
        if clipped_count:
            axis.text(
                0.01, 0.01, f"{clipped_count} extreme points clipped for display",
                transform=axis.transAxes, fontsize=5.8, color="#555555", va="bottom"
            )
        axis.legend(
            handles=legend_handles, frameon=True, framealpha=0.82, facecolor="white",
            edgecolor="none", fontsize=5.6, ncol=2 if len(legend_handles) > 4 else 1,
            loc="best", handletextpad=0.3, columnspacing=0.6, borderpad=0.25,
        )
        axis.set_xlabel("Local PC1")
        axis.set_ylabel("Local PC2")
        _despine(axis)
    for axis in axes[len(animals):]:
        axis.axis("off")
    category_count = int(coordinates["local_type"].astype(str).nunique()) if "local_type" in coordinates else int(len(animals))
    fig.suptitle(
        f"Independent-channel within-seizure waveform categories ({category_count} labels): one point per event-category cell\n"
        "Coordinates are fitted separately per mouse/channel and cannot be compared between panels",
        fontsize=11, y=0.985,
    )
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.07, top=0.88, hspace=0.38, wspace=0.25)
    return _save(fig, output_stem, dpi)


def plot_common_category_space(
    coordinates: pd.DataFrame,
    evidence: pd.DataFrame,
    output_stem: str | Path,
    dpi: int = 300,
) -> list[Path]:
    fig, axis = plt.subplots(figsize=(9.2, 6.8), constrained_layout=True)
    color_by_type = {
        value: OKABE_ITO[index % len(OKABE_ITO)]
        for index, value in enumerate(
            sorted(coordinates["analysis_type_id"].astype(str).unique())
        )
    }
    x_low, x_high = _robust_limits(coordinates["common_pc1"].to_numpy(float), 0.02)
    y_low, y_high = _robust_limits(coordinates["common_pc2"].to_numpy(float), 0.02)
    display = coordinates.copy()
    display["display_pc1"] = np.clip(display["common_pc1"], x_low, x_high)
    display["display_pc2"] = np.clip(display["common_pc2"], y_low, y_high)
    coordinate_map = display.set_index("local_type")
    supported = evidence.loc[
        evidence["pair_decision"].isin(
            ["confirmed_shared_candidate", "provisional_shared_candidate"]
        )
    ]
    uncertain = evidence.loc[
        evidence["pair_decision"].isin(
            ["compatible_uncertain", "compatible_low_event_support"]
        )
    ].head(8)
    for row in uncertain.itertuples(index=False):
        first = coordinate_map.loc[row.local_type_a]
        second = coordinate_map.loc[row.local_type_b]
        axis.plot(
            [first.display_pc1, second.display_pc1],
            [first.display_pc2, second.display_pc2],
            color="#999999",
            linewidth=0.8,
            linestyle=":",
            alpha=0.7,
            zorder=1,
        )
    for row in supported.itertuples(index=False):
        first = coordinate_map.loc[row.local_type_a]
        second = coordinate_map.loc[row.local_type_b]
        line_color = (
            "#009E73"
            if row.pair_decision == "confirmed_shared_candidate"
            else "#E69F00"
        )
        axis.plot(
            [first.display_pc1, second.display_pc1],
            [first.display_pc2, second.display_pc2],
            color=line_color,
            linewidth=2.0,
            zorder=1,
        )
    # Keep every stream/category point in the map, but avoid an unreadable wall
    # of text when the independent-channel expansion yields hundreds of cells.
    # One highest-support stream is labelled for each frozen global category;
    # the complete point-to-label mapping remains in the accompanying CSV.
    support = (
        pd.to_numeric(display["independent_events"], errors="coerce").fillna(0.0)
        if "independent_events" in display
        else pd.Series(0.0, index=display.index)
    )
    category_column = "local_type_label" if "local_type_label" in display else "local_type"
    representatives = (
        display.assign(_support=support)
        .sort_values("_support", ascending=False, kind="stable")
        .drop_duplicates(category_column)
    )
    # Keep the complete point cloud and source table, but annotate only a
    # small, high-support set.  A label for every independent stream/category
    # cell quickly becomes unreadable in the central PCA cloud.
    max_annotated_categories = 12
    annotation_keys = set(
        representatives["local_type"].astype(str).head(max_annotated_categories)
    )
    offsets = ((5, 6), (5, -11), (-35, 6), (-35, -11), (7, 14), (-35, 14))
    for row_index, row in enumerate(display.itertuples(index=False)):
        color = color_by_type[str(row.analysis_type_id)]
        edge = "#111111" if row.local_resolution_status == "validated_stable" else "#777777"
        axis.scatter(
            row.display_pc1,
            row.display_pc2,
            marker=MOUSE_MARKERS.get(str(row.animal_id), "o"),
            s=95 if row.event_support_status == "adequate_events" else 65,
            facecolor=color,
            edgecolor=edge,
            linewidth=1.1,
            alpha=1.0 if row.event_support_status == "adequate_events" else 0.62,
            zorder=3,
        )
        if str(row.local_type) in annotation_keys:
            annotation_label = str(getattr(row, "local_type_label", row.local_type))
            axis.annotate(
                annotation_label,
                (row.display_pc1, row.display_pc2),
                xytext=offsets[row_index % len(offsets)],
                textcoords="offset points",
                fontsize=6.4,
                fontweight="bold",
                arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.4},
            )
    mouse_handles = [
        mpl.lines.Line2D(
            [],
            [],
            marker=marker,
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#222222",
            markersize=7,
            label=animal,
        )
        for animal, marker in MOUSE_MARKERS.items()
    ]
    evidence_handles = [
        mpl.lines.Line2D([], [], color="#009E73", linewidth=2, label="Confirmed link"),
        mpl.lines.Line2D([], [], color="#E69F00", linewidth=2, label="Provisional link"),
        mpl.lines.Line2D(
            [], [], color="#999999", linestyle=":", linewidth=1, label="Compatible, uncertain"
        ),
    ]
    first_legend = axis.legend(
        handles=mouse_handles,
        title="Mouse",
        frameon=False,
        ncol=3,
        loc="upper right",
    )
    axis.add_artist(first_legend)
    axis.legend(handles=evidence_handles, frameon=False, loc="lower right")
    variance_1 = 100 * float(coordinates["common_pc1_variance_fraction"].iloc[0])
    variance_2 = 100 * float(coordinates["common_pc2_variance_fraction"].iloc[0])
    axis.set_xlabel(f"Common feature PC1 ({variance_1:.1f}% variance)")
    axis.set_ylabel(f"Common feature PC2 ({variance_2:.1f}% variance)")
    # ``local_type`` is the independent stream/category cell (287 in this
    # cohort), whereas ``analysis_type_id``/``local_type_label`` is the frozen
    # global morphology category (45).  Report both explicitly in the title.
    if "analysis_type_id" in coordinates:
        category_count = int(coordinates["analysis_type_id"].astype(str).nunique())
    elif "local_type_label" in coordinates:
        category_count = int(coordinates["local_type_label"].astype(str).nunique())
    else:
        category_count = int(coordinates["local_type"].astype(str).nunique()) if "local_type" in coordinates else int(len(coordinates))
    stream_cell_count = int(coordinates["local_type"].astype(str).nunique()) if "local_type" in coordinates else int(len(coordinates))
    label_note = (
        f"Top {max_annotated_categories} categories are labelled; all {stream_cell_count} stream/category cells are retained"
        if category_count > max_annotated_categories
        else "Shared full waveform shape and physiology blocks are equally represented"
    )
    axis.set_title(
        f"Cross-mouse map of {category_count} frozen global morphology categories\n"
        f"{label_note}"
    )
    axis.set_xlim(x_low, x_high)
    axis.set_ylim(y_low, y_high)
    _despine(axis)
    return _save(fig, output_stem, dpi)


def plot_candidate_evidence_matrix(
    evidence: pd.DataFrame,
    output_stem: str | Path,
    dpi: int = 300,
) -> list[Path]:
    """Show why the strongest cross-mouse pairs did or did not merge."""

    rows = _strongest_candidate_pairs(evidence, maximum=12)
    blocks = [
        block
        for block in ("raw_shape", "amplitude", "shape", "frequency", "bilateral")
        if f"{block}_threshold" in rows
    ]
    values = np.full((len(rows), len(blocks)), np.nan, dtype=float)
    for row_index, row in enumerate(rows.itertuples(index=False)):
        for column_index, block in enumerate(blocks):
            high = getattr(row, f"{block}_ci_high")
            threshold = getattr(row, f"{block}_threshold")
            values[row_index, column_index] = (
                float(high) / float(threshold)
                if np.isfinite(high) and np.isfinite(threshold) and threshold > 0
                else np.nan
            )
    fig, axis = plt.subplots(
        figsize=(9.5, max(4.8, 0.48 * len(rows) + 1.8)), constrained_layout=True
    )
    masked = np.ma.masked_invalid(values)
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad("#D9D9D9")
    image = axis.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=2.0)
    axis.set_xticks(
        np.arange(len(blocks)),
        [
            {
                "raw_shape": "Full waveform shape",
                "amplitude": "Relative amplitude",
                "shape": "Time-domain statistics",
                "frequency": "Frequency",
                "bilateral": "Bilateral consistency",
            }[block]
            for block in blocks
        ],
    )
    pair_labels = [
        f"{row.local_type_a} <-> {row.local_type_b}"
        for row in rows.itertuples(index=False)
    ]
    axis.set_yticks(np.arange(len(rows)), pair_labels)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = "N/A" if not np.isfinite(value) else f"{value:.2f}"
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=7.2,
                color="white" if np.isfinite(value) and value > 1.35 else "black",
            )
    axis.axvline(0.5, color="white", lw=0.4)
    axis.set_xlabel("95% bootstrap upper bound / repeatability threshold (≤1 passes)")
    axis.set_ylabel("Strongest cross-mouse category pairs")
    axis.set_title(
        "Strongest cross-mouse candidates: evidence that passed and evidence blocking a merge\n"
        "Grey = unavailable (E77 bilateral evidence is intentionally omitted)"
    )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colorbar.set_label("Distance-to-threshold ratio")
    return _save(fig, output_stem, dpi)


def plot_within_mouse_raw_shape_audit(
    audit: pd.DataFrame,
    inventory: pd.DataFrame,
    output_stem: str | Path,
    dpi: int = 300,
) -> list[Path]:
    """Clinician-readable matrix of possible within-mouse duplicate labels."""

    animals = sorted(inventory["animal_id"].astype(str).unique())
    relation_codes = {
        "possible_duplicate_subdivision": 1,
        "same_shape_family_physiology_distinct": 2,
        "low_sample_boundary_transient": 3,
        "shape_distinct": 4,
        "insufficient_or_mixed_evidence": 5,
    }
    colors = ["#FFFFFF", "#009E73", "#56B4E9", "#E69F00", "#6C6C6C", "#CFCFCF"]
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, 6.5), cmap.N)
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.5))
    for axis, animal in zip(axes.ravel(), animals):
        labels = (
            inventory.loc[inventory["animal_id"].eq(animal), "local_type"]
            .sort_values(key=lambda values: values.map(_label_number))
            .astype(str)
            .tolist()
        )
        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        ratio = np.full(matrix.shape, np.nan)
        label_index = {label: index for index, label in enumerate(labels)}
        for row in audit.loc[audit["animal_id"].eq(animal)].itertuples(index=False):
            i = label_index[str(row.local_type_a)]
            j = label_index[str(row.local_type_b)]
            code = relation_codes[str(row.within_mouse_relation)]
            matrix[i, j] = matrix[j, i] = code
            value = float(row.raw_shape_distance) / float(row.raw_shape_threshold)
            ratio[i, j] = ratio[j, i] = value
        axis.imshow(matrix, cmap=cmap, norm=norm, aspect="equal")
        axis.set_xticks(np.arange(len(labels)), [x.split("-")[-1] for x in labels])
        axis.set_yticks(np.arange(len(labels)), [x.split("-")[-1] for x in labels])
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i == j:
                    axis.text(j, i, "—", ha="center", va="center", fontsize=7)
                elif np.isfinite(ratio[i, j]):
                    axis.text(
                        j, i, f"{ratio[i, j]:.1f}", ha="center", va="center",
                        fontsize=6.3, color="white" if matrix[i, j] == 4 else "black"
                    )
        status = (
            "validated"
            if bool(
                inventory.loc[inventory["animal_id"].eq(animal), "review_stability_ok"].iloc[0]
            )
            else "exploratory"
        )
        axis.set_title(f"{animal} ({status})")
        axis.set_xlabel("Local category")
        axis.set_ylabel("Local category")
    handles = [
        mpl.patches.Patch(color=colors[code], label=label)
        for label, code in [
            ("Possible duplicate subdivision", 1),
            ("Same shape, physiology differs", 2),
            ("Low-sample boundary transient", 3),
            ("Raw waveform shape distinct", 4),
            ("Mixed/insufficient evidence", 5),
        ]
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.915),
        ncol=3, frameon=False, fontsize=7
    )
    fig.suptitle(
        "Within-mouse audit of all 21 labels (cell number = raw-shape distance / threshold)\n"
        "This checks duplicate subdivision; it does not redefine the 21 labels",
        fontsize=11, y=0.99,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.82, hspace=0.48, wspace=0.30)
    return _save(fig, output_stem, dpi)


def plot_exploratory_candidate_presence(
    evidence: pd.DataFrame,
    output_stem: str | Path,
    dpi: int = 300,
) -> list[Path]:
    """Separate pair-level similarities from the formal shared-type registry."""

    decisions = [
        "confirmed_shared_candidate",
        "provisional_shared_candidate",
        "compatible_low_event_support",
        "compatible_uncertain",
        "not_merged_mixed_evidence",
    ]
    candidates = evidence.loc[evidence["pair_decision"].isin(decisions)].copy()
    if candidates.empty:
        return []
    candidates = _strongest_candidate_pairs(candidates, maximum=12)
    animals = sorted(
        set(candidates["animal_a"].astype(str)) | set(candidates["animal_b"].astype(str))
        | set(MOUSE_MARKERS)
    )
    values = np.zeros((len(candidates), len(animals)), dtype=float)
    annotations = np.full(values.shape, "", dtype=object)
    for row_index, row in enumerate(candidates.itertuples(index=False)):
        for animal, label in (
            (str(row.animal_a), str(row.local_type_a)),
            (str(row.animal_b), str(row.local_type_b)),
        ):
            column = animals.index(animal)
            values[row_index, column] = 1.0
            annotations[row_index, column] = label
    fig, axis = plt.subplots(
        figsize=(8.8, max(3.3, 0.65 * len(candidates) + 1.8)),
        constrained_layout=True,
    )
    cmap = mpl.colors.ListedColormap(["#F4F4F4", "#56B4E9"])
    axis.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(animals)), animals)
    row_labels = [
        f"CP{index + 1:02d} · 2/6 mice · {row.pair_decision}"
        for index, row in enumerate(candidates.itertuples(index=False))
    ]
    axis.set_yticks(np.arange(len(candidates)), row_labels)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if annotations[row, column]:
                axis.text(
                    column, row, annotations[row, column], ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color="#111111"
                )
    axis.set_xlabel("Mouse")
    axis.set_ylabel("Pair-level candidate (not a merged global type)")
    axis.set_title(
        "Exploratory cross-mouse similarities kept separate from the formal registry\n"
        "Rows may overlap and must not be counted as independent shared seizure types"
    )
    return _save(fig, output_stem, dpi)


def plot_cross_mouse_presence(
    presence: pd.DataFrame,
    output_stem: str | Path,
    dpi: int = 300,
) -> list[Path]:
    # ``presence`` has one row per category × independent stream.  Collapse it
    # to one row per frozen global category for the registry axis; stream-level
    # fractions remain in the heatmap values and source table.
    ordered = (
        presence.groupby(["analysis_type_id", "analysis_type_label"], as_index=False)
        .agg(mouse_count=("mouse_count", "max"))
        .assign(
            _label_number=lambda frame: frame["analysis_type_label"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .astype(float)
        )
        .sort_values(["mouse_count", "_label_number"], ascending=[False, True], kind="stable")
        .drop(columns="_label_number")
    )
    animals = sorted(presence["animal_id"].astype(str).unique())
    values = np.zeros((len(ordered), len(animals)), dtype=float)
    annotations = np.full(values.shape, "", dtype=object)
    for row_index, type_row in enumerate(ordered.itertuples(index=False)):
        group = presence.loc[presence["analysis_type_id"].eq(type_row.analysis_type_id)]
        for column_index, animal in enumerate(animals):
            row = group.loc[group["animal_id"].eq(animal)].iloc[0]
            values[row_index, column_index] = float(row["events_with_category_fraction"])
            if bool(row["present"]):
                annotations[row_index, column_index] = (
                    f"{row['local_type']}\n{int(row['independent_events'])} events"
                )
    height = max(5.4, 0.46 * len(ordered) + 1.5)
    fig, axis = plt.subplots(figsize=(8.6, height), constrained_layout=True)
    image = axis.imshow(values, aspect="auto", cmap="cividis", vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(animals)), animals)
    row_labels = [
        f"{row.analysis_type_label} ({int(row.mouse_count)}/{len(animals)} mice)"
        for row in ordered.itertuples(index=False)
    ]
    axis.set_yticks(np.arange(len(ordered)), row_labels)
    # A cell label for every category/stream is useful for a small registry,
    # but becomes unreadable once the formal taxonomy is large.  For the
    # 45-category run the heatmap remains complete and the exact counts stay
    # in ``legacy_style_presence.csv``.
    annotate_cells = len(ordered) <= 25
    if annotate_cells:
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                if annotations[row, column]:
                    axis.text(
                        column,
                        row,
                        annotations[row, column],
                        ha="center",
                        va="center",
                        fontsize=6.1,
                        color="white" if values[row, column] > 0.50 else "black",
                    )
    axis.set_xlabel("Mouse")
    axis.tick_params(axis="y", labelsize=6.5)
    axis.set_ylabel("Frozen exploratory morphology category")
    axis.set_title(
        f"Cross-mouse presence registry ({len(ordered)} categories; independent channel streams)\n"
        "Cell value is the fraction of a stream's annotated events assigned to the category"
    )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colorbar.set_label("Event fraction within mouse")
    return _save(fig, output_stem, dpi)


def plot_interpretable_category_profile(
    result: CrossMouseWaveformResult,
    output_stem: str | Path,
    dpi: int = 300,
) -> list[Path]:
    selected = result.representative_dimensions.sort_values("rank")
    features = selected["feature"].astype(str).tolist()
    prototypes = result.category_prototypes.set_index("local_type")
    scaling = result.category_prototypes.attrs.get("feature_scaling")
    if not isinstance(scaling, pd.DataFrame):
        scaling = pd.DataFrame()
    centers = scaling.set_index("feature")["center_across_local_prototypes"].to_dict()
    scales = scaling.set_index("feature")["iqr_scale_across_local_prototypes"].to_dict()
    matrix = np.column_stack(
        [
            (pd.to_numeric(prototypes[feature], errors="coerce").to_numpy(float) - centers.get(feature, 0.0))
            / max(scales.get(feature, 1.0), 1e-8)
            for feature in features
        ]
    )
    order = sorted(prototypes.index.astype(str), key=_natural_label_sort_key)
    positions = [prototypes.index.get_loc(label) for label in order]
    matrix = np.clip(matrix[positions], -3.0, 3.0)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, max(5.5, 0.30 * len(order) + 1.8)),
        gridspec_kw={"width_ratios": [1.8, 1.0]},
        constrained_layout=True,
    )
    image = axes[0].imshow(matrix, aspect="auto", cmap="PuOr", vmin=-3, vmax=3)
    axes[0].set_xticks(
        np.arange(len(features)), [_short_feature_name(feature) for feature in features], rotation=30, ha="right"
    )
    axes[0].set_yticks(np.arange(len(order)), order)
    axes[0].set_xlabel("Most discriminative non-redundant dimensions")
    axes[0].set_ylabel("Local waveform category")
    axes[0].set_title("Event-balanced category profiles")
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.035, pad=0.02)
    colorbar.set_label("Robust z-score across 21 prototypes")

    threshold = result.threshold_calibration.loc[
        result.threshold_calibration["quantile"].eq(0.95)
    ].set_index("feature_block")["distance_threshold"]
    block_order = [
        block
        for block in ("raw_shape", "amplitude", "shape", "frequency", "bilateral")
        if block in threshold.index
    ]
    values = [float(threshold.loc[block]) for block in block_order]
    bars = axes[1].barh(
        np.arange(len(block_order)),
        values,
        color=["#000000", "#D55E00", "#0072B2", "#009E73", "#CC79A7"][:len(block_order)],
        alpha=0.85,
    )
    axes[1].bar_label(bars, fmt="%.2f", padding=3, fontsize=7)
    axes[1].set_yticks(np.arange(len(block_order)), [value.title() for value in block_order])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("95th percentile split-half distance")
    axes[1].set_title("Empirical repeatability limits")
    _despine(axes[1])
    fig.suptitle(
        "Which dimensions separate the 21 categories, and how close must a cross-mouse match be?\n"
        "Thresholds are dataset-operational, not biological standards",
        fontsize=10.5,
    )
    return _save(fig, output_stem, dpi)


def plot_category_count_and_evidence(
    inventory: pd.DataFrame,
    registry: pd.DataFrame,
    output_stem: str | Path,
    dpi: int = 300,
) -> list[Path]:
    animals = sorted(inventory["animal_id"].astype(str).unique())
    counts = inventory.groupby("animal_id")["local_type"].nunique().reindex(animals)
    stable = (
        inventory.groupby("animal_id")["review_stability_ok"].first().reindex(animals)
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    colors = ["#009E73" if bool(value) else "#E69F00" for value in stable]
    bars = axes[0].bar(animals, counts, color=colors, edgecolor="#333333", linewidth=0.6)
    axes[0].bar_label(bars, labels=[str(int(value)) for value in counts], padding=3)
    axes[0].set_ylim(0, max(counts) + 1.3)
    axes[0].set_ylabel("Local waveform categories")
    # Stream identifiers are long (for example ``E50_CH7``); split the
    # identifier over two lines instead of letting adjacent tick labels merge.
    axes[0].set_xticks(
        np.arange(len(animals)),
        [animal.replace("_", "\n") for animal in animals],
    )
    axes[0].tick_params(axis="x", labelsize=6.3, pad=2)
    axes[0].set_title("Existing categories retained for every mouse")
    legend = [
        mpl.patches.Patch(color="#009E73", label="Validated/stable resolution"),
        mpl.patches.Patch(color="#E69F00", label="Exploratory subdivision"),
    ]
    axes[0].legend(handles=legend, frameon=False, fontsize=7)
    _despine(axes[0])

    # The registry passed by the adapter may contain one row per stream.  Use
    # a category-level view so the evidence bar is not inflated by channel
    # expansion.
    registry_view = (
        registry.groupby("analysis_type_id", as_index=False)
        .agg(
            analysis_type_label=("analysis_type_label", "first"),
            mouse_count=("mouse_count", "max"),
            analysis_type_status=("analysis_type_status", "first"),
        )
    )
    # Preserve explicit confirmed/provisional statuses when provided.  The
    # strict all-channel adapter otherwise records a conservative binary
    # status based on presence in at least two mice.
    registry_view["analysis_type_status"] = registry_view["analysis_type_status"].where(
        registry_view["analysis_type_status"].notna()
        & registry_view["analysis_type_status"].astype(str).ne(""),
        np.where(
            registry_view["mouse_count"].ge(2),
            "retained_exploratory",
            "animal_specific_no_supported_match",
        ),
    )
    status_counts = registry_view["analysis_type_status"].value_counts()
    status_order = [
        "confirmed_shared_candidate",
        "provisional_shared_candidate",
        "retained_exploratory",
        "animal_specific_no_supported_match",
        "animal_specific_insufficient_evidence",
    ]
    status_labels = [
        "Confirmed shared",
        "Provisional shared",
        "Present in >=2 mice (exploratory)",
        "Animal-specific, one mouse",
        "Insufficient evidence",
    ]
    values = [int(status_counts.get(status, 0)) for status in status_order]
    bars = axes[1].barh(
        np.arange(len(values)),
        values,
        color=["#009E73", "#E69F00", "#0072B2", "#999999", "#BDBDBD"],
    )
    axes[1].bar_label(bars, padding=3)
    axes[1].set_yticks(np.arange(len(values)), status_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Registered analysis types")
    axes[1].set_title("Cross-mouse matching evidence")
    _despine(axes[1])
    fig.suptitle(
        "Local discovery and cross-mouse matching are separate statistical steps",
        fontsize=10.5,
    )
    return _save(fig, output_stem, dpi)


def build_atlas_contact_sheet(
    atlas_dir: str | Path,
    output_stem: str | Path,
    dpi: int = 300,
) -> list[Path]:
    atlas_root = Path(atlas_dir)
    images = [
        atlas_root / f"{animal}_candidate_waveform_atlas.png"
        for animal in MOUSE_MARKERS
    ]
    images = [path for path in images if path.exists()]
    if not images:
        return []
    fig, axes = plt.subplots(3, 2, figsize=(15.0, 18.0), constrained_layout=True)
    for axis, path in zip(axes.ravel(), images):
        axis.imshow(plt.imread(path))
        axis.set_title(path.name.split("_")[0], fontsize=11, fontweight="bold")
        axis.axis("off")
    for axis in axes.ravel()[len(images) :]:
        axis.axis("off")
    fig.suptitle(
        "Filtered representative waveforms for all 21 local categories\n"
        "0.5-70 Hz zero-phase band-pass plus 50 Hz notch; amplitude shown in a.u.",
        fontsize=13,
    )
    return _save(fig, output_stem, dpi)


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
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


def _label_number(value: Any) -> int:
    try:
        return int(str(value).rsplit("C", 1)[1])
    except (IndexError, ValueError):
        return 999999


def _natural_label_sort_key(value: Any) -> tuple[str, int]:
    text = str(value)
    return text.split("-", 1)[0], _label_number(text)


def _short_feature_name(feature: str) -> str:
    labels = {
        "log_amplitude_vs_baseline": "Amplitude / baseline",
        "baseline_z__aggregate_line_length": "Line length / baseline",
        "baseline_z__aggregate_log_rms": "RMS / baseline",
        "aggregate_hjorth_mobility": "Hjorth mobility",
        "aggregate_hjorth_complexity": "Hjorth complexity",
        "aggregate_crest_factor": "Crest factor",
        "aggregate_excursion_rate_hz": "Excursion rate",
        "aggregate_peak_frequency_hz": "Peak frequency",
        "aggregate_median_frequency_hz": "Median frequency",
        "aggregate_spectral_entropy": "Spectral entropy",
        "aggregate_spectral_concentration": "Spectral concentration",
        "relative_delta": "Relative delta",
        "relative_theta": "Relative theta",
        "relative_alpha": "Relative alpha",
        "relative_beta": "Relative beta",
        "relative_low_gamma_clean": "Relative low gamma",
        "theta_delta_ratio": "Theta / delta",
        "high_low_ratio": "High / low",
        "channel_correlation": "Bilateral correlation",
        "spectral_cosine_similarity": "Bilateral spectral similarity",
        "abs_amplitude_log_ratio": "Bilateral amplitude difference",
        "abs_line_length_log_ratio": "Bilateral line-length difference",
    }
    return labels.get(feature, feature)


def _strongest_candidate_pairs(
    evidence: pd.DataFrame, maximum: int
) -> pd.DataFrame:
    """Retain the best pair for each local type, then fill by closeness."""

    rows = evidence.copy()
    ratio_columns = [
        f"{block}_ci_high"
        for block in ("raw_shape", "amplitude", "shape", "frequency")
        if f"{block}_ci_high" in rows
    ]
    if ratio_columns:
        ratios = []
        for row in rows.itertuples(index=False):
            values = []
            for high_column in ratio_columns:
                block = high_column.removesuffix("_ci_high")
                high = getattr(row, high_column)
                threshold = getattr(row, f"{block}_threshold")
                if np.isfinite(high) and np.isfinite(threshold) and threshold > 0:
                    values.append(float(high) / float(threshold))
            ratios.append(max(values) if values else np.inf)
        rows["candidate_ci_max_ratio"] = ratios
    else:
        rows["candidate_ci_max_ratio"] = rows["maximum_required_distance_ratio"]
    rows = rows.sort_values(
        ["candidate_ci_max_ratio", "maximum_required_distance_ratio"], kind="stable"
    )
    return rows.head(maximum).reset_index(drop=True)


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


__all__ = ["generate_cross_mouse_waveform_figures"]
