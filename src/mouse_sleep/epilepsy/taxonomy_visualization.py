from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from mouse_sleep.epilepsy.taxonomy_statistics import TaxonomyStatisticsResult


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

BLOCK_COLORS = {
    "shape": "#0072B2",
    "amplitude": "#D55E00",
    "frequency": "#009E73",
    "bilateral": "#CC79A7",
}

FEATURE_LABELS = {
    "amplitude_vs_reference_log": "Amplitude vs reference (log ratio)",
    "line_length_vs_reference_log": "Line length vs reference (log ratio)",
    "aggregate_peak_frequency_hz": "Peak frequency (Hz)",
    "aggregate_median_frequency_hz": "Median frequency (Hz)",
    "aggregate_spectral_entropy": "Spectral entropy",
    "aggregate_crest_factor": "Crest factor",
    "aggregate_hjorth_mobility": "Hjorth mobility",
    "aggregate_hjorth_complexity": "Hjorth complexity",
    "relative_delta": "Delta relative power",
    "relative_theta": "Theta relative power",
    "relative_beta": "Beta relative power",
    "relative_low_gamma_clean": "Low-gamma relative power",
    "theta_delta_ratio": "Theta/delta ratio (log)",
    "high_low_ratio": "High/low ratio (log)",
    "channel_correlation": "CH7-CH8 correlation",
    "spectral_cosine_similarity": "CH7-CH8 spectral similarity",
    "amplitude_log_ratio": "CH7-CH8 amplitude difference",
    "line_length_log_ratio": "CH7-CH8 line-length difference",
}


def apply_taxonomy_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 0.9,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def _save_figure(fig: plt.Figure, output_stem: str | Path) -> list[Path]:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    fig.savefig(outputs[0], dpi=300)
    fig.savefig(outputs[1])
    plt.close(fig)
    return outputs


def plot_mouse_waveform_atlas(
    signals: np.ndarray,
    assignments: pd.DataFrame,
    state_summary: pd.DataFrame,
    output_stem: str | Path,
) -> list[Path]:
    """Plot unnormalized filtered waveforms with one fixed scale per mouse."""

    apply_taxonomy_figure_style()
    if assignments.empty or state_summary.empty:
        raise ValueError("Waveform atlas needs state assignments and summaries")
    animal = str(assignments["animal_id"].iloc[0])
    representatives = state_summary.sort_values("state", kind="stable")
    active_channels = [
        index
        for index, column in enumerate(("ch7_valid", "ch8_valid"))
        if column in assignments and bool(assignments[column].mean() >= 0.5)
    ]
    if not active_channels:
        raise ValueError(f"No valid channel remains for {animal}")
    states = len(representatives)
    fig, axes = plt.subplots(
        len(active_channels),
        states,
        figsize=(max(6.8, states * 2.25), max(2.2, len(active_channels) * 1.7)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    sample_indices = np.linspace(
        0, len(signals) - 1, min(5000, len(signals)), dtype=np.int64
    )
    scale_values = np.asarray(signals[sample_indices][:, active_channels], dtype=float)
    scale = float(np.quantile(np.abs(scale_values), 0.995))
    representative_indices = representatives["representative_row_index"].to_numpy(
        np.int64
    )
    representative_scale = float(
        np.max(np.abs(np.asarray(signals[representative_indices][:, active_channels])))
    )
    scale = max(scale, 1.05 * representative_scale)
    scale = max(scale, 1e-6)
    for column_index, row in enumerate(representatives.itertuples(index=False)):
        signal_index = int(row.representative_row_index)
        rate = float(assignments.loc[signal_index, "sampling_rate_hz"])
        time = np.arange(signals.shape[-1]) / rate
        for axis_row, channel_index in enumerate(active_channels):
            axis = axes[axis_row, column_index]
            axis.plot(
                time,
                signals[signal_index, channel_index],
                color=OKABE_ITO[column_index % len(OKABE_ITO)],
                linewidth=0.8,
            )
            axis.axhline(0.0, color="#B8B8B8", linewidth=0.45)
            axis.set_ylim(-scale, scale)
            if column_index == 0:
                axis.set_ylabel(f"CH{channel_index + 7}\nAmplitude (a.u.)")
            if axis_row == 0:
                axis.set_title(
                    f"S{int(row.state) + 1}\n{int(row.windows):,} windows, "
                    f"{int(row.events)} events"
                )
            if axis_row == len(active_channels) - 1:
                axis.set_xlabel("Time (s)")
    fig.suptitle(
        f"Mouse {animal}: representative filtered waveforms\n"
        "Fixed vertical scale within mouse; ADC-to-microvolt calibration unavailable",
        y=1.04,
        fontsize=10,
    )
    fig.tight_layout()
    return _save_figure(fig, output_stem)


def _representative_events(assignments: pd.DataFrame, maximum: int) -> list[str]:
    durations = (
        assignments.groupby("event_id", sort=True)["event_duration_s"]
        .first()
        .sort_values(kind="stable")
    )
    if len(durations) <= maximum:
        return durations.index.astype(str).tolist()
    positions = np.linspace(0, len(durations) - 1, maximum, dtype=np.int64)
    return durations.iloc[positions].index.astype(str).tolist()


def plot_mouse_state_timelines(
    assignments: pd.DataFrame,
    selected_k: int,
    output_stem: str | Path,
    maximum_events: int = 12,
) -> list[Path]:
    apply_taxonomy_figure_style()
    if assignments.empty:
        raise ValueError("State timeline needs assignments")
    animal = str(assignments["animal_id"].iloc[0])
    event_ids = _representative_events(assignments, maximum_events)
    fig, axes = plt.subplots(
        len(event_ids),
        1,
        figsize=(7.2, max(2.8, 0.42 * len(event_ids) + 1.2)),
        sharex=True,
        squeeze=False,
    )
    cmap = ListedColormap(OKABE_ITO[:selected_k])
    for row_index, event_id in enumerate(event_ids):
        axis = axes[row_index, 0]
        event = assignments.loc[assignments["event_id"].astype(str).eq(event_id)].sort_values(
            "start_time_ms", kind="stable"
        )
        path = event["state"].to_numpy(int)[None, :]
        axis.imshow(
            path,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=-0.5,
            vmax=selected_k - 0.5,
            extent=(0.0, 1.0, 0.0, 1.0),
        )
        duration = float(event["event_duration_s"].iloc[0])
        short_id = event_id.split(":")[-1]
        axis.set_yticks([0.5], [f"{short_id}\n{duration:.1f} s"])
        axis.tick_params(axis="y", length=0)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_visible(row_index == len(event_ids) - 1)
        if row_index < len(event_ids) - 1:
            axis.tick_params(axis="x", bottom=False, labelbottom=False)
    axes[-1, 0].set_xlabel("Relative position within annotated seizure")
    handles = [
        mpl.patches.Patch(color=OKABE_ITO[state], label=f"S{state + 1}")
        for state in range(selected_k)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=selected_k,
        frameon=False,
    )
    fig.suptitle(
        f"Mouse {animal}: state paths across duration-stratified seizures",
        y=1.025,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return _save_figure(fig, output_stem)


def plot_local_event_type_inventory(
    event_assignments: pd.DataFrame,
    output_stem: str | Path,
) -> list[Path]:
    """Show validated within-mouse event types without asserting cross-mouse identity."""

    apply_taxonomy_figure_style()
    required = {"animal_id", "event_id", "event_type", "event_type_label"}
    missing = required - set(event_assignments.columns)
    if missing:
        raise ValueError(f"Event inventory is missing {sorted(missing)}")
    counts = (
        event_assignments.groupby(
            ["animal_id", "event_type", "event_type_label"], as_index=False
        )["event_id"]
        .nunique()
        .rename(columns={"event_id": "event_count"})
    )
    totals = counts.groupby("animal_id")["event_count"].transform("sum")
    counts["fraction"] = counts["event_count"] / totals
    mice = sorted(counts["animal_id"].astype(str).unique())
    fig, axis = plt.subplots(figsize=(6.8, 3.2))
    left = np.zeros(len(mice), dtype=float)
    for mouse_index, mouse in enumerate(mice):
        rows = counts.loc[counts["animal_id"].astype(str).eq(mouse)].sort_values(
            "event_type", kind="stable"
        )
        for local_index, row in enumerate(rows.itertuples(index=False)):
            fraction = float(row.fraction)
            axis.barh(
                mouse_index,
                fraction,
                left=left[mouse_index],
                color=OKABE_ITO[local_index % len(OKABE_ITO)],
                edgecolor="white",
                linewidth=0.8,
                height=0.66,
            )
            label = f"T{int(row.event_type) + 1}: {fraction:.1%}\n(n={int(row.event_count)})"
            text_color = "white" if local_index in {0, 1, 2, 3, 4} else "black"
            axis.text(
                left[mouse_index] + fraction / 2,
                mouse_index,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )
            left[mouse_index] += fraction
    axis.set_yticks(np.arange(len(mice)), mice)
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Fraction of annotated seizures within mouse")
    axis.set_ylabel("Mouse")
    axis.set_title("Validated local whole-seizure types (no cross-mouse identity assumed)")
    axis.invert_yaxis()
    axis.spines["left"].set_visible(False)
    axis.tick_params(axis="y", length=0)
    fig.tight_layout()
    return _save_figure(fig, output_stem)


def plot_mouse_dimension_summary(
    statistics: TaxonomyStatisticsResult,
    output_stem: str | Path,
) -> list[Path]:
    apply_taxonomy_figure_style()
    selected = statistics.selected_dimensions.sort_values("rank", kind="stable")
    descriptives = statistics.state_descriptives
    animal = (
        str(descriptives["animal_id"].iloc[0]) if len(descriptives) else "unknown"
    )
    if selected.empty:
        fig, axis = plt.subplots(figsize=(5.0, 2.2))
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            f"Mouse {animal}: one supported state; no between-state dimension contrast",
            ha="center",
            va="center",
        )
        return _save_figure(fig, output_stem)

    features = selected["feature"].astype(str).tolist()
    state_values = descriptives.loc[descriptives["feature"].isin(features)].pivot(
        index="feature", columns="state", values="event_median"
    )
    state_values = state_values.reindex(features)
    raw = state_values.to_numpy(float)
    center = np.nanmedian(raw, axis=1, keepdims=True)
    scale = np.nanmedian(np.abs(raw - center), axis=1, keepdims=True) * 1.4826
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (raw - center) / scale

    figure_height = max(3.0, 0.55 * len(features) + 1.4)
    fig, (heat_axis, forest_axis) = plt.subplots(
        1,
        2,
        figsize=(8.0, figure_height),
        gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.55},
        constrained_layout=True,
    )
    image = heat_axis.imshow(standardized, aspect="auto", cmap="PuOr", vmin=-3, vmax=3)
    heat_axis.set_xticks(
        np.arange(state_values.shape[1]),
        [f"S{int(state) + 1}" if str(state).isdigit() else str(state) for state in state_values.columns],
    )
    heat_axis.set_yticks(
        np.arange(len(features)), [FEATURE_LABELS.get(feature, feature) for feature in features]
    )
    heat_axis.set_title("Event-weighted state medians")
    heat_axis.set_xlabel("Local state")
    heat_axis.spines["top"].set_visible(True)
    heat_axis.spines["right"].set_visible(True)
    colorbar = fig.colorbar(image, ax=heat_axis, fraction=0.045, pad=0.04)
    colorbar.set_label("Robust z-score")

    y = np.arange(len(selected))
    effects = selected["effect_size"].to_numpy(float)
    lower = selected["effect_ci_low"].to_numpy(float)
    upper = selected["effect_ci_high"].to_numpy(float)
    colors = [BLOCK_COLORS[str(block)] for block in selected["feature_block"]]
    forest_axis.axvline(0.0, color="#666666", linewidth=0.7)
    for index in range(len(selected)):
        forest_axis.errorbar(
            effects[index],
            y[index],
            xerr=np.asarray(
                [[effects[index] - lower[index]], [upper[index] - effects[index]]]
            ),
            fmt="o",
            color=colors[index],
            ecolor=colors[index],
            capsize=2.5,
            markersize=4,
        )
    forest_axis.set_yticks(y, [FEATURE_LABELS.get(feature, feature) for feature in features])
    forest_axis.invert_yaxis()
    forest_axis.set_xlabel("Robust standardized effect (95% event-bootstrap CI)")
    forest_axis.set_title("Most discriminative non-redundant dimensions")
    block_handles = [
        mpl.lines.Line2D(
            [], [], marker="o", linestyle="none", color=color, label=block.title()
        )
        for block, color in BLOCK_COLORS.items()
    ]
    forest_axis.legend(handles=block_handles, frameon=False, loc="best")
    fig.suptitle(f"Mouse {animal}: interpretable state dimensions", fontsize=10)
    return _save_figure(fig, output_stem)


def plot_mouse_model_selection(
    state_grid: pd.DataFrame,
    event_grid: pd.DataFrame,
    output_stem: str | Path,
) -> list[Path]:
    apply_taxonomy_figure_style()
    animal = str(state_grid["animal_id"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for axis, grid, count_column, title in (
        (axes[0], state_grid, "k", "Within-seizure states"),
        (axes[1], event_grid, "event_type_count", "Whole-seizure types"),
    ):
        x = grid[count_column].to_numpy(int)
        y = grid["event_cv_log_likelihood"].to_numpy(float)
        error = grid["event_cv_standard_error"].to_numpy(float)
        colors = ["#009E73" if bool(value) else "#9E9E9E" for value in grid["support_ok"]]
        axis.errorbar(x, y, yerr=error, color="#666666", linewidth=0.7, capsize=2)
        axis.scatter(x, y, c=colors, s=28, zorder=3)
        chosen = grid.loc[grid["selected"]]
        axis.scatter(
            chosen[count_column],
            chosen["event_cv_log_likelihood"],
            facecolors="none",
            edgecolors="#D55E00",
            linewidths=1.5,
            s=80,
            zorder=4,
        )
        axis.set_xticks(x)
        axis.set_xlabel("Candidate count")
        axis.set_ylabel("Held-out event log-likelihood")
        axis.set_title(title)
    fig.suptitle(
        f"Mouse {animal}: event-grouped model selection\n"
        "Green passes support/stability; orange ring is one-SE selection",
        fontsize=10,
    )
    fig.tight_layout()
    return _save_figure(fig, output_stem)


def plot_cross_mouse_presence(
    presence: pd.DataFrame,
    kind: str,
    output_stem: str | Path,
    status_label: str | None = None,
) -> list[Path]:
    apply_taxonomy_figure_style()
    if kind not in {"event", "state"}:
        raise ValueError("kind must be 'event' or 'state'")
    if kind == "event":
        label_column = "global_event_type_label"
        value_column = "event_fraction"
        count_column = "event_count"
        title = "Cross-mouse seizure-type presence"
    else:
        label_column = "global_state_label"
        value_column = "state_fraction"
        count_column = "window_count"
        title = "Cross-mouse state-type presence"
    grouped = (
        presence.groupby([label_column, "animal_id"], as_index=False)[
            [value_column, count_column]
        ]
        .sum()
    )
    fractions = grouped.pivot(index=label_column, columns="animal_id", values=value_column)
    counts = grouped.pivot(index=label_column, columns="animal_id", values=count_column)
    fractions = fractions.fillna(0.0)
    counts = counts.reindex(index=fractions.index, columns=fractions.columns).fillna(0)
    coverage = (counts > 0).sum(axis=1)
    row_labels = [
        f"{label} ({int(coverage.loc[label])}/{len(fractions.columns)} mice)"
        for label in fractions.index
    ]
    fig, axis = plt.subplots(
        figsize=(max(5.5, 0.75 * len(fractions.columns) + 2.5), max(2.5, 0.5 * len(fractions) + 1.4))
    )
    image = axis.imshow(fractions.to_numpy(float), aspect="auto", cmap="cividis", vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(fractions.columns)), fractions.columns)
    axis.set_yticks(np.arange(len(fractions)), row_labels)
    for row in range(len(fractions)):
        for column in range(len(fractions.columns)):
            fraction = float(fractions.iloc[row, column])
            count = int(counts.iloc[row, column])
            color = "white" if fraction > 0.55 else "black"
            axis.text(
                column,
                row,
                f"{fraction:.2f}\n(n={count})",
                ha="center",
                va="center",
                color=color,
                fontsize=6.5,
            )
    axis.set_xlabel("Mouse")
    axis.set_ylabel("Global matched type")
    if status_label:
        title = f"{title}\n{status_label}"
    axis.set_title(title)
    axis.spines["top"].set_visible(True)
    axis.spines["right"].set_visible(True)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Within-mouse fraction")
    fig.tight_layout()
    return _save_figure(fig, output_stem)


__all__ = [
    "FEATURE_LABELS",
    "OKABE_ITO",
    "apply_taxonomy_figure_style",
    "plot_cross_mouse_presence",
    "plot_local_event_type_inventory",
    "plot_mouse_dimension_summary",
    "plot_mouse_model_selection",
    "plot_mouse_state_timelines",
    "plot_mouse_waveform_atlas",
]
