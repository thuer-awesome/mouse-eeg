from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.signal import spectrogram, welch

# Keep the established morphology layout while making vector text editable and
# reproducible across the PNG/SVG/PDF export bundle.
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)


OKABE_ITO = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
    "#7A7A7A",
    "#6A3D9A",
    "#1B9E77",
    "#A6761D",
]

REPRESENTATIVES_PER_MORPHOSTATE = 3
TIMELINE_BINS = 120
TIMELINE_EVENTS_PER_PAGE = 40


@dataclass(frozen=True)
class SignalCalibration:
    signal_unit: str = "raw"
    raw_units_per_uv: float | None = None
    reference_sensitivity_uv_per_mm: float = 10.0
    reference_paper_speed_mm_per_s: float = 30.0

    @property
    def unit_verified(self) -> bool:
        return (
            self.signal_unit.lower() == "uv"
            and self.raw_units_per_uv is not None
            and np.isfinite(self.raw_units_per_uv)
            and self.raw_units_per_uv > 0
        )

    def to_display(self, signal: np.ndarray) -> np.ndarray:
        if self.unit_verified:
            return np.asarray(signal, dtype=float) / float(self.raw_units_per_uv)
        return np.asarray(signal, dtype=float)

    @property
    def display_unit(self) -> str:
        return "uV" if self.unit_verified else "raw unit"


def generate_morphology_figures(
    assignments: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    output_dir: str | Path,
    calibration: SignalCalibration,
    dpi: int = 600,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    outputs.extend(_global_overview(assignments, summaries, output_dir, dpi))
    for animal in sorted(assignments["animal_id"].astype(str).unique()):
        outputs.extend(
            _mouse_atlas(assignments, summaries, animal, output_dir, calibration, dpi)
        )
        outputs.extend(_mouse_timelines(assignments, animal, output_dir, dpi))
    return outputs


def generate_event_morphology_figures(
    events: pd.DataFrame,
    output_dir: str | Path,
    dpi: int = 600,
) -> list[Path]:
    """Summarize one-event-one-vector assignments without counting 4-s windows."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {
        "event_id",
        "animal_id",
        "review_event_type_cluster",
        "review_event_type_label",
        "duration_s",
        "dominant_review_fraction",
        "review_state_entropy",
        "short_unresolved",
        "limited_context",
        "review_event_type_uncertain",
    }
    missing = required - set(events)
    if missing:
        raise ValueError(f"Event morphology figure input is missing: {sorted(missing)}")
    classified = events.loc[events["review_event_type_cluster"].ge(0)].copy()
    animals = sorted(events["animal_id"].astype(str).unique())
    labels = sorted(classified["review_event_type_label"].astype(str).unique())
    colors = {label: OKABE_ITO[index % len(OKABE_ITO)] for index, label in enumerate(labels)}
    fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)

    heatmap_ax = fig.add_subplot(grid[0, 0])
    table = (
        classified.groupby(["animal_id", "review_event_type_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=animals, columns=labels, fill_value=0)
    )
    fractions = table.div(table.sum(axis=1).replace(0, 1), axis=0)
    image = heatmap_ax.imshow(fractions.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    heatmap_ax.set_xticks(range(len(labels)), labels)
    heatmap_ax.set_yticks(range(len(animals)), animals)
    heatmap_ax.set_xlabel("Exploratory complete-event group")
    heatmap_ax.set_ylabel("Animal")
    heatmap_ax.set_title("A  Event-count prevalence", loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=heatmap_ax, shrink=0.75)
    colorbar.set_label("Fraction of complete events")

    duration_ax = fig.add_subplot(grid[0, 1])
    duration_values = [
        classified.loc[
            classified["review_event_type_label"].eq(label), "duration_s"
        ].clip(lower=0.1).to_numpy()
        for label in labels
    ]
    if duration_values:
        boxes = duration_ax.boxplot(
            duration_values,
            tick_labels=labels,
            patch_artist=True,
            showfliers=False,
        )
        for patch, label in zip(boxes["boxes"], labels):
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.75)
    duration_ax.set_yscale("log")
    duration_ax.set_ylabel("Complete-event duration (s, log scale)")
    duration_ax.set_title("B  Duration distribution", loc="left", fontweight="bold")
    duration_ax.spines[["top", "right"]].set_visible(False)

    composition_ax = fig.add_subplot(grid[1, 0])
    rng = np.random.default_rng(20260811)
    selection = rng.choice(len(classified), min(5000, len(classified)), replace=False)
    sampled = classified.iloc[selection]
    for label in labels:
        group = sampled.loc[sampled["review_event_type_label"].eq(label)]
        composition_ax.scatter(
            group["dominant_review_fraction"],
            group["review_state_entropy"],
            s=7,
            alpha=0.35,
            linewidths=0,
            color=colors[label],
            label=label,
        )
    composition_ax.axvline(0.8, color="#555555", lw=0.8, ls="--")
    composition_ax.set_xlabel("Dominant 4-s morphostate fraction")
    composition_ax.set_ylabel("Within-event morphostate entropy")
    composition_ax.set_title("C  Single-state versus mixed events", loc="left", fontweight="bold")
    composition_ax.legend(frameon=False, fontsize=7)
    composition_ax.spines[["top", "right"]].set_visible(False)

    quality_ax = fig.add_subplot(grid[1, 1])
    quality = events.groupby("animal_id").agg(
        short=("short_unresolved", "sum"),
        limited=("limited_context", "sum"),
        uncertain=("review_event_type_uncertain", "sum"),
        total=("event_id", "size"),
    ).reindex(animals)
    positions = np.arange(len(animals))
    short = quality["short"].to_numpy()
    limited_only = np.maximum(quality["limited"].to_numpy() - short, 0)
    uncertain_only = np.maximum(
        quality["uncertain"].to_numpy() - quality[["short", "limited"]].max(axis=1).to_numpy(),
        0,
    )
    quality_ax.bar(positions, short, color="#D55E00", label="<4 s unresolved")
    quality_ax.bar(positions, limited_only, bottom=short, color="#E69F00", label="one-window context")
    quality_ax.bar(
        positions,
        uncertain_only,
        bottom=short + limited_only,
        color="#7A7A7A",
        label="other uncertain",
    )
    quality_ax.set_xticks(positions, animals)
    quality_ax.set_ylabel("Complete events")
    quality_ax.set_title("D  Events requiring caution", loc="left", fontweight="bold")
    quality_ax.legend(frameon=False, fontsize=6)
    quality_ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Complete-event candidate groups (exploratory, not clinical seizure types)",
        fontsize=10,
        fontweight="bold",
    )
    return _save_figure(fig, output_dir / "complete_event_morphology_overview", dpi)


def _global_overview(
    assignments: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    known = assignments.loc[assignments["morphology_cluster"].ge(0)].copy()
    labels = sorted(known["morphology_label"].unique(), key=_label_number)
    animals = sorted(assignments["animal_id"].astype(str).unique())
    colors = {label: OKABE_ITO[index % len(OKABE_ITO)] for index, label in enumerate(labels)}

    fig = plt.figure(figsize=(7.2, 6.5), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])
    fig.subplots_adjust(
        left=0.09,
        right=0.94,
        bottom=0.08,
        top=0.80,
        hspace=0.52,
        wspace=0.34,
    )
    scatter = fig.add_subplot(grid[0, 0])
    rng = np.random.default_rng(20260811)
    selection = rng.choice(len(known), min(10000, len(known)), replace=False) if len(known) else []
    sampled = known.iloc[selection] if len(known) else known
    for label in labels:
        group = sampled.loc[sampled["morphology_label"].eq(label)]
        scatter.scatter(
            group["pc1"],
            group["pc2"],
            s=4,
            alpha=0.35,
            linewidths=0,
            color=colors[label],
            label=label,
        )
    scatter.set_xlabel("PCA component 1")
    scatter.set_ylabel("PCA component 2")
    scatter.set_title("A  Candidate embedding", loc="left", fontweight="bold")
    if labels:
        scatter.legend(frameon=False, ncol=min(4, len(labels)), fontsize=6)

    heatmap = fig.add_subplot(grid[0, 1])
    fractions = _event_equal_prevalence(assignments, animals, labels)
    if labels:
        image = heatmap.imshow(
            fractions.to_numpy(), cmap="viridis", aspect="auto", vmin=0, vmax=1
        )
        colorbar = fig.colorbar(image, ax=heatmap, shrink=0.75)
        colorbar.set_label("Mean event prevalence")
    else:
        heatmap.text(
            0.5,
            0.5,
            "No assigned candidate morphostates",
            transform=heatmap.transAxes,
            ha="center",
            va="center",
            fontsize=7,
        )
    heatmap.set_xticks(range(len(labels)), labels, rotation=90, ha="center", fontsize=5)
    heatmap.set_yticks(range(len(animals)), animals)
    heatmap.set_xlabel("Global candidate morphostate")
    heatmap.set_ylabel("Animal")
    heatmap.set_title(
        "B  Event-equal prevalence", loc="left", fontweight="bold"
    )

    counts_ax = fig.add_subplot(grid[1, 0])
    counts = summaries["per_mouse_global_type_counts"].set_index("animal_id")
    local_path = output_dir.parent / "tables" / "per_mouse_local_type_counts.csv"
    local = pd.read_csv(local_path).set_index("animal_id") if local_path.exists() else pd.DataFrame()
    positions = np.arange(len(animals))
    global_values = counts.reindex(animals)["shared_global_types"].fillna(0).to_numpy()
    counts_ax.bar(
        positions - 0.18,
        global_values,
        0.36,
        color="#0072B2",
        label="Shared global candidates",
    )
    if not local.empty:
        local_values = local.reindex(animals)["local_candidate_types"].fillna(0).to_numpy()
        counts_ax.bar(
            positions + 0.18,
            local_values,
            0.36,
            color="#E69F00",
            label="Within-mouse candidates",
        )
    counts_ax.set_xticks(
        positions,
        [str(animal).replace("_", "\n") for animal in animals],
        fontsize=6,
    )
    counts_ax.set_ylabel("Candidate count")
    counts_ax.set_title("C  Candidates per animal", loc="left", fontweight="bold")
    counts_ax.legend(frameon=False, fontsize=7)
    counts_ax.spines[["top", "right"]].set_visible(False)

    cluster_ax = fig.add_subplot(grid[1, 1])
    cluster = summaries["global_cluster_summary"].sort_values("morphology_cluster")
    x = np.arange(len(cluster))
    cluster_ax.bar(
        x,
        cluster["events"],
        color=[colors.get(label, "#777777") for label in cluster["morphology_label"]],
    )
    cluster_ax.set_xticks(
        x,
        cluster["morphology_label"],
        rotation=90,
        ha="center",
        fontsize=5,
    )
    cluster_ax.set_ylabel("Distinct event components")
    cluster_ax.set_title("D  Candidate event support", loc="left", fontweight="bold")
    cluster_ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Candidate seizure morphostates\n"
        "Research candidates, not clinically confirmed seizure types",
        fontsize=10,
        fontweight="bold",
        y=0.97,
    )
    return _save_figure(fig, output_dir / "global_morphology_overview", dpi)


def _event_equal_prevalence(
    assignments: pd.DataFrame,
    animals: list[str],
    labels: list[str],
) -> pd.DataFrame:
    """Average within-event state fractions so every event has equal weight."""

    if not labels:
        return pd.DataFrame(index=animals, columns=labels, dtype=float)
    frame = assignments.copy()
    frame["_animal_id"] = frame["animal_id"].astype(str)
    frame["_event_id"] = frame["event_id"].astype(str)
    event_keys = ["_animal_id", "_event_id"]
    event_totals = frame.groupby(event_keys, sort=False).size().astype(float)
    classified = frame.loc[
        frame["morphology_cluster"].ge(0)
        & frame["morphology_label"].astype(str).isin(labels)
    ].copy()
    classified["_morphology_label"] = classified["morphology_label"].astype(str)
    if classified.empty:
        return pd.DataFrame(0.0, index=animals, columns=labels)
    counts = (
        classified.groupby(event_keys + ["_morphology_label"], sort=False)
        .size()
        .rename("assigned_windows")
        .reset_index()
    )
    counts = counts.merge(
        event_totals.rename("event_windows").reset_index(),
        on=event_keys,
        how="left",
        validate="many_to_one",
    )
    counts["within_event_fraction"] = (
        counts["assigned_windows"] / counts["event_windows"].clip(lower=1.0)
    )
    wide = counts.pivot_table(
        index=event_keys,
        columns="_morphology_label",
        values="within_event_fraction",
        aggfunc="first",
        fill_value=0.0,
    ).reindex(
        index=event_totals.index,
        columns=labels,
        fill_value=0.0,
    )
    wide = wide.fillna(0.0)
    return (
        wide.groupby(level="_animal_id", sort=False)
        .mean()
        .reindex(index=animals, columns=labels, fill_value=0.0)
    )


def _mouse_atlas(
    assignments: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    animal: str,
    output_dir: Path,
    calibration: SignalCalibration,
    dpi: int,
) -> list[Path]:
    all_animal_rows = assignments.loc[
        assignments["animal_id"].astype(str).eq(animal)
    ].copy()
    animal_rows = all_animal_rows.loc[
        all_animal_rows["morphology_cluster"].ge(0)
    ].copy()
    support = summaries["mouse_cluster_summary"]
    support = support.loc[
        support["animal_id"].astype(str).eq(animal)
    ].sort_values("morphology_cluster")
    if support.empty or animal_rows.empty:
        return []
    labels = support["morphology_label"].astype(str).tolist()
    event_prevalence = _event_equal_prevalence(all_animal_rows, [animal], labels)
    unit_note = _calibration_note(calibration)
    outputs: list[Path] = []
    support_rows = support.to_dict("records")
    for page_index, row in enumerate(support_rows):
        label = str(row["morphology_label"])
        candidates = animal_rows.loc[
            animal_rows["morphology_label"].astype(str).eq(label)
        ]
        representatives = _select_event_representatives(
            candidates, REPRESENTATIVES_PER_MORPHOSTATE
        )
        if representatives.empty:
            continue
        figure_height = max(5.0, 2.25 * len(representatives) + 1.5)
        fig = plt.figure(figsize=(7.2, figure_height), constrained_layout=False)
        grid = fig.add_gridspec(
            len(representatives), 3, width_ratios=[2.7, 1.0, 1.35]
        )
        fig.subplots_adjust(
            left=0.08,
            right=0.96,
            bottom=0.07,
            top=0.84,
            hspace=0.88,
            wspace=0.48,
        )
        for row_index, (_, representative) in enumerate(
            representatives.iterrows()
        ):
            signal = _load_signal(representative)
            display = calibration.to_display(signal)
            rate = float(representative["sampling_rate_hz"])
            time = np.arange(display.shape[1]) / rate
            waveform_ax = fig.add_subplot(grid[row_index, 0])
            _plot_stacked_waveform(
                waveform_ax, time, display, representative, calibration
            )
            confidence = _confidence_note(representative).removeprefix(" | ")
            phase = str(representative.get("event_phase", "phase unknown"))
            confidence_line = f"\n{confidence}" if confidence else ""
            waveform_ax.set_title(
                f"Candidate {label}, exemplar {row_index + 1}/{len(representatives)}\n"
                f"Event {_short_event_id(representative['event_id'], 22)} | {phase}"
                f"{confidence_line}",
                loc="left",
                fontsize=6.5,
                fontweight="bold",
            )
            waveform_ax.set_xlabel("Time (s)")
            waveform_ax.set_ylabel("DC-centered waveform", fontsize=7)

            psd_ax = fig.add_subplot(grid[row_index, 1])
            _plot_psd(psd_ax, display, rate, representative, calibration)
            psd_ax.set_xlabel("Frequency (Hz)")
            if row_index == 0:
                psd_ax.set_title("PSD", fontsize=8, fontweight="bold")

            stft_ax = fig.add_subplot(grid[row_index, 2])
            _plot_stft(stft_ax, display, rate, representative, calibration)
            stft_ax.set_xlabel("Time (s)")
            if row_index == 0:
                stft_ax.set_title(
                    "STFT (mean of valid channels)", fontsize=8, fontweight="bold"
                )

        event_fraction = float(event_prevalence.loc[animal, label])
        review_status = (
            "passes automated support screen"
            if _bool_value(row.get("present_for_review", False))
            else "low-support candidate"
        )
        fig.suptitle(
            f"{animal} candidate seizure morphostate {label} "
            f"(not clinically confirmed)\n"
            f"{int(row['windows'])} windows from {int(row['events'])} events; "
            f"event-equal prevalence {100 * event_fraction:.1f}%; {review_status}\n"
            f"{unit_note}",
            fontsize=8.5,
            fontweight="bold",
            y=0.985,
        )
        root = output_dir / f"mouse_{animal}_morphology_atlas"
        if len(support_rows) > 1:
            root = output_dir / (
                f"mouse_{animal}_morphology_atlas_p{page_index + 1:02d}_{label}"
            )
        outputs.extend(_save_figure(fig, root, dpi))
    return outputs


def _mouse_timelines(
    assignments: pd.DataFrame,
    animal: str,
    output_dir: Path,
    dpi: int,
    events_per_page: int = TIMELINE_EVENTS_PER_PAGE,
    normalized_bins: int = TIMELINE_BINS,
) -> list[Path]:
    group = assignments.loc[assignments["animal_id"].astype(str).eq(animal)].copy()
    if group.empty:
        return []
    events_per_page = max(1, int(events_per_page))
    normalized_bins = max(2, int(normalized_bins))
    event_order = (
        group.groupby("event_id", sort=False)
        .agg(
            session_id=("session_id", "first"),
            start_time_ms=("start_time_ms", "min"),
            event_duration_s=("event_duration_s", "max"),
        )
        .reset_index()
        .sort_values(["session_id", "start_time_ms", "event_id"], kind="stable")
        .reset_index(drop=True)
    )
    label_table = (
        group.loc[group["morphology_cluster"].ge(0), ["morphology_cluster", "morphology_label"]]
        .drop_duplicates("morphology_cluster")
        .sort_values("morphology_cluster")
    )
    clusters = label_table["morphology_cluster"].astype(int).tolist()
    cluster_names = label_table["morphology_label"].astype(str).tolist()
    color_values = ["#BDBDBD"] + [
        OKABE_ITO[cluster % len(OKABE_ITO)] for cluster in clusters
    ]
    mapping = {-1: 0, **{cluster: index + 1 for index, cluster in enumerate(clusters)}}
    page_count = max(1, ceil(len(event_order) / events_per_page))
    outputs: list[Path] = []
    for page_index in range(page_count):
        start = page_index * events_per_page
        stop = min(len(event_order), start + events_per_page)
        page = event_order.iloc[start:stop]
        raster = np.zeros((len(page), normalized_bins), dtype=np.int16)
        uncertainty = np.zeros_like(raster, dtype=bool)
        ylabels: list[str] = []
        for row_index, event_meta in enumerate(page.to_dict("records")):
            event_id = str(event_meta["event_id"])
            event = group.loc[group["event_id"].astype(str).eq(event_id)].sort_values(
                "start_time_ms"
            )
            positions = _normalized_event_indices(event, normalized_bins)
            clusters_for_event = pd.to_numeric(
                event["morphology_cluster"], errors="coerce"
            ).fillna(-1).astype(int).to_numpy()
            raster[row_index] = np.asarray(
                [mapping.get(int(value), 0) for value in clusters_for_event[positions]],
                dtype=np.int16,
            )
            if "morphology_uncertain" in event:
                uncertain_values = np.asarray(
                    [_bool_value(value) for value in event["morphology_uncertain"]],
                    dtype=bool,
                )
                uncertainty[row_index] = uncertain_values[positions]
            ylabels.append(
                _timeline_event_label(
                    event_id,
                    str(event_meta["session_id"]),
                    float(event_meta["event_duration_s"]),
                )
            )

        figure_height = max(2.8, 0.20 * len(page) + 1.8)
        fig, ax = plt.subplots(
            figsize=(7.2, figure_height), constrained_layout=True
        )
        cmap = ListedColormap(color_values)
        ax.imshow(
            raster,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=-0.5,
            vmax=max(0.5, len(color_values) - 0.5),
        )
        if uncertainty.any():
            overlay = np.ma.masked_where(~uncertainty, np.ones_like(raster))
            ax.imshow(
                overlay,
                aspect="auto",
                interpolation="nearest",
                cmap=ListedColormap(["#202020"]),
                alpha=0.30,
                vmin=0,
                vmax=1,
            )
        ax.set_yticks(range(len(ylabels)), ylabels, fontsize=6)
        tick_positions = np.linspace(0, normalized_bins - 1, 5)
        ax.set_xticks(tick_positions, ["0", "25", "50", "75", "100"])
        ax.set_xlabel(
            "Normalized within-event progress (%) - each event has equal display width"
        )
        ax.set_ylabel("Event component (duration)")
        ax.set_title(
            f"{animal}: all-event candidate morphostate index, page "
            f"{page_index + 1}/{page_count} (events {start + 1}-{stop} of {len(event_order)})\n"
            "Candidate morphostates are not clinically confirmed seizure types",
            loc="left",
            fontweight="bold",
            fontsize=9,
        )
        handles: list[Patch] = [
            Patch(facecolor=color_values[0], label="unknown / unassigned")
        ]
        handles.extend(
            Patch(facecolor=color_values[index + 1], label=f"candidate {name}")
            for index, name in enumerate(cluster_names)
        )
        if uncertainty.any():
            handles.append(
                Patch(facecolor="#202020", alpha=0.30, label="uncertain assignment")
            )
        ax.legend(
            handles=handles,
            frameon=False,
            ncol=1,
            fontsize=6,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )
        root = output_dir / f"mouse_{animal}_event_timelines"
        if page_count > 1:
            root = output_dir / f"mouse_{animal}_event_timelines_p{page_index + 1:03d}"
        outputs.extend(_save_figure(fig, root, dpi))
    return outputs


def _select_event_representatives(
    candidates: pd.DataFrame,
    count: int,
) -> pd.DataFrame:
    """Select the best window from distinct events before any within-event repeat."""

    if candidates.empty:
        return candidates.copy()
    ranked = candidates.copy()
    ranked["_event_key"] = ranked["event_id"].astype(str)
    if "morphology_uncertain" in ranked:
        ranked["_rank_uncertain"] = [
            int(_bool_value(value)) for value in ranked["morphology_uncertain"]
        ]
    else:
        ranked["_rank_uncertain"] = 0
    if "morphology_probability" in ranked:
        probability = pd.to_numeric(
            ranked["morphology_probability"], errors="coerce"
        )
        ranked["_rank_probability"] = -probability.fillna(-np.inf)
    else:
        ranked["_rank_probability"] = 0.0
    distance = (
        ranked["morphology_distance"]
        if "morphology_distance" in ranked
        else pd.Series(np.inf, index=ranked.index)
    )
    start_time = (
        ranked["start_time_ms"]
        if "start_time_ms" in ranked
        else pd.Series(0.0, index=ranked.index)
    )
    ranked["_rank_distance"] = pd.to_numeric(
        distance, errors="coerce"
    ).fillna(np.inf)
    ranked["_rank_start"] = pd.to_numeric(
        start_time, errors="coerce"
    ).fillna(0.0)
    ranked = ranked.sort_values(
        [
            "_rank_uncertain",
            "_rank_probability",
            "_rank_distance",
            "_rank_start",
        ],
        kind="stable",
    )
    representatives = ranked.drop_duplicates("_event_key", keep="first").head(count)
    return representatives.drop(
        columns=[
            "_event_key",
            "_rank_uncertain",
            "_rank_probability",
            "_rank_distance",
            "_rank_start",
        ]
    )


def _plot_psd(
    ax: plt.Axes,
    signal: np.ndarray,
    rate: float,
    row: pd.Series,
    calibration: SignalCalibration,
) -> None:
    plotted = False
    for channel_index, (channel, color) in enumerate(
        (("CH7", "#0072B2"), ("CH8", "#D55E00"))
    ):
        if channel_index >= signal.shape[0] or not _row_channel_valid(row, channel):
            continue
        values = signal[channel_index] - np.median(signal[channel_index])
        frequency, power = welch(
            values,
            fs=rate,
            nperseg=min(512, signal.shape[1]),
        )
        limit = min(100.0, rate / 2.0)
        mask = (frequency >= 0.5) & (frequency <= limit)
        ax.semilogy(
            frequency[mask], power[mask] + 1e-12, color=color, lw=0.75, label=channel
        )
        plotted = True
    ax.set_xlim(0, min(100.0, rate / 2.0))
    ax.set_ylabel(f"PSD ({calibration.display_unit}^2/Hz)", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    if plotted:
        ax.legend(frameon=False, fontsize=6)
    else:
        ax.text(
            0.5,
            0.5,
            "No valid channel",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6,
        )


def _plot_stft(
    ax: plt.Axes,
    signal: np.ndarray,
    rate: float,
    row: pd.Series,
    calibration: SignalCalibration,
) -> None:
    channel_power: list[np.ndarray] = []
    frequencies: np.ndarray | None = None
    times: np.ndarray | None = None
    nperseg = min(256, signal.shape[1])
    if nperseg < 8:
        ax.text(0.5, 0.5, "Window too short for STFT", ha="center", va="center")
        return
    for channel_index, channel in enumerate(("CH7", "CH8")):
        if channel_index >= signal.shape[0] or not _row_channel_valid(row, channel):
            continue
        frequencies, times, power = spectrogram(
            signal[channel_index] - np.median(signal[channel_index]),
            fs=rate,
            window="hann",
            nperseg=nperseg,
            noverlap=min(nperseg - 1, int(round(0.75 * nperseg))),
            detrend="constant",
            scaling="density",
            mode="psd",
        )
        channel_power.append(power)
    if not channel_power or frequencies is None or times is None:
        ax.text(
            0.5,
            0.5,
            "No valid channel",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6,
        )
        return
    mean_power = np.mean(np.stack(channel_power), axis=0)
    limit = min(100.0, rate / 2.0)
    mask = (frequencies >= 0.5) & (frequencies <= limit)
    log_power = 10.0 * np.log10(np.maximum(mean_power[mask], 1e-20))
    image = ax.pcolormesh(
        times,
        frequencies[mask],
        log_power,
        shading="auto",
        cmap="cividis",
        rasterized=True,
    )
    ax.set_ylim(0.5, limit)
    ax.set_ylabel("Frequency (Hz)", fontsize=7)
    colorbar = ax.figure.colorbar(image, ax=ax, pad=0.02, fraction=0.06)
    colorbar.set_label(
        f"dB re 1 {calibration.display_unit}^2/Hz", fontsize=6
    )
    colorbar.ax.tick_params(labelsize=5)


def _normalized_event_indices(event: pd.DataFrame, bins: int) -> np.ndarray:
    if event.empty:
        raise ValueError("Cannot normalize an empty event")
    midpoint = 0.5 * (
        pd.to_numeric(event["start_time_ms"], errors="coerce").to_numpy(float)
        + pd.to_numeric(event["end_time_ms"], errors="coerce").to_numpy(float)
    )
    event_start = (
        float(pd.to_numeric(event["event_start_ms"], errors="coerce").iloc[0])
        if "event_start_ms" in event
        else float(np.nanmin(midpoint))
    )
    event_end = (
        float(pd.to_numeric(event["event_end_ms"], errors="coerce").iloc[0])
        if "event_end_ms" in event
        else float(np.nanmax(midpoint))
    )
    duration = max(event_end - event_start, 1.0)
    progress = np.clip((midpoint - event_start) / duration, 0.0, 1.0)
    progress = np.maximum.accumulate(np.nan_to_num(progress, nan=0.0))
    targets = np.linspace(0.0, 1.0, bins)
    right = np.searchsorted(progress, targets, side="left")
    right = np.clip(right, 0, len(progress) - 1)
    left = np.clip(right - 1, 0, len(progress) - 1)
    choose_right = np.abs(progress[right] - targets) < np.abs(progress[left] - targets)
    return np.where(choose_right, right, left).astype(np.int64)


def _calibration_note(calibration: SignalCalibration) -> str:
    if calibration.unit_verified:
        return (
            f"Amplitude converted to uV (1 uV = {calibration.raw_units_per_uv:g} raw units); "
            f"ACNS display reference {calibration.reference_sensitivity_uv_per_mm:g} uV/mm, "
            f"{calibration.reference_paper_speed_mm_per_s:g} mm/s"
        )
    return (
        "Amplitude is DC-centered raw acquisition units (a.u.); calibration is "
        "unverified, so a uV/mm display scale is not applied"
    )


def _confidence_note(row: pd.Series) -> str:
    parts: list[str] = []
    if "morphology_probability" in row.index:
        probability = pd.to_numeric(
            pd.Series([row.get("morphology_probability")]), errors="coerce"
        ).iloc[0]
        if np.isfinite(probability):
            parts.append(f"p={float(probability):.2f}")
    if "morphology_distance" in row.index:
        distance = pd.to_numeric(
            pd.Series([row.get("morphology_distance")]), errors="coerce"
        ).iloc[0]
        if np.isfinite(distance):
            parts.append(f"distance={float(distance):.2f}")
    if "morphology_uncertain" in row.index and _bool_value(
        row.get("morphology_uncertain")
    ):
        parts.append("uncertain")
    return "" if not parts else " | " + ", ".join(parts)


def _row_channel_valid(row: pd.Series, channel: str) -> bool:
    return _bool_value(row.get(f"{channel.lower()}_valid", True), default=True)


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    try:
        if bool(pd.isna(value)):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


def _short_event_id(event_id: Any, maximum_length: int = 32) -> str:
    text = str(event_id)
    if len(text) <= maximum_length:
        return text
    half = (maximum_length - 3) // 2
    return f"{text[:half]}...{text[-half:]}"


def _timeline_event_label(event_id: str, session_id: str, duration_s: float) -> str:
    event_name = event_id.split(":")[-1]
    session_name = session_id if len(session_id) <= 12 else session_id[:9] + "..."
    return f"{session_name}/{event_name} ({duration_s:.1f}s)"


def _plot_stacked_waveform(
    ax: plt.Axes,
    time: np.ndarray,
    signal: np.ndarray,
    row: pd.Series,
    calibration: SignalCalibration,
) -> None:
    centered = signal - np.median(signal, axis=1, keepdims=True)
    valid_flags = [
        channel_index < signal.shape[0] and _row_channel_valid(row, channel)
        for channel_index, channel in enumerate(("CH7", "CH8"))
    ]
    display_channels = centered[: len(valid_flags)]
    valid_signal = display_channels[
        np.asarray(valid_flags[: len(display_channels)], dtype=bool)
    ]
    amplitude_source = valid_signal if valid_signal.size else centered
    span = float(np.nanquantile(np.abs(amplitude_source), 0.995))
    span = max(span, 1.0)
    offset = 2.5 * span
    for channel_index, (channel, color) in enumerate(
        (("CH7", "#0072B2"), ("CH8", "#D55E00"))
    ):
        valid = channel_index < signal.shape[0] and _row_channel_valid(row, channel)
        base = (1 - channel_index) * offset
        if valid:
            ax.plot(time, centered[channel_index] + base, color=color, lw=0.65)
        else:
            ax.plot(time, np.full_like(time, base), color="#999999", lw=0.65)
            ax.text(time[-1], base, " invalid", color="#777777", fontsize=6, va="center")
        ax.text(time[0], base + 0.72 * span, channel, color=color if valid else "#777777", fontsize=6)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(-1.3 * span, offset + 1.3 * span)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    horizontal = min(1.0, float(time[-1] - time[0]) / 3.0)
    vertical = _nice_scale(span, calibration.unit_verified)
    x0 = time[-1] - horizontal - 0.08 * (time[-1] - time[0])
    y0 = -0.9 * span
    ax.plot([x0, x0 + horizontal], [y0, y0], color="black", lw=1.2, clip_on=False)
    ax.plot([x0, x0], [y0, y0 + vertical], color="black", lw=1.2, clip_on=False)
    ax.text(x0 + horizontal / 2, y0 - 0.12 * span, f"{horizontal:g} s", ha="center", va="top", fontsize=6)
    ax.text(x0 - 0.01 * (time[-1] - time[0]), y0 + vertical / 2, f"{vertical:g} {calibration.display_unit}", ha="right", va="center", fontsize=6)


def _nice_scale(span: float, verified_uv: bool) -> float:
    if verified_uv:
        candidates = np.asarray([20, 50, 100, 200, 500, 1000, 2000, 5000], dtype=float)
    else:
        exponent = 10 ** math_floor_log10(span)
        candidates = exponent * np.asarray([0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    target = max(span * 0.6, candidates[0])
    return float(candidates[np.argmin(np.abs(candidates - target))])


def math_floor_log10(value: float) -> int:
    return int(np.floor(np.log10(max(abs(value), 1e-12))))


def _load_signal(row: pd.Series) -> np.ndarray:
    path = Path(str(row["cache_dir"])) / "signals.npy"
    signals = np.load(path, mmap_mode="r")
    return np.array(
        signals[int(row["cache_local_index"])], dtype=np.float32, copy=True
    )


def _label_number(label: Any) -> int:
    text = str(label)
    try:
        digits = "".join(character for character in text if character.isdigit())
        return int(digits)
    except ValueError:
        return 10_000


def _save_figure(fig: plt.Figure, root: Path, dpi: int) -> list[Path]:
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


__all__ = [
    "SignalCalibration",
    "generate_event_morphology_figures",
    "generate_morphology_figures",
]
