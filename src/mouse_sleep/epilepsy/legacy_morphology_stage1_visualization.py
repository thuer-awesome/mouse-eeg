"""Compact publication-style figures for the legacy morphology stage-1 audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl

# This is a batch-only scientific export pipeline.  Select the backend before
# pyplot import so figure generation is reproducible on headless sessions.
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from mouse_sleep.epilepsy.legacy_morphology_stage1 import ANIMALS, OLD_LABEL


COLORS = {
    "E50": "#0072B2",
    "E76": "#D55E00",
    "E77": "#009E73",
    "E79": "#CC79A7",
    "E81": "#E69F00",
    "E82": "#56B4E9",
}
MARKERS = {"E50": "o", "E76": "s", "E77": "^", "E79": "D", "E81": "P", "E82": "X"}
WIDTH_MM = 183


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 5.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(ext) for ext in (".svg", ".pdf", ".tiff", ".png")]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], dpi=600, bbox_inches="tight")
    fig.savefig(paths[3], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def _label_number(value: str) -> int:
    tail = str(value).rsplit("-C", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return 999


def _feature_label(value: str) -> str:
    """Use compact, human-readable feature labels at journal figure scale."""

    labels = {
        "norm__ch8_hjorth_mobility": "Hjorth mobility",
        "norm__ch8_spectral_entropy": "spectral entropy",
        "norm__ch8_relative_delta": "delta power",
        "norm__ch8_zero_crossing_rate": "zero-crossing rate",
        "norm__ch8_relative_low_gamma_clean": "low gamma power",
        "norm__ch8_peak_frequency_hz": "peak frequency",
        "norm__ch8_median_frequency_hz": "median frequency",
        "norm__ch8_relative_theta": "theta power",
        "norm__ch8_relative_alpha": "alpha power",
        "norm__ch8_relative_beta": "beta power",
        "norm__ch8_relative_high_gamma": "high gamma power",
        "norm__ch8_theta_delta_ratio": "theta/delta",
    }
    return labels.get(value, value.replace("norm__ch8_", "").replace("relative_", ""))


def _state_resolution(grid: pd.DataFrame) -> pd.DataFrame:
    selected = grid.loc[grid["selected"].astype(bool)].copy()
    return selected.drop_duplicates("animal_id").set_index("animal_id").reindex(ANIMALS)


def figure_1(
    pca: pd.DataFrame,
    eligibility: pd.DataFrame,
    candidates: pd.DataFrame,
    audit: dict[str, Any],
    loadings: pd.DataFrame,
    stem: Path,
) -> list[Path]:
    """Main figure: event-balanced PCA hero plus minimum supporting evidence."""

    _style()
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 5.15))
    grid = fig.add_gridspec(2, 3, width_ratios=(1.72, 1.0, 1.0), hspace=0.55, wspace=0.55)
    hero = fig.add_subplot(grid[:, 0])
    for mouse in ANIMALS:
        rows = pca.loc[pca["animal_id"].eq(mouse)]
        if rows.empty:
            continue
        hero.scatter(rows["weighted_pc1"], rows["weighted_pc2"], s=8, alpha=0.28, marker=MARKERS[mouse], color=COLORS[mouse], linewidths=0, rasterized=True)
        hero.scatter(rows["weighted_pc1"].median(), rows["weighted_pc2"].median(), s=45, marker=MARKERS[mouse], color=COLORS[mouse], edgecolor="white", linewidth=0.7, zorder=3)
    legend = [mpl.lines.Line2D([], [], linestyle="none", marker=MARKERS[m], markerfacecolor=COLORS[m], markeredgecolor="white", markersize=5.5, label=m) for m in ANIMALS]
    hero.legend(handles=legend, title="mouse", frameon=False, ncol=3, loc="best", columnspacing=0.8, handletextpad=0.25)
    hero.set_xlabel("Mouse-balanced PC1")
    hero.set_ylabel("Mouse-balanced PC2")
    hero.set_title("a  Cross-mouse event geometry", loc="left", fontweight="bold")
    hero.text(0.02, 0.02, "one point = one independent clinical event\nCH8 only; each mouse has equal total weight", transform=hero.transAxes, fontsize=5.7, va="bottom", color="#404040")

    ax = fig.add_subplot(grid[0, 1])
    all_events = eligibility.groupby("animal_id").size().reindex(ANIMALS, fill_value=0)
    selected_events = eligibility.loc[eligibility["eligibility"].eq("eligible")].groupby("animal_id").size().reindex(ANIMALS, fill_value=0)
    y = np.arange(len(ANIMALS))
    ax.barh(y, selected_events, color="#0072B2", label="eligible")
    ax.barh(y, all_events - selected_events, left=selected_events, color="#D9D9D9", label="not in primary geometry")
    ax.set_yticks(y, ANIMALS)
    ax.invert_yaxis()
    ax.set_xlabel("events")
    ax.set_title("b  CH8 event admission", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=5.2, loc="lower right")

    ax = fig.add_subplot(grid[0, 2])
    resolution = _state_resolution(candidates)
    old = audit["legacy_local_candidate_counts_by_mouse"]
    x = np.arange(len(ANIMALS))
    ax.plot(x, [old.get(mouse, np.nan) for mouse in ANIMALS], "o--", color="#6E6E6E", lw=0.8, ms=3.8, label="legacy local candidates")
    ax.plot(x, resolution["components"].to_numpy(float), "o", color="#D55E00", ms=4.3, label="event-level stable resolution")
    ax.set_xticks(x, ANIMALS)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("state count")
    ax.set_title("c  Resolution audit", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=5.0, loc="upper right")

    ax = fig.add_subplot(grid[1, 1])
    stability = candidates.loc[candidates["components"].gt(1)].copy()
    for mouse in ANIMALS:
        rows = stability.loc[stability["animal_id"].eq(mouse)]
        if rows.empty:
            continue
        ax.plot(rows["components"], rows["bootstrap_median_ari"], marker=MARKERS[mouse], color=COLORS[mouse], lw=0.8, ms=3.4)
    ax.axhline(0.60, color="#666666", ls="--", lw=0.7)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("candidate components")
    ax.set_ylabel("event-bootstrap ARI")
    ax.set_title("d  Local stability", loc="left", fontweight="bold")
    ax.text(0.98, 0.05, "dashed: prespecified\nstability threshold", ha="right", va="bottom", transform=ax.transAxes, fontsize=5.2, color="#505050")

    ax = fig.add_subplot(grid[1, 2])
    ranked = loadings.assign(abs_pc1=loadings["pc1_loading"].abs()).sort_values("abs_pc1", ascending=False).head(7).iloc[::-1]
    ax.barh(np.arange(len(ranked)), ranked["pc1_loading"], color=np.where(ranked["pc1_loading"] >= 0, "#0072B2", "#D55E00"))
    ax.axvline(0, color="#333333", lw=0.5)
    ax.set_yticks(np.arange(len(ranked)), [_feature_label(value) for value in ranked["feature"]])
    ax.tick_params(axis="y", labelsize=5.7, pad=1.5)
    ax.set_xlabel("PC1 loading")
    ax.set_title("e  Main physiological axis", loc="left", fontweight="bold")

    cohort_title = (
        "Combined seven-mouse morphology: event-balanced CH8 reassessment"
        if "E07" in ANIMALS
        else "Legacy six-mouse morphology: event-balanced CH8 reassessment"
    )
    fig.suptitle(cohort_title, x=0.51, y=0.99, fontsize=10, fontweight="bold")
    return _save(fig, stem)


def figure_s1(prototypes: pd.DataFrame, stem: Path) -> list[Path]:
    """Read-only mapping for every historical local state in the 21-state inventory."""

    _style()
    fig, ax = plt.subplots(figsize=(WIDTH_MM / 25.4, 4.25), constrained_layout=True)
    main = prototypes.loc[
        prototypes["weighted_pc1"].between(-5.4, 3.3)
        & prototypes["weighted_pc2"].between(-2.1, 2.1)
    ].copy()
    offsets = {
        "E50-C1": (4, 4),
        "E50-C2": (4, 6),
        "E50-C3": (4, 4),
        "E50-C4": (4, 4),
        "E50-C5": (4, 4),
        "E50-C6": (4, 4),
        "E50-C7": (4, 4),
        "E76-C1": (4, 4),
        "E76-C2": (4, 4),
        "E76-C3": (4, 4),
        "E77-C1": (5, 8),
        "E77-C2": (4, 4),
        "E77-C3": (4, -9),
        "E79-C1": (4, 4),
        "E79-C2": (4, 4),
        "E79-C3": (4, 4),
        "E79-C4": (4, 4),
        "E81-C1": (4, 4),
        "E82-C1": (4, 4),
        "E82-C2": (4, 4),
    }
    for mouse in ANIMALS:
        rows = main.loc[main["animal_id"].eq(mouse)].sort_values(OLD_LABEL, key=lambda item: item.map(_label_number))
        if rows.empty:
            continue
        ax.scatter(rows["weighted_pc1"], rows["weighted_pc2"], s=56, color=COLORS[mouse], marker=MARKERS[mouse], edgecolor="white", linewidth=0.7, label=mouse, zorder=3)
        for row in rows.itertuples(index=False):
            state = str(getattr(row, OLD_LABEL))
            ax.annotate(state, (row.weighted_pc1, row.weighted_pc2), xytext=offsets.get(state, (4, 4)), textcoords="offset points", fontsize=5.4, color="#202020")
    ax.set_xlim(-5.4, 3.3)
    ax.set_ylim(-2.1, 2.1)
    ax.set_xlabel("Mouse-balanced PC1")
    ax.set_ylabel("Mouse-balanced PC2")
    ax.set_title("Historical local candidates in the common event space (zoomed)\nEach label is a read-only mapping, not a global clinical class", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, title="mouse", loc="best")
    # The upper-left region is empty in the zoomed view, so the full-range
    # inset does not obscure a mapped candidate in the primary panel.
    inset = inset_axes(ax, width="30%", height="35%", loc="upper left", borderpad=1.2)
    for mouse in ANIMALS:
        rows = prototypes.loc[prototypes["animal_id"].eq(mouse)]
        inset.scatter(rows["weighted_pc1"], rows["weighted_pc2"], s=18, color=COLORS[mouse], marker=MARKERS[mouse], linewidths=0, zorder=3)
    outlier = prototypes.loc[~prototypes.index.isin(main.index)]
    for row in outlier.itertuples(index=False):
        inset.annotate(str(getattr(row, OLD_LABEL)), (row.weighted_pc1, row.weighted_pc2), xytext=(2, 2), textcoords="offset points", fontsize=5.0)
    inset.set_title("all 21 (full range)", fontsize=5.2, pad=2)
    inset.tick_params(labelsize=5.0, length=2)
    inset.spines[["top", "right"]].set_visible(True)
    return _save(fig, stem)


def figure_s2(cells: pd.DataFrame, stem: Path) -> list[Path]:
    """Compact spectral prototype heatmap for all historical candidate states."""

    _style()
    features = [
        "norm__ch8_relative_delta",
        "norm__ch8_relative_theta",
        "norm__ch8_relative_alpha",
        "norm__ch8_relative_beta",
        "norm__ch8_relative_low_gamma_clean",
        "norm__ch8_relative_high_gamma",
        "norm__ch8_theta_delta_ratio",
        "norm__ch8_spectral_entropy",
    ]
    labels = (
        cells.groupby(["animal_id", OLD_LABEL], sort=True)[features].median().reset_index()
        .sort_values(["animal_id", OLD_LABEL], key=lambda col: col.map(_label_number) if col.name == OLD_LABEL else col)
    )
    matrix = labels[features].to_numpy(float)
    limit = max(1.0, float(np.nanquantile(np.abs(matrix), 0.98)))
    fig, ax = plt.subplots(figsize=(WIDTH_MM / 25.4, 5.3), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_yticks(np.arange(len(labels)), [str(getattr(row, OLD_LABEL)) for row in labels.itertuples(index=False)])
    ax.set_xticks(np.arange(len(features)), [_feature_label(name) for name in features], rotation=35, ha="right")
    ax.set_title("Read-only spectral/physiological profiles of the historical local candidates\nValues are median CH8 session-baseline standardized features", loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("relative baseline (robust z)")
    for start in np.cumsum(labels.groupby("animal_id", sort=True).size().to_numpy())[:-1]:
        ax.axhline(start - 0.5, color="white", lw=1.2)
    return _save(fig, stem)


def generate_figures(output_root: str | Path) -> dict[str, list[str]]:
    root = Path(output_root).resolve()
    audit = json_load(root / "01_protocol" / "source_audit.json")
    pca = pd.read_parquet(root / "04_cross_mouse" / "mouse_balanced_event_pca.parquet")
    eligibility = pd.read_csv(root / "02_feature_inventory" / "event_eligibility_audit.csv")
    candidates = pd.read_csv(root / "03_per_mouse" / "event_state_resolution_candidates.csv")
    loadings = pd.read_csv(root / "04_cross_mouse" / "mouse_balanced_pca_loadings.csv")
    prototypes = pd.read_csv(root / "04_cross_mouse" / "legacy_21_read_only_pca_centroids.csv")
    cells = pd.read_parquet(root / "04_cross_mouse" / "legacy_21_event_state_cells.parquet")
    # Keep the historical layout and color grammar, but derive the cohort
    # legend from the actual input.  This prevents a combined run from
    # silently dropping a newly admitted animal (for example E07) merely
    # because the original six-mouse constant predates it.
    global ANIMALS
    observed = tuple(sorted(pca["animal_id"].astype(str).unique()))
    if observed:
        ANIMALS = observed
        palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#6A3D9A", "#1B9E77"]
        markers = ["o", "s", "^", "D", "P", "X", "v", "<"]
        for i, animal in enumerate(ANIMALS):
            COLORS.setdefault(animal, palette[i % len(palette)])
            MARKERS.setdefault(animal, markers[i % len(markers)])
    figure_root = root / "05_figures"
    outputs = {
        "figure_1": [str(path) for path in figure_1(pca, eligibility, candidates, audit, loadings, figure_root / "figure_1_event_balanced_cross_mouse")],
        "figure_s1": [str(path) for path in figure_s1(prototypes, figure_root / "figure_s1_legacy_21_mapping")],
        "figure_s2": [str(path) for path in figure_s2(cells, figure_root / "figure_s2_legacy_21_spectral_profiles")],
    }
    (figure_root / "FIGURE_CONTRACT.md").write_text(
        """# Figure Contract

Core conclusion: the historical 21 labels are within-mouse exploratory candidates; an independent-event, CH8-primary and mouse-balanced audit is required before any global taxonomy claim.

Archetype: asymmetric quantitative figure with a cross-mouse PCA hero panel.

Panel logic: Figure 1a is the hero event geometry; 1b verifies source admission; 1c-d audit local resolution and stability; 1e names the dominant physiological axis. Figure S1 provides every historical local-state mapping. Figure S2 provides compact spectral/physiological prototypes.

Statistics: one point is one independent clinical event. PCA weights give every mouse equal total influence. Counts are descriptive; no classifier accuracy is plotted.

Integrity: existing scalar features only; no raw waveform image adjustment or regenerated signal.
""",
        encoding="utf-8",
    )
    (figure_root / "FIGURE_QA.md").write_text(
        """# Figure QA

- Python/matplotlib only; SVG/PDF editable text is configured.
- Every figure is exported as SVG, PDF, TIFF at 600 dpi, and PNG preview at 300 dpi.
- Values trace to the event eligibility, event PCA, old-state centroid, and old-state-cell tables in numbered output folders.
- Historical local labels are explicitly marked read-only and non-clinical.
""",
        encoding="utf-8",
    )
    return outputs


def json_load(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["generate_figures"]
