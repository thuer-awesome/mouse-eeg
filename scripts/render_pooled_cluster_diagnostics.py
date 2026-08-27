#!/usr/bin/env python
"""Render the statistical audit behind the pooled PG taxonomy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "output" / "epilepsy_program_20260821" / "02_morphology_taxonomy" / "all_mice_old_plus_new" / "strict_all_channel_v1"
FEATURES = [
    "log_robust_amplitude", "log_rms", "log_line_length", "hjorth_mobility", "hjorth_complexity", "peak_frequency_hz", "spectral_entropy", "relative_delta", "relative_theta", "relative_beta", "relative_low_gamma_clean", "relative_high_gamma",
    "baseline_z__log_robust_amplitude", "baseline_z__log_rms", "baseline_z__log_line_length", "baseline_z__hjorth_mobility", "baseline_z__hjorth_complexity", "baseline_z__peak_frequency_hz", "baseline_z__spectral_entropy", "baseline_z__relative_delta", "baseline_z__relative_theta", "baseline_z__relative_beta", "baseline_z__relative_low_gamma_clean", "baseline_z__relative_high_gamma",
]
MOUSE_COLOURS = {"E07": "#0072B2", "E50": "#D55E00", "E76": "#009E73", "E77": "#CC79A7", "E79": "#E69F00", "E81": "#56B4E9", "E82": "#6A3D9A"}

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7, "axes.spines.right": False, "axes.spines.top": False, "svg.fonttype": "none", "pdf.fonttype": 42})


def _layout_paths(root: Path) -> tuple[Path, Path, Path, Path]:
    """Resolve the source-study or compact-release table layout."""

    if (root / "taxonomy").is_dir():
        return (
            root / "tables" / "channel_event_features.parquet",
            root / "taxonomy" / "pooled_event_labels.parquet",
            root / "taxonomy" / "pooled_category_registry.csv",
            root / "tables" / "pooled_mouse_event_assignments.parquet",
        )
    return (
        root / "01_tables" / "channel_event_features.parquet",
        root / "04_label_freeze_pooled" / "pooled_event_labels.parquet",
        root / "04_label_freeze_pooled" / "pooled_category_registry.csv",
        root / "03_cross_mouse" / "pooled_mouse_event_assignments.parquet",
    )


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def finite(frame: pd.DataFrame) -> np.ndarray:
    x = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    return x.fillna(x.median()).fillna(0.0).to_numpy(float)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--figure-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    fig_root = (args.figure_dir.resolve() if args.figure_dir is not None else root / "06_figures" / "paper_ready_morphology_pooled") / "06_cluster_diagnostics"
    feature_path, labels_path, registry_path, assignments_path = _layout_paths(root)
    events = pd.read_parquet(feature_path)
    events = events.loc[events.channel_name.astype(str).isin(["CH7", "CH8"])].copy()
    events = events.loc[~(events.animal_id.astype(str).eq("E77") & events.channel_name.astype(str).eq("CH7"))].copy()
    labels = pd.read_parquet(labels_path)
    events = events.merge(labels[["animal_id", "session_id", "event_id", "channel_name", "global_label", "pooled_label"]], on=["animal_id", "session_id", "event_id", "channel_name"], how="inner", validate="one_to_one")
    # Subsample only for display; the complete feature table remains in the
    # parquet audit and the PCA explained-variance report.
    # Deterministic display thinning is used only if a future cohort exceeds
    # the plotting limit; the complete coordinates remain in the parquet file.
    display_n = min(24000, len(events))
    take = np.linspace(0, len(events) - 1, display_n, dtype=int)
    display = events.iloc[np.sort(take)].copy()
    z = RobustScaler(quantile_range=(10, 90)).fit_transform(finite(events))
    pca = PCA(n_components=3, random_state=20260823).fit(z)
    coords = pca.transform(z)
    source_dir = fig_root / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"animal_id": events.animal_id.to_numpy(), "channel_name": events.channel_name.to_numpy(), "global_label": events.global_label.to_numpy(), "pc1": coords[:, 0], "pc2": coords[:, 1], "pc3": coords[:, 2]}).to_parquet(source_dir / "pooled_all_event_3d_pca_coordinates.parquet", index=False)
    display_coords = coords[np.sort(take)]

    # Panel A: true three-dimensional joint structure, with mouse and channel
    # encoded redundantly as colour and marker.
    fig = plt.figure(figsize=(183 / 25.4, 82 / 25.4))
    ax = fig.add_subplot(121, projection="3d")
    for mouse in sorted(display.animal_id.astype(str).unique()):
        for channel, marker in (("CH7", "o"), ("CH8", "^")):
            mask = display.animal_id.astype(str).eq(mouse) & display.channel_name.astype(str).eq(channel)
            idx = np.flatnonzero(mask.to_numpy())
            if len(idx):
                ax.scatter(display_coords[idx, 0], display_coords[idx, 1], display_coords[idx, 2], s=4, alpha=.22, marker=marker, color=MOUSE_COLOURS.get(mouse, "#777777"), linewidths=0)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", labelpad=2)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)", labelpad=2)
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2] * 100:.1f}%)", labelpad=2)
    ax.set_title("All event×channel points", loc="left", fontsize=7.5, fontweight="bold")
    ax.view_init(elev=22, azim=-58)
    handles = [Line2D([], [], marker="o", linestyle="none", color="#444444", label="CH7"), Line2D([], [], marker="^", linestyle="none", color="#444444", label="CH8")]
    ax.legend(handles=handles, frameon=False, fontsize=5.8, loc="upper left")

    # Panel B: pooled component counts and training support per mouse.
    summary_path = root / "03_cross_mouse" / "pooled_mouse_cluster_summary.csv"
    if summary_path.is_file():
        summary = pd.read_csv(summary_path).sort_values("animal_id")
    elif assignments_path.is_file():
        assignments = pd.read_parquet(assignments_path)
        summary = (
            assignments.groupby("animal_id", sort=True)
            .agg(n_pooled_clusters=("pooled_cluster", "nunique"), n_train_events=("event_id", "nunique"))
            .reset_index()
        )
    else:
        summary = pd.DataFrame(columns=["animal_id", "n_pooled_clusters", "n_train_events"])
    ax2 = fig.add_subplot(122)
    x = np.arange(len(summary)); bars = ax2.bar(x, summary.n_pooled_clusters, color=[MOUSE_COLOURS.get(str(v), "#777777") for v in summary.animal_id], width=.62)
    ax2.set_xticks(x, summary.animal_id.astype(str)); ax2.set_ylabel("Pooled clusters selected"); ax2.set_title("Within-mouse CH7 + CH8 pooled fits", loc="left", fontsize=7.5, fontweight="bold")
    ax2.grid(axis="y", color="#D7DADD", linewidth=.45); ax2.set_axisbelow(True)
    for bar, support in zip(bars, summary.n_train_events): ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .1, f"n={int(support)}", ha="center", va="bottom", fontsize=5.2)
    fig.suptitle("Statistical structure of the pooled morphology taxonomy", fontsize=9, fontweight="bold", x=.01, ha="left")
    fig.text(.01, .01, "Pxx is fitted jointly within each mouse from CH7 + CH8 event points; display points are subsampled, while all coordinates and counts are retained in the audit tables.", fontsize=5.8, color="#5F6972")
    fig.subplots_adjust(left=.04, right=.98, top=.86, bottom=.14, wspace=.28)
    save(fig, fig_root / "figure_6_pooled_3d_pca_and_cluster_counts")

    # Model-selection diagnostics: BIC gain and bootstrap stability for each
    # mouse-level pooled GMM, kept as a supplementary small-multiple figure.
    grids = sorted((root / "03_cross_mouse" / "pooled_mouse_gmm").glob("pooled_gmm_grid_*.csv"))
    fig, axes = plt.subplots(2, 4, figsize=(183 / 25.4, 88 / 25.4), sharex=False, sharey=False, constrained_layout=True)
    for ax, path in zip(axes.ravel(), grids):
        grid = pd.read_csv(path); mouse = path.stem.split("_")[-1]
        ax2b = ax.twinx(); ax.plot(grid.components, grid.bic_gain_vs_k1, "o-", color="#0072B2", lw=1, ms=3); ax2b.plot(grid.components, grid.stability_ari, "s--", color="#D55E00", lw=.9, ms=2.7)
        selected = grid.loc[grid.selected]
        if not selected.empty:
            ax.axvline(float(selected.components.iloc[0]), color="#333333", lw=.6, ls=":")
        ax.set_title(mouse, loc="left", fontsize=7, fontweight="bold"); ax.set_xlabel("k", fontsize=6); ax.set_ylabel("BIC gain", fontsize=6); ax2b.set_ylabel("Stability ARI", fontsize=6); ax.tick_params(labelsize=5.5); ax2b.tick_params(labelsize=5.5); ax.grid(axis="y", color="#E0E3E6", lw=.4)
    for ax in axes.ravel()[len(grids):]: ax.axis("off")
    handles = [Line2D([], [], color="#0072B2", marker="o", label="BIC gain vs k=1"), Line2D([], [], color="#D55E00", marker="s", ls="--", label="Bootstrap stability ARI")]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.03), ncol=2, fontsize=6)
    fig.suptitle("Pooled GMM model-selection audit", fontsize=9, fontweight="bold", y=1.08)
    fig.text(.01, -.02, "Vertical dotted line marks the selected component count; this is a statistical resolution audit, not a clinical subtype claim.", fontsize=5.8, color="#5F6972")
    save(fig, fig_root / "figure_7_pooled_gmm_bic_and_stability")

    registry = pd.read_csv(registry_path).sort_values("n_event_channel", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(183 / 25.4, 76 / 25.4), constrained_layout=True)
    colours = ["#D55E00" if bool(v) else "#0072B2" for v in registry.mouse_specific_flag]
    ax.bar(np.arange(len(registry)), registry.n_event_channel, color=colours, width=.8)
    ax.set_xlabel("PG category ranked by event×channel support"); ax.set_ylabel("Event×channel support"); ax.set_title(f"Frozen PG support ({len(registry)} categories)", loc="left", fontsize=7.5, fontweight="bold"); ax.set_xticks(np.arange(len(registry)), registry.pooled_global_label.astype(str), rotation=75, ha="right", fontsize=5.2); ax.grid(axis="y", color="#D7DADD", lw=.45); ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker="s", color="#0072B2", linestyle="none", label="shared across ≥2 mice"), Line2D([], [], marker="s", color="#D55E00", linestyle="none", label="animal-specific")], frameon=False, fontsize=6, ncol=2, loc="upper right")
    fig.text(.01, -.03, "PG labels are exploratory morphology categories; no category is interpreted as a clinical seizure subtype without review.", fontsize=5.8, color="#5F6972")
    save(fig, fig_root / "figure_8_pg_support_ranked")
    manifest = {"figures": ["figure_6_pooled_3d_pca_and_cluster_counts", "figure_7_pooled_gmm_bic_and_stability", "figure_8_pg_support_ranked"], "n_event_channel": int(len(events)), "n_display_points": int(len(display)), "pca_components": 3, "n_pg_categories": int(len(registry)), "pca_explained_variance": pca.explained_variance_ratio_.tolist(), "source_tables": {"event_features": str(feature_path), "event_labels": str(labels_path), "category_registry": str(registry_path), "pooled_assignments": str(assignments_path)}}
    (fig_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
