#!/usr/bin/env python
"""Render the corrected PG01-PG36 morphology/classification atlas.

This renderer is deliberately separate from the earlier exploratory figure
packages.  The formal visual contract is:

* one point = one independent event x valid channel observation;
* PG01-PG36 are the only formal morphology labels;
* CH7 and CH8 are encoded independently (marker and colour), although both
  streams from a mouse can be displayed in the same mouse-level panel;
* one PG category = one point in the global centroid PCA figure;
* every class is retained in the PR/ROC audit, with no ``typical`` class
  filtering.

The script never modifies the frozen labels, model predictions, or previous
figure directories.  It writes a new, auditable v2 package.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
TAX_ROOT = (
    ROOT
    / "output"
    / "epilepsy_program_20260821"
    / "02_morphology_taxonomy"
    / "all_mice_old_plus_new"
    / "strict_all_channel_v1"
)
DEFAULT_OUT = TAX_ROOT / "06_figures" / "paper_ready_morphology_pooled_v2"

FEATURES = [
    "log_robust_amplitude",
    "log_rms",
    "log_line_length",
    "hjorth_mobility",
    "hjorth_complexity",
    "peak_frequency_hz",
    "spectral_entropy",
    "relative_delta",
    "relative_theta",
    "relative_beta",
    "relative_low_gamma_clean",
    "relative_high_gamma",
    "baseline_z__log_robust_amplitude",
    "baseline_z__log_rms",
    "baseline_z__log_line_length",
    "baseline_z__hjorth_mobility",
    "baseline_z__hjorth_complexity",
    "baseline_z__peak_frequency_hz",
    "baseline_z__spectral_entropy",
    "baseline_z__relative_delta",
    "baseline_z__relative_theta",
    "baseline_z__relative_beta",
    "baseline_z__relative_low_gamma_clean",
    "baseline_z__relative_high_gamma",
]
KEYS = ["animal_id", "session_id", "event_id", "channel_name"]
CHANNELS = ["CH7", "CH8"]
CHANNEL_COLOURS = {"CH7": "#0072B2", "CH8": "#D55E00"}
CHANNEL_MARKERS = {"CH7": "o", "CH8": "^"}
MOUSE_COLOURS = {
    "E07": "#0072B2",
    "E50": "#D55E00",
    "E76": "#009E73",
    "E77": "#CC79A7",
    "E79": "#E69F00",
    "E81": "#56B4E9",
    "E82": "#6A3D9A",
}
PG_ORDER = [f"PG{i:02d}" for i in range(1, 37)]
PG_COLOURS = {
    label: to_hex(mpl.colormaps["turbo"](idx / max(1, len(PG_ORDER) - 1)))
    for idx, label in enumerate(PG_ORDER)
}
INK = "#252A2E"
MID_GREY = "#66717A"
LIGHT_GREY = "#D9DEE3"


def natural(value: object) -> int:
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else 999999


def save_figure(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def finite_features(frame: pd.DataFrame) -> np.ndarray:
    values = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    values = values.apply(pd.to_numeric, errors="coerce")
    values = values.fillna(values.median()).fillna(0.0)
    return values.to_numpy(float)


def robust_pca(frame: pd.DataFrame, n_components: int = 2) -> tuple[np.ndarray, PCA]:
    z = RobustScaler(quantile_range=(10, 90)).fit_transform(finite_features(frame))
    model = PCA(n_components=n_components, random_state=20260823)
    return model.fit_transform(z), model


def display_limits(values: Iterable[float], quantile: tuple[float, float] = (0.01, 0.99)) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -1.0, 1.0, 0.0
    lo_full, hi_full = float(np.min(arr)), float(np.max(arr))
    qlo, qhi = np.quantile(arr, quantile)
    span = max(float(qhi - qlo), 1e-8)
    # Keep the complete range when tails are modest; otherwise use a clearly
    # documented central display window while retaining all coordinates.
    tail_fraction = float(((arr < qlo) | (arr > qhi)).mean())
    if (hi_full - lo_full) <= 4.0 * span:
        lo, hi = lo_full, hi_full
    else:
        lo, hi = float(qlo - 0.10 * span), float(qhi + 0.10 * span)
    pad = max(0.04 * (hi - lo), 0.15)
    return lo - pad, hi + pad, tail_fraction


def repel_positions(points: np.ndarray, min_dist: float = 0.075, iterations: int = 350) -> np.ndarray:
    """Deterministic label relaxation in normalized display coordinates."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return points.copy()
    positions = points.copy()
    anchors = points.copy()
    for _ in range(iterations):
        delta = np.zeros_like(positions)
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = positions[i] - positions[j]
                dist = float(np.linalg.norm(d))
                if dist < min_dist:
                    if dist < 1e-9:
                        direction = np.array([1.0, 0.0]) if i % 2 == 0 else np.array([0.0, 1.0])
                    else:
                        direction = d / dist
                    push = 0.5 * (min_dist - dist) * direction
                    delta[i] += push
                    delta[j] -= push
        # Tether labels to their category centroid to avoid drifting to the
        # page edge when a mouse has many nearby categories.
        delta += 0.045 * (anchors - positions)
        positions += 0.30 * delta
        positions[:, 0] = np.clip(positions[:, 0], 0.035, 0.965)
        positions[:, 1] = np.clip(positions[:, 1], 0.055, 0.945)
    return positions


def annotate_repelled(ax: plt.Axes, centers: pd.DataFrame, label_col: str = "global_label") -> None:
    if centers.empty:
        return
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    norm = np.column_stack(
        [
            (centers.pc1.to_numpy(float) - xlo) / max(xhi - xlo, 1e-9),
            (centers.pc2.to_numpy(float) - ylo) / max(yhi - ylo, 1e-9),
        ]
    )
    # Text boxes are wider than the centroid markers; use a conservative
    # separation so adjacent PG labels remain individually readable.
    positions = repel_positions(norm, min_dist=max(0.075, 1.20 / math.sqrt(len(centers))), iterations=700)
    for idx, row in centers.reset_index(drop=True).iterrows():
        px = xlo + positions[idx, 0] * (xhi - xlo)
        py = ylo + positions[idx, 1] * (yhi - ylo)
        cx, cy = float(row.pc1), float(row.pc2)
        label = str(row[label_col])
        colour = PG_COLOURS.get(label, INK)
        if abs(px - cx) + abs(py - cy) > 0.02 * max(xhi - xlo, yhi - ylo):
            ax.plot([cx, px], [cy, py], color="#9AA3AA", lw=0.35, zorder=3)
        ax.text(
            px,
            py,
            label,
            ha="center",
            va="center",
            fontsize=5.0 if len(centers) > 12 else 6.0,
            fontweight="bold",
            color=INK,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": colour, "linewidth": 0.55, "alpha": 0.94},
            zorder=5,
        )


def _layout_paths(tax_root: Path) -> tuple[Path, Path, Path, Path]:
    """Resolve either the source-study tree or the cleaned release tree."""
    if (tax_root / "taxonomy").is_dir():
        return (
            tax_root / "tables" / "channel_event_features.parquet",
            tax_root / "taxonomy" / "pooled_event_labels.parquet",
            tax_root / "taxonomy" / "pooled_category_registry.csv",
            tax_root / "classification",
        )
    return (
        tax_root / "01_tables" / "channel_event_features.parquet",
        tax_root / "04_label_freeze_pooled" / "pooled_event_labels.parquet",
        tax_root / "04_label_freeze_pooled" / "pooled_category_registry.csv",
        tax_root / "05_classification_pooled",
    )


def load_events(tax_root: Path) -> pd.DataFrame:
    feature_path, label_path, _, _ = _layout_paths(tax_root)
    features = pd.read_parquet(feature_path)
    labels = pd.read_parquet(label_path)
    labels = labels[KEYS + ["global_label", "global_category", "global_distance", "low_confidence"]]
    merged = features.merge(labels, on=KEYS, how="inner", validate="one_to_one")
    merged = merged.loc[merged.channel_name.astype(str).isin(CHANNELS)].copy()
    merged = merged.loc[~(merged.animal_id.astype(str).eq("E77") & merged.channel_name.astype(str).eq("CH7"))].copy()
    merged["global_label"] = merged["global_label"].astype(str)
    merged = merged.loc[merged.global_label.isin(PG_ORDER)].copy()
    return merged.reset_index(drop=True)


def choose_representatives(events: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Choose one real event row nearest each group feature centroid."""
    work = events.copy()
    work["_event_key"] = work[KEYS].astype(str).agg("|".join, axis=1)
    work = work.drop_duplicates("_event_key", keep="first")
    rows: list[pd.Series] = []
    for group_key, group in work.groupby(keys, sort=True):
        group = group.copy()
        mat = finite_features(group)
        if len(group) > 1:
            z = RobustScaler(quantile_range=(10, 90)).fit_transform(mat)
            center = np.nanmedian(z, axis=0)
            distance = np.sqrt(np.nanmean((z - center) ** 2, axis=1))
            row = group.iloc[int(np.nanargmin(distance))].copy()
        else:
            row = group.iloc[0].copy()
        row["event_channel_support"] = int(len(group))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=list(events.columns) + ["event_channel_support"])
    return pd.DataFrame(rows).drop(columns=["_event_key"], errors="ignore").reset_index(drop=True)


def load_signal(row: pd.Series) -> np.ndarray:
    signals = np.load(Path(str(row["cache_dir"])) / "signals.npy", mmap_mode="r")
    signal = np.asarray(signals[int(row["cache_local_index"])], dtype=float)
    if signal.ndim == 2:
        idx = 0 if str(row["channel_name"]).upper() == "CH7" else 1
        signal = signal[idx]
    signal = signal.reshape(-1)
    signal = signal - np.nanmedian(signal)
    scale = np.nanpercentile(signal, 95) - np.nanpercentile(signal, 5)
    return signal / max(float(scale), 1e-8)


def plot_wave_panel(ax: plt.Axes, row: pd.Series | None, label: str, channel: str, support: int) -> None:
    colour = CHANNEL_COLOURS[channel]
    ax.set_xlim(0, 4)
    ax.set_xticks([0, 2, 4])
    ax.set_yticks([])
    ax.set_ylim(-3.2, 3.2)
    ax.axhline(0, color="#B9BEC3", lw=0.4, zorder=0)
    if row is None:
        ax.set_title(f"{label} | {channel} | n=0", loc="left", fontsize=6.5, fontweight="bold", color=colour)
        ax.text(0.5, 0.50, "not observed", transform=ax.transAxes, ha="center", va="center", color=MID_GREY, fontsize=7)
        ax.text(0.01, 0.05, "PG category absent in this channel", transform=ax.transAxes, fontsize=5.0, color=MID_GREY)
        return
    signal = load_signal(row)
    fs = float(row.get("sampling_rate_hz", 250.0))
    time = np.arange(len(signal), dtype=float) / fs
    ax.plot(time, signal, color=colour, lw=0.65)
    ax.set_title(f"{label} | {channel} | n={support:,}", loc="left", fontsize=6.5, fontweight="bold", color=colour)
    ax.text(0.01, 0.05, "one independent event x channel representative", transform=ax.transAxes, fontsize=5.0, color=MID_GREY)
    freq, power = welch(signal, fs=fs, nperseg=min(512, len(signal)))
    mask = (freq >= 0.5) & (freq <= 70)
    inset = ax.inset_axes([0.62, 0.60, 0.33, 0.30])
    denom = max(float(np.nanmax(power[mask])) if np.any(mask) else 1.0, 1e-12)
    inset.plot(freq[mask], power[mask] / denom, color=colour, lw=0.55)
    inset.set_xlim(0.5, 70); inset.set_xticks([]); inset.set_yticks([]); inset.set_title("PSD", fontsize=5, pad=1)
    for spine in inset.spines.values():
        spine.set_color("#B8BEC4"); spine.set_linewidth(0.45)


def render_per_mouse_pca(events: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for animal, group in events.groupby("animal_id", sort=True):
        group = group.copy()
        coords, pca = robust_pca(group)
        group["pc1"], group["pc2"] = coords[:, 0], coords[:, 1]
        group["pc1_variance_fraction"] = float(pca.explained_variance_ratio_[0])
        group["pc2_variance_fraction"] = float(pca.explained_variance_ratio_[1])
        rows.append(group)
    coords = pd.concat(rows, ignore_index=True)
    source_dir = out / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    coords.to_parquet(source_dir / "per_mouse_pg_coordinates.parquet", index=False)
    support = (
        coords.groupby(["animal_id", "global_label", "channel_name"], sort=True)
        .size()
        .rename("n_event_channel")
        .reset_index()
    )
    support.to_csv(source_dir / "per_mouse_pg_channel_support.csv", index=False, encoding="utf-8-sig")
    root = out / "01_per_mouse_pg_pca"
    for animal, group in coords.groupby("animal_id", sort=True):
        fig = plt.figure(figsize=(183 / 25.4, 88 / 25.4))
        gs = fig.add_gridspec(1, 2, width_ratios=[3.8, 1.25], wspace=0.12)
        ax = fig.add_subplot(gs[0, 0])
        for channel in CHANNELS:
            h = group.loc[group.channel_name.astype(str).eq(channel)]
            if not h.empty:
                ax.scatter(h.pc1, h.pc2, s=6, alpha=0.17, color=CHANNEL_COLOURS[channel], marker=CHANNEL_MARKERS[channel], linewidths=0, rasterized=True)
        centers = group.groupby("global_label", sort=True).agg(pc1=("pc1", "median"), pc2=("pc2", "median"), support=("event_id", "size")).reset_index()
        for _, row in centers.iterrows():
            ax.scatter([row.pc1], [row.pc2], s=35, facecolor="white", edgecolor=PG_COLOURS.get(str(row.global_label), INK), linewidth=0.9, zorder=4)
        xlo, xhi, xtail = display_limits(group.pc1); ylo, yhi, ytail = display_limits(group.pc2)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        annotate_repelled(ax, centers)
        ax.set_xlabel(f"Mouse-local PC1 ({group.pc1_variance_fraction.iloc[0] * 100:.1f}%)")
        ax.set_ylabel(f"Mouse-local PC2 ({group.pc2_variance_fraction.iloc[0] * 100:.1f}%)")
        ax.set_title(f"{animal} | all independent event×channel points | {group.global_label.nunique()} PG categories", loc="left", fontsize=8.2, fontweight="bold")
        ax.text(0.01, 0.01, f"PG labels are global; CH7/CH8 remain independent markers. Display tail fractions: PC1 {xtail:.1%}, PC2 {ytail:.1%}.", transform=ax.transAxes, fontsize=5.2, color=MID_GREY)
        handles = [Line2D([], [], marker="o", linestyle="none", color=CHANNEL_COLOURS["CH7"], label="CH7"), Line2D([], [], marker="^", linestyle="none", color=CHANNEL_COLOURS["CH8"], label="CH8")]
        ax.legend(handles=handles, frameon=False, fontsize=6, loc="upper right", ncol=2)
        table = fig.add_subplot(gs[0, 1]); table.axis("off")
        table.set_title(f"{animal} PG support", loc="left", fontsize=7.5, fontweight="bold")
        pivot = support.loc[support.animal_id.eq(animal)].pivot(index="global_label", columns="channel_name", values="n_event_channel").fillna(0)
        for channel in CHANNELS:
            if channel not in pivot.columns: pivot[channel] = 0
        pivot["total"] = pivot[CHANNELS].sum(axis=1)
        pivot = pivot.reindex(sorted(pivot.index, key=natural))
        y = 0.94
        for label, row in pivot.iterrows():
            colour = PG_COLOURS.get(str(label), INK)
            table.add_patch(Rectangle((0.01, y - 0.018), 0.035, 0.035, transform=table.transAxes, facecolor=colour, edgecolor="none"))
            table.text(0.065, y, f"{label}  total={int(row.total)}", transform=table.transAxes, va="center", fontsize=5.8, fontweight="bold")
            table.text(0.065, y - 0.020, f"CH7={int(row.CH7)}   CH8={int(row.CH8)}", transform=table.transAxes, va="center", fontsize=5.1, color=MID_GREY)
            y -= 0.105 if len(pivot) <= 8 else 0.085
        table.text(0.01, 0.02, "Union across this mouse is a subset of PG01–PG36; it is not expected to contain all 36 in every mouse.", transform=table.transAxes, fontsize=5.2, color=MID_GREY, wrap=True)
        fig.suptitle("Within-mouse PG audit: both valid channels shown, no channel pairing", x=0.01, ha="left", y=0.995, fontsize=9, fontweight="bold")
        fig.subplots_adjust(left=0.06, right=0.99, top=0.89, bottom=0.13)
        save_figure(fig, root / f"{animal}_pg36_independent_points")
    return coords


def render_per_mouse_waveforms(events: pd.DataFrame, out: Path) -> None:
    reps = choose_representatives(events, ["animal_id", "global_label", "channel_name"])
    reps.to_csv(out / "source_data" / "per_mouse_pg_waveform_representatives.csv", index=False, encoding="utf-8-sig")
    root = out / "02_per_mouse_pg_waveforms"
    for animal, group in events.groupby("animal_id", sort=True):
        labels = sorted(group.global_label.astype(str).unique(), key=natural)
        support = group.groupby(["global_label", "channel_name"]).size().to_dict()
        lookup = {(str(r.global_label), str(r.channel_name)): r for _, r in reps.loc[reps.animal_id.eq(animal)].iterrows()}
        for page, start in enumerate(range(0, len(labels), 8), start=1):
            subset = labels[start : start + 8]
            fig, axes = plt.subplots(len(subset), 2, figsize=(183 / 25.4, max(60 / 25.4, len(subset) * 22 / 25.4)), squeeze=False)
            for ridx, label in enumerate(subset):
                for cidx, channel in enumerate(CHANNELS):
                    row = lookup.get((label, channel))
                    plot_wave_panel(axes[ridx, cidx], row, label, channel, int(support.get((label, channel), 0)))
                    if ridx == len(subset) - 1:
                        axes[ridx, cidx].set_xlabel("Time (s)")
                    else:
                        axes[ridx, cidx].set_xlabel("")
            fig.suptitle(f"{animal} | all observed global PG categories | independent CH7/CH8 waveforms", x=0.01, ha="left", y=0.995, fontsize=9, fontweight="bold")
            fig.text(0.01, 0.012, "Rows are global PG labels, not local Pxx/Lxx labels. Each cell is one real event×channel representative; absent channel combinations are explicit.", fontsize=5.8, color=MID_GREY)
            fig.subplots_adjust(left=0.035, right=0.995, top=0.955, bottom=0.085, wspace=0.12, hspace=0.52)
            save_figure(fig, root / f"{animal}_pg_waveforms_p{page:02d}")


def render_global_centroid_pca(events: pd.DataFrame, out: Path) -> pd.DataFrame:
    # Use the frozen training-fitted prototypes for the one-point-per-PG
    # geometry.  Event support is joined from the complete event table only
    # for annotation; held-out events never redefine a centroid.
    proto_path = TAX_ROOT / "04_label_freeze_pooled" / "pooled_global_prototypes.parquet"
    prototypes = pd.read_parquet(proto_path)
    support = events.groupby("global_label", sort=True).agg(
        n_event_channel=("event_id", "size"),
        n_observed_animals=("animal_id", "nunique"),
        observed_animals=("animal_id", lambda x: ";".join(sorted(x.astype(str).unique(), key=natural))),
    ).reset_index()
    proto_label = "pooled_global_label" if "pooled_global_label" in prototypes.columns else "global_label"
    rows = []
    for _, row in prototypes.iterrows():
        label = str(row[proto_label])
        record = {"global_label": label, "n_train_events": int(row.get("n_train_events", 0)), "n_member_animals": int(row.get("n_member_animals", row.get("n_animals", 0))), "member_animals": str(row.get("member_animals", ""))}
        for feature in FEATURES:
            source = f"f_{feature}"
            record[feature] = float(pd.to_numeric(row.get(source, np.nan), errors="coerce"))
        rows.append(record)
    centroids = pd.DataFrame(rows).set_index("global_label").reindex(PG_ORDER).reset_index()
    centroids = centroids.merge(support, on="global_label", how="left")
    centroids["n_event_channel"] = centroids["n_event_channel"].fillna(0).astype(int)
    centroids["n_observed_animals"] = centroids["n_observed_animals"].fillna(0).astype(int)
    centroids["observed_animals"] = centroids["observed_animals"].fillna("")
    centroids["n_animals"] = centroids["n_member_animals"]
    centroids["member_animals"] = centroids["member_animals"].replace("nan", "")
    centroids[FEATURES] = centroids[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    z = RobustScaler(quantile_range=(10, 90)).fit_transform(centroids[FEATURES].to_numpy(float))
    pca = PCA(n_components=2, random_state=20260823).fit(z)
    coords = pca.transform(z)
    centroids["pc1"], centroids["pc2"] = coords[:, 0], coords[:, 1]
    distances = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).mean(axis=2))
    np.fill_diagonal(distances, np.inf)
    centroids["nearest_centroid_distance"] = np.min(distances, axis=1)
    centroids["shared_across_mice"] = centroids.n_member_animals >= 2
    centroids.to_csv(out / "source_data" / "pg36_centroid_coordinates.csv", index=False, encoding="utf-8-sig")
    centroids.to_parquet(out / "source_data" / "pg36_centroid_coordinates.parquet", index=False)
    fig = plt.figure(figsize=(183 / 25.4, 88 / 25.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.26)
    ax = fig.add_subplot(gs[0, 0])
    for shared, h in centroids.groupby("shared_across_mice"):
        # Marker area encodes support only; category identity is carried by
        # the PG label border, so the plot remains readable for 36 classes.
        colour = "#0072B2" if bool(shared) else "#D55E00"
        ax.scatter(h.pc1, h.pc2, s=28 + 18 * np.log1p(h.n_event_channel), color=colour, alpha=0.82, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xlabel(f"PCA of PG centroids: PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("One point per global PG category", loc="left", fontsize=8, fontweight="bold")
    ax.set_xlim(*display_limits(centroids.pc1, (0.0, 1.0))[:2]); ax.set_ylim(*display_limits(centroids.pc2, (0.0, 1.0))[:2])
    annotate_repelled(ax, centroids)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="none", color="#0072B2", label="shared across ≥2 mice"), Line2D([], [], marker="o", linestyle="none", color="#D55E00", label="animal-specific")], frameon=False, fontsize=5.8, loc="best")
    ax2 = fig.add_subplot(gs[0, 1])
    ranked = centroids.sort_values(["nearest_centroid_distance", "global_label"], ascending=[True, True]).reset_index(drop=True)
    y = np.arange(len(ranked))
    colours = ["#0072B2" if bool(v) else "#D55E00" for v in ranked.shared_across_mice]
    ax2.barh(y, ranked.nearest_centroid_distance, color=colours, height=0.72)
    ax2.set_yticks(y, ranked.global_label.astype(str), fontsize=5.2)
    ax2.invert_yaxis(); ax2.set_xlabel("Nearest PG centroid distance (robust units)")
    ax2.set_title("Centroid separation audit", loc="left", fontsize=8, fontweight="bold")
    ax2.grid(axis="x", color="#D9DEE3", lw=0.4); ax2.set_axisbelow(True)
    fig.suptitle("Global PG36 morphology geometry", x=0.01, ha="left", y=0.995, fontsize=9, fontweight="bold")
    fig.text(0.01, 0.006, "All 36 global categories are retained as one feature centroid each. Event-level coordinates and support are in the source table; PG labels are exploratory morphology labels.", fontsize=5.8, color=MID_GREY)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.14)
    save_figure(fig, out / "03_global_pg_centroid_pca" / "figure_3_pg36_centroid_pca_and_separation")
    return centroids


def render_support_presence(events: pd.DataFrame, out: Path) -> None:
    streams = sorted((events.animal_id.astype(str) + "/" + events.channel_name.astype(str)).unique(), key=lambda x: (natural(x.split("/", 1)[0]), x))
    counts = events.assign(stream=events.animal_id.astype(str) + "/" + events.channel_name.astype(str)).groupby(["global_label", "stream"]).size().unstack(fill_value=0).reindex(index=PG_ORDER, columns=streams, fill_value=0)
    counts.to_csv(out / "source_data" / "pg36_mouse_channel_support_matrix.csv", encoding="utf-8-sig")
    membership = (
        events.groupby(["global_label", "animal_id"], sort=True)
        .agg(
            n_event_channel=("event_id", "size"),
            channels=("channel_name", lambda x: ";".join(sorted(set(x.astype(str))))),
        )
        .reset_index()
    )
    membership.to_csv(out / "source_data" / "pg36_category_mouse_membership.csv", index=False, encoding="utf-8-sig")
    root = out / "04_pg_support_presence"
    for page, labels in enumerate([PG_ORDER[:18], PG_ORDER[18:]], start=1):
        sub = counts.loc[labels]
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(183 / 25.4, 98 / 25.4), gridspec_kw={"width_ratios": [1.45, 1.0]})
        image = ax.imshow(np.log10(sub.to_numpy(float) + 1), aspect="auto", cmap="viridis")
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=6); ax.set_xticks(np.arange(len(streams)), [s.replace("/", "\n") for s in streams], fontsize=5.2)
        ax.set_xlabel("Independent mouse/channel stream"); ax.set_ylabel("Global PG label")
        ax.set_title("log10(event×channel support + 1)", loc="left", fontsize=7.5, fontweight="bold")
        for i in range(len(labels)):
            for j in range(len(streams)):
                value = int(sub.iloc[i, j])
                if value:
                    ax.text(j, i, str(value), ha="center", va="center", fontsize=5.0, color="white" if np.log10(value + 1) > 1.2 else INK)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="log10(count + 1)")
        presence = (sub.to_numpy(float) > 0).astype(float)
        ax2.imshow(presence, aspect="auto", cmap=mpl.colors.ListedColormap(["#F2F3F4", "#0072B2"]), vmin=0, vmax=1)
        ax2.set_yticks(np.arange(len(labels)), labels, fontsize=6); ax2.set_xticks(np.arange(len(streams)), [s.replace("/", "\n") for s in streams], fontsize=5.2)
        ax2.set_xlabel("Observed (blue) vs not observed"); ax2.set_title("Presence audit", loc="left", fontsize=7.5, fontweight="bold")
        fig.suptitle(f"PG36 support and channel presence | page {page}/2", x=0.01, ha="left", y=0.995, fontsize=9, fontweight="bold")
        fig.text(0.01, 0.006, "The union across all streams contains 36 PG labels. A mouse/channel stream can contain only its observed subset; absent cells are not missing labels.", fontsize=5.8, color=MID_GREY)
        fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.17, wspace=0.30)
        save_figure(fig, root / f"figure_4_pg_support_presence_p{page:02d}")


def plot_pr_or_roc_grid(pred: pd.DataFrame, model: str, out: Path, kind: str) -> None:
    y = pred["y_true"].astype(int).to_numpy()
    prob = pred[[f"prob_{label}" for label in PG_ORDER]].to_numpy(float)
    fig, axes = plt.subplots(6, 6, figsize=(183 / 25.4, 183 / 25.4), sharex=True, sharey=True)
    axes = axes.ravel()
    for idx, label in enumerate(PG_ORDER):
        ax = axes[idx]
        positive = (y == idx).astype(int)
        support = int(positive.sum())
        if kind == "pr":
            precision, recall, _ = precision_recall_curve(positive, prob[:, idx])
            ap = average_precision_score(positive, prob[:, idx]) if positive.sum() else float("nan")
            ax.plot(recall, precision, color=PG_COLOURS[label], lw=0.8)
            ax.axhline(positive.mean(), color="#B8BEC4", lw=0.45, ls="--")
            ax.set_ylabel("P", fontsize=5) if idx % 6 == 0 else None
            ax.set_xlabel("R", fontsize=5) if idx >= 30 else None
            ax.set_title(f"{label}  n={support}\nAP={ap:.3f}", fontsize=5.1, loc="left", color=INK)
        else:
            fpr, tpr, _ = roc_curve(positive, prob[:, idx])
            au = roc_auc_score(positive, prob[:, idx]) if positive.sum() and positive.sum() < len(positive) else float("nan")
            ax.plot(fpr, tpr, color=PG_COLOURS[label], lw=0.8)
            ax.plot([0, 1], [0, 1], color="#B8BEC4", lw=0.45, ls="--")
            ax.set_ylabel("TPR", fontsize=5) if idx % 6 == 0 else None
            ax.set_xlabel("FPR", fontsize=5) if idx >= 30 else None
            ax.set_title(f"{label}  n={support}\nAUROC={au:.3f}", fontsize=5.1, loc="left", color=INK)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.tick_params(labelsize=5.0, length=2); ax.grid(color="#EEF0F1", lw=0.3)
    fig.suptitle(f"{model} | all 36 PG classes | {'precision–recall' if kind == 'pr' else 'ROC'}", x=0.01, ha="left", y=0.995, fontsize=9, fontweight="bold")
    fig.text(0.01, 0.006, "Every PG class is shown in its own one-vs-rest panel. No class is removed as atypical; the old pale lines were only a readability de-emphasis.", fontsize=5.8, color=MID_GREY)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.95, bottom=0.06, wspace=0.30, hspace=0.58)
    save_figure(fig, out / model / f"figure_{'pr' if kind == 'pr' else 'roc'}_all36")


def render_all_class_pr_roc(tax_root: Path, out: Path) -> list[str]:
    _, _, _, pred_root = _layout_paths(tax_root)
    models = ["eegwavenet_sc", "physio_tf", "physio_stats", "eegnetv4", "eegconformer"]
    root = out / "06_classifier_pr_roc_all36"
    for model in models:
        path = pred_root / model / "all_window_test_predictions.parquet"
        if not path.exists():
            continue
        pred = pd.read_parquet(path)
        plot_pr_or_roc_grid(pred, model, root, "pr")
        plot_pr_or_roc_grid(pred, model, root, "roc")
    readme = (
        "# Five-model all-class PR/ROC audit\n\n"
        "Each model has two 6×6 figures: `figure_pr_all36` and `figure_roc_all36`. "
        "Every PG01–PG36 class is a separate one-vs-rest panel with its support and AP/AUROC. "
        "The pale curves in the earlier overview were the remaining classes, not atypical or excluded classes.\n"
    )
    (root / "README_zh.md").write_text(readme, encoding="utf-8")
    return models


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy-root", type=Path, default=TAX_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    tax_root = args.taxonomy_root.resolve(); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    events = load_events(tax_root)
    coords = render_per_mouse_pca(events, out)
    render_per_mouse_waveforms(events, out)
    centroids = render_global_centroid_pca(events, out)
    render_support_presence(events, out)
    models = render_all_class_pr_roc(tax_root, out)
    _, _, registry_path, _ = _layout_paths(tax_root)
    registry = pd.read_csv(registry_path)
    manifest = {
        "schema_version": 2,
        "renderer": "render_task2_pg36_corrected_v2.py",
        "formal_label_prefix": "PG",
        "formal_global_category_count": int(events.global_label.nunique()),
        "formal_labels": sorted(events.global_label.unique(), key=natural),
        "event_channel_points": int(len(events)),
        "mice": sorted(events.animal_id.astype(str).unique(), key=natural),
        "valid_channels": sorted(events.channel_name.astype(str).unique()),
        "channel_independence": True,
        "same_mouse_display": "CH7 and CH8 are shown together only as independent points/columns; no channel concatenation or paired observation",
        "per_mouse_category_semantics": "each mouse displays its observed subset of PG01-PG36; the union across mice is PG01-PG36",
        "global_centroid_semantics": "one feature centroid point per PG category (36 points total)",
        "global_centroid_fit_scope": "frozen training-event PG prototypes; held-out events contribute support annotations only",
        "local_pxx_and_lxx": "excluded from formal figures; retained only in old audit directories",
        "unobserved_pg_channel_combinations": int((events[["global_label", "channel_name"]].drop_duplicates().shape[0]) and (36 * 2 - events[["global_label", "channel_name"]].drop_duplicates().shape[0])),
        "five_models_all_class_pr_roc": models,
        "source_tables": {
            "event_features": str(_layout_paths(tax_root)[0]),
            "frozen_event_labels": str(_layout_paths(tax_root)[1]),
            "category_registry": str(registry_path),
        },
        "excluded_rule": "E77/CH7 absent from frozen protocol",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = (
        "# Task 2 formal PG36 corrected figures\n\n"
        "This is the canonical visual package for the pooled morphology taxonomy. "
        "All formal figures use PG01–PG36. One observation is one independent event×channel point. "
        "CH7 and CH8 can be displayed in the same mouse panel, but they are never concatenated or treated as a paired classifier input.\n\n"
        "## Figure semantics\n\n"
        "- `01_per_mouse_pg_pca`: each mouse's observed PG subset, with both valid channels as independent markers and every present PG label annotated.\n"
        "- `02_per_mouse_pg_waveforms`: one row per observed PG category in that mouse; CH7 and CH8 columns are independent evidence.\n"
        "- `03_global_pg_centroid_pca`: exactly 36 points, one robust feature centroid per PG category, plus nearest-centroid separation.\n"
        "- `04_pg_support_presence`: full PG01–PG36 × mouse/channel support matrix; zero means not observed in that stream.\n"
        "- `06_classifier_pr_roc_all36`: all five models, all 36 classes, one-vs-rest PR and ROC small multiples.\n\n"
        "The 36-class union is not expected to appear in every single mouse. Most PG categories are animal-specific; the support matrix makes that explicit. Pxx/Lxx are exploratory local labels and are not used in these formal figures.\n"
    )
    (out / "README_zh.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(out), "n_event_channel": len(events), "n_pg": int(events.global_label.nunique()), "models": models}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
