#!/usr/bin/env python
"""Render the pooled taxonomy split and leakage audit figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "output" / "epilepsy_program_20260821" / "02_morphology_taxonomy" / "all_mice_old_plus_new" / "strict_all_channel_v1"
COLOURS = {"train": "#0072B2", "val": "#D55E00", "test": "#009E73"}

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7, "axes.spines.right": False, "axes.spines.top": False, "svg.fonttype": "none", "pdf.fonttype": 42})


def _layout_paths(root: Path) -> tuple[Path, Path]:
    """Resolve source-study and compact-release taxonomy layouts."""

    if (root / "taxonomy").is_dir():
        return (
            root / "taxonomy" / "pooled_channel_window_labels.parquet",
            root / "taxonomy" / "pooled_event_labels.parquet",
        )
    return (
        root / "04_label_freeze_pooled" / "pooled_channel_window_labels.parquet",
        root / "04_label_freeze_pooled" / "pooled_event_labels.parquet",
    )


def _cross_split_window_overlaps(labels: pd.DataFrame) -> int:
    """Count adjacent cross-split overlaps without assuming a zero audit."""

    required = {"split", "start_time_ms", "end_time_ms", "animal_id", "channel_name"}
    if not required.issubset(labels.columns):
        return -1
    keys = ["animal_id", "channel_name"]
    if "session_id" in labels.columns:
        keys.insert(1, "session_id")
    count = 0
    for _, group in labels.groupby(keys, sort=False):
        group = group.sort_values(["start_time_ms", "end_time_ms"], kind="stable")
        previous_end = None
        previous_split = None
        for row in group.itertuples(index=False):
            start = float(row.start_time_ms)
            end = float(row.end_time_ms)
            split = str(row.split)
            if previous_end is not None and start < previous_end and split != previous_split:
                count += 1
            previous_end = max(float(previous_end or end), end)
            previous_split = split
    return int(count)


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--figure-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    labels_path, events_path = _layout_paths(root)
    labels = pd.read_parquet(labels_path)
    events = pd.read_parquet(events_path)
    fig_root = (args.figure_dir.resolve() if args.figure_dir is not None else root / "06_figures" / "paper_ready_classifier_pooled") / "figure_0_data_split_audit"
    fig_root.mkdir(parents=True, exist_ok=True)
    order = ["train", "val", "test"]
    animals = sorted(labels.animal_id.astype(str).unique())
    window_counts = labels.groupby(["animal_id", "split"], sort=True).size().unstack(fill_value=0).reindex(index=animals, columns=order, fill_value=0)
    event_counts = events.groupby(["animal_id", "channel_name", "split"], sort=True).size().reset_index(name="n")
    event_counts.to_csv(fig_root / "event_channel_split_counts.csv", index=False, encoding="utf-8-sig")
    window_counts.to_csv(fig_root / "window_split_counts_by_animal.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 72 / 25.4), gridspec_kw={"width_ratios": [1.05, 1.15, .95]}, constrained_layout=True)
    x = np.arange(len(animals)); bottom = np.zeros(len(animals))
    for split in order:
        vals = window_counts[split].to_numpy(float)
        axes[0].bar(x, vals, bottom=bottom, color=COLOURS[split], width=.65, label=split)
        bottom += vals
    axes[0].set_xticks(x, animals); axes[0].set_ylabel("Complete windows"); axes[0].set_title("Window allocation", loc="left", fontsize=7.5, fontweight="bold"); axes[0].grid(axis="y", color="#D7DADD", linewidth=.45); axes[0].set_axisbelow(True)
    streams = sorted((events.animal_id.astype(str) + "/" + events.channel_name.astype(str)).unique())
    emat = event_counts.assign(stream=event_counts.animal_id.astype(str) + "/" + event_counts.channel_name.astype(str)).pivot_table(index="stream", columns="split", values="n", fill_value=0).reindex(index=streams, columns=order, fill_value=0)
    im = axes[1].imshow(emat.to_numpy(float), aspect="auto", cmap="Blues")
    axes[1].set_yticks(np.arange(len(streams)), streams, fontsize=5.8); axes[1].set_xticks(np.arange(3), order); axes[1].set_xlabel("Split"); axes[1].set_title("Event×channel allocation", loc="left", fontsize=7.5, fontweight="bold"); fig.colorbar(im, ax=axes[1], fraction=.045, pad=.03, label="Events")
    ratios = window_counts.div(window_counts.sum(axis=1), axis=0)
    for split in order:
        axes[2].plot(np.arange(len(animals)), ratios[split], "o-", color=COLOURS[split], lw=1.1, ms=3, label=split)
    axes[2].axhline(1/len(order), color="#999999", ls="--", lw=.7); axes[2].set_ylim(0, 1); axes[2].set_xticks(x, animals); axes[2].set_ylabel("Within-animal fraction"); axes[2].set_title("Split ratio by animal", loc="left", fontsize=7.5, fontweight="bold"); axes[2].grid(axis="y", color="#D7DADD", linewidth=.45); axes[2].legend(frameon=False, fontsize=6)
    axes[0].text(-.15, 1.05, "a", transform=axes[0].transAxes, fontsize=9, fontweight="bold"); axes[1].text(-.15, 1.05, "b", transform=axes[1].transAxes, fontsize=9, fontweight="bold"); axes[2].text(-.15, 1.05, "c", transform=axes[2].transAxes, fontsize=9, fontweight="bold")
    fig.suptitle("Event-level mixed split and leakage audit", fontsize=9, fontweight="bold", x=.01, ha="left")
    event_key_columns = [column for column in ("animal_id", "session_id", "event_id", "channel_name") if column in events.columns]
    mixed_event_keys = int(events.groupby(event_key_columns, sort=False)["split"].nunique().gt(1).sum()) if event_key_columns else -1
    cross_window_overlaps = _cross_split_window_overlaps(labels)
    overlap_text = "unavailable" if cross_window_overlaps < 0 else str(cross_window_overlaps)
    fig.text(.01, -.02, f"Splits are assigned at clinical-event level within each animal; event keys in multiple splits={mixed_event_keys}; adjacent cross-split window overlaps={overlap_text}.", fontsize=5.8, color="#5F6972")
    fig.legend(handles=[Patch(facecolor=COLOURS[s], label=s) for s in order], frameon=False, loc="upper right", bbox_to_anchor=(.99, 1.04), ncol=3, fontsize=6)
    save(fig, fig_root / "figure_0_event_level_split_audit")
    manifest = {"figure": "figure_0_event_level_split_audit", "event_rows": int(len(events)), "window_rows": int(len(labels)), "split_counts_events": events.split.value_counts().to_dict(), "split_counts_windows": labels.split.value_counts().to_dict(), "cross_split_mixed_event_keys": mixed_event_keys, "cross_split_overlapping_windows": cross_window_overlaps, "context_scope": "same event and same channel", "source_labels": str(labels_path), "source_events": str(events_path), "figure_dir": str(fig_root)}
    (fig_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
