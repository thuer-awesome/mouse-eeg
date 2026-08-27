#!/usr/bin/env python
"""Render waveform/PSD pages for every within-mouse pooled Pxx cluster."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "output" / "epilepsy_program_20260821" / "02_morphology_taxonomy" / "all_mice_old_plus_new" / "strict_all_channel_v1"
FEATURES = ["log_robust_amplitude", "log_rms", "log_line_length", "hjorth_mobility", "hjorth_complexity", "peak_frequency_hz", "spectral_entropy", "relative_delta", "relative_theta", "relative_beta", "relative_low_gamma_clean", "relative_high_gamma"]
COLOUR = {"CH7": "#0072B2", "CH8": "#D55E00"}

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7, "axes.spines.right": False, "axes.spines.top": False, "svg.fonttype": "none", "pdf.fonttype": 42})


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def shape_normalize(signal: np.ndarray) -> np.ndarray:
    centered = np.asarray(signal, dtype=float) - np.nanmedian(signal)
    scale = np.nanpercentile(centered, 95) - np.nanpercentile(centered, 5)
    return centered / max(float(scale), 1e-8)


def representative(group: pd.DataFrame, features: list[str]) -> pd.Series:
    x = group[features].replace([np.inf, -np.inf], np.nan).astype(float)
    x = x.fillna(x.median()).fillna(0.0).to_numpy(float)
    centre = np.nanmedian(x, axis=0)
    distance = np.sqrt(np.nanmean((x - centre) ** 2, axis=1))
    return group.iloc[int(np.nanargmin(distance))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--figure-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    events = pd.read_parquet(root / "01_tables" / "channel_event_features.parquet")
    events = events.loc[events.channel_name.astype(str).isin(["CH7", "CH8"])].copy()
    events = events.loc[~(events.animal_id.astype(str).eq("E77") & events.channel_name.astype(str).eq("CH7"))].copy()
    pooled = pd.read_parquet(root / "03_cross_mouse" / "pooled_mouse_event_assignments.parquet")
    joined = events.merge(pooled[["animal_id", "session_id", "event_id", "channel_name", "pooled_label"]], on=["animal_id", "session_id", "event_id", "channel_name"], how="inner", validate="one_to_one")
    reps = []
    for (animal, label, channel), group in joined.groupby(["animal_id", "pooled_label", "channel_name"], sort=True):
        row = representative(group, [f for f in FEATURES if f in group.columns])
        reps.append(row)
    reps = pd.DataFrame(reps)
    out_root = (args.figure_dir.resolve() if args.figure_dir is not None else root / "06_figures" / "paper_ready_morphology_pooled") / "02_pooled_within_mouse" / "waveforms"
    out_root.mkdir(parents=True, exist_ok=True)
    labels = sorted(joined.pooled_label.astype(str).unique(), key=lambda v: ("".join(c for c in v if c.isdigit()), v))
    support = joined.groupby("pooled_label").size().to_dict()
    for page, start in enumerate(range(0, len(labels), 12), start=1):
        subset = labels[start:start + 12]
        fig, axes = plt.subplots(4, 3, figsize=(183 / 25.4, 116 / 25.4), constrained_layout=False)
        for ax, label in zip(axes.ravel(), subset):
            h = reps.loc[reps.pooled_label.astype(str).eq(label)]
            inset = ax.inset_axes([.57, .57, .37, .34]); inset.set_xlim(.5, 70); inset.set_xticks([]); inset.set_yticks([]); inset.set_title("PSD", fontsize=5, pad=1)
            for spine in inset.spines.values(): spine.set_color("#B8BEC4"); spine.set_linewidth(.45)
            for channel in ("CH7", "CH8"):
                row = h.loc[h.channel_name.astype(str).eq(channel)]
                if row.empty: continue
                row = row.iloc[0]
                try:
                    signal = np.load(Path(str(row.cache_dir)) / "signals.npy", mmap_mode="r")[int(row.cache_local_index), 0 if channel == "CH7" else 1]
                    signal = shape_normalize(signal); fs = float(row.sampling_rate_hz); time = np.arange(signal.size) / fs
                    ax.plot(time, signal, color=COLOUR[channel], lw=.65, label=channel)
                    freq, power = welch(signal, fs=fs, nperseg=min(512, signal.size)); mask = (freq >= .5) & (freq <= 70); inset.plot(freq[mask], power[mask] / max(float(np.nanmax(power[mask])), 1e-12), color=COLOUR[channel], lw=.55)
                except Exception:
                    continue
            ax.axhline(0, color="#B9BEC3", lw=.45); ax.set_xlim(0, 4); ax.set_ylim(-3.2, 3.2); ax.set_xticks([0, 2, 4]); ax.set_yticks([]); ax.set_title(f"{label} | n={int(support.get(label, 0)):,}", loc="left", fontsize=6.3, fontweight="bold"); ax.text(.01, .03, "CH7 / CH8", transform=ax.transAxes, fontsize=5, color="#5F6972")
        for ax in axes.ravel()[len(subset):]: ax.axis("off")
        for i, ax in enumerate(axes.ravel()):
            if i < len(subset) and i // 3 == 3:
                ax.set_xlabel("Time (s)")
            elif i < len(subset):
                ax.set_xticklabels([])
        fig.suptitle("Within-mouse pooled Pxx waveforms and display-only PSD", fontsize=9, fontweight="bold", x=.01, ha="left", y=.98)
        fig.text(.01, .01, "Each Pxx cluster was fitted jointly from all eligible CH7 + CH8 event points within one mouse; traces are shape-normalized after 0.5-70 Hz band-pass + 50 Hz notch and shown in a.u.", fontsize=5.8, color="#5F6972")
        fig.subplots_adjust(left=.035, right=.995, top=.88, bottom=.11, wspace=.12, hspace=.38)
        save(fig, out_root / f"figure_2_pooled_waveforms_p{page:02d}")
    reps.to_csv(out_root / "pooled_waveform_representatives.csv", index=False, encoding="utf-8-sig")
    (out_root / "README_zh.md").write_text(f"Pooled local waveform atlas: {len(labels)} Pxx clusters, {len(reps)} available representative stream-cluster traces. Each panel uses a representative event nearest the cluster feature median; PSD is display-only.\n", encoding="utf-8")
    print(f"wrote {len(labels)} pooled waveform clusters to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
