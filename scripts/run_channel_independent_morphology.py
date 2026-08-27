#!/usr/bin/env python
"""Build an event-balanced, channel-independent morphology audit.

The paired legacy tables remain the primary historical renderer input.  This
companion explicitly unfolds CH7/CH8 into independent event observations so a
valid channel is never represented only through a bilateral aggregate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "output" / "epilepsy_program_20260821" / "02_morphology_taxonomy" / "all_mice_old_plus_new"


FEATURE_SUFFIXES = ("log_robust_amplitude", "log_rms", "log_line_length", "hjorth_mobility", "hjorth_complexity", "peak_frequency_hz", "spectral_entropy", "relative_delta", "relative_theta", "relative_beta", "relative_low_gamma_clean", "relative_high_gamma")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--root", type=Path, default=DEFAULT_ROOT); p.add_argument("--seed", type=int, default=20260822); p.add_argument("--bootstrap-repeats", type=int, default=12); args = p.parse_args(argv)
    root = args.root.resolve(); tables = root / "tables"; out = root / "channel_independent"; out.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(tables / "combined_morphology_features_relative.parquet")
    rows = []
    for channel in ("ch7", "ch8"):
        valid = source[f"{channel}_valid"].astype(bool)
        if channel == "ch7": valid &= ~source["animal_id"].astype(str).eq("E77")
        cols = [f"{channel}_{s}" for s in FEATURE_SUFFIXES]
        baseline_cols = [f"baseline_z__{channel}_{s}" for s in FEATURE_SUFFIXES if f"baseline_z__{channel}_{s}" in source]
        use = cols + baseline_cols
        metadata_cols = ["animal_id", "session_id", "event_id", "event_duration_s"]
        if "event_position_fraction" in source:
            metadata_cols.append("event_position_fraction")
        frame = source.loc[valid, [*metadata_cols, *use]].copy()
        frame["channel_name"] = channel.upper()
        frame["source_positive_channels"] = source.loc[valid, "source_positive_channels"].astype(str).to_numpy()
        frame = frame.rename(columns={c: c.replace(f"{channel}_", "") for c in cols})
        frame = frame.rename(columns={c: c.replace(f"baseline_z__{channel}_", "baseline_z__") for c in baseline_cols})
        rows.append(frame)
    unfolded = pd.concat(rows, ignore_index=True, sort=False)
    feature_cols = [s for s in FEATURE_SUFFIXES if s in unfolded] + [f"baseline_z__{s}" for s in FEATURE_SUFFIXES if f"baseline_z__{s}" in unfolded]
    # One median vector per animal x event x channel prevents long events from
    # dominating the channel-level morphology clustering.
    keys = ["animal_id", "session_id", "event_id", "channel_name"]
    event_features = unfolded.groupby(keys, sort=True)[feature_cols].median().reset_index()
    metadata_fields = ["event_duration_s"] + (["event_position_fraction"] if "event_position_fraction" in unfolded else [])
    metadata = unfolded.groupby(keys, sort=True)[metadata_fields].median().reset_index()
    event_features = event_features.merge(metadata, on=keys, validate="one_to_one")
    event_features.to_parquet(out / "channel_event_features.parquet", index=False)
    assignments = []; model_rows = []; rng = np.random.default_rng(int(args.seed))
    for animal, g in event_features.groupby("animal_id", sort=True):
        g = g.reset_index(drop=True); matrix = g[feature_cols].replace([np.inf, -np.inf], np.nan); matrix = matrix.fillna(matrix.median()).fillna(0.0)
        scaler = RobustScaler(quantile_range=(10, 90)).fit(matrix); z = scaler.transform(matrix)
        maximum = min(8, max(1, len(g) // 30)); candidates = []
        models = {}
        for k in range(1, maximum + 1):
            model = GaussianMixture(n_components=k, covariance_type="diag", n_init=3, max_iter=300, reg_covar=1e-4, random_state=int(args.seed + 101*k + len(g))).fit(z); labels = model.predict(z); models[k] = (model, labels)
            if k == 1: med = q25 = 1.0
            else:
                scores = []
                for rep in range(int(args.bootstrap_repeats)):
                    sample = rng.integers(0, len(z), len(z))
                    try: pred = GaussianMixture(n_components=k, covariance_type="diag", n_init=2, max_iter=200, reg_covar=1e-4, random_state=int(args.seed + rep + k*1000)).fit(z[sample]).predict(z)
                    except ValueError: continue
                    scores.append(adjusted_rand_score(labels, pred))
                med = float(np.median(scores)) if scores else np.nan; q25 = float(np.quantile(scores, .25)) if scores else np.nan
            candidates.append({"animal_id": animal, "components": k, "bic": float(model.bic(z)), "min_events": int(np.bincount(labels, minlength=k).min()), "bootstrap_median_ari": med, "bootstrap_q25_ari": q25})
        grid = pd.DataFrame(candidates); one = float(grid.loc[grid.components.eq(1), "bic"].iloc[0]); grid["bic_improvement_vs_one"] = one - grid.bic; grid["stability_pass"] = grid.components.eq(1) | (grid.min_events.ge(8) & grid.bootstrap_median_ari.ge(.60) & grid.bootstrap_q25_ari.ge(.40) & grid.bic_improvement_vs_one.ge(10))
        viable = grid.loc[grid.stability_pass & grid.components.gt(1)]; selected = int(viable.sort_values(["bic", "components"]).iloc[0].components) if not viable.empty else 1; grid["selected"] = grid.components.eq(selected); model, labels = models[selected]
        order = np.argsort(model.means_[:, 0], kind="stable"); remap = {int(old): int(new) for new, old in enumerate(order)}; labels = np.asarray([remap[int(v)] for v in labels], dtype=int); probs = model.predict_proba(z)[:, order].max(axis=1)
        a = g[keys].copy(); a["channel_cluster"] = labels; a["channel_label"] = [f"{animal}-{str(ch)}-T{int(v)+1}" for ch, v in zip(a.channel_name, labels)]; a["assignment_probability"] = probs; assignments.append(a); grid.to_csv(out / f"channel_gmm_grid_{animal}.csv", index=False); model_rows.append({"animal_id": animal, "events": len(g), "selected_components": selected, "bootstrap_median_ari": float(grid.loc[grid.selected, "bootstrap_median_ari"].iloc[0]), "bootstrap_q25_ari": float(grid.loc[grid.selected, "bootstrap_q25_ari"].iloc[0])})
    assigned = pd.concat(assignments, ignore_index=True); assigned.to_parquet(out / "channel_event_assignments.parquet", index=False); pd.DataFrame(model_rows).to_csv(out / "channel_model_selection.csv", index=False, encoding="utf-8-sig")
    # Mouse-balanced PCA for the unfolded channel event vectors.
    mat = event_features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(event_features[feature_cols].median()).fillna(0.0); z = RobustScaler(quantile_range=(10, 90)).fit_transform(mat); pca = PCA(n_components=min(8, z.shape[1], len(z)-1), random_state=int(args.seed)); coords = pca.fit_transform(z); coord = event_features[keys].copy(); coord[[f"PC{i+1}" for i in range(coords.shape[1])]] = coords; coord.to_parquet(out / "channel_event_pca.parquet", index=False); (out / "channel_pca_summary.json").write_text(json.dumps({"rows": len(coord), "features": feature_cols, "explained_variance_fraction": pca.explained_variance_ratio_.tolist(), "mouse_unit": "channel-independent event", "e77_ch7_excluded": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    mpl.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "axes.spines.right": False, "axes.spines.top": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(183/25.4, 72/25.4), constrained_layout=True); colors = {"CH7": "#D55E00", "CH8": "#0072B2"}
    for ch in ("CH7", "CH8"):
        g = coord.loc[coord.channel_name.eq(ch)]; ax.scatter(g.PC1, g.PC2, s=8, alpha=.45, color=colors[ch], label=ch, linewidths=0, rasterized=True)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Channel-independent event morphology", loc="left", fontweight="bold"); ax.legend(frameon=False, title="channel")
    figdir = out / "figures"; figdir.mkdir(exist_ok=True)
    for ext, dpi in (("svg", 300), ("pdf", 300), ("tiff", 600), ("png", 300)): fig.savefig(figdir / f"channel_event_pca.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig); (out / "CHANNEL_MORPHOLOGY_REPORT_zh.md").write_text(f"# Channel-independent morphology\n\n- Event x channel observations: {len(event_features):,}\n- CH7 observations: {int(event_features.channel_name.eq('CH7').sum()):,}; CH8: {int(event_features.channel_name.eq('CH8').sum()):,}.\n- E77/CH7 excluded.\n- One median vector per event x channel was used for clustering and PCA; labels remain exploratory.\n", encoding="utf-8")
    print(f"[channel-morphology] complete rows={len(event_features):,} output={out}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
