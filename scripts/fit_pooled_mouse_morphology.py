#!/usr/bin/env python
"""Fit a true within-mouse pooled morphology clustering.

This is deliberately separate from the frozen 45-category classifier labels.
For each mouse, every eligible CH7 and CH8 event is one point in the same
feature space, and one GMM is fitted to the training events from both channels.
The resulting Pxx labels are a morphology view, not clinical seizure names.
"""
from __future__ import annotations

import json
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "output"
    / "epilepsy_program_20260821"
    / "02_morphology_taxonomy"
    / "all_mice_old_plus_new"
    / "strict_all_channel_v1"
)
FEATURES = [
    "log_robust_amplitude", "log_rms", "log_line_length", "hjorth_mobility",
    "hjorth_complexity", "peak_frequency_hz", "spectral_entropy",
    "relative_delta", "relative_theta", "relative_beta",
    "relative_low_gamma_clean", "relative_high_gamma",
    "baseline_z__log_robust_amplitude", "baseline_z__log_rms",
    "baseline_z__log_line_length", "baseline_z__hjorth_mobility",
    "baseline_z__hjorth_complexity", "baseline_z__peak_frequency_hz",
    "baseline_z__spectral_entropy", "baseline_z__relative_delta",
    "baseline_z__relative_theta", "baseline_z__relative_beta",
    "baseline_z__relative_low_gamma_clean", "baseline_z__relative_high_gamma",
]


def finite(frame: pd.DataFrame) -> np.ndarray:
    x = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    return x.fillna(x.median()).fillna(0.0).to_numpy(float)


def choose_gmm(z: np.ndarray, seed: int) -> tuple[GaussianMixture, np.ndarray, pd.DataFrame]:
    n = len(z)
    max_k = min(8, max(1, n // 20))
    rows = []
    models: dict[int, tuple[GaussianMixture, np.ndarray]] = {}
    for k in range(1, max_k + 1):
        model = GaussianMixture(n_components=k, covariance_type="diag", n_init=4, max_iter=400, reg_covar=1e-4, random_state=seed + k)
        labels = model.fit_predict(z)
        models[k] = (model, labels)
        stability = 1.0
        if k > 1 and n >= 30:
            aris = []
            rng = np.random.default_rng(seed + 1000 * k)
            for rep in range(6):
                ix = rng.integers(0, n, n)
                boot = GaussianMixture(n_components=k, covariance_type="diag", n_init=2, max_iter=250, reg_covar=1e-4, random_state=seed + rep + 2000 * k).fit(z[ix])
                aris.append(adjusted_rand_score(labels, boot.predict(z)))
            if aris:
                stability = float(np.median(aris))
        counts = np.bincount(labels, minlength=k)
        rows.append({"components": k, "bic": float(model.bic(z)), "min_events": int(counts.min()), "stability_ari": stability})
    grid = pd.DataFrame(rows)
    bic1 = float(grid.loc[grid.components.eq(1), "bic"].iloc[0])
    grid["bic_gain_vs_k1"] = bic1 - grid.bic
    grid["pass"] = grid.components.eq(1) | (grid.min_events.ge(12) & grid.stability_ari.ge(0.55) & grid.bic_gain_vs_k1.ge(10))
    viable = grid.loc[grid.components.gt(1) & grid["pass"]]
    selected = int(viable.sort_values(["bic", "components"]).iloc[0].components) if not viable.empty else 1
    grid["selected"] = grid.components.eq(selected)
    return models[selected][0], models[selected][1], grid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    event_path = root / "01_tables" / "channel_event_features.parquet"
    if not event_path.is_file():
        # Accept the intermediate channel-independent layout as a convenience,
        # but always materialize the canonical 01_tables output below.
        event_path = root / "channel_independent" / "channel_event_features.parquet"
    event = pd.read_parquet(event_path)
    if "split" not in event.columns:
        raise ValueError(
            "channel_event_features.parquet must contain an event-level split; "
            "do not fit pooled clusters from a table with an implicit split"
        )
    event = event.loc[event.channel_name.astype(str).isin(["CH7", "CH8"])].copy()
    assignments = []
    prototypes = []
    summaries = []
    exclusions = []
    grids = root / "03_cross_mouse" / "pooled_mouse_gmm"
    grids.mkdir(parents=True, exist_ok=True)
    for idx, (animal, group0) in enumerate(event.groupby("animal_id", sort=True)):
        group = group0.reset_index(drop=True).copy()
        # The taxonomy is frozen from the training partition only.  The old
        # fallback to ``group`` when a mouse had fewer than twelve training
        # events leaked validation/test morphology into the fitted scaler and
        # GMM.  A small training set is still valid (choose_gmm will select
        # k=1); a mouse with no training events is explicitly excluded and
        # causes the stage to fail rather than silently leaking held-out data.
        train = group.loc[group.split.astype(str).eq("train")].copy()
        if train.empty:
            exclusions.append(
                {
                    "animal_id": str(animal),
                    "n_events": int(len(group)),
                    "n_train_events": 0,
                    "reason": "no_train_events",
                }
            )
            summaries.append(
                {
                    "animal_id": str(animal),
                    "n_events": int(len(group)),
                    "n_train_events": 0,
                    "n_pooled_clusters": 0,
                    "selected_bic": np.nan,
                    "selected_stability_ari": np.nan,
                    "channels_in_fit": "",
                    "status": "excluded_no_train_events",
                }
            )
            continue
        scaler = RobustScaler(quantile_range=(10, 90)).fit(finite(train))
        z_train = scaler.transform(finite(train))
        model, train_raw, grid = choose_gmm(z_train, int(args.seed) + idx * 101)
        order = np.argsort(model.means_[:, 0], kind="stable")
        remap = {int(old): int(new) for new, old in enumerate(order)}
        train_labels = np.asarray([remap[int(v)] for v in train_raw], dtype=int)
        z_all = scaler.transform(finite(group))
        raw = model.predict(z_all)
        labels = np.asarray([remap[int(v)] for v in raw], dtype=int)
        probs = model.predict_proba(z_all)[:, order].max(axis=1)
        group["pooled_cluster"] = labels
        group["pooled_label"] = [f"{animal}-P{int(v) + 1:02d}" for v in labels]
        group["pooled_probability"] = probs
        assignments.append(group[["animal_id", "session_id", "event_id", "channel_name", "split", "pooled_cluster", "pooled_label", "pooled_probability"]])
        train["_cluster"] = train_labels
        for cluster in sorted(np.unique(train_labels)):
            subset = train.loc[train._cluster.eq(cluster)]
            values = np.nanmedian(finite(subset), axis=0)
            prototypes.append({"animal_id": animal, "pooled_cluster": int(cluster), "pooled_label": f"{animal}-P{int(cluster) + 1:02d}", "n_train_events": int(len(subset)), **{f"f_{name}": float(value) for name, value in zip(FEATURES, values)}})
        grid.to_csv(grids / f"pooled_gmm_grid_{animal}.csv", index=False)
        selected = grid.loc[grid.selected].iloc[0]
        summaries.append({"animal_id": animal, "n_events": int(len(group)), "n_train_events": int(len(train)), "n_pooled_clusters": int(selected.components), "selected_bic": float(selected.bic), "selected_stability_ari": float(selected.stability_ari), "channels_in_fit": "+".join(sorted(group.channel_name.astype(str).unique())), "status": "included"})
    exclusions_frame = pd.DataFrame(exclusions, columns=["animal_id", "n_events", "n_train_events", "reason"])
    exclusions_frame.to_csv(root / "03_cross_mouse" / "pooled_mouse_exclusions.csv", index=False, encoding="utf-8-sig")
    if exclusions:
        # Keep the explicit exclusion artifact for audit/recovery, then stop
        # before producing a taxonomy that would have silently dropped rows.
        raise ValueError(
            "pooled clustering requires at least one training event per mouse; "
            f"excluded={','.join(row['animal_id'] for row in exclusions)} "
            "(reason=no_train_events). "
            "See 03_cross_mouse/pooled_mouse_exclusions.csv"
        )
    result = pd.concat(assignments, ignore_index=True)
    result.to_parquet(root / "03_cross_mouse" / "pooled_mouse_event_assignments.parquet", index=False)
    pd.DataFrame(prototypes).to_parquet(root / "03_cross_mouse" / "pooled_mouse_prototypes.parquet", index=False)
    pd.DataFrame(summaries).to_csv(root / "03_cross_mouse" / "pooled_mouse_cluster_summary.csv", index=False, encoding="utf-8-sig")
    manifest = {"unit": "one eligible event x channel point; CH7 and CH8 fitted jointly within each mouse", "fit_split": "training events only", "label_prefix": "Pxx", "not_clinical": True, "n_event_channel": int(len(result)), "animals": sorted(result.animal_id.astype(str).unique()), "excluded_mice": exclusions}
    (root / "03_cross_mouse" / "pooled_mouse_clustering_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"pooled mouse clustering written to {root / '03_cross_mouse'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
