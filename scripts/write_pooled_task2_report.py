#!/usr/bin/env python
"""Write one concise index/report for the pooled morphology classification run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "output" / "epilepsy_program_20260821" / "02_morphology_taxonomy" / "all_mice_old_plus_new" / "strict_all_channel_v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    pooled = root / "04_label_freeze_pooled"
    cls = root / "05_classification_pooled"
    fig = root / "06_figures"
    manifest = json.loads((pooled / "pooled_label_freeze_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((pooled / "pooled_label_freeze_audit.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(cls / "all_window_model_comparison.csv").sort_values("macro_f1", ascending=False)
    model_lines = [
        f"| {row.model} | {row.accuracy:.3f} | {row.balanced_accuracy:.3f} | {row.macro_f1:.3f} | {row.macro_auprc:.3f} | {row.event_macro_f1:.3f} |"
        for _, row in metrics.iterrows()
    ]
    event_best = metrics.sort_values("event_macro_f1", ascending=False).iloc[0]
    report = f"""# Task 2 pooled morphology classification

## Data and label contract

- Seven mice: E07, E50, E76, E77, E79, E81, E82; valid channels are CH7 and CH8. E77/CH7 is excluded.
- Each valid channel remains an independent classifier sample. CH7 and CH8 are never concatenated at model input.
- Within each mouse, all eligible CH7+CH8 event points are fitted jointly (`Pxx` pooled clusters): {manifest['n_pooled_prototypes']} pooled prototypes.
- Cross-mouse PG freeze: {manifest['n_global_categories']} exploratory categories at robust distance threshold {manifest['selected_distance_threshold']:.2f}; {manifest['n_shared_categories']} shared across at least two mice and {manifest['n_animal_specific_categories']} animal-specific; same-mouse merge violations = {manifest['same_mouse_merge_violations']}.
- PG labels are morphology categories for exploratory classification, not clinical seizure subtypes.
- Model input note: Physio-TF and Physio+statistics consume the full 21-epoch sequence; EEGWaveNet-SC, EEGNetv4 and EEGConformer remain single-window paper baselines. The shared loader enforces the same event-contained context contract but does not silently change baseline architectures.
- Preprocessing: 250 Hz, 0.5-70 Hz zero-phase band-pass and 50 Hz notch. Amplitude is shape-normalized a.u.; ADC-to-uV calibration is unconfirmed.

## Split and denominators

| Unit | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Event x channel labels | {manifest['train_event_count']:,} | {manifest['val_event_count']:,} | {manifest['test_event_count']:,} | {manifest['n_event_channel']:,} |
| Complete windows | {audit['split_counts_windows_output']['train']:,} | {audit['split_counts_windows_output']['val']:,} | {audit['split_counts_windows_output']['test']:,} | {manifest['n_windows']:,} |

Training used an event-balanced sensitivity cap of at most eight windows per event (5,331 capped test windows). Formal performance below uses every eligible held-out test window (`n=32,583`).

## Full-window model results

| Model | Accuracy | Balanced accuracy | Macro-F1 | Macro-AUPRC | Event Macro-F1 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(model_lines)}

The best full-window Macro-F1 in this run is **{metrics.iloc[0].model} ({metrics.iloc[0].macro_f1:.3f})**. The best event-aggregated Macro-F1 is **{event_best.model} ({event_best.event_macro_f1:.3f})**. Macro-AUROC is retained in the CSV but is not used alone as a clinical conclusion.

## Canonical outputs

- Taxonomy and label audit: [`04_label_freeze_pooled`](04_label_freeze_pooled)
- Classifier checkpoints, full-window predictions and per-class tables: [`05_classification_pooled`](05_classification_pooled)
- Main morphology figures: [`06_figures/paper_ready_morphology_pooled`](06_figures/paper_ready_morphology_pooled)
- Statistical cluster diagnostics (3D PCA, BIC/stability, PG support): [`06_figures/paper_ready_morphology_pooled/06_cluster_diagnostics`](06_figures/paper_ready_morphology_pooled/06_cluster_diagnostics)
- Main paper-style classifier figures: [`06_figures/paper_ready_classifier_pooled`](06_figures/paper_ready_classifier_pooled)
- Split/leakage audit figure: [`06_figures/paper_ready_classifier_pooled/figure_0_data_split_audit`](06_figures/paper_ready_classifier_pooled/figure_0_data_split_audit)
- Complete PR/ROC, support-recall and per-model confusion figures: [`06_figures/strict_classifier_full_pooled`](06_figures/strict_classifier_full_pooled)
- Legacy 21-state and previous G taxonomy are retained under `historical_21_read_only`, `legacy_style_morphology`, `legacy_style_cross_mouse`, and `04_label_freeze`; they are priors/audit references, not the PG final classifier.

Task 1 binary seizure onset/offset decoding remains a separate model line. Its legacy figures are not overwritten by this task-2 run.

Per-stream Macro-F1 is intentionally strict: it averages over the complete PG01-PG36 label set for each animal/channel stream, assigning zero to categories absent from that stream. It is therefore a domain-robustness audit, not a replacement for the pooled overall test metric.
"""
    reports = root / "07_reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "TASK2_POOLED_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    index = f"""# Task 2 canonical index

Start with [`07_reports/TASK2_POOLED_FINAL_REPORT.md`](../07_reports/TASK2_POOLED_FINAL_REPORT.md). The formal taxonomy is PG01-PG{manifest['n_global_categories']:02d}; all model metrics use 32,583 held-out windows.

1. Morphology process: [`paper_ready_morphology_pooled`](paper_ready_morphology_pooled)
2. Cluster statistics: [`paper_ready_morphology_pooled/06_cluster_diagnostics`](paper_ready_morphology_pooled/06_cluster_diagnostics)
3. Main classifier figures: [`paper_ready_classifier_pooled`](paper_ready_classifier_pooled)
4. Complete deep-learning figures: [`strict_classifier_full_pooled`](strict_classifier_full_pooled)
5. Tables/checkpoints: [`05_classification_pooled`](../05_classification_pooled)
    """
    (fig / "README_TASK2_INDEX_zh.md").write_text(index, encoding="utf-8")
    print(reports / "TASK2_POOLED_FINAL_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
