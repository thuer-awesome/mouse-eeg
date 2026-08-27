# Task 2 pooled morphology classification

## Data and label contract

- Seven mice: E07, E50, E76, E77, E79, E81, E82; valid channels are CH7 and CH8. E77/CH7 is excluded.
- Each valid channel remains an independent classifier sample. CH7 and CH8 are never concatenated at model input.
- Within each mouse, all eligible CH7+CH8 event points are fitted jointly (`Pxx` pooled clusters): 40 pooled prototypes.
- Cross-mouse PG freeze: 36 exploratory categories at robust distance threshold 0.90; 3 shared across at least two mice and 33 animal-specific; same-mouse merge violations = 0.
- PG labels are morphology categories for exploratory classification, not clinical seizure subtypes.
- The formal waveform atlas is rendered as a complete `PG x channel` grid: 36 global PG categories x 2 valid channel identities = 72 possible combinations; 61 have observed event support and 11 are explicitly marked `not observed`. Each observed panel is one independent channel stream.
- Model input note: Physio-TF and Physio+statistics consume the full 21-epoch sequence; EEGWaveNet-SC, EEGNetv4 and EEGConformer remain single-window paper baselines. The shared loader enforces the same event-contained context contract but does not silently change baseline architectures.
- Preprocessing: 250 Hz, 0.5-70 Hz zero-phase band-pass and 50 Hz notch. Amplitude is shape-normalized a.u.; ADC-to-uV calibration is unconfirmed.

## Split and denominators

| Unit | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Event x channel labels | 6,393 | 1,375 | 1,376 | 9,144 |
| Complete windows | 130,216 | 31,731 | 32,583 | 194,530 |

Training used an event-balanced sensitivity cap of at most eight windows per event (5,331 capped test windows). Formal performance below uses every eligible held-out test window (`n=32,583`).

## Full-window model results

| Model | Accuracy | Balanced accuracy | Macro-F1 | Macro-AUPRC | Event Macro-F1 |
|---|---:|---:|---:|---:|---:|
| physio_stats | 0.605 | 0.593 | 0.465 | 0.513 | 0.625 |
| physio_tf | 0.522 | 0.582 | 0.434 | 0.497 | 0.647 |
| eegconformer | 0.266 | 0.365 | 0.207 | 0.279 | 0.409 |
| eegnetv4 | 0.229 | 0.328 | 0.188 | 0.246 | 0.369 |
| eegwavenet_sc | 0.219 | 0.260 | 0.154 | 0.221 | 0.247 |

The best full-window Macro-F1 in this run is **physio_stats (0.465)**. The best event-aggregated Macro-F1 is **physio_tf (0.647)**. Macro-AUROC is retained in the CSV but is not used alone as a clinical conclusion.

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
