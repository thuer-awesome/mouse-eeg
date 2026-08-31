# Task 2 pooled morphology classification

## Data and label contract

- Seven mice: E07, E50, E76, E77, E79, E81, E82; valid channels are CH7 and CH8. E77/CH7 is excluded.
- Each valid channel remains an independent classifier sample. CH7 and CH8 are never concatenated at model input.
- Within each mouse, all eligible CH7+CH8 event points are fitted jointly (`Pxx` pooled clusters): 40 pooled prototypes.
- Cross-mouse PG freeze: 36 exploratory categories at robust distance threshold 0.90; 3 shared across at least two mice and 33 animal-specific; same-mouse merge violations = 0.
- PG labels are morphology categories for exploratory classification, not clinical seizure subtypes.
- Model input note: Physio-TF and Physio+statistics consume the full 21-epoch sequence; EEGWaveNet-SC, EEGNetv4 and EEGConformer remain single-window paper baselines. The shared loader enforces the same event-contained context contract but does not silently change baseline architectures.
- Preprocessing: 250 Hz, 0.5-70 Hz zero-phase band-pass and 50 Hz notch. Amplitude is shape-normalized a.u.; ADC-to-uV calibration is unconfirmed.

## Split and denominators

| Unit | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Event x channel labels | 6,393 | 1,375 | 1,376 | 9,144 |
| Complete windows | 130,216 | 31,731 | 32,583 | 194,530 |

Training used an event-balanced sensitivity cap of at most eight windows per event.
The capped sensitivity table contains 5,331 test windows.
Reported full-window performance below uses every eligible held-out test window (`n=32,583`), with the denominator read from the all-window comparison table.

### Alignment caveat

This PG36 freeze uses its own event split (6,393/1,375/1,376 train/validation/test event×channel labels), not the Task-1 strict protocol split (4,495/960/957). The key audit found only 111 of 1,376 PG36 test keys with an exact `event_id + channel + split` match to the strict protocol; 642 keys were present with a split mismatch and 591 were absent. These classifier results are therefore a morphology-exploration snapshot, not a strict Task-1 generalization estimate, and must not be combined with Task-1 metrics.

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

- Taxonomy and label audit: [`taxonomy`](../taxonomy)
- Classifier checkpoints, full-window predictions and per-class tables: [`classification`](../classification)
- Main morphology figures: [`figures`](../figures)
- Global centroid and support diagnostics: [`figures/03_global_pg_centroid_pca`](../figures/03_global_pg_centroid_pca) and [`figures/04_pg_support_presence`](../figures/04_pg_support_presence)
- Main paper-style classifier figures: [`figures/model_summary`](../figures/model_summary)
- Split/leakage audit figure: [`figures/model_summary/figure_0_data_split_audit`](../figures/model_summary/figure_0_data_split_audit)
- Complete PR/ROC, support-recall and per-model confusion figures: [`figures/confusion_all_windows`](../figures/confusion_all_windows)
- Historical 21-state and previous G taxonomy artifacts are intentionally excluded from this PG36 release and remain available only in the source study archive.

Task 1 binary seizure onset/offset decoding remains a separate model line. Its legacy figures are not overwritten by this task-2 run.

Per-stream Macro-F1 is intentionally strict: it averages over the complete PG01-PG36 label set for each animal/channel stream, assigning zero to categories absent from that stream. It is therefore a domain-robustness audit, not a replacement for the pooled overall test metric.
