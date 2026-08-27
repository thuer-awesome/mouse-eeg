# Pooled taxonomy classifier figures

- Taxonomy: PG01-PG36; each valid channel is an independent sample.
- Figure 0: event-level mixed split, independent channel-stream allocation, and leakage audit (cross-split overlapping event pairs = 0).
- Figure 1: train/validation NLL curves with selected checkpoints.
- Figure 2: train/validation Macro-F1 curves.
- Figure 3: full held-out-window balanced accuracy, Macro-F1 and Macro-AUPRC.
- Figure 4: Physio+statistics row-normalized confusion and support/recall.
- Figure 5: animal/channel stream robustness.
- Supplement: complete 36x36 confusion matrices for all five models.

The formal denominator is the complete test-window table, not the capped event-balanced training sensitivity sample. PG labels are exploratory morphology categories, not clinical seizure subtypes.
