# Pooled global morphology taxonomy (PG labels)

This directory is an independent label-freeze product derived from the
within-mouse pooled CH7/CH8 morphology clusters. It does **not** overwrite the
strict all-channel `04_label_freeze` directory or its G01-G45 taxonomy.

## Frozen result

- `9,144` event-by-channel rows and `194,530` source windows.
- `40` pooled mouse prototypes from `7` mice.
- `36` frozen categories labelled `PG01`-`PG36`.
- `3` categories are observed in at least two mice;
  `33` are animal-specific.
- The selected robust prototype distance threshold is `0.90`.

## Merge constraint

Pooled clusters are the atomic units. Candidate merges are allowed only between
different mice and only when the maximum (complete-linkage) distance between
all member prototypes is below the threshold. Two components are merged only
when their sets of mice are disjoint. Therefore a single PG category can
contain at most one pooled cluster from any mouse; transitive chains cannot
merge two pooled states from the same mouse.

Event labels inherit the category of their pooled mouse cluster. They are not
reassigned to a nearest global centroid, so this constraint remains true for
all splits. `pooled_global_distance` and `pooled_low_confidence` are QC fields;
they do not silently remove events or windows.

## Files

- `pooled_event_labels.parquet`: one row per event x valid channel, with pooled
  cluster and PG label plus split and QC fields.
- `pooled_channel_window_labels.parquet`: every source window with PG labels;
  previous strict G labels are retained as `source_strict_*` columns.
- `pooled_category_registry.csv/.parquet`: support, split counts, mouse/channel
  presence and merge status for each PG category.
- `pooled_global_prototypes.parquet`: robust-space weighted centroid per PG.
- `pooled_prototype_members.parquet`: exact pooled mouse prototype membership.
- `pooled_global_category_grid.csv/.parquet`: threshold sensitivity grid.
- `pooled_label_freeze_manifest.json`: reproducibility and invariant audit.

PG labels are exploratory morphology states. They must not be described as
clinical seizure subtypes without independent clinical review.
