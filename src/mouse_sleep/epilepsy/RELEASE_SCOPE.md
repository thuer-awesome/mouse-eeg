# Release scope

The `mouse_sleep.epilepsy` modules are the audited implementation copied from
the source study.  Public release wrappers live in `epilepsy_eeg`:

- `raw_adapter.py` creates the label-independent 250-Hz cache;
- `task1.py` runs channel-independent binary inference and the physical-window
  decoder/PDF renderer;
- `task2.py` validates and materializes the frozen PG36 artifact lineage.

Do not import historical sleep-staging labels or the old 45-class lineage into
the two formal workflows.  Every change to filtering, split construction,
decoder thresholds, feature definitions or model topology must produce a new
manifest and a new result directory.
