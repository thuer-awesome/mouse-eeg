# Changelog

## 0.1.1 - 2026-08-28

- Added source-first `run-task1` and `run-pg36` orchestration from raw EEG/labels through preprocessing, training, inference, evaluation and figure rendering.
- Added reproducibility manifests, stage logs, configurable full-run overrides and regression tests; frozen weights remain optional release snapshots.

## 0.1.0 - 2026-08-27

- Added the audited Task 1 binary continuous annotation release for seven mice and twelve valid channel streams.
- Added the audited Task 2 pooled PG36 morphology release with five classifiers and full-window denominator tables.
- Added configurable preprocessing, strict protocol construction, training, annotation and audit CLI commands.
- Added environment snapshots, release manifests, LFS rules, CI checks and handoff documentation.
