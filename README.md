# mouse-eeg: Epilepsy EEG Research

This repository is the cleaned, reproducible release of the mouse epilepsy EEG study. It contains two deliberately separate workflows:

1. **Task 1 — binary continuous annotation.** Every valid channel is an independent input stream. Confirmed seizure intervals are mapped to `seizure=1`; all eligible background is `non-seizure=0`. The three frozen models are EEGWaveNet-SC, Physio-Sequence, and Physio + statistics. A validation-only interval-aware decoder converts window probabilities into physical onset/offset intervals and a clinician-facing PDF atlas.
2. **Task 2 — PG36 morphology classification.** Every event x valid channel is an independent observation. Within-mouse clustering, pooled prototypes, cross-mouse PCA/UMAP, representative waveforms and five frozen classifiers (EEGWaveNet-SC, Physio-TF, Physio + statistics, EEGNetv4 and EEGConformer) are retained. PG01–PG36 are exploratory morphology categories, not clinically validated seizure subtypes.

The repository contains code, configuration, reports, and selected final derived artifacts. Raw EEG and institution-sensitive recordings are referenced through manifests and are not committed by default. Large PDFs, Parquet tables and weights use Git LFS when they are included in a release clone.

## Layout

```text
configs/        frozen experiment defaults and override points
data/           input contract and de-identified source manifests
src/            reusable adapters, protocol builders and model wrappers
scripts/        paper-result renderers and reproducible training scripts
results/        two clean audited release bundles only
docs/           workflow, configuration, API and clinical limitations
tests/          lightweight adapter and release-invariant checks
environment/    pip/conda dependency snapshots
```

## Install

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[all,dev]"
```

For a CPU-only audit, `pip install -e ".[parquet,viz,dev]"` is sufficient. CUDA and the exact PyTorch build must be selected for the target machine.

## CLI entry points

```text
epilepsy-eeg prepare           raw EEG -> filtered cache + offline protocol
epilepsy-eeg build-task1-protocol  cache + event components -> purged train/val/test protocol
epilepsy-eeg annotate          raw/protocol input -> channel predictions + PDF atlas
epilepsy-eeg train-task1       configurable binary training
epilepsy-eeg render-task1      corrected event atlas rendering
epilepsy-eeg materialize-task1 copy the frozen Task-1 release
epilepsy-eeg materialize-task2 copy the frozen PG36 release
epilepsy-eeg pg36-report       validate/copy PG36 results
epilepsy-eeg train-pg36        configurable five-model PG36 training
epilepsy-eeg render-pg36       regenerate the PG36 figure package
epilepsy-eeg audit              validate both release roots
```

Run `epilepsy-eeg <command> --help` for every parameter. The YAML files in `configs/` are the documented defaults; command-line values override them.

`configs/annotation_detection_final.yaml` controls preprocessing, protocol
purge, decoder and three-model training. `configs/morphology_pg36_final.yaml`
controls PG36 training. The `prepare` and `build-task1-protocol` commands make
the preprocessing and leakage-safe split steps explicit; `annotate` remains an
offline/replay doctor-in-the-loop interface and is not a validated real-time
clinical deployment.

## Reproducibility

The formal results are frozen snapshots from the audited source project. Their exact source paths, hashes, denominator definitions and excluded legacy lineages are recorded in `results/**/release_manifest.json`. The release does not silently retrain a model when rendering a figure. Task 1 corrected decoding is a separate operation from model training; Task 2 PG36 training cannot be used to claim corrected onset/offset performance.

See [docs/WORKFLOW_zh.md](docs/WORKFLOW_zh.md), [docs/TASK1_ANNOTATION_DETECTION_zh.md](docs/TASK1_ANNOTATION_DETECTION_zh.md), [docs/TASK2_MORPHOLOGY_PG36_zh.md](docs/TASK2_MORPHOLOGY_PG36_zh.md), and [docs/API_zh.md](docs/API_zh.md).
Configuration and environment details are in [docs/CONFIGURATION_zh.md](docs/CONFIGURATION_zh.md).
