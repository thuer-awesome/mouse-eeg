# Contributing

Thank you for improving the reproducibility of this mouse EEG project.

## Before opening a change

- Do not commit raw recordings, institution-sensitive labels, local caches, or credentials.
- Keep Task 1 binary annotation and Task 2 PG36 morphology classification as separate interfaces and result trees.
- Treat every valid channel as an independent stream/observation; do not introduce animal or channel identity as a model feature.
- Put changed parameters in a YAML config and write a new output directory and manifest. Do not overwrite a frozen release.

## Required checks

```powershell
python -m compileall -q src scripts
python -m pytest -q
epilepsy-eeg audit --task1-root results/task1_annotation_detection --task2-root results/task2_morphology_pg36
```

Figure changes should include the source table or manifest used to make the figure. New metrics must state the denominator and whether the table is a primary full-window result or an event-balanced sensitivity analysis.

## Pull requests

Describe the data contract, configuration diff, expected compute environment, and any change to the formal result scope. A reviewer must be able to reproduce the command without access to a hidden local path.
