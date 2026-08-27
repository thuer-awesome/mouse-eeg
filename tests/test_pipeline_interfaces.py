from __future__ import annotations

from pathlib import Path

from epilepsy_eeg.pipeline import run_pg36_pipeline, run_task1_pipeline


def test_task1_dry_run_does_not_require_mounted_sensitive_files(capsys) -> None:
    output = run_task1_pipeline(
        input_path=Path("Z:/sensitive/raw.parquet"),
        labels_path=Path("Z:/sensitive/labels.csv"),
        output_root=Path("Z:/planned/task1"),
        models=("eegwavenet_sc",),
        dry_run=True,
    )
    assert output == Path("Z:/planned/task1").resolve()
    assert "run-task1" in capsys.readouterr().out


def test_pg36_dry_run_exposes_expected_class_contract(capsys) -> None:
    run_pg36_pipeline(
        taxonomy_root=Path("Z:/frozen/taxonomy"),
        output_root=Path("Z:/planned/pg36"),
        models=("eegwavenet_sc",),
        dry_run=True,
    )
    assert "expected_classes" in capsys.readouterr().out
