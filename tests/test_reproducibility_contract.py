from __future__ import annotations

import json
from pathlib import Path

from epilepsy_eeg.cli import main as cli_main
from epilepsy_eeg.pipeline import run_pg36_pipeline, run_task1_pipeline


ROOT = Path(__file__).parents[1]


def test_source_model_and_training_files_are_checked_in() -> None:
    required = (
        ROOT / "src/mouse_sleep/epilepsy/single_channel_models.py",
        ROOT / "src/mouse_sleep/epilepsy/single_channel_train.py",
        ROOT / "src/mouse_sleep/epilepsy/single_channel_inference.py",
        ROOT / "scripts/train_pg36.py",
        ROOT / "scripts/evaluate_strict_classifiers_all_windows.py",
        ROOT / "scripts/render_task1_corrected_model_performance.py",
        ROOT / "scripts/render_task2_pg36_corrected_v2.py",
    )
    assert all(path.is_file() for path in required)


def test_task1_dry_run_records_effective_training_and_decoder(capsys) -> None:
    run_task1_pipeline(
        input_path="Z:/raw/recording.parquet",
        labels_path="Z:/labels",
        output_root="Z:/planned/task1",
        config_path=ROOT / "configs/annotation_detection_final.yaml",
        models=("eegwavenet_sc",),
        epochs=3,
        primary_iou=0.5,
        dry_run=True,
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["resolved"]["effective_training"]["eegwavenet_sc"]["epochs"] == 3
    assert payload["resolved"]["effective_decoder"]["primary_match_iou"] == 0.5


def test_task1_dry_run_propagates_full_run_overrides(capsys) -> None:
    """The end-to-end entry point must use (and record) CLI overrides."""

    run_task1_pipeline(
        input_path="Z:/raw/recording.parquet",
        labels_path="Z:/labels",
        output_root="Z:/planned/task1_overrides",
        models=("eegwavenet_sc",),
        eegwavenet_batch_size=77,
        learning_rate_factor=0.25,
        gradient_clip_norm=2.5,
        normalization_clip=8.0,
        maximum_evaluation_windows=123,
        mixed_precision=False,
        filter_overrides={
            "highpass_hz": 1.25,
            "lowpass_hz": 65.0,
            "notch_hz": 60.0,
            "notch_quality_factor": 28.0,
            "bandpass_order": 3,
        },
        dry_run=True,
    )
    payload = json.loads(capsys.readouterr().out)
    resolved = payload["resolved"]
    effective = resolved["effective_training"]["eegwavenet_sc"]
    assert effective["batch_size"] == 77
    assert effective["learning_rate_factor"] == 0.25
    assert effective["gradient_clip_norm"] == 2.5
    assert effective["normalization_clip"] == 8.0
    assert effective["maximum_evaluation_windows"] == 123
    assert effective["mixed_precision"] is False
    assert resolved["preprocessing"]["filter"]["notch_hz"] == 60.0
    assert resolved["effective_protocol"]["filter"]["highpass_hz"] == 1.25
    assert resolved["cli_overrides"]["filter"]["bandpass_order"] == 3


def test_pg36_dry_run_propagates_filter_overrides(capsys) -> None:
    run_pg36_pipeline(
        input_path="Z:/raw",
        labels_path="Z:/labels",
        output_root="Z:/planned/pg36_overrides",
        models=("eegwavenet_sc",),
        filter_overrides={
            "highpass_hz": 1.25,
            "lowpass_hz": 65.0,
            "notch_hz": 60.0,
            "notch_quality_factor": 28.0,
            "bandpass_order": 3,
        },
        dry_run=True,
    )
    payload = json.loads(capsys.readouterr().out)
    resolved = payload["resolved"]
    assert resolved["preprocessing"]["filter"]["highpass_hz"] == 1.25
    assert resolved["cli_overrides"]["filter"]["bandpass_order"] == 3


def test_cli_full_run_accepts_source_and_training_overrides(capsys) -> None:
    rc = cli_main(
        [
            "run-task1",
            "--input",
            "Z:/raw",
            "--labels",
            "Z:/labels",
            "--output",
            "Z:/planned/cli",
            "--models",
            "eegwavenet_sc",
            "--highpass-hz",
            "1.25",
            "--eegwavenet-batch-size",
            "77",
            "--gradient-clip-norm",
            "2.5",
            "--no-mixed-precision",
            "--dry-run",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["resolved"]["effective_training"]["eegwavenet_sc"]["batch_size"] == 77
    assert payload["resolved"]["preprocessing"]["filter"]["highpass_hz"] == 1.25


def test_pg36_dry_run_marks_raw_mode(capsys) -> None:
    run_pg36_pipeline(
        input_path="Z:/raw",
        labels_path="Z:/labels",
        output_root="Z:/planned/pg36",
        models=("eegwavenet_sc",),
        dry_run=True,
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["resolved"]["input_mode"] == "raw_to_result"
    assert payload["resolved"]["effective_training"]["epochs"] > 0
