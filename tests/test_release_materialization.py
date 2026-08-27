from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from epilepsy_eeg.figure_manifest import task1_figure_families, write_figure_manifest
from epilepsy_eeg.task2 import materialize_pg36, validate_pg36


def _write(path: Path, value: str = "ok") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _make_task2_source(root: Path) -> None:
    taxonomy = root / "04_label_freeze_pooled"
    taxonomy.mkdir(parents=True)
    (taxonomy / "pooled_label_freeze_manifest.json").write_text(
        json.dumps(
            {
                "taxonomy_version": "test",
                "n_global_categories": 36,
                "n_event_channel": 12,
                "n_windows": 30,
                "same_mouse_merge_violations": 0,
            }
        ),
        encoding="utf-8",
    )
    classification = root / "05_classification_pooled"
    rows = []
    for model in ("eegwavenet_sc", "physio_tf", "physio_stats", "eegnetv4", "eegconformer"):
        model_dir = classification / model
        model_dir.mkdir(parents=True)
        _write(model_dir / "best_model.pt", "checkpoint")
        rows.append({"model": model, "n": 30, "accuracy": 0.5, "balanced_accuracy": 0.5, "macro_f1": 0.5})
    pd.DataFrame(rows).to_csv(classification / "all_window_model_comparison.csv", index=False)

    # Both products intentionally coexist in the source study tree.  The
    # corrected v2 product must win; the old product is not copied into the
    # clean release.
    legacy = root / "06_figures" / "paper_ready_morphology_pooled"
    _write(legacy / "01_independent_channel_pca" / "old_marker.txt", "legacy")
    corrected = root / "06_figures" / "paper_ready_morphology_pooled_v2"
    for directory in (
        "01_per_mouse_pg_pca",
        "02_per_mouse_pg_waveforms",
        "03_global_pg_centroid_pca",
        "04_pg_support_presence",
        "06_classifier_pr_roc_all36",
    ):
        _write(corrected / directory / "new_marker.txt")
    for model in ("eegwavenet_sc", "physio_tf", "physio_stats", "eegnetv4", "eegconformer"):
        _write(corrected / "06_classifier_pr_roc_all36" / model / "figure_pr_all36.png")
        _write(corrected / "06_classifier_pr_roc_all36" / model / "figure_roc_all36.png")

    for source_dir, name in (
        ("06_figures/paper_ready_classifier_pooled", "summary.png"),
        ("06_figures/strict_classifier_full_pooled", "figure_pr_all_classes.png"),
    ):
        for model in ("eegwavenet_sc", "physio_tf", "physio_stats", "eegnetv4", "eegconformer"):
            _write(root / source_dir / model / name)
            if "strict" in source_dir:
                _write(root / source_dir / model / "figure_roc_all_classes.png")
                _write(root / source_dir / model / "figure_confusion_matrix_all_windows.png")

    # Historical source artifacts intentionally coexist with the corrected
    # taxonomy.  A PG36 release must regenerate, not copy, these files.
    _write(root / "07_reports" / "CLASSIFIER_TASK2_REPORT_zh.md", "G01-G45 old report")
    _write(root / "07_reports" / "MORPHOLOGY_TASK2_REPORT_zh.md", "45 个全局类别")
    (root / "08_audit").mkdir(parents=True, exist_ok=True)
    _write(root / "08_audit" / "run_audit.json", json.dumps({"global_categories": 45}))
    _write(root / "08_audit" / "split_audit.json", json.dumps({"classes_present": 45}))
    _write(root / "08_audit" / "classifier_environment.json", json.dumps({"classes": 45}))


def test_materialize_pg36_prefers_corrected_morphology_and_validates(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "release"
    _make_task2_source(source)

    materialize_pg36(source, target)
    assert (target / "figures" / "01_per_mouse_pg_pca" / "new_marker.txt").is_file()
    assert not (target / "figures" / "01_independent_channel_pca" / "old_marker.txt").exists()
    payload = validate_pg36(target)
    assert payload["ok"] is True
    assert payload["primary_test_windows"] == 30
    assert not (target / "reports" / "CLASSIFIER_TASK2_REPORT_zh.md").exists()
    assert not (target / "reports" / "MORPHOLOGY_TASK2_REPORT_zh.md").exists()
    audit = json.loads((target / "audit" / "classifier_environment.json").read_text(encoding="utf-8"))
    assert audit["classes"] == 36
    assert all(value == 36 for value in audit["checkpoint_classes"].values())
    manifest = json.loads((target / "figures" / "FIGURE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["figure_families"]
    assert all(
        not Path(value).is_absolute()
        for family in manifest["figure_families"]
        for value in family.get("source_inputs", [])
    )


def test_figure_manifest_distinguishes_skip_render_from_generated(tmp_path: Path) -> None:
    root = tmp_path / "run"
    figure = root / "figures" / "model_performance"
    _write(figure / "figure.png")
    stages = [{"name": "06_render_performance", "status": "completed"}]

    generated = task1_figure_families(root, stages, requested=True)[0]
    skipped = task1_figure_families(root, stages, requested=False)[0]
    assert generated["status"] == "generated"
    assert skipped["status"] == "not_requested"

    manifest_path = root / "figures" / "FIGURE_MANIFEST.json"
    write_figure_manifest(root=root, task="annotation_detection", families=[generated], path=manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["figure_families"][0]["outputs"][0]["path"].endswith("figure.png")


def test_validate_pg36_rejects_stale_class_count(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "release"
    _make_task2_source(source)
    materialize_pg36(source, target)
    (target / "audit" / "run_audit.json").write_text(
        json.dumps({"global_categories": 45}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="global_categories=45"):
        validate_pg36(target)
