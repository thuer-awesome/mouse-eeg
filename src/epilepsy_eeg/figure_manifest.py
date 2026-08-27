"""Stable, machine-readable figure lineage for the two research tasks.

The project keeps figures as derived artifacts.  This module records *how* a
figure family was produced and which denominator it uses, without importing
matplotlib or touching the source data.  It is intentionally small so an
audit-only installation can validate a run before the optional plotting
dependencies are installed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


FIGURE_SUFFIXES = frozenset({".pdf", ".png", ".svg", ".tiff", ".eps"})


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _safe(value.item())
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value


def _stage_status(stages: Iterable[Mapping[str, Any]], names: Iterable[str]) -> str | None:
    wanted = set(names)
    if not wanted:
        return None
    records = [stage for stage in stages if str(stage.get("name")) in wanted]
    if not records:
        return None
    if any(str(stage.get("status")) == "failed" for stage in records):
        return "failed"
    if all(str(stage.get("status")) == "completed" for stage in records):
        return "generated"
    return "running"


def _output_records(directory: Path, root: Path) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in FIGURE_SUFFIXES:
            continue
        rows.append(
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "bytes": int(path.stat().st_size),
            }
        )
    return rows


def _portable_source(root: Path, value: str | Path) -> str:
    """Store run-internal inputs relative to the run root.

    Figure manifests are committed with the release snapshot.  Absolute
    workstation paths make those manifests misleading after a clone, while
    inputs outside the run (for example a mounted raw recording) still need a
    provenance path.  Resolve only paths that are actually inside ``root``;
    leave external paths untouched so the audit can still identify them.
    """

    path = Path(value)
    try:
        relative = path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return str(path).replace("\\", "/")
    return str(relative).replace("\\", "/")


def figure_family(
    *,
    root: str | Path,
    name: str,
    relative_path: str | Path,
    renderer: str,
    source_inputs: Iterable[str | Path] = (),
    denominator: str | None = None,
    semantics: str | None = None,
    stages: Iterable[Mapping[str, Any]] = (),
    stage_names: Iterable[str] = (),
    requested: bool = True,
) -> dict[str, Any]:
    """Describe one figure family without guessing from a stale directory.

    ``requested=False`` is used for ``--skip-render`` runs.  Existing files in
    an explicitly reused output directory are then reported as ``not_requested``
    rather than being falsely attributed to the current run.
    """

    root_path = Path(root).resolve()
    directory = (root_path / Path(relative_path)).resolve()
    outputs = _output_records(directory, root_path)
    status = _stage_status(stages, stage_names)
    if not requested:
        status = "not_requested"
    elif status is None:
        status = "generated" if outputs else "not_generated"
    elif status == "generated" and not outputs:
        status = "empty"
    return {
        "name": str(name),
        "path": str(Path(relative_path)).replace("\\", "/"),
        "renderer": str(renderer),
        "status": status,
        "source_inputs": [_portable_source(root_path, item) for item in source_inputs],
        "denominator": denominator,
        "semantics": semantics,
        "outputs": outputs,
    }


def write_figure_manifest(
    *,
    root: str | Path,
    task: str,
    families: Iterable[Mapping[str, Any]],
    path: str | Path | None = None,
    schema_version: int = 1,
) -> dict[str, Any]:
    """Write a deterministic manifest and return its JSON-compatible payload."""

    root_path = Path(root).resolve()
    family_rows = [_safe(dict(family)) for family in families]
    payload: dict[str, Any] = {
        "schema_version": int(schema_version),
        "task": str(task),
        "generated_at_utc": _utc_now(),
        "figure_families": family_rows,
    }
    target = Path(path).resolve() if path is not None else root_path / "FIGURE_MANIFEST.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def task1_figure_families(
    root: str | Path,
    stages: Iterable[Mapping[str, Any]],
    *,
    requested: bool = True,
) -> list[dict[str, Any]]:
    """Return the canonical Task-1 figure lineage."""

    root_path = Path(root).resolve()
    return [
        figure_family(
            root=root_path,
            name="model_performance",
            relative_path="figures/model_performance",
            renderer="scripts/render_task1_corrected_model_performance.py",
            source_inputs=(
                root_path / "models",
                root_path / "decoder",
            ),
            denominator="supervised held-out 4-s windows; event TN is undefined",
            semantics="corrected window PR/ROC, window confusion, event metrics and training diagnostics",
            stages=stages,
            stage_names=("06_render_performance",),
            requested=requested,
        ),
        figure_family(
            root=root_path,
            name="clinical_event_atlas",
            relative_path="figures/clinical_event_atlas",
            renderer="scripts/render_corrected_event_atlas.py",
            source_inputs=(
                root_path / "protocol",
                root_path / "decoder",
            ),
            denominator="all manual test event×channel rows; formal exclusions remain explicit",
            semantics="raw waveform, split-safe probability and interval-aware decoded state",
            stages=stages,
            stage_names=("07_render_event_atlas",),
            requested=requested,
        ),
    ]


def task2_figure_families(
    root: str | Path,
    stages: Iterable[Mapping[str, Any]],
    *,
    requested: bool = True,
    expected_classes: int = 36,
) -> list[dict[str, Any]]:
    """Return the canonical Task-2 morphology and classifier figure lineage."""

    root_path = Path(root).resolve()
    morphology = root_path / "06_figures" / "paper_ready_morphology_pooled"
    classifier = root_path / "06_figures" / "paper_ready_classifier_pooled"
    full = root_path / "06_figures" / "strict_classifier_full_pooled"
    return [
        figure_family(
            root=root_path,
            name="morphology_atlas",
            relative_path="06_figures/paper_ready_morphology_pooled",
            renderer="scripts/render_task2_pg36_corrected_v2.py",
            source_inputs=(root_path / "01_tables", root_path / "04_label_freeze_pooled"),
            denominator="all eligible event×valid-channel morphology rows",
            semantics=f"per-mouse independent-channel PCA/waveforms, one centroid per PG category (expected {expected_classes}) and support matrix",
            stages=stages,
            stage_names=("08_render_pg36",),
            requested=requested,
        ),
        figure_family(
            root=root_path,
            name="cluster_diagnostics",
            relative_path="06_figures/paper_ready_morphology_pooled/06_cluster_diagnostics",
            renderer="scripts/render_pooled_cluster_diagnostics.py",
            source_inputs=(root_path / "01_tables", root_path / "03_cross_mouse", root_path / "04_label_freeze_pooled"),
            denominator="all event×valid-channel points; display thinning is recorded by renderer",
            semantics="pooled feature PCA, BIC/stability and category support audit",
            stages=stages,
            stage_names=("09_render_cluster_diagnostics",),
            requested=requested,
        ),
        figure_family(
            root=root_path,
            name="split_audit",
            relative_path="06_figures/paper_ready_classifier_pooled/figure_0_data_split_audit",
            renderer="scripts/render_pooled_split_audit.py",
            source_inputs=(root_path / "04_label_freeze_pooled" / "pooled_channel_window_labels.parquet", root_path / "04_label_freeze_pooled" / "pooled_event_labels.parquet"),
            denominator="all frozen event×channel rows and complete windows by train/val/test",
            semantics="event-level mixed split, independent streams and leakage audit",
            stages=stages,
            stage_names=("10_render_split_audit",),
            requested=requested,
        ),
        figure_family(
            root=root_path,
            name="classifier_summary",
            relative_path="06_figures/paper_ready_classifier_pooled",
            renderer="scripts/render_pg36_classifier_summary.py",
            source_inputs=(root_path / "05_classification_pooled",),
            denominator="all eligible held-out test windows (full-window evaluator)",
            semantics="training curves, model comparison, best-model confusion/support and per-stream robustness",
            stages=stages,
            stage_names=("11_render_classifier_summary",),
            requested=requested,
        ),
        figure_family(
            root=root_path,
            name="classifier_full_window",
            relative_path="06_figures/strict_classifier_full_pooled",
            renderer="scripts/render_pg36_classifier_full.py",
            source_inputs=(root_path / "05_classification_pooled", root_path / "04_label_freeze_pooled"),
            denominator="all eligible held-out test windows; capped-at-8 is sensitivity-only",
            semantics=f"all {expected_classes} PG classes: one-vs-rest PR/ROC, support/recall, confusion and stream metrics",
            stages=stages,
            stage_names=("12_render_classifier_full",),
            requested=requested,
        ),
    ]


__all__ = [
    "figure_family",
    "task1_figure_families",
    "task2_figure_families",
    "write_figure_manifest",
]
