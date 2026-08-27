#!/usr/bin/env python
"""Run the interval-aware clinical event decoder on persisted probabilities.

This runner is intentionally a new audit line.  It never overwrites model
weights, historical ``events`` directories, or historical figures.  A single
decoder operating point is fitted on validation data for each model, then
applied unchanged to the held-out test trace and to the complete recording.
The model is not retrained here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mouse_sleep.epilepsy.corrected_event_decoder import (  # noqa: E402
    build_sequences,
    evaluate_sequences,
    fit_corrected_config,
)


MODEL_ORDER = (
    "eegwavenet_sc",
    "physio_sequence_no_aux",
    "physio_sequence_stats",
)


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_clean(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=PROJECT_ROOT
        / "output/epilepsy_program_20260821/01_annotation_detection/04_model_runs/combined_old_plus_new",
        help="Root containing <model>/continuous/continuous_predictions.parquet.",
    )
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=PROJECT_ROOT
        / "output/epilepsy_program_20260821/01_annotation_detection/03_combined_old_plus_new/protocol",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "output/epilepsy_program_20260821/01_annotation_detection/06_decoder_audit_v2/corrected_combined_old_plus_new",
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    parser.add_argument("--primary-iou", type=float, default=0.30)
    parser.add_argument("--minimum-sensitivity", type=float, default=0.85)
    parser.add_argument("--minimum-precision", type=float, default=0.15)
    parser.add_argument("--maximum-fp-per-hour", type=float, default=40.0)
    parser.add_argument("--allow-existing", action="store_true")
    return parser


def _scope_audit(
    sequences: tuple[Any, ...],
    scope_events: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return events not fully observable in a continuous decoder segment."""

    # Event IDs are intentionally repeated for CH7 and CH8: each valid channel
    # is an independent clinical observation.  A global event-id set would
    # therefore let a represented CH7 row incorrectly mark an excluded CH8 row
    # as observable.  Keep the complete track key in the denominator audit.
    represented: set[tuple[str, str, str, str]] = set()
    for sequence in sequences:
        if not sequence.truth_rows.empty and "event_id" in sequence.truth_rows.columns:
            for row in sequence.truth_rows.itertuples(index=False):
                represented.add(
                    (str(sequence.key[0]), str(sequence.key[1]), str(sequence.key[2]), str(row.event_id))
                )
    source = scope_events.copy()
    source["event_id"] = source["event_id"].astype(str)
    source_keys = list(
        zip(
            source["animal_id"].astype(str),
            source["session_id"].astype(str),
            source["channel_name"].astype(str),
            source["event_id"].astype(str),
            strict=True,
        )
    )
    excluded = source.loc[[key not in represented for key in source_keys]].copy()
    excluded["exclusion_reason"] = "not_fully_contained_in_one_continuous_decoder_segment"
    counts = {
        "clinical_event_denominator": int(len(source)),
        "formal_event_denominator": int(len(represented)),
        "excluded_event_count": int(len(excluded)),
    }
    return excluded, counts


def _write_evaluation(
    model_dir: Path,
    scope: str,
    evaluated: dict[str, Any],
    sequences: tuple[Any, ...],
    scope_events: pd.DataFrame,
    *,
    probability_column: str,
) -> dict[str, Any]:
    scope_dir = model_dir / f"events_corrected_v1_{scope}"
    scope_dir.mkdir(parents=True, exist_ok=True)
    evaluated["predicted_events"].to_parquet(scope_dir / "predicted_events.parquet", index=False)
    evaluated["matches"].to_parquet(scope_dir / "event_matches.parquet", index=False)
    evaluated["true_assessments"].to_parquet(
        scope_dir / "true_event_assessments.parquet", index=False
    )
    track_parts: list[pd.DataFrame] = []
    for key, frame in evaluated["decoded_tracks"].items():
        copy = frame.copy()
        copy["animal_id"] = str(key[0])
        copy["session_id"] = str(key[1])
        copy["channel_name"] = str(key[2])
        copy["decoder_scope"] = scope
        copy["probability_column_used"] = probability_column
        track_parts.append(copy)
    tracks = (
        pd.concat(track_parts, ignore_index=True, sort=False)
        if track_parts
        else pd.DataFrame()
    )
    # Full-recording traces can contain millions of windows.  The interval
    # tables and metrics are sufficient for the audit; preserve an existing
    # track parquet (written by the atlas pass) instead of materialising a
    # second multi-gigabyte frame in memory.
    track_path = scope_dir / "decoded_tracks.parquet"
    if track_parts or not track_path.is_file():
        tracks.to_parquet(track_path, index=False)
    track_rows = int(len(tracks))
    if not track_parts and track_path.is_file():
        try:
            import pyarrow.parquet as pq

            track_rows = int(pq.ParquetFile(track_path).metadata.num_rows)
        except Exception:
            # The audit remains valid if an older parquet engine cannot expose
            # metadata; only this convenience count becomes unknown.
            track_rows = -1
    excluded, denominator = _scope_audit(sequences, scope_events)
    excluded.to_csv(scope_dir / "excluded_events.csv", index=False, encoding="utf-8-sig")
    metrics = evaluated["metrics"]
    record = {
        "scope": scope,
        "probability_column": probability_column,
        "sequence_count": int(len(sequences)),
        "decoded_track_rows": track_rows,
        "predicted_interval_count": int(len(evaluated["predicted_events"])),
        **denominator,
        "metrics": metrics,
    }
    _write_json(scope_dir / "event_metrics.json", record)
    return record


def _run_model(
    model_name: str,
    run_root: Path,
    protocol_dir: Path,
    output_root: Path,
    events: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    trace_path = run_root / model_name / "continuous" / "continuous_predictions.parquet"
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)
    print(f"[corrected-v2] {model_name}: reading {trace_path.name}", flush=True)
    predictions = pd.read_parquet(trace_path)
    model_dir = output_root / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    validation_sequences = build_sequences(predictions, events, split="val")
    print(
        f"[corrected-v2] {model_name}: fitting validation decoder on "
        f"{len(validation_sequences)} sequences",
        flush=True,
    )
    config, decoder_record, candidates = fit_corrected_config(
        validation_sequences,
        minimum_sensitivity=args.minimum_sensitivity,
        minimum_precision=args.minimum_precision,
        maximum_false_positive_events_per_hour=args.maximum_fp_per_hour,
        minimum_iou=args.primary_iou,
    )
    candidates.to_csv(model_dir / "decoder_candidates.csv", index=False, encoding="utf-8-sig")
    _write_json(model_dir / "decoder_fit.json", decoder_record)

    scopes = {
        "validation": (validation_sequences, "split_safe_probability", "val"),
        "test": (build_sequences(predictions, events, split="test"), "split_safe_probability", "test"),
        "full_recording": (build_sequences(predictions, events, split="all"), "seizure_probability", "full_recording"),
    }
    scope_records: dict[str, Any] = {}
    for scope, (sequences, probability_column, split_label) in scopes.items():
        print(
            f"[corrected-v2] {model_name}: decoding {scope} ({len(sequences)} sequences)",
            flush=True,
        )
        evaluated = evaluate_sequences(
            sequences,
            config,
            probability_column=probability_column,
            split_label=split_label,
            minimum_iou=args.primary_iou,
            include_decoded_tracks=(scope != "full_recording"),
        )
        scope_events = events if scope == "full_recording" else events.loc[events["split"].astype(str).eq("val" if scope == "validation" else "test")]
        scope_records[scope] = _write_evaluation(
            model_dir,
            scope,
            evaluated,
            sequences,
            scope_events,
            probability_column=probability_column,
        )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "decoder_kind": "interval_aware_full_window_union",
        "model": model_name,
        "trace_path": str(trace_path.resolve()),
        "trace_sha256": _sha256(trace_path),
        "protocol_dir": str(protocol_dir.resolve()),
        "selected_config": asdict(config),
        "fit_split": "val",
        "test_used_for_selection": False,
        "primary_match_iou": float(args.primary_iou),
        "window_support_seconds": 4.0,
        "boundary_uncertainty_seconds": 2.0,
        "validation_constraints": {
            "sensitivity_floor": float(args.minimum_sensitivity),
            "precision_floor": float(args.minimum_precision),
            "false_positive_events_per_hour_cap": float(args.maximum_fp_per_hour),
        },
        "scopes": scope_records,
        "legacy_directories_untouched": True,
    }
    _write_json(model_dir / "model_manifest.json", manifest)
    return manifest


def main() -> int:
    args = _parser().parse_args()
    run_root = args.run_root.resolve()
    protocol_dir = args.protocol_dir.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.allow_existing:
        raise FileExistsError(
            f"output is non-empty: {output_root}; use --allow-existing to rerun"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(protocol_dir / "channel_events.parquet")
    if "split" not in events.columns:
        raise ValueError("channel_events.parquet must contain split")
    manifests = []
    for model_name in args.models:
        manifests.append(_run_model(model_name, run_root, protocol_dir, output_root, events, args))
    run_manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "protocol_dir": str(protocol_dir),
        "run_root": str(run_root),
        "output_root": str(output_root),
        "models": [item["model"] for item in manifests],
        "manual_event_counts": {
            "train": int((events["split"].astype(str) == "train").sum()),
            "val": int((events["split"].astype(str) == "val").sum()),
            "test": int((events["split"].astype(str) == "test").sum()),
            "all": int(len(events)),
        },
        "decoder_kind": "interval_aware_full_window_union",
        "legacy_directories_untouched": True,
        "notes": [
            "Each valid channel is an independent input stream.",
            "Clinical atlas denominator uses all test channel-events; formal metrics exclude events not fully observable in one decoder segment and report that exclusion explicitly.",
            "External predicted intervals are review candidates, not automatically confirmed false positives.",
        ],
    }
    _write_json(output_root / "manifest.json", run_manifest)
    readme = "\n".join(
        [
            "# Corrected decoder audit v2",
            "",
            "This is a new interval-aware audit line. Historical model/event directories and figures were not overwritten.",
            "",
            "- Each positive decision retains the physical EEG window `[window_start, window_end)` (4 s), then overlapping/adjacent positive windows are unioned.",
            "- Decoder operating point is fitted on validation only; held-out test is never used for selection.",
            "- Test clinical atlas denominator: all rows in protocol/channel_events.parquet with split=test (computed for this run).",
            "- Formal event metrics separately report the fully observable denominator and excluded events.",
            "- `models/<model>/events_corrected_v1_<scope>/decoded_tracks.parquet` is the source for the corrected waveform/probability/state atlas.",
            "- `external` candidates mean zero positive-time overlap with the current manual labels on that channel; they require clinical review.",
        ]
    )
    (output_root / "README_zh.md").write_text(readme + "\n", encoding="utf-8")
    print(f"[corrected-v2] output={output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
