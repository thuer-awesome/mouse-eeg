#!/usr/bin/env python
"""Render the corrected clinical event atlas with the historical layout.

The renderer adapts corrected interval-aware parquet outputs to the existing
clinical atlas implementation.  This deliberately reuses the established
three-layer waveform/probability/decoded-state design instead of creating a
new contact-sheet style.  The temporary adapter contains hardlinks only and
is removed after rendering; no legacy output is touched.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mouse_sleep.epilepsy.event_coverage_atlas import (  # noqa: E402
    MODEL_ORDER,
    generate_event_coverage_atlas,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corrected-root",
        type=Path,
        default=PROJECT_ROOT
        / "output/epilepsy_program_20260821/01_annotation_detection/06_decoder_audit_v2/corrected_combined_old_plus_new",
    )
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=PROJECT_ROOT
        / "output/epilepsy_program_20260821/01_annotation_detection/03_combined_old_plus_new/protocol",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "output/epilepsy_program_20260821/01_annotation_detection/06_decoder_audit_v2/corrected_combined_old_plus_new/figures/clinical_event_atlas",
    )
    parser.add_argument("--models", nargs="+", default=list(MODEL_ORDER))
    parser.add_argument("--allow-existing", action="store_true")
    return parser


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    ordered = sorted((float(a), float(b)) for a, b in intervals if float(b) > float(a))
    if not ordered:
        return 0.0
    total = 0.0
    left, right = ordered[0]
    for start, end in ordered[1:]:
        if start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + right - left


def _materialize_prediction_table(source: Path, target: Path) -> None:
    """Materialize a schema-valid candidate table for the atlas adapter.

    A short smoke run can legitimately produce zero decoded intervals.  Some
    parquet writers then emit an empty table with no columns, while the atlas
    still needs the grouping keys to create explicit zero-coverage rows.
    """

    frame = pd.read_parquet(source)
    required = [
        "animal_id",
        "session_id",
        "channel_name",
        "start_time_ms",
        "end_time_ms",
        "duration_s",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing and not frame.empty:
        raise ValueError(f"predicted event table missing columns: {missing}")
    if missing:
        for column in missing:
            frame[column] = pd.Series(dtype="float64")
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)


def _complete_assessments(
    protocol_dir: Path, predicted_path: Path, assessment_path: Path, target: Path
) -> None:
    """Fill formal-gate exclusions so the atlas denominator remains all test labels."""

    manual = pd.read_parquet(protocol_dir / "channel_events.parquet")
    manual = manual.loc[manual["split"].astype(str).eq("test")].copy()
    predicted = pd.read_parquet(predicted_path)
    required = ["animal_id", "session_id", "channel_name", "start_time_ms", "end_time_ms"]
    missing = [column for column in required if column not in predicted.columns]
    if missing and not predicted.empty:
        raise ValueError(f"predicted event table missing columns: {missing}")
    if missing:
        for column in missing:
            predicted[column] = pd.Series(dtype="float64")
    assessments = pd.read_parquet(assessment_path)
    keys = ["animal_id", "session_id", "channel_name"]
    groups = {
        tuple(map(str, key)): frame
        for key, frame in predicted.groupby(keys, sort=False)
    }
    existing = set(assessments.get("event_id", pd.Series(dtype="string")).astype(str))
    rows: list[dict[str, object]] = []
    for event in manual.itertuples(index=False):
        event_id = str(event.event_id)
        if event_id in existing:
            continue
        key = (str(event.animal_id), str(event.session_id), str(event.channel_name))
        group = groups.get(key, predicted.iloc[0:0])
        start, end = float(event.start_time_ms), float(event.end_time_ms)
        overlap = group.loc[
            group["end_time_ms"].astype(float).gt(start)
            & group["start_time_ms"].astype(float).lt(end)
        ]
        intervals = [
            (max(start, float(row.start_time_ms)), min(end, float(row.end_time_ms)))
            for row in overlap.itertuples(index=False)
        ]
        covered = _union_duration(intervals)
        duration = max(1.0, end - start)
        rows.append(
            {
                **{key_name: getattr(event, key_name) for key_name in keys},
                "event_id": event_id,
                "annotated_duration_ms": duration,
                "covered_duration_ms": covered,
                "coverage": covered / duration,
                "overlapping_predicted_segments": int(len(overlap)),
            }
        )
    if rows:
        assessments = pd.concat([assessments, pd.DataFrame(rows)], ignore_index=True, sort=False)
    assessments.to_parquet(target, index=False)


def main() -> int:
    args = _parser().parse_args()
    corrected_root = args.corrected_root.resolve()
    protocol_dir = args.protocol_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_existing:
        raise FileExistsError(f"output is non-empty: {output_dir}; use --allow-existing")
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="corrected_atlas_adapter_") as temporary:
        adapter = Path(temporary)
        for model in args.models:
            source = corrected_root / "models" / model
            event_source = source / "events_corrected_v1_test"
            required = (
                event_source / "predicted_events.parquet",
                event_source / "true_event_assessments.parquet",
                event_source / "decoded_tracks.parquet",
                source / "decoder_fit.json",
            )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError("corrected model output missing: " + ", ".join(missing))
            model_adapter = adapter / "models" / model
            _link_or_copy(event_source / "decoded_tracks.parquet", model_adapter / "continuous" / "continuous_predictions.parquet")
            event_adapter = model_adapter / "events_corrected_v1_test"
            predicted_adapter = event_adapter / "test_predicted_events.parquet"
            _materialize_prediction_table(
                event_source / "predicted_events.parquet", predicted_adapter
            )
            _complete_assessments(
                protocol_dir,
                predicted_adapter,
                event_source / "true_event_assessments.parquet",
                event_adapter / "test_true_event_assessments.parquet",
            )
            _link_or_copy(source / "decoder_fit.json", event_adapter / "decoder_fit.json")
        rendered = generate_event_coverage_atlas(
            adapter,
            protocol_dir,
            output_dir,
            models=tuple(args.models),
            event_directory_name="events_corrected_v1_test",
        )

    # Derive denominators from the actual protocol/audit instead of carrying
    # the historical combined-run counts into a new experiment.
    manual_test = pd.read_parquet(protocol_dir / "channel_events.parquet")
    manual_test = manual_test.loc[manual_test["split"].astype(str).eq("test")]
    clinical_denominator = int(len(manual_test))
    formal_denominator = None
    first_metrics = corrected_root / "models" / str(args.models[0]) / "events_corrected_v1_test" / "event_metrics.json"
    if first_metrics.is_file():
        try:
            formal_denominator = int(json.loads(first_metrics.read_text(encoding="utf-8")).get("formal_event_denominator"))
        except (TypeError, ValueError, json.JSONDecodeError):
            formal_denominator = None
    if formal_denominator is None:
        formal_denominator = clinical_denominator
    manifest_path = output_dir / "corrected_atlas_manifest.json"
    manifest = {
        "schema_version": 1,
        "decoder_kind": "interval_aware_full_window_union",
        "corrected_root": str(corrected_root),
        "protocol_dir": str(protocol_dir),
        "output_dir": str(rendered),
        "models": list(args.models),
        "layout_reference": "mouse_sleep.epilepsy.event_coverage_atlas historical clinical atlas",
        "window_support_seconds": 4.0,
        "boundary_uncertainty_seconds": 2.0,
        "clinical_atlas_denominator": clinical_denominator,
        "formal_metric_denominator": formal_denominator,
        "formal_gate_exclusions": max(0, clinical_denominator - formal_denominator),
        "legacy_atlas_untouched": True,
        "external_candidates_are_review_only": True,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = output_dir / "README_zh_corrected.md"
    readme.write_text(
        "\n".join(
            [
                "# 修正后的临床事件图集",
                "",
                "本图集沿用旧 clinical event atlas 的三层版式：原始波形、概率轨迹、decoded state。",
                "模型阳性窗口按真实物理区间 `[window_start, window_end)`（4 秒）绘制并做区间并集，不再把窗口中心压缩为 1 秒事件。",
                "",
                "- 蓝色半透明：人工事件区间。",
                "- 橙色边框：模型预测区间。",
                "- 橙色状态带：interval-aware decoder 的完整窗口支持。",
                "- label_external_prediction 仅表示与当前人工标签零时间重叠，仍需临床复核，不等同于已确认误报。",
                "- 旧目录保持只读；本目录是 corrected v1 的唯一新图集。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    # Keep the two denominators explicit for clinical readers: the atlas
    # includes every held-out label, while formal metrics exclude only rows
    # that cannot be fully observed within a single decoder segment.
    (output_dir / "DENOMINATOR_NOTE_zh.txt").write_text(
                f"临床图集分母：{clinical_denominator} 个测试通道事件。\n"
        f"正式 interval-aware 指标分母：{formal_denominator} 个完整可观测事件。\n"
        f"形式审计排除事件：{max(0, clinical_denominator - formal_denominator)} 个；它们保留在 true_event_assessments 和 excluded_events 中，并作为零覆盖审计记录，不从图集分母中删除。\n",
        encoding="utf-8",
    )
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
