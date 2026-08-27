#!/usr/bin/env python
"""Train and audit the old-plus-new channel-blind binary epilepsy models.

This runner deliberately reuses the historical single-channel training,
inference, decoder and figure code.  It adds one controlled decoder ablation:
the legacy validation-calibrated operating point is retained alongside a
``repair_long_boundary`` operating point that explicitly searches lower
thresholds, short persistence and small interior-gap bridging.  The latter is
selected on validation only and is never allowed to replace the raw
probability trace.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mouse_sleep.epilepsy.single_channel_inference import (  # noqa: E402
    fit_and_evaluate_global_event_decoder,
    run_continuous_single_channel_inference,
)
from mouse_sleep.epilepsy.single_channel_train import (  # noqa: E402
    SingleChannelTrainingConfig,
    run_single_channel_training,
)


MODEL_ORDER = (
    "eegwavenet_sc",
    "physio_sequence_no_aux",
    "physio_sequence_stats",
)
MODEL_LABELS = {
    "eegwavenet_sc": "EEGWaveNet-SC",
    "physio_sequence_no_aux": "Physio-Sequence",
    "physio_sequence_stats": "Physio + statistics",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "all", "summarize"))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT
        / "output/epilepsy_program_20260821/01_annotation_detection/03_combined_old_plus_new/protocol",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=PROJECT_ROOT
        / "output/epilepsy_program_20260821/01_annotation_detection/04_model_runs/combined_old_plus_new",
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--physio-samples-per-epoch",
        type=int,
        default=262144,
        help="weighted event-region training samples per epoch for 21-window Physio models",
    )
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-decoder", action="store_true")
    return parser


def _train_config(model: str, args: argparse.Namespace) -> SingleChannelTrainingConfig:
    is_stats = model == "physio_sequence_stats"
    is_physio = model != "eegwavenet_sc"
    return SingleChannelTrainingConfig(
        model_name=model,
        epochs=1 if args.smoke else int(args.epochs),
        batch_size=min(int(args.batch_size), 32) if args.smoke else int(args.batch_size),
        samples_per_epoch=(
            1024
            if args.smoke
            else (int(args.physio_samples_per_epoch) if is_physio else None)
        ),
        sampling_mode=(
            "weighted_smoke" if (args.smoke or is_physio) else "all_windows"
        ),
        class_weight_mode="hierarchical_event_region",
        auxiliary_statistics_weight=0.15 if is_stats else 0.0,
        early_stopping_patience=2 if args.smoke else 4,
        learning_rate_patience=1 if args.smoke else 2,
        maximum_evaluation_windows=512 if args.smoke else None,
        seed=int(args.seed),
        device=str(args.device),
        num_workers=int(args.num_workers),
    )


def _copy_tree_files(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)


def _run_decoder(
    model_root: Path,
    protocol: Path,
    *,
    profile: str,
    output_name: str,
    sensitivity_floor: float,
    primary_iou: float,
) -> dict[str, Any]:
    trace = model_root / "continuous" / "reviewed_supervised_trace.parquet"
    if not trace.is_file():
        trace = model_root / "continuous" / "continuous_predictions.parquet"
    if not trace.is_file():
        raise FileNotFoundError(trace)
    output = model_root / output_name
    metrics_path = fit_and_evaluate_global_event_decoder(
        trace,
        protocol,
        output,
        decoder_grid_profile=profile,
        minimum_validation_event_sensitivity=sensitivity_floor,
        primary_match_iou=primary_iou,
        reported_match_ious=(0.0, 0.1, 0.3, 0.5),
    )
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _promote_selected_decoder(model_root: Path) -> str:
    """Promote the validation-better decoder; retain both ablations."""

    candidates: dict[str, dict[str, Any]] = {}
    for name in ("events_baseline", "events_repaired"):
        fit_path = model_root / name / "decoder_fit.json"
        if fit_path.is_file():
            payload = json.loads(fit_path.read_text(encoding="utf-8"))
            metrics = payload.get("validation_metrics", {})
            candidates[name] = metrics
    if not candidates:
        raise FileNotFoundError(f"decoder fits missing under {model_root}")
    # Event F1 is the primary clinical operating-point criterion.  If tied,
    # prefer fewer zero-coverage events, then lower false-positive rate.
    def score(name: str) -> tuple[float, float, float]:
        m = candidates[name]
        return (
            float(m.get("event_f1", float("-inf"))),
            -float(m.get("zero_coverage_events", float("inf"))),
            -float(m.get("false_positive_events_per_hour", float("inf"))),
        )

    selected_name = max(candidates, key=score)
    selected = model_root / selected_name
    canonical = model_root / "events"
    if canonical.exists():
        for path in canonical.iterdir():
            if path.is_file():
                path.unlink()
    canonical.mkdir(parents=True, exist_ok=True)
    for path in selected.iterdir():
        if path.is_file():
            shutil.copy2(path, canonical / path.name)
    return selected_name


def _compact(
    event_metrics: dict[str, Any],
    model: str,
    decoder: str,
    window_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten held-out window and event metrics without mixing denominators."""

    test_window = (
        (window_metrics or {}).get("test", {}).get("pooled", {})
        if window_metrics is not None
        else {}
    )
    test_event = event_metrics.get("test", {}).get("pooled", {})
    return {
        "model": model,
        "model_label": MODEL_LABELS.get(model, model),
        "decoder": decoder,
        "window_auprc": test_window.get("auprc"),
        "window_auroc": test_window.get("auroc"),
        "window_positive_fraction": (
            float(test_window["positive_windows"] / test_window["n_windows"])
            if test_window.get("n_windows")
            else None
        ),
        "event_sensitivity": test_event.get("event_sensitivity"),
        "event_precision": test_event.get("event_precision"),
        "event_f1": test_event.get("event_f1"),
        "annotated_event_coverage": test_event.get("annotated_event_coverage"),
        "zero_coverage_events": test_event.get("zero_coverage_events"),
        "long_events_ge_30s_zero_coverage": test_event.get("long_events_ge_30s_zero_coverage"),
        "false_positive_events_per_hour": test_event.get("false_positive_events_per_hour"),
        "median_absolute_onset_error_seconds": test_event.get("median_absolute_onset_error_seconds"),
        "median_absolute_offset_error_seconds": test_event.get("median_absolute_offset_error_seconds"),
    }


def _run_one(model: str, args: argparse.Namespace, protocol: Path, root: Path) -> list[dict[str, Any]]:
    model_root = root / model
    config = _train_config(model, args)
    run_single_channel_training(protocol, model_root, config)
    if args.smoke or args.skip_decoder:
        return []
    run_continuous_single_channel_inference(
        protocol,
        model_root,
        model_root / "continuous",
        device=str(args.device),
        window_batch_size=1024,
        context_batch_size=2048,
    )
    baseline = _run_decoder(
        model_root,
        protocol,
        profile="calibrated",
        output_name="events_baseline",
        sensitivity_floor=0.70,
        primary_iou=0.30,
    )
    repaired = _run_decoder(
        model_root,
        protocol,
        profile="repair_long_boundary",
        output_name="events_repaired",
        sensitivity_floor=0.85,
        primary_iou=0.10,
    )
    selected_decoder = _promote_selected_decoder(model_root)
    window_metrics = json.loads(
        (model_root / "metrics.json").read_text(encoding="utf-8")
    )
    (model_root / "decoder_comparison.json").write_text(
        json.dumps(
            {
                "selection": f"{selected_decoder}_is_canonical; other_decoder_retained_for_ablation",
                "canonical_decoder": selected_decoder,
                "baseline": baseline,
                "repaired": repaired,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return [
        _compact(baseline, model, "baseline_calibrated", window_metrics),
        _compact(repaired, model, "repaired_long_boundary", window_metrics),
    ]


def _summarize(root: Path) -> None:
    rows: list[dict[str, Any]] = []
    canonical_by_model: dict[str, str] = {}
    for model in MODEL_ORDER:
        path = root / model / "decoder_comparison.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        canonical_by_model[model] = str(payload.get("canonical_decoder", "events_repaired"))
        metrics_path = root / model / "metrics.json"
        window_metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.is_file()
            else None
        )
        rows.extend(
            [
                _compact(payload["baseline"], model, "baseline_calibrated", window_metrics),
                _compact(payload["repaired"], model, "repaired_long_boundary", window_metrics),
            ]
        )
    if rows:
        pd.DataFrame(rows).to_csv(root / "combined_decoder_comparison.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "models": [model for model in MODEL_ORDER if (root / model / "completed.json").is_file()],
        "canonical_decoder": canonical_by_model,
        "baseline_decoder": "events_baseline",
        "training_started": bool(rows),
        "metrics_source": "validation-fitted decoder; held-out test never used for selection",
    }
    (root / "RUN_SUMMARY.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    protocol = args.protocol.resolve()
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if args.command in {"train", "all"}:
        for model in args.models:
            print(f"[combined] start {model}", flush=True)
            rows.extend(_run_one(model, args, protocol, root))
            print(f"[combined] done {model}", flush=True)
    if args.command in {"all", "summarize"}:
        _summarize(root)
    elif rows:
        pd.DataFrame(rows).to_csv(root / "combined_decoder_comparison.csv", index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
