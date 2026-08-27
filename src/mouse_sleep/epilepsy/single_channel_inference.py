from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from mouse_sleep.epilepsy.seizure_annotation import (
    DecoderConfig,
    EventEvaluation,
    EventInterval,
    ValidationSequence,
    _pool_event_evaluations,
    decode_window_probabilities,
    decoder_config_grid,
    evaluate_event_annotations,
    fit_global_decoder_config,
)
from mouse_sleep.epilepsy.single_channel_models import (
    PhysioSequenceNet,
    build_single_channel_model,
)


def run_continuous_single_channel_inference(
    protocol_dir: str | Path,
    model_run_dir: str | Path,
    output_dir: str | Path,
    *,
    device: str = "auto",
    window_batch_size: int = 1024,
    context_batch_size: int = 2048,
    mixed_precision: bool = True,
) -> Path:
    """Infer every valid channel window with both continuous and split-safe context."""

    if min(window_batch_size, context_batch_size) < 1:
        raise ValueError("inference batch sizes must be positive")
    protocol = Path(protocol_dir).resolve()
    model_run = Path(model_run_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_run / "model.pt"
    full_index_path = protocol / "full_index.parquet"
    baseline_path = protocol / "training_baseline.json"
    for path in (checkpoint_path, full_index_path, baseline_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    signature = {
        "schema_version": 3,
        "model_checkpoint_sha256": _sha256(checkpoint_path),
        "full_index_sha256": _sha256(full_index_path),
        "training_baseline_sha256": _sha256(baseline_path),
        "window_batch_size": int(window_batch_size),
        "context_batch_size": int(context_batch_size),
        "mixed_precision": bool(mixed_precision),
    }
    signature_hash = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode("utf-8")
    ).hexdigest()
    completed_path = output / "completed.json"
    candidate_predictions_path = output / "all_valid_candidate_trace.parquet"
    formal_predictions_path = output / "reviewed_supervised_trace.parquet"
    # Compatibility alias for older callers.  It contains the formal trace,
    # never the all-valid clinical candidate trace.
    predictions_path = output / "continuous_predictions.parquet"
    if completed_path.is_file():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if completed.get("signature_sha256") != signature_hash:
            raise ValueError(f"completed inference has a different signature: {output}")
        for path in (candidate_predictions_path, formal_predictions_path, predictions_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"completed inference is missing required trace: {path}"
                )
        return formal_predictions_path

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = str(checkpoint["model_name"])
    sequence_length = int(checkpoint["sequence_length"])
    model = build_single_channel_model(
        model_name,
        classes=2,
        samples_per_window=1000,
        sample_rate_hz=250.0,
        sequence_length=sequence_length,
        **dict(checkpoint.get("model_kwargs", {})),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    torch_device = _resolve_device(device)
    model = model.to(torch_device).eval()
    amp_enabled = bool(mixed_precision and torch_device.type == "cuda")
    frame = pd.read_parquet(full_index_path)
    frame = frame.loc[frame["channel_valid"]].copy()
    frame = frame.sort_values(
        ["animal_id", "session_id", "channel_name", "start_time_ms"], kind="stable"
    ).reset_index(drop=True)
    # Every valid waveform window is kept for blinded clinical candidate review.
    # Formal training/validation/test metrics use only explicitly reviewed,
    # gate-approved windows.  Legacy protocols have only the older eligibility
    # flag; new v4 protocols supply formal_metric_eligible.
    formal_eligible = _formal_metric_eligibility(frame)
    context_segment_ids = _context_segment_ids(frame)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    stats = {
        (str(row["animal_id"]), str(row["channel_name"])): (
            float(row["center"]), float(row["scale"])
        )
        for row in baseline["channels"]
    }
    probabilities = np.empty(len(frame), dtype=np.float32)
    split_safe_probabilities = np.empty(len(frame), dtype=np.float32)
    arrays: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for track_number, (_, group) in enumerate(
            frame.groupby("continuity_id", sort=False), start=1
        ):
            indices = group.index.to_numpy(np.int64)
            values = _read_normalized_track(group, arrays, stats, clip=20.0)
            if isinstance(model, PhysioSequenceNet):
                continuous_scores, safe_scores = _infer_physio_track(
                    model,
                    values,
                    group["split"].astype(str).to_numpy(),
                    context_segment_ids[indices],
                    torch_device,
                    window_batch_size,
                    context_batch_size,
                    amp_enabled,
                )
            else:
                continuous_scores = _infer_window_track(
                    model,
                    values,
                    torch_device,
                    window_batch_size,
                    amp_enabled,
                )
                safe_scores = continuous_scores.copy()
            probabilities[indices] = continuous_scores
            split_safe_probabilities[indices] = safe_scores
            if track_number % 20 == 0:
                print(
                    f"[continuous-inference] model={model_name} "
                    f"tracks={track_number} windows={int(indices[-1]) + 1}/{len(frame)}",
                    flush=True,
                )
    keep_columns = [
        "sample_id",
        "animal_id",
        "session_id",
        "shard",
        "channel_name",
        "start_time_ms",
        "end_time_ms",
        "center_time_ms",
        "label",
        "channel_event_ids",
        "region_id",
        "region_kind",
        "split",
        "continuity_id",
        "continuity_position",
        "sampling_rate_hz",
        "window_seconds",
        "stride_seconds",
    ]
    optional_columns = (
        "supervised_center_eligible",
        "training_eligible_after_gate",
        "supervision_state",
        "training_label",
        "formal_metric_eligible",
        "context_eligible",
        "context_segment_id",
        "decoder_segment_id",
        "review_source_id",
        "filter_provenance_id",
    )
    keep_columns.extend(column for column in optional_columns if column in frame.columns)
    candidate_result = frame.loc[:, keep_columns].copy()
    candidate_result["formal_metric_eligible_effective"] = formal_eligible
    # This is the effective acquisition/QC context used by sequence models.
    # It differs from a source gate ID when a reused ID is separated by a
    # physical discontinuity or a retained-window gap.
    candidate_result["inference_context_segment_id"] = context_segment_ids
    candidate_result["seizure_probability"] = probabilities
    candidate_result["split_safe_probability"] = split_safe_probabilities
    candidate_result.to_parquet(candidate_predictions_path, index=False)
    formal_result = candidate_result.loc[
        candidate_result["formal_metric_eligible_effective"].astype(bool)
    ].copy()
    formal_result.to_parquet(formal_predictions_path, index=False)
    # Existing reporting modules consume this name.  Keep it as an exact formal
    # trace alias until their callers move to reviewed_supervised_trace.parquet.
    formal_result.to_parquet(predictions_path, index=False)
    manifest = {
        **signature,
        "signature_sha256": signature_hash,
        "model_name": model_name,
        "device": str(torch_device),
        "mixed_precision_effective": amp_enabled,
        "all_valid_candidate_trace": {
            "path": candidate_predictions_path.name,
            "windows": int(len(candidate_result)),
            "tracks": int(candidate_result["continuity_id"].nunique()),
            "sha256": _sha256(candidate_predictions_path),
            "intended_use": "blinded clinical candidate review only; never a formal metric denominator",
        },
        "reviewed_supervised_trace": {
            "path": formal_predictions_path.name,
            "windows": int(len(formal_result)),
            "tracks": int(formal_result["continuity_id"].nunique()),
            "sha256": _sha256(formal_predictions_path),
            "intended_use": "formal supervised validation/test decoding and metrics",
            "eligibility_column": _formal_eligibility_source(frame),
        },
        "continuous_predictions_compatibility_alias": {
            "path": predictions_path.name,
            "targets": formal_predictions_path.name,
            "sha256": _sha256(predictions_path),
        },
        "continuous_probability_context": "adjacent windows in the same physical/QC inference context segment only",
        "split_safe_probability_context": "adjacent windows from the center split and physical/QC inference context segment only",
        "unknown_or_unreviewed_windows_are_excluded_from_formal_trace": True,
        "decoder_must_use_reviewed_supervised_trace": True,
        "identity_is_model_input": False,
    }
    (output / "inference_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    completed_path.write_text(
        json.dumps({"signature_sha256": signature_hash}, indent=2), encoding="utf-8"
    )
    return predictions_path


def fit_and_evaluate_global_event_decoder(
    predictions: str | Path | pd.DataFrame,
    protocol_dir: str | Path,
    output_dir: str | Path,
    *,
    false_positive_penalty: float = 0.02,
    minimum_validation_event_sensitivity: float = 0.70,
    decoder_grid_profile: str = "calibrated",
    primary_match_iou: float = 0.30,
    reported_match_ious: Iterable[float] = (0.0, 0.1, 0.3, 0.5),
) -> Path:
    """Fit one validation-only decoder and evaluate held-out event boundaries.

    A positive overlap is retained only as a descriptive clinical-review
    endpoint.  Decoder selection and the primary validation/test endpoint use
    ``primary_match_iou`` so a long, poorly aligned prediction cannot be
    reported as a detected event merely because it touches an annotation.
    """

    prediction_frame = (
        predictions.copy()
        if isinstance(predictions, pd.DataFrame)
        else pd.read_parquet(predictions)
    )
    protocol = Path(protocol_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    events, event_source = _formal_event_metric_events(protocol)
    regions = pd.read_parquet(protocol / "split_regions.parquet")
    validation_inputs, validation_keys = _build_validation_sequences(
        prediction_frame, events, split="val"
    )
    if decoder_grid_profile == "calibrated":
        # All values are selected before looking at the test split.  This
        # deliberately spans the probability calibration range observed in
        # the validation traces while keeping the fit auditable and fast.
        grid_kwargs = {
            "high_thresholds": (0.30, 0.40, 0.50, 0.60, 0.70),
            "low_thresholds": (0.10, 0.20, 0.30, 0.40, 0.50),
            "median_width_windows": (3,),
            "persistence_values": ((2, 3),),
            "max_gap_seconds": (0.0, 1.0, 2.0),
            "min_event_seconds": (3.0,),
            "boundary_search_seconds": (4.0,),
        }
    elif decoder_grid_profile == "exhaustive":
        grid_kwargs = {
            "high_thresholds": (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975),
            "low_thresholds": (0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.85),
            "median_width_windows": (3, 5, 7),
            "persistence_values": ((2, 3), (3, 5)),
            "max_gap_seconds": (0.0, 1.0, 2.0),
            "min_event_seconds": (3.0, 6.0, 10.0),
            "boundary_search_seconds": (4.0,),
        }
    elif decoder_grid_profile == "repair_long_boundary":
        # Operating points dedicated to the two failure modes found in the
        # legacy clinical atlas: long events disappearing after a high
        # threshold and short probability holes splitting one event.  The
        # grid is still fitted on validation only; it does not inspect test
        # traces or labels.  ``max_gap_seconds`` is deliberately capped at
        # three seconds so a genuine short seizure cannot be bridged across a
        # long non-seizure interval.
        grid_kwargs = {
            # 48 validation-only candidates: enough resolution to test the
            # two observed failure modes without making decoder fitting scale
            # quadratically with the number of long recordings.
            "high_thresholds": (0.25, 0.55, 0.85),
            "low_thresholds": (0.05, 0.30),
            "median_width_windows": (1,),
            "persistence_values": ((1, 2), (2, 3)),
            "max_gap_seconds": (0.0, 2.0),
            "min_event_seconds": (1.0, 3.0),
            "boundary_search_seconds": (6.0,),
        }
    else:
        raise ValueError(
            "decoder_grid_profile must be calibrated, exhaustive, or repair_long_boundary"
        )
    candidates = decoder_config_grid(**grid_kwargs)
    if not 0.0 <= minimum_validation_event_sensitivity <= 1.0:
        raise ValueError("minimum validation event sensitivity must lie in [0, 1]")
    thresholds = _normalise_match_iou_thresholds(
        primary_match_iou, reported_match_ious
    )
    fit = fit_global_decoder_config(
        validation_inputs,
        candidates,
        objective="event_f1_times_mean_iou",
        minimum_iou=primary_match_iou,
        false_positive_penalty=false_positive_penalty,
    )
    chosen, constraint_satisfied, selection_fallback, maximum_sensitivity = (
        _select_decoder_candidate(
            fit.candidates,
            minimum_validation_event_sensitivity=minimum_validation_event_sensitivity,
            primary_match_iou=primary_match_iou,
        )
    )
    candidate_rows = []
    for result in fit.candidates:
        risk = _decoder_candidate_risk_metrics(
            result, primary_match_iou=primary_match_iou
        )
        selection_pool = (
            "sensitivity_floor"
            if float(result.metrics["event_sensitivity"])
            + 1e-12
            >= minimum_validation_event_sensitivity
            else "below_sensitivity_floor"
        )
        if not constraint_satisfied and np.isclose(
            float(result.metrics["event_sensitivity"]),
            maximum_sensitivity,
            rtol=0.0,
            atol=1e-12,
        ):
            selection_pool = "maximum_sensitivity_fallback"
        candidate_rows.append(
            {
                **asdict(result.config),
                "score": float(result.score),
                **{f"metric_{key}": value for key, value in result.metrics.items()},
                **{f"risk_{key}": value for key, value in risk.items()},
                "selection_pool": selection_pool,
                "selected": bool(result is chosen),
            }
        )
    pd.DataFrame(candidate_rows).to_csv(output / "decoder_candidates.csv", index=False)
    decoder_record = {
        "schema_version": 3,
        "fit_split": "val",
        "identity_or_channel_specific_thresholds": False,
        "objective": fit.objective,
        "selection_policy": "validation_only_risk_aware_lexicographic",
        "selection_order": [
            "meet_minimum_validation_event_sensitivity_if_possible",
            "minimize_long_events_ge_30s_zero_coverage",
            "minimize_zero_coverage_events",
            "minimize_low_coverage_events_lt_0p5",
            "maximize_primary_strict_iou_event_f1",
            "maximize_event_f1",
            "maximize_mean_iou",
            "minimize_label_external_candidates_per_channel_hour",
            "maximize_legacy_decoder_score_as_final_tiebreak",
        ],
        "primary_match_iou": float(primary_match_iou),
        "reported_match_ious": [float(value) for value in thresholds],
        "decoder_grid_profile": decoder_grid_profile,
        "decoder_candidate_count": len(candidates),
        "false_positive_penalty": float(false_positive_penalty),
        "minimum_validation_event_sensitivity": float(
            minimum_validation_event_sensitivity
        ),
        "validation_sensitivity_constraint_satisfied": constraint_satisfied,
        "selection_fallback": selection_fallback,
        "maximum_validation_event_sensitivity": maximum_sensitivity,
        "score": float(chosen.score),
        "config": asdict(chosen.config),
        "validation_metrics": _json_clean(chosen.metrics),
        "validation_risk_metrics": _json_clean(
            _decoder_candidate_risk_metrics(
                chosen, primary_match_iou=primary_match_iou
            )
        ),
        "unconstrained_best": {
            "score": float(fit.score),
            "config": asdict(fit.config),
            "validation_metrics": _json_clean(fit.metrics),
        },
        "validation_sequences": [list(key) for key in validation_keys],
        "event_metric_event_source": event_source,
    }
    (output / "decoder_fit.json").write_text(
        json.dumps(decoder_record, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    validation_result = _evaluate_probability_split(
        prediction_frame,
        events,
        "val",
        chosen.config,
        minimum_iou=primary_match_iou,
    )
    test_result = _evaluate_probability_split(
        prediction_frame,
        events,
        "test",
        chosen.config,
        minimum_iou=primary_match_iou,
    )
    full_result = _evaluate_full_recordings(
        prediction_frame,
        events,
        chosen.config,
        minimum_iou=primary_match_iou,
    )
    for name, result in (
        ("validation", validation_result),
        ("test", test_result),
        ("full_recording", full_result),
    ):
        result["predicted_events"].to_parquet(
            output / f"{name}_predicted_events.parquet", index=False
        )
        result["matches"].to_parquet(output / f"{name}_event_matches.parquet", index=False)
        result["true_assessments"].to_parquet(
            output / f"{name}_true_event_assessments.parquet", index=False
        )
    metrics_by_iou: dict[str, Any] = {}
    for threshold in thresholds:
        if np.isclose(threshold, primary_match_iou, rtol=0.0, atol=1e-12):
            threshold_validation = validation_result
            threshold_test = test_result
            threshold_full = full_result
        else:
            threshold_validation = _evaluate_probability_split(
                prediction_frame,
                events,
                "val",
                chosen.config,
                minimum_iou=threshold,
            )
            threshold_test = _evaluate_probability_split(
                prediction_frame,
                events,
                "test",
                chosen.config,
                minimum_iou=threshold,
            )
            threshold_full = _evaluate_full_recordings(
                prediction_frame,
                events,
                chosen.config,
                minimum_iou=threshold,
            )
        metrics_by_iou[_match_iou_label(threshold)] = {
            "minimum_iou": float(threshold),
            "validation": threshold_validation["metrics"],
            "test": threshold_test["metrics"],
            "full_recording_descriptive_only": threshold_full["metrics"],
        }
    metrics = {
        "schema_version": 2,
        "decoder_config": asdict(chosen.config),
        "primary_match_iou": float(primary_match_iou),
        "reported_match_ious": [float(value) for value in thresholds],
        "validation": validation_result["metrics"],
        "test": test_result["metrics"],
        "full_recording_descriptive_only": full_result["metrics"],
        "metrics_by_match_iou": metrics_by_iou,
        "test_is_primary_generalization_result": True,
        "boundary_uncertainty_seconds": chosen.config.boundary_uncertainty_seconds,
        "event_metric_event_source": event_source,
    }
    metrics_path = output / "event_metrics.json"
    metrics_path.write_text(
        json.dumps(_json_clean(metrics), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return metrics_path


def _formal_event_metric_events(protocol: Path) -> tuple[pd.DataFrame, str]:
    """Load the explicit formal event denominator when a protocol supplies it."""

    formal_path = protocol / "formal_channel_events.parquet"
    source_path = formal_path if formal_path.is_file() else protocol / "channel_events.parquet"
    events = pd.read_parquet(source_path)
    if formal_path.is_file():
        required = {
            "event_gate_status",
            "event_gate_reason",
            "formal_decoder_segment_id",
        }
        missing = required - set(events.columns)
        if missing:
            raise ValueError(
                "formal event denominator is missing required audit columns: "
                f"{sorted(missing)}"
            )
        if not events["event_gate_status"].eq("included").all():
            raise ValueError("formal event denominator contains an excluded event")
        if not events["event_gate_reason"].eq(
            "fully_observable_one_decoder_segment"
        ).all():
            raise ValueError("formal event denominator contains an unobservable event")
    return events, source_path.name


def _select_decoder_candidate(
    candidates: Iterable[Any],
    *,
    minimum_validation_event_sensitivity: float,
    primary_match_iou: float = 0.30,
) -> tuple[Any, bool, str | None, float]:
    """Choose a validation-only clinical operating point.

    Event F1 alone can hide exactly the failure mode under review: a decoder
    that scores well on short events while completely missing long seizures.
    Candidates therefore pass a sensitivity floor first and are then ranked by
    clinically relevant failures before conventional event metrics.  When the
    requested floor is impossible, only maximum-sensitivity candidates remain
    eligible; a high-threshold, nearly silent decoder can never win merely by
    producing few false alarms.
    """

    results = tuple(candidates)
    if not results:
        raise ValueError("decoder candidate collection is empty")
    sensitivities = np.asarray(
        [_finite_metric(result.metrics, "event_sensitivity", default=-np.inf) for result in results],
        dtype=float,
    )
    maximum_sensitivity = float(np.nanmax(sensitivities))
    eligible = tuple(
        result
        for result, sensitivity in zip(results, sensitivities, strict=True)
        if sensitivity + 1e-12 >= minimum_validation_event_sensitivity
    )
    constraint_satisfied = bool(eligible)
    if constraint_satisfied:
        candidate_pool = eligible
        fallback = None
    else:
        candidate_pool = tuple(
            result
            for result, sensitivity in zip(results, sensitivities, strict=True)
            if np.isclose(sensitivity, maximum_sensitivity, rtol=0.0, atol=1e-12)
        )
        fallback = "maximum_validation_sensitivity"
    chosen = min(
        candidate_pool,
        key=lambda result: _decoder_candidate_selection_key(
            result, primary_match_iou=primary_match_iou
        ),
    )
    return chosen, constraint_satisfied, fallback, maximum_sensitivity


def _decoder_candidate_risk_metrics(
    result: Any, *, primary_match_iou: float
) -> dict[str, float]:
    """Return the auditable validation criteria used for decoder selection."""

    metrics = result.metrics
    primary_key = f"f1_iou_ge_{_strict_iou_suffix(primary_match_iou)}"
    return {
        "long_events_ge_30s_zero_coverage": _finite_metric(
            metrics, "long_events_ge_30s_zero_coverage", default=np.inf
        ),
        "zero_coverage_events": _finite_metric(
            metrics, "zero_coverage_events", default=np.inf
        ),
        "low_coverage_events_lt_0p5": _finite_metric(
            metrics, "low_coverage_events_lt_0p5", default=np.inf
        ),
        "primary_strict_iou_event_f1": _finite_metric(
            metrics,
            primary_key,
            default=_finite_metric(metrics, "event_f1", default=-np.inf),
        ),
        "event_f1": _finite_metric(metrics, "event_f1", default=-np.inf),
        "mean_iou": _finite_metric(metrics, "mean_iou", default=-np.inf),
        "label_external_candidates_per_channel_hour": _finite_metric(
            metrics, "false_positive_events_per_hour", default=np.inf
        ),
        "legacy_decoder_score": float(result.score)
        if np.isfinite(float(result.score))
        else -np.inf,
    }


def _decoder_candidate_selection_key(
    result: Any, *, primary_match_iou: float
) -> tuple[float, ...]:
    """Lexicographic validation rank, ordered from clinical risk to ties."""

    risk = _decoder_candidate_risk_metrics(
        result, primary_match_iou=primary_match_iou
    )
    config = result.config
    return (
        risk["long_events_ge_30s_zero_coverage"],
        risk["zero_coverage_events"],
        risk["low_coverage_events_lt_0p5"],
        -risk["primary_strict_iou_event_f1"],
        -risk["event_f1"],
        -risk["mean_iou"],
        risk["label_external_candidates_per_channel_hour"],
        -risk["legacy_decoder_score"],
        float(config.high_threshold),
        float(config.low_threshold),
        int(config.median_width_windows),
        int(config.persistence_k),
        int(config.persistence_n),
        float(config.max_gap_seconds),
        float(config.min_event_seconds),
    )


def _strict_iou_suffix(value: float) -> str:
    """Match the stable metric labels emitted by seizure_annotation.py."""

    return f"{float(value):g}".replace(".", "p")


def _finite_metric(
    metrics: Mapping[str, Any], name: str, *, default: float
) -> float:
    value = metrics.get(name, default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _read_normalized_track(
    group: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    stats: Mapping[tuple[str, str], tuple[float, float]],
    *,
    clip: float,
) -> np.ndarray:
    first = group.iloc[0]
    center, scale = stats[(str(first["animal_id"]), str(first["channel_name"]))]
    output = np.empty((len(group), 1, 1000), dtype=np.float32)
    for output_index, row in enumerate(group.itertuples(index=False)):
        root = str(row.filtered_cache_dir)
        if root not in arrays:
            arrays[root] = np.load(Path(root) / "signals.npy", mmap_mode="r")
        values = np.asarray(
            arrays[root][int(row.filtered_local_index), int(row.channel_index)],
            dtype=np.float32,
        )
        output[output_index, 0] = np.clip(
            (values - center) / max(scale, 1e-6), -clip, clip
        )
    return output


def _infer_window_track(
    model: torch.nn.Module,
    values: np.ndarray,
    device: torch.device,
    batch_size: int,
    amp_enabled: bool,
) -> np.ndarray:
    output = np.empty(len(values), dtype=np.float32)
    for left in range(0, len(values), batch_size):
        right = min(left + batch_size, len(values))
        batch = torch.from_numpy(values[left:right]).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(batch)["logits"]
        output[left:right] = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
    return output


def _infer_physio_track(
    model: PhysioSequenceNet,
    values: np.ndarray,
    split_names: np.ndarray,
    context_segment_ids: np.ndarray | None,
    device: torch.device,
    window_batch_size: int,
    context_batch_size: int,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.empty((len(values), model.model_dim), dtype=np.float32)
    statistics = np.empty((len(values), model.statistics_dim), dtype=np.float32)
    for left in range(0, len(values), window_batch_size):
        right = min(left + window_batch_size, len(values))
        batch = torch.from_numpy(values[left:right]).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            batch_embeddings, batch_statistics = model.encode_windows(batch)
        embeddings[left:right] = batch_embeddings.detach().float().cpu().numpy()
        statistics[left:right] = batch_statistics.detach().float().cpu().numpy()

    half = model.sequence_length // 2
    padded_embeddings = np.zeros(
        (len(values) + 2 * half, model.model_dim), dtype=np.float32
    )
    padded_embeddings[half : half + len(values)] = embeddings
    embedding_windows = np.lib.stride_tricks.sliding_window_view(
        padded_embeddings, model.sequence_length, axis=0
    ).transpose(0, 2, 1)
    padded_valid = np.zeros(len(values) + 2 * half, dtype=bool)
    padded_valid[half : half + len(values)] = True
    valid_windows = np.lib.stride_tricks.sliding_window_view(
        padded_valid, model.sequence_length
    )
    split_codes, _ = pd.factorize(split_names, sort=True)
    padded_splits = np.full(len(values) + 2 * half, -1, dtype=np.int16)
    padded_splits[half : half + len(values)] = split_codes.astype(np.int16)
    split_windows = np.lib.stride_tricks.sliding_window_view(
        padded_splits, model.sequence_length
    )
    segment_codes = _segment_codes(context_segment_ids, len(values))
    padded_segments = np.full(len(values) + 2 * half, -1, dtype=np.int64)
    padded_segments[half : half + len(values)] = segment_codes
    segment_windows = np.lib.stride_tricks.sliding_window_view(
        padded_segments, model.sequence_length
    )
    continuous_output = np.empty(len(values), dtype=np.float32)
    safe_output = np.empty(len(values), dtype=np.float32)
    for left in range(0, len(values), context_batch_size):
        right = min(left + context_batch_size, len(values))
        sequence = torch.from_numpy(
            np.array(embedding_windows[left:right], copy=True, order="C")
        ).to(device, non_blocking=True)
        center_codes = split_codes[left:right, None]
        center_segments = segment_codes[left:right, None]
        # The clinical continuous trace may span train/validation/test labels,
        # but must never cross a physical/QC context boundary.  The split-safe
        # trace applies that same physical mask plus its split boundary.
        continuous_mask_numpy = (~valid_windows[left:right]) | (
            segment_windows[left:right] != center_segments
        )
        safe_mask_numpy = continuous_mask_numpy | (
            split_windows[left:right] != center_codes
        )
        continuous_mask = torch.from_numpy(
            np.array(continuous_mask_numpy, copy=True, order="C")
        ).to(device, non_blocking=True)
        safe_mask = torch.from_numpy(np.array(safe_mask_numpy, copy=True, order="C")).to(
            device, non_blocking=True
        )
        center_statistics = torch.from_numpy(statistics[left:right]).to(
            device, non_blocking=True
        )
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            continuous_logits = model.contextualize_embeddings(
                sequence,
                continuous_mask,
                center_statistics=center_statistics,
            )["logits"]
            safe_logits = model.contextualize_embeddings(
                sequence,
                safe_mask,
                center_statistics=center_statistics,
            )["logits"]
        continuous_output[left:right] = (
            torch.softmax(continuous_logits.detach().float(), dim=1)[:, 1].cpu().numpy()
        )
        safe_output[left:right] = (
            torch.softmax(safe_logits.detach().float(), dim=1)[:, 1].cpu().numpy()
        )
    return continuous_output, safe_output


def _formal_eligibility_source(frame: pd.DataFrame) -> str:
    """Return the protocol flag governing formal supervised metrics."""

    for column in (
        "formal_metric_eligible",
        "training_eligible_after_gate",
        "supervised_center_eligible",
    ):
        if column in frame.columns:
            return column
    return "implicit_all_valid_legacy_fallback"


def _formal_metric_eligibility(frame: pd.DataFrame) -> np.ndarray:
    """Select only reviewed, gate-approved rows for formal evaluation.

    A v4 protocol must provide ``formal_metric_eligible``.  The two fallback
    names keep frozen pre-v4 protocols readable, but a new protocol must never
    rely on the implicit all-valid fallback.
    """

    source = _formal_eligibility_source(frame)
    if source == "implicit_all_valid_legacy_fallback":
        return np.ones(len(frame), dtype=bool)
    eligible = frame[source].fillna(False).astype(bool).to_numpy()
    if "supervision_state" in frame.columns:
        approved = frame["supervision_state"].isin(
            ("seizure", "verified_nonseizure")
        ).to_numpy()
        eligible &= approved
    if "training_label" in frame.columns:
        label = pd.to_numeric(frame["training_label"], errors="coerce")
        eligible &= label.isin((0, 1)).to_numpy()
    return eligible


def _context_segment_ids(frame: pd.DataFrame) -> np.ndarray:
    """Build inference-local segments from physical/QC continuity only.

    A source gate ID can reappear after a rejected window, a sample-coordinate
    discontinuity, or a reused recording segment ID.  Treating that recurrence
    as one transformer sequence lets the model borrow invalid context across a
    hard boundary.  This function makes every such run a distinct token while
    leaving unknown IDs missing so :func:`_segment_codes` isolates each one.
    """

    if frame.empty:
        return np.empty(0, dtype=object)
    source = (
        frame["context_segment_id"]
        if "context_segment_id" in frame.columns
        else frame["continuity_id"]
    ).astype("string")
    continuity = (
        frame["continuity_id"].astype("string")
        if "continuity_id" in frame.columns
        else pd.Series("legacy_continuity", index=frame.index, dtype="string")
    )
    if "continuity_position" in frame.columns:
        positions = pd.to_numeric(frame["continuity_position"], errors="coerce")
    else:
        positions = pd.Series(np.nan, index=frame.index, dtype=float)

    output: list[object] = [pd.NA] * len(frame)
    previous_known = False
    previous_source: str | None = None
    previous_continuity: str | None = None
    previous_position = float("nan")
    segment_number = -1
    for row_index, (raw_source, raw_continuity, raw_position) in enumerate(
        zip(source.tolist(), continuity.tolist(), positions.tolist(), strict=True)
    ):
        if pd.isna(raw_source) or not str(raw_source).strip():
            previous_known = False
            previous_source = None
            previous_continuity = None
            previous_position = float("nan")
            continue
        source_id = str(raw_source)
        continuity_id = "__unknown_continuity__" if pd.isna(raw_continuity) else str(raw_continuity)
        position = float(raw_position) if pd.notna(raw_position) else float("nan")
        position_break = (
            "continuity_position" in frame.columns
            and (
                not np.isfinite(position)
                or not np.isfinite(previous_position)
                or not np.isclose(position, previous_position + 1.0)
            )
        )
        new_segment = (
            not previous_known
            or source_id != previous_source
            or continuity_id != previous_continuity
            or position_break
        )
        if new_segment:
            segment_number += 1
        output[row_index] = f"inference_context_{segment_number:08d}"
        previous_known = True
        previous_source = source_id
        previous_continuity = continuity_id
        previous_position = position
    return np.asarray(output, dtype=object)


def _normalise_match_iou_thresholds(
    primary_match_iou: float, reported_match_ious: Iterable[float]
) -> tuple[float, ...]:
    """Validate and deterministically order descriptive IoU endpoints."""

    thresholds = {float(primary_match_iou)}
    thresholds.update(float(value) for value in reported_match_ious)
    if not thresholds or any(
        not np.isfinite(value) or not 0.0 <= value <= 1.0
        for value in thresholds
    ):
        raise ValueError("event-match IoU thresholds must be finite values in [0, 1]")
    return tuple(sorted(thresholds))


def _match_iou_label(value: float) -> str:
    return f"iou_ge_{value:.2f}".replace(".", "p")


def _segment_codes(
    context_segment_ids: np.ndarray | None, length: int
) -> np.ndarray:
    if context_segment_ids is None:
        return np.zeros(length, dtype=np.int64)
    values = np.asarray(context_segment_ids, dtype=object)
    if values.ndim != 1 or len(values) != length:
        raise ValueError("context_segment_ids must align with the waveform track")
    # Factorize gives a stable code for every non-null segment; missing IDs are
    # intentionally made unique so context never leaks through an unknown gap.
    normalized = pd.Series(values, dtype="string")
    missing = normalized.isna().to_numpy()
    normalized = normalized.fillna("__unknown_context_segment__")
    codes, _ = pd.factorize(normalized, sort=True)
    if missing.any():
        codes = codes.astype(np.int64, copy=True)
        codes[missing] = -(np.flatnonzero(missing) + 2)
    return codes.astype(np.int64, copy=False)


def _decoder_segments(group: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    """Yield decoder traces split only at signal/QC discontinuities.

    A decoder segment is deliberately independent of clinical event labels.
    When a v4 context gate is present, its conflict boundaries are hard signal
    boundaries.  Manual event IDs, labels, and split labels are never used to
    divide a continuous trace, so the event decoder is genuinely responsible
    for separating adjacent clinical events.
    """

    ordered = group.sort_values("start_time_ms", kind="stable")
    # Keep the public decoder ID compatible with the frozen event gate.  Its
    # formal_decoder_segment_id is derived from context_segment_id (plus an
    # occurrence fragment where needed), so inference-local IDs must never be
    # emitted as report IDs.  They are still the strictest available boundary
    # signal and are used below only to decide where a decoder trace breaks.
    if "context_segment_id" in ordered.columns:
        public_source = ordered["context_segment_id"]
    elif "continuity_id" in ordered.columns:
        public_source = ordered["continuity_id"]
    else:
        # Small in-memory inputs used by downstream callers may contain just
        # one already-isolated trace.  Preserve that legacy API without
        # inventing a label-derived boundary.
        public_source = pd.Series(
            "legacy_continuity", index=ordered.index, dtype="string"
        )
    if "inference_context_segment_id" in ordered.columns:
        boundary_source = ordered["inference_context_segment_id"]
    else:
        boundary_source = public_source

    public_ids = public_source.astype("string")
    boundary_ids = boundary_source.astype("string")
    known = (
        public_ids.notna()
        & public_ids.ne("")
        & boundary_ids.notna()
        & boundary_ids.ne("")
    ).fillna(False).to_numpy(dtype=bool)
    if not known.any():
        return

    # Formal prediction tables omit unreviewed or QC-rejected windows.  A
    # shared context/continuity ID must therefore not reconnect positions that
    # are no longer physically observable as one trace.  This is intentionally
    # based on acquisition/QC continuity alone, never on label or event data.
    position_break = np.zeros(len(ordered), dtype=bool)
    if "continuity_position" in ordered.columns:
        positions = pd.to_numeric(ordered["continuity_position"], errors="coerce")
        previous_positions = positions.shift()
        adjacent = (
            positions.notna()
            & previous_positions.notna()
            & positions.eq(previous_positions + 1)
        ).to_numpy(dtype=bool)
        position_break = ~adjacent
    else:
        # Infer a genuine temporal acquisition gap for lightweight callers
        # that have timestamps but no explicit continuity positions.  Nominal
        # overlapping model windows are spaced by their start-time stride;
        # using the median makes this robust to a single rejected row.
        starts = ordered["start_time_ms"].to_numpy(dtype=float)
        if len(starts) > 1:
            differences = np.diff(starts)
            positive = differences[differences > 0.0]
            if len(positive):
                nominal_stride = float(np.median(positive))
                position_break[1:] = differences > nominal_stride * 1.5
    position_break[0] = True

    id_changed = boundary_ids.ne(boundary_ids.shift()).fillna(True).to_numpy(dtype=bool)
    previous_known = np.r_[False, known[:-1]]
    run_start = (~known) | (~previous_known) | id_changed | position_break
    run_ids = np.cumsum(run_start, dtype=np.int64)
    emitted_ids: dict[str, int] = {}
    for _, segment in ordered.loc[known].groupby(run_ids[known], sort=False):
        source_id = str(public_source.loc[segment.index].iloc[0])
        occurrence = emitted_ids.get(source_id, 0)
        emitted_ids[source_id] = occurrence + 1
        # A repeated ID after a physical/QC gap is a distinct temporal trace.
        # Keep the first ID unchanged for backwards-compatible report keys.
        segment_id = (
            source_id if occurrence == 0 else f"{source_id}:fragment{occurrence:04d}"
        )
        yield segment_id, segment


def _truth_for_key(
    events: pd.DataFrame, key: tuple[Any, ...], split: str
) -> tuple[EventInterval, ...]:
    truth = _truth_frame_for_key(events, key, split)
    return tuple(
        EventInterval(float(row.start_time_ms), float(row.end_time_ms))
        for row in truth.itertuples(index=False)
    )


def _truth_frame_for_key(
    events: pd.DataFrame, key: tuple[Any, ...], split_label: str
) -> pd.DataFrame:
    mask = (
        events["animal_id"].eq(key[0])
        & events["session_id"].eq(key[1])
        & events["channel_name"].eq(key[2])
    )
    if split_label != "all":
        if "split" not in events.columns:
            raise ValueError("split-scoped event evaluation requires events['split']")
        mask &= events["split"].eq(split_label)
    return events.loc[mask].sort_values(
        "start_time_ms", kind="stable"
    ).reset_index(drop=True)


def _covered_duration_seconds(ordered: pd.DataFrame) -> float:
    """Duration of reviewed waveform coverage, not count times nominal stride."""

    if ordered.empty:
        raise ValueError("cannot compute coverage duration for an empty trace")
    if "end_time_ms" not in ordered.columns:
        raise ValueError("trace must contain end_time_ms for coverage duration")
    starts = ordered["start_time_ms"].to_numpy(float)
    ends = ordered["end_time_ms"].to_numpy(float)
    if np.any(~np.isfinite(starts)) or np.any(~np.isfinite(ends)) or np.any(ends <= starts):
        raise ValueError("trace intervals must have finite positive duration")
    # Consecutive overlapping four-second windows contribute their union once.
    total_ms = 0.0
    current_start = float(starts[0])
    current_end = float(ends[0])
    for start, end in zip(starts[1:], ends[1:], strict=True):
        start = float(start)
        end = float(end)
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total_ms += current_end - current_start
            current_start, current_end = start, end
    total_ms += current_end - current_start
    if total_ms <= 0.0:
        raise ValueError("reviewed trace coverage must be positive")
    return total_ms / 1000.0


def _truth_frame_overlapping_coverage(
    truth_frame: pd.DataFrame, ordered: pd.DataFrame
) -> pd.DataFrame:
    """Return only manual events fully observable in a QC-bounded trace.

    This applies only to true signal/QC discontinuities.  It must not be used
    to manufacture segment boundaries at manual event edges.  A partial event
    at a hard trace boundary remains unscored rather than being treated as a
    partially observed miss.
    """

    if truth_frame.empty:
        return truth_frame.copy()
    start = float(ordered["start_time_ms"].min())
    end = float(ordered["end_time_ms"].max())
    contained = truth_frame["start_time_ms"].ge(start - 1e-6) & truth_frame[
        "end_time_ms"
    ].le(end + 1e-6)
    if "event_gate_status" in truth_frame.columns:
        overlaps = truth_frame["end_time_ms"].gt(start) & truth_frame[
            "start_time_ms"
        ].lt(end)
        if (overlaps & ~contained).any():
            raise ValueError(
                "a formal event crosses a decoder trace boundary; rerun the event gate"
            )
    return truth_frame.loc[contained].reset_index(drop=True)


def _build_validation_sequences(
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    split: str,
) -> tuple[tuple[ValidationSequence, ...], tuple[tuple[str, str, str], ...]]:
    selected = predictions.loc[predictions["split"].eq(split)]
    sequences = []
    keys = []
    for key, group in selected.groupby(
        ["animal_id", "session_id", "channel_name"], sort=True
    ):
        truth = _truth_for_key(events, key, split)
        for segment_key, ordered in _decoder_segments(group):
            segment_truth_frame = _truth_frame_overlapping_coverage(
                _truth_frame_for_key(events, key, split), ordered
            )
            segment_truth = tuple(
                EventInterval(float(row.start_time_ms), float(row.end_time_ms))
                for row in segment_truth_frame.itertuples(index=False)
            )
            sequences.append(
                ValidationSequence(
                    probabilities=ordered["split_safe_probability"].to_numpy(float),
                    window_start_time_ms=ordered["start_time_ms"].to_numpy(float),
                    true_events=segment_truth,
                    recording_duration_seconds=_covered_duration_seconds(ordered),
                )
            )
            keys.append(tuple(map(str, key)) + (str(segment_key),))
    return tuple(sequences), tuple(keys)


def _evaluate_probability_split(
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    split: str,
    config: DecoderConfig,
    *,
    minimum_iou: float = 0.0,
) -> dict[str, Any]:
    return _evaluate_sequences(
        predictions.loc[predictions["split"].eq(split)],
        events.loc[events["split"].eq(split)],
        config,
        probability_column="split_safe_probability",
        split_label=split,
        minimum_iou=minimum_iou,
    )


def _evaluate_full_recordings(
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    config: DecoderConfig,
    *,
    minimum_iou: float = 0.0,
) -> dict[str, Any]:
    return _evaluate_sequences(
        predictions,
        events,
        config,
        probability_column="seizure_probability",
        split_label="all",
        minimum_iou=minimum_iou,
    )


def _evaluate_sequences(
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    config: DecoderConfig,
    *,
    probability_column: str,
    split_label: str,
    minimum_iou: float = 0.0,
) -> dict[str, Any]:
    evaluations: list[EventEvaluation] = []
    evaluation_keys: list[tuple[str, str, str]] = []
    predicted_rows = []
    match_rows = []
    assessment_rows = []
    for key, group in predictions.groupby(
        ["animal_id", "session_id", "channel_name"], sort=True
    ):
        truth_frame = _truth_frame_for_key(events, key, split_label)
        for segment_key, ordered in _decoder_segments(group):
            segment_truth_frame = _truth_frame_overlapping_coverage(truth_frame, ordered)
            decoded = decode_window_probabilities(
                ordered[probability_column].to_numpy(float),
                ordered["start_time_ms"].to_numpy(float),
                config,
            )
            truth = tuple(
                EventInterval(float(row.start_time_ms), float(row.end_time_ms))
                for row in segment_truth_frame.itertuples(index=False)
            )
            evaluation = evaluate_event_annotations(
                truth,
                decoded.events,
                _covered_duration_seconds(ordered),
                minimum_iou=minimum_iou,
            )
            evaluations.append(evaluation)
            evaluation_key = tuple(map(str, key)) + (str(segment_key),)
            evaluation_keys.append(evaluation_key)
            for predicted_index, event in enumerate(decoded.events):
                predicted_rows.append(
                    {
                        "animal_id": key[0],
                        "session_id": key[1],
                        "channel_name": key[2],
                        "split": split_label,
                        "decoder_segment_id": str(segment_key),
                        "predicted_event_index": predicted_index,
                        "start_time_ms": event.start_ms,
                        "end_time_ms": event.end_ms,
                        "duration_s": event.duration_ms / 1000.0,
                    }
                )
            for match in evaluation.matches:
                true_row = segment_truth_frame.iloc[match.true_index]
                match_rows.append(
                    {
                        "animal_id": key[0],
                        "session_id": key[1],
                        "channel_name": key[2],
                        "split": split_label,
                        "decoder_segment_id": str(segment_key),
                        "match_minimum_iou": float(minimum_iou),
                        "event_id": str(true_row["event_id"]),
                        "true_index": match.true_index,
                        "predicted_event_index": match.predicted_index,
                        "overlap_ms": match.overlap_ms,
                        "iou": match.iou,
                        "dice": match.dice,
                        "annotated_coverage": match.annotated_coverage,
                        "onset_error_seconds": match.onset_error_seconds,
                        "offset_error_seconds": match.offset_error_seconds,
                        "duration_error_seconds": match.duration_error_seconds,
                        "duration_ratio": match.duration_ratio,
                    }
                )
            for assessment in evaluation.true_event_assessments:
                true_row = segment_truth_frame.iloc[assessment.true_index]
                assessment_rows.append(
                    {
                        "animal_id": key[0],
                        "session_id": key[1],
                        "channel_name": key[2],
                        "split": split_label,
                        "decoder_segment_id": str(segment_key),
                        "event_id": str(true_row["event_id"]),
                        "annotated_duration_ms": assessment.annotated_duration_ms,
                        "covered_duration_ms": assessment.covered_duration_ms,
                        "coverage": assessment.coverage,
                        "overlapping_predicted_segments": assessment.overlapping_predicted_segments,
                    }
                )
    pooled = _pool_event_evaluations(evaluations)
    per_animal = {}
    per_channel = {}
    for animal in sorted({key[0] for key in evaluation_keys}):
        subset = [
            evaluation
            for key, evaluation in zip(evaluation_keys, evaluations, strict=True)
            if key[0] == animal
        ]
        per_animal[animal] = _json_clean(_pool_event_evaluations(subset))
    for key, evaluation in zip(evaluation_keys, evaluations, strict=True):
        per_channel[":".join(key)] = _json_clean(evaluation.metrics)
    return {
        "metrics": {
            "pooled": _json_clean(pooled),
            "per_animal": per_animal,
            "per_recording_channel": per_channel,
        },
        "predicted_events": pd.DataFrame(predicted_rows),
        "matches": pd.DataFrame(match_rows),
        "true_assessments": pd.DataFrame(assessment_rows),
    }


def _json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "fit_and_evaluate_global_event_decoder",
    "run_continuous_single_channel_inference",
]
