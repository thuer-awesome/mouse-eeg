from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mouse_sleep.epilepsy.event_morphology import (
    EventMorphologyConfig,
    build_event_signatures,
    fit_event_morphology,
)
from mouse_sleep.epilepsy.morphology import (
    MorphologyConfig,
    build_morphology_index,
    extract_morphology_features,
    fit_morphology_discovery,
    summarize_morphologies,
)
from mouse_sleep.epilepsy.morphology_visualization import (
    SignalCalibration,
    generate_event_morphology_figures,
    generate_morphology_figures,
)
from mouse_sleep.epilepsy.signal_processing import EEGFilterConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Discover candidate seizure waveform morphologies across all labeled mice "
            "without treating overlapping windows as independent events."
        )
    )
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=Path("data/epilepsy_windows_exact/window_manifest.csv"),
    )
    parser.add_argument(
        "--session-manifest",
        type=Path,
        default=Path("data/epilepsy_session_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/epilepsy_morphology_v1"),
    )
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--morphology-stride-seconds", type=float, default=4.0)
    parser.add_argument("--minimum-overlap-fraction", type=float, default=0.75)
    parser.add_argument("--fit-windows-per-event", type=int, default=32)
    parser.add_argument("--max-fit-windows-per-animal", type=int, default=6000)
    parser.add_argument("--pca-components", type=int, default=12)
    parser.add_argument("--highpass-hz", type=float, default=0.5)
    parser.add_argument("--lowpass-hz", type=float, default=70.0)
    parser.add_argument("--notch-hz", type=float, default=50.0)
    parser.add_argument("--notch-quality-factor", type=float, default=30.0)
    parser.add_argument("--bandpass-order", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--signal-unit",
        choices=("raw", "uV"),
        default="raw",
        help="Use uV only when a vendor conversion factor is documented.",
    )
    parser.add_argument(
        "--raw-units-per-uv",
        type=float,
        help="Required for --signal-unit uV; raw value divided by this factor gives uV.",
    )
    parser.add_argument("--reference-sensitivity-uv-per-mm", type=float, default=10.0)
    parser.add_argument("--reference-paper-speed-mm-per-s", type=float, default=30.0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Build tables/models only; useful for pipeline diagnostics.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.signal_unit == "uV" and (
        args.raw_units_per_uv is None or args.raw_units_per_uv <= 0
    ):
        raise ValueError("--signal-unit uV requires a positive --raw-units-per-uv")
    if not 0 < args.minimum_overlap_fraction <= 1:
        raise ValueError("--minimum-overlap-fraction must be in (0, 1]")
    if args.morphology_stride_seconds < args.window_seconds:
        raise ValueError(
            "Morphology discovery windows must be non-overlapping; set stride >= window"
        )

    output = args.output_dir.resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    config = MorphologyConfig(
        window_seconds=args.window_seconds,
        morphology_stride_seconds=args.morphology_stride_seconds,
        minimum_overlap_fraction=args.minimum_overlap_fraction,
        fit_windows_per_event=args.fit_windows_per_event,
        max_fit_windows_per_animal=args.max_fit_windows_per_animal,
        pca_components=args.pca_components,
        filter_highpass_hz=args.highpass_hz,
        filter_lowpass_hz=args.lowpass_hz,
        filter_notch_hz=args.notch_hz,
        filter_notch_quality_factor=args.notch_quality_factor,
        filter_bandpass_order=args.bandpass_order,
        random_seed=args.seed,
    )
    filter_config = EEGFilterConfig(
        highpass_hz=args.highpass_hz,
        lowpass_hz=args.lowpass_hz,
        notch_hz=args.notch_hz,
        notch_quality_factor=args.notch_quality_factor,
        bandpass_order=args.bandpass_order,
    )

    index_path = tables / "morphology_window_index.csv"
    event_path = tables / "event_components.csv"
    feature_path = tables / "morphology_features.parquet"
    feature_preprocessing_path = tables / "feature_preprocessing.json"
    assignment_path = tables / "morphology_assignments.parquet"
    if args.resume and index_path.exists() and event_path.exists():
        print("[morphology] loading cached event/window index", flush=True)
        events = pd.read_csv(event_path)
        index = pd.read_csv(index_path)
        index["is_fit_window"] = index["is_fit_window"].astype(bool)
    else:
        print("[morphology] mapping closed intervals to non-overlapping windows", flush=True)
        events, index = build_morphology_index(
            args.cache_manifest.resolve(), args.session_manifest.resolve(), config
        )
        events.to_csv(event_path, index=False, encoding="utf-8-sig")
        index.to_csv(index_path, index=False, encoding="utf-8-sig")
    _validate_index(events, index)
    print(
        f"[morphology] {len(events):,} event components, {len(index):,} "
        f"non-overlapping windows, {int(index['is_fit_window'].sum()):,} fit windows",
        flush=True,
    )

    requested_preprocessing = {
        "schema_version": 1,
        "method": "continuous_shard_zero_phase_filter_before_window_extraction",
        "filter": filter_config.to_dict(),
    }
    cached_preprocessing = (
        json.loads(feature_preprocessing_path.read_text(encoding="utf-8"))
        if feature_preprocessing_path.exists()
        else None
    )
    if (
        args.resume
        and feature_path.exists()
        and cached_preprocessing == requested_preprocessing
    ):
        print("[morphology] loading cached physiology features", flush=True)
        features = pd.read_parquet(feature_path)
    else:
        print(
            "[morphology] reconstructing continuous shards, filtering, and extracting features",
            flush=True,
        )
        features = extract_morphology_features(index, filter_config=filter_config)
        features.to_parquet(feature_path, index=False)
        feature_preprocessing_path.write_text(
            json.dumps(requested_preprocessing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    _validate_features(features)

    if args.resume and assignment_path.exists():
        print("[morphology] loading cached morphology assignments", flush=True)
        assignments = pd.read_parquet(assignment_path)
        diagnostics = json.loads(
            (tables / "clustering_diagnostics.json").read_text(encoding="utf-8")
        )
    else:
        print(
            "[morphology] fitting event-block-validated GMM and HDBSCAN sensitivity",
            flush=True,
        )
        assignments, diagnostics = fit_morphology_discovery(features, tables, config)
        assignments.to_parquet(assignment_path, index=False)
    summaries = summarize_morphologies(assignments, events)
    for name, frame in summaries.items():
        frame.to_csv(tables / f"{name}.csv", index=False, encoding="utf-8-sig")
    review_assignments = _review_assignment_view(assignments)
    review_summaries = summarize_morphologies(review_assignments, events)
    for name, frame in review_summaries.items():
        frame.to_csv(
            tables / f"review_{name}.csv", index=False, encoding="utf-8-sig"
        )

    print("[morphology] aggregating complete events to one-event-one-vector", flush=True)
    event_signatures = build_event_signatures(assignments, events)
    event_signatures.to_parquet(tables / "event_signatures.parquet", index=False)
    event_assignments, event_type_summary, event_diagnostics = fit_event_morphology(
        event_signatures,
        tables,
        EventMorphologyConfig(random_seed=args.seed),
    )

    calibration = SignalCalibration(
        signal_unit=args.signal_unit,
        raw_units_per_uv=args.raw_units_per_uv,
        reference_sensitivity_uv_per_mm=args.reference_sensitivity_uv_per_mm,
        reference_paper_speed_mm_per_s=args.reference_paper_speed_mm_per_s,
    )
    figure_paths: list[Path] = []
    if not args.skip_figures:
        window_marker = figures / "global_morphology_overview.png"
        review_marker = figures / "review_resolution" / "global_morphology_overview.png"
        review_needed = (
            diagnostics["review_cluster_count"] != diagnostics["global_cluster_count"]
        )
        if args.resume and window_marker.exists() and (
            not review_needed or review_marker.exists()
        ):
            print("[morphology] reusing completed window-level figure atlases", flush=True)
            figure_paths.extend(
                sorted(
                    path
                    for path in figures.rglob("*")
                    if path.is_file()
                    and path.suffix.lower() in {".png", ".pdf", ".svg"}
                    and "complete_events" not in path.parts
                )
            )
        else:
            print("[morphology] rendering global and per-animal clinical atlases", flush=True)
            figure_paths.extend(
                generate_morphology_figures(
                    assignments, summaries, figures, calibration, dpi=args.dpi
                )
            )
            if review_needed:
                print(
                    "[morphology] rendering maximum-supported review-resolution atlases",
                    flush=True,
                )
                figure_paths.extend(
                    generate_morphology_figures(
                        review_assignments,
                        review_summaries,
                        figures / "review_resolution",
                        calibration,
                        dpi=args.dpi,
                    )
                )
        event_marker = figures / "complete_events" / "complete_event_morphology_overview.png"
        if args.resume and event_marker.exists():
            figure_paths.extend(
                sorted(event_marker.parent.glob("complete_event_morphology_overview.*"))
            )
        else:
            figure_paths.extend(
                generate_event_morphology_figures(
                    event_assignments,
                    figures / "complete_events",
                    dpi=args.dpi,
                )
            )
    report_path = _write_report(
        output,
        assignments,
        events,
        summaries,
        diagnostics,
        event_diagnostics,
        event_type_summary,
        calibration,
        len(figure_paths),
    )
    manifest = _write_manifest(
        output,
        args,
        config,
        events,
        assignments,
        diagnostics,
        event_diagnostics,
        calibration,
        report_path,
    )
    print(f"[morphology] report: {report_path}", flush=True)
    print(f"[morphology] manifest: {manifest}", flush=True)
    return 0


def _review_assignment_view(assignments: pd.DataFrame) -> pd.DataFrame:
    result = assignments.copy()
    mapping = {
        "review_morphology_cluster": "morphology_cluster",
        "review_morphology_label": "morphology_label",
        "review_morphology_probability": "morphology_probability",
        "review_morphology_uncertain": "morphology_uncertain",
        "review_morphology_distance": "morphology_distance",
    }
    missing = set(mapping) - set(result)
    if missing:
        raise ValueError(f"Review-resolution assignments are missing: {sorted(missing)}")
    for source, target in mapping.items():
        result[target] = result[source]
    return result


def _validate_index(events: pd.DataFrame, index: pd.DataFrame) -> None:
    required = {
        "animal_id",
        "session_id",
        "event_id",
        "cache_dir",
        "cache_local_index",
        "start_time_ms",
        "event_overlap_fraction",
        "is_fit_window",
    }
    missing = required - set(index)
    if missing:
        raise ValueError(f"Morphology index is missing columns: {sorted(missing)}")
    event_ids = set(events["event_id"].astype(str))
    indexed = set(index["event_id"].astype(str))
    if event_ids != indexed:
        raise ValueError(
            f"Event coverage mismatch: missing={sorted(event_ids - indexed)[:5]}, "
            f"unexpected={sorted(indexed - event_ids)[:5]}"
        )
    duplicates = index.duplicated(["cache_dir", "cache_local_index"])
    if duplicates.any():
        raise ValueError(f"Morphology index contains {int(duplicates.sum())} repeated windows")
    if not index["event_overlap_fraction"].between(0, 1).all():
        raise ValueError("Event overlap fractions must remain within [0, 1]")
    missing_cache = [path for path in index["cache_dir"].unique() if not Path(path).is_dir()]
    if missing_cache:
        raise FileNotFoundError(f"Missing cache directories: {missing_cache[:3]}")


def _validate_features(features: pd.DataFrame) -> None:
    if len(features) < 100:
        raise ValueError("Too few morphology windows for discovery")
    if features["animal_id"].nunique() < 2:
        raise ValueError("Morphology discovery needs more than one animal")
    if features["event_id"].nunique() < 2:
        raise ValueError("Morphology discovery needs more than one event")
    if features["valid_channel_count"].eq(0).all():
        raise ValueError("Every indexed waveform has zero valid EEG channels")


def _write_report(
    output: Path,
    assignments: pd.DataFrame,
    events: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    diagnostics: dict[str, Any],
    event_diagnostics: dict[str, Any],
    event_type_summary: pd.DataFrame,
    calibration: SignalCalibration,
    figure_files: int,
) -> Path:
    type_counts = summaries["per_mouse_global_type_counts"].copy()
    local_path = output / "tables" / "per_mouse_local_type_counts.csv"
    if local_path.exists():
        type_counts = type_counts.merge(pd.read_csv(local_path), on="animal_id", how="left")
    mouse_table = _markdown_table(type_counts)
    cluster_table = _markdown_table(summaries["global_cluster_summary"])
    event_type_table = _markdown_table(event_type_summary)
    unit_statement = (
        "The source values were converted to uV using the documented factor "
        f"{calibration.raw_units_per_uv:g} raw units/uV."
        if calibration.unit_verified
        else (
            "The Parquet schema and recording manifests contain no gain, ADC scale, or "
            "physical unit. Amplitude is therefore reported in raw acquisition units. "
            "The ACNS reference of 10 uV/mm (100 uV/10 mm) is recorded but is not applied."
        )
    )
    report = f"""# Mouse seizure morphology discovery v2

## Scope

This is an exploratory waveform morphology analysis, not a clinical seizure-subtype diagnosis.
All closed CH7/CH8 annotations were merged only when their time intervals overlapped. Every
resulting event component is represented by at least one morphology window. Four-second
windows are non-overlapping. The primary GMM is selected with event-block cross-validation,
event bootstrap stability, cross-animal support, and per-component event support. HDBSCAN is
retained only as a density/outlier sensitivity analysis.

## Dataset

- Animals: {assignments['animal_id'].nunique()}
- Event components: {len(events):,}
- Non-overlapping morphology windows: {len(assignments):,}
- Event-balanced fitting windows: {int(assignments['is_fit_window'].sum()):,}
- Stable coarse GMM families: {diagnostics['global_cluster_count']}
- Maximum supported clinical-review resolution: {diagnostics['review_cluster_count']}
- Raw BIC-only GMM choice (not used as biological count): {diagnostics['gmm_bic_selected_components']}
- Low-posterior-probability windows: {100 * diagnostics['uncertain_fraction']:.2f}%
- HDBSCAN sensitivity clusters: {diagnostics['hdbscan_cluster_count']}
- HDBSCAN sensitivity outlier fraction: {100 * diagnostics['hdbscan_unknown_fraction']:.2f}%
- CH8-only stable families: {diagnostics['ch8_only_cluster_count']}
- CH8-only/main-fit agreement (ARI): {diagnostics['ch8_only_agreement_with_primary_fit_ari']:.3f}
- Cluster/animal normalized mutual information: {diagnostics['animal_cluster_nmi']:.3f}
- Selected GMM event-bootstrap median ARI: {diagnostics['selected_gmm']['bootstrap_median_ari']:.3f}
- Selected GMM event-bootstrap lower quartile ARI: {diagnostics['selected_gmm']['bootstrap_q25_ari']:.3f}
- Figure files: {figure_files}

## Amplitude calibration

{unit_statement}

ACNS Guideline One specifies a routine clinical EEG sensitivity range of 5-10 uV/mm,
commonly 7 uV/mm, and explicitly states that 10 uV/mm gives a 10-mm deflection for
100 uV. It also gives a routine paper speed of 3 cm/s or 10 s/page. Those human clinical
display values are references, not seizure thresholds and not proof of the rodent device unit.

## Candidate types per animal

{mouse_table}

`shared_global_types` counts global morphotypes with at least 10 windows, at least 1% of
the animal's classified windows, and support from at least two event components.
`local_candidate_types` is a separate within-animal HDBSCAN sensitivity analysis. The
maximum-supported review resolution is exploratory and must not be called a confirmed
number of seizure types.

## Global cluster support

{cluster_table}

## Complete-event candidate types

The window families above describe recurring 4-second morphostates. They are not counted as
separate seizures. Every original union event is also collapsed to one vector containing its
morphostate composition, transitions, and robust physiology summaries before event-level
clustering.

- Event-level candidate groups: {event_diagnostics['selected_components']}
- Maximum stable event-review resolution: {event_diagnostics['review_components']}
- Events eligible for balanced event discovery: {event_diagnostics['eligible_training_events']}
- Events per animal in each balanced repeat: {event_diagnostics['balanced_events_per_animal']}
- Events shorter than 4 s (`short_unresolved`): {event_diagnostics['short_unresolved_events']}
- One-window/low-context events: {event_diagnostics['limited_context_events']}
- Event assignments requiring caution: {event_diagnostics['uncertain_events']}

{event_type_table}

## Interpretation boundary

The algorithm can show recurrent waveform families, complete-event composition groups, and
their animal/event support. A clinician must still determine whether a family represents an
independent seizure morphotype, a temporal phase of one seizure, a channel artifact, or a
drug-induced state. Low-context, low-probability, and short events remain explicitly flagged
and must not be converted into clinical names before review.
"""
    path = output / "MORPHOLOGY_DISCOVERY_REPORT.md"
    path.write_text(report, encoding="utf-8")
    return path


def _markdown_table(frame: pd.DataFrame) -> str:
    """Format compact Markdown without requiring the optional tabulate package."""

    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4g}"
            )
        else:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )
    columns = [str(column).replace("|", "\\|") for column in display.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_manifest(
    output: Path,
    args: argparse.Namespace,
    config: MorphologyConfig,
    events: pd.DataFrame,
    assignments: pd.DataFrame,
    diagnostics: dict[str, Any],
    event_diagnostics: dict[str, Any],
    calibration: SignalCalibration,
    report_path: Path,
) -> Path:
    files = sorted(path for path in output.rglob("*") if path.is_file())
    hashes = {
        str(path.relative_to(output)).replace("\\", "/"): _sha256(path)
        for path in files
    }
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache_manifest": str(args.cache_manifest.resolve()),
        "session_manifest": str(args.session_manifest.resolve()),
        "output_dir": str(output),
        "config": config.__dict__,
        "calibration": calibration.__dict__,
        "signal_unit_verified": calibration.unit_verified,
        "animals": sorted(assignments["animal_id"].astype(str).unique()),
        "event_components": int(len(events)),
        "morphology_windows": int(len(assignments)),
        "global_cluster_count": int(diagnostics["global_cluster_count"]),
        "review_cluster_count": int(diagnostics["review_cluster_count"]),
        "event_type_count": int(event_diagnostics["selected_components"]),
        "event_review_type_count": int(event_diagnostics["review_components"]),
        "short_unresolved_events": int(event_diagnostics["short_unresolved_events"]),
        "report": str(report_path),
        "sha256": hashes,
    }
    path = output / "morphology_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
