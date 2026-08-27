"""Command-line entry point for paper-replication work only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mouse_sleep.epilepsy.replication.bonn import (
    build_dataset_manifest,
    build_grouped_stratified_folds,
    manifest_summary,
    resolve_bonn_root,
)
from mouse_sleep.epilepsy.replication.chbmit import (
    build_chbmit_manifest,
    build_chbmit_patient_folds,
    chbmit_manifest_summary,
)
from mouse_sleep.epilepsy.replication.chbmit_edf import (
    build_chbmit_window_manifest,
    inspect_chbmit_edf,
)
from mouse_sleep.epilepsy.replication.training import (
    train_bonn_cross_validation,
    train_bonn_fold,
)
from mouse_sleep.epilepsy.replication.reporting import generate_replication_report
from mouse_sleep.epilepsy.replication.verify import verify_replicated_architectures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="epilepsy-replication",
        description="Isolated public-dataset seizure-model replication tools",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    verify = subcommands.add_parser("verify", help="Verify architecture and checkpoint equivalence")
    verify.add_argument("--output", type=Path, required=True)

    prepare = subcommands.add_parser("prepare-bonn", help="Validate Bonn data and freeze grouped folds")
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument(
        "--task",
        choices=[
            "five_class",
            "ictal_vs_all",
            "ictal_vs_interictal",
            "healthy_vs_epilepsy",
        ],
        default="ictal_vs_all",
    )
    prepare.add_argument("--folds", type=int, default=5)
    prepare.add_argument("--seed", type=int, default=20260804)

    train = subcommands.add_parser("train-bonn", help="Train one frozen Bonn fold")
    train.add_argument("--dataset-root", type=Path, required=True)
    train.add_argument("--split-manifest", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument(
        "--model",
        choices=["eegwavenet", "eegnet82"],
        default="eegwavenet",
    )
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument(
        "--normalization",
        choices=["none", "per_record_zscore", "global_train_zscore"],
        default="none",
    )
    train.add_argument("--seed", type=int, default=20260804)
    train.add_argument("--device", default="auto")

    cv = subcommands.add_parser("train-bonn-cv", help="Train and aggregate every frozen Bonn fold")
    cv.add_argument("--dataset-root", type=Path, required=True)
    cv.add_argument("--folds-dir", type=Path, required=True)
    cv.add_argument("--output", type=Path, required=True)
    cv.add_argument(
        "--model", choices=["eegwavenet", "eegnet82"], default="eegwavenet"
    )
    cv.add_argument("--epochs", type=int, default=50)
    cv.add_argument("--batch-size", type=int, default=32)
    cv.add_argument("--learning-rate", type=float, default=1e-3)
    cv.add_argument("--patience", type=int, default=10)
    cv.add_argument(
        "--normalization",
        choices=["none", "per_record_zscore", "global_train_zscore"],
        default="none",
    )
    cv.add_argument("--seed", type=int, default=20260804)
    cv.add_argument("--device", default="auto")

    report = subcommands.add_parser(
        "report", help="Compare default and validation-calibrated CV metrics"
    )
    report.add_argument(
        "--aggregate",
        type=Path,
        action="append",
        required=True,
        help="Path to aggregate_metrics.json; repeat for multiple runs",
    )
    report.add_argument("--output", type=Path, required=True)

    chbmit = subcommands.add_parser(
        "prepare-chbmit", help="Freeze CHB-MIT metadata and patient-disjoint folds"
    )
    chbmit.add_argument("--metadata-root", type=Path, required=True)
    chbmit.add_argument("--output", type=Path, required=True)
    chbmit.add_argument("--dataset-root", type=Path, default=None)
    chbmit.add_argument("--records", type=Path, default=None)
    chbmit.add_argument("--sha256", type=Path, default=None)
    chbmit.add_argument("--verify-checksums", action="store_true")

    inspect_edf = subcommands.add_parser(
        "inspect-chbmit-edf", help="Inspect one CHB-MIT EDF header and canonical montage"
    )
    inspect_edf.add_argument("--edf", type=Path, required=True)
    inspect_edf.add_argument("--output", type=Path, default=None)

    windows = subcommands.add_parser(
        "build-chbmit-windows", help="Index fixed CHB-MIT EDF windows without preloading"
    )
    windows.add_argument("--file-manifest", type=Path, required=True)
    windows.add_argument("--dataset-root", type=Path, required=True)
    windows.add_argument("--output", type=Path, required=True)
    windows.add_argument("--window-seconds", type=float, default=4.0)
    windows.add_argument("--stride-seconds", type=float, default=4.0)
    windows.add_argument(
        "--label-policy",
        choices=["center", "any_overlap", "majority_overlap"],
        default="center",
    )
    windows.add_argument("--max-files", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        report = verify_replicated_architectures(args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "prepare-bonn":
        dataset_root = resolve_bonn_root(args.dataset_root)
        args.output.mkdir(parents=True, exist_ok=True)
        manifest = build_dataset_manifest(
            dataset_root, args.output / "dataset_manifest.csv", task=args.task
        )
        folds = build_grouped_stratified_folds(
            manifest, args.output / "folds", n_splits=args.folds, seed=args.seed
        )
        print(
            json.dumps(
                {
                    "dataset_root": str(dataset_root),
                    "manifest": str(manifest),
                    "summary": manifest_summary(manifest),
                    "folds": folds.to_dict(orient="records"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "train-bonn":
        run_dir = train_bonn_fold(
            args.dataset_root,
            args.split_manifest,
            args.output,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            normalization=args.normalization,
            seed=args.seed,
            device=args.device,
        )
        print(json.dumps({"run_dir": str(run_dir)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "train-bonn-cv":
        cv_dir = train_bonn_cross_validation(
            args.dataset_root,
            args.folds_dir,
            args.output,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            normalization=args.normalization,
            seed=args.seed,
            device=args.device,
        )
        print(json.dumps({"cv_dir": str(cv_dir)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "report":
        csv_path, markdown_path = generate_replication_report(
            args.aggregate, args.output
        )
        print(
            json.dumps(
                {
                    "summary_csv": str(csv_path),
                    "summary_markdown": str(markdown_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "prepare-chbmit":
        args.output.mkdir(parents=True, exist_ok=True)
        metadata_root = args.metadata_root.resolve()
        sha256_path = args.sha256
        if sha256_path is None:
            candidate = metadata_root / "SHA256SUMS.txt"
            sha256_path = candidate if candidate.exists() else None
        manifest_path = args.output / "dataset_manifest.csv"
        manifest = build_chbmit_manifest(
            metadata_root,
            manifest_path,
            dataset_root=args.dataset_root,
            records_path=args.records,
            sha256_path=sha256_path,
            verify_checksums=args.verify_checksums,
        )
        folds = build_chbmit_patient_folds(manifest, args.output / "folds")
        print(
            json.dumps(
                {
                    "manifest": str(manifest_path.resolve()),
                    "summary": chbmit_manifest_summary(manifest),
                    "folds": folds.to_dict(orient="records"),
                    "paper_performance_claim_allowed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "inspect-chbmit-edf":
        inspection = inspect_chbmit_edf(args.edf)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(inspection, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(json.dumps(inspection, ensure_ascii=False, indent=2))
        return 0
    if args.command == "build-chbmit-windows":
        windows = build_chbmit_window_manifest(
            args.file_manifest,
            args.dataset_root,
            args.output,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
            label_policy=args.label_policy,
            max_files=args.max_files,
        )
        print(
            json.dumps(
                {
                    "window_manifest": str(args.output.resolve()),
                    "windows": int(len(windows)),
                    "class_counts": {
                        str(key): int(value)
                        for key, value in windows["label"].value_counts().sort_index().items()
                    },
                    "label_policy": args.label_policy,
                    "paper_performance_claim_allowed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
