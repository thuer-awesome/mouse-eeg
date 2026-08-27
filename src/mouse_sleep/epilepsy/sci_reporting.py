"""Publication-ready reporting for completed mouse epilepsy experiments.

The module is intentionally read-only with respect to training artifacts. It
collects frozen LOSO splits, per-fold histories, predictions, and metrics, then
exports source-data tables, journal-size figures, and a Chinese SCI-style
experimental report. Formal mouse LOSO results are kept separate from Bonn
engineering validation and capped AITRICS smoke tests.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.metrics import precision_recall_curve, roc_curve

from mouse_sleep.epilepsy.loso_reporting import generate_loso_report

__all__ = [
    "SciReportOutputs",
    "build_split_tables",
    "build_training_tables",
    "generate_epilepsy_sci_report",
]


MODEL_LABELS = {
    "eegwavenet_paper_exact": "EEGWaveNet",
    "eegnet_8_2_paper_exact": "EEGNet-8,2",
}
MODEL_COLORS = {
    "eegwavenet_paper_exact": "#0072B2",
    "eegnet_8_2_paper_exact": "#E69F00",
}
ANIMAL_ORDER = ("E50", "E76", "E77", "E79", "E81", "E82")
ANIMAL_COLORS = {
    animal: color
    for animal, color in zip(
        ANIMAL_ORDER,
        ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"),
        strict=True,
    )
}
LINE_STYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)))
FORMAL_MODELS = tuple(MODEL_LABELS)


@dataclass(frozen=True)
class SciReportOutputs:
    report: Path
    figures_dir: Path
    tables_dir: Path
    manifest: Path
    figures: tuple[Path, ...]
    tables: tuple[Path, ...]


def generate_epilepsy_sci_report(
    runs_root: str | Path,
    output_dir: str | Path | None = None,
    *,
    bonn_root: str | Path | None = None,
    aitrics_smoke_root: str | Path | None = None,
    dpi: int = 600,
) -> SciReportOutputs:
    """Generate tables, figures, and a manuscript-style report.

    Only launcher-completed formal LOSO runs are admitted to the main result
    tables. Bonn and AITRICS outputs are inventoried separately with their
    validation status retained.
    """

    root = Path(runs_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (root / "sci_report").resolve()
    )
    figures_dir = output / "figures"
    tables_dir = output / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    clinical_dir = root / "clinical_report"
    generate_loso_report(root, clinical_dir)
    queue_manifest = _read_json(root / "queue_manifest.json")
    inventory = pd.read_csv(
        clinical_dir / "run_inventory.csv", encoding="utf-8-sig"
    )
    per_animal = pd.read_csv(
        clinical_dir / "per_animal_metrics.csv", encoding="utf-8-sig"
    )
    per_session = pd.read_csv(
        clinical_dir / "per_session_metrics.csv", encoding="utf-8-sig"
    )
    fold_summary = pd.read_csv(
        clinical_dir / "model_fold_summary.csv", encoding="utf-8-sig"
    )
    pooled = pd.read_csv(
        clinical_dir / "model_pooled_metrics.csv", encoding="utf-8-sig"
    )
    confusion = pd.read_csv(
        clinical_dir / "confusion_matrices.csv", encoding="utf-8-sig"
    )
    _validate_formal_inventory(inventory, per_animal, fold_summary)

    split_design, split_summary, animal_distribution = build_split_tables(
        queue_manifest
    )
    history_long, training_diagnostics = build_training_tables(inventory)
    model_configuration = _build_model_configuration(inventory)
    bonn = _build_bonn_summary(Path(bonn_root).resolve() if bonn_root else None)
    aitrics = _build_aitrics_smoke_summary(
        Path(aitrics_smoke_root).resolve() if aitrics_smoke_root else None
    )
    experiment_inventory = _build_experiment_inventory(
        inventory, bonn, aitrics
    )

    table_frames: dict[str, pd.DataFrame] = {
        "split_design.csv": split_design,
        "split_summary.csv": split_summary,
        "animal_distribution.csv": animal_distribution,
        "model_configuration.csv": model_configuration,
        "training_history_long.csv": history_long,
        "training_diagnostics.csv": training_diagnostics,
        "per_animal_metrics.csv": per_animal,
        "per_session_metrics.csv": per_session,
        "model_fold_summary.csv": fold_summary,
        "model_pooled_metrics.csv": pooled,
        "confusion_matrices.csv": confusion,
        "bonn_public_engineering_validation.csv": bonn,
        "aitrics_smoke_summary.csv": aitrics,
        "trained_experiment_inventory.csv": experiment_inventory,
    }
    table_paths: list[Path] = []
    for name, frame in table_frames.items():
        path = tables_dir / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        table_paths.append(path)

    _set_publication_style()
    figure_bases: list[Path] = []
    figure_bases.append(
        _plot_protocol_and_distribution(
            queue_manifest,
            animal_distribution,
            figures_dir / "figure1_protocol_and_data_distribution",
            dpi=dpi,
        )
    )
    for model, name in (
        ("eegwavenet_paper_exact", "figure2_eegwavenet_loss_curves"),
        ("eegnet_8_2_paper_exact", "figure3_eegnet_loss_curves"),
    ):
        figure_bases.append(
            _plot_loss_curves(
                history_long,
                model,
                figures_dir / name,
                dpi=dpi,
            )
        )
    figure_bases.append(
        _plot_animal_performance(
            per_animal,
            figures_dir / "figure4_animal_level_performance",
            dpi=dpi,
        )
    )
    figure_bases.append(
        _plot_confusion_and_events(
            per_animal,
            confusion,
            figures_dir / "figure5_confusion_and_event_performance",
            dpi=dpi,
        )
    )
    figure_bases.append(
        _plot_roc_pr_curves(
            inventory,
            figures_dir / "figure6_test_roc_pr_curves",
            dpi=dpi,
        )
    )
    figure_bases.append(
        _plot_threshold_and_convergence(
            per_animal,
            training_diagnostics,
            figures_dir / "figureS1_threshold_and_convergence",
            dpi=dpi,
        )
    )
    if not bonn.empty:
        figure_bases.append(
            _plot_bonn_engineering_validation(
                bonn,
                figures_dir / "figureS2_bonn_engineering_validation",
                dpi=dpi,
            )
        )

    figure_paths = tuple(
        path
        for base in figure_bases
        for path in (base.with_suffix(".png"), base.with_suffix(".pdf"), base.with_suffix(".svg"))
    )
    _verify_figure_outputs(figure_paths)

    report_path = output / "SCI_RESULTS_REPORT_zh.md"
    report_path.write_text(
        _render_sci_report(
            split_design=split_design,
            split_summary=split_summary,
            model_configuration=model_configuration,
            training_diagnostics=training_diagnostics,
            per_animal=per_animal,
            fold_summary=fold_summary,
            pooled=pooled,
            bonn=bonn,
            aitrics=aitrics,
        ),
        encoding="utf-8",
    )

    manifest_payload = {
        "schema_version": 1,
        "scope": "mouse_epilepsy_same_day_acute_post_drug_loso",
        "formal_models": list(FORMAL_MODELS),
        "formal_model_folds": int(len(inventory)),
        "biological_units": int(per_animal["test_animal"].nunique()),
        "figures": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in figure_paths
        ],
        "tables": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "rows": int(len(table_frames[path.name])),
            }
            for path in table_paths
        ],
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
        "notes": [
            "Formal mouse results are animal-level LOSO on same-day acute post-drug data.",
            "Bonn outputs are public-dataset engineering validation, not original-paper performance replication.",
            "AITRICS output is a capped one-epoch smoke test and is excluded from formal model comparison.",
        ],
    }
    manifest_path = output / "sci_report_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return SciReportOutputs(
        report=report_path,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        manifest=manifest_path,
        figures=figure_paths,
        tables=tuple(table_paths),
    )


def build_split_tables(
    queue_manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build fold roles, split counts, and per-animal label distributions."""

    folds = queue_manifest.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("queue_manifest must contain a non-empty folds list")
    design_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    reference_frame: pd.DataFrame | None = None
    for fold_payload in folds:
        fold = int(fold_payload["fold"])
        manifest = Path(str(fold_payload["manifest"]))
        frame = pd.read_csv(manifest, encoding="utf-8-sig")
        required_columns = {
            "animal_id",
            "session_id",
            "split",
            "n_windows",
            "positive_windows",
        }
        missing = sorted(required_columns.difference(frame.columns))
        if missing:
            raise ValueError(f"{manifest} is missing columns: {', '.join(missing)}")
        if reference_frame is None:
            reference_frame = frame.copy()
        role_animals: dict[str, list[str]] = {}
        split_values: dict[str, dict[str, Any]] = {}
        for split in ("train", "val", "test"):
            group = frame.loc[frame["split"].astype(str).eq(split)].copy()
            if group.empty:
                raise ValueError(f"fold {fold} has no {split} rows")
            animals = sorted(group["animal_id"].astype(str).unique())
            role_animals[split] = animals
            windows = int(pd.to_numeric(group["n_windows"]).sum())
            positives = int(pd.to_numeric(group["positive_windows"]).sum())
            values = {
                "fold": fold,
                "split": split,
                "animals": ",".join(animals),
                "n_animals": len(animals),
                "n_sessions": int(group["session_id"].astype(str).nunique()),
                "n_windows": windows,
                "positive_windows": positives,
                "negative_windows": windows - positives,
                "positive_fraction": positives / windows,
            }
            split_values[split] = values
            summary_rows.append(values)
        design_rows.append(
            {
                "fold": fold,
                "train_animals": ",".join(role_animals["train"]),
                "validation_animal": ",".join(role_animals["val"]),
                "test_animal": ",".join(role_animals["test"]),
                "train_windows": split_values["train"]["n_windows"],
                "validation_windows": split_values["val"]["n_windows"],
                "test_windows": split_values["test"]["n_windows"],
                "train_positive_fraction": split_values["train"]["positive_fraction"],
                "validation_positive_fraction": split_values["val"]["positive_fraction"],
                "test_positive_fraction": split_values["test"]["positive_fraction"],
            }
        )
    assert reference_frame is not None
    animal_rows: list[dict[str, Any]] = []
    for animal, group in reference_frame.groupby("animal_id", sort=False):
        windows = int(pd.to_numeric(group["n_windows"]).sum())
        positives = int(pd.to_numeric(group["positive_windows"]).sum())
        animal_rows.append(
            {
                "animal_id": str(animal),
                "n_sessions": int(group["session_id"].astype(str).nunique()),
                "n_windows": windows,
                "positive_windows": positives,
                "negative_windows": windows - positives,
                "positive_fraction": positives / windows,
            }
        )
    animals = pd.DataFrame(animal_rows)
    animals["animal_id"] = pd.Categorical(
        animals["animal_id"], categories=ANIMAL_ORDER, ordered=True
    )
    animals = animals.sort_values("animal_id").reset_index(drop=True)
    animals["animal_id"] = animals["animal_id"].astype(str)
    return (
        pd.DataFrame(design_rows).sort_values("fold").reset_index(drop=True),
        pd.DataFrame(summary_rows).sort_values(["fold", "split"]).reset_index(drop=True),
        animals,
    )


def build_training_tables(
    inventory: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect all epoch histories and derive checkpoint diagnostics."""

    history_parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for row in inventory.sort_values(["model", "fold"]).to_dict(orient="records"):
        run_dir = Path(str(row["run_dir"]))
        history_path = run_dir / "history.csv"
        history = pd.read_csv(history_path, encoding="utf-8-sig")
        required = {"epoch", "train_loss", "val_loss"}
        missing = sorted(required.difference(history.columns))
        if missing:
            raise ValueError(f"{history_path} is missing columns: {', '.join(missing)}")
        for column in required:
            history[column] = pd.to_numeric(history[column], errors="raise")
        if not np.isfinite(history[list(required)].to_numpy(dtype=float)).all():
            raise ValueError(f"{history_path} contains non-finite values")
        epochs = history["epoch"].astype(int).to_numpy()
        if not np.array_equal(epochs, np.arange(1, len(history) + 1)):
            raise ValueError(f"{history_path} epochs are not contiguous from 1")
        model = str(row["model"])
        fold = int(row["fold"])
        test_animal = str(row["test_animal"])
        augmented = history.assign(
            model=model,
            model_label=MODEL_LABELS.get(model, model),
            fold=fold,
            test_animal=test_animal,
            run_dir=str(run_dir),
        )
        history_parts.append(augmented)
        best_index = int(history["val_loss"].idxmin())
        best = history.loc[best_index]
        final = history.iloc[-1]
        diagnostics.append(
            {
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "fold": fold,
                "test_animal": test_animal,
                "epochs": int(len(history)),
                "best_epoch": int(best["epoch"]),
                "train_loss_at_best": float(best["train_loss"]),
                "best_val_loss": float(best["val_loss"]),
                "final_train_loss": float(final["train_loss"]),
                "final_val_loss": float(final["val_loss"]),
                "final_generalization_gap": float(
                    final["val_loss"] - final["train_loss"]
                ),
                "run_dir": str(run_dir),
            }
        )
    return (
        pd.concat(history_parts, ignore_index=True),
        pd.DataFrame(diagnostics).sort_values(["model", "fold"]).reset_index(drop=True),
    )


def _build_model_configuration(inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    architecture = {
        "eegwavenet_paper_exact": {
            "input_adapter": "2x1000 @250 Hz -> linear interpolation to 2x1024",
            "architecture": "6 depthwise temporal scales + five Conv1d heads + MLP",
        },
        "eegnet_8_2_paper_exact": {
            "input_adapter": "2x1000 @250 Hz -> linear interpolation to 2x1024",
            "architecture": "EEGNet F1=8, D=2, F2=16; temporal/spatial/separable blocks",
        },
    }
    for model, group in inventory.groupby("model", sort=True):
        manifests = [
            _read_json(Path(str(path)) / "run_manifest.json")
            for path in group["run_dir"]
        ]
        fields = (
            "parameters",
            "epochs",
            "batch_size",
            "learning_rate",
            "seed",
            "device",
        )
        values: dict[str, Any] = {}
        for field in fields:
            unique = {json.dumps(item.get(field), sort_keys=True) for item in manifests}
            if len(unique) != 1:
                raise ValueError(f"{field} differs across formal {model} folds")
            values[field] = manifests[0].get(field)
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS.get(str(model), str(model)),
                **architecture.get(str(model), {}),
                **values,
                "optimizer": "AdamW",
                "weight_decay": 1e-4,
                "training_sampler": "inverse-frequency WeightedRandomSampler",
                "loss": "class-weighted cross-entropy",
                "checkpoint_rule": "minimum validation loss",
                "threshold_rule": "validation F1 maximum over 0.10-0.90, step 0.01",
            }
        )
    return pd.DataFrame(rows)


def _build_bonn_summary(root: Path | None) -> pd.DataFrame:
    columns = (
        "model",
        "model_label",
        "protocol",
        "normalization",
        "folds",
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "auroc_mean",
        "auroc_std",
        "auprc_mean",
        "auprc_std",
        "seizure_recall_mean",
        "seizure_recall_std",
        "replication_status",
        "paper_performance_claim_allowed",
        "run_dir",
    )
    if root is None or not root.is_dir():
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/aggregate_metrics.json")):
        payload = _read_json(path)
        aggregate = payload.get("aggregate_metrics")
        if not isinstance(aggregate, Mapping):
            continue
        folds_dir = str(payload.get("folds_dir", ""))
        if "ictal_vs_all" in folds_dir:
            protocol = "S vs Z/O/N/F"
        elif "ictal_vs_interictal" in folds_dir:
            protocol = "S vs N/F"
        else:
            protocol = "unknown"

        def stat(name: str, key: str) -> float:
            item = aggregate.get(name, {})
            return float(item.get(key, np.nan)) if isinstance(item, Mapping) else np.nan

        model = str(payload.get("model", "unknown"))
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "protocol": protocol,
                "normalization": str(payload.get("normalization", "unknown")),
                "folds": int(payload.get("folds", 0)),
                "accuracy_mean": stat("test_accuracy", "mean"),
                "accuracy_std": stat("test_accuracy", "std"),
                "balanced_accuracy_mean": stat("test_balanced_accuracy", "mean"),
                "balanced_accuracy_std": stat("test_balanced_accuracy", "std"),
                "macro_f1_mean": stat("test_macro_f1", "mean"),
                "macro_f1_std": stat("test_macro_f1", "std"),
                "auroc_mean": stat("test_auroc", "mean"),
                "auroc_std": stat("test_auroc", "std"),
                "auprc_mean": stat("test_auprc", "mean"),
                "auprc_std": stat("test_auprc", "std"),
                "seizure_recall_mean": stat("test_recall_class_1", "mean"),
                "seizure_recall_std": stat("test_recall_class_1", "std"),
                "replication_status": str(payload.get("replication_status", "")),
                "paper_performance_claim_allowed": bool(
                    payload.get("paper_performance_claim_allowed", False)
                ),
                "run_dir": str(path.parent.resolve()),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["protocol", "model", "normalization"]
    ).reset_index(drop=True)


def _build_aitrics_smoke_summary(root: Path | None) -> pd.DataFrame:
    columns = (
        "model",
        "status",
        "epochs",
        "train_sequences",
        "validation_sequences",
        "test_sequences",
        "train_windows",
        "validation_windows",
        "test_windows",
        "train_loss",
        "validation_loss",
        "threshold",
        "test_accuracy",
        "test_sensitivity",
        "test_specificity",
        "test_f1",
        "test_auroc",
        "test_auprc",
        "run_dir",
    )
    if root is None or not root.is_dir():
        return pd.DataFrame(columns=columns)
    candidates = [
        path.parent
        for path in root.glob("*/metrics.json")
        if (path.parent / "history.csv").is_file()
        and (path.parent / "run_manifest.json").is_file()
    ]
    if not candidates:
        return pd.DataFrame(columns=columns)
    run_dir = max(candidates, key=lambda path: path.stat().st_mtime)
    metrics = _read_json(run_dir / "metrics.json")
    manifest = _read_json(run_dir / "run_manifest.json")
    history = pd.read_csv(run_dir / "history.csv", encoding="utf-8-sig")
    datasets = metrics.get("datasets", {})
    test_window = metrics.get("test", {}).get("window", {})
    row = {
        "model": str(manifest.get("model", "aitrics")),
        "status": "capped_one_epoch_smoke_only",
        "epochs": int(manifest.get("epochs", len(history))),
        "train_sequences": _nested_number(datasets, "train", "sequences"),
        "validation_sequences": _nested_number(datasets, "val", "sequences"),
        "test_sequences": _nested_number(datasets, "test", "sequences"),
        "train_windows": _nested_number(datasets, "train", "windows"),
        "validation_windows": _nested_number(datasets, "val", "windows"),
        "test_windows": _nested_number(datasets, "test", "windows"),
        "train_loss": float(history.iloc[-1]["train_loss"]),
        "validation_loss": float(history.iloc[-1]["val_loss"]),
        "threshold": float(metrics.get("threshold_selected_on_validation", np.nan)),
        "test_accuracy": _mapping_number(test_window, "accuracy"),
        "test_sensitivity": _mapping_number(test_window, "sensitivity"),
        "test_specificity": _mapping_number(test_window, "specificity"),
        "test_f1": _mapping_number(test_window, "f1"),
        "test_auroc": _mapping_number(test_window, "auroc"),
        "test_auprc": _mapping_number(test_window, "auprc"),
        "run_dir": str(run_dir.resolve()),
    }
    return pd.DataFrame([row], columns=columns)


def _build_experiment_inventory(
    formal: pd.DataFrame, bonn: pd.DataFrame, aitrics: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in formal.to_dict(orient="records"):
        rows.append(
            {
                "family": "mouse_formal_loso",
                "model": row["model"],
                "unit": f"fold_{int(row['fold']):02d}_test_{row['test_animal']}",
                "status": "formal_completed",
                "included_in_primary_comparison": True,
                "path": row["run_dir"],
            }
        )
    for row in bonn.to_dict(orient="records"):
        rows.append(
            {
                "family": "bonn_public_engineering_validation",
                "model": row["model"],
                "unit": f"{row['protocol']}__{row['normalization']}__{int(row['folds'])}fold",
                "status": row["replication_status"],
                "included_in_primary_comparison": False,
                "path": row["run_dir"],
            }
        )
    for row in aitrics.to_dict(orient="records"):
        rows.append(
            {
                "family": "mouse_capped_smoke",
                "model": row["model"],
                "unit": "fold_01_one_epoch_capped",
                "status": row["status"],
                "included_in_primary_comparison": False,
                "path": row["run_dir"],
            }
        )
    return pd.DataFrame(rows)


def _plot_protocol_and_distribution(
    queue_manifest: Mapping[str, Any],
    animal_distribution: pd.DataFrame,
    output_base: Path,
    *,
    dpi: int,
) -> Path:
    folds = queue_manifest["folds"]
    matrix = np.zeros((len(folds), len(ANIMAL_ORDER)), dtype=int)
    for row_index, fold in enumerate(folds):
        train = set(map(str, fold["train_animals"]))
        val = str(fold["val_animal"])
        test = str(fold["test_animal"])
        for column, animal in enumerate(ANIMAL_ORDER):
            matrix[row_index, column] = 0 if animal in train else 1 if animal == val else 2 if animal == test else -1
    fig = plt.figure(figsize=(7.2, 3.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.05, 1.35))
    ax_roles = fig.add_subplot(grid[0, 0])
    ax_dist = fig.add_subplot(grid[0, 1])
    cmap = ListedColormap(("#D9D9D9", "#56B4E9", "#E69F00"))
    norm = BoundaryNorm((-0.5, 0.5, 1.5, 2.5), cmap.N)
    ax_roles.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax_roles.text(
                column,
                row,
                ("Tr", "Val", "Test")[matrix[row, column]],
                ha="center",
                va="center",
                fontsize=6.5,
                color="black",
            )
    ax_roles.set_xticks(range(len(ANIMAL_ORDER)), ANIMAL_ORDER)
    ax_roles.set_yticks(range(len(folds)), [f"Fold {int(item['fold'])}" for item in folds])
    ax_roles.set_xlabel("Animal")
    ax_roles.set_ylabel("Outer LOSO fold")
    ax_roles.set_title("Animal-level split roles")
    for spine in ax_roles.spines.values():
        spine.set_visible(False)

    ordered = animal_distribution.set_index("animal_id").loc[list(ANIMAL_ORDER)]
    x = np.arange(len(ordered))
    negatives = ordered["negative_windows"].to_numpy(dtype=float) / 1000.0
    positives = ordered["positive_windows"].to_numpy(dtype=float) / 1000.0
    ax_dist.bar(x, negatives, color="#A7A9AC", label="Non-seizure", width=0.72)
    ax_dist.bar(
        x,
        positives,
        bottom=negatives,
        color="#D55E00",
        label="Seizure",
        width=0.72,
    )
    for index, fraction in enumerate(ordered["positive_fraction"]):
        ax_dist.text(
            index,
            negatives[index] + positives[index] + 2.5,
            f"{100 * float(fraction):.1f}%",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    ax_dist.set_xticks(x, ordered.index)
    ax_dist.set_xlabel("Animal")
    ax_dist.set_ylabel("Windows (thousands)")
    ax_dist.set_title("Out-of-fold data and seizure prevalence")
    ax_dist.legend(frameon=False, ncol=2, loc="upper center")
    _panel_label(ax_roles, "A")
    _panel_label(ax_dist, "B")
    _export_figure(fig, output_base, dpi=dpi)
    return output_base


def _plot_loss_curves(
    history: pd.DataFrame,
    model: str,
    output_base: Path,
    *,
    dpi: int,
) -> Path:
    subset = history.loc[history["model"].astype(str).eq(model)].copy()
    if subset["fold"].nunique() != 6:
        raise ValueError(f"expected six histories for {model}")
    minimum = max(float(subset[["train_loss", "val_loss"]].min().min()) * 0.75, 1e-3)
    maximum = float(subset[["train_loss", "val_loss"]].max().max()) * 1.25
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.0), sharex=True, sharey=True, constrained_layout=True)
    for panel, (fold, group) in enumerate(subset.groupby("fold", sort=True)):
        ax = axes.flat[panel]
        group = group.sort_values("epoch")
        ax.plot(
            group["epoch"],
            group["train_loss"],
            color="#0072B2",
            linewidth=1.3,
            label="Train",
        )
        ax.plot(
            group["epoch"],
            group["val_loss"],
            color="#D55E00",
            linewidth=1.3,
            linestyle="--",
            label="Validation",
        )
        best = group.loc[group["val_loss"].idxmin()]
        ax.scatter(
            [best["epoch"]],
            [best["val_loss"]],
            color="#000000",
            marker="*",
            s=34,
            zorder=5,
            label="Selected checkpoint" if panel == 0 else None,
        )
        ax.annotate(
            f"best={int(best['epoch'])}",
            (best["epoch"], best["val_loss"]),
            xytext=(3, 5),
            textcoords="offset points",
            fontsize=6.2,
        )
        animal = str(group["test_animal"].iloc[0])
        ax.set_title(f"Fold {int(fold)}: held-out {animal}")
        ax.set_yscale("log")
        ax.set_ylim(minimum, maximum)
        ax.set_xlim(1, 30)
        ax.grid(axis="y", which="major", color="#D9D9D9", linewidth=0.5)
        _panel_label(ax, chr(ord("A") + panel))
    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("Weighted CE loss (log scale)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.015))
    _export_figure(fig, output_base, dpi=dpi)
    return output_base


def _plot_animal_performance(
    per_animal: pd.DataFrame,
    output_base: Path,
    *,
    dpi: int,
) -> Path:
    metrics = (
        ("balanced_accuracy", "Balanced accuracy"),
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("f1", "F1 score"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), constrained_layout=True)
    for panel, (metric, label) in enumerate(metrics):
        ax = axes.flat[panel]
        _paired_model_axis(ax, per_animal, metric, ylabel=label, ylim=(0.0, 1.03))
        if metric in {"balanced_accuracy", "auroc"}:
            ax.axhline(0.5, color="#777777", linestyle=":", linewidth=0.8)
        _panel_label(ax, chr(ord("A") + panel))
    _export_figure(fig, output_base, dpi=dpi)
    return output_base


def _plot_confusion_and_events(
    per_animal: pd.DataFrame,
    confusion: pd.DataFrame,
    output_base: Path,
    *,
    dpi: int,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
    for column, model in enumerate(FORMAL_MODELS):
        row = confusion.loc[confusion["model"].astype(str).eq(model)]
        if len(row) != 1:
            raise ValueError(f"missing pooled confusion matrix for {model}")
        item = row.iloc[0]
        counts = np.asarray(
            [[item["tn"], item["fp"]], [item["fn"], item["tp"]]], dtype=int
        )
        normalized = counts / counts.sum(axis=1, keepdims=True)
        ax = axes[0, column]
        image = ax.imshow(normalized, cmap="cividis", vmin=0.0, vmax=1.0)
        for truth in range(2):
            for prediction in range(2):
                value = normalized[truth, prediction]
                color = "white" if value > 0.55 else "black"
                ax.text(
                    prediction,
                    truth,
                    f"{counts[truth, prediction]:,}\n{100 * value:.1f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                )
        ax.set_xticks((0, 1), ("Non-seizure", "Seizure"), rotation=15, ha="right")
        ax.set_yticks((0, 1), ("Non-seizure", "Seizure"))
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class" if column == 0 else "")
        ax.set_title(MODEL_LABELS[model])
        if column == 1:
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row fraction")
        _panel_label(ax, chr(ord("A") + column))
    _paired_model_axis(
        axes[1, 0],
        per_animal,
        "event_sensitivity",
        ylabel="Event sensitivity",
        ylim=(0.0, 1.05),
    )
    _panel_label(axes[1, 0], "C")
    _paired_model_axis(
        axes[1, 1],
        per_animal,
        "false_alarms_per_hour",
        ylabel="False alarms per hour",
        ylim=(0.0, max(140.0, float(per_animal["false_alarms_per_hour"].max()) * 1.08)),
    )
    _panel_label(axes[1, 1], "D")
    _export_figure(fig, output_base, dpi=dpi)
    return output_base


def _plot_roc_pr_curves(
    inventory: pd.DataFrame,
    output_base: Path,
    *,
    dpi: int,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.5), constrained_layout=True)
    for row_index, model in enumerate(FORMAL_MODELS):
        model_rows = inventory.loc[inventory["model"].astype(str).eq(model)].sort_values("fold")
        prevalences: list[float] = []
        for curve_index, item in enumerate(model_rows.to_dict(orient="records")):
            frame = pd.read_csv(
                Path(str(item["run_dir"])) / "predictions.csv",
                usecols=["split", "label", "seizure_probability"],
                encoding="utf-8-sig",
            )
            frame = frame.loc[frame["split"].astype(str).eq("test")]
            y = frame["label"].to_numpy(dtype=np.int8)
            probability = frame["seizure_probability"].to_numpy(dtype=float)
            animal = str(item["test_animal"])
            color = ANIMAL_COLORS[animal]
            style = LINE_STYLES[curve_index % len(LINE_STYLES)]
            fpr, tpr, _ = roc_curve(y, probability)
            precision, recall, _ = precision_recall_curve(y, probability)
            per_run_metrics = _read_json(Path(str(item["run_dir"])) / "metrics.json")
            test_window = per_run_metrics["test"]["window"]
            axes[row_index, 0].plot(
                fpr,
                tpr,
                color=color,
                linestyle=style,
                linewidth=1.1,
                label=f"{animal} (AUC {float(test_window['auroc']):.2f})",
            )
            axes[row_index, 1].plot(
                recall,
                precision,
                color=color,
                linestyle=style,
                linewidth=1.1,
                label=f"{animal} (AP {float(test_window['auprc']):.2f})",
            )
            prevalences.append(float(y.mean()))
        roc_ax = axes[row_index, 0]
        pr_ax = axes[row_index, 1]
        roc_ax.plot((0, 1), (0, 1), color="#777777", linestyle=":", linewidth=0.8)
        pr_ax.axhline(
            float(np.mean(prevalences)),
            color="#777777",
            linestyle=":",
            linewidth=0.8,
            label=f"Mean prevalence {np.mean(prevalences):.2f}",
        )
        for ax in (roc_ax, pr_ax):
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.grid(color="#E6E6E6", linewidth=0.5)
            ax.legend(frameon=False, fontsize=5.7, loc="best")
        roc_ax.set_xlabel("False-positive rate")
        roc_ax.set_ylabel("True-positive rate")
        pr_ax.set_xlabel("Recall")
        pr_ax.set_ylabel("Precision")
        roc_ax.set_title(f"{MODEL_LABELS[model]}: ROC")
        pr_ax.set_title(f"{MODEL_LABELS[model]}: precision-recall")
    for panel, ax in enumerate(axes.flat):
        _panel_label(ax, chr(ord("A") + panel))
    _export_figure(fig, output_base, dpi=dpi)
    return output_base


def _plot_threshold_and_convergence(
    per_animal: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_base: Path,
    *,
    dpi: int,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
    _animal_series_axis(
        axes[0],
        per_animal,
        "threshold_selected_on_validation",
        ylabel="Selected threshold",
        ylim=(0.0, 0.7),
    )
    _animal_series_axis(
        axes[1], diagnostics, "best_epoch", ylabel="Best epoch", ylim=(0.0, 31.0)
    )
    _animal_series_axis(
        axes[2],
        diagnostics,
        "final_generalization_gap",
        ylabel="Final val - train loss",
        ylim=(0.0, max(1.0, float(diagnostics["final_generalization_gap"].max()) * 1.08)),
    )
    for panel, ax in enumerate(axes):
        _panel_label(ax, chr(ord("A") + panel))
    _export_figure(fig, output_base, dpi=dpi)
    return output_base


def _plot_bonn_engineering_validation(
    bonn: pd.DataFrame,
    output_base: Path,
    *,
    dpi: int,
) -> Path:
    metrics = (
        ("balanced_accuracy_mean", "Balanced accuracy"),
        ("macro_f1_mean", "Macro F1"),
        ("auroc_mean", "AUROC"),
        ("auprc_mean", "AUPRC"),
    )
    ordered = bonn.copy().reset_index(drop=True)
    ordered["configuration"] = (
        ordered["protocol"]
        + " | "
        + ordered["normalization"].replace(
            {
                "global_train_zscore": "global train z-score",
                "per_record_zscore": "per-record z-score",
            }
        )
        + " | "
        + ordered["model_label"]
    )
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
    for panel, (metric, title) in enumerate(metrics):
        ax = axes.flat[panel]
        colors = [MODEL_COLORS[str(model)] for model in ordered["model"]]
        ax.scatter(ordered[metric], y, c=colors, s=30, edgecolor="black", linewidth=0.35)
        ax.set_xlim(0.55 if "balanced" in metric or "f1" in metric else 0.85, 1.01)
        ax.set_xlabel(title)
        ax.set_yticks(y)
        if panel % 2 == 0:
            ax.set_yticklabels(ordered["configuration"])
        else:
            ax.set_yticklabels([])
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.5)
        _panel_label(ax, chr(ord("A") + panel))
    _export_figure(fig, output_base, dpi=dpi)
    return output_base


def _paired_model_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    *,
    ylabel: str,
    ylim: tuple[float, float],
) -> None:
    pivot = frame.pivot(index="test_animal", columns="model", values=metric).reindex(ANIMAL_ORDER)
    if pivot[list(FORMAL_MODELS)].isna().any().any():
        raise ValueError(f"paired metric {metric} is incomplete")
    for _, values in pivot.iterrows():
        ax.plot((0, 1), values[list(FORMAL_MODELS)], color="#B8B8B8", linewidth=0.7, zorder=1)
    right_values = pivot[FORMAL_MODELS[1]].to_numpy(dtype=float)
    adjusted = _spread_label_positions(right_values, ylim)
    for animal, actual, label_y in zip(pivot.index, right_values, adjusted, strict=True):
        ax.plot(
            (1.0, 1.025),
            (actual, label_y),
            color="#777777",
            linewidth=0.45,
            clip_on=False,
        )
        ax.text(1.035, label_y, animal, fontsize=5.5, va="center", clip_on=False)
    for x, model in enumerate(FORMAL_MODELS):
        values = pivot[model].to_numpy(dtype=float)
        ax.scatter(
            np.full(len(values), x),
            values,
            color=MODEL_COLORS[model],
            edgecolor="black",
            linewidth=0.35,
            s=24,
            zorder=3,
        )
        ax.errorbar(
            x,
            values.mean(),
            yerr=values.std(ddof=1),
            fmt="D",
            color="black",
            markerfacecolor="white",
            markersize=4,
            capsize=3,
            linewidth=0.9,
            zorder=4,
        )
    ax.set_xticks((0, 1), [MODEL_LABELS[model] for model in FORMAL_MODELS])
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.10, 1.13)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)


def _spread_label_positions(
    values: np.ndarray,
    ylim: tuple[float, float],
) -> np.ndarray:
    """Spread clustered endpoint labels while preserving their rank."""

    values = np.asarray(values, dtype=float)
    lower, upper = map(float, ylim)
    span = upper - lower
    separation = 0.035 * span
    margin = 0.02 * span
    order = np.argsort(values)
    placed = values[order].copy()
    placed[0] = max(placed[0], lower + margin)
    for index in range(1, len(placed)):
        placed[index] = max(placed[index], placed[index - 1] + separation)
    if placed[-1] > upper - margin:
        placed -= placed[-1] - (upper - margin)
    for index in range(len(placed) - 2, -1, -1):
        placed[index] = min(placed[index], placed[index + 1] - separation)
    if placed[0] < lower + margin:
        placed += lower + margin - placed[0]
    result = np.empty_like(placed)
    result[order] = placed
    return result


def _animal_series_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    *,
    ylabel: str,
    ylim: tuple[float, float],
) -> None:
    x = np.arange(len(ANIMAL_ORDER))
    for model, marker, style in zip(FORMAL_MODELS, ("o", "s"), ("-", "--"), strict=True):
        subset = frame.loc[frame["model"].astype(str).eq(model)].set_index("test_animal").reindex(ANIMAL_ORDER)
        ax.plot(
            x,
            subset[metric],
            color=MODEL_COLORS[model],
            marker=marker,
            linestyle=style,
            linewidth=1.0,
            markersize=3.5,
            label=MODEL_LABELS[model],
        )
    ax.set_xticks(x, ANIMAL_ORDER, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    ax.legend(frameon=False, fontsize=6)


def _render_sci_report(
    *,
    split_design: pd.DataFrame,
    split_summary: pd.DataFrame,
    model_configuration: pd.DataFrame,
    training_diagnostics: pd.DataFrame,
    per_animal: pd.DataFrame,
    fold_summary: pd.DataFrame,
    pooled: pd.DataFrame,
    bonn: pd.DataFrame,
    aitrics: pd.DataFrame,
) -> str:
    total_windows = int(split_summary.loc[split_summary["fold"].eq(1), "n_windows"].sum())
    total_positive = int(split_summary.loc[split_summary["fold"].eq(1), "positive_windows"].sum())
    lines: list[str] = [
        "# 小鼠癫痫检测：SCI 论文级完整实验结果",
        "",
        "## 摘要式结论",
        "",
        "本报告汇总两种正式窗口级深度学习基线在同一冻结动物级 LOSO 协议下的全部 12 个训练折。"
        "EEGWaveNet 的动物折均值 Balanced Accuracy、AUROC 和 AUPRC 分别为 "
        f"{_summary_value(fold_summary, 'eegwavenet_paper_exact', 'balanced_accuracy')}、"
        f"{_summary_value(fold_summary, 'eegwavenet_paper_exact', 'auroc')} 和 "
        f"{_summary_value(fold_summary, 'eegwavenet_paper_exact', 'auprc')}；"
        "EEGNet-8,2 对应为 "
        f"{_summary_value(fold_summary, 'eegnet_8_2_paper_exact', 'balanced_accuracy')}、"
        f"{_summary_value(fold_summary, 'eegnet_8_2_paper_exact', 'auroc')} 和 "
        f"{_summary_value(fold_summary, 'eegnet_8_2_paper_exact', 'auprc')}。"
        "两模型均表现出显著的动物间波动和不稳定阈值；事件误报率仍过高，因此这些结果是工程基线和负结果证据，不是临床可用性能。",
        "",
        "## 1. 研究设计与统计单位",
        "",
        f"- 生物学样本量为 6 只小鼠（{', '.join(ANIMAL_ORDER)}），共 8 个 recording segment。",
        "- 所有记录均来自 2026-07-01 同一天急性给药后采集；本实验只检验同日条件下对未见动物的迁移，没有跨日期或常规慢性场景证据。",
        f"- 全数据包含 {total_windows:,} 个有效 4 秒窗口，其中 {total_positive:,} 个阳性窗口（{100 * total_positive / total_windows:.2f}%）；窗口步长为 1 秒，因此窗口彼此相关，不能把窗口数当作独立样本量。",
        "- 每折使用 4 只鼠训练、1 只鼠验证、1 只鼠测试。同一动物的所有片段始终处于同一集合。验证鼠负责 checkpoint 与阈值选择；测试鼠不参与任何选择。",
        "",
        "![Figure 1](figures/figure1_protocol_and_data_distribution.png)",
        "",
        "### 1.1 冻结六折划分",
        "",
        _markdown_table(
            split_design,
            (
                ("fold", "折", "int"),
                ("train_animals", "训练动物", "text"),
                ("validation_animal", "验证动物", "text"),
                ("test_animal", "测试动物", "text"),
                ("train_windows", "训练窗", "integer"),
                ("validation_windows", "验证窗", "integer"),
                ("test_windows", "测试窗", "integer"),
                ("train_positive_fraction", "训练阳性率", "percent"),
                ("validation_positive_fraction", "验证阳性率", "percent"),
                ("test_positive_fraction", "测试阳性率", "percent"),
            ),
        ),
        "",
        "## 2. 数据处理、模型与训练协议",
        "",
        "原始 CH7/CH8 约 1000 Hz 信号经抗混叠重采样到 250 Hz；生成 4 秒、1000 点窗口，步长 1 秒。任一 closed 通道事件与窗口重叠至少 0.5 秒定义为阳性；0 到 0.5 秒的过渡窗排除。每窗每通道独立执行 median/MAD 标准化并裁剪到 [-20, 20]。两个正式模型均把 2x1000 输入插值到 2x1024 后进入论文拓扑。",
        "",
        _markdown_table(
            model_configuration,
            (
                ("model_label", "模型", "text"),
                ("parameters", "参数量", "integer"),
                ("architecture", "核心结构", "text"),
                ("epochs", "Epoch", "int"),
                ("batch_size", "Batch", "int"),
                ("learning_rate", "学习率", "decimal4"),
                ("optimizer", "优化器", "text"),
                ("device", "设备", "text"),
            ),
        ),
        "",
        "训练使用 AdamW（weight decay=1e-4）、30 epochs、batch size=256、seed=20260804 和 CUDA。训练数据同时使用逆频率 WeightedRandomSampler 与训练集计数导出的平方根类权重交叉熵；这构成双重类别平衡，是当前高误报的重要候选原因。checkpoint 以验证加权交叉熵最低选择，阈值仅在验证鼠上从 0.10 到 0.90（步长 0.01）最大化窗口 F1。",
        "",
        "## 3. 训练收敛与验证曲线",
        "",
        "![Figure 2](figures/figure2_eegwavenet_loss_curves.png)",
        "",
        "![Figure 3](figures/figure3_eegnet_loss_curves.png)",
        "",
        "12 个折中多数最佳验证 epoch 出现在训练早期。注意训练 loss 在类别平衡采样分布上计算，而验证 loss 在自然类别分布上计算；因此二者绝对值不能被视为同分布估计。持续下降的训练 loss 与上升/波动的验证 loss 仍与过拟合和动物域偏移一致，但不能仅凭 loss gap 区分两者。",
        "",
        _markdown_table(
            training_diagnostics,
            (
                ("model_label", "模型", "text"),
                ("fold", "折", "int"),
                ("test_animal", "测试鼠", "text"),
                ("best_epoch", "最佳 epoch", "int"),
                ("train_loss_at_best", "最佳点训练 loss", "decimal3"),
                ("best_val_loss", "最佳验证 loss", "decimal3"),
                ("final_train_loss", "最终训练 loss", "decimal3"),
                ("final_val_loss", "最终验证 loss", "decimal3"),
                ("final_generalization_gap", "最终 gap", "decimal3"),
            ),
        ),
        "",
        "## 4. 未见动物测试性能",
        "",
        "### 4.1 动物折均值与跨动物波动",
        "",
        _fold_summary_markdown(fold_summary),
        "",
        "EEGWaveNet 的 Accuracy、Specificity、F1、Balanced Accuracy、MCC、AUROC、AUPRC 和事件敏感度描述性均值高于 EEGNet-8,2；EEGNet-8,2 的窗口 Sensitivity 更高且误报/小时更低。两者存在明确取舍，且各指标折间标准差较大。本数据只有 6 个生物学单位，也没有预注册推断检验，因此不报告模型优越性的 p 值，不声称统计学显著或存在单一总体胜者。",
        "",
        "![Figure 4](figures/figure4_animal_level_performance.png)",
        "",
        "### 4.2 每只测试动物的全部正式结果",
        "",
        _markdown_table(
            per_animal.sort_values(["model", "fold"]),
            (
                ("model", "模型", "model"),
                ("fold", "折", "int"),
                ("test_animal", "测试鼠", "text"),
                ("threshold_selected_on_validation", "阈值", "decimal2"),
                ("accuracy", "Acc", "decimal3"),
                ("sensitivity", "Sens", "decimal3"),
                ("specificity", "Spec", "decimal3"),
                ("f1", "F1", "decimal3"),
                ("balanced_accuracy", "Bal Acc", "decimal3"),
                ("mcc", "MCC", "decimal3"),
                ("auroc", "AUROC", "decimal3"),
                ("auprc", "AUPRC", "decimal3"),
                ("event_sensitivity", "事件敏感度", "decimal3"),
                ("false_alarms_per_hour", "误报/小时", "decimal1"),
            ),
        ),
        "",
        "### 4.3 Pooled 描述性结果与混淆矩阵",
        "",
        _markdown_table(
            pooled,
            (
                ("model", "模型", "model"),
                ("n_windows", "窗数", "integer"),
                ("accuracy", "Acc", "decimal3"),
                ("sensitivity", "Sens", "decimal3"),
                ("specificity", "Spec", "decimal3"),
                ("f1", "F1", "decimal3"),
                ("balanced_accuracy", "Bal Acc", "decimal3"),
                ("mcc", "MCC", "decimal3"),
                ("auroc", "AUROC", "decimal3"),
                ("auprc", "AUPRC", "decimal3"),
                ("event_sensitivity", "事件敏感度", "decimal3"),
                ("false_alarms_per_hour", "误报/小时", "decimal1"),
            ),
        ),
        "",
        "pooled 离散指标保留每折各自在验证鼠上选择的阈值。不同折的概率未做共同校准，因此 pooled AUROC/AUPRC 仅作描述，模型主结论应依据动物折均值与逐动物结果。",
        "",
        "![Figure 5](figures/figure5_confusion_and_event_performance.png)",
        "",
        "![Figure 6](figures/figure6_test_roc_pr_curves.png)",
        "",
        "![Figure S1](figures/figureS1_threshold_and_convergence.png)",
        "",
        "## 5. 事件级解释",
        "",
        "EEGWaveNet 的动物折平均事件敏感度为 "
        f"{_summary_value(fold_summary, 'eegwavenet_paper_exact', 'event_sensitivity')}，"
        "但平均误报为 "
        f"{_summary_value(fold_summary, 'eegwavenet_paper_exact', 'false_alarms_per_hour')} 次/小时；"
        "EEGNet-8,2 对应为 "
        f"{_summary_value(fold_summary, 'eegnet_8_2_paper_exact', 'event_sensitivity')} 和 "
        f"{_summary_value(fold_summary, 'eegnet_8_2_paper_exact', 'false_alarms_per_hour')} 次/小时。"
        "因此即使某些动物窗口 AUROC 较高，现有逐窗阈值和事件合并仍不能形成可用报警系统。",
        "",
        "## 6. 其他已训练结果：与主结果严格分开",
        "",
        "### 6.1 Bonn 公共数据工程验证",
        "",
        "Bonn 结果用于验证模型实现和训练链路，数据简单、片段级且不等价于本小鼠跨动物任务，也不等价于 EEGWaveNet/EEGNet 原论文性能复现。所有 aggregate JSON 均明确记录 `public_dataset_engineering_validation` 与 `paper_performance_claim_allowed=false`。",
        "",
        _bonn_markdown(bonn),
        "",
        "![Figure S2](figures/figureS2_bonn_engineering_validation.png)" if not bonn.empty else "未发现 Bonn 正式工程验证产物。",
        "",
        "### 6.2 AITRICS stateful 冒烟测试",
        "",
        _aitrics_markdown(aitrics),
        "",
        "AITRICS 仅运行 1 epoch、8 个训练 sequence、4 个验证 sequence 和 4 个测试 sequence；它只证明状态化数据流、CUDA 前反向和产物保存可运行。该结果不进入正式六折比较，也不能用于模型排名。",
        "",
        "## 7. 数据质量与结论边界",
        "",
        "1. 当前 6 只鼠全部来自同日急性给药，不能代表跨日期、慢性癫痫或下一批动物。",
        "2. E77 的 CH7 已审计为硬件轨/平坦通道，但当前正式模型仍按双通道输入；缺少通道质量 mask。",
        "3. 5 个 session 的标签文件没有机器可读的完整复核范围；未复核阴性与已复核阴性尚未实现三态区分。",
        "4. 每窗 median/MAD 标准化删除绝对幅值和慢变基线，可能损失急性发作的重要信息。",
        "5. 训练同时使用平衡采样和类权重，可能推高误报；下一轮应只保留一种不平衡修正并在内层验证冻结。",
        "6. 单一验证鼠同时决定 checkpoint 与阈值，阳性率跨鼠从 5.29% 到 49.50%，导致阈值显著不稳定。",
        "7. 重叠窗口非独立，真正的统计单位是动物；当前 n=6 只适合探索性基线和工程验证。",
        "",
        "允许的结论是：当前工程链路已在冻结动物级 LOSO 下完整运行；EEGWaveNet 在若干区分度指标上具有较高描述性均值，EEGNet-8,2 则具有较高窗口敏感度和较低事件误报，但两者跨动物稳定性与报警性能均未达到可用要求。禁止声称：已达到临床水平、已证明跨日期泛化、已复现原论文数值、存在总体最优模型、或 EEGWaveNet 在统计学上显著优于 EEGNet。",
        "",
        "## 8. 复现入口与源数据",
        "",
        "- 正式训练队列：`scripts/run_mouse_epilepsy_loso.py`",
        "- SCI 报告生成：`scripts/generate_mouse_epilepsy_sci_report.py`",
        "- 冻结划分：`data/epilepsy_splits_exact/loso_fold*.csv`",
        "- 训练产物：`runs/epilepsy_exact_loso_adapted/<model>/<fold>/artifacts/<run>/`",
        "- 本报告源数据：`sci_report/tables/`",
        "- 矢量图：`sci_report/figures/*.pdf` 与 `*.svg`；600 DPI 图：`*.png`。",
        "",
        "## 9. 图注",
        "",
        "**Figure 1.** 动物级六折设计与每只动物的数据分布。A，灰色、蓝色和橙色分别表示训练、验证和测试角色；B，非发作/发作窗口堆叠计数及阳性比例。",
        "",
        "**Figure 2.** EEGWaveNet 六折训练与验证加权交叉熵。纵轴为对数尺度；星号表示最低验证 loss checkpoint。",
        "",
        "**Figure 3.** EEGNet-8,2 六折训练与验证加权交叉熵，定义同 Figure 2。",
        "",
        "**Figure 4.** 未见动物测试的成对模型比较。每条灰线连接同一测试动物，黑色空心菱形与误差线表示 n=6 动物的均值±样本标准差。",
        "",
        "**Figure 5.** Pooled 混淆矩阵与事件级性能。混淆矩阵同时标注窗口计数和按真实类别归一化比例；下排为逐动物事件敏感度和误报/小时。",
        "",
        "**Figure 6.** 每个独立测试动物的 ROC 与 precision-recall 曲线。每个折在不同动物上训练/验证后测试；PR 虚线为六个测试动物阳性率的算术均值，动物特异基线因阳性率不同而变化。",
        "",
        "**Figure S1.** 验证集选择阈值、最佳 epoch 和最终验证-训练 loss gap 的动物间变化。",
        "",
        "**Figure S2.** Bonn 公共数据工程验证的配置级结果，仅用于实现链路验证，不构成原论文数值复现。",
        "",
    ]
    return "\n".join(lines)


def _fold_summary_markdown(frame: pd.DataFrame) -> str:
    headers = (
        "模型",
        "Acc",
        "Sens",
        "Spec",
        "F1",
        "Bal Acc",
        "MCC",
        "AUROC",
        "AUPRC",
        "事件敏感度",
        "误报/小时",
    )
    rows: list[list[str]] = []
    for model in FORMAL_MODELS:
        item = frame.loc[frame["model"].astype(str).eq(model)].iloc[0]
        rows.append(
            [
                MODEL_LABELS[model],
                *[
                    _mean_std(item, metric)
                    for metric in (
                        "accuracy",
                        "sensitivity",
                        "specificity",
                        "f1",
                        "balanced_accuracy",
                        "mcc",
                        "auroc",
                        "auprc",
                        "event_sensitivity",
                        "false_alarms_per_hour",
                    )
                ],
            ]
        )
    return _markdown_rows(headers, rows)


def _bonn_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "未发现 Bonn 工程验证产物。"
    return _markdown_table(
        frame,
        (
            ("model_label", "模型", "text"),
            ("protocol", "协议", "text"),
            ("normalization", "归一化", "text"),
            ("accuracy_mean", "Acc", "decimal3"),
            ("balanced_accuracy_mean", "Bal Acc", "decimal3"),
            ("macro_f1_mean", "Macro F1", "decimal3"),
            ("auroc_mean", "AUROC", "decimal3"),
            ("auprc_mean", "AUPRC", "decimal3"),
            ("seizure_recall_mean", "发作召回", "decimal3"),
        ),
    )


def _aitrics_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "未发现 AITRICS stateful smoke 产物。"
    return _markdown_table(
        frame,
        (
            ("epochs", "Epoch", "int"),
            ("train_sequences", "训练 sequence", "int"),
            ("validation_sequences", "验证 sequence", "int"),
            ("test_sequences", "测试 sequence", "int"),
            ("train_loss", "训练 loss", "decimal3"),
            ("validation_loss", "验证 loss", "decimal3"),
            ("test_accuracy", "测试 Acc", "decimal3"),
            ("test_sensitivity", "测试 Sens", "decimal3"),
            ("test_specificity", "测试 Spec", "decimal3"),
            ("test_auroc", "测试 AUROC", "decimal3"),
            ("test_auprc", "测试 AUPRC", "decimal3"),
        ),
    )


def _markdown_table(
    frame: pd.DataFrame,
    columns: Sequence[tuple[str, str, str]],
) -> str:
    rows: list[list[str]] = []
    for item in frame.to_dict(orient="records"):
        rows.append([_format_cell(item.get(key), kind) for key, _, kind in columns])
    return _markdown_rows([label for _, label, _ in columns], rows)


def _markdown_rows(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(map(_escape_markdown, row)) + " |" for row in rows]
    return "\n".join((header, separator, *body))


def _format_cell(value: Any, kind: str) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    if kind == "model":
        return MODEL_LABELS.get(str(value), str(value))
    if kind == "text":
        return str(value)
    if kind in {"int", "integer"}:
        return f"{int(value):,}"
    if kind == "percent":
        return f"{100 * float(value):.2f}%"
    if kind == "decimal1":
        return f"{float(value):.1f}"
    if kind == "decimal2":
        return f"{float(value):.2f}"
    if kind == "decimal3":
        return f"{float(value):.3f}"
    if kind == "decimal4":
        return f"{float(value):.4f}"
    return str(value)


def _summary_value(frame: pd.DataFrame, model: str, metric: str) -> str:
    row = frame.loc[frame["model"].astype(str).eq(model)].iloc[0]
    return _mean_std(row, metric)


def _mean_std(row: Mapping[str, Any], metric: str) -> str:
    mean = float(row[f"{metric}_mean"])
    std = float(row[f"{metric}_std"])
    precision = 1 if metric == "false_alarms_per_hour" else 3
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def _validate_formal_inventory(
    inventory: pd.DataFrame,
    per_animal: pd.DataFrame,
    fold_summary: pd.DataFrame,
) -> None:
    expected = {(model, fold) for model in FORMAL_MODELS for fold in range(1, 7)}
    observed = {
        (str(row.model), int(row.fold))
        for row in inventory.itertuples(index=False)
        if str(row.launcher_status) == "completed" and str(row.artifact_state) == "complete"
    }
    if observed != expected:
        raise ValueError(
            f"formal inventory is incomplete; missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    if len(per_animal) != 12:
        raise ValueError("per_animal_metrics.csv must contain 12 formal rows")
    for model in FORMAL_MODELS:
        summary = fold_summary.loc[fold_summary["model"].astype(str).eq(model)]
        if len(summary) != 1 or int(summary.iloc[0]["completed_folds"]) != 6:
            raise ValueError(f"fold summary is incomplete for {model}")


def _set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.1,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _export_figure(fig: plt.Figure, output_base: Path, *, dpi: int) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _verify_figure_outputs(paths: Sequence[Path]) -> None:
    for path in paths:
        if not path.is_file() or path.stat().st_size < 1000:
            raise ValueError(f"figure output is missing or unexpectedly small: {path}")
        if path.suffix.lower() == ".png":
            array = plt.imread(path)
            if array.size == 0 or not np.isfinite(array).all():
                raise ValueError(f"invalid PNG figure: {path}")
            if float(np.nanstd(array)) < 1e-4:
                raise ValueError(f"blank PNG figure: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"top-level JSON must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nested_number(value: Any, outer: str, inner: str) -> float:
    if not isinstance(value, Mapping):
        return np.nan
    item = value.get(outer)
    return _mapping_number(item, inner)


def _mapping_number(value: Any, key: str) -> float:
    if not isinstance(value, Mapping):
        return np.nan
    raw = value.get(key)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return np.nan


def _escape_markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
