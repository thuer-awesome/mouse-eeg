"""Deterministic public-data training harness for seizure model replication."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# CUDA requires this setting before its first cuBLAS operation for seeded,
# deterministic matrix multiplication.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mouse_sleep.epilepsy.replication.bonn import (
    BONN_SAMPLES,
    compute_train_stats,
    load_bonn_split,
)
from mouse_sleep.epilepsy.replication.eegnet_exact import EEGNet82
from mouse_sleep.epilepsy.replication.eegwavenet import EEGWaveNetPaperExact


MODEL_ALIASES = {
    "eegwavenet": "eegwavenet_paper_exact",
    "eegwavenet_paper_exact": "eegwavenet_paper_exact",
    "eegnet": "eegnet_8_2_paper_exact",
    "eegnet82": "eegnet_8_2_paper_exact",
    "eegnet_8_2_paper_exact": "eegnet_8_2_paper_exact",
}


def _sha256(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def build_replication_model(
    name: str, *, channels: int, samples: int, classes: int
) -> tuple[nn.Module, bool, str]:
    """Return model, whether it emits log-probabilities, and canonical name."""

    key = MODEL_ALIASES.get(name.lower().replace("-", "_"))
    if key == "eegwavenet_paper_exact":
        return EEGWaveNetPaperExact(channels, classes), True, key
    if key == "eegnet_8_2_paper_exact":
        return EEGNet82(channels, samples, classes), False, key
    raise ValueError(f"Unknown replication model {name!r}; choose from {sorted(MODEL_ALIASES)}")


def _loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=pin_memory,
    )


def _probabilities(output: torch.Tensor, log_probabilities: bool) -> torch.Tensor:
    return output.exp() if log_probabilities else output.softmax(dim=1)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    log_probabilities: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            output = model(batch_x)
            losses.append(float(criterion(output, batch_y).detach().cpu()))
            labels.append(batch_y.cpu().numpy())
            probabilities.append(_probabilities(output, log_probabilities).cpu().numpy())
    return float(np.mean(losses)), np.concatenate(labels), np.concatenate(probabilities)


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Compute class-wise and aggregate metrics without hiding class failures."""

    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if threshold is not None:
        if probabilities.shape[1] != 2:
            raise ValueError("A scalar decision threshold is only valid for binary tasks")
        y_pred = (probabilities[:, 1] >= threshold).astype(np.int64)
    else:
        y_pred = probabilities.argmax(axis=1)
    labels = np.arange(probabilities.shape[1])
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {}
    total = matrix.sum()
    for index, label in enumerate(labels):
        tp = matrix[index, index]
        fn = matrix[index].sum() - tp
        fp = matrix[:, index].sum() - tp
        tn = total - tp - fn - fp
        per_class[str(label)] = {
            "precision": float(precision[index]),
            "recall_sensitivity": float(recall[index]),
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }
    if threshold is not None:
        metrics["decision_threshold"] = float(threshold)
    if probabilities.shape[1] == 2 and len(np.unique(y_true)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_true, probabilities[:, 1]))
        metrics["auprc"] = float(average_precision_score(y_true, probabilities[:, 1]))
    elif len(np.unique(y_true)) == probabilities.shape[1]:
        metrics["macro_ovr_auroc"] = float(
            roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
        )
    return metrics


def select_validation_youden_threshold(
    y_true: np.ndarray, probabilities: np.ndarray
) -> float:
    """Select a binary threshold on validation data only via Youden's J."""

    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Youden threshold selection requires binary probabilities")
    if len(np.unique(y_true)) != 2:
        raise ValueError("Validation labels must contain both binary classes")
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_true, probabilities[:, 1]
    )
    finite = np.isfinite(thresholds)
    if not finite.any():
        raise ValueError("No finite ROC threshold was available")
    scores = true_positive_rate[finite] - false_positive_rate[finite]
    candidates = thresholds[finite]
    best_score = scores.max()
    tied = candidates[np.isclose(scores, best_score, rtol=0.0, atol=1e-12)]
    # A tie is resolved toward the threshold closest to the conventional 0.5.
    return float(tied[np.argmin(np.abs(tied - 0.5))])


def train_bonn_fold(
    dataset_root: str | Path,
    split_manifest: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "eegwavenet",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 10,
    normalization: str = "none",
    seed: int = 20260804,
    device: str = "auto",
) -> Path:
    """Train one frozen Bonn fold and save a complete audit bundle."""

    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")
    _set_seed(seed)
    resolved_device = _resolve_device(device)
    split_path = Path(split_manifest).resolve()
    split_frame = pd.read_csv(split_path)
    if set(split_frame["split"].unique()) != {"train", "val", "test"}:
        raise ValueError("A replication fold must contain train, val, and test")

    train_stats = None
    if normalization == "global_train_zscore":
        train_stats = compute_train_stats(split_frame, dataset_root)
    arrays = {}
    for split_name in ("train", "val", "test"):
        arrays[split_name] = load_bonn_split(
            split_frame,
            dataset_root,
            split_name,
            normalization=normalization,
            train_stats=train_stats,
        )
    x_train, y_train, train_meta, _ = arrays["train"]
    x_val, y_val, val_meta, _ = arrays["val"]
    x_test, y_test, test_meta, _ = arrays["test"]
    classes = int(split_frame["label"].max()) + 1

    model, log_probabilities, canonical_name = build_replication_model(
        model_name,
        channels=x_train.shape[1],
        samples=x_train.shape[2],
        classes=classes,
    )
    model.to(resolved_device)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(classes), y=y_train
    )
    weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=resolved_device)
    criterion: nn.Module = (
        nn.NLLLoss(weight=weight_tensor)
        if log_probabilities
        else nn.CrossEntropyLoss(weight=weight_tensor)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = _loader(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        pin_memory=resolved_device.type == "cuda",
    )
    val_loader = _loader(
        x_val,
        y_val,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        pin_memory=resolved_device.type == "cuda",
    )
    test_loader = _loader(
        x_test,
        y_test,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        pin_memory=resolved_device.type == "cuda",
    )

    history: list[dict[str, float | int]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    wait = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(resolved_device, non_blocking=True)
            batch_y = batch_y.to(resolved_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            if isinstance(model, EEGNet82):
                model.apply_max_norm_constraints()
            train_losses.append(float(loss.detach().cpu()))
        val_loss, _, val_probabilities = _evaluate(
            model, val_loader, resolved_device, criterion, log_probabilities
        )
        val_metrics = classification_metrics(y_val, val_probabilities)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    val_loss, val_true, val_probabilities = _evaluate(
        model, val_loader, resolved_device, criterion, log_probabilities
    )
    test_loss, test_true, test_probabilities = _evaluate(
        model, test_loader, resolved_device, criterion, log_probabilities
    )
    metrics = {
        "validation": {"loss": val_loss, **classification_metrics(val_true, val_probabilities)},
        "test": {"loss": test_loss, **classification_metrics(test_true, test_probabilities)},
    }
    validation_threshold = None
    if classes == 2:
        validation_threshold = select_validation_youden_threshold(
            val_true, val_probabilities
        )
        metrics["validation_calibrated"] = classification_metrics(
            val_true, val_probabilities, threshold=validation_threshold
        )
        metrics["test_calibrated"] = classification_metrics(
            test_true, test_probabilities, threshold=validation_threshold
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        Path(output_dir).resolve()
        / f"{timestamp}_{split_path.stem}_{canonical_name}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), run_dir / "model.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    predictions = test_meta[["path", "recording_id", "set_code", "label"]].copy()
    predictions["prediction"] = test_probabilities.argmax(axis=1)
    if validation_threshold is not None:
        predictions["validation_selected_threshold"] = validation_threshold
        predictions["calibrated_prediction"] = (
            test_probabilities[:, 1] >= validation_threshold
        ).astype(np.int64)
    for class_index in range(classes):
        predictions[f"probability_{class_index}"] = test_probabilities[:, class_index]
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    run_manifest = {
        "created_utc": timestamp,
        "model": canonical_name,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "dataset": "Bonn EEG",
        "dataset_root": str(Path(dataset_root).resolve()),
        "split_manifest": str(split_path),
        "split_manifest_sha256": _sha256(split_path),
        "split_counts": {
            "train": int(len(train_meta)),
            "val": int(len(val_meta)),
            "test": int(len(test_meta)),
        },
        "classes": classes,
        "source_samples": BONN_SAMPLES,
        "normalization": normalization,
        "normalization_train_stats": train_stats,
        "validation_selected_threshold": validation_threshold,
        "threshold_selection": (
            "validation_youden_j" if validation_threshold is not None else None
        ),
        "class_weights": class_weights.tolist(),
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "requested_epochs": epochs,
        "completed_epochs": len(history),
        "early_stopping_patience": patience,
        "seed": seed,
        "device": str(resolved_device),
        "torch_version": torch.__version__,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "replication_status": "public_dataset_engineering_validation",
        "paper_performance_claim_allowed": False,
        "claim_blockers": [
            "Bonn source files do not expose patient IDs.",
            "The closed EEGWaveNet paper protocol and target table have not been fully recovered.",
        ],
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return run_dir


def train_bonn_cross_validation(
    dataset_root: str | Path,
    folds_dir: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "eegwavenet",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 10,
    normalization: str = "none",
    seed: int = 20260804,
    device: str = "auto",
) -> Path:
    """Run every frozen fold sequentially and aggregate test metrics."""

    fold_paths = sorted(Path(folds_dir).glob("fold_[0-9][0-9].csv"))
    if not fold_paths:
        raise FileNotFoundError(f"No fold_XX.csv files in {folds_dir}")
    canonical = MODEL_ALIASES.get(model_name.lower().replace("-", "_"))
    if canonical is None:
        raise ValueError(f"Unknown replication model {model_name!r}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cv_dir = (
        Path(output_dir).resolve()
        / f"{timestamp}_{canonical}_{normalization}_cv"
    )
    cv_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    run_dirs: list[str] = []
    for fold_index, fold_path in enumerate(fold_paths, start=1):
        run_dir = train_bonn_fold(
            dataset_root,
            fold_path,
            cv_dir / "fold_runs",
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            normalization=normalization,
            seed=seed + fold_index - 1,
            device=device,
        )
        run_dirs.append(str(run_dir))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        row: dict[str, Any] = {"fold": fold_index, "run_dir": str(run_dir)}
        for split_name in (
            "validation",
            "test",
            "validation_calibrated",
            "test_calibrated",
        ):
            if split_name not in metrics:
                continue
            for metric_name in (
                "loss",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "auroc",
                "auprc",
                "macro_ovr_auroc",
                "decision_threshold",
            ):
                if metric_name in metrics[split_name]:
                    row[f"{split_name}_{metric_name}"] = metrics[split_name][metric_name]
            for class_name, values in metrics[split_name]["per_class"].items():
                row[f"{split_name}_recall_class_{class_name}"] = values[
                    "recall_sensitivity"
                ]
                row[f"{split_name}_specificity_class_{class_name}"] = values[
                    "specificity"
                ]
        rows.append(row)

    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(cv_dir / "fold_metrics.csv", index=False)
    aggregate: dict[str, dict[str, float]] = {}
    for column in fold_metrics.select_dtypes(include=[np.number]).columns:
        if column == "fold":
            continue
        values = fold_metrics[column].dropna().to_numpy(dtype=float)
        aggregate[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "n": int(len(values)),
        }
    cv_manifest = {
        "created_utc": timestamp,
        "model": canonical,
        "normalization": normalization,
        "folds_dir": str(Path(folds_dir).resolve()),
        "folds": len(fold_paths),
        "base_seed": seed,
        "run_dirs": run_dirs,
        "aggregate_metrics": aggregate,
        "replication_status": "public_dataset_engineering_validation",
        "paper_performance_claim_allowed": False,
    }
    (cv_dir / "aggregate_metrics.json").write_text(
        json.dumps(cv_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return cv_dir
