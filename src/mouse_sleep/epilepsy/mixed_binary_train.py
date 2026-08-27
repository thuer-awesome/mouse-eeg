from __future__ import annotations

import hashlib
import json
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from mouse_sleep.epilepsy.metrics import window_metrics
from mouse_sleep.epilepsy.mixed_binary import (
    MixedBinaryDataset,
    audit_mixed_binary_split,
)
from mouse_sleep.epilepsy.models import build_epilepsy_model
from mouse_sleep.epilepsy.seizure_physio import SeizurePhysioNet


@dataclass(frozen=True)
class MixedBinaryTrainingConfig:
    """Common protocol for pooled-mouse binary architecture comparisons."""

    model_name: str = "seizure_physio"
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    auxiliary_statistics_weight: float = 0.0
    class_weight_mode: str = "balanced"
    selection_metric: str = "auprc"
    threshold_metric: str = "mcc"
    early_stopping_patience: int = 8
    seed: int = 20260814
    device: str = "auto"
    mixed_precision: bool = True
    num_workers: int = 0
    normalization_clip: float = 20.0
    maximum_train_windows: int | None = None
    maximum_evaluation_windows: int | None = None

    def validated(self) -> "MixedBinaryTrainingConfig":
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if self.auxiliary_statistics_weight < 0:
            raise ValueError("auxiliary_statistics_weight cannot be negative")
        if self.class_weight_mode not in {
            "balanced",
            "balanced_mouse_sqrt",
            "none",
        }:
            raise ValueError(
                "class_weight_mode must be balanced, balanced_mouse_sqrt, or none"
            )
        if self.selection_metric not in {"auprc", "auroc", "val_loss"}:
            raise ValueError("selection_metric must be auprc, auroc, or val_loss")
        if self.threshold_metric not in {"mcc", "balanced_accuracy", "f1"}:
            raise ValueError("threshold_metric must be mcc, balanced_accuracy, or f1")
        if self.early_stopping_patience < 1 or self.num_workers < 0:
            raise ValueError("patience must be positive and num_workers nonnegative")
        if self.normalization_clip <= 0:
            raise ValueError("normalization_clip must be positive")
        for name, value in (
            ("maximum_train_windows", self.maximum_train_windows),
            ("maximum_evaluation_windows", self.maximum_evaluation_windows),
        ):
            if value is not None and value < 2:
                raise ValueError(f"{name} must be at least two")
        key = _model_key(self.model_name)
        if "aitrics" in key:
            raise ValueError(
                "Paper-faithful AITRICS requires complete ordered 26-window "
                "sequences and explicit recurrent-state boundaries; the mixed "
                "independent-window view cannot make that comparison."
            )
        if self.auxiliary_statistics_weight > 0 and key != "seizure_physio":
            raise ValueError(
                "The continuous-statistics auxiliary loss is only defined for "
                "SeizurePhysioNet; baseline architectures must remain unchanged."
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["model_kwargs"] = dict(self.model_kwargs)
        return values


def run_mixed_binary_training(
    metadata: str | Path,
    training_baseline: str | Path,
    output_dir: str | Path,
    config: MixedBinaryTrainingConfig | None = None,
) -> Path:
    """Train one model on the frozen pooled split and evaluate without test tuning."""

    config = (config or MixedBinaryTrainingConfig()).validated()
    metadata_path = Path(metadata).resolve()
    baseline_path = Path(training_baseline).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    signature = {
        "schema_version": 1,
        "metadata": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "training_baseline": str(baseline_path),
        "training_baseline_sha256": _sha256(baseline_path),
        "config": config.to_dict(),
    }
    signature_hash = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode("utf-8")
    ).hexdigest()
    completed_path = output / "completed.json"
    if completed_path.is_file():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if completed.get("signature_sha256") != signature_hash:
            raise ValueError(f"completed run has a different configuration: {output}")
        return output

    _seed_everything(config.seed)
    full_metadata = pd.read_parquet(metadata_path)
    audit_mixed_binary_split(full_metadata)
    datasets = {
        split: MixedBinaryDataset(
            full_metadata,
            baseline_path,
            split=split,
            maximum_windows=(
                config.maximum_train_windows
                if split == "train"
                else config.maximum_evaluation_windows
            ),
            normalization_clip=config.normalization_clip,
        )
        for split in ("train", "val", "test")
    }
    train_labels = datasets["train"].labels()
    if set(np.unique(train_labels).tolist()) != {0, 1}:
        raise ValueError("training data must contain both binary classes")

    device = _resolve_device(config.device)
    sample = datasets["train"][0]["signals"]
    model = _build_model(config, sample).to(device)
    class_weights, sample_weight_lookup = _loss_weighting(
        datasets["train"].frame, config.class_weight_mode, device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    amp_enabled = bool(config.mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loader_generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        split: torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            generator=loader_generator if split == "train" else None,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
        )
        for split, dataset in datasets.items()
    }

    progress_path = output / "progress.pt"
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    if progress_path.is_file():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("signature_sha256") != signature_hash:
            raise ValueError(f"progress checkpoint has a different configuration: {output}")
        model.load_state_dict(progress["model_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        _optimizer_to(optimizer, device)
        scaler.load_state_dict(progress["scaler_state"])
        loader_generator.set_state(progress["loader_generator_state"])
        history = list(progress["history"])
        best_state = progress["best_state"]
        best_score = float(progress["best_score"])
        best_epoch = int(progress["best_epoch"])
        stale_epochs = int(progress["stale_epochs"])
        start_epoch = int(progress["completed_epoch"]) + 1
        _restore_rng(progress, device)

    for epoch in range(start_epoch, config.epochs + 1):
        train_result = _run_loader(
            model,
            loaders["train"],
            device,
            class_weights,
            sample_weight_lookup,
            config.auxiliary_statistics_weight,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        validation_result = _run_loader(
            model,
            loaders["val"],
            device,
            class_weights,
            sample_weight_lookup,
            config.auxiliary_statistics_weight,
            amp_enabled=amp_enabled,
            collect_predictions=True,
        )
        validation_metrics = window_metrics(
            validation_result["predictions"]["label"].to_numpy(),
            validation_result["predictions"]["seizure_probability"].to_numpy(),
            threshold=0.5,
        )
        score = _selection_score(config.selection_metric, validation_result, validation_metrics)
        improved = score > best_score + 1e-10
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "train_classification_loss": train_result["classification_loss"],
                "train_statistics_loss": train_result["statistics_loss"],
                "validation_loss": validation_result["loss"],
                "validation_classification_loss": validation_result[
                    "classification_loss"
                ],
                "validation_statistics_loss": validation_result["statistics_loss"],
                "validation_auprc": validation_metrics["auprc"],
                "validation_auroc": validation_metrics["auroc"],
                "selection_score": score,
                "best": improved,
            }
        )
        _save_progress(
            progress_path,
            signature_hash,
            epoch,
            model,
            optimizer,
            scaler,
            loader_generator,
            history,
            best_state,
            best_score,
            best_epoch,
            stale_epochs,
        )
        print(
            f"[mixed-binary] model={config.model_name} epoch={epoch}/{config.epochs} "
            f"train={train_result['loss']:.5f} val={validation_result['loss']:.5f} "
            f"val_auprc={validation_metrics['auprc']:.5f}",
            flush=True,
        )
        if stale_epochs >= config.early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    predictions: dict[str, pd.DataFrame] = {}
    for split in ("val", "test"):
        result = _run_loader(
            model,
            loaders[split],
            device,
            class_weights,
            sample_weight_lookup,
            config.auxiliary_statistics_weight,
            amp_enabled=amp_enabled,
            collect_predictions=True,
        )
        predictions[split] = result["predictions"].assign(split=split)
    threshold = _select_validation_threshold(
        predictions["val"], config.threshold_metric
    )
    metrics = {
        "schema_version": 1,
        "evaluation_scope": "event_balanced_group_sampled_windows",
        "continuous_monitoring_metrics_valid": False,
        "continuous_monitoring_note": (
            "False alarms/hour and latency require inference on complete continuous "
            "held-out timelines; they are intentionally not estimated from capped groups."
        ),
        "checkpoint_selected_on": f"validation_{config.selection_metric}",
        "best_epoch": best_epoch,
        "threshold_selected_on": f"validation_{config.threshold_metric}",
        "threshold": threshold,
        "val": _evaluate_predictions(predictions["val"], threshold),
        "test": _evaluate_predictions(predictions["test"], threshold),
    }
    combined_predictions = pd.concat(
        [predictions["val"], predictions["test"]], ignore_index=True
    )
    predictions_path = output / "predictions.parquet"
    combined_predictions.to_parquet(predictions_path, index=False)
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    torch.save(
        {
            "model_name": config.model_name,
            "model_kwargs": dict(config.model_kwargs),
            "state_dict": best_state,
            "threshold": threshold,
            "statistics_auxiliary_weight": config.auxiliary_statistics_weight,
        },
        output / "model.pt",
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    manifest = {
        **signature,
        "signature_sha256": signature_hash,
        "device": str(device),
        "mixed_precision_effective": amp_enabled,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "sampling": "uniform_shuffle_without_weighted_sampler",
        "imbalance_correction": config.class_weight_mode,
        "class_counts": np.bincount(train_labels, minlength=2).astype(int).tolist(),
        "class_weights": (
            class_weights.detach().cpu().tolist() if class_weights is not None else None
        ),
        "mouse_class_sample_weights": [
            {"animal_id": animal, "label": label, "weight": weight}
            for (animal, label), weight in sorted(sample_weight_lookup.items())
        ],
        "prediction_sha256": _sha256(predictions_path),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    completed_path.write_text(
        json.dumps(
            {"signature_sha256": signature_hash, "best_epoch": best_epoch},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def _build_model(
    config: MixedBinaryTrainingConfig, sample: torch.Tensor
) -> nn.Module:
    key = _model_key(config.model_name)
    if key == "seizure_physio":
        kwargs = {
            "channels": int(sample.shape[0]),
            "classes": 2,
            "samples_per_window": int(sample.shape[-1]),
            "normalization_clip": config.normalization_clip,
            **dict(config.model_kwargs),
        }
        return SeizurePhysioNet(**kwargs)
    return build_epilepsy_model(
        config.model_name,
        channels=int(sample.shape[0]),
        classes=2,
        **dict(config.model_kwargs),
    )


def _run_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_weights: torch.Tensor | None,
    sample_weight_lookup: Mapping[tuple[str, int], float],
    auxiliary_weight: float,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    collect_predictions: bool = False,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_classification = 0.0
    total_statistics = 0.0
    seen = 0
    prediction_rows: list[dict[str, Any]] = []
    context = nullcontext() if training else torch.no_grad()
    with context:
        for batch in loader:
            signals = batch["signals"].to(device, non_blocking=True)
            masks = batch["channel_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = _forward(model, signals, masks)
                logits = outputs["logits"]
                classification_loss = _classification_loss(
                    logits,
                    labels,
                    batch["animal_id"],
                    class_weights,
                    sample_weight_lookup,
                )
                statistics_loss = _statistics_loss(outputs)
                loss = classification_loss + auxiliary_weight * statistics_loss
            if training:
                if scaler is None:
                    raise AssertionError("training requires a gradient scaler")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                constraint = getattr(model, "apply_max_norm_constraints", None)
                if constraint is not None:
                    constraint()
            batch_size = int(labels.shape[0])
            total_loss += float(loss.detach()) * batch_size
            total_classification += float(classification_loss.detach()) * batch_size
            total_statistics += float(statistics_loss.detach()) * batch_size
            seen += batch_size
            if collect_predictions:
                probabilities = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
                labels_cpu = labels.detach().cpu().numpy()
                for index in range(batch_size):
                    prediction_rows.append(
                        {
                            "animal_id": str(batch["animal_id"][index]),
                            "session_id": str(batch["session_id"][index]),
                            "group_id": str(batch["group_id"][index]),
                            "start_time_ms": float(batch["start_time_ms"][index]),
                            "end_time_ms": float(batch["end_time_ms"][index]),
                            "label": int(labels_cpu[index]),
                            "seizure_probability": float(probabilities[index]),
                        }
                    )
    result: dict[str, Any] = {
        "loss": total_loss / max(seen, 1),
        "classification_loss": total_classification / max(seen, 1),
        "statistics_loss": total_statistics / max(seen, 1),
    }
    if collect_predictions:
        result["predictions"] = pd.DataFrame(prediction_rows)
    return result


def _forward(
    model: nn.Module, signals: torch.Tensor, channel_mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    if isinstance(model, SeizurePhysioNet):
        return model(signals, channel_mask)
    logits = model(signals)
    return {"logits": logits}


def _statistics_loss(outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    logits = outputs["logits"]
    if "statistics_prediction" not in outputs:
        return logits.new_zeros(())
    prediction = outputs["statistics_prediction"]
    target = outputs["statistics_target"]
    mask = outputs.get("statistics_mask", torch.ones_like(target, dtype=torch.bool))
    elementwise = F.smooth_l1_loss(prediction, target, reduction="none")
    weights = mask.to(elementwise.dtype)
    return (elementwise * weights).sum() / weights.sum().clamp_min(1.0)


def _classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    animal_ids: list[str] | tuple[str, ...],
    class_weights: torch.Tensor | None,
    sample_weight_lookup: Mapping[tuple[str, int], float],
) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    if not sample_weight_lookup:
        if class_weights is not None:
            denominator = class_weights[labels].sum().clamp_min(1e-8)
            return losses.sum() / denominator
        return losses.mean()
    label_values = labels.detach().cpu().tolist()
    sample_weights = torch.as_tensor(
        [
            sample_weight_lookup[(str(animal), int(label))]
            for animal, label in zip(animal_ids, label_values, strict=True)
        ],
        dtype=losses.dtype,
        device=losses.device,
    )
    return (losses * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)


def _evaluate_predictions(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    overall = window_metrics(
        frame["label"].to_numpy(),
        frame["seizure_probability"].to_numpy(),
        threshold,
    )
    per_mouse = {
        str(animal): window_metrics(
            group["label"].to_numpy(),
            group["seizure_probability"].to_numpy(),
            threshold,
        )
        for animal, group in frame.groupby("animal_id", sort=True)
    }
    group_scores = (
        frame.groupby(["group_id", "animal_id"], as_index=False)
        .agg(label=("label", "first"), maximum_probability=("seizure_probability", "max"))
    )
    positive = group_scores.loc[group_scores["label"].eq(1)]
    background = group_scores.loc[group_scores["label"].eq(0)]
    group_metrics = {
        "positive_groups": int(len(positive)),
        "detected_positive_groups": int(
            positive["maximum_probability"].ge(threshold).sum()
        ),
        "positive_group_sensitivity": float(
            positive["maximum_probability"].ge(threshold).mean()
        ),
        "background_groups": int(len(background)),
        "false_positive_background_groups": int(
            background["maximum_probability"].ge(threshold).sum()
        ),
        "false_positive_background_group_fraction": float(
            background["maximum_probability"].ge(threshold).mean()
        ),
    }
    return {"window": overall, "per_mouse_window": per_mouse, "group": group_metrics}


def _select_validation_threshold(frame: pd.DataFrame, metric: str) -> float:
    labels = frame["label"].to_numpy()
    probabilities = frame["seizure_probability"].to_numpy()
    candidates = np.unique(
        np.r_[0.01, np.linspace(0.02, 0.98, 193), 0.99, probabilities]
    )
    if len(candidates) > 2000:
        candidates = np.quantile(candidates, np.linspace(0.0, 1.0, 2000))
    scored = []
    for threshold in candidates:
        values = window_metrics(labels, probabilities, float(threshold))
        score = float(values[metric])
        if not np.isfinite(score):
            score = -float("inf")
        scored.append((score, float(values["balanced_accuracy"]), -abs(float(threshold) - 0.5), float(threshold)))
    return max(scored)[-1]


def _selection_score(
    metric: str, validation_result: Mapping[str, Any], metrics: Mapping[str, Any]
) -> float:
    if metric == "val_loss":
        return -float(validation_result["loss"])
    value = float(metrics[metric])
    return value if np.isfinite(value) else -float("inf")


def _loss_weighting(
    frame: pd.DataFrame, mode: str, device: torch.device
) -> tuple[torch.Tensor | None, dict[tuple[str, int], float]]:
    labels = frame["label"].to_numpy(np.int64)
    if mode == "none":
        return None, {}
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2).astype(float)
    if mode == "balanced":
        weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
        return torch.as_tensor(weights, dtype=torch.float32, device=device), {}

    stratum_counts = (
        frame.assign(animal_id=frame["animal_id"].astype(str))
        .groupby(["animal_id", "label"])
        .size()
    )
    animals = sorted(frame["animal_id"].astype(str).unique())
    expected = {(animal, label) for animal in animals for label in (0, 1)}
    observed = {(str(animal), int(label)) for animal, label in stratum_counts.index}
    if observed != expected:
        raise ValueError("balanced_mouse_sqrt requires every mouse/class stratum")
    total = float(len(frame))
    lookup: dict[tuple[str, int], float] = {}
    for label in (0, 1):
        class_counts = {
            animal: float(stratum_counts.loc[(animal, label)]) for animal in animals
        }
        class_total = sum(class_counts.values())
        raw = {animal: 1.0 / np.sqrt(value) for animal, value in class_counts.items()}
        weighted_average = sum(
            class_counts[animal] * raw[animal] for animal in animals
        ) / class_total
        class_balance = total / (2.0 * class_total)
        for animal in animals:
            lookup[(animal, label)] = float(
                class_balance * raw[animal] / weighted_average
            )
    return None, lookup


def _save_progress(
    path: Path,
    signature_hash: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loader_generator: torch.Generator,
    history: list[dict[str, Any]],
    best_state: dict[str, torch.Tensor] | None,
    best_score: float,
    best_epoch: int,
    stale_epochs: int,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "signature_sha256": signature_hash,
            "completed_epoch": epoch,
            "model_state": {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "loader_generator_state": loader_generator.get_state(),
            "history": history,
            "best_state": best_state,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        temporary,
    )
    temporary.replace(path)


def _restore_rng(progress: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(progress["python_rng_state"])
    np.random.set_state(progress["numpy_rng_state"])
    torch.set_rng_state(progress["torch_rng_state"])
    if device.type == "cuda" and progress.get("cuda_rng_states") is not None:
        torch.cuda.set_rng_state_all(progress["cuda_rng_states"])


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(device)


def _model_key(name: str) -> str:
    key = name.lower().replace("-", "_")
    if key in {"seizure_physionet", "physio", "physio_fusion"}:
        return "seizure_physio"
    return key


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["MixedBinaryTrainingConfig", "run_mixed_binary_training"]
