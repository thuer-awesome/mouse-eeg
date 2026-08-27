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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F

from mouse_sleep.epilepsy.continuous_single_channel import (
    ContinuousSingleChannelDataset,
)
from mouse_sleep.epilepsy.metrics import window_metrics
from mouse_sleep.epilepsy.single_channel_models import build_single_channel_model


@dataclass(frozen=True)
class SingleChannelTrainingConfig:
    model_name: str
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    epochs: int = 15
    batch_size: int = 128
    # ``None`` is the strict protocol: every supervised center is visited once
    # per epoch.  A finite value is retained only for explicit smoke tests.
    samples_per_epoch: int | None = None
    sampling_mode: str = "all_windows"
    class_weight_mode: str = "hierarchical_event_region"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    auxiliary_statistics_weight: float = 0.0
    early_stopping_patience: int = 5
    learning_rate_patience: int = 2
    learning_rate_factor: float = 0.5
    gradient_clip_norm: float = 1.0
    seed: int = 20260814
    device: str = "auto"
    mixed_precision: bool = True
    num_workers: int = 0
    normalization_clip: float = 20.0
    maximum_evaluation_windows: int | None = None

    def validated(self) -> "SingleChannelTrainingConfig":
        valid_models = {
            "eegwavenet_sc",
            "physio_sequence_no_aux",
            "physio_sequence_stats",
            "physio_sequence+stats",
            "physio_sequence_time_frequency",
            "physio_sequence_stats_input",
            "physio_sequence_stats_aux",
        }
        if self.model_name not in valid_models:
            raise ValueError(f"unsupported single-channel model {self.model_name!r}")
        if min(self.epochs, self.batch_size) < 1:
            raise ValueError("epochs, batch size, and samples per epoch must be positive")
        if self.samples_per_epoch is not None and self.samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive or None")
        if self.maximum_evaluation_windows is not None and self.maximum_evaluation_windows < 2:
            raise ValueError("maximum_evaluation_windows must be at least two or None")
        if self.sampling_mode not in {"all_windows", "weighted_smoke"}:
            raise ValueError("sampling_mode must be all_windows or weighted_smoke")
        if self.sampling_mode == "weighted_smoke" and self.samples_per_epoch is None:
            raise ValueError("weighted_smoke requires samples_per_epoch")
        if self.class_weight_mode not in {
            "none",
            "balanced",
            "balanced_mouse_sqrt",
            "hierarchical_event_region",
        }:
            raise ValueError(
                "class_weight_mode must be none, balanced, balanced_mouse_sqrt, "
                "or hierarchical_event_region"
            )
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning rate must be positive and weight decay nonnegative")
        if self.auxiliary_statistics_weight < 0:
            raise ValueError("auxiliary weight cannot be negative")
        is_aux = self.model_name in {
            "physio_sequence_stats",
            "physio_sequence+stats",
            "physio_sequence_stats_aux",
        }
        if is_aux != (self.auxiliary_statistics_weight > 0):
            raise ValueError(
                "the stats model requires a positive auxiliary weight and all other models require zero"
            )
        if min(self.early_stopping_patience, self.learning_rate_patience) < 1:
            raise ValueError("training patience values must be positive")
        if not 0 < self.learning_rate_factor < 1:
            raise ValueError("learning_rate_factor must lie in (0, 1)")
        if self.gradient_clip_norm <= 0 or self.normalization_clip <= 0:
            raise ValueError("gradient clip and normalization clip must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        return self

    @property
    def sequence_length(self) -> int:
        return 1 if self.model_name == "eegwavenet_sc" else 21

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["model_kwargs"] = dict(self.model_kwargs)
        return result


def run_single_channel_training(
    protocol_dir: str | Path,
    output_dir: str | Path,
    config: SingleChannelTrainingConfig,
) -> Path:
    """Train one blinded channel model and preserve validation/test predictions."""

    config = config.validated()
    protocol = Path(protocol_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "full_index": protocol / "full_index.parquet",
        "centers": protocol / "supervised_centers.parquet",
        "baseline": protocol / "training_baseline.json",
        "protocol_audit": protocol / "protocol_audit.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"protocol is incomplete: {missing}")
    signature = {
        "schema_version": 1,
        "protocol": str(protocol),
        "inputs": {name: _sha256(path) for name, path in paths.items()},
        "config": config.to_dict(),
    }
    signature_hash = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode("utf-8")
    ).hexdigest()
    completed_path = output / "completed.json"
    if completed_path.is_file():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if completed.get("signature_sha256") != signature_hash:
            raise ValueError(f"completed output has a different signature: {output}")
        return output

    _seed_everything(config.seed)
    # Read/sort the large full index once and share it across split datasets.
    # This preserves the exact 21-window context while avoiding three copies
    # of the multi-million-row frame during pooled training.
    context_frame = pd.read_parquet(paths["full_index"])
    context_frame = context_frame.sort_values(
        ["continuity_id", "continuity_position"], kind="stable"
    ).reset_index(drop=True)
    context_frame["_context_row"] = np.arange(len(context_frame), dtype=np.int64)
    datasets = {
        split: ContinuousSingleChannelDataset(
            paths["full_index"],
            paths["centers"],
            paths["baseline"],
            split=split,
            sequence_length=config.sequence_length,
            normalization_clip=config.normalization_clip,
            mask_other_splits=True,
            context_frame=context_frame,
        )
        for split in ("train", "val", "test")
    }
    for split, dataset in datasets.items():
        if split != "train" and config.maximum_evaluation_windows is not None:
            dataset.audit_frame = _stratified_limit_frame(
                dataset.audit_frame, config.maximum_evaluation_windows
            )
        labels = set(dataset.labels().tolist())
        if labels != {0, 1}:
            raise ValueError(f"{split} dataset does not contain both classes: {labels}")
    generator = torch.Generator().manual_seed(config.seed)
    sampler = None
    if config.sampling_mode == "weighted_smoke":
        train_weights = _hierarchical_sample_weights(datasets["train"].audit_frame)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(train_weights, dtype=torch.double),
            num_samples=int(config.samples_per_epoch),
            replacement=True,
            generator=generator,
        )
    device = _resolve_device(config.device)
    class_weights, training_sample_weights = _loss_weighting(
        datasets["train"].audit_frame, config.class_weight_mode, device
    )
    loaders = {
        "train": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=config.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            generator=generator if sampler is None else None,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.num_workers > 0,
        ),
        **{
            split: torch.utils.data.DataLoader(
                datasets[split],
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=config.num_workers > 0,
            )
            for split in ("val", "test")
        },
    }

    model = build_single_channel_model(
        config.model_name,
        classes=2,
        samples_per_window=1000,
        sample_rate_hz=250.0,
        sequence_length=config.sequence_length,
        **dict(config.model_kwargs),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.learning_rate_factor,
        patience=config.learning_rate_patience,
        min_lr=1e-6,
    )
    amp_enabled = bool(config.mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    progress_path = output / "progress.pt"
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    if progress_path.is_file():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("signature_sha256") == signature_hash:
            model.load_state_dict(progress["model_state"])
            optimizer.load_state_dict(progress["optimizer_state"])
            _optimizer_to(optimizer, device)
            scheduler.load_state_dict(progress["scheduler_state"])
            scaler.load_state_dict(progress["scaler_state"])
            generator.set_state(progress["sampler_generator_state"])
            history = list(progress["history"])
            best_state = progress["best_state"]
            best_score = float(progress["best_score"])
            best_epoch = int(progress["best_epoch"])
            stale_epochs = int(progress["stale_epochs"])
            start_epoch = int(progress["completed_epoch"]) + 1
            _restore_rng(progress)
        else:
            # A stale checkpoint can be left by an intentionally changed
            # batch size or sampler.  Start a fresh run in the same model
            # directory; the first completed epoch atomically overwrites it.
            print(
                f"[single-channel] ignoring stale progress for model={config.model_name} "
                "because the training signature changed",
                flush=True,
            )

    for epoch in range(start_epoch, config.epochs + 1):
        train_result = _run_loader(
            model,
            loaders["train"],
            datasets["train"],
            device,
            config.auxiliary_statistics_weight,
            config.gradient_clip_norm,
            class_weights,
            training_sample_weights,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            collect_predictions=False,
        )
        val_result = _run_loader(
            model,
            loaders["val"],
            datasets["val"],
            device,
            config.auxiliary_statistics_weight,
            config.gradient_clip_norm,
            None,
            None,
            amp_enabled=amp_enabled,
            collect_predictions=True,
        )
        validation_metrics = _prediction_metrics(val_result["predictions"])
        score = float(validation_metrics["macro_animal_auprc"])
        improved = score > best_score + 1e-8
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
        scheduler.step(score)
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": train_result["loss"],
                "train_classification_loss": train_result["classification_loss"],
                "train_statistics_loss": train_result["statistics_loss"],
                "validation_loss": val_result["loss"],
                "validation_classification_loss": val_result["classification_loss"],
                "validation_statistics_loss": val_result["statistics_loss"],
                "validation_auprc": validation_metrics["pooled"]["auprc"],
                "validation_auroc": validation_metrics["pooled"]["auroc"],
                "validation_macro_animal_auprc": score,
                "best": improved,
            }
        )
        _save_progress(
            progress_path,
            signature_hash,
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            generator,
            history,
            best_state,
            best_score,
            best_epoch,
            stale_epochs,
        )
        print(
            f"[single-channel] model={config.model_name} epoch={epoch}/{config.epochs} "
            f"train={train_result['loss']:.5f} val={val_result['loss']:.5f} "
            f"val_auprc={validation_metrics['pooled']['auprc']:.5f} "
            f"macro_mouse_auprc={score:.5f}",
            flush=True,
        )
        if stale_epochs >= config.early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError("training produced no valid validation checkpoint")
    model.load_state_dict(best_state)
    prediction_frames = []
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_selected_on": "validation_macro_animal_auprc",
        "best_epoch": best_epoch,
        "best_validation_macro_animal_auprc": best_score,
        "accuracy_is_selection_metric": False,
    }
    for split in ("val", "test"):
        result = _run_loader(
            model,
            loaders[split],
            datasets[split],
            device,
            config.auxiliary_statistics_weight,
            config.gradient_clip_norm,
            None,
            None,
            amp_enabled=amp_enabled,
            collect_predictions=True,
        )
        prediction = result["predictions"].assign(split=split)
        prediction_frames.append(prediction)
        metrics[split] = _prediction_metrics(prediction)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions_path = output / "predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    torch.save(
        {
            "schema_version": 1,
            "model_name": config.model_name,
            "model_kwargs": dict(config.model_kwargs),
            "sequence_length": config.sequence_length,
            "state_dict": best_state,
            "auxiliary_statistics_weight": config.auxiliary_statistics_weight,
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
        "training_sampler": config.sampling_mode,
        "class_weight_mode": config.class_weight_mode,
        "train_windows_per_epoch": int(len(datasets["train"]))
        if config.sampling_mode == "all_windows"
        else int(config.samples_per_epoch),
        "dataset_windows": {
            split: int(len(dataset)) for split, dataset in datasets.items()
        },
        "prediction_sha256": _sha256(predictions_path),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    completed_path.write_text(
        json.dumps(
            {
                "signature_sha256": signature_hash,
                "best_epoch": best_epoch,
                "best_validation_macro_animal_auprc": best_score,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def _hierarchical_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    required = {
        "animal_id",
        "label",
        "region_id",
        "channel_name",
        "channel_event_ids",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"training audit frame lacks sampler columns: {sorted(missing)}")
    audit = frame.copy()
    audit["_stratum"] = (
        audit["animal_id"].astype(str) + ":label" + audit["label"].astype(str)
    )
    event_ids = audit["channel_event_ids"].fillna("").astype(str)
    missing_positive_event = audit["label"].eq(1) & event_ids.eq("")
    if missing_positive_event.any():
        raise ValueError("positive training rows require channel_event_ids")
    audit["_group"] = np.where(
        audit["label"].eq(1),
        "event:" + event_ids + ":" + audit["channel_name"].astype(str),
        "region:"
        + audit["region_id"].astype(str)
        + ":"
        + audit["channel_name"].astype(str),
    )
    group_sizes = audit.groupby(["_stratum", "_group"]).size()
    group_counts = audit.groupby("_stratum")["_group"].nunique()
    weights = np.asarray(
        [
            1.0 / (
                float(group_counts.loc[stratum])
                * float(group_sizes.loc[(stratum, group)])
            )
            for stratum, group in audit[["_stratum", "_group"]].itertuples(
                index=False, name=None
            )
        ],
        dtype=np.float64,
    )
    weights /= weights.mean()
    return weights


def _stratified_limit_frame(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    """Deterministically cap only smoke-test evaluation rows."""

    if len(frame) <= maximum:
        return frame.reset_index(drop=True)
    selected_indices: list[Any] = []
    labels = sorted(frame["label"].astype(int).unique().tolist())
    remaining = int(maximum)
    for position, label in enumerate(labels):
        group = frame.loc[frame["label"].eq(label)]
        slots = len(labels) - position
        take = min(len(group), max(1, remaining // slots))
        indices = np.linspace(0, len(group) - 1, take, dtype=np.int64)
        selected_indices.extend(group.index[indices].tolist())
        remaining -= take
    if len(selected_indices) < maximum:
        selected_set = set(selected_indices)
        selected_indices.extend(
            index for index in frame.index if index not in selected_set
        )
    return frame.loc[selected_indices[:maximum]].reset_index(drop=True)


def _run_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    dataset: ContinuousSingleChannelDataset,
    device: torch.device,
    auxiliary_weight: float,
    gradient_clip_norm: float,
    class_weights: torch.Tensor | None,
    sample_weights: np.ndarray | None,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool,
    collect_predictions: bool,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_classification = 0.0
    total_statistics = 0.0
    seen = 0
    rows = []
    context = nullcontext() if training else torch.no_grad()
    with context:
        for batch in loader:
            signals = batch["signals"].to(device, non_blocking=True)
            padding = batch["padding_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(signals, padding)
                batch_row_indices = batch["row_index"].detach().cpu().numpy()
                classification_loss = _classification_loss(
                    outputs["logits"],
                    labels,
                    class_weights if training else None,
                    sample_weights[batch_row_indices]
                    if training and sample_weights is not None
                    else None,
                )
                statistics_loss = _statistics_loss(outputs)
                loss = classification_loss + auxiliary_weight * statistics_loss
            if training:
                if scaler is None:
                    raise AssertionError("training requires a gradient scaler")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            batch_size = int(labels.shape[0])
            seen += batch_size
            total_loss += float(loss.detach()) * batch_size
            total_classification += float(classification_loss.detach()) * batch_size
            total_statistics += float(statistics_loss.detach()) * batch_size
            if collect_predictions:
                probability = torch.softmax(outputs["logits"].detach().float(), dim=1)[:, 1]
                for row_index, label, score in zip(
                    batch["row_index"].tolist(),
                    labels.detach().cpu().tolist(),
                    probability.cpu().tolist(),
                    strict=True,
                ):
                    rows.append((int(row_index), int(label), float(score)))
    result: dict[str, Any] = {
        "loss": total_loss / max(seen, 1),
        "classification_loss": total_classification / max(seen, 1),
        "statistics_loss": total_statistics / max(seen, 1),
    }
    if collect_predictions:
        row_indices = [row[0] for row in rows]
        prediction = dataset.audit_frame.iloc[row_indices].copy().reset_index(drop=True)
        prediction["label"] = [row[1] for row in rows]
        prediction["seizure_probability"] = [row[2] for row in rows]
        prediction = prediction.drop(columns=["_context_row"], errors="ignore")
        result["predictions"] = prediction
    return result


def _classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None,
    sample_weights: np.ndarray | None,
) -> torch.Tensor:
    losses = F.cross_entropy(
        logits, labels, weight=class_weights, reduction="none"
    )
    if sample_weights is None:
        if class_weights is not None:
            denominator = class_weights[labels].sum().clamp_min(1e-8)
            return losses.sum() / denominator
        return losses.mean()
    batch_weights = torch.as_tensor(
        sample_weights,
        dtype=losses.dtype,
        device=losses.device,
    )
    return (losses * batch_weights).sum() / batch_weights.sum().clamp_min(1e-8)


def _loss_weighting(
    frame: pd.DataFrame,
    mode: str,
    device: torch.device,
) -> tuple[torch.Tensor | None, np.ndarray | None]:
    labels = frame["label"].to_numpy(np.int64)
    if mode == "none":
        return None, None
    counts = np.bincount(labels, minlength=2).astype(float)
    if mode == "balanced":
        weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
        return torch.as_tensor(weights, dtype=torch.float32, device=device), None
    if mode == "hierarchical_event_region":
        return None, _hierarchical_sample_weights(frame)
    strata = (
        frame.assign(animal_id=frame["animal_id"].astype(str))
        .groupby(["animal_id", "label"])
        .size()
    )
    animals = sorted(frame["animal_id"].astype(str).unique())
    expected = {(animal, label) for animal in animals for label in (0, 1)}
    observed = {(str(animal), int(label)) for animal, label in strata.index}
    if observed != expected:
        raise ValueError("balanced_mouse_sqrt requires every mouse/class stratum")
    total = float(len(frame))
    lookup: dict[tuple[str, int], float] = {}
    for label in (0, 1):
        class_counts = {animal: float(strata.loc[(animal, label)]) for animal in animals}
        class_total = sum(class_counts.values())
        raw = {animal: 1.0 / np.sqrt(value) for animal, value in class_counts.items()}
        average = sum(class_counts[a] * raw[a] for a in animals) / class_total
        class_balance = total / (2.0 * class_total)
        for animal in animals:
            lookup[(animal, label)] = float(class_balance * raw[animal] / average)
    sample_weights = np.asarray(
        [
            lookup[(str(animal), int(label))]
            for animal, label in frame[["animal_id", "label"]].itertuples(
                index=False, name=None
            )
        ],
        dtype=np.float64,
    )
    return None, sample_weights


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


def _prediction_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    pooled = window_metrics(
        frame["label"].to_numpy(np.int64),
        frame["seizure_probability"].to_numpy(float),
        threshold=0.5,
    )
    per_animal = {
        str(animal): _rank_metrics(group)
        for animal, group in frame.groupby("animal_id", sort=True)
    }
    per_channel = {
        f"{animal}:{channel}": _rank_metrics(group)
        for (animal, channel), group in frame.groupby(
            ["animal_id", "channel_name"], sort=True
        )
    }
    animal_auprc = [
        values["auprc"] for values in per_animal.values() if values["auprc"] is not None
    ]
    animal_auroc = [
        values["auroc"] for values in per_animal.values() if values["auroc"] is not None
    ]
    return {
        "pooled": pooled,
        "macro_animal_auprc": float(np.mean(animal_auprc)),
        "macro_animal_auroc": float(np.mean(animal_auroc)),
        "per_animal": per_animal,
        "per_animal_channel": per_channel,
    }


def _rank_metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    labels = frame["label"].to_numpy(np.int64)
    scores = frame["seizure_probability"].to_numpy(float)
    classes = np.unique(labels)
    if len(classes) < 2:
        return {
            "windows": int(len(frame)),
            "positives": int(labels.sum()),
            "positive_fraction": float(labels.mean()),
            "auprc": None,
            "auroc": None,
        }
    return {
        "windows": int(len(frame)),
        "positives": int(labels.sum()),
        "positive_fraction": float(labels.mean()),
        "auprc": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
    }


def _save_progress(
    path: Path,
    signature_hash: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    generator: torch.Generator,
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
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "sampler_generator_state": generator.get_state(),
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


def _restore_rng(progress: Mapping[str, Any]) -> None:
    random.setstate(progress["python_rng_state"])
    np.random.set_state(progress["numpy_rng_state"])
    torch.set_rng_state(progress["torch_rng_state"])
    if torch.cuda.is_available() and progress.get("cuda_rng_states") is not None:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["SingleChannelTrainingConfig", "run_single_channel_training"]
