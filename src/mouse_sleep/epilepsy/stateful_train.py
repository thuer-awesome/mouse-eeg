"""Continuous-sequence training for the paper-faithful AITRICS raw model.

The released AITRICS trainer presents one 30-second source segment at a time
and evaluates ``30 - window_size`` windows.  With four-second windows and a
one-second shift this is **26 windows**, not the mathematically tempting 27.
This module keeps that protocol separate from the ordinary independent-window
trainer used by the other mouse adapters.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn

from mouse_sleep.epilepsy.metrics import window_metrics
from mouse_sleep.epilepsy.models import build_epilepsy_model
from mouse_sleep.epilepsy.train import (
    _evaluate,
    _save_training_progress,
    _select_threshold,
    robust_normalize_window,
)


AITRICS_SEQUENCE_WINDOWS = 26
AITRICS_WINDOW_SECONDS = 4.0
AITRICS_STRIDE_SECONDS = 1.0


class CachedSequenceDataset(torch.utils.data.Dataset):
    """Read non-overlapping 26-window sequences from the existing cache.

    A cache manifest row is one recording segment (one source parquet shard).
    Sequence construction never crosses rows.  Within a row, a new run starts
    after a timestamp discontinuity; this prevents a transition window removed
    during cache construction from silently connecting two unrelated states.
    The cache itself is already in raw-sample order, so timestamps are used for
    continuity checks but the signal order is not reordered.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        sequence_windows: int = AITRICS_SEQUENCE_WINDOWS,
        min_stride_fraction: float = 0.25,
        max_stride_fraction: float = 1.75,
        max_sequences: int | None = None,
    ) -> None:
        if sequence_windows < 1:
            raise ValueError("sequence_windows must be positive")
        if not 0.0 < min_stride_fraction < max_stride_fraction:
            raise ValueError("invalid stride continuity bounds")
        required = {"cache_dir", "n_windows", "animal_id", "session_id"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Sequence manifest is missing columns: {missing}")

        self.rows = frame.to_dict("records")
        self.sequence_windows = int(sequence_windows)
        self.min_stride_fraction = float(min_stride_fraction)
        self.max_stride_fraction = float(max_stride_fraction)
        self._arrays: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._sequences: list[dict[str, Any]] = []
        self._sequence_counts: dict[int, int] = {}
        self._build_sequence_index()
        if max_sequences is not None:
            if max_sequences < 1:
                raise ValueError("max_sequences must be positive")
            self._select_sequences(int(max_sequences))

    def __len__(self) -> int:
        return len(self._sequences)

    @property
    def sequence_metadata(self) -> pd.DataFrame:
        """Return a compact, auditable index of the selected sequences."""

        return pd.DataFrame(
            [
                {
                    "sequence_id": sequence["sequence_id"],
                    "animal_id": sequence["animal_id"],
                    "session_id": sequence["session_id"],
                    "shard": sequence["shard"],
                    "start_time_ms": float(sequence["start_time_ms"]),
                    "end_time_ms": float(sequence["end_time_ms"]),
                    "positive_windows": int(sequence["positive_windows"]),
                    "sequence_windows": self.sequence_windows,
                }
                for sequence in self._sequences
            ]
        )

    def labels(self) -> np.ndarray:
        if not self._sequences:
            return np.asarray([], dtype=np.int8)
        return np.concatenate(
            [sequence["labels"].astype(np.int8, copy=False) for sequence in self._sequences]
        )

    def sequence_labels(self) -> np.ndarray:
        """Binary sequence-level labels used only for sampler balancing."""

        return np.asarray(
            [int(sequence["positive_windows"] > 0) for sequence in self._sequences],
            dtype=np.int8,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence = self._sequences[index]
        signals, labels, starts = self._open(int(sequence["row_index"]))
        windows = np.stack(
            [robust_normalize_window(signals[int(local)]) for local in sequence["indices"]],
            axis=0,
        ).astype(np.float32, copy=False)
        local_indices = sequence["indices"]
        return {
            "signals": torch.from_numpy(windows),
            "labels": torch.from_numpy(labels[local_indices].astype(np.int64, copy=False)),
            "start_time_ms": torch.from_numpy(starts[local_indices].astype(np.float64, copy=False)),
            "animal_id": str(sequence["animal_id"]),
            "session_id": str(sequence["session_id"]),
            "shard": int(sequence["shard"]),
            "sequence_id": str(sequence["sequence_id"]),
        }

    def _open(self, row_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if row_index not in self._arrays:
            root = Path(self.rows[row_index]["cache_dir"])
            self._arrays[row_index] = (
                np.load(root / "signals.npy", mmap_mode="r"),
                np.load(root / "labels.npy", mmap_mode="r"),
                np.load(root / "start_time_ms.npy", mmap_mode="r"),
            )
        return self._arrays[row_index]

    def _build_sequence_index(self) -> None:
        for row_index, row in enumerate(self.rows):
            signals, labels, starts = self._open(row_index)
            n_windows = min(
                int(row["n_windows"]), len(signals), len(labels), len(starts)
            )
            if n_windows < self.sequence_windows:
                continue
            expected_ms = float(row.get("stride_seconds", AITRICS_STRIDE_SECONDS)) * 1000.0
            if not np.isfinite(expected_ms) or expected_ms <= 0:
                expected_ms = AITRICS_STRIDE_SECONDS * 1000.0
            starts_view = np.asarray(starts[:n_windows], dtype=np.float64)
            diffs = np.diff(starts_view)
            contiguous = (
                np.isfinite(diffs)
                & (diffs >= expected_ms * self.min_stride_fraction)
                & (diffs <= expected_ms * self.max_stride_fraction)
            )
            # A run is represented as [start, end) in cache-local indices.
            run_start = 0
            run_breaks = np.flatnonzero(~contiguous)
            for break_index in run_breaks:
                self._append_row_sequences(
                    row_index,
                    row,
                    labels,
                    starts_view,
                    run_start,
                    int(break_index) + 1,
                )
                run_start = int(break_index) + 1
            self._append_row_sequences(
                row_index,
                row,
                labels,
                starts_view,
                run_start,
                n_windows,
            )

    def _append_row_sequences(
        self,
        row_index: int,
        row: dict[str, Any],
        labels: np.ndarray,
        starts: np.ndarray,
        run_start: int,
        run_end: int,
    ) -> None:
        run_length = run_end - run_start
        for offset in range(0, run_length - self.sequence_windows + 1, self.sequence_windows):
            begin = run_start + offset
            end = begin + self.sequence_windows
            local_indices = np.arange(begin, end, dtype=np.int64)
            label_slice = np.asarray(labels[local_indices], dtype=np.int8)
            shard = int(row.get("shard", -1))
            sequence_number = self._sequence_counts.get(row_index, 0)
            self._sequence_counts[row_index] = sequence_number + 1
            self._sequences.append(
                {
                    "row_index": row_index,
                    "indices": local_indices,
                    "labels": label_slice,
                    "positive_windows": int((label_slice == 1).sum()),
                    "animal_id": str(row["animal_id"]),
                    "session_id": str(row["session_id"]),
                    "shard": shard,
                    "sequence_id": (
                        f"{row['animal_id']}/{row['session_id']}/"
                        f"shard{shard:04d}/seq{sequence_number:06d}"
                    ),
                    "start_time_ms": starts[begin],
                    "end_time_ms": starts[end - 1]
                    + float(row.get("window_seconds", AITRICS_WINDOW_SECONDS)) * 1000.0,
                }
            )

    def _select_sequences(self, maximum: int) -> None:
        if len(self._sequences) <= maximum:
            return
        sequence_labels = self.sequence_labels()
        chosen: list[int] = []
        for label in (0, 1):
            candidates = np.flatnonzero(sequence_labels == label)
            target = min(len(candidates), maximum // 2)
            if target:
                chosen.extend(candidates[np.linspace(0, len(candidates) - 1, target, dtype=np.int64)].tolist())
        if len(chosen) < maximum:
            remaining = np.setdiff1d(np.arange(len(self._sequences)), np.asarray(chosen, dtype=np.int64))
            chosen.extend(remaining[: maximum - len(chosen)].tolist())
        self._sequences = [self._sequences[index] for index in sorted(chosen[:maximum])]


def _stateful_loader_pass(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    *,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    collect_predictions: bool,
) -> tuple[float, int, pd.DataFrame]:
    """Run one sequence pass; optimizer steps occur once per window."""

    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    seen = 0
    prediction_rows: list[dict[str, Any]] = []
    for batch in loader:
        signals = batch["signals"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if signals.ndim != 4 or labels.ndim != 2:
            raise ValueError(
                "Stateful loader must return signals [B,26,C,S] and labels [B,26]"
            )
        for window_index in range(signals.shape[1]):
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
            context = torch.enable_grad() if training else torch.no_grad()
            with context:
                logits, _ = model.forward_stateful(
                    signals[:, window_index], reset_state=window_index == 0
                )
                loss = loss_fn(logits, labels[:, window_index])
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            batch_size = int(labels.shape[0])
            total_loss += float(loss.detach()) * batch_size
            seen += batch_size
            if collect_predictions:
                probability = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
                starts = batch["start_time_ms"][:, window_index].cpu().numpy()
                for item_index, value in enumerate(probability):
                    prediction_rows.append(
                        {
                            "animal_id": str(batch["animal_id"][item_index]),
                            "session_id": str(batch["session_id"][item_index]),
                            "sequence_id": str(batch["sequence_id"][item_index]),
                            "shard": int(batch["shard"][item_index]),
                            "start_time_ms": float(starts[item_index]),
                            "label": int(labels[item_index, window_index].detach().cpu()),
                            "seizure_probability": float(value),
                            "window_index": window_index,
                        }
                    )
    return total_loss / max(seen, 1), seen, pd.DataFrame(prediction_rows)


def run_aitrics_stateful_training(
    split_manifest: str | Path,
    output_root: str | Path,
    *,
    model_name: str = "aitrics_raw_cnn2d_lstm_paper_exact",
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    seed: int = 20260804,
    device: str = "auto",
    max_train_sequences: int | None = None,
    max_eval_sequences: int | None = None,
    progress_checkpoint: str | Path | None = None,
) -> Path:
    """Train/evaluate AITRICS with 26-window stateful 30-second segments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if model_name not in {
        "aitrics_raw_cnn2d_lstm_paper_exact",
        "rodent_aitrics_stateful",
    }:
        raise ValueError(f"Unsupported stateful AITRICS model: {model_name}")
    frame = pd.read_csv(split_manifest)
    required = {"train", "val", "test"}
    if set(frame["split"].astype(str)) != required:
        raise ValueError(f"Split manifest must contain exactly {sorted(required)}")
    datasets = {
        split: CachedSequenceDataset(
            frame.loc[frame.split.eq(split)].copy(),
            max_sequences=max_train_sequences if split == "train" else max_eval_sequences,
        )
        for split in required
    }
    if not len(datasets["train"]):
        raise ValueError("Training split has no complete 26-window sequences")
    if not len(datasets["val"]):
        raise ValueError("Validation split has no complete 26-window sequences")
    if not len(datasets["test"]):
        raise ValueError("Test split has no complete 26-window sequences")
    train_labels = datasets["train"].labels()
    if len(np.unique(train_labels)) < 2:
        raise ValueError("Training split needs both seizure and non-seizure windows")
    sample = datasets["train"][0]["signals"]
    channels = int(sample.shape[1])
    model = build_epilepsy_model("rodent_aitrics_stateful", channels=channels, classes=2)
    if not hasattr(model, "forward_stateful"):
        raise TypeError("rodent_aitrics_stateful must expose forward_stateful")
    target = torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )
    model.to(target)
    counts = np.bincount(train_labels, minlength=2).astype(float)
    class_weights = torch.tensor(
        np.sqrt(counts.sum() / np.maximum(counts, 1.0)),
        dtype=torch.float32,
        device=target,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    sequence_labels = datasets["train"].sequence_labels()
    seq_counts = np.bincount(sequence_labels, minlength=2).astype(float)
    sampler_weights = 1.0 / np.maximum(seq_counts[sequence_labels], 1.0)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(sampler_weights, dtype=torch.double),
        len(sampler_weights),
        replacement=True,
    )
    # The released code uses drop_last=True.  It also needs at least two
    # samples for the classifier BatchNorm during training.
    train_batch_size = min(int(batch_size), len(datasets["train"]))
    if train_batch_size < 2:
        raise ValueError("Stateful training needs at least two sequences per batch")
    loaders = {
        "train": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=train_batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=0,
            pin_memory=target.type == "cuda",
        ),
        "val": torch.utils.data.DataLoader(
            datasets["val"],
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            num_workers=0,
            pin_memory=target.type == "cuda",
        ),
        "test": torch.utils.data.DataLoader(
            datasets["test"],
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            num_workers=0,
            pin_memory=target.type == "cuda",
        ),
    }

    progress_path = Path(progress_checkpoint).resolve() if progress_checkpoint else None
    progress_config = {
        "split_manifest": str(Path(split_manifest).resolve()),
        "model": model_name,
        "implementation": "rodent_aitrics_stateful",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "device": str(target),
        "max_train_sequences": max_train_sequences,
        "max_eval_sequences": max_eval_sequences,
        "channels": channels,
        "sequence_windows": AITRICS_SEQUENCE_WINDOWS,
    }
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    start_epoch = 1
    resumed_from_epoch = 0
    if progress_path is not None and progress_path.is_file():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("config") != progress_config:
            raise ValueError(
                f"Progress checkpoint configuration does not match this run: {progress_path}"
            )
        model.load_state_dict(progress["model_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(target)
        best_state = progress["best_state"]
        best_val = float(progress["best_val"])
        history = list(progress["history"])
        resumed_from_epoch = int(progress["completed_epoch"])
        start_epoch = resumed_from_epoch + 1
        random.setstate(progress["python_rng_state"])
        np.random.set_state(progress["numpy_rng_state"])
        torch.set_rng_state(progress["torch_rng_state"])
        if target.type == "cuda" and progress.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(progress["cuda_rng_states"])
        print(
            f"[epilepsy-stateful] resuming after epoch "
            f"{resumed_from_epoch}/{epochs} from {progress_path}",
            flush=True,
        )
    for epoch in range(start_epoch, int(epochs) + 1):
        train_loss, train_seen, _ = _stateful_loader_pass(
            model,
            loaders["train"],
            device=target,
            loss_fn=loss_fn,
            optimizer=optimizer,
            collect_predictions=False,
        )
        val_loss, val_seen, _ = _stateful_loader_pass(
            model,
            loaders["val"],
            device=target,
            loss_fn=loss_fn,
            optimizer=None,
            collect_predictions=False,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_windows": train_seen,
            "val_windows": val_seen,
            "sequence_windows": AITRICS_SEQUENCE_WINDOWS,
        }
        history.append(row)
        print(
            f"[epilepsy-stateful] epoch={epoch}/{epochs} "
            f"train={train_loss:.5f} val={val_loss:.5f}",
            flush=True,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if progress_path is not None:
            _save_training_progress(
                progress_path,
                config=progress_config,
                completed_epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_state=best_state,
                best_val=best_val,
                history=history,
            )
    if best_state is None:
        raise RuntimeError("No stateful checkpoint was produced")
    model.load_state_dict(best_state)
    _, _, val_predictions = _stateful_loader_pass(
        model,
        loaders["val"],
        device=target,
        loss_fn=loss_fn,
        optimizer=None,
        collect_predictions=True,
    )
    _, _, test_predictions = _stateful_loader_pass(
        model,
        loaders["test"],
        device=target,
        loss_fn=loss_fn,
        optimizer=None,
        collect_predictions=True,
    )
    threshold = _select_threshold(val_predictions)
    predictions = {"val": val_predictions, "test": test_predictions}
    results = {
        split: _evaluate(
            prediction,
            threshold,
            frame.loc[frame.split.eq(split)],
        )
        for split, prediction in predictions.items()
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(output_root).resolve() / f"{timestamp}_{model_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "model_name": model_name,
            "implementation": "rodent_aitrics_stateful",
            "state_dict": best_state,
            "channels": channels,
            "classes": 2,
            "threshold": threshold,
            "sequence_windows": AITRICS_SEQUENCE_WINDOWS,
            "window_seconds": AITRICS_WINDOW_SECONDS,
            "stride_seconds": AITRICS_STRIDE_SECONDS,
        },
        run_dir / "model.pt",
    )
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    pd.concat(
        [value.assign(split=key) for key, value in predictions.items()],
        ignore_index=True,
    ).to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    metrics = {
        "threshold_selected_on_validation": threshold,
        "protocol": {
            "name": "aitrics_raw_stateful_30s",
            "sequence_windows": AITRICS_SEQUENCE_WINDOWS,
            "window_seconds": AITRICS_WINDOW_SECONDS,
            "stride_seconds": AITRICS_STRIDE_SECONDS,
            "reset_state_per_recording_segment": True,
            "detach_hidden_before_each_subsequent_window": True,
            "optimizer_step_per_window": True,
        },
        "datasets": {
            split: {
                "sequences": len(dataset),
                "windows": int(len(dataset) * dataset.sequence_windows),
                "animals": sorted(dataset.sequence_metadata["animal_id"].unique().tolist())
                if len(dataset)
                else [],
            }
            for split, dataset in datasets.items()
        },
        **results,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "split_manifest": str(Path(split_manifest).resolve()),
                "model": model_name,
                "implementation": "rodent_aitrics_stateful",
                "epochs": epochs,
                "batch_size": batch_size,
                "effective_train_batch_size": train_batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "device": str(target),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "sequence_windows": AITRICS_SEQUENCE_WINDOWS,
                "normalization": "channelwise median/MAD, clip [-20,20]",
                "max_train_sequences": max_train_sequences,
                "max_eval_sequences": max_eval_sequences,
                "resumed_from_epoch": resumed_from_epoch,
                "progress_checkpoint": (
                    str(progress_path) if progress_path is not None else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir


__all__ = [
    "AITRICS_SEQUENCE_WINDOWS",
    "CachedSequenceDataset",
    "run_aitrics_stateful_training",
]
