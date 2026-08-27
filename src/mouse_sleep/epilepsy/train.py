from __future__ import annotations

import bisect
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from mouse_sleep.epilepsy.metrics import event_metrics, window_metrics
from mouse_sleep.epilepsy.models import build_epilepsy_model


def robust_normalize_window(x: np.ndarray) -> np.ndarray:
    """Apply the cache trainer's channel-wise robust normalization."""

    values = np.asarray(x, dtype=np.float32).copy()
    median = np.median(values, axis=1, keepdims=True)
    scale = np.median(np.abs(values - median), axis=1, keepdims=True) * 1.4826
    return np.clip(
        (values - median) / np.maximum(scale, 1e-6), -20.0, 20.0
    ).astype(np.float32, copy=False)


class CachedWindowDataset(torch.utils.data.Dataset):
    def __init__(self, frame: pd.DataFrame, max_windows: int | None = None) -> None:
        self.rows = frame.to_dict("records")
        self._selected_indices: list[np.ndarray | None] = [None] * len(self.rows)
        self.lengths = [int(row["n_windows"]) for row in self.rows]
        self._arrays: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if max_windows is not None:
            if max_windows < 1:
                raise ValueError("max_windows must be positive")
            self._select_stratified_indices(max_windows)
        self.ends = np.cumsum(self.lengths).tolist()

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def _open(self, row_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if row_index not in self._arrays:
            root = Path(self.rows[row_index]["cache_dir"])
            self._arrays[row_index] = (
                np.load(root / "signals.npy", mmap_mode="r"),
                np.load(root / "labels.npy", mmap_mode="r"),
                np.load(root / "start_time_ms.npy", mmap_mode="r"),
            )
        return self._arrays[row_index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index = bisect.bisect_right(self.ends, index)
        previous = self.ends[row_index - 1] if row_index else 0
        local = index - previous
        signals, labels, starts = self._open(row_index)
        selected = self._selected_indices[row_index]
        source_index = int(selected[local]) if selected is not None else local
        x = robust_normalize_window(signals[source_index])
        row = self.rows[row_index]
        return {
            "signals": torch.from_numpy(x), "label": int(labels[source_index]),
            "start_time_ms": float(starts[source_index]), "session_id": str(row["session_id"]),
            "animal_id": str(row["animal_id"]),
        }

    def labels(self) -> np.ndarray:
        values: list[np.ndarray] = []
        for i in range(len(self.rows)):
            labels = np.asarray(self._open(i)[1])
            selected = self._selected_indices[i]
            values.append(labels[selected] if selected is not None else labels[: self.lengths[i]])
        return np.concatenate(values) if values else np.asarray([], dtype=np.int8)

    def _select_stratified_indices(self, maximum: int) -> None:
        """Choose a deterministic, approximately balanced smoke subset.

        The full training path leaves ``max_windows`` unset and remains fully
        memory-mapped.  When a cap is requested, selecting evenly spaced
        positive and negative indices prevents the old failure mode where the
        first chronological shard contained only one class.
        """

        total = sum(self.lengths)
        if total <= maximum:
            return
        references: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
        for row_index in range(len(self.rows)):
            labels = np.asarray(self._open(row_index)[1])[: self.lengths[row_index]]
            for label in (0, 1):
                references[label].extend(
                    (row_index, int(local))
                    for local in np.flatnonzero(labels == label)
                )
        available = sum(bool(items) for items in references.values())
        if available < 2:
            raise ValueError("Capped dataset needs both seizure and non-seizure windows")
        if references[0] and references[1]:
            target_each = maximum // 2
            counts = {0: min(len(references[0]), target_each), 1: min(len(references[1]), target_each)}
            remainder = maximum - counts[0] - counts[1]
            for label in (0, 1):
                extra = min(remainder, len(references[label]) - counts[label])
                counts[label] += extra
                remainder -= extra
        else:
            label = 0 if references[0] else 1
            counts = {label: min(maximum, len(references[label]))}
        for label, items in references.items():
            count = counts.get(label, 0)
            if not count:
                continue
            chosen = np.linspace(0, len(items) - 1, count, dtype=np.int64)
            for row_index, local in [items[int(i)] for i in chosen]:
                current = self._selected_indices[row_index]
                if current is None:
                    self._selected_indices[row_index] = np.asarray([local], dtype=np.int64)
                else:
                    self._selected_indices[row_index] = np.append(current, local)
        for row_index, selected in enumerate(self._selected_indices):
            if selected is not None:
                selected.sort()
                self.lengths[row_index] = int(selected.size)
            else:
                self.lengths[row_index] = 0


def run_training(
    split_manifest: str | Path,
    output_root: str | Path,
    model_name: str = "aitrics_cnn2d_lstm",
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 20260804,
    device: str = "auto",
    max_train_windows: int | None = None,
    max_eval_windows: int | None = None,
    progress_checkpoint: str | Path | None = None,
) -> Path:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Keep fold comparisons reproducible on CUDA.  This is deliberately
        # set inside the epilepsy entry point so other project pipelines keep
        # their existing performance-oriented backend settings.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    frame = pd.read_csv(split_manifest)
    required = {"train", "val", "test"}
    if set(frame["split"].astype(str)) != required:
        raise ValueError(f"Split manifest must contain exactly {sorted(required)}")
    datasets = {
        split: CachedWindowDataset(
            frame.loc[frame.split.eq(split)].copy(),
            max_windows=max_train_windows if split == "train" else max_eval_windows,
        )
        for split in required
    }
    train_labels = datasets["train"].labels()
    if len(np.unique(train_labels)) < 2:
        raise ValueError("Training split needs both seizure and non-seizure windows")
    sample = datasets["train"][0]["signals"]
    model = build_epilepsy_model(model_name, channels=int(sample.shape[0]), classes=2)
    target = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
    model.to(target)
    counts = np.bincount(train_labels, minlength=2).astype(float)
    class_weights = torch.tensor(np.sqrt(counts.sum() / np.maximum(counts, 1.0)), dtype=torch.float32, device=target)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    sample_weights = 1.0 / np.maximum(counts[train_labels], 1.0)
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True)
    loaders = {
        "train": torch.utils.data.DataLoader(datasets["train"], batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=target.type == "cuda"),
        "val": torch.utils.data.DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=target.type == "cuda"),
        "test": torch.utils.data.DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=target.type == "cuda"),
    }
    progress_path = Path(progress_checkpoint).resolve() if progress_checkpoint else None
    progress_config = {
        "split_manifest": str(Path(split_manifest).resolve()),
        "model": model_name,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "device": str(target),
        "max_train_windows": max_train_windows,
        "max_eval_windows": max_eval_windows,
        "channels": int(sample.shape[0]),
    }
    best_state = None; best_val = float("inf"); history = []; start_epoch = 1
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
            f"[epilepsy] resuming after epoch {resumed_from_epoch}/{epochs} from {progress_path}",
            flush=True,
        )
    for epoch in range(start_epoch, epochs + 1):
        model.train(); total = 0.; seen = 0
        for batch in loaders["train"]:
            x = batch["signals"].to(target, non_blocking=True); y = batch["label"].to(target)
            optimizer.zero_grad(set_to_none=True); loss = loss_fn(model(x), y); loss.backward(); optimizer.step()
            # EEGNet's published Keras implementation applies max-norm after
            # every update.  Adapters expose the projection without changing
            # other legacy model behavior.
            constrain = getattr(model, "apply_max_norm_constraints", None)
            if constrain is not None:
                constrain()
            total += float(loss.detach()) * len(y); seen += len(y)
        val_loss = _loss(model, loaders["val"], loss_fn, target)
        history.append({"epoch": epoch, "train_loss": total / max(seen, 1), "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
        print(f"[epilepsy] epoch={epoch}/{epochs} train={history[-1]['train_loss']:.5f} val={val_loss:.5f}", flush=True)
    if best_state is None: raise RuntimeError("No checkpoint was produced")
    model.load_state_dict(best_state)
    predictions = {split: _predict(model, loaders[split], target) for split in ("val", "test")}
    threshold = _select_threshold(predictions["val"])
    results = {split: _evaluate(predictions[split], threshold, frame.loc[frame.split.eq(split)]) for split in ("val", "test")}
    run_dir = Path(output_root).resolve() / f"{datetime.now():%Y%m%d_%H%M%S}_{model_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.save({"model_name": model_name, "state_dict": best_state, "channels": int(sample.shape[0]), "classes": 2, "threshold": threshold}, run_dir / "model.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    pd.concat([value.assign(split=key) for key, value in predictions.items()], ignore_index=True).to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    (run_dir / "metrics.json").write_text(json.dumps({"threshold_selected_on_validation": threshold, **results}, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps({"split_manifest": str(Path(split_manifest).resolve()), "model": model_name, "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "seed": seed, "device": str(target), "parameters": sum(p.numel() for p in model.parameters()), "resumed_from_epoch": resumed_from_epoch, "progress_checkpoint": str(progress_path) if progress_path is not None else None}, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def _save_training_progress(
    path: Path,
    *,
    config: dict[str, Any],
    completed_epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_state: dict[str, torch.Tensor] | None,
    best_val: float,
    history: list[dict[str, float | int]],
) -> None:
    """Atomically persist enough state to resume at the next epoch."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = {
        "schema_version": 1,
        "config": config,
        "completed_epoch": int(completed_epoch),
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "best_state": best_state,
        "best_val": float(best_val),
        "history": list(history),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, temporary)
    temporary.replace(path)


def _loss(model: nn.Module, loader: Any, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval(); total = 0.; seen = 0
    with torch.no_grad():
        for batch in loader:
            y = batch["label"].to(device); loss = loss_fn(model(batch["signals"].to(device)), y); total += float(loss) * len(y); seen += len(y)
    return total / max(seen, 1)


def _predict(model: nn.Module, loader: Any, device: torch.device) -> pd.DataFrame:
    rows = []; model.eval()
    with torch.no_grad():
        for batch in loader:
            prob = torch.softmax(model(batch["signals"].to(device)), dim=1)[:, 1].cpu().numpy()
            for i, value in enumerate(prob): rows.append({"animal_id": batch["animal_id"][i], "session_id": batch["session_id"][i], "start_time_ms": float(batch["start_time_ms"][i]), "label": int(batch["label"][i]), "seizure_probability": float(value)})
    return pd.DataFrame(rows)


def _select_threshold(frame: pd.DataFrame) -> float:
    candidates = np.linspace(.1, .9, 81); y = frame["label"].to_numpy(); p = frame["seizure_probability"].to_numpy()
    if len(np.unique(y)) < 2:
        return 0.5
    return float(max(candidates, key=lambda t: window_metrics(y, p, float(t))["f1"]))


def _evaluate(predictions: pd.DataFrame, threshold: float, split_rows: pd.DataFrame) -> dict[str, Any]:
    y = predictions["label"].to_numpy(); p = predictions["seizure_probability"].to_numpy()
    result: dict[str, Any] = {"window": window_metrics(y, p, threshold), "sessions": {}}
    aggregate = {"true_events": 0, "predicted_events": 0, "false_alarms": 0}; sensitivities = []; latencies = []
    total_hours = 0.
    for session, group in predictions.groupby("session_id", sort=True):
        row = split_rows.loc[split_rows.session_id.astype(str).eq(str(session))].iloc[0]
        metric = event_metrics(group.label.to_numpy(), group.seizure_probability.to_numpy(), group.start_time_ms.to_numpy(), float(row.get("window_seconds", 4.0)), threshold)
        result["sessions"][str(session)] = metric
        for key in aggregate: aggregate[key] += int(metric[key])
        if metric["true_events"]: sensitivities.append((metric["event_sensitivity"], metric["true_events"]))
        if np.isfinite(metric["median_latency_s"]): latencies.append(metric["median_latency_s"])
        starts = group.start_time_ms.to_numpy(); total_hours += (starts.max() - starts.min() + float(row.get("window_seconds", 4.0)) * 1000) / 3_600_000 if len(starts) else 0
    result["event"] = {**aggregate, "event_sensitivity": float(sum(v * n for v, n in sensitivities) / max(sum(n for _, n in sensitivities), 1)), "false_alarms_per_hour": float(aggregate["false_alarms"] / max(total_hours, 1e-9)), "median_session_latency_s": float(np.median(latencies)) if latencies else float("nan")}
    return result
