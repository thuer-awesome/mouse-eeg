#!/usr/bin/env python
"""Evaluate frozen-category models on every eligible held-out test window."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
# ``train_pg36.py`` is a checked-in script rather than an installed package
# module.  Put both the script directory and the project ``src`` directory on
# the import path so this evaluator works from the repository root, from an
# arbitrary working directory, and via ``python -m`` style wrappers.  The
# imports themselves remain lazy (inside ``main``) so ``--help`` does not
# require PyTorch/Braindecode to be installed.
SCRIPT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
for _path in (SCRIPT_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

OUT = ROOT / "output" / "epilepsy_program_20260821" / "02_morphology_taxonomy" / "all_mice_old_plus_new" / "strict_all_channel_v1"

MODEL_ORDER = ("eegwavenet_sc", "physio_tf", "physio_stats", "eegnetv4", "eegconformer")


def score(y: np.ndarray, p: np.ndarray, classes: int) -> tuple[dict[str, float], np.ndarray]:
    if y.ndim != 1 or p.ndim != 2 or len(y) != len(p):
        raise ValueError(f"invalid score shapes: y={y.shape}, p={p.shape}")
    if p.shape[1] != classes:
        raise ValueError(f"probability matrix has {p.shape[1]} classes; expected {classes}")
    if len(y) == 0:
        raise ValueError("cannot evaluate an empty split")
    if np.any((y < 0) | (y >= classes)):
        raise ValueError("labels are outside the configured class range")
    pred = p.argmax(1)
    cm = confusion_matrix(y, pred, labels=np.arange(classes))
    ybin = np.eye(classes, dtype=int)[y]
    aps, auc = [], []
    for c in range(classes):
        if ybin[:, c].sum() and ybin[:, c].sum() < len(y):
            aps.append(average_precision_score(ybin[:, c], p[:, c]))
            auc.append(roc_auc_score(ybin[:, c], p[:, c]))
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        # ``balanced_accuracy_score`` emits a warning when a per-stream
        # diagnostic has no true examples for a class that the model predicts.
        # Computing the same mean recall from the already materialized matrix
        # keeps the formal metric identical while avoiding noisy logs.
        "balanced_accuracy": float(
            np.mean(
                np.divide(
                    np.diag(cm),
                    cm.sum(axis=1),
                    out=np.zeros(classes, dtype=float),
                    where=cm.sum(axis=1) > 0,
                )[cm.sum(axis=1) > 0]
            )
        ),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_auprc": float(np.mean(aps)) if aps else float("nan"),
        "macro_auroc": float(np.mean(auc)) if auc else float("nan"),
    }, cm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--label-path", type=Path, default=None)
    parser.add_argument("--output-subdir", default="05_classification")
    parser.add_argument("--label-prefix", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=None)
    parser.add_argument("--device", default="auto", help="auto、cpu 或 CUDA 设备名（如 cuda:0）")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-non36", action="store_true", help="仅调试用：允许冻结类别数不是 36")
    args = parser.parse_args(argv)

    # Heavy imports are intentionally delayed until after argparse.  This
    # keeps ``evaluate_strict_classifiers_all_windows.py --help`` useful in a
    # CPU/audit-only environment where torch is not installed.
    try:
        import torch
        from torch.utils.data import DataLoader
        from train_pg36 import (
            ChannelWindowDataset,
            build_model,
            infer_input_shape,
            output_log_probs,
            seed_all,
        )
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "全窗口模型评估需要 PyTorch 以及仓库内 scripts/train_pg36.py；"
            "请安装 `pip install -e '.[all]'` 后重试"
        ) from exc

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    root = args.out.resolve()
    label_path = args.label_path.resolve() if args.label_path is not None else root / "04_label_freeze" / "frozen_channel_window_labels.parquet"
    result_root = root / args.output_subdir
    if not label_path.is_file():
        raise FileNotFoundError(f"PG36 label table not found: {label_path}")
    if not result_root.is_dir():
        raise FileNotFoundError(f"classification output directory not found: {result_root}")
    seed_all(int(args.seed))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")
    labels = pd.read_parquet(label_path)
    required = {"split", "global_category", "animal_id", "session_id", "event_id", "channel_name", "cache_dir", "cache_local_index"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"PG36 label table is missing columns: {sorted(missing)}")
    labels = labels.loc[labels.global_category.notna()].copy()
    labels["global_category"] = pd.to_numeric(labels.global_category, errors="raise").astype(int)
    labels["split"] = labels["split"].astype(str).str.lower()
    if set(labels["split"]) - {"train", "val", "test"}:
        raise ValueError("PG36 label table contains unknown split values")
    observed_classes = sorted(labels["global_category"].unique().tolist())
    if not observed_classes or observed_classes != list(range(observed_classes[-1] + 1)):
        raise ValueError("global_category values must be contiguous and start at zero")
    classes = int(observed_classes[-1] + 1)
    context = labels.copy()
    samples_per_window, sample_rate_hz = infer_input_shape(labels)
    test_labels = labels.loc[labels.split.eq("test")].copy().reset_index(drop=True)
    if test_labels.empty:
        raise ValueError("PG36 label table has no test rows")
    # Keep the formal evaluator tied to the frozen PG36 contract while still
    # deriving the class count from the complete label table (the test split
    # may not contain every rare class).
    if classes != 36 and not args.allow_non36:
        raise ValueError(f"strict PG36 evaluation requires 36 classes; found {classes}")
    label_prefix = args.label_prefix or ("PG" if labels.get("global_label", pd.Series(dtype=str)).astype(str).str.startswith("PG").any() else "G")
    results = []
    stream_results = []
    model_names = tuple(args.models or MODEL_ORDER)
    model_dirs = [result_root / name for name in model_names]
    missing_models = [str(path / "best_model.pt") for path in model_dirs if not (path / "best_model.pt").is_file()]
    if missing_models:
        raise FileNotFoundError("missing model checkpoints: " + ", ".join(missing_models))
    for model_dir in model_dirs:
        model_name = model_dir.name
        meta = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=False)
        checkpoint_classes = int(meta.get("classes", classes))
        if checkpoint_classes != classes:
            raise ValueError(f"{model_name} checkpoint has {checkpoint_classes} classes; label table has {classes}")
        checkpoint_samples = meta.get("samples_per_window")
        if checkpoint_samples is not None and int(checkpoint_samples) != samples_per_window:
            raise ValueError(
                f"{model_name} checkpoint has {checkpoint_samples} samples/window; "
                f"label cache has {samples_per_window}"
            )
        if checkpoint_samples is None and samples_per_window != 1000:
            raise ValueError(
                f"{model_name} is a legacy checkpoint without input-shape metadata; "
                f"retrain before evaluating a {samples_per_window}-sample cache"
            )
        checkpoint_rate = meta.get("sample_rate_hz")
        if checkpoint_rate is not None and not np.isclose(float(checkpoint_rate), sample_rate_hz, rtol=0, atol=1e-4):
            raise ValueError(
                f"{model_name} checkpoint has sample_rate_hz={checkpoint_rate}; "
                f"label cache has {sample_rate_hz}"
            )
        if checkpoint_rate is None and not np.isclose(sample_rate_hz, 250.0, rtol=0, atol=1e-4):
            raise ValueError(
                f"{model_name} is a legacy checkpoint without sample-rate metadata; "
                f"retrain before evaluating a {sample_rate_hz:g}-Hz cache"
            )
        model, needs_seq, log_model = build_model(model_name, classes, samples_per_window, 21)
        model.load_state_dict(meta["state_dict"])
        model.to(device).eval()
        dataset = ChannelWindowDataset(test_labels, context, 21)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
        ys, ps = [], []
        with torch.no_grad():
            for batch in loader:
                signals = batch["signals"].to(device)
                mask = batch["mask"].to(device)
                inp = signals if needs_seq else signals[:, 10]
                out = model(inp, padding_mask=mask if needs_seq else None)
                log_probs = output_log_probs(model, out, log_model)
                ys.append(batch["label"].numpy())
                ps.append(torch.exp(log_probs).cpu().numpy())
        y = np.concatenate(ys)
        p = np.concatenate(ps)
        metrics, cm = score(y, p, classes)
        metrics.update({"model": model_name, "device": str(device), "split": "test_all_eligible_windows", "best_epoch": int(meta.get("best_epoch", -1)), "label_path": str(label_path), "label_prefix": label_prefix, "samples_per_window": int(samples_per_window), "sample_rate_hz": float(sample_rate_hz)})
        results.append(metrics)
        labels_text = [f"{label_prefix}{i + 1:02d}" for i in range(classes)]
        pd.DataFrame(cm, index=labels_text, columns=labels_text).to_csv(model_dir / "confusion_matrix_all_windows.csv")
        (model_dir / "all_window_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        all_pred = test_labels[["animal_id", "session_id", "event_id", "channel_name", "start_time_ms", "end_time_ms", "global_category"]].copy()
        all_pred.rename(columns={"global_category": "y_true"}, inplace=True)
        all_pred["y_pred"] = p.argmax(1)
        for c in range(classes):
            all_pred[f"prob_{label_prefix}{c + 1:02d}"] = p[:, c]
        all_pred.to_parquet(model_dir / "all_window_test_predictions.parquet", index=False)

        pred_df = test_labels[["animal_id", "session_id", "event_id", "channel_name", "global_category"]].copy()
        for c in range(classes):
            pred_df[f"p{c}"] = p[:, c]
        pred_df["pred"] = p.argmax(1)
        agg = {"true": ("global_category", "first"), "pred": ("pred", lambda x: int(np.bincount(x.astype(int), minlength=classes).argmax()))}
        for c in range(classes):
            agg[f"p{c}"] = (f"p{c}", "mean")
        event = pred_df.groupby(["animal_id", "session_id", "event_id", "channel_name"], sort=False).agg(**agg).reset_index()
        event_probs = np.column_stack([event[f"p{c}"].to_numpy() for c in range(classes)])
        event_metrics, _ = score(event.true.to_numpy(), event_probs, classes)
        metrics.update({f"event_{key}": value for key, value in event_metrics.items() if key != "n"})
        event.to_parquet(model_dir / "event_level_test_predictions.parquet", index=False)
        (model_dir / "all_window_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        stream_frame = test_labels[["animal_id", "channel_name", "global_category"]].copy()
        for (animal, channel), indices in stream_frame.groupby(["animal_id", "channel_name"], sort=True).groups.items():
            ix = np.asarray(list(indices), dtype=int)
            stream_metrics, _ = score(y[ix], p[ix], classes)
            stream_results.append({"model": model_name, "animal_id": str(animal), "channel_name": str(channel), **stream_metrics})
    pd.DataFrame(results).to_csv(result_root / "all_window_model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stream_results).to_csv(result_root / "per_animal_channel_metrics.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(results).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
