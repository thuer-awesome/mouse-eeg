#!/usr/bin/env python
"""Evaluate frozen-category models on every eligible held-out test window."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from train_strict_morphology_classifiers import (
    ChannelWindowDataset,
    build_model,
    output_log_probs,
    seed_all,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "epilepsy_program_20260821" / "02_morphology_taxonomy" / "all_mice_old_plus_new" / "strict_all_channel_v1"


def score(y: np.ndarray, p: np.ndarray, classes: int) -> tuple[dict[str, float], np.ndarray]:
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
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
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
    args = parser.parse_args(argv)
    root = args.out.resolve()
    label_path = args.label_path.resolve() if args.label_path is not None else root / "04_label_freeze" / "frozen_channel_window_labels.parquet"
    result_root = root / args.output_subdir
    seed_all(20260823)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = pd.read_parquet(label_path)
    labels = labels.loc[labels.split.eq("test") & labels.global_category.notna()].copy()
    labels["global_category"] = labels.global_category.astype(int)
    context = pd.read_parquet(label_path)
    classes = int(labels.global_category.max() + 1)
    label_prefix = args.label_prefix or ("PG" if labels.get("global_label", pd.Series(dtype=str)).astype(str).str.startswith("PG").any() else "G")
    results = []
    stream_results = []
    model_dirs = sorted(p for p in result_root.iterdir() if p.is_dir() and (p / "best_model.pt").exists())
    for model_dir in model_dirs:
        model_name = model_dir.name
        meta = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=False)
        model, needs_seq, log_model = build_model(model_name, classes, 1000, 21)
        model.load_state_dict(meta["state_dict"])
        model.to(device).eval()
        dataset = ChannelWindowDataset(labels, context, 21)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
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
        metrics.update({"model": model_name, "device": str(device), "split": "test_all_eligible_windows", "best_epoch": int(meta.get("best_epoch", -1)), "label_path": str(label_path), "label_prefix": label_prefix})
        results.append(metrics)
        labels_text = [f"{label_prefix}{i + 1:02d}" for i in range(classes)]
        pd.DataFrame(cm, index=labels_text, columns=labels_text).to_csv(model_dir / "confusion_matrix_all_windows.csv")
        (model_dir / "all_window_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        all_pred = labels[["animal_id", "session_id", "event_id", "channel_name", "start_time_ms", "end_time_ms", "global_category"]].reset_index(drop=True).copy()
        all_pred.rename(columns={"global_category": "y_true"}, inplace=True)
        all_pred["y_pred"] = p.argmax(1)
        for c in range(classes):
            all_pred[f"prob_{label_prefix}{c + 1:02d}"] = p[:, c]
        all_pred.to_parquet(model_dir / "all_window_test_predictions.parquet", index=False)

        pred_df = labels[["animal_id", "session_id", "event_id", "channel_name", "global_category"]].reset_index(drop=True).copy()
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
        stream_frame = labels[["animal_id", "channel_name", "global_category"]].reset_index(drop=True)
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
