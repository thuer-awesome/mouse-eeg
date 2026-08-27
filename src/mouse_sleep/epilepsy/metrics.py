from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score


def window_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float = .5) -> dict[str, float | int]:
    y_true, probability = np.asarray(y_true).astype(int), np.asarray(probability, dtype=float)
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else float("nan")
    specificity = float(tn / (tn + fp)) if tn + fp else float("nan")
    balanced = float(np.nanmean([sensitivity, specificity]))
    accuracy = float((tp + tn) / max(len(y_true), 1))
    out: dict[str, float | int] = {"n_windows": int(len(y_true)), "positive_windows": int(y_true.sum()), "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity, "f1": float(f1_score(y_true, pred, zero_division=0)), "balanced_accuracy": balanced, "mcc": float(matthews_corrcoef(y_true, pred)) if len(np.unique(y_true)) == 2 else float("nan")}
    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, probability)); out["auprc"] = float(average_precision_score(y_true, probability))
    else:
        out["auroc"] = float("nan"); out["auprc"] = float("nan")
    return out


def event_metrics(y_true: np.ndarray, probability: np.ndarray, start_time_ms: np.ndarray, window_seconds: float = 4.0, threshold: float = .5, merge_gap_seconds: float = 2.0) -> dict[str, float | int]:
    """Approximate event sensitivity, false alarms/hour, and detection latency."""
    starts = np.asarray(start_time_ms, dtype=float); p = np.asarray(probability, dtype=float); y = np.asarray(y_true).astype(int)
    order = np.argsort(starts); starts, p, y = starts[order], p[order], y[order]
    truth = _runs(starts, y.astype(bool), window_seconds * 1000, merge_gap_seconds * 1000)
    pred = _runs(starts, p >= threshold, window_seconds * 1000, merge_gap_seconds * 1000)
    matched = 0; latencies = []; used = set()
    for true_start, true_end in truth:
        hits = [(i, max(0., min(true_end, b) - max(true_start, a))) for i, (a, b) in enumerate(pred) if i not in used and min(true_end, b) > max(true_start, a)]
        if hits:
            i, _ = max(hits, key=lambda x: x[1]); used.add(i); matched += 1; latencies.append(max(0., pred[i][0] - true_start) / 1000.)
    duration_h = max((starts[-1] - starts[0] + window_seconds * 1000) / 3_600_000., 1e-9) if len(starts) else 0.
    return {"true_events": len(truth), "predicted_events": len(pred), "event_sensitivity": float(matched / max(len(truth), 1)), "false_alarms": int(len(pred) - len(used)), "false_alarms_per_hour": float((len(pred) - len(used)) / max(duration_h, 1e-9)), "median_latency_s": float(np.median(latencies)) if latencies else float("nan")}


def _runs(starts: np.ndarray, mask: np.ndarray, window_ms: float, gap_ms: float) -> list[tuple[float, float]]:
    out = []; current = None
    for start, active in zip(starts, mask):
        if not active:
            continue
        end = start + window_ms
        if current is None or start > current[1] + gap_ms: out.append((float(start), float(end))); current = list(out[-1])
        else: out[-1] = (out[-1][0], float(max(out[-1][1], end))); current = list(out[-1])
    return out
