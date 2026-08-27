from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import resample_poly, welch

from mouse_sleep.epilepsy.data import _require_pyarrow, read_events


_BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma": (30.0, 100.0)}


def build_physiology_qc(
    session_manifest: str | Path,
    output: str | Path,
    min_duration_seconds: float = 300.0,
    segment_seconds: float = 10.0,
    max_events: int | None = None,
) -> Path:
    """Compare suspicious label segments with nearby baseline physiology.

    The score is a screening aid, not an automatic include/exclude decision.
    It reports robust amplitude, line length, relative band power, spectral
    entropy and CH7/CH8 correlation for onset/middle/end samples.
    """
    manifest = pd.read_csv(session_manifest); rows: list[dict[str, Any]] = []
    for session in manifest.to_dict("records"):
        events = read_events(session.get("label_files", ""))
        events = events.loc[events.get("status", pd.Series(dtype=str)).astype(str).str.lower().eq("closed")].copy()
        events["duration_s"] = (events["end_time_ms"] - events["start_time_ms"]) / 1000.0
        events = events.loc[events["duration_s"] >= min_duration_seconds].sort_values("duration_s", ascending=False)
        for event in events.to_dict("records"):
            if max_events is not None and len(rows) >= max_events: break
            start, end = float(event["start_time_ms"]), float(event["end_time_ms"])
            width = segment_seconds * 1000.0
            centers = {"onset": start, "middle": max(start, (start + end - width) / 2.0), "end": max(start, end - width)}
            event_features = []
            for phase, phase_start in centers.items():
                signal, rate = _read_interval(Path(session["data_dir"]), phase_start, min(phase_start + width, end))
                features = _features(signal, rate)
                if features: event_features.append(features)
            baseline_start = max(0.0, start - 20_000.0)
            baseline_end = max(0.0, start - 10_000.0)
            if baseline_end <= baseline_start:
                baseline_start, baseline_end = end + 10_000.0, end + 20_000.0
            baseline_signal, baseline_rate = _read_interval(Path(session["data_dir"]), baseline_start, baseline_end)
            baseline = _features(baseline_signal, baseline_rate)
            combined = _mean_features(event_features)
            comparison = _compare(combined, baseline)
            rows.append({
                "animal_id": session["animal_id"], "session_id": session["session_id"],
                "event_id": event.get("event_id"), "channel_name": event.get("channel_name"),
                "start_time_s": start / 1000.0, "end_time_s": end / 1000.0,
                "duration_s": event["duration_s"], "sampled_phases": len(event_features),
                **{f"event_{k}": v for k, v in combined.items()},
                **{f"baseline_{k}": v for k, v in baseline.items()}, **comparison,
                "review_decision": "needs_clinician_review",
            })
        if max_events is not None and len(rows) >= max_events: break
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    return output


def _read_interval(data_dir: Path, start_ms: float, end_ms: float) -> tuple[np.ndarray, float]:
    pq = _require_pyarrow(); chunks = []
    first, last = int(max(0, start_ms) // 3_600_000), int(max(0, end_ms) // 3_600_000)
    for shard in range(first, last + 1):
        path = data_dir / f"eeg_{shard:04d}.parquet"
        if not path.exists(): continue
        table = pq.read_table(path, columns=["timestamp_ms", "ch7", "ch8"], filters=[("timestamp_ms", ">=", int(start_ms)), ("timestamp_ms", "<", int(end_ms))])
        if table.num_rows: chunks.append(table.to_pandas())
    if not chunks: return np.empty((2, 0), dtype=np.float32), 1000.0
    frame = pd.concat(chunks, ignore_index=True).sort_values("timestamp_ms")
    times = frame["timestamp_ms"].to_numpy(dtype=float); rate = 1000.0 / max(float(np.median(np.diff(times))), 1e-6) if len(times) > 1 else 1000.0
    return frame[["ch7", "ch8"]].to_numpy(np.float32).T, rate


def _features(signal: np.ndarray, rate: float) -> dict[str, float]:
    if signal.shape[1] < max(64, rate): return {}
    down = max(1, int(round(rate / 250.0)))
    if down > 1: signal = resample_poly(signal, 1, down, axis=1); rate /= down
    signal = signal - np.median(signal, axis=1, keepdims=True)
    robust_amplitude = float(np.mean(np.quantile(signal, .99, axis=1) - np.quantile(signal, .01, axis=1)))
    line_length = float(np.mean(np.abs(np.diff(signal, axis=1))) * rate)
    frequency, psd = welch(signal, fs=rate, nperseg=min(512, signal.shape[1]), axis=1)
    total_mask = (frequency >= 1) & (frequency <= min(100, rate / 2))
    total = np.trapezoid(psd[:, total_mask], frequency[total_mask], axis=1) + 1e-12
    features = {"robust_amplitude": robust_amplitude, "line_length_per_s": line_length, "channel_correlation": float(np.corrcoef(signal)[0, 1])}
    for name, (low, high) in _BANDS.items():
        mask = (frequency >= low) & (frequency < min(high, rate / 2))
        features[f"relative_{name}"] = float(np.mean(np.trapezoid(psd[:, mask], frequency[mask], axis=1) / total)) if mask.any() else float("nan")
    normalized = psd[:, total_mask] / np.maximum(psd[:, total_mask].sum(axis=1, keepdims=True), 1e-12)
    features["spectral_entropy"] = float(np.mean(-np.sum(normalized * np.log(normalized + 1e-12), axis=1) / math.log(normalized.shape[1])))
    return features


def _mean_features(values: list[dict[str, float]]) -> dict[str, float]:
    if not values: return {}
    return {key: float(np.nanmean([value.get(key, np.nan) for value in values])) for key in values[0]}


def _compare(event: dict[str, float], baseline: dict[str, float]) -> dict[str, float | str]:
    if not event or not baseline: return {"physiology_change_score": float("nan"), "physiology_flag": "missing_signal_or_baseline"}
    keys = ["line_length_per_s", "relative_delta", "relative_theta", "relative_beta", "relative_gamma"]
    ratios = {f"log2_ratio_{key}": float(np.log2((event[key] + 1e-12) / (baseline[key] + 1e-12))) for key in keys}
    score = float(np.nanmax(np.abs(list(ratios.values()))))
    return {**ratios, "physiology_change_score": score, "physiology_flag": "clear_change_candidate" if score >= .5 else "weak_change_candidate"}
