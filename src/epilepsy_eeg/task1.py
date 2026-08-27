"""Task 1: channel-blind binary seizure annotation and clinical PDF output."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from mouse_sleep.epilepsy.corrected_event_decoder import (
    build_sequences,
    evaluate_sequences,
)
from mouse_sleep.epilepsy.seizure_annotation import DecoderConfig, decode_window_intervals
from mouse_sleep.epilepsy.single_channel_inference import run_continuous_single_channel_inference

from .paths import MODEL_ORDER_TASK1, read_yaml, sha256, write_json
from .raw_adapter import build_protocol

MODEL_LABELS = {
    "eegwavenet_sc": "EEGWaveNet-SC",
    "physio_sequence_no_aux": "Physio-Sequence",
    "physio_sequence_stats": "Physio + statistics",
}


def _decoder_config(path: str | Path | None, fallback: dict[str, Any] | None = None) -> DecoderConfig:
    values: dict[str, Any] = {}
    if path is not None and Path(path).is_file():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        values = dict(data.get("config", data.get("selected_config", {})))
    if not values and fallback:
        values = dict(fallback)
    allowed = {name for name in DecoderConfig.__dataclass_fields__}
    values = {key: value for key, value in values.items() if key in allowed}
    return DecoderConfig(**values)


def _candidate_events(frame: pd.DataFrame, config: DecoderConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    tracks: list[pd.DataFrame] = []
    required = {"animal_id", "session_id", "channel_name", "continuity_id", "start_time_ms", "end_time_ms", "seizure_probability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"预测表缺少列: {sorted(missing)}")
    for key, group in frame.groupby(["animal_id", "session_id", "channel_name", "continuity_id"], sort=True):
        group = group.sort_values("start_time_ms", kind="stable").copy()
        decoded = decode_window_intervals(
            group["seizure_probability"].to_numpy(float),
            group["start_time_ms"].to_numpy(float),
            config,
            group["end_time_ms"].to_numpy(float),
        )
        for idx, event in enumerate(decoded.events):
            rows.append({
                "animal_id": str(key[0]),
                "session_id": str(key[1]),
                "channel_name": str(key[2]),
                "continuity_id": str(key[3]),
                "predicted_event_index": idx,
                "start_time_ms": float(event.start_ms),
                "end_time_ms": float(event.end_ms),
                "duration_s": float(event.duration_ms / 1000.0),
                "decoder_interval_support": "full_model_window_union",
            })
        group["smoothed_probability"] = decoded.smoothed_probability
        group["decoded_mask"] = decoded.seizure_mask.astype(np.int8)
        tracks.append(group)
    events = pd.DataFrame(rows, columns=["animal_id", "session_id", "channel_name", "continuity_id", "predicted_event_index", "start_time_ms", "end_time_ms", "duration_s", "decoder_interval_support"])
    track = pd.concat(tracks, ignore_index=True) if tracks else frame.iloc[:0].copy()
    return events, track


def _event_overlap_summary(events: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    columns = ["animal_id", "session_id", "channel_name", "event_id", "start_time_ms", "end_time_ms", "coverage", "max_iou", "category"]
    if events.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for truth in events.itertuples(index=False):
        start, end = float(truth.start_time_ms), float(truth.end_time_ms)
        candidates = predicted.loc[
            predicted["animal_id"].astype(str).eq(str(truth.animal_id))
            & predicted["session_id"].astype(str).eq(str(truth.session_id))
            & predicted["channel_name"].astype(str).eq(str(truth.channel_name))
        ]
        max_iou = 0.0
        intervals = []
        for pred in candidates.itertuples(index=False):
            ps, pe = float(pred.start_time_ms), float(pred.end_time_ms)
            overlap = max(0.0, min(end, pe) - max(start, ps))
            if overlap <= 0:
                continue
            intervals.append((max(start, ps), min(end, pe)))
            iou = overlap / max(1.0, (end - start) + (pe - ps) - overlap)
            max_iou = max(max_iou, iou)
        merged: list[list[float]] = []
        for left, right in sorted(intervals):
            if not merged or left > merged[-1][1]:
                merged.append([left, right])
            else:
                merged[-1][1] = max(merged[-1][1], right)
        covered = sum(right - left for left, right in merged)
        coverage = covered / max(1.0, end - start)
        category = "full" if coverage >= 0.9 else "partial" if coverage >= 0.5 else "low" if coverage > 0 else "missed"
        rows.append({"animal_id": str(truth.animal_id), "session_id": str(truth.session_id), "channel_name": str(truth.channel_name), "event_id": str(getattr(truth, "event_id", len(rows))), "start_time_ms": start, "end_time_ms": end, "coverage": coverage, "max_iou": max_iou, "category": category})
    return pd.DataFrame(rows, columns=columns)


def render_clinical_pdf(track: pd.DataFrame, events: pd.DataFrame, output: str | Path, *, model_label: str, page_seconds: float = 120.0, cache_dir_column: str = "filtered_cache_dir", local_index_column: str = "filtered_local_index") -> Path:
    """Render one channel per page: waveform, probability, and decoded state."""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(target) as pdf:
        for key, group in track.groupby(["animal_id", "session_id", "channel_name", "continuity_id"], sort=True):
            group = group.sort_values("start_time_ms", kind="stable").reset_index(drop=True)
            if group.empty:
                continue
            first, last = float(group.start_time_ms.iloc[0]), float(group.end_time_ms.iloc[-1])
            page_start = first
            while page_start < last:
                page_end = page_start + page_seconds * 1000.0
                part = group.loc[(group.start_time_ms < page_end) & (group.end_time_ms > page_start)].copy()
                if part.empty:
                    page_start = page_end
                    continue
                fig, axes = plt.subplots(3, 1, figsize=(11.0, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2, 0.55]})
                # Stitch the one-second overlap into a compact trace.  The
                # plot is downsampled only for display; source windows remain
                # untouched in the parquet output.
                signal_parts = []
                for row in part.itertuples(index=False):
                    root = getattr(row, cache_dir_column, None)
                    local = getattr(row, local_index_column, None)
                    if root is None or local is None:
                        continue
                    try:
                        arr = np.load(Path(str(root)) / "signals.npy", mmap_mode="r")
                        ci = 0 if str(row.channel_name).upper() == "CH7" else 1
                        signal_parts.append(np.asarray(arr[int(local), ci], dtype=float))
                    except (FileNotFoundError, OSError, ValueError, IndexError):
                        signal_parts.append(np.zeros(1000, dtype=float))
                if signal_parts:
                    raw = np.concatenate([signal_parts[0], *[item[-250:] for item in signal_parts[1:]]])
                    x = np.linspace(float(part.start_time_ms.iloc[0]), float(part.end_time_ms.iloc[-1]), len(raw))
                    keep = max(1, len(raw) // 6000)
                    raw = raw[::keep]
                    x = x[::keep]
                    med = float(np.nanmedian(raw)); scale = float(np.nanpercentile(raw, 98) - np.nanpercentile(raw, 2))
                    scale = max(scale, 1e-6)
                    axes[0].plot((x - page_start) / 1000.0, (raw - med) / scale, color="#30363b", linewidth=0.45, rasterized=True)
                axes[0].set_ylabel("EEG (robust a.u.)")
                axes[1].plot((part.center_time_ms - page_start) / 1000.0, part.seizure_probability, color="#1f6f8b", linewidth=0.8, label="probability")
                if "smoothed_probability" in part:
                    axes[1].plot((part.center_time_ms - page_start) / 1000.0, part.smoothed_probability, color="#c45a32", linewidth=0.8, label="smoothed")
                axes[1].axhline(0.7, color="#8a9298", linestyle="--", linewidth=0.55)
                axes[1].set_ylim(-0.02, 1.02); axes[1].set_ylabel("P(seizure)")
                axes[2].step((part.center_time_ms - page_start) / 1000.0, part.decoded_mask, where="mid", color="#c45a32", linewidth=1.2)
                axes[2].set_ylim(-0.05, 1.05); axes[2].set_yticks([0, 1], ["non-seizure", "seizure"])
                axes[2].set_xlabel("Time within page (s)")
                truth = events.loc[
                    events["animal_id"].astype(str).eq(str(key[0]))
                    & events["session_id"].astype(str).eq(str(key[1]))
                    & events["channel_name"].astype(str).eq(str(key[2]))
                    & events["start_time_ms"].astype(float).lt(page_end)
                    & events["end_time_ms"].astype(float).gt(page_start)
                ] if not events.empty else events
                for row in truth.itertuples(index=False):
                    left = (max(page_start, float(row.start_time_ms)) - page_start) / 1000.0
                    right = (min(page_end, float(row.end_time_ms)) - page_start) / 1000.0
                    axes[0].axvspan(left, right, color="#2f6b8a", alpha=0.16)
                    axes[2].axvspan(left, right, color="#2f6b8a", alpha=0.16)
                for row in pd.DataFrame(_event_rows_in_page(group, page_start, page_end)).itertuples(index=False):
                    axes[0].axvspan(row.left, row.right, facecolor="none", edgecolor="#c45a32", linewidth=1.0)
                    axes[2].axvspan(row.left, row.right, facecolor="none", edgecolor="#c45a32", linewidth=1.0)
                axes[0].set_title(f"{model_label} | {key[0]} / {key[1]} / {key[2]} | {page_start/1000:.1f}-{page_end/1000:.1f}s", loc="left", fontsize=9, fontweight="bold")
                for axis in axes:
                    axis.spines[["top", "right"]].set_visible(False)
                    axis.grid(axis="x", color="#e5e8ea", linewidth=0.4)
                fig.tight_layout(h_pad=0.8)
                pdf.savefig(fig, dpi=200)
                plt.close(fig)
                page_start = page_end
    return target


def _event_rows_in_page(group: pd.DataFrame, page_start: float, page_end: float) -> list[dict[str, float]]:
    if "predicted_event_start_time_ms" not in group.columns:
        # Recover contiguous mask runs from the decoded window intervals.
        rows = []
        mask = group.decoded_mask.to_numpy(bool)
        starts = group.start_time_ms.to_numpy(float)
        ends = group.end_time_ms.to_numpy(float)
        begin = None
        for i, active in enumerate(np.r_[mask, False]):
            if active and begin is None:
                begin = i
            if not active and begin is not None:
                left = (max(page_start, starts[begin]) - page_start) / 1000.0
                right = (min(page_end, ends[i - 1]) - page_start) / 1000.0
                if right > left:
                    rows.append({"left": left, "right": right})
                begin = None
        return rows
    return []


def run_annotation(
    *,
    protocol_dir: str | Path | None,
    source: str | Path | None,
    labels: str | Path | None,
    checkpoint_root: str | Path,
    output_dir: str | Path,
    models: Iterable[str] = MODEL_ORDER_TASK1,
    decoder_root: str | Path | None = None,
    device: str = "auto",
    sampling_rate_hz: float | None = None,
    labels_time_base: str = "absolute_ms",
    page_seconds: float = 120.0,
    allow_existing: bool = False,
    config_path: str | Path | None = None,
    target_rate_hz: float | None = None,
    window_seconds: float | None = None,
    stride_seconds: float | None = None,
    ch7_column: str | None = None,
    ch8_column: str | None = None,
    filter_overrides: dict[str, Any] | None = None,
) -> Path:
    """Run inference for one or more models and write a clinical review bundle."""
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not allow_existing:
        raise FileExistsError(f"输出目录非空: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cfg_data = read_yaml(config_path) if config_path else {}
    if protocol_dir is None:
        if source is None:
            raise ValueError("必须提供 --protocol-dir 或 --input")
        preprocessing = dict(cfg_data.get("preprocessing", {}))
        filter_values = dict(preprocessing.get("filter", cfg_data.get("filter", {})))
        if filter_overrides:
            filter_values.update({key: value for key, value in filter_overrides.items() if value is not None})
        from mouse_sleep.epilepsy.signal_processing import EEGFilterConfig
        protocol = build_protocol(
            source,
            output / "prepared_input",
            labels=labels,
            sampling_rate_hz=sampling_rate_hz,
            target_rate_hz=float(target_rate_hz if target_rate_hz is not None else preprocessing.get("target_rate_hz", cfg_data.get("sampling_rate_hz", 250.0))),
            window_seconds=float(window_seconds if window_seconds is not None else preprocessing.get("window_seconds", cfg_data.get("window_seconds", 4.0))),
            stride_seconds=float(stride_seconds if stride_seconds is not None else preprocessing.get("stride_seconds", cfg_data.get("stride_seconds", 1.0))),
            labels_time_base=labels_time_base,
            ch7_column=ch7_column or preprocessing.get("ch7_column") or cfg_data.get("ch7_column"),
            ch8_column=ch8_column or preprocessing.get("ch8_column") or cfg_data.get("ch8_column"),
            filter_config=EEGFilterConfig(**filter_values) if filter_values else None,
        )
    else:
        protocol = Path(protocol_dir).resolve()
    checkpoint_base = Path(checkpoint_root).resolve()
    decoder_base = Path(decoder_root).resolve() if decoder_root else None
    all_records: list[dict[str, Any]] = []
    for model in models:
        if model not in MODEL_ORDER_TASK1:
            raise ValueError(f"未知任务一模型: {model}")
        checkpoint_dir = checkpoint_base / model
        checkpoint_file = checkpoint_dir / "model.pt"
        # A freshly trained Task-2-style export may call its checkpoint
        # ``best_model.pt``.  Keep the adapter permissive while preserving the
        # audited Task-1 naming in the release bundle.
        if not checkpoint_file.is_file() and (checkpoint_dir / "best_model.pt").is_file():
            checkpoint_file = checkpoint_dir / "best_model.pt"
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"缺少任务一权重: {checkpoint_dir / 'model.pt'}")
        inference_dir = output / "inference" / model
        inference_model_dir = checkpoint_dir
        if checkpoint_file.name != "model.pt":
            inference_model_dir = output / "_checkpoint_adapters" / model
            inference_model_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint_file, inference_model_dir / "model.pt")
        prediction_path = run_continuous_single_channel_inference(protocol, inference_model_dir, inference_dir, device=device)
        predictions = pd.read_parquet(prediction_path)
        frozen_decoder = None
        if decoder_base:
            frozen_decoder = decoder_base / "models" / model / "decoder_fit.json"
        if frozen_decoder is None or not frozen_decoder.is_file():
            candidate = checkpoint_dir / "decoder_fit.json"
            frozen_decoder = candidate if candidate.is_file() else None
        decoder = _decoder_config(frozen_decoder, cfg_data.get("decoder"))
        predicted_events, track = _candidate_events(predictions, decoder)
        predicted_events.insert(0, "model", model)
        track.insert(0, "model", model)
        model_out = output / "models" / model
        model_out.mkdir(parents=True, exist_ok=True)
        predicted_events.to_parquet(model_out / "predicted_events.parquet", index=False)
        track.to_parquet(model_out / "decoded_tracks.parquet", index=False)
        labels_frame = pd.read_parquet(protocol / "channel_events.parquet") if (protocol / "channel_events.parquet").is_file() else pd.DataFrame()
        summary = _event_overlap_summary(labels_frame, predicted_events) if not labels_frame.empty else pd.DataFrame()
        if not summary.empty:
            summary.insert(0, "model", model)
            summary.to_csv(model_out / "event_assessments.csv", index=False, encoding="utf-8-sig")
        pdf = render_clinical_pdf(track, labels_frame, model_out / "clinical_event_atlas.pdf", model_label=MODEL_LABELS[model], page_seconds=page_seconds)
        all_records.append({"model": model, "checkpoint": str(checkpoint_file), "prediction_path": str(prediction_path), "predicted_events": int(len(predicted_events)), "labels": int(len(labels_frame)), "decoder": asdict(decoder), "pdf": str(pdf), "checkpoint_sha256": sha256(checkpoint_file)})
    manifest = {"schema_version": 1, "task": "annotation_detection", "protocol_dir": str(protocol), "source": str(source) if source else None, "models": all_records, "offline_only": True, "amplitude_unit": "a.u.", "decoder_selection": "frozen validation-only configuration; no test fitting"}
    write_json(output / "run_manifest.json", manifest)
    (output / "README_zh.md").write_text("# 任务一临床复核输出\n\n每个有效通道独立推理。PDF 每页依次显示滤波 EEG、概率轨迹和连续解码状态；蓝色区间是输入标签（若提供），橙色边框是模型区间。无标签输入只生成模型候选，不能据此计算灵敏度或精确率。\n\n本 CLI 使用离线零相位滤波，适合回放和医生在环复核，不是图机实时报警接口。\n", encoding="utf-8")
    return output
