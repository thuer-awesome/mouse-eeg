from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

_ANIMAL_RE = re.compile(r"(?:^|[-_])E(?P<animal>\d+)(?:[-_]|$)", re.I)
_TIME_RE = re.compile(r"(?<!\d)(?P<time>\d{6})(?!\d)")
_SHARD_RE = re.compile(r"(?P<start>\d{4})-(?P<end>\d{4})")


def _require_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "读取癫痫 Parquet 需要 pyarrow。请在 sleep_staging\\.venv 中安装 "
            "pyarrow，或使用已配置 pyarrow 的 Conda Python。"
        ) from exc
    return pq


def _animal_id(path: Path) -> str:
    for part in reversed(path.parts):
        match = _ANIMAL_RE.search(part)
        if match:
            return f"E{int(match.group('animal')):02d}"
    raise ValueError(f"Cannot infer animal id from {path}")


def _session_time(path: Path) -> str:
    # The last six-digit component is the actual recording start directory.
    matches = [m.group("time") for part in path.parts for m in _TIME_RE.finditer(part)]
    return matches[-1] if matches else "unknown"


def _label_group(label_root: Path, animal: str, session_time: str) -> list[Path]:
    candidates = []
    for directory in sorted(label_root.iterdir()):
        if not directory.is_dir() or not directory.name.upper().startswith(animal):
            continue
        if session_time != "unknown" and session_time not in directory.name:
            continue
        candidates.extend(sorted(directory.rglob("*.csv")))
    return candidates


def _reviewed_shards(label_files: Iterable[Path]) -> set[int] | None:
    ranges: set[int] = set()
    found_range = False
    for path in label_files:
        match = _SHARD_RE.search(path.name)
        if match:
            found_range = True
            ranges.update(range(int(match.group("start")), int(match.group("end")) + 1))
    return ranges if found_range else None


def discover_sessions(raw_root: str | Path, label_root: str | Path) -> pd.DataFrame:
    """Discover data sessions and the matching external label files.

    Matching uses animal id plus the six-digit recording start directory.  This
    handles the nested E50/E76 export layout without guessing from file order.
    """
    raw_root, label_root = Path(raw_root), Path(label_root)
    rows: list[dict[str, Any]] = []
    for data_dir in sorted(raw_root.rglob("data")):
        parquet_files = sorted(data_dir.glob("*.parquet"))
        if not parquet_files:
            continue
        animal = _animal_id(data_dir)
        session_time = _session_time(data_dir)
        session_id = data_dir.parent.name
        label_files = _label_group(label_root, animal, session_time)
        rows.append(
            {
                "animal_id": animal,
                "session_id": session_id,
                "session_time": session_time,
                "data_dir": str(data_dir.resolve()),
                "label_files": ";".join(str(p.resolve()) for p in label_files),
                "reviewed_shards": ",".join(str(i) for i in sorted(_reviewed_shards(label_files) or [])),
                "reviewed_scope": "shards" if _reviewed_shards(label_files) is not None else "all",
                "sampling_rate_hz": 1000.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["animal_id", "session_time", "session_id"]).reset_index(drop=True)


def read_events(label_files: str | Iterable[str] | None) -> pd.DataFrame:
    paths = [] if label_files is None or (not isinstance(label_files, (str, list, tuple, set)) and pd.isna(label_files)) else (
        str(label_files).split(";") if isinstance(label_files, str) else list(label_files)
    )
    frames = []
    for path in paths:
        if not path:
            continue
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["start_time_ms", "end_time_ms", "status", "channel_name"])
    events = pd.concat(frames, ignore_index=True)
    for col in ("start_time_ms", "end_time_ms"):
        events[col] = pd.to_numeric(events[col], errors="coerce")
    return events


def merge_closed_events(events: pd.DataFrame, merge_gap_ms: int = 0) -> list[tuple[float, float]]:
    """Return channel-independent closed intervals, merging overlaps."""
    if events.empty:
        return []
    usable = events.loc[
        events.get("status", pd.Series(dtype=str)).astype(str).str.lower().eq("closed")
        & events["start_time_ms"].notna()
        & events["end_time_ms"].notna()
    ]
    intervals = sorted((float(a), float(b)) for a, b in usable[["start_time_ms", "end_time_ms"]].to_numpy() if b >= a)
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + merge_gap_ms:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(a, b) for a, b in merged]


def _window_label(start_ms: float, end_ms: float, intervals: list[tuple[float, float]], min_positive_ms: float) -> int:
    overlap = sum(max(0.0, min(end_ms, b) - max(start_ms, a)) for a, b in intervals)
    if overlap >= min_positive_ms:
        return 1
    return -1 if overlap > 0 else 0  # transition windows are excluded from training/evaluation


def _shard_number(path: Path) -> int:
    match = re.search(r"_(\d{4})\.parquet$", path.name)
    return int(match.group(1)) if match else -1


def build_window_cache(
    session_manifest: str | Path | pd.DataFrame,
    output_dir: str | Path,
    window_seconds: float = 4.0,
    stride_seconds: float = 1.0,
    target_rate_hz: float = 250.0,
    min_positive_seconds: float = 0.5,
    max_files: int | None = None,
    max_files_per_session: int | None = None,
    provisional_max_event_seconds: float | None = None,
    allow_unreviewed_long_events: bool = False,
) -> Path:
    """Materialize reviewed windows as memory-mappable NumPy arrays.

    Only CH7/CH8 are used.  The six saturated channels and vendor ``on/off/wave``
    columns are intentionally excluded.  Every valid ``closed`` interval is
    retained, including long events: the clinical team confirmed that prolonged
    seizures are expected in these acute post-drug recordings.  ``open``,
    ``orphan_end`` and ``point`` annotations remain excluded.  Every cache row
    records session, shard, start time, and label so a later split cannot
    accidentally cross animals.

    ``allow_unreviewed_long_events`` is retained as a backwards-compatible CLI
    argument but no longer gates the clinically confirmed long intervals.
    """
    pq = _require_pyarrow()
    frame = pd.read_csv(session_manifest) if not isinstance(session_manifest, pd.DataFrame) else session_manifest.copy()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    files_seen = 0
    from scipy.signal import resample_poly

    for session in frame.to_dict("records"):
        raw_events = read_events(session.get("label_files", ""))
        closed = raw_events.loc[raw_events.get("status", pd.Series(dtype=str)).astype(str).str.lower().eq("closed")].copy()
        duration_seconds = (closed["end_time_ms"] - closed["start_time_ms"]) / 1000.0
        long_count = int((duration_seconds > 300.0).sum())
        # ``allow_unreviewed_long_events`` is kept for API compatibility.
        # Clinical confirmation supersedes the old duration-only guard; duration
        # by itself is not a label-error criterion.
        if provisional_max_event_seconds is not None:
            raw_events = raw_events.loc[
                ~raw_events.get("status", pd.Series(index=raw_events.index, dtype=str)).astype(str).str.lower().eq("closed")
                | ((raw_events["end_time_ms"] - raw_events["start_time_ms"]) <= provisional_max_event_seconds * 1000.0)
            ].copy()
        events = merge_closed_events(raw_events)
        if provisional_max_event_seconds is not None and not events:
            continue
        event_shards = {int(start // 3_600_000) for start, _ in events}
        reviewed_value = session.get("reviewed_shards", "")
        reviewed_text = "" if pd.isna(reviewed_value) else str(reviewed_value)
        reviewed = set(int(x) for x in reviewed_text.split(",") if x.strip())
        restrict = str(session.get("reviewed_scope", "all")) == "shards"
        parquet_files = sorted(
            Path(session["data_dir"]).glob("*.parquet"),
            key=lambda path: (_shard_number(path) not in event_shards, _shard_number(path)),
        )
        session_files_seen = 0
        for parquet_path in parquet_files:
            shard = _shard_number(parquet_path)
            if restrict and shard not in reviewed:
                continue
            if max_files is not None and files_seen >= max_files:
                break
            if max_files_per_session is not None and session_files_seen >= max_files_per_session:
                break
            table = pq.read_table(parquet_path, columns=["timestamp_ms", "ch7", "ch8"])
            values = table.to_pandas()
            timestamps = values["timestamp_ms"].to_numpy(np.float64)
            signals = values[["ch7", "ch8"]].to_numpy(np.float32).T
            if len(timestamps) < 2 or signals.shape[1] < 100:
                continue
            native_rate = 1000.0 / max(np.median(np.diff(timestamps)), 1e-6)
            down = max(1, int(round(native_rate / target_rate_hz)))
            actual_rate = native_rate / down
            if down > 1:
                signals = resample_poly(signals, up=1, down=down, axis=1).astype(np.float32)
                timestamps = timestamps[::down][: signals.shape[1]]
            samples = int(round(window_seconds * actual_rate))
            step = max(1, int(round(stride_seconds * actual_rate)))
            count = max(0, (signals.shape[1] - samples) // step + 1)
            if count == 0:
                continue
            starts = np.arange(count, dtype=np.int64) * step
            start_ms = timestamps[starts]
            end_ms = start_ms + window_seconds * 1000.0
            labels = np.asarray([_window_label(a, b, events, min_positive_seconds * 1000.0) for a, b in zip(start_ms, end_ms)], dtype=np.int8)
            keep = labels >= 0
            if not keep.any():
                continue
            x = np.stack([signals[:, int(i): int(i) + samples] for i in starts[keep]], axis=0).astype(np.float32)
            y = labels[keep]
            cache_id = f"{session['animal_id']}_{session['session_id']}_shard{shard:04d}"
            cache_dir = output_dir / cache_id
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_dir / "signals.npy", x)
            np.save(cache_dir / "labels.npy", y)
            np.save(cache_dir / "start_time_ms.npy", start_ms[keep])
            meta = {
                "animal_id": session["animal_id"], "session_id": session["session_id"],
                "shard": shard, "sampling_rate_hz": actual_rate, "window_seconds": window_seconds,
                "stride_seconds": stride_seconds, "source": str(parquet_path), "events": len(events),
                "label_policy": (
                    "provisional_duration_filter_smoke_only"
                    if provisional_max_event_seconds is not None
                    else "all_closed_acute_post_drug_clinically_confirmed"
                ),
                "long_closed_events_over_300s_in_session": long_count,
                "clinical_context": "acute_post_drug_single_day_20260701",
            }
            (cache_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append({**meta, "cache_dir": str(cache_dir), "n_windows": int(y.size), "positive_windows": int((y == 1).sum())})
            files_seen += 1
            session_files_seen += 1
        if max_files is not None and files_seen >= max_files:
            break
    manifest_path = output_dir / "window_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return manifest_path
