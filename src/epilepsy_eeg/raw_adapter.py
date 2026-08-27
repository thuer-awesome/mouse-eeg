"""Conservative raw EEG input adapter used by the annotation CLI."""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from mouse_sleep.epilepsy.signal_processing import EEGFilterConfig, filter_continuous_eeg

CHANNELS = ("CH7", "CH8")


def _lookup(columns: Iterable[Any]) -> dict[str, Any]:
    return {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in columns}


def _find(columns: Iterable[Any], aliases: Iterable[str]) -> Any | None:
    lookup = _lookup(columns)
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]", "", alias.lower())
        if key in lookup:
            return lookup[key]
    return None


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")[:120] or "recording"


def _animal(path: Path, explicit: str | None) -> str:
    if explicit:
        return str(explicit).upper()
    # Include both POSIX and Windows path separators.  Matching the complete
    # path (rather than only ``path.name``) keeps names such as ``E50\r0``
    # discoverable regardless of the host platform.
    match = re.search(r"(?:^|[-_\\/])E(\d+)(?:[-_\\/.]|$)", str(path), flags=re.I)
    return f"E{int(match.group(1)):02d}" if match else "UNKNOWN"


def discover_files(source: str | Path) -> list[Path]:
    path = Path(source).expanduser().resolve()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".parquet", ".csv", ".npy", ".npz", ".edf", ".bdf", ".fif"}]
    # Vendor exports often contain ``name(1).parquet`` alongside the canonical
    # file.  Prefer the canonical file so an input directory is deterministic.
    names = {p.name.lower() for p in files}
    unique = []
    for p in sorted(files):
        base = re.sub(r"\(\d+\)(?=\.[^.]+$)", "", p.name).lower()
        if p.name.lower() != base and base in names:
            continue
        unique.append(p)
    if not unique:
        raise FileNotFoundError(f"没有可识别的 EEG 文件: {path}")
    return unique


def discover_label_files(source: str | Path) -> list[Path]:
    """Return deterministic CSV/Parquet label tables from a file or folder.

    Clinical exports are often split into one table per mouse or recording.
    Keeping discovery here (rather than in the orchestration layer) means the
    lower-level ``prepare`` command and the end-to-end runners use exactly the
    same input semantics.  The directory must contain label tables only; raw
    EEG files are intentionally ignored.
    """

    path = Path(source).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in {".csv", ".parquet", ".pq"}:
            raise ValueError(f"不支持的标签格式: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = [
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in {".csv", ".parquet", ".pq"}
    ]
    # Apply the same vendor-duplicate rule as EEG discovery.
    names = {item.name.lower() for item in files}
    unique: list[Path] = []
    for item in sorted(files):
        base = re.sub(r"\(\d+\)(?=\.[^.]+$)", "", item.name).lower()
        if item.name.lower() != base and base in names:
            continue
        unique.append(item)
    if not unique:
        raise FileNotFoundError(f"没有可识别的标签表: {path}")
    return unique


def _read(path: Path, rate: float | None, ch7: str | None, ch8: str | None) -> tuple[np.ndarray, np.ndarray, float]:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".csv"}:
        frame = pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path)
        c7 = ch7 or _find(frame.columns, ("ch7", "channel7", "channel_7", "eeg7"))
        c8 = ch8 or _find(frame.columns, ("ch8", "channel8", "channel_8", "eeg8"))
        if c7 is None or c8 is None:
            raise ValueError(f"{path} 缺少 CH7/CH8 列，请用 --ch7-column/--ch8-column 指定")
        values = frame[[c7, c8]].apply(pd.to_numeric, errors="coerce").to_numpy(float).T
        time_col = _find(frame.columns, ("timestamp_ms", "time_ms", "timestamp", "time"))
        if time_col is None:
            if rate is None or rate <= 0:
                raise ValueError(f"{path} 无时间列，必须提供 --sampling-rate")
            timestamps = np.arange(values.shape[1], dtype=float) * 1000.0 / rate
            native = float(rate)
        else:
            timestamps = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(float)
            keep = np.isfinite(timestamps)
            timestamps, values = timestamps[keep], values[:, keep]
            order = np.argsort(timestamps, kind="stable")
            timestamps, values = timestamps[order], values[:, order]
            if len(timestamps) < 2:
                raise ValueError(f"{path} 时间列有效值不足")
            delta = float(np.nanmedian(np.diff(timestamps)))
            if not np.isfinite(delta) or delta <= 0:
                raise ValueError(f"{path} 时间戳不是递增毫秒轴")
            native = 1000.0 / delta
    elif suffix in {".npy", ".npz"}:
        if rate is None or rate <= 0:
            raise ValueError("NPY/NPZ 输入必须提供 --sampling-rate")
        if suffix == ".npy":
            raw = np.load(path, mmap_mode="r")
            timestamp = None
        else:
            archive = np.load(path, allow_pickle=False)
            key = next((k for k in ("signals", "data", "eeg", "x") if k in archive), None)
            if key is None:
                raise ValueError(f"{path} NPZ 缺少 signals/data/eeg/x 数组")
            raw, timestamp = archive[key], archive["timestamp_ms"] if "timestamp_ms" in archive else None
        raw = np.asarray(raw)
        if raw.ndim != 2:
            raise ValueError(f"{path} 数组必须二维，收到 {raw.shape}")
        if raw.shape[0] >= 2 and raw.shape[1] > raw.shape[0]:
            values = raw[:2].astype(float)
        elif raw.shape[1] >= 2 and raw.shape[0] > raw.shape[1]:
            values = raw[:, :2].T.astype(float)
        else:
            raise ValueError(f"{path} 无法判断通道轴: {raw.shape}")
        native = float(rate)
        timestamps = np.asarray(timestamp, float) if timestamp is not None else np.arange(values.shape[1], dtype=float) * 1000.0 / native
    elif suffix in {".edf", ".bdf", ".fif"}:
        try:
            import mne
        except ImportError as exc:
            raise RuntimeError("EDF/BDF/FIF 需要可选依赖，请安装 `pip install -e '.[edf]'`") from exc
        raw = mne.io.read_raw(path, preload=True, verbose="ERROR")
        names = {str(n).upper(): n for n in raw.ch_names}
        c7 = next((n for key, n in names.items() if re.search(r"(?:CH|CHANNEL)\s*7$", key)), None)
        c8 = next((n for key, n in names.items() if re.search(r"(?:CH|CHANNEL)\s*8$", key)), None)
        if c7 is None or c8 is None:
            raise ValueError(f"{path} 未找到 CH7/CH8 通道")
        values = raw.get_data(picks=[c7, c8]).astype(float)
        native = float(raw.info["sfreq"])
        timestamps = np.arange(values.shape[1], dtype=float) * 1000.0 / native
    else:
        raise ValueError(f"不支持的输入格式: {path.suffix}")
    if len(timestamps) != values.shape[1]:
        raise ValueError(f"{path} 时间轴长度与信号长度不一致")
    return values, timestamps, native


def _resample(values: np.ndarray, timestamps: np.ndarray, native: float, target: float) -> tuple[np.ndarray, np.ndarray]:
    if abs(native - target) < 1e-3:
        return values.astype(np.float32), timestamps.astype(float)
    ratio = Fraction(float(target) / float(native)).limit_denominator(2000)
    output = resample_poly(values, ratio.numerator, ratio.denominator, axis=-1).astype(np.float32)
    start = float(timestamps[0]) if len(timestamps) else 0.0
    out_time = start + np.arange(output.shape[-1], dtype=float) * 1000.0 / target
    return output, out_time


def _valid_channels(values: np.ndarray) -> np.ndarray:
    result = np.ones(2, dtype=bool)
    for i, channel in enumerate(values):
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            result[i] = False
            continue
        rail = float((np.abs(finite) >= 1e5).mean())
        spread = float(np.percentile(finite, 95) - np.percentile(finite, 5))
        result[i] = float(np.isfinite(channel).mean()) >= 0.995 and rail < 0.01 and spread > 1e-6
    return result


def _baseline(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    center = float(np.median(finite))
    scale = float(np.median(np.abs(finite - center)) * 1.4826)
    return center, max(scale, 1e-3)


def _source_identity_matches(source: Path, *, animal: str, session: str) -> bool:
    """Conservatively route identity-free label files by filename.

    A directory of clinical exports is commonly split into one file per mouse
    or recording, but those files do not always contain explicit identity
    columns.  We only filter when the filename contains an unambiguous mouse
    token (``E50``/``E77`` etc.); generic names such as ``01_first.csv`` remain
    shared tables and are merged for the current recording.  This avoids both
    silent cross-mouse label leakage and surprising exclusion of generic label
    tables.
    """

    stem = source.stem.upper()
    tokens = re.findall(r"(?<![A-Z0-9])E\d+(?![A-Z0-9])", stem)
    if tokens and str(animal).upper() not in set(tokens):
        return False
    # Session names are vendor-specific and often contain punctuation or
    # generated counters.  We therefore route identity-free files by animal
    # token only; the preflight above rejects ambiguous same-animal/multi-
    # session inputs instead of guessing from a filename.
    return True


def _read_label_source(source: Path) -> pd.DataFrame:
    """Read one label table with a deterministic, extension-aware loader."""

    suffix = source.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffix == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"不支持的标签格式: {source.suffix}")


def _label_identity_columns(frame: pd.DataFrame) -> tuple[Any | None, Any | None]:
    """Return the optional animal/session columns used for routing."""

    return (
        _find(frame.columns, ("animal_id", "animal", "mouse", "mouse_id")),
        _find(frame.columns, ("session_id", "session", "recording", "recording_id")),
    )


def _norm_identity(value: Any, *, session: bool = False) -> str:
    """Normalize common spreadsheet identity spellings for exact joins."""

    if pd.isna(value):
        return ""
    text = str(value).strip()
    if session:
        text = Path(text).stem
    # Excel often serializes identifiers such as E50 or 1 as ``E50.0``/``1.0``.
    if text.endswith(".0") and text[:-2]:
        text = text[:-2]
    return re.sub(r"\s+", "", text).upper()


def _label_source_preflight(
    sources: list[Path],
    recordings: list[tuple[str, str, Path]],
) -> list[dict[str, Any]]:
    """Validate label-to-recording routing before any cache is written.

    A single identity-free table is acceptable for one recording (the common
    offline use case).  With multiple recordings, every identity-free table
    must be unambiguously named for exactly one recording; otherwise applying
    it to each recording would duplicate labels and silently corrupt event
    counts.  Tables with identity columns are checked for rows that cannot be
    mapped to an input recording.  The returned records are persisted as an
    audit artifact by :func:`build_protocol`.
    """

    known = {(str(animal).upper(), str(session)) for animal, session, _ in recordings}
    audit: list[dict[str, Any]] = []
    errors: list[str] = []
    for source in sources:
        frame = _read_label_source(source)
        animal_col, session_col = _label_identity_columns(frame)
        row_count = int(len(frame))
        profile: dict[str, Any] = {
            "source": str(source.resolve()),
            "rows": row_count,
            "animal_column": str(animal_col) if animal_col is not None else None,
            "session_column": str(session_col) if session_col is not None else None,
            "routing": "identity_columns" if animal_col is not None or session_col is not None else "filename_or_shared",
            "status": "ok",
            "unmatched_rows": 0,
            "ambiguous_rows": 0,
            "candidate_recordings": [],
        }
        if animal_col is None and session_col is None:
            candidates = [
                (animal, session)
                for animal, session, _ in recordings
                if _source_identity_matches(source, animal=animal, session=session)
            ]
            profile["candidate_recordings"] = [f"{a}/{s}" for a, s in candidates]
            if len(recordings) > 1 and len(candidates) != 1:
                profile["status"] = "ambiguous_identity_free"
                errors.append(
                    f"标签表 {source.name} 无 animal/session 列，无法唯一映射到一个原始记录 "
                    f"(候选 {len(candidates)} 个；请拆分文件或补充 identity 列)"
                )
            audit.append(profile)
            continue

        # Check each row independently.  Missing identity values are not
        # guessed when several raw recordings are present.
        unmatched = 0
        ambiguous = 0
        for _, row in frame.iterrows():
            animal_value = _norm_identity(row[animal_col]) if animal_col is not None else ""
            session_value = _norm_identity(row[session_col], session=True) if session_col is not None else ""
            animal_value = animal_value or None
            session_value = session_value or None
            matches = [
                (a, s)
                for a, s in known
                if (animal_value is None or _norm_identity(a) == animal_value)
                and (session_value is None or _norm_identity(s, session=True) == session_value)
            ]
            if not matches:
                unmatched += 1
            elif len(matches) > 1:
                ambiguous += 1
        profile["unmatched_rows"] = int(unmatched)
        profile["ambiguous_rows"] = int(ambiguous)
        if unmatched or ambiguous:
            profile["status"] = (
                "unmatched_identity_rows" if unmatched else "ambiguous_identity_rows"
            )
            errors.append(
                f"标签表 {source.name} 有 {unmatched}/{row_count} 行无法映射、"
                f"{ambiguous}/{row_count} 行映射到多个输入记录；"
                "请核对 animal_id/session_id 或输入文件名"
            )
        audit.append(profile)
    if errors:
        detail = "\n".join(f"- {item}" for item in errors[:8])
        if len(errors) > 8:
            detail += f"\n- ...另有 {len(errors) - 8} 项"
        raise ValueError("标签映射预检失败，未写入训练协议：\n" + detail)
    return audit


def _read_labels(
    path: str | Path | None,
    *,
    timestamps: np.ndarray,
    animal: str,
    session: str,
    valid: np.ndarray,
    time_base: str,
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    columns = ["animal_id", "session_id", "channel_name", "channels", "event_id", "source_event_id", "start_time_ms", "end_time_ms", "status", "split"]
    if path is None:
        empty = pd.DataFrame(columns=columns)
        return (empty, {"source_files": [], "rows_read": 0, "rows_selected": 0, "rows_invalid_time": 0, "rows_dropped_invalid_channel": 0, "channel_rows": 0}) if return_audit else empty
    sources = discover_label_files(path)
    selected_frames: list[pd.DataFrame] = []
    rows_read = 0
    rows_selected = 0
    source_records: list[dict[str, Any]] = []
    for source in sources:
        frame = _read_label_source(source)
        source_rows = int(len(frame))
        rows_read += source_rows
        # A label file may cover several recordings.  Restrict each source
        # before concatenation so rows from identity-free files do not become
        # NaN metadata and get dropped when another source has identity cols.
        animal_col = _find(frame.columns, ("animal_id", "animal", "mouse", "mouse_id"))
        session_col = _find(frame.columns, ("session_id", "session", "recording", "recording_id"))
        identity_routed = True
        if animal_col is not None:
            frame = frame.loc[
                frame[animal_col].map(_norm_identity).eq(_norm_identity(animal))
            ].copy()
        elif not _source_identity_matches(source, animal=animal, session=session):
            source_records.append({
                "source": str(source.resolve()),
                "rows": source_rows,
                "rows_selected": 0,
                "identity_routed": False,
            })
            continue
        if session_col is not None and len(frame):
            session_values = frame[session_col].map(lambda value: _norm_identity(value, session=True))
            target_session = _norm_identity(session, session=True)
            frame = frame.loc[
                session_values.eq(target_session)
            ].copy()
        if len(frame):
            selected_frames.append(frame)
            rows_selected += int(len(frame))
        source_records.append({
            "source": str(source.resolve()),
            "rows": source_rows,
            "rows_selected": int(len(frame)),
            "identity_routed": identity_routed,
        })
    if not selected_frames:
        frame = pd.DataFrame()
    elif len(selected_frames) == 1:
        frame = selected_frames[0].copy()
    else:
        frame = pd.concat(selected_frames, ignore_index=True, sort=False)
    frame = frame.reset_index(drop=True)
    if frame.empty:
        empty = pd.DataFrame(columns=columns)
        audit = {"source_files": source_records, "rows_read": rows_read, "rows_selected": rows_selected, "rows_invalid_time": 0, "rows_dropped_invalid_channel": 0, "channel_rows": 0}
        return (empty, audit) if return_audit else empty
    start_col = _find(frame.columns, ("start_time_ms", "start_ms", "start", "onset_ms", "on"))
    end_col = _find(frame.columns, ("end_time_ms", "end_ms", "end", "offset_ms", "off"))
    if start_col is None or end_col is None:
        raise ValueError("标签必须包含 start/end 时间列")
    out = pd.DataFrame({"start_time_ms": pd.to_numeric(frame[start_col], errors="coerce"), "end_time_ms": pd.to_numeric(frame[end_col], errors="coerce")})
    source_event_col = _find(frame.columns, ("event_id", "event", "episode_id", "id"))
    if source_event_col is None:
        out["source_event_id"] = [f"row_{i:06d}" for i in range(len(out))]
    else:
        out["source_event_id"] = frame[source_event_col].astype(str).to_numpy()
    if time_base == "relative_ms":
        out[["start_time_ms", "end_time_ms"]] += float(timestamps[0])
    elif time_base != "absolute_ms":
        raise ValueError("--labels-time-base 必须为 absolute_ms 或 relative_ms")
    channel_col = _find(frame.columns, ("channel_name", "channel", "ch"))
    out["channel_name"] = frame[channel_col].astype(str).to_numpy() if channel_col is not None else "ALL"
    out["channel_name"] = out["channel_name"].str.upper().str.replace(r"CHANNEL\s*", "CH", regex=True).str.replace(r"^CH0*", "CH", regex=True)
    out.loc[~out["channel_name"].isin(CHANNELS), "channel_name"] = "ALL"
    status_col = _find(frame.columns, ("status", "state"))
    out["status"] = frame[status_col].astype(str).str.strip().str.lower().to_numpy() if status_col is not None else "closed"
    out["status"] = out["status"].replace({"": "closed", "nan": "closed", "none": "closed"})
    split_col = _find(frame.columns, ("split", "partition", "set"))
    out["split"] = frame[split_col].astype(str).str.lower().to_numpy() if split_col is not None else "test"
    out.loc[~out["split"].isin(("train", "val", "test")), "split"] = "test"
    explicit = out.loc[out["channel_name"].isin(CHANNELS)].copy()
    explicit_before_qc = int(len(explicit))
    # Explicit channel annotations for a QC-invalid stream must not survive
    # merely because the source table named that channel.  They are retained
    # in the alignment audit as dropped channel rows.
    valid_lookup = {channel: bool(valid[index]) for index, channel in enumerate(CHANNELS)}
    if not explicit.empty:
        explicit = explicit.loc[
            explicit["channel_name"].map(valid_lookup).fillna(False).astype(bool)
        ].copy()
    all_rows = out.loc[out["channel_name"].eq("ALL")].drop(columns="channel_name")
    dropped_invalid_channel = explicit_before_qc - int(len(explicit))
    dropped_invalid_channel += int(len(all_rows)) * int(sum(not ok for ok in valid_lookup.values()))
    expanded = []
    for i, channel in enumerate(CHANNELS):
        if bool(valid[i]) and not all_rows.empty:
            copy = all_rows.copy()
            copy["channel_name"] = channel
            expanded.append(copy)
    out = pd.concat([explicit, *expanded], ignore_index=True) if expanded else explicit
    before_time_filter = int(len(out))
    out = out.loc[out["start_time_ms"].notna() & out["end_time_ms"].notna() & (out["end_time_ms"] > out["start_time_ms"])].copy()
    invalid_time = before_time_filter - int(len(out))
    out["animal_id"], out["session_id"] = animal, session
    # Both channel views of the same clinical interval keep one event id.  The
    # channel column remains part of channel-level joins, while shared event
    # identity prevents CH7 and CH8 from being assigned to different splits.
    out["event_id"] = [f"{session}:event_{src}" for src in out["source_event_id"]]
    out["channels"] = out["channel_name"]
    result = out.loc[:, columns]
    audit = {
        "source_files": source_records,
        "rows_read": int(rows_read),
        "rows_selected": int(rows_selected),
        "rows_invalid_time": int(invalid_time),
        "rows_dropped_invalid_channel": int(dropped_invalid_channel),
        "channel_rows": int(len(result)),
        "valid_channels": [channel for channel, ok in valid_lookup.items() if ok],
    }
    return (result, audit) if return_audit else result


def build_protocol(
    source: str | Path,
    output_dir: str | Path,
    *,
    labels: str | Path | None = None,
    animal_id: str | None = None,
    session_id: str | None = None,
    sampling_rate_hz: float | None = None,
    target_rate_hz: float = 250.0,
    window_seconds: float = 4.0,
    stride_seconds: float = 1.0,
    labels_time_base: str = "absolute_ms",
    ch7_column: str | None = None,
    ch8_column: str | None = None,
    filter_config: EEGFilterConfig | None = None,
    strict_label_mapping: bool = True,
) -> Path:
    """Create the minimal protocol expected by the audited inference code."""
    if min(target_rate_hz, window_seconds, stride_seconds) <= 0:
        raise ValueError("target rate, window and stride must be positive")
    files = discover_files(source)
    label_sources = discover_label_files(labels) if labels is not None else []
    # It is convenient for small studies to place raw exports and label tables
    # in one folder. ``discover_files`` intentionally accepts CSV/Parquet for
    # raw table inputs, so remove the explicitly supplied label tables before
    # deriving recording identities (otherwise a label table itself could be
    # mistaken for an EEG recording).
    if labels is not None:
        label_paths = {item.resolve() for item in label_sources}
        files = [item for item in files if item.resolve() not in label_paths]
    if not files:
        raise FileNotFoundError("输入目录中没有可识别的 EEG 文件（标签表已排除）")
    # Resolve all raw identities before reading labels.  This lets us reject
    # an identity-free multi-recording label directory before any filtered
    # cache is produced, rather than silently applying the same table to every
    # mouse/recording.
    recording_specs = [
        (_animal(path, animal_id), str(session_id or path.stem), path)
        for path in files
    ]
    if len({(animal, session) for animal, session, _ in recording_specs}) != len(recording_specs):
        raise ValueError(
            "多个原始文件解析为同一 animal/session；请通过 --animal-id/--session-id "
            "或规范化文件名提供唯一身份"
        )
    label_preflight: list[dict[str, Any]] = []
    if labels is not None and strict_label_mapping:
        label_preflight = _label_source_preflight(label_sources, recording_specs)
    output = Path(output_dir).resolve()
    cache_root = output / "cache" / "shards"
    cache_root.mkdir(parents=True, exist_ok=True)
    cfg = filter_config or EEGFilterConfig()
    window_samples = int(round(target_rate_hz * window_seconds))
    stride_samples = int(round(target_rate_hz * stride_seconds))
    index_parts: list[pd.DataFrame] = []
    event_parts: list[pd.DataFrame] = []
    baseline_rows: dict[tuple[str, str], tuple[float, float]] = {}
    source_records: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    label_alignment_rows: list[dict[str, Any]] = []
    for file_index, path in enumerate(files):
        animal_name = _animal(path, animal_id)
        session_name = str(session_id or path.stem)
        values, timestamps, native = _read(path, sampling_rate_hz, ch7_column, ch8_column)
        values, timestamps = _resample(values, timestamps, native, target_rate_hz)
        if values.shape[-1] < max(window_samples, 64):
            raise ValueError(f"{path} 太短，无法形成 {window_seconds:g}s 窗口")
        valid = _valid_channels(values)
        filtered = filter_continuous_eeg(values, target_rate_hz, cfg)
        filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
        filtered[~valid, :] = 0.0
        count = (filtered.shape[-1] - window_samples) // stride_samples + 1
        starts_idx = np.arange(count, dtype=np.int64) * stride_samples
        starts = timestamps[starts_idx]
        shard = cache_root / f"{_safe(animal_name)}_{_safe(session_name)}_shard{file_index:04d}"
        shard.mkdir(parents=True, exist_ok=True)
        windows = np.stack([filtered[:, i : i + window_samples] for i in starts_idx], axis=0).astype(np.float32)
        np.save(shard / "signals.npy", windows)
        np.save(shard / "start_time_ms.npy", starts.astype(float))
        np.save(shard / "channel_valid.npy", np.broadcast_to(valid, (count, 2)).copy())
        metadata = {"source": str(path), "animal_id": animal_name, "session_id": session_name, "sampling_rate_hz": target_rate_hz, "window_seconds": window_seconds, "stride_seconds": stride_seconds, "n_windows": int(count), "channel_valid": {CHANNELS[i]: bool(valid[i]) for i in range(2)}, "filter": cfg.to_dict(), "amplitude_unit": "a.u."}
        (shard / "meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        rows = []
        for channel_index, channel in enumerate(CHANNELS):
            if not valid[channel_index]:
                continue
            center, scale = _baseline(filtered[channel_index])
            baseline_rows[(animal_name, channel)] = (center, scale)
            for local, start in enumerate(starts):
                rows.append({"sample_id": f"{animal_name}:{session_name}:{channel}:{float(start):.3f}", "animal_id": animal_name, "session_id": session_name, "shard": file_index, "channel_name": channel, "channel_index": channel_index, "start_time_ms": float(start), "end_time_ms": float(start + 1000 * window_seconds), "center_time_ms": float(start + 500 * window_seconds), "label": -1, "channel_event_overlap_ms": 0.0, "channel_event_ids": "", "region_id": f"{animal_name}:{session_name}:shard{file_index:04d}", "region_kind": "unlabelled_recording", "split": "test", "continuity_id": f"{animal_name}:{session_name}:{channel}:shard{file_index:04d}", "continuity_position": local, "sampling_rate_hz": target_rate_hz, "window_seconds": window_seconds, "stride_seconds": stride_seconds, "filtered_cache_dir": str(shard), "filtered_local_index": local, "channel_valid": True, "full_split_context": True, "formal_metric_eligible": True, "supervised_center_eligible": False})
        index_parts.append(pd.DataFrame(rows))
        event_frame, label_audit = _read_labels(
            labels,
            timestamps=timestamps,
            animal=animal_name,
            session=session_name,
            valid=valid,
            time_base=labels_time_base,
            return_audit=True,
        )
        event_parts.append(event_frame)
        label_alignment_rows.append(
            {
                "animal_id": animal_name,
                "session_id": session_name,
                "source_file": str(path.resolve()),
                "raw_label_files": int(len(label_sources)),
                "rows_read": int(label_audit.get("rows_read", 0)),
                "rows_selected": int(label_audit.get("rows_selected", 0)),
                "rows_invalid_time": int(label_audit.get("rows_invalid_time", 0)),
                "rows_dropped_invalid_channel": int(label_audit.get("rows_dropped_invalid_channel", 0)),
                "channel_event_rows": int(label_audit.get("channel_rows", 0)),
                "valid_channels": ";".join(label_audit.get("valid_channels", [])),
            }
        )
        manifest_rows.append({"animal_id": animal_name, "session_id": session_name, "shard": int(file_index), "filtered_cache_dir": str(shard), "sampling_rate_hz": float(target_rate_hz), "window_seconds": float(window_seconds), "stride_seconds": float(stride_seconds), "n_windows": int(count), "valid_ch7_windows": int(valid[0] * count), "valid_ch8_windows": int(valid[1] * count)})
        source_records.append({"path": str(path), "animal_id": animal_name, "session_id": session_name, "native_rate_hz": native, "target_rate_hz": target_rate_hz, "n_samples": int(values.shape[-1]), "n_windows": int(count), "channel_valid": valid.tolist()})
    protocol = output / "protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    index = pd.concat(index_parts, ignore_index=True) if index_parts else pd.DataFrame()
    events = pd.concat(event_parts, ignore_index=True) if event_parts else pd.DataFrame(columns=["animal_id", "session_id", "channel_name", "event_id", "start_time_ms", "end_time_ms", "status", "split"])
    index.to_parquet(protocol / "full_index.parquet", index=False)
    events.to_parquet(protocol / "channel_events.parquet", index=False)
    if events.empty:
        event_components = pd.DataFrame(columns=["animal_id", "session_id", "event_id", "start_time_ms", "end_time_ms", "channels", "status"])
    else:
        event_components = (
            events.groupby(["animal_id", "session_id", "event_id", "start_time_ms", "end_time_ms", "status"], as_index=False)
            .agg(channels=("channel_name", lambda values: ",".join(sorted(set(map(str, values))))))
        )
    event_components.to_parquet(protocol / "event_components.parquet", index=False)
    pd.DataFrame(manifest_rows).to_csv(output / "complete_filtered_manifest.csv", index=False, encoding="utf-8-sig")
    regions = index.groupby(["animal_id", "session_id", "shard"], as_index=False).agg(start_time_ms=("start_time_ms", "min"), end_time_ms=("end_time_ms", "max"), split=("split", "first"))
    regions["region_id"] = regions.apply(lambda r: f"{r.animal_id}:{r.session_id}:shard{int(r.shard):04d}", axis=1)
    regions["region_kind"] = "unlabelled_recording"
    regions.to_parquet(protocol / "split_regions.parquet", index=False)
    baseline = {"schema_version": 1, "method": "per_channel_median_mad_from_input_recording", "channels": [{"animal_id": a, "channel_name": c, "center": center, "scale": scale} for (a, c), (center, scale) in sorted(baseline_rows.items())]}
    (protocol / "training_baseline.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {"schema_version": 1, "protocol": "raw_adapter_offline_inference", "source_files": source_records, "n_windows": int(len(index)), "n_valid_channel_streams": int(index[["animal_id", "session_id", "channel_name", "continuity_id"]].drop_duplicates().shape[0]) if not index.empty else 0, "filter": cfg.to_dict(), "target_rate_hz": target_rate_hz, "window_seconds": window_seconds, "stride_seconds": stride_seconds, "labels_supplied": labels is not None, "amplitude_unit": "a.u.", "note": "Offline/replay adapter; not a clinical real-time acquisition interface."}
    alignment_frame = pd.DataFrame(label_alignment_rows)
    alignment_frame.to_csv(protocol / "label_alignment.csv", index=False, encoding="utf-8-sig")
    alignment_audit = {
        "schema_version": 1,
        "strict_label_mapping": bool(strict_label_mapping),
        "source_files": label_preflight,
        "recordings": label_alignment_rows,
        "rows_read_total": int(alignment_frame["rows_read"].sum()) if not alignment_frame.empty else 0,
        "rows_selected_total": int(alignment_frame["rows_selected"].sum()) if not alignment_frame.empty else 0,
        "rows_invalid_time_total": int(alignment_frame["rows_invalid_time"].sum()) if not alignment_frame.empty else 0,
        "rows_dropped_invalid_channel_total": int(alignment_frame["rows_dropped_invalid_channel"].sum()) if not alignment_frame.empty else 0,
        "channel_event_rows_total": int(alignment_frame["channel_event_rows"].sum()) if not alignment_frame.empty else 0,
        "note": "Rows dropped by identity routing, invalid time bounds, or invalid channels are explicit in this audit; no missing labels are imputed as non-seizure.",
    }
    (protocol / "label_alignment.json").write_text(json.dumps(alignment_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    audit.update(
        {
            "label_alignment": "label_alignment.json",
            "label_alignment_csv": "label_alignment.csv",
            "label_rows_read": alignment_audit["rows_read_total"],
            "label_rows_selected": alignment_audit["rows_selected_total"],
            "label_channel_event_rows": alignment_audit["channel_event_rows_total"],
            "label_rows_invalid_time": alignment_audit["rows_invalid_time_total"],
            "label_rows_dropped_invalid_channel": alignment_audit["rows_dropped_invalid_channel_total"],
            "strict_label_mapping": bool(strict_label_mapping),
        }
    )
    (protocol / "protocol_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (protocol / "config.json").write_text(json.dumps({"target_rate_hz": target_rate_hz, "window_seconds": window_seconds, "stride_seconds": stride_seconds, "filter": cfg.to_dict()}, ensure_ascii=False, indent=2), encoding="utf-8")
    return protocol
