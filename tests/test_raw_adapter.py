from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from epilepsy_eeg.raw_adapter import _animal, _read_labels, _valid_channels, build_protocol, discover_label_files


def test_channel_qc_rejects_sentinel_channel() -> None:
    values = np.vstack([np.sin(np.linspace(0, 30, 1000)), np.full(1000, -400000.0)])
    result = _valid_channels(values)
    assert result.tolist() == [True, False]


def test_animal_id_is_detected_after_windows_path_separator() -> None:
    from pathlib import Path

    assert _animal(Path(r"C:\\study\\E01_recording.npz"), None) == "E01"


def test_labels_without_channel_are_expanded() -> None:
    timestamps = np.arange(10000, dtype=float)
    labels = pd.DataFrame({"start_time_ms": [100.0], "end_time_ms": [500.0]})
    path = "test_labels_tmp.csv"
    labels.to_csv(path, index=False)
    try:
        out = _read_labels(path, timestamps=timestamps, animal="E01", session="r0", valid=np.array([True, True]), time_base="absolute_ms")
    finally:
        import pathlib
        pathlib.Path(path).unlink(missing_ok=True)
    assert set(out["channel_name"]) == {"CH7", "CH8"}
    assert len(out) == 2
    assert out["event_id"].nunique() == 1
    assert set(out["channels"]) == {"CH7", "CH8"}


def test_label_metadata_is_aligned_after_channel_expansion(tmp_path) -> None:
    labels = pd.DataFrame(
        {
            "animal_id": ["E02", "E01"],
            "session_id": ["other", "r1"],
            "event_id": ["wrong", "clinical-7"],
            "start_time_ms": [10.0, 100.0],
            "end_time_ms": [20.0, 500.0],
            "status": ["open", "reviewed"],
            "split": ["train", "val"],
        }
    )
    path = tmp_path / "labels.csv"
    labels.to_csv(path, index=False)
    out = _read_labels(path, timestamps=np.arange(10000, dtype=float), animal="E01", session="r1", valid=np.array([True, True]), time_base="absolute_ms")
    assert len(out) == 2
    assert set(out["status"]) == {"reviewed"}
    assert set(out["split"]) == {"val"}
    assert out["event_id"].nunique() == 1


def test_label_directory_is_discovered_in_stable_order(tmp_path) -> None:
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    pd.DataFrame({"start_time_ms": [300.0], "end_time_ms": [400.0], "event_id": ["b"]}).to_csv(label_dir / "02_second.csv", index=False)
    pd.DataFrame({"start_time_ms": [100.0], "end_time_ms": [200.0], "event_id": ["a"]}).to_csv(label_dir / "01_first.csv", index=False)
    assert [path.name for path in discover_label_files(label_dir)] == ["01_first.csv", "02_second.csv"]
    out = _read_labels(label_dir, timestamps=np.arange(1000, dtype=float), animal="E01", session="r1", valid=np.array([True, False]), time_base="absolute_ms")
    assert out["source_event_id"].tolist() == ["a", "b"]
    assert out["channel_name"].tolist() == ["CH7", "CH7"]


def test_identity_free_label_files_are_routed_by_mouse_token(tmp_path) -> None:
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    pd.DataFrame(
        {"start_time_ms": [100.0], "end_time_ms": [200.0], "event_id": ["e50"]}
    ).to_csv(label_dir / "E50_events.csv", index=False)
    pd.DataFrame(
        {"start_time_ms": [300.0], "end_time_ms": [400.0], "event_id": ["e76"]}
    ).to_csv(label_dir / "E76_events.csv", index=False)
    out = _read_labels(
        label_dir,
        timestamps=np.arange(1000, dtype=float),
        animal="E50",
        session="r1",
        valid=np.array([True, False]),
        time_base="absolute_ms",
    )
    assert out["source_event_id"].tolist() == ["e50"]


def test_build_protocol_writes_channel_and_event_component_tables(tmp_path) -> None:
    n = 6000
    recording = tmp_path / "E03_session.csv"
    labels = tmp_path / "labels.csv"
    t = np.arange(n, dtype=float) * 1.0
    pd.DataFrame({"timestamp_ms": t, "CH7": np.sin(t / 37.0), "CH8": np.cos(t / 41.0)}).to_csv(recording, index=False)
    pd.DataFrame({"event_id": ["evt-1"], "start_time_ms": [1000.0], "end_time_ms": [2500.0]}).to_csv(labels, index=False)
    protocol = build_protocol(recording, tmp_path / "prepared", labels=labels, sampling_rate_hz=1000.0)
    assert (protocol / "full_index.parquet").is_file()
    assert (protocol / "event_components.parquet").is_file()
    components = pd.read_parquet(protocol / "event_components.parquet")
    assert len(components) == 1
    assert components.iloc[0]["channels"] == "CH7,CH8"
    assert components.iloc[0]["event_id"].endswith("evt-1")
    assert (protocol.parent / "complete_filtered_manifest.csv").is_file()


def test_build_protocol_rejects_ambiguous_identity_free_label_directory(tmp_path) -> None:
    """A generic label directory must not be silently reused for two records."""

    n = 20000
    t = np.arange(n, dtype=float)
    for animal in ("E10", "E11"):
        pd.DataFrame(
            {"timestamp_ms": t, "CH7": np.sin(t / 17.0), "CH8": np.cos(t / 19.0)}
        ).to_csv(tmp_path / f"{animal}_session.csv", index=False)
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    pd.DataFrame({"event_id": ["x"], "start_time_ms": [100.0], "end_time_ms": [300.0]}).to_csv(label_dir / "events.csv", index=False)
    with pytest.raises(ValueError, match="标签映射预检失败"):
        build_protocol(tmp_path, tmp_path / "prepared", labels=label_dir, sampling_rate_hz=1000.0)


def test_build_protocol_writes_label_alignment_audit_and_filters_invalid_channel(tmp_path) -> None:
    n = 20000
    t = np.arange(n, dtype=float)
    recording = tmp_path / "E77_session.csv"
    # CH7 is a rail/sentinel stream and must be excluded from explicit labels.
    pd.DataFrame({"timestamp_ms": t, "CH7": np.full(n, 400000.0), "CH8": np.sin(t / 19.0)}).to_csv(recording, index=False)
    labels = tmp_path / "events.csv"
    pd.DataFrame({"event_id": ["bad-channel", "good"], "channel_name": ["CH7", "CH8"], "start_time_ms": [100.0, 200.0], "end_time_ms": [300.0, 400.0]}).to_csv(labels, index=False)
    protocol = build_protocol(recording, tmp_path / "prepared", labels=labels, sampling_rate_hz=1000.0)
    events = pd.read_parquet(protocol / "channel_events.parquet")
    assert set(events["channel_name"]) == {"CH8"}
    assert (protocol / "label_alignment.json").is_file()
    audit = json.loads((protocol / "label_alignment.json").read_text(encoding="utf-8"))
    assert audit["channel_event_rows_total"] == 1


def test_build_protocol_excludes_label_csv_when_raw_and_labels_share_directory(tmp_path) -> None:
    n = 6000
    recording = tmp_path / "E04_session.csv"
    labels = tmp_path / "labels.csv"
    t = np.arange(n, dtype=float)
    pd.DataFrame(
        {"timestamp_ms": t, "CH7": np.sin(t / 37.0), "CH8": np.cos(t / 41.0)}
    ).to_csv(recording, index=False)
    pd.DataFrame(
        {"event_id": ["evt-1"], "start_time_ms": [1000.0], "end_time_ms": [2500.0]}
    ).to_csv(labels, index=False)
    protocol = build_protocol(
        tmp_path,
        tmp_path / "prepared",
        labels=labels,
        sampling_rate_hz=1000.0,
    )
    assert len(pd.read_csv(protocol.parent / "complete_filtered_manifest.csv")) == 1


def test_label_preflight_rejects_animal_only_rows_for_multiple_sessions(tmp_path) -> None:
    t = np.arange(6000, dtype=float)
    for session in ("r1", "r2"):
        pd.DataFrame(
            {"timestamp_ms": t, "CH7": np.sin(t / 37.0), "CH8": np.cos(t / 41.0)}
        ).to_csv(tmp_path / f"E05_{session}.csv", index=False)
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "animal_id": ["E05"],
            "start_time_ms": [1000.0],
            "end_time_ms": [2500.0],
        }
    ).to_csv(labels, index=False)
    import pytest

    with pytest.raises(ValueError, match="多个输入记录"):
        build_protocol(
            tmp_path,
            tmp_path / "prepared",
            labels=labels,
            sampling_rate_hz=1000.0,
        )
