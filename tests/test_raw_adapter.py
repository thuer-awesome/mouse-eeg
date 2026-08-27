from __future__ import annotations

import numpy as np
import pandas as pd

from epilepsy_eeg.raw_adapter import _read_labels, _valid_channels, build_protocol


def test_channel_qc_rejects_sentinel_channel() -> None:
    values = np.vstack([np.sin(np.linspace(0, 30, 1000)), np.full(1000, -400000.0)])
    result = _valid_channels(values)
    assert result.tolist() == [True, False]


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
