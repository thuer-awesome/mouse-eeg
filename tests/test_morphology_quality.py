from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import epilepsy_eeg.morphology_adapter as adapter


def _fake_features(values: np.ndarray, rate: float, filtered_signals=None) -> dict[str, np.ndarray]:
    del rate, filtered_signals
    n = int(values.shape[0])
    # Encode the cache row in every feature so baseline calculations are easy
    # to check without relying on signal-processing tolerances.
    encoded = values[:, 0, 0].astype(float)
    output: dict[str, np.ndarray] = {}
    for channel in ("ch7", "ch8"):
        output[f"{channel}_valid"] = np.ones(n, dtype=bool)
        for suffix in adapter.FEATURE_SUFFIXES:
            output[f"{channel}_{suffix}"] = encoded.copy()
    output["has_nan_or_inf"] = np.zeros(n, dtype=bool)
    return output


def _rows(cache: Path, indices: list[int], split: str, *, label: int = 1) -> pd.DataFrame:
    rows = []
    for index in indices:
        for channel in ("CH7", "CH8"):
            rows.append(
                {
                    "animal_id": "E01",
                    "session_id": "S01",
                    "event_id": f"event_{index}",
                    "channel_name": channel,
                    "start_time_ms": float(index * 1000),
                    "end_time_ms": float(index * 1000 + 4000),
                    "label": label,
                    "channel_valid": True,
                    "filtered_cache_dir": str(cache),
                    "filtered_local_index": index,
                    "shard": 0,
                    "sampling_rate_hz": 250.0,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def test_baseline_uses_train_rows_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(adapter, "_feature_batch", _fake_features)
    signals = np.zeros((5, 2, 1000), dtype=np.float32)
    signals[:, :, 0] = np.arange(5, dtype=np.float32)[:, None]
    np.save(tmp_path / "signals.npy", signals)

    event_rows = pd.concat(
        [_rows(tmp_path, [0], "train"), _rows(tmp_path, [1], "val")], ignore_index=True
    )
    # Two train backgrounds define the reference (indices 2, 3).  The large
    # validation value (index 4) must not move the center/scale.
    background = pd.concat(
        [_rows(tmp_path, [2, 3], "train", label=0), _rows(tmp_path, [4], "val", label=0)],
        ignore_index=True,
    )
    result = adapter._feature_rows(
        event_rows,
        baseline_rows=background,
        baseline_samples_per_stream=256,
    )
    ch7 = result.loc[result["channel_name"].eq("CH7")].sort_values("filtered_local_index")
    assert result.attrs["baseline_reference_scope"] == "train_background_only"
    assert result.attrs["baseline_fallback_groups"] == 0
    # q10/q90 of [2, 3] are [2.1, 2.9], so row 0 has z=(0-2.5)/0.8.
    assert float(ch7.iloc[0]["ch7_baseline_z__log_rms"]) == pytest.approx(-3.125)
    # A validation background at 4 would produce a materially different z.
    assert float(ch7.iloc[1]["ch7_baseline_z__log_rms"]) == pytest.approx(-1.875)


def test_invalid_channel_windows_are_excluded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_with_bad_channel(values: np.ndarray, rate: float, filtered_signals=None):
        output = _fake_features(values, rate, filtered_signals)
        # The second physical window is invalid only for CH7; CH8 remains a
        # valid independent observation.
        output["ch7_valid"][np.isclose(values[:, 0, 0], 1.0)] = False
        return output

    monkeypatch.setattr(adapter, "_feature_batch", fake_with_bad_channel)
    signals = np.random.default_rng(1).normal(size=(2, 2, 1000)).astype(np.float32)
    signals[:, :, 0] = np.arange(2, dtype=np.float32)[:, None]
    np.save(tmp_path / "signals.npy", signals)
    event_rows = _rows(tmp_path, [0, 1], "train")
    result = adapter._feature_rows(event_rows, baseline_rows=event_rows.assign(label=0), baseline_samples_per_stream=256)
    assert result.attrs["invalid_event_windows"] == 1
    assert result["feature_valid"].all()
    assert not ((result["channel_name"] == "CH7") & (result["filtered_local_index"] == 1)).any()
    assert ((result["channel_name"] == "CH8") & (result["filtered_local_index"] == 1)).any()


def test_pooled_clustering_refuses_mouse_without_train_events(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "fit_pooled_mouse_morphology.py"
    spec = importlib.util.spec_from_file_location("fit_pooled_mouse_morphology", script)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    table = tmp_path / "01_tables"
    table.mkdir()
    row = {
        "animal_id": "E01",
        "session_id": "S01",
        "event_id": "event_1",
        "channel_name": "CH7",
        "split": "val",
        **{feature: 0.0 for feature in module.FEATURES},
    }
    pd.DataFrame([row]).to_parquet(table / "channel_event_features.parquet", index=False)
    with pytest.raises(ValueError, match="no_train_events"):
        module.main(["--root", str(tmp_path)])
    exclusions = pd.read_csv(tmp_path / "03_cross_mouse" / "pooled_mouse_exclusions.csv")
    assert exclusions.to_dict("records") == [
        {"animal_id": "E01", "n_events": 1, "n_train_events": 0, "reason": "no_train_events"}
    ]
