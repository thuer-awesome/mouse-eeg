from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos


@dataclass(frozen=True)
class EEGFilterConfig:
    highpass_hz: float = 0.5
    lowpass_hz: float = 70.0
    notch_hz: float = 50.0
    notch_quality_factor: float = 30.0
    bandpass_order: int = 4

    def validated(self, sampling_rate_hz: float) -> "EEGFilterConfig":
        nyquist = sampling_rate_hz / 2.0
        if not 0 < self.highpass_hz < self.lowpass_hz < nyquist:
            raise ValueError(
                "EEG filter requires 0 < highpass < lowpass < Nyquist; "
                f"received {self.highpass_hz}, {self.lowpass_hz}, Nyquist={nyquist}"
            )
        if not self.highpass_hz < self.notch_hz < self.lowpass_hz:
            raise ValueError("The notch frequency must lie inside the EEG passband")
        if self.notch_quality_factor <= 0 or self.bandpass_order < 1:
            raise ValueError("Filter quality factor and order must be positive")
        return self

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def filter_continuous_eeg(
    signal: np.ndarray,
    sampling_rate_hz: float,
    config: EEGFilterConfig | None = None,
) -> np.ndarray:
    """Zero-phase band-pass and notch filtering along the last axis."""

    config = (config or EEGFilterConfig()).validated(sampling_rate_hz)
    values = np.asarray(signal, dtype=np.float64)
    if values.shape[-1] < 64:
        raise ValueError("EEG filtering needs at least 64 samples")
    finite = np.isfinite(values)
    median = np.nanmedian(np.where(finite, values, np.nan), axis=-1, keepdims=True)
    median = np.nan_to_num(median, nan=0.0)
    centered = np.where(finite, values, median) - median
    bandpass = butter(
        config.bandpass_order,
        [config.highpass_hz, config.lowpass_hz],
        btype="bandpass",
        fs=sampling_rate_hz,
        output="sos",
    )
    filtered = sosfiltfilt(bandpass, centered, axis=-1)
    notch_b, notch_a = iirnotch(
        config.notch_hz,
        config.notch_quality_factor,
        fs=sampling_rate_hz,
    )
    filtered = sosfiltfilt(tf2sos(notch_b, notch_a), filtered, axis=-1)
    return np.asarray(filtered, dtype=np.float32)


def reconstruct_continuous_cache(
    windowed_signals: np.ndarray,
    stride_samples: int,
) -> np.ndarray:
    """Rebuild a continuous shard from deterministic overlapping cache windows."""

    signals = windowed_signals
    if signals.ndim != 3:
        raise ValueError(f"Expected [windows, channels, samples], received {signals.shape}")
    if not 0 < stride_samples <= signals.shape[-1]:
        raise ValueError("Cache stride must be within the window length")
    first = np.asarray(signals[0], dtype=np.float32)
    if len(signals) == 1:
        return first.copy()
    tails = np.asarray(signals[1:, :, -stride_samples:], dtype=np.float32)
    appended = tails.transpose(1, 0, 2).reshape(signals.shape[1], -1)
    return np.concatenate([first, appended], axis=1)


def extract_filtered_windows(
    continuous_signal: np.ndarray,
    local_indices: np.ndarray,
    stride_samples: int,
    window_samples: int,
) -> np.ndarray:
    result = np.empty(
        (len(local_indices), continuous_signal.shape[0], window_samples),
        dtype=np.float32,
    )
    for output_index, local_index in enumerate(np.asarray(local_indices, dtype=np.int64)):
        start = int(local_index) * stride_samples
        stop = start + window_samples
        if start < 0 or stop > continuous_signal.shape[-1]:
            raise IndexError(
                f"Cached window {local_index} maps outside reconstructed continuous signal"
            )
        result[output_index] = continuous_signal[:, start:stop]
    return result


__all__ = [
    "EEGFilterConfig",
    "extract_filtered_windows",
    "filter_continuous_eeg",
    "reconstruct_continuous_cache",
]
