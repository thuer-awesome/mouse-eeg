from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from mouse_sleep.epilepsy.replication.aitrics_exact import (
    AITRICSRawCNN2DLSTMPaperExact,
)
from mouse_sleep.epilepsy.replication.eegnet_exact import EEGNet82
from mouse_sleep.epilepsy.replication.eegwavenet import EEGWaveNetPaperExact


class EpilepsyEEGNet(nn.Module):
    """Small EEGNet-style baseline for two-channel rodent EEG."""
    def __init__(self, channels: int = 2, classes: int = 2, dropout: float = 0.25) -> None:
        super().__init__()
        self.temporal = nn.Sequential(nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False), nn.BatchNorm2d(16))
        self.depthwise = nn.Sequential(nn.Conv2d(16, 32, (channels, 1), groups=16, bias=False), nn.BatchNorm2d(32), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.separable = nn.Sequential(nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32, bias=False), nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        self.head = nn.LazyLinear(classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3: raise ValueError("Expected [batch, channels, samples]")
        z = self.separable(self.depthwise(self.temporal(x.unsqueeze(1))))
        return self.head(z.flatten(1))


class EpilepsyTCN(nn.Module):
    def __init__(self, channels: int = 2, classes: int = 2, hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        layers = [nn.Conv1d(channels, hidden, 7, padding=3), nn.GELU()]
        for dilation in (1, 2, 4, 8):
            layers.extend([nn.Conv1d(hidden, hidden, 5, padding=2 * dilation, dilation=dilation), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers); self.head = nn.Linear(hidden, classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x).mean(-1))


class AITRICSConv2DLSTM(nn.Module):
    """Corrected local reproduction of AITRICS CNN2D-LSTM/raw.

    The downloaded official file contains debugging ``print`` and ``exit(1)``
    statements and pools away the LSTM sequence. This implementation keeps the
    published convolutional family and preserves a temporal token sequence.
    """
    def __init__(self, channels: int = 2, classes: int = 2, hidden: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        def block(inp: int, out: int, kernel: int, stride: int = 1) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(inp, out, (1, kernel), stride=(1, stride), padding=(0, kernel // 2), bias=False), nn.BatchNorm2d(out), nn.ReLU(), nn.Dropout(dropout))
        self.features = nn.Sequential(block(1, 64, 51, 4), nn.MaxPool2d((1, 4)), block(64, 128, 21, 2), block(128, 256, 9, 2))
        self.lstm = nn.LSTM(256 * channels, hidden, num_layers=1, batch_first=True, bidirectional=False)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, classes))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x.unsqueeze(1))
        b, f, c, t = z.shape
        z = z.permute(0, 3, 1, 2).reshape(b, t, f * c)
        z, _ = self.lstm(z)
        return self.head(z[:, -1])


class TimeFrequencyConformer(nn.Module):
    """Lightweight dual-branch time/frequency model for the adapted dataset."""
    def __init__(self, channels: int = 2, classes: int = 2, dim: int = 96, heads: int = 4, dropout: float = 0.15, sample_rate: float = 250.0) -> None:
        super().__init__(); self.sample_rate = sample_rate
        self.time = nn.Sequential(nn.Conv1d(channels, dim // 2, 9, padding=4), nn.GELU(), nn.Conv1d(dim // 2, dim, 31, padding=15), nn.BatchNorm1d(dim), nn.GELU())
        self.freq = nn.Sequential(nn.Conv2d(channels, dim, (1, 9), padding=(0, 4)), nn.BatchNorm2d(dim), nn.GELU())
        self.register_buffer("stft_window", torch.hann_window(128), persistent=False)
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 3, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.context = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, classes))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.time(x).transpose(1, 2)
        spec = torch.stft(x.reshape(-1, x.shape[-1]), n_fft=128, hop_length=32, window=self.stft_window, return_complex=True).abs().reshape(x.shape[0], x.shape[1], 65, -1)
        f = self.freq(spec).mean(2).transpose(1, 2)
        q, _ = self.cross(t[:, :: max(1, t.shape[1] // 64)], f, f, need_weights=False)
        return self.head(self.context(q).mean(1))


def _resize_samples(x: torch.Tensor, samples: int) -> torch.Tensor:
    """Explicitly adapt the 250 Hz rodent cache to a paper input length."""

    if x.shape[-1] == samples:
        return x
    return F.interpolate(x, size=samples, mode="linear", align_corners=False)


class RodentEEGWaveNetPaperAdapter(nn.Module):
    """EEGWaveNet paper topology with an explicit 2-channel rodent adapter."""

    paper_input_samples = 1024
    input_layout = "[batch, channels, samples]"

    def __init__(self, channels: int = 2, classes: int = 2) -> None:
        super().__init__()
        self.core = EEGWaveNetPaperExact(channels, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.core.n_chans:
            raise ValueError(
                f"expected [batch, {self.core.n_chans}, samples], received {tuple(x.shape)}"
            )
        return self.core(_resize_samples(x, self.paper_input_samples))


class RodentEEGNet82PaperAdapter(nn.Module):
    """EEGNet-8,2 with only channel/sample-count adaptation for rodents."""

    paper_input_samples = 1024
    input_layout = "[batch, channels, samples]"

    def __init__(self, channels: int = 2, classes: int = 2) -> None:
        super().__init__()
        self.core = EEGNet82(channels, self.paper_input_samples, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.core.channels:
            raise ValueError(
                f"expected [batch, {self.core.channels}, samples], received {tuple(x.shape)}"
            )
        return self.core(_resize_samples(x, self.paper_input_samples))

    @torch.no_grad()
    def apply_max_norm_constraints(self) -> None:
        self.core.apply_max_norm_constraints()


class RodentAITRICSRawPaperAdapter(nn.Module):
    """AITRICS raw CNN2D-LSTM adapter for CH7/CH8 windows.

    The source model consumes ``[B, 800, C]``.  The generic rodent cache emits
    ``[B, C, 1000]`` at 250 Hz, so this wrapper resamples to 800 points and
    transposes explicitly.  The ordinary :meth:`forward` path resets state for
    every independent window.  The :meth:`forward_stateful` path is used by the
    dedicated 30-second trainer: callers reset once at a sequence boundary and
    then retain state through the remaining 25 windows.
    """

    paper_input_samples = 800
    input_layout = "[batch, channels, samples] -> [batch, 800, channels]"

    def __init__(self, channels: int = 2, classes: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        del channels  # The source raw model's convolution has height 1.
        self.core = AITRICSRawCNN2DLSTMPaperExact(classes=classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_stateful(x, reset_state=True)
        return logits

    def forward_stateful(
        self, x: torch.Tensor, *, reset_state: bool = False
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run one window while preserving the paper model's recurrent state.

        ``reset_state=True`` must be used at the beginning of every logical
        30-second recording segment.  The core detaches its hidden tensors on
        subsequent calls, matching the released trainer's truncated BPTT
        behavior.  Sequence batching is implemented by the stateful trainer so
        every batch slot represents one independent recording segment.
        """

        if x.ndim != 3 or x.shape[1] < 1:
            raise ValueError("expected [batch, channels, samples]")
        adapted = _resize_samples(x, self.paper_input_samples).transpose(1, 2)
        return self.core(adapted, reset_state=reset_state)

    def reset_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the paper core's explicit sequence-boundary reset."""

        return self.core.reset_state(batch_size, device=device, dtype=dtype)


def build_epilepsy_model(name: str, channels: int = 2, classes: int = 2, **config: Any) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key in {"eegwavenet", "eegwavenet_paper_exact", "rodent_eegwavenet"}:
        return RodentEEGWaveNetPaperAdapter(channels, classes)
    if key in {
        "eegnet_8_2_paper_exact",
        "eegnet82_paper_exact",
        "rodent_eegnet82",
    }:
        return RodentEEGNet82PaperAdapter(channels, classes)
    if key in {
        "aitrics_raw_cnn2d_lstm_paper_exact",
        "aitrics_raw",
        "rodent_aitrics_stateful",
    }:
        return RodentAITRICSRawPaperAdapter(
            channels, classes, float(config.get("dropout", 0.1))
        )
    if key in {"eegnet", "epilepsy_eegnet"}: return EpilepsyEEGNet(channels, classes, float(config.get("dropout", .25)))
    if key in {"tcn", "epilepsy_tcn"}: return EpilepsyTCN(channels, classes, int(config.get("hidden", 64)), float(config.get("dropout", .2)))
    if key in {"aitrics", "cnn2d_lstm", "aitrics_cnn2d_lstm"}: return AITRICSConv2DLSTM(channels, classes, int(config.get("hidden", 128)), float(config.get("dropout", .2)))
    if key in {"time_frequency", "tf_conformer", "conformer"}: return TimeFrequencyConformer(channels, classes, int(config.get("dim", 96)), int(config.get("heads", 4)), float(config.get("dropout", .15)), float(config.get("sample_rate", 250.0)))
    raise ValueError(f"Unknown epilepsy model {name!r}; use eegnet, tcn, aitrics_cnn2d_lstm, or time_frequency")
