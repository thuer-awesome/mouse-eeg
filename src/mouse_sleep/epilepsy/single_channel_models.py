"""Channel-blind, single-channel seizure models.

This module is intentionally separate from :mod:`seizure_physio`.  The legacy
model consumes two simultaneous channels, whereas the models here treat every
valid EEG channel as an independent observation and never receive animal or
channel identity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from mouse_sleep.epilepsy.replication.eegwavenet import EEGWaveNetPaperExact


SINGLE_CHANNEL_STATISTIC_NAMES = (
    "log1p_rms_scaled",
    "log1p_line_length_scaled",
    "log1p_hjorth_mobility_scaled",
    "log1p_hjorth_complexity_scaled",
    "relative_delta_power",
    "relative_theta_power",
    "relative_alpha_power",
    "relative_beta_power",
    "relative_gamma_power",
    "log_theta_delta_scaled",
    "spectral_entropy",
    "dominant_frequency_fraction",
    "spectral_edge_95_fraction",
)


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(channels, maximum), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _position_encoding(values: torch.Tensor) -> torch.Tensor:
    length, dimension = values.shape[1:]
    positions = torch.arange(
        length, device=values.device, dtype=values.dtype
    ).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, dimension, 2, device=values.device, dtype=values.dtype)
        * (-math.log(10000.0) / dimension)
    )
    encoding = torch.zeros(
        length, dimension, device=values.device, dtype=values.dtype
    )
    encoding[:, 0::2] = torch.sin(positions * divisor)
    odd_width = encoding[:, 1::2].shape[1]
    encoding[:, 1::2] = torch.cos(positions * divisor[:odd_width])
    return values + encoding.unsqueeze(0)


class EEGWaveNetSC(nn.Module):
    """Single-channel wrapper around the paper-faithful EEGWaveNet baseline.

    A direct call accepts ``[batch, 1, samples]``.  For use with a shared
    sequence loader, ``[batch, epochs, 1, samples]`` is also accepted and only
    the valid centre epoch is classified.  The returned ``logits`` are the
    class log-probabilities produced by the published architecture; they are
    valid inputs to both cross entropy and NLL loss.
    """

    input_layout = "[batch, 1, samples]"
    uses_sequence_context = False
    auxiliary_statistics = False
    output_is_log_probability = True

    def __init__(
        self,
        *,
        samples_per_window: int = 1000,
        classes: int = 2,
        sequence_length: int = 21,
    ) -> None:
        super().__init__()
        if samples_per_window <= 0:
            raise ValueError("samples_per_window must be positive")
        if classes < 2:
            raise ValueError("classes must be at least 2")
        if sequence_length <= 0 or sequence_length % 2 == 0:
            raise ValueError("sequence_length must be a positive odd integer")
        self.samples_per_window = int(samples_per_window)
        self.classes = int(classes)
        self.sequence_length = int(sequence_length)
        self.center_index = self.sequence_length // 2
        self.network = EEGWaveNetPaperExact(n_chans=1, n_classes=self.classes)

    def forward(
        self,
        signals: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if signals.ndim == 4:
            self._validate_sequence(signals, padding_mask)
            signals = signals[:, self.center_index]
        elif signals.ndim == 3:
            self._validate_windows(signals)
            if padding_mask is not None:
                raise ValueError("padding_mask is only valid for sequence input")
        else:
            raise ValueError(
                "expected signals [batch, 1, samples] or "
                "[batch, epochs, 1, samples]"
            )
        clean_signals = torch.nan_to_num(
            signals, nan=0.0, posinf=0.0, neginf=0.0
        )
        return {"logits": self.network(clean_signals)}

    def _validate_windows(self, signals: torch.Tensor) -> None:
        expected = (1, self.samples_per_window)
        if tuple(signals.shape[1:]) != expected:
            raise ValueError(
                f"expected signals [batch, 1, {self.samples_per_window}], "
                f"received {tuple(signals.shape)}"
            )
        if not signals.is_floating_point():
            raise TypeError("signals must be a floating-point tensor")

    def _validate_sequence(
        self, signals: torch.Tensor, padding_mask: torch.Tensor | None
    ) -> None:
        expected = (self.sequence_length, 1, self.samples_per_window)
        if tuple(signals.shape[1:]) != expected:
            raise ValueError(
                "expected signals "
                f"[batch, {self.sequence_length}, 1, {self.samples_per_window}], "
                f"received {tuple(signals.shape)}"
            )
        if not signals.is_floating_point():
            raise TypeError("signals must be a floating-point tensor")
        if padding_mask is None:
            return
        if padding_mask.dtype != torch.bool:
            raise TypeError("padding_mask must be a boolean tensor")
        if tuple(padding_mask.shape) != tuple(signals.shape[:2]):
            raise ValueError(
                f"expected padding_mask {tuple(signals.shape[:2])}, "
                f"received {tuple(padding_mask.shape)}"
            )
        if bool(padding_mask[:, self.center_index].any()):
            raise ValueError("the centre epoch cannot be padding")


class _TemporalScale(nn.Module):
    def __init__(
        self,
        branch_channels: int,
        kernel_size: int,
        output_tokens: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                1,
                branch_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.GroupNorm(_group_count(branch_channels), branch_channels),
            nn.GELU(),
            nn.Conv1d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                groups=branch_channels,
                bias=False,
            ),
            nn.GroupNorm(_group_count(branch_channels), branch_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(output_tokens),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class _MultiScaleTemporalEncoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        branch_channels: int,
        kernel_sizes: Sequence[int],
        output_tokens: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.scales = nn.ModuleList(
            [
                _TemporalScale(
                    branch_channels,
                    int(kernel_size),
                    output_tokens,
                    dropout,
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.projection = nn.Sequential(
            nn.Conv1d(
                len(kernel_sizes) * branch_channels,
                model_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(model_dim), model_dim),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = torch.cat([scale(values) for scale in self.scales], dim=1)
        return self.projection(encoded).transpose(1, 2)


class _FrequencyEncoder(nn.Module):
    def __init__(
        self, model_dim: int, branch_channels: int, dropout: float
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(
                1,
                branch_channels,
                kernel_size=(5, 3),
                padding=(2, 1),
                bias=False,
            ),
            nn.GroupNorm(_group_count(branch_channels), branch_channels),
            nn.GELU(),
            nn.Conv2d(
                branch_channels,
                model_dim,
                kernel_size=(5, 3),
                stride=(2, 1),
                padding=(2, 1),
                bias=False,
            ),
            nn.GroupNorm(_group_count(model_dim), model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, power: torch.Tensor) -> torch.Tensor:
        log_power = torch.log1p(power.clamp_min(0.0)).unsqueeze(1)
        encoded = self.network(log_power)
        return encoded.mean(dim=2).transpose(1, 2)


class _SingleChannelStatistics(nn.Module):
    bands_hz = (
        (0.5, 4.0),
        (4.0, 8.0),
        (8.0, 13.0),
        (13.0, 30.0),
        (30.0, 70.0),
    )

    def __init__(
        self, sample_rate_hz: float, n_fft: int, normalization_clip: float
    ) -> None:
        super().__init__()
        self.normalization_clip = float(normalization_clip)
        frequencies = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)
        band_masks = []
        for index, (low, high) in enumerate(self.bands_hz):
            if index == len(self.bands_hz) - 1:
                mask = (frequencies >= low) & (frequencies <= high)
            else:
                mask = (frequencies >= low) & (frequencies < high)
            if not bool(mask.any()):
                raise ValueError(
                    f"n_fft={n_fft} has no bins in the {low:g}-{high:g} Hz band"
                )
            band_masks.append(mask.to(torch.float32))
        analysis_mask = (frequencies >= 0.5) & (frequencies <= 70.0)
        if int(analysis_mask.sum()) < 2:
            raise ValueError("STFT must provide at least two 0.5-70 Hz bins")
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.register_buffer("band_masks", torch.stack(band_masks), persistent=False)
        self.register_buffer("analysis_mask", analysis_mask, persistent=False)

    @property
    def feature_dim(self) -> int:
        return len(SINGLE_CHANNEL_STATISTIC_NAMES)

    def forward(
        self, values: torch.Tensor, stft_power: torch.Tensor
    ) -> torch.Tensor:
        # values: [windows, 1, samples]; stft_power: [windows, freq, frames]
        waveform = values[:, 0]
        epsilon = torch.finfo(waveform.dtype).eps
        rms = waveform.square().mean(dim=-1).sqrt()
        first_difference = waveform.diff(dim=-1)
        second_difference = first_difference.diff(dim=-1)
        line_length = first_difference.abs().mean(dim=-1)
        centered = waveform - waveform.mean(dim=-1, keepdim=True)
        activity = centered.square().mean(dim=-1)
        difference_activity = first_difference.square().mean(dim=-1)
        second_difference_activity = second_difference.square().mean(dim=-1)
        mobility = torch.sqrt(difference_activity / activity.clamp_min(epsilon))
        derivative_mobility = torch.sqrt(
            second_difference_activity / difference_activity.clamp_min(epsilon)
        )
        complexity = derivative_mobility / mobility.clamp_min(epsilon)

        rms = torch.log1p(rms.clamp_min(0.0)) / math.log1p(
            self.normalization_clip
        )
        line_length = torch.log1p(line_length.clamp_min(0.0)) / math.log1p(
            2.0 * self.normalization_clip
        )
        mobility = torch.log1p(
            mobility.clamp(0.0, self.normalization_clip)
        ) / math.log1p(self.normalization_clip)
        complexity = torch.log1p(
            complexity.clamp(0.0, self.normalization_clip)
        ) / math.log1p(self.normalization_clip)
        temporal = torch.stack([rms, line_length, mobility, complexity], dim=-1)

        mean_power = stft_power.mean(dim=-1)
        band_masks = self.band_masks.to(mean_power.dtype)
        band_power = torch.einsum("nf,kf->nk", mean_power, band_masks)
        total_band_power = band_power.sum(dim=-1, keepdim=True)
        relative_band_power = band_power / total_band_power.clamp_min(epsilon)
        relative_band_power = torch.where(
            total_band_power > epsilon,
            relative_band_power,
            torch.zeros_like(relative_band_power),
        )

        theta_delta = torch.log(
            (band_power[:, 1] + epsilon) / (band_power[:, 0] + epsilon)
        ).clamp(-4.0, 4.0) / 4.0
        theta_delta = torch.where(
            total_band_power[:, 0] > epsilon,
            theta_delta,
            torch.zeros_like(theta_delta),
        ).unsqueeze(-1)

        analysis_power = mean_power[:, self.analysis_mask]
        analysis_total = analysis_power.sum(dim=-1, keepdim=True)
        probabilities = analysis_power / analysis_total.clamp_min(epsilon)
        spectral_entropy = -(
            probabilities * torch.log(probabilities.clamp_min(epsilon))
        ).sum(dim=-1) / math.log(analysis_power.shape[-1])
        spectral_entropy = torch.where(
            analysis_total[:, 0] > epsilon,
            spectral_entropy,
            torch.zeros_like(spectral_entropy),
        ).unsqueeze(-1)

        analysis_frequencies = self.frequencies[self.analysis_mask].to(
            mean_power.dtype
        )
        maximum_frequency = float(self.bands_hz[-1][1])
        dominant_index = analysis_power.argmax(dim=-1)
        dominant_frequency = analysis_frequencies[dominant_index] / maximum_frequency
        dominant_frequency = torch.where(
            analysis_total[:, 0] > epsilon,
            dominant_frequency,
            torch.zeros_like(dominant_frequency),
        ).unsqueeze(-1)

        cumulative_power = analysis_power.cumsum(dim=-1)
        edge_index = (
            cumulative_power >= 0.95 * analysis_total
        ).to(torch.int64).argmax(dim=-1)
        edge_frequency = analysis_frequencies[edge_index] / maximum_frequency
        edge_frequency = torch.where(
            analysis_total[:, 0] > epsilon,
            edge_frequency,
            torch.zeros_like(edge_frequency),
        ).unsqueeze(-1)

        statistics = torch.cat(
            [
                temporal,
                relative_band_power,
                theta_delta,
                spectral_entropy,
                dominant_frequency,
                edge_frequency,
            ],
            dim=-1,
        )
        return torch.nan_to_num(
            statistics, nan=0.0, posinf=20.0, neginf=-20.0
        )


class _CrossModalFusion(nn.Module):
    def __init__(self, model_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.time_from_frequency = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.frequency_from_time = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.time_norm = nn.LayerNorm(model_dim)
        self.frequency_norm = nn.LayerNorm(model_dim)
        self.statistics_norm = nn.LayerNorm(model_dim)
        self.base_norm = nn.LayerNorm(model_dim)
        self.statistics_residual = nn.Linear(model_dim, model_dim)
        self.gate = nn.Linear(model_dim * 2, model_dim)
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        time_tokens: torch.Tensor,
        frequency_tokens: torch.Tensor,
        statistics_token: torch.Tensor | None,
    ) -> torch.Tensor:
        fused, _ = self.forward_with_base(
            time_tokens, frequency_tokens, statistics_token
        )
        return fused

    def forward_with_base(
        self,
        time_tokens: torch.Tensor,
        frequency_tokens: torch.Tensor,
        statistics_token: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_update, _ = self.time_from_frequency(
            time_tokens,
            frequency_tokens,
            frequency_tokens,
            need_weights=False,
        )
        frequency_update, _ = self.frequency_from_time(
            frequency_tokens,
            time_tokens,
            time_tokens,
            need_weights=False,
        )
        time_summary = self.time_norm(time_tokens + time_update).mean(dim=1)
        frequency_summary = self.frequency_norm(
            frequency_tokens + frequency_update
        ).mean(dim=1)
        base = self.base_norm(0.5 * (time_summary + frequency_summary))
        if statistics_token is None:
            return base, base
        statistics = self.statistics_norm(statistics_token)
        gate = torch.sigmoid(self.gate(torch.cat([base, statistics], dim=-1)))
        fused = self.output_norm(base + gate * self.statistics_residual(statistics))
        return fused, base


class _SingleWindowPhysioEncoder(nn.Module):
    def __init__(
        self,
        *,
        samples_per_window: int,
        sample_rate_hz: float,
        model_dim: int,
        branch_channels: int,
        waveform_tokens: int,
        temporal_kernel_sizes: Sequence[int],
        attention_heads: int,
        dropout: float,
        stft_n_fft: int,
        stft_hop_length: int,
        normalization_clip: float,
        include_statistics_input: bool,
    ) -> None:
        super().__init__()
        self.samples_per_window = int(samples_per_window)
        self.stft_n_fft = int(stft_n_fft)
        self.stft_hop_length = int(stft_hop_length)
        self.include_statistics_input = bool(include_statistics_input)
        self.temporal = _MultiScaleTemporalEncoder(
            model_dim,
            branch_channels,
            temporal_kernel_sizes,
            waveform_tokens,
            dropout,
        )
        self.statistics = _SingleChannelStatistics(
            sample_rate_hz, stft_n_fft, normalization_clip
        )
        self.frequency = _FrequencyEncoder(model_dim, branch_channels, dropout)
        self.statistics_projection = nn.Sequential(
            nn.Linear(self.statistics.feature_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.fusion = _CrossModalFusion(model_dim, attention_heads, dropout)
        self.register_buffer(
            "stft_window", torch.hann_window(stft_n_fft), persistent=False
        )

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stft_power = self._stft_power(values)
        time_tokens = self.temporal(values)
        frequency_tokens = self.frequency(
            stft_power[:, self.statistics.analysis_mask]
        )
        statistics = self.statistics(values.to(stft_power.dtype), stft_power)
        statistics_token = self.statistics_projection(statistics)
        embedding, time_frequency_base = self.fusion.forward_with_base(
            time_tokens,
            frequency_tokens,
            statistics_token if self.include_statistics_input else None,
        )
        return embedding, statistics, time_frequency_base

    def _stft_power(self, values: torch.Tensor) -> torch.Tensor:
        stft_values = values[:, 0]
        if stft_values.dtype in {torch.float16, torch.bfloat16}:
            stft_values = stft_values.float()
        window = self.stft_window.to(
            device=stft_values.device, dtype=stft_values.dtype
        )
        spectrum = torch.stft(
            stft_values,
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop_length,
            window=window,
            center=True,
            return_complex=True,
        )
        power = spectrum.abs().square() / window.square().sum().clamp_min(1e-8)
        return torch.nan_to_num(power, nan=0.0, posinf=1e12, neginf=0.0)


class _MaskedEpochTCNBlock(nn.Module):
    def __init__(self, model_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=model_dim,
            bias=False,
        )
        self.pointwise = nn.Conv1d(model_dim, model_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, values: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        update = self.depthwise(values.transpose(1, 2))
        update = self.pointwise(update).transpose(1, 2)
        update = self.dropout(F.gelu(self.norm(update)))
        output = values + update
        return output.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class PhysioSequenceNet(nn.Module):
    """Single-channel physiology-informed seizure sequence classifier.

    The model first encodes each four-second window independently from its raw
    waveform, differentiable STFT, and same-channel physiology statistics.
    A non-causal dilated TCN and Transformer then operate across consecutive
    windows.  Only the centre window is classified.  ``padding_mask`` follows
    the PyTorch convention: ``True`` denotes a padded context window.

    Signals must already use robust baseline normalization fitted without
    validation/test leakage.  No per-window scaling is performed here, so the
    within-channel relative-amplitude information remains available.
    """

    input_layout = "[batch, epochs, 1, samples]"
    statistic_names = SINGLE_CHANNEL_STATISTIC_NAMES
    uses_sequence_context = True

    def __init__(
        self,
        *,
        classes: int = 2,
        samples_per_window: int = 1000,
        sample_rate_hz: float = 250.0,
        sequence_length: int = 21,
        model_dim: int = 64,
        branch_channels: int = 16,
        waveform_tokens: int = 24,
        temporal_kernel_sizes: Sequence[int] = (9, 31, 101),
        attention_heads: int = 4,
        epoch_tcn_dilations: Sequence[int] = (1, 2, 4, 8),
        transformer_layers: int = 2,
        dropout: float = 0.15,
        stft_n_fft: int = 128,
        stft_hop_length: int = 32,
        normalization_clip: float = 20.0,
        include_statistics_input: bool = True,
        auxiliary_statistics: bool = False,
    ) -> None:
        super().__init__()
        self._validate_configuration(
            classes=classes,
            samples_per_window=samples_per_window,
            sample_rate_hz=sample_rate_hz,
            sequence_length=sequence_length,
            model_dim=model_dim,
            attention_heads=attention_heads,
            temporal_kernel_sizes=temporal_kernel_sizes,
            epoch_tcn_dilations=epoch_tcn_dilations,
            transformer_layers=transformer_layers,
            stft_n_fft=stft_n_fft,
            stft_hop_length=stft_hop_length,
            normalization_clip=normalization_clip,
        )
        self.classes = int(classes)
        self.samples_per_window = int(samples_per_window)
        self.sample_rate_hz = float(sample_rate_hz)
        self.sequence_length = int(sequence_length)
        self.center_index = self.sequence_length // 2
        self.model_dim = int(model_dim)
        self.include_statistics_input = bool(include_statistics_input)
        self.auxiliary_statistics = bool(auxiliary_statistics)

        self.window_encoder = _SingleWindowPhysioEncoder(
            samples_per_window=self.samples_per_window,
            sample_rate_hz=self.sample_rate_hz,
            model_dim=self.model_dim,
            branch_channels=int(branch_channels),
            waveform_tokens=int(waveform_tokens),
            temporal_kernel_sizes=temporal_kernel_sizes,
            attention_heads=int(attention_heads),
            dropout=float(dropout),
            stft_n_fft=int(stft_n_fft),
            stft_hop_length=int(stft_hop_length),
            normalization_clip=float(normalization_clip),
            include_statistics_input=self.include_statistics_input,
        )
        self.epoch_tcn = nn.ModuleList(
            [
                _MaskedEpochTCNBlock(
                    self.model_dim, int(dilation), float(dropout)
                )
                for dilation in epoch_tcn_dilations
            ]
        )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(attention_heads),
            dim_feedforward=self.model_dim * 3,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            transformer_layer,
            num_layers=int(transformer_layers),
            enable_nested_tensor=False,
        )
        self.embedding_norm = nn.LayerNorm(self.model_dim)
        hidden_dim = max(self.model_dim // 2, self.classes)
        self.classifier = nn.Sequential(
            nn.Linear(self.model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.classes),
        )
        if self.auxiliary_statistics:
            self.statistics_regression: nn.Module | None = nn.Sequential(
                nn.Linear(self.model_dim, self.model_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.model_dim, self.statistics_dim),
            )
        else:
            self.statistics_regression = None

    @property
    def statistics_dim(self) -> int:
        return len(self.statistic_names)

    def forward(
        self,
        signals: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        padding_mask = self._validate_inputs(signals, padding_mask)
        clean_signals = torch.nan_to_num(
            signals, nan=0.0, posinf=0.0, neginf=0.0
        )
        clean_signals = clean_signals.masked_fill(
            padding_mask[:, :, None, None], 0.0
        )

        batch_size = signals.shape[0]
        flat_signals = clean_signals.reshape(
            batch_size * self.sequence_length, 1, self.samples_per_window
        )
        flat_embeddings, flat_statistics, flat_auxiliary_source = self.encode_windows(
            flat_signals, return_auxiliary_source=True
        )
        sequence = flat_embeddings.reshape(
            batch_size, self.sequence_length, self.model_dim
        )
        center_statistics = flat_statistics.reshape(
            batch_size, self.sequence_length, self.statistics_dim
        )[:, self.center_index]
        center_auxiliary_source = flat_auxiliary_source.reshape(
            batch_size, self.sequence_length, self.model_dim
        )[:, self.center_index]
        return self.contextualize_embeddings(
            sequence,
            padding_mask,
            center_statistics=center_statistics,
            auxiliary_source=center_auxiliary_source,
        )

    def encode_windows(
        self,
        signals: torch.Tensor,
        *,
        return_auxiliary_source: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Encode independent windows once for efficient continuous inference."""

        if signals.ndim != 3 or tuple(signals.shape[1:]) != (
            1,
            self.samples_per_window,
        ):
            raise ValueError(
                f"expected independent windows [batch, 1, {self.samples_per_window}], "
                f"received {tuple(signals.shape)}"
            )
        if not signals.is_floating_point():
            raise TypeError("signals must be a floating-point tensor")
        clean = torch.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
        embedding, statistics, auxiliary_source = self.window_encoder(clean)
        if return_auxiliary_source:
            return embedding, statistics, auxiliary_source
        return embedding, statistics

    def contextualize_embeddings(
        self,
        sequence: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        center_statistics: torch.Tensor | None = None,
        auxiliary_source: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Apply the unchanged TCN/Transformer head to precomputed embeddings."""

        if sequence.ndim != 3 or tuple(sequence.shape[1:]) != (
            self.sequence_length,
            self.model_dim,
        ):
            raise ValueError(
                "expected embeddings "
                f"[batch, {self.sequence_length}, {self.model_dim}], "
                f"received {tuple(sequence.shape)}"
            )
        if padding_mask.dtype != torch.bool or tuple(padding_mask.shape) != tuple(
            sequence.shape[:2]
        ):
            raise ValueError("padding_mask must be boolean and align with embeddings")
        if bool(padding_mask[:, self.center_index].any()):
            raise ValueError("the centre epoch cannot be padding")
        sequence = sequence.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        for block in self.epoch_tcn:
            sequence = block(sequence, padding_mask)

        sequence = _position_encoding(sequence)
        sequence = sequence.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        contextualized = self.context(
            sequence, src_key_padding_mask=padding_mask
        )
        contextualized = contextualized.masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        embedding = self.embedding_norm(contextualized[:, self.center_index])
        outputs = {
            "logits": self.classifier(embedding),
            "embedding": embedding,
        }

        if self.statistics_regression is not None:
            if center_statistics is None or tuple(center_statistics.shape) != (
                sequence.shape[0],
                self.statistics_dim,
            ):
                raise ValueError(
                    "the auxiliary model requires center_statistics "
                    f"[batch, {self.statistics_dim}]"
                )
            target = center_statistics.detach()
            regression_source = embedding if auxiliary_source is None else auxiliary_source
            if tuple(regression_source.shape) != tuple(embedding.shape):
                raise ValueError("auxiliary_source must align with the centre embedding")
            outputs.update(
                {
                    # Predict physiology from the time-frequency representation,
                    # before the explicit statistics token is fused.  This avoids
                    # the previous shortcut of reconstructing a feature that was
                    # already present in the regression input.
                    "statistics_prediction": self.statistics_regression(
                        regression_source
                    ),
                    "statistics_target": target,
                    "statistics_mask": torch.isfinite(target),
                }
            )
        return outputs

    def _validate_inputs(
        self, signals: torch.Tensor, padding_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if signals.ndim != 4:
            raise ValueError(
                "expected signals "
                f"[batch, {self.sequence_length}, 1, {self.samples_per_window}], "
                f"received {tuple(signals.shape)}"
            )
        expected = (self.sequence_length, 1, self.samples_per_window)
        if tuple(signals.shape[1:]) != expected:
            raise ValueError(
                "expected signals "
                f"[batch, {self.sequence_length}, 1, {self.samples_per_window}], "
                f"received {tuple(signals.shape)}"
            )
        if not signals.is_floating_point():
            raise TypeError("signals must be a floating-point tensor")
        if padding_mask is None:
            padding_mask = torch.zeros(
                signals.shape[:2], device=signals.device, dtype=torch.bool
            )
        else:
            if padding_mask.dtype != torch.bool:
                raise TypeError("padding_mask must be a boolean tensor")
            if tuple(padding_mask.shape) != tuple(signals.shape[:2]):
                raise ValueError(
                    f"expected padding_mask {tuple(signals.shape[:2])}, "
                    f"received {tuple(padding_mask.shape)}"
                )
            padding_mask = padding_mask.to(device=signals.device)
        if bool(padding_mask[:, self.center_index].any()):
            raise ValueError("the centre epoch cannot be padding")
        return padding_mask

    @staticmethod
    def _validate_configuration(
        *,
        classes: int,
        samples_per_window: int,
        sample_rate_hz: float,
        sequence_length: int,
        model_dim: int,
        attention_heads: int,
        temporal_kernel_sizes: Sequence[int],
        epoch_tcn_dilations: Sequence[int],
        transformer_layers: int,
        stft_n_fft: int,
        stft_hop_length: int,
        normalization_clip: float,
    ) -> None:
        if classes < 2:
            raise ValueError("classes must be at least 2")
        if samples_per_window <= 0 or sample_rate_hz <= 0:
            raise ValueError("sample counts and sample rate must be positive")
        if sequence_length <= 0 or sequence_length % 2 == 0:
            raise ValueError("sequence_length must be a positive odd integer")
        if model_dim <= 0 or attention_heads <= 0:
            raise ValueError("model and attention dimensions must be positive")
        if model_dim % attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        if not temporal_kernel_sizes or any(
            int(size) <= 0 or int(size) % 2 == 0
            for size in temporal_kernel_sizes
        ):
            raise ValueError(
                "temporal kernel sizes must be non-empty positive odd values"
            )
        if not epoch_tcn_dilations or any(
            int(dilation) <= 0 for dilation in epoch_tcn_dilations
        ):
            raise ValueError("epoch_tcn_dilations must be non-empty and positive")
        if transformer_layers <= 0:
            raise ValueError("transformer_layers must be positive")
        if stft_n_fft <= 0 or stft_hop_length <= 0:
            raise ValueError("STFT sizes must be positive")
        if stft_n_fft > samples_per_window:
            raise ValueError("stft_n_fft cannot exceed samples_per_window")
        if normalization_clip <= 0:
            raise ValueError("normalization_clip must be positive")


def build_single_channel_model(
    name: str,
    *,
    classes: int = 2,
    samples_per_window: int = 1000,
    sample_rate_hz: float = 250.0,
    sequence_length: int = 21,
    **model_kwargs: object,
) -> nn.Module:
    """Construct a channel-blind model from a training-config friendly name."""

    key = name.strip().lower().replace("-", "_").replace("+", "_")
    if key in {"eegwavenet_sc", "eegwavenet_single_channel"}:
        if model_kwargs:
            unexpected = ", ".join(sorted(model_kwargs))
            raise TypeError(f"unexpected EEGWaveNetSC arguments: {unexpected}")
        return EEGWaveNetSC(
            samples_per_window=samples_per_window,
            classes=classes,
            sequence_length=sequence_length,
        )
    if key in {"physio_sequence", "physio_sequence_no_aux", "physio_sequence_stats_input"}:
        include_statistics_input = True
        auxiliary_statistics = False
    elif key in {
        "physio_sequence_stats",
        "physio_sequence_with_stats",
        "physio_sequence_stats_aux",
    }:
        include_statistics_input = True
        auxiliary_statistics = True
    elif key in {"physio_sequence_time_frequency", "physio_sequence_time_freq"}:
        include_statistics_input = False
        auxiliary_statistics = False
    else:
        raise ValueError(f"unknown single-channel model: {name!r}")

    supplied_auxiliary = model_kwargs.pop("auxiliary_statistics", None)
    if supplied_auxiliary is not None and bool(supplied_auxiliary) != auxiliary_statistics:
        raise ValueError(
            "model name and auxiliary_statistics specify conflicting variants"
        )
    supplied_statistics_input = model_kwargs.pop("include_statistics_input", None)
    if (
        supplied_statistics_input is not None
        and bool(supplied_statistics_input) != include_statistics_input
    ):
        raise ValueError(
            "model name and include_statistics_input specify conflicting variants"
        )
    return PhysioSequenceNet(
        classes=classes,
        samples_per_window=samples_per_window,
        sample_rate_hz=sample_rate_hz,
        sequence_length=sequence_length,
        include_statistics_input=include_statistics_input,
        auxiliary_statistics=auxiliary_statistics,
        **model_kwargs,
    )


EEGWaveNetSingleChannel = EEGWaveNetSC
PhysioSequenceSingleChannel = PhysioSequenceNet


__all__ = [
    "EEGWaveNetSC",
    "EEGWaveNetSingleChannel",
    "PhysioSequenceNet",
    "PhysioSequenceSingleChannel",
    "SINGLE_CHANNEL_STATISTIC_NAMES",
    "build_single_channel_model",
]
