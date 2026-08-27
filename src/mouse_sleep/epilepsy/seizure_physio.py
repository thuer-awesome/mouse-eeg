from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


SEIZURE_STATISTIC_NAMES = (
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
    "interchannel_correlation",
)


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(channels, maximum), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _position_encoding(values: torch.Tensor) -> torch.Tensor:
    length, dimension = values.shape[1], values.shape[2]
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


class _TemporalScale(nn.Module):
    def __init__(
        self,
        channels: int,
        branch_channels: int,
        kernel_size: int,
        output_tokens: int,
        dropout: float,
    ) -> None:
        super().__init__()
        expanded_channels = channels * branch_channels
        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                expanded_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(_group_count(expanded_channels), expanded_channels),
            nn.GELU(),
            nn.Conv1d(expanded_channels, branch_channels, kernel_size=1, bias=False),
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
        channels: int,
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
                    channels,
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
        self.availability = nn.Linear(channels, model_dim, bias=False)

    def forward(
        self, values: torch.Tensor, channel_mask: torch.Tensor
    ) -> torch.Tensor:
        encoded = torch.cat([scale(values) for scale in self.scales], dim=1)
        tokens = self.projection(encoded).transpose(1, 2)
        availability = self.availability(channel_mask.to(tokens.dtype)).unsqueeze(1)
        return tokens + availability


class _FrequencyEncoder(nn.Module):
    def __init__(
        self,
        channels: int,
        model_dim: int,
        branch_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(
                channels * 2,
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
        self.availability = nn.Linear(channels, model_dim, bias=False)

    def forward(
        self, power: torch.Tensor, channel_mask: torch.Tensor
    ) -> torch.Tensor:
        log_power = torch.log1p(power.clamp_min(0.0))
        availability_maps = channel_mask[:, :, None, None].to(log_power.dtype)
        availability_maps = availability_maps.expand_as(log_power)
        encoded = self.network(torch.cat([log_power, availability_maps], dim=1))
        tokens = encoded.mean(dim=2).transpose(1, 2)
        availability = self.availability(channel_mask.to(tokens.dtype)).unsqueeze(1)
        return tokens + availability


class _BidirectionalCrossFusion(nn.Module):
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
        self.cross_projection = nn.Linear(model_dim, model_dim)
        self.gate = nn.Linear(model_dim * 2, model_dim)
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self, time_tokens: torch.Tensor, frequency_tokens: torch.Tensor
    ) -> torch.Tensor:
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
        attended_time = self.time_norm(time_tokens + time_update)
        attended_frequency = self.frequency_norm(
            frequency_tokens + frequency_update
        )

        frequency_base = F.interpolate(
            frequency_tokens.transpose(1, 2),
            size=time_tokens.shape[1],
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        frequency_cross = F.interpolate(
            attended_frequency.transpose(1, 2),
            size=time_tokens.shape[1],
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        base = 0.5 * (time_tokens + frequency_base)
        cross = 0.5 * (attended_time + frequency_cross)
        gate = torch.sigmoid(self.gate(torch.cat([base, cross], dim=-1)))
        return self.output_norm(base + gate * self.cross_projection(cross))


class _DilatedResidualBlock(nn.Module):
    def __init__(self, model_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                model_dim,
                model_dim,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=model_dim,
                bias=False,
            ),
            nn.Conv1d(model_dim, model_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(model_dim), model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class _SeizureStatistics(nn.Module):
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
        if normalization_clip <= 0:
            raise ValueError("normalization_clip must be positive")
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
        return len(SEIZURE_STATISTIC_NAMES)

    def forward(
        self,
        values: torch.Tensor,
        channel_mask: torch.Tensor,
        stft_power: torch.Tensor,
    ) -> torch.Tensor:
        epsilon = torch.finfo(values.dtype).eps
        channel_weights = channel_mask.to(values.dtype)

        rms = values.square().mean(dim=-1).sqrt()
        first_difference = values.diff(dim=-1)
        second_difference = first_difference.diff(dim=-1)
        line_length = first_difference.abs().mean(dim=-1)
        centered = values - values.mean(dim=-1, keepdim=True)
        activity = centered.square().mean(dim=-1)
        difference_activity = first_difference.square().mean(dim=-1)
        second_difference_activity = second_difference.square().mean(dim=-1)
        mobility = torch.sqrt(difference_activity / activity.clamp_min(epsilon))
        derivative_mobility = torch.sqrt(
            second_difference_activity / difference_activity.clamp_min(epsilon)
        )
        complexity = derivative_mobility / mobility.clamp_min(epsilon)
        mobility = mobility.clamp(0.0, self.normalization_clip)
        complexity = complexity.clamp(0.0, self.normalization_clip)
        rms = torch.log1p(rms.clamp_min(0.0)) / math.log1p(
            self.normalization_clip
        )
        line_length = torch.log1p(line_length.clamp_min(0.0)) / math.log1p(
            2.0 * self.normalization_clip
        )
        mobility = torch.log1p(mobility) / math.log1p(self.normalization_clip)
        complexity = torch.log1p(complexity) / math.log1p(
            self.normalization_clip
        )
        temporal = torch.stack(
            [rms, line_length, mobility, complexity], dim=-1
        )
        temporal = self._masked_mean(temporal, channel_weights)

        mean_power = stft_power.mean(dim=-1)
        band_masks = self.band_masks.to(mean_power.dtype)
        band_power = torch.einsum("bcf,kf->bck", mean_power, band_masks)
        total_band_power = band_power.sum(dim=-1, keepdim=True)
        relative_band_power = band_power / total_band_power.clamp_min(epsilon)
        relative_band_power = torch.where(
            total_band_power > epsilon,
            relative_band_power,
            torch.zeros_like(relative_band_power),
        )
        relative_band_power = self._masked_mean(
            relative_band_power, channel_weights
        )

        theta_delta = torch.log(
            (band_power[..., 1] + epsilon) / (band_power[..., 0] + epsilon)
        ).clamp(-4.0, 4.0) / 4.0
        theta_delta = torch.where(
            total_band_power.squeeze(-1) > epsilon,
            theta_delta,
            torch.zeros_like(theta_delta),
        )
        theta_delta = self._masked_mean(
            theta_delta.unsqueeze(-1), channel_weights
        )

        analysis_power = mean_power[..., self.analysis_mask]
        analysis_total = analysis_power.sum(dim=-1, keepdim=True)
        probabilities = analysis_power / analysis_total.clamp_min(epsilon)
        spectral_entropy = -(
            probabilities * torch.log(probabilities.clamp_min(epsilon))
        ).sum(dim=-1)
        spectral_entropy = spectral_entropy / math.log(analysis_power.shape[-1])
        spectral_entropy = torch.where(
            analysis_total.squeeze(-1) > epsilon,
            spectral_entropy,
            torch.zeros_like(spectral_entropy),
        )
        spectral_entropy = self._masked_mean(
            spectral_entropy.unsqueeze(-1), channel_weights
        )

        analysis_frequencies = self.frequencies[self.analysis_mask].to(
            mean_power.dtype
        )
        maximum_frequency = float(self.bands_hz[-1][1])
        dominant_index = analysis_power.argmax(dim=-1)
        dominant_frequency = analysis_frequencies[dominant_index] / maximum_frequency
        dominant_frequency = torch.where(
            analysis_total.squeeze(-1) > epsilon,
            dominant_frequency,
            torch.zeros_like(dominant_frequency),
        )
        dominant_frequency = self._masked_mean(
            dominant_frequency.unsqueeze(-1), channel_weights
        )

        cumulative_power = analysis_power.cumsum(dim=-1)
        edge_threshold = 0.95 * analysis_total
        edge_index = (cumulative_power >= edge_threshold).to(torch.int64).argmax(
            dim=-1
        )
        edge_frequency = analysis_frequencies[edge_index] / maximum_frequency
        edge_frequency = torch.where(
            analysis_total.squeeze(-1) > epsilon,
            edge_frequency,
            torch.zeros_like(edge_frequency),
        )
        edge_frequency = self._masked_mean(
            edge_frequency.unsqueeze(-1), channel_weights
        )

        channel_0 = centered[:, 0]
        channel_1 = centered[:, 1]
        covariance = (channel_0 * channel_1).mean(dim=-1)
        variance_product = channel_0.square().mean(dim=-1) * channel_1.square().mean(
            dim=-1
        )
        correlation = covariance / variance_product.sqrt().clamp_min(epsilon)
        both_valid = channel_mask[:, 0] & channel_mask[:, 1]
        correlation = torch.where(
            both_valid & (variance_product > epsilon),
            correlation.clamp(-1.0, 1.0),
            torch.zeros_like(correlation),
        ).unsqueeze(-1)

        statistics = torch.cat(
            [
                temporal,
                relative_band_power,
                theta_delta,
                spectral_entropy,
                dominant_frequency,
                edge_frequency,
                correlation,
            ],
            dim=-1,
        )
        return torch.nan_to_num(
            statistics, nan=0.0, posinf=20.0, neginf=-20.0
        )

    @staticmethod
    def _masked_mean(
        values: torch.Tensor, channel_weights: torch.Tensor
    ) -> torch.Tensor:
        weights = channel_weights.unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (values * weights).sum(dim=1) / denominator


class SeizurePhysioNet(nn.Module):
    """Single-window two-channel EEG seizure detector with physiology targets.

    Inputs are expected to have already been normalized with statistics fitted
    only on training non-seizure data. No per-window normalization is applied,
    so relative amplitude information remains available to the network.
    """

    input_layout = "[batch, 2, 1000]"
    statistic_names = SEIZURE_STATISTIC_NAMES

    def __init__(
        self,
        channels: int = 2,
        classes: int = 2,
        samples_per_window: int = 1000,
        sample_rate_hz: float = 250.0,
        model_dim: int = 64,
        branch_channels: int = 16,
        waveform_tokens: int = 32,
        temporal_kernel_sizes: Sequence[int] = (9, 31, 101),
        attention_heads: int = 4,
        tcn_dilations: Sequence[int] = (1, 2, 4),
        transformer_layers: int = 1,
        dropout: float = 0.15,
        stft_n_fft: int = 128,
        stft_hop_length: int = 32,
        normalization_clip: float = 20.0,
    ) -> None:
        super().__init__()
        if channels != 2:
            raise ValueError("SeizurePhysioNet requires exactly two EEG channels")
        if classes != 2:
            raise ValueError("SeizurePhysioNet is defined for binary classification")
        if samples_per_window <= 0 or waveform_tokens <= 0:
            raise ValueError("sample and token counts must be positive")
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if model_dim % attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        if stft_n_fft <= 0 or stft_hop_length <= 0:
            raise ValueError("STFT sizes must be positive")
        if normalization_clip <= 0:
            raise ValueError("normalization_clip must be positive")
        if stft_n_fft > samples_per_window:
            raise ValueError("stft_n_fft cannot exceed samples_per_window")
        if not temporal_kernel_sizes or any(
            int(size) <= 0 or int(size) % 2 == 0 for size in temporal_kernel_sizes
        ):
            raise ValueError("temporal kernel sizes must be non-empty positive odd values")
        if transformer_layers <= 0:
            raise ValueError("transformer_layers must be positive")

        self.channels = channels
        self.classes = classes
        self.samples_per_window = samples_per_window
        self.sample_rate_hz = float(sample_rate_hz)
        self.stft_n_fft = int(stft_n_fft)
        self.stft_hop_length = int(stft_hop_length)
        self.temporal = _MultiScaleTemporalEncoder(
            channels,
            model_dim,
            branch_channels,
            temporal_kernel_sizes,
            waveform_tokens,
            dropout,
        )
        self.frequency = _FrequencyEncoder(
            channels, model_dim, branch_channels, dropout
        )
        self.fusion = _BidirectionalCrossFusion(
            model_dim, attention_heads, dropout
        )
        self.tcn = nn.Sequential(
            *[
                _DilatedResidualBlock(model_dim, int(dilation), dropout)
                for dilation in tcn_dilations
            ]
        )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=attention_heads,
            dim_feedforward=model_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            transformer_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.embedding_norm = nn.LayerNorm(model_dim)
        hidden_dim = max(model_dim // 2, classes)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, classes),
        )
        self.statistics = _SeizureStatistics(
            sample_rate_hz, stft_n_fft, normalization_clip
        )
        self.statistics_regression = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, self.statistics.feature_dim),
        )
        self.register_buffer(
            "stft_window", torch.hann_window(stft_n_fft), persistent=False
        )

    @property
    def statistics_dim(self) -> int:
        return self.statistics.feature_dim

    def forward(
        self, signals: torch.Tensor, channel_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        self._validate_inputs(signals, channel_mask)
        mask = torch.nan_to_num(
            channel_mask.to(device=signals.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).gt(0.5)
        clean_signals = torch.nan_to_num(
            signals, nan=0.0, posinf=0.0, neginf=0.0
        )
        masked_signals = torch.where(
            mask.unsqueeze(-1), clean_signals, torch.zeros_like(clean_signals)
        )

        stft_power = self._stft_power(masked_signals)
        time_tokens = self.temporal(masked_signals, mask)
        analysis_power = stft_power[:, :, self.statistics.analysis_mask]
        frequency_tokens = self.frequency(analysis_power, mask)
        fused_tokens = self.fusion(time_tokens, frequency_tokens)
        temporal_context = self.tcn(fused_tokens.transpose(1, 2)).transpose(1, 2)
        contextualized = self.context(_position_encoding(temporal_context))
        embedding = self.embedding_norm(contextualized.mean(dim=1))
        statistics_target = self.statistics(
            masked_signals.to(stft_power.dtype), mask, stft_power
        ).detach()
        any_valid = mask.any(dim=1, keepdim=True)
        statistics_mask = any_valid.expand(-1, self.statistics.feature_dim).clone()
        statistics_mask[:, -1] = mask.all(dim=1)
        return {
            "logits": self.classifier(embedding),
            "statistics_prediction": self.statistics_regression(embedding),
            "statistics_target": statistics_target,
            "statistics_mask": statistics_mask,
            "embedding": embedding,
        }

    def _stft_power(self, values: torch.Tensor) -> torch.Tensor:
        stft_values = values
        if stft_values.dtype in {torch.float16, torch.bfloat16}:
            stft_values = stft_values.float()
        window = self.stft_window.to(
            device=stft_values.device, dtype=stft_values.dtype
        )
        spectrum = torch.stft(
            stft_values.reshape(-1, stft_values.shape[-1]),
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop_length,
            window=window,
            center=True,
            return_complex=True,
        )
        power = spectrum.abs().square() / window.square().sum().clamp_min(1e-8)
        power = power.reshape(
            values.shape[0], values.shape[1], power.shape[-2], power.shape[-1]
        )
        return torch.nan_to_num(power, nan=0.0, posinf=1e12, neginf=0.0)

    def _validate_inputs(
        self, signals: torch.Tensor, channel_mask: torch.Tensor
    ) -> None:
        if signals.ndim != 3:
            raise ValueError(
                f"expected signals [batch, 2, {self.samples_per_window}], "
                f"received {tuple(signals.shape)}"
            )
        if signals.shape[1] != self.channels or signals.shape[2] != self.samples_per_window:
            raise ValueError(
                f"expected signals [batch, {self.channels}, {self.samples_per_window}], "
                f"received {tuple(signals.shape)}"
            )
        if not signals.is_floating_point():
            raise TypeError("signals must be a floating-point tensor")
        if channel_mask.ndim != 2 or channel_mask.shape != signals.shape[:2]:
            raise ValueError(
                f"expected channel_mask {tuple(signals.shape[:2])}, "
                f"received {tuple(channel_mask.shape)}"
            )


__all__ = ["SEIZURE_STATISTIC_NAMES", "SeizurePhysioNet"]
