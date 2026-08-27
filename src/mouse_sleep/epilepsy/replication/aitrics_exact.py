"""Clean-room reconstruction of the AITRICS CHIL 2022 raw CNN2D-LSTM.

The public repository contains a blocking ``exit(1)`` in ``forward``.  This
module removes only that debug interruption and preserves the disclosed raw
architecture, tensor order, state propagation, and truncated recurrent-state
semantics.  It does not claim paper-performance replication without the
licensed TUSZ v1.5.2 data and the original patient protocol.
"""

from __future__ import annotations

from typing import TypeAlias

import torch
from torch import nn

HiddenState: TypeAlias = tuple[torch.Tensor, torch.Tensor]


class AITRICSRawCNN2DLSTMPaperExact(nn.Module):
    """Raw CNN2D-LSTM architecture released by Lee et al. (CHIL 2022).

    The source-model input layout is ``[batch, 800, channels]``: four seconds
    sampled at 200 Hz.  The raw paper setting uses 20 bipolar channels, but no
    convolution mixes the montage-height axis before adaptive pooling, so the
    same published parameterization can ingest another positive channel count.

    A 30-second source batch is evaluated as 26 overlapping four-second
    windows.  Call :meth:`reset_state` once at the beginning of that sequence,
    then call :meth:`forward` once per window.  The recurrent state is detached
    before every call, exactly matching the released trainer: values propagate
    across windows, gradients do not.
    """

    input_samples = 800
    hidden_dim = 256
    num_layers = 2

    def __init__(self, classes: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        if classes < 2:
            raise ValueError("classes must be at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.classes = int(classes)
        self.dropout_rate = float(dropout)
        # Preserve the released module registry, including its one unused
        # trainable PReLU scalar.  The raw path always selects ``relu``; the
        # distinction is exposed in verification as registered vs active-path
        # parameter counts.
        self.activations = nn.ModuleDict(
            {
                "lrelu": nn.LeakyReLU(),
                "prelu": nn.PReLU(),
                "relu": nn.ReLU(inplace=True),
                "tanh": nn.Tanh(),
                "sigmoid": nn.Sigmoid(),
                "leaky_relu": nn.LeakyReLU(0.2),
                "elu": nn.ELU(),
            }
        )

        def block(
            in_channels: int,
            out_channels: int,
            kernel: int,
            stride: int,
        ) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, kernel),
                    stride=(1, stride),
                    padding=(0, kernel // 2),
                    bias=True,
                ),
                nn.BatchNorm2d(out_channels),
                self.activations["relu"],
                nn.Dropout(self.dropout_rate),
            )

        self.features = nn.Sequential(
            block(1, 64, kernel=51, stride=4),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
            block(64, 128, kernel=21, stride=2),
            block(128, 256, kernel=9, stride=2),
        )
        # Attribute spelling intentionally follows the released repository.
        self.agvpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 64, bias=True),
            nn.BatchNorm1d(64),
            self.activations["relu"],
            nn.Linear(64, self.classes, bias=True),
        )
        self._hidden: HiddenState | None = None

    @property
    def hidden(self) -> HiddenState | None:
        """Expose the state under the name used by the released model."""

        return self._hidden

    def reset_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> HiddenState:
        """Reset recurrent state at a 30-second sequence boundary."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        parameter = next(self.parameters())
        resolved_device = parameter.device if device is None else torch.device(device)
        resolved_dtype = parameter.dtype if dtype is None else dtype
        shape = (self.num_layers, int(batch_size), self.hidden_dim)
        self._hidden = (
            torch.zeros(shape, device=resolved_device, dtype=resolved_dtype),
            torch.zeros(shape, device=resolved_device, dtype=resolved_dtype),
        )
        return self._hidden

    def detach_state(self) -> HiddenState | None:
        """Truncate back-propagation while retaining recurrent values."""

        if self._hidden is not None:
            self._hidden = tuple(value.detach() for value in self._hidden)  # type: ignore[assignment]
        return self._hidden

    def reset_or_detach_state(self, x: torch.Tensor) -> HiddenState:
        """Match source behavior while tolerating a changed last batch size."""

        batch_size = int(x.shape[0])
        state = self._hidden
        if (
            state is None
            or state[0].shape[1] != batch_size
            or state[0].device != x.device
            or state[0].dtype != x.dtype
        ):
            return self.reset_state(batch_size, device=x.device, dtype=x.dtype)
        detached = self.detach_state()
        assert detached is not None
        return detached

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the one-token ``[batch, 1, 256]`` window embedding."""

        self._validate_input(x)
        # Released forward: [B,T,C] -> [B,C,T] -> [B,1,C,T].
        z = x.permute(0, 2, 1).unsqueeze(1)
        z = self.features(z)
        z = self.agvpool(z)
        z = torch.squeeze(z, 2).permute(0, 2, 1)
        return z

    def forward(
        self,
        x: torch.Tensor,
        *,
        reset_state: bool = False,
    ) -> tuple[torch.Tensor, HiddenState]:
        """Return ``(logits, hidden)`` with source-faithful state handling."""

        tokens = self.forward_features(x)
        if reset_state:
            hidden = self.reset_state(
                int(x.shape[0]), device=x.device, dtype=x.dtype
            )
        else:
            hidden = self.reset_or_detach_state(x)
        output, self._hidden = self.lstm(tokens, hidden)
        logits = self.classifier(output[:, -1, :])
        return logits, self._hidden

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                f"expected [batch, 800, channels], received shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.input_samples:
            raise ValueError(
                f"expected {self.input_samples} time samples, received {x.shape[1]}"
            )
        if x.shape[2] < 1:
            raise ValueError("at least one EEG channel is required")


__all__ = ["AITRICSRawCNN2DLSTMPaperExact", "HiddenState"]
