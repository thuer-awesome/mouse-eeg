"""Clean-room implementation of the published EEGWaveNet architecture.

The module layout follows the architecture disclosed by Thuwajit et al.
(IEEE TII 2021, DOI: 10.1109/TII.2021.3133307). Attribute names intentionally
match the released checkpoints so they can be audited with strict state-dict
loading; no external repository source text is included here.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class EEGWaveNetPaperExact(nn.Module):
    """EEGWaveNet as released for fixed-length EEG seizure classification.

    Inputs have shape ``[batch, channels, samples]``. The returned tensor
    contains class log-probabilities. For the paper's 4-second, 256 Hz setup,
    ``samples`` is 1024.
    """

    _TEMPORAL_NAMES = tuple(f"temp_conv{index}" for index in range(1, 7))
    _SCALE_NAMES = tuple(f"chpool{index}" for index in range(1, 6))

    def __init__(self, n_chans: int, n_classes: int) -> None:
        super().__init__()
        if n_chans < 1:
            raise ValueError("n_chans must be positive")
        if n_classes < 1:
            raise ValueError("n_classes must be positive")

        self.n_chans = int(n_chans)
        self.n_classes = int(n_classes)

        for name in self._TEMPORAL_NAMES:
            setattr(
                self,
                name,
                nn.Conv1d(
                    self.n_chans,
                    self.n_chans,
                    kernel_size=2,
                    stride=2,
                    groups=self.n_chans,
                ),
            )

        for name in self._SCALE_NAMES:
            setattr(self, name, self._make_scale_extractor(self.n_chans))

        self.classifier = nn.Sequential(
            nn.Linear(5 * 32, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, self.n_classes),
        )

    @staticmethod
    def _make_scale_extractor(n_chans: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(n_chans, 32, kernel_size=4),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(32, 32, kernel_size=4),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.01),
        )

    def forward(
        self, x: torch.Tensor, training: bool | None = None
    ) -> torch.Tensor:
        """Return log-probabilities; ``training`` is a compatibility argument."""
        del training
        if x.ndim != 3:
            raise ValueError("Expected input with shape [batch, channels, samples]")
        if x.shape[1] != self.n_chans:
            raise ValueError(
                f"Expected {self.n_chans} channels, received {x.shape[1]}"
            )

        temporal_scales: list[torch.Tensor] = []
        features = x
        for index, name in enumerate(self._TEMPORAL_NAMES):
            features = getattr(self, name)(features)
            if index > 0:
                temporal_scales.append(features)

        pooled = [
            getattr(self, name)(scale).mean(dim=-1)
            for name, scale in zip(self._SCALE_NAMES, temporal_scales, strict=True)
        ]
        logits = self.classifier(torch.cat(pooled, dim=1))
        return F.log_softmax(logits, dim=1)
