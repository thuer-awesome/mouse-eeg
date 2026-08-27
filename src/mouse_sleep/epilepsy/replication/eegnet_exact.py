"""Clean-room PyTorch implementation of the published EEGNet-8,2 network.

The layer order and defaults follow Table 2 and the reference implementation
from Lawhern et al., *Journal of Neural Engineering* 15 (2018) 056013.  This
module is deliberately independent of the rodent-data adapters in this project.

The paper ends the network with a softmax.  ``forward`` returns the pre-softmax
logits so that PyTorch's numerically stable ``CrossEntropyLoss`` can be used;
``predict_proba`` applies that paper-level softmax for inference.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["EEGNet82", "EEGNet82PaperExact"]


class EEGNet82(nn.Module):
    """EEGNet-8,2 with the hyperparameters reported by Lawhern et al.

    Parameters
    ----------
    channels:
        Number of EEG electrodes in each trial.
    samples:
        Number of time samples in each trial.  It fixes the classifier input
        size and must therefore match every input passed to the model.
    classes:
        Number of mutually exclusive output classes.
    dropout_rate:
        Dropout probability.  The reference implementation defaults to 0.5.
    dropout_type:
        ``"dropout"`` is the paper default.  ``"spatial"`` exposes the
        SpatialDropout2D option shipped by the authors for later experiments.
    norm_rate:
        Maximum L2 norm for each output unit of the final dense layer.

    Notes
    -----
    The architecture constants are fixed to ``F1=8``, ``D=2``, ``F2=16``, a
    64-sample temporal kernel, and a 16-sample separable temporal kernel.
    Keras applies weight constraints immediately after each optimizer update.
    Call ``apply_max_norm_constraints`` after ``optimizer.step()`` to preserve
    that behavior in a PyTorch training loop.
    """

    F1 = 8
    D = 2
    F2 = F1 * D
    TEMPORAL_KERNEL_LENGTH = 64
    SEPARABLE_KERNEL_LENGTH = 16
    SPATIAL_MAX_NORM = 1.0

    def __init__(
        self,
        channels: int,
        samples: int,
        classes: int,
        dropout_rate: float = 0.5,
        dropout_type: str = "dropout",
        norm_rate: float = 0.25,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if samples < 32:
            raise ValueError("samples must be at least 32 for 4x and 8x pooling")
        if classes < 2:
            raise ValueError("classes must be at least 2")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")
        if norm_rate <= 0.0:
            raise ValueError("norm_rate must be positive")

        dropout_key = dropout_type.lower().replace("_", "")
        if dropout_key == "dropout":
            dropout_layer: type[nn.Module] = nn.Dropout
        elif dropout_key in {"spatial", "spatialdropout2d", "dropout2d"}:
            dropout_layer = nn.Dropout2d
        else:
            raise ValueError("dropout_type must be 'dropout' or 'spatial'")

        self.channels = int(channels)
        self.samples = int(samples)
        self.classes = int(classes)
        self.norm_rate = float(norm_rate)

        # Block 1: temporal filter bank followed by one spatial filter per
        # temporal feature map and depth multiplier.
        self.temporal_conv = nn.Conv2d(
            1,
            self.F1,
            kernel_size=(1, self.TEMPORAL_KERNEL_LENGTH),
            padding="same",
            bias=False,
        )
        self.temporal_bn = nn.BatchNorm2d(
            self.F1, eps=1e-3, momentum=0.01, affine=True
        )
        self.spatial_conv = nn.Conv2d(
            self.F1,
            self.F2,
            kernel_size=(self.channels, 1),
            groups=self.F1,
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm2d(
            self.F2, eps=1e-3, momentum=0.01, affine=True
        )
        self.block1_activation = nn.ELU()
        self.block1_pool = nn.AvgPool2d(kernel_size=(1, 4))
        self.block1_dropout = dropout_layer(p=dropout_rate)

        # Block 2: a depthwise temporal convolution followed by the 1x1
        # pointwise convolution that forms a separable convolution.
        self.separable_depthwise_conv = nn.Conv2d(
            self.F2,
            self.F2,
            kernel_size=(1, self.SEPARABLE_KERNEL_LENGTH),
            padding="same",
            groups=self.F2,
            bias=False,
        )
        self.separable_pointwise_conv = nn.Conv2d(
            self.F2, self.F2, kernel_size=(1, 1), bias=False
        )
        self.separable_bn = nn.BatchNorm2d(
            self.F2, eps=1e-3, momentum=0.01, affine=True
        )
        self.block2_activation = nn.ELU()
        self.block2_pool = nn.AvgPool2d(kernel_size=(1, 8))
        self.block2_dropout = dropout_layer(p=dropout_rate)

        pooled_samples = (self.samples // 4) // 8
        self.feature_dim = self.F2 * pooled_samples
        self.classifier = nn.Linear(self.feature_dim, self.classes, bias=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use the Glorot/BatchNorm initializers of the Keras reference."""

        for module in self.modules():
            if isinstance(module, nn.Conv2d | nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the flattened EEGNet representation before classification."""

        self._validate_input(x)
        z = x.unsqueeze(1)
        z = self.temporal_bn(self.temporal_conv(z))
        z = self.spatial_conv(z)
        z = self.spatial_bn(z)
        z = self.block1_activation(z)
        z = self.block1_pool(z)
        z = self.block1_dropout(z)

        z = self.separable_depthwise_conv(z)
        z = self.separable_pointwise_conv(z)
        z = self.separable_bn(z)
        z = self.block2_activation(z)
        z = self.block2_pool(z)
        z = self.block2_dropout(z)
        return z.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class logits with shape ``[batch, classes]``."""

        return self.classifier(self.forward_features(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return the softmax probabilities specified by the paper."""

        return torch.softmax(self(x), dim=1)

    @torch.no_grad()
    def apply_max_norm_constraints(self) -> None:
        """Project constrained filters onto the paper's L2-norm balls.

        Invoke this once immediately after every optimizer update, matching
        Keras' ``depthwise_constraint=max_norm(1)`` and the final dense
        ``kernel_constraint=max_norm(norm_rate)``.
        """

        self._project_output_filter_norms_(
            self.spatial_conv.weight, self.SPATIAL_MAX_NORM
        )
        self._project_output_filter_norms_(self.classifier.weight, self.norm_rate)

    @staticmethod
    def _project_output_filter_norms_(weight: torch.Tensor, maximum: float) -> None:
        flat = weight.reshape(weight.shape[0], -1)
        norms = torch.linalg.vector_norm(flat, ord=2, dim=1, keepdim=True)
        scales = (maximum / norms.clamp_min(torch.finfo(weight.dtype).eps)).clamp(
            max=1.0
        )
        weight.mul_(scales.reshape((-1,) + (1,) * (weight.ndim - 1)))

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                f"expected [batch, channels, samples], received shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, received {x.shape[1]}"
            )
        if x.shape[2] != self.samples:
            raise ValueError(f"expected {self.samples} samples, received {x.shape[2]}")


# Descriptive alias used by replication manifests and reports.
EEGNet82PaperExact = EEGNet82
