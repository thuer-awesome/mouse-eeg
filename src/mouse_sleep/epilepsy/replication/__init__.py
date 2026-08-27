"""Paper-faithful seizure-model replications kept separate from adaptations."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from mouse_sleep.epilepsy.replication.eegnet_exact import EEGNet82, EEGNet82PaperExact
from mouse_sleep.epilepsy.replication.eegwavenet import EEGWaveNetPaperExact
from mouse_sleep.epilepsy.replication.aitrics_exact import (
    AITRICSRawCNN2DLSTMPaperExact,
)

__all__ = [
    "AITRICSRawCNN2DLSTMPaperExact",
    "EEGNet82",
    "EEGNet82PaperExact",
    "EEGWaveNetPaperExact",
]
