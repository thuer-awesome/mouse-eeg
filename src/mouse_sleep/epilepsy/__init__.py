"""Independent mouse EEG seizure-detection pipeline.

This package deliberately does not share sleep labels, splits, or metrics.  The
public AITRICS model is kept as an external reference; the local implementation
is a corrected, configurable reproduction suitable for the two-channel data.
"""

from mouse_sleep.epilepsy.data import build_window_cache, discover_sessions


def build_epilepsy_model(*args, **kwargs):
    """Import the torch-dependent model registry only when it is requested.

    Data audit and morphology modules use the same package namespace but do
    not require a deep-learning runtime.  Deferring this import keeps those
    read-only workflows usable in a scientific Python environment while
    preserving the public model-construction API for training code.
    """

    from mouse_sleep.epilepsy.models import build_epilepsy_model as _build

    return _build(*args, **kwargs)

__all__ = ["discover_sessions", "build_window_cache", "build_epilepsy_model"]
