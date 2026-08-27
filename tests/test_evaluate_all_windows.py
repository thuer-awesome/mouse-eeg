from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_evaluator():
    path = Path(__file__).parents[1] / "scripts" / "evaluate_strict_classifiers_all_windows.py"
    spec = importlib.util.spec_from_file_location("evaluate_all_windows", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load evaluator module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluator_imports_without_training_dependencies() -> None:
    module = _load_evaluator()
    assert module.MODEL_ORDER[-1] == "eegconformer"
    metrics, matrix = module.score(
        np.asarray([0, 1]),
        np.asarray([[0.9, 0.1], [0.2, 0.8]]),
        classes=2,
    )
    assert metrics["accuracy"] == 1.0
    assert matrix.tolist() == [[1, 0], [0, 1]]
