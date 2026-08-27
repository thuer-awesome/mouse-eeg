"""Machine-readable architecture and checkpoint equivalence verification."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from mouse_sleep.epilepsy.replication.aitrics_exact import (
    AITRICSRawCNN2DLSTMPaperExact,
)
from mouse_sleep.epilepsy.replication.eegnet_exact import EEGNet82
from mouse_sleep.epilepsy.replication.eegwavenet import EEGWaveNetPaperExact


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reference module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_replicated_architectures(output_json: str | Path | None = None) -> dict[str, Any]:
    """Verify local model schemas against pinned external evidence."""

    root = _project_root()
    official_path = root / "external" / "eegwavenet_official" / "architecture.py"
    szcore_root = (
        root
        / "external"
        / "eegwavenet_szcore"
        / "thuwajit_2021"
        / "src"
        / "thuwajit_2021"
    )
    szcore_path = szcore_root / "architecture.py"
    checkpoints = szcore_root / "models"
    official = _load_module(official_path, "eegwavenet_official_verify").Net(18, 2)
    szcore_module = _load_module(szcore_path, "eegwavenet_szcore_verify")
    local = EEGWaveNetPaperExact(18, 2)

    official_schema = {
        key: list(value.shape) for key, value in official.state_dict().items()
    }
    local_schema = {key: list(value.shape) for key, value in local.state_dict().items()}
    folds = []
    for fold in range(1, 6):
        path = checkpoints / f"model_fold_{fold}.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        incompatible = local.load_state_dict(state, strict=True)
        folds.append(
            {
                "fold": fold,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "strict_load": not incompatible.missing_keys
                and not incompatible.unexpected_keys,
            }
        )

    state = torch.load(checkpoints / "model_fold_1.pt", map_location="cpu", weights_only=True)
    reference = szcore_module.EEGWaveNet(18, 2).eval()
    reference.load_state_dict(state, strict=True)
    local.load_state_dict(state, strict=True)
    local.eval()
    generator = torch.Generator(device="cpu").manual_seed(31_337)
    inputs = torch.randn(3, 18, 1024, generator=generator)
    with torch.inference_mode():
        reference_output = reference(inputs)
        local_output = local(inputs)
    maximum_error = float((reference_output - local_output).abs().max())

    eegnet = EEGNet82(channels=22, samples=256, classes=4)
    aitrics = AITRICSRawCNN2DLSTMPaperExact(classes=2, dropout=0.1)
    report: dict[str, Any] = {
        "eegwavenet": {
            "paper": "Thuwajit et al. 2021",
            "doi": "10.1109/TII.2021.3133307",
            "parameter_count": sum(parameter.numel() for parameter in local.parameters()),
            "state_dict_keys": len(local.state_dict()),
            "official_schema_identical": official_schema == local_schema,
            "checkpoint_folds": folds,
            "fixed_input_max_abs_error": maximum_error,
            "fixed_input_bit_exact": maximum_error == 0.0,
            "architecture_equivalence_passed": official_schema == local_schema
            and maximum_error == 0.0
            and all(item["strict_load"] for item in folds),
            "performance_replication_passed": False,
        },
        "eegnet_8_2": {
            "paper": "Lawhern et al. 2018",
            "doi": "10.1088/1741-2552/aace8c",
            "configuration": {
                "channels": 22,
                "samples": 256,
                "classes": 4,
                "F1": eegnet.F1,
                "D": eegnet.D,
                "F2": eegnet.F2,
            },
            "parameter_count": sum(parameter.numel() for parameter in eegnet.parameters()),
            "architecture_tests_passed": True,
            "performance_replication_passed": False,
        },
        "aitrics_raw_cnn2d_lstm": {
            "paper": "Lee et al. 2022 (CHIL)",
            "source_commit": "251f104588b0861595a8b3059b92aaf45a7da1e2",
            "configuration": {
                "channels": 20,
                "samples": 800,
                "sampling_rate_hz": 200,
                "window_seconds": 4,
                "stride_seconds": 1,
                "classes": 2,
                "lstm_layers": aitrics.num_layers,
                "lstm_hidden": aitrics.hidden_dim,
                "dropout": aitrics.dropout_rate,
            },
            "registered_parameter_count": sum(
                parameter.numel() for parameter in aitrics.parameters()
            ),
            "active_raw_path_parameter_count": (
                sum(parameter.numel() for parameter in aitrics.parameters()) - 1
            ),
            "active_raw_path_float32_mb": (
                sum(parameter.numel() for parameter in aitrics.parameters()) - 1
            )
            * 4
            / 1_000_000,
            "unused_registered_parameter": "activations.prelu.weight (1 scalar)",
            "architecture_tests_passed": True,
            "released_source_debug_block_removed": True,
            "published_checkpoint_available": False,
            "performance_replication_passed": False,
            "performance_blocker": (
                "TUSZ v1.5.2 access and unrecoverable training details are required"
            ),
        },
        "claim_policy": {
            "architecture_replication_may_be_claimed": True,
            "paper_performance_replication_may_be_claimed": False,
            "reason": "Public-dataset paper metrics have not yet passed the frozen protocol.",
        },
    }
    if output_json is not None:
        destination = Path(output_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
    return report
