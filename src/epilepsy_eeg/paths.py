from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MODEL_ORDER_TASK1 = ("eegwavenet_sc", "physio_sequence_no_aux", "physio_sequence_stats")
MODEL_ORDER_TASK2 = ("eegwavenet_sc", "physio_tf", "physio_stats", "eegnetv4", "eegconformer")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default, allow_nan=False), encoding="utf-8")
    return target


def read_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("读取 YAML 配置需要 PyYAML") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"配置文件必须是 mapping: {path}")
    return value
