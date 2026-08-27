"""Reproducible loader and split builder for the Bonn EEG corpus.

The Bonn recordings are already band-limited by the data provider (0.53-40 Hz)
and contain one 4097-sample channel per file.  This module deliberately keeps
the source sampling and amplitude values unchanged.  Any resampling or
normalization is an explicit training option and is written to the run record.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


BONN_SAMPLE_RATE_HZ = 173.61
BONN_SAMPLES = 4097
BONN_SET_CODES = ("Z", "O", "N", "F", "S")

# The task names are intentionally explicit.  In particular, ``ictal_vs_all``
# is not interchangeable with a healthy-vs-epilepsy task.
TASK_LABELS: dict[str, dict[str, int]] = {
    "five_class": {"Z": 0, "O": 1, "N": 2, "F": 3, "S": 4},
    "ictal_vs_all": {"Z": 0, "O": 0, "N": 0, "F": 0, "S": 1},
    "ictal_vs_interictal": {"N": 0, "F": 0, "S": 1},
    "healthy_vs_epilepsy": {"Z": 0, "O": 0, "N": 1, "F": 1, "S": 1},
}


@dataclass(frozen=True)
class BonnRecord:
    """One source recording and its provenance."""

    path: Path
    relative_path: str
    set_code: str
    recording_id: str


def resolve_bonn_root(root: str | Path) -> Path:
    """Find the directory containing all five ``Set_*`` folders."""

    candidate = Path(root).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    if all((candidate / f"Set_{code}").is_dir() for code in BONN_SET_CODES):
        return candidate
    matches = []
    for z_dir in candidate.rglob("Set_Z"):
        parent = z_dir.parent
        if all((parent / f"Set_{code}").is_dir() for code in BONN_SET_CODES):
            matches.append(parent)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one Bonn root with Set_Z/Set_O/Set_N/Set_F/Set_S; found {matches}"
        )
    return matches[0]


def discover_bonn_records(root: str | Path, *, validate: bool = True) -> list[BonnRecord]:
    """Discover and optionally validate all Bonn text recordings."""

    resolved = resolve_bonn_root(root)
    records: list[BonnRecord] = []
    pattern = re.compile(r"^(?P<code>[ZONFS])(?P<number>\d{3})\.txt$", re.IGNORECASE)
    for code in BONN_SET_CODES:
        folder = resolved / f"Set_{code}"
        files = sorted(folder.glob("*.txt"))
        if len(files) != 100:
            raise ValueError(f"Expected 100 files in {folder}, found {len(files)}")
        for path in files:
            match = pattern.match(path.name)
            if match is None or match.group("code").upper() != code:
                raise ValueError(f"Unexpected Bonn filename: {path}")
            if validate:
                values = np.loadtxt(path, dtype=np.float32)
                if values.ndim != 1 or values.size != BONN_SAMPLES:
                    raise ValueError(f"Expected {BONN_SAMPLES} values in {path}")
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite values in {path}")
            records.append(
                BonnRecord(
                    path=path,
                    relative_path=path.relative_to(resolved).as_posix(),
                    set_code=code,
                    recording_id=path.stem.upper(),
                )
            )
    return records


def _sha256(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def build_dataset_manifest(
    root: str | Path,
    output_csv: str | Path,
    *,
    task: str = "ictal_vs_all",
    include_hashes: bool = True,
) -> Path:
    """Write a source manifest with labels and immutable file hashes."""

    if task not in TASK_LABELS:
        raise ValueError(f"Unknown Bonn task {task!r}; choose from {sorted(TASK_LABELS)}")
    resolved = resolve_bonn_root(root)
    records = discover_bonn_records(resolved, validate=True)
    mapping = TASK_LABELS[task]
    rows = []
    for record in records:
        if record.set_code not in mapping:
            continue
        row = {
            "path": record.relative_path,
            "recording_id": record.recording_id,
            "set_code": record.set_code,
            "label": mapping[record.set_code],
            "sample_rate_hz": BONN_SAMPLE_RATE_HZ,
            "n_samples": BONN_SAMPLES,
        }
        if include_hashes:
            row["sha256"] = _sha256(record.path)
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    metadata = {
        "dataset": "Bonn EEG",
        "dataset_root": str(resolved),
        "task": task,
        "label_mapping": mapping,
        "sample_rate_hz": BONN_SAMPLE_RATE_HZ,
        "n_samples": BONN_SAMPLES,
        "n_records": int(len(frame)),
    }
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return destination


def build_grouped_stratified_folds(
    manifest: str | Path | pd.DataFrame,
    output_dir: str | Path,
    *,
    n_splits: int = 5,
    seed: int = 20260804,
    validation_fraction: float | None = 0.2,
) -> pd.DataFrame:
    """Create deterministic outer test folds and an inner validation fold.

    Bonn does not publish patient identifiers.  We therefore group by the
    recording ID (one complete 23.6-second record) and state that limitation in
    the generated index rather than pretending to provide patient-level CV.
    """

    frame = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
    required = {"path", "recording_id", "set_code", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if frame["recording_id"].duplicated().any():
        raise ValueError("Each Bonn recording_id must occur exactly once")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if validation_fraction is not None and not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")

    y = frame["label"].to_numpy()
    groups = frame["recording_id"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []

    for fold, (train_val_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(frame)), y, groups), start=1
    ):
        split = np.full(len(frame), "test", dtype=object)
        split[test_idx] = "test"
        if validation_fraction is None:
            split[train_val_idx] = "train"
        else:
            # Pick the closest available number of inner groups while retaining
            # a stratified, group-disjoint validation set.
            inner_splits = max(2, round(1.0 / validation_fraction))
            inner_splits = min(inner_splits, len(train_val_idx))
            inner = StratifiedGroupKFold(
                n_splits=inner_splits, shuffle=True, random_state=seed + fold
            )
            relative_train, relative_val = next(
                inner.split(
                    np.zeros(len(train_val_idx)),
                    y[train_val_idx],
                    groups[train_val_idx],
                )
            )
            split[train_val_idx[relative_train]] = "train"
            split[train_val_idx[relative_val]] = "val"

        fold_frame = frame.copy()
        fold_frame["split"] = split
        fold_frame["fold"] = fold
        destination = output / f"fold_{fold:02d}.csv"
        fold_frame.to_csv(destination, index=False, encoding="utf-8")
        index_rows.append(
            {
                "fold": fold,
                "manifest": destination.name,
                "n_train": int((split == "train").sum()),
                "n_val": int((split == "val").sum()),
                "n_test": int((split == "test").sum()),
                "seed": seed,
                "group_column": "recording_id",
                "patient_level_cv_available": False,
            }
        )

    index = pd.DataFrame(index_rows)
    index_path = output / "fold_index.csv"
    index.to_csv(index_path, index=False, encoding="utf-8")
    (output / "split_metadata.json").write_text(
        json.dumps(
            {
                "protocol": "stratified_group_kfold",
                "n_splits": n_splits,
                "seed": seed,
                "group_column": "recording_id",
                "patient_level_cv_available": False,
                "note": "Bonn source files do not expose patient IDs.",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return index


def load_bonn_split(
    split_manifest: str | Path | pd.DataFrame,
    dataset_root: str | Path,
    split: str,
    *,
    normalization: str = "none",
    train_stats: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, float]]:
    """Load one split as ``[N, 1, 4097]`` float32 tensors plus metadata."""

    frame = pd.read_csv(split_manifest) if not isinstance(split_manifest, pd.DataFrame) else split_manifest.copy()
    if "split" in frame.columns:
        frame = frame.loc[frame["split"].eq(split)].copy()
    if frame.empty:
        raise ValueError(f"No records in split {split!r}")
    root = resolve_bonn_root(dataset_root)
    signals = []
    for relative in frame["path"].astype(str):
        path = root / relative
        values = np.loadtxt(path, dtype=np.float32)
        if values.shape != (BONN_SAMPLES,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid Bonn recording: {path}")
        signals.append(values)
    x = np.stack(signals, axis=0)[:, None, :].astype(np.float32, copy=False)
    stats: dict[str, float] = {}
    key = normalization.lower()
    if key == "none":
        pass
    elif key == "per_record_zscore":
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        x = (x - mean) / np.maximum(std, 1e-6)
    elif key == "global_train_zscore":
        if train_stats is None:
            raise ValueError("train_stats is required for global_train_zscore")
        mean = float(train_stats["mean"])
        std = max(float(train_stats["std"]), 1e-6)
        x = (x - mean) / std
        stats = {"mean": mean, "std": std}
    else:
        raise ValueError(
            "normalization must be none, per_record_zscore, or global_train_zscore"
        )
    return x, frame["label"].to_numpy(dtype=np.int64), frame, stats


def compute_train_stats(
    split_manifest: str | Path | pd.DataFrame, dataset_root: str | Path
) -> dict[str, float]:
    """Compute global normalization statistics from the train records only."""

    x, _, _, _ = load_bonn_split(
        split_manifest, dataset_root, "train", normalization="none"
    )
    return {"mean": float(x.mean()), "std": float(x.std())}


def manifest_summary(manifest: str | Path | pd.DataFrame) -> dict[str, object]:
    """Return a compact, JSON-friendly source manifest summary."""

    frame = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest
    return {
        "records": int(len(frame)),
        "classes": {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()},
        "sets": {str(k): int(v) for k, v in frame["set_code"].value_counts().sort_index().items()},
        "sha256_manifest": hashlib.sha256(
            frame.to_csv(index=False).encode("utf-8")
        ).hexdigest(),
    }

