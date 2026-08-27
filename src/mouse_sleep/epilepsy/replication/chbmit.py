"""Auditable CHB-MIT metadata parsing and patient-level split preparation.

This module deliberately stops before reading EDF samples.  It freezes the
source file list, seizure annotations, channel metadata, and patient groups so
that a later EDF window adapter cannot accidentally split a patient across
training and evaluation.  The EEGWaveNet paper's exact CHB-MIT protocol is
not public in the released code, so generated protocols are labelled as an
independent benchmark rather than paper-exact replication.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

CHBMIT_BASE_URL = "https://physionet.org/files/chbmit/1.0.0"
CHBMIT_SAMPLING_RATE = 256
CHBMIT_EPOCH_SAMPLES = 1024

# The first 18 standard bipolar channels are shared by the canonical CHB-MIT
# montages often used by seizure benchmarks.  EDF files can contain extra or
# missing channels; the later EDF adapter must verify this list per recording.
DEFAULT_BIPOLAR_CHANNELS_18 = (
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
)

# The database contains two monitoring cases from the same child.  Keeping
# them in one group is necessary for an honest subject-independent estimate.
PATIENT_GROUP_OVERRIDES = {"chb01": "patient_chb01_chb21", "chb21": "patient_chb01_chb21"}

_FILE_RE = re.compile(r"^File Name:\s*(?P<name>\S+\.edf)\s*$", re.I)
_RATE_RE = re.compile(r"^Data Sampling Rate:\s*(?P<rate>[0-9]+(?:\.[0-9]+)?)\s*Hz", re.I)
_CHANNEL_RE = re.compile(r"^Channel\s+(?P<index>[0-9]+):\s*(?P<name>.*)\s*$", re.I)
_N_SEIZURES_RE = re.compile(
    r"^Number of Seizures in File:\s*(?P<count>[0-9]+)\s*$", re.I
)
_SEIZURE_START_RE = re.compile(
    r"^Seizure(?:\s+[0-9]+)?\s+Start Time:\s*(?P<seconds>[0-9]+(?:\.[0-9]+)?)\s*seconds?\s*$",
    re.I,
)
_SEIZURE_END_RE = re.compile(
    r"^Seizure(?:\s+[0-9]+)?\s+End Time:\s*(?P<seconds>[0-9]+(?:\.[0-9]+)?)\s*seconds?\s*$",
    re.I,
)


def patient_id_for_case(case_id: str) -> str:
    """Return the leakage-safe patient group for a CHB-MIT case directory."""

    normalized = case_id.strip().lower()
    return PATIENT_GROUP_OVERRIDES.get(normalized, f"patient_{normalized}")


def parse_chbmit_summary(path: str | Path) -> dict[str, Any]:
    """Parse one PhysioNet ``*-summary.txt`` file without reading EDF data."""

    summary_path = Path(path)
    case_id = summary_path.stem.removesuffix("-summary").lower()
    lines = summary_path.read_text(encoding="utf-8", errors="replace").splitlines()
    sampling_rate: float | None = None
    channels: dict[int, str] = {}
    files: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    pending_starts: list[float] = []
    pending_ends: list[float] = []

    def flush_seizures() -> None:
        nonlocal pending_starts, pending_ends
        if current is None:
            pending_starts = []
            pending_ends = []
            return
        if len(pending_starts) != len(pending_ends):
            raise ValueError(
                f"Unpaired seizure boundaries in {summary_path} for {current['file_name']}"
            )
        current["seizures"] = [
            {"start_seconds": start, "end_seconds": end}
            for start, end in zip(pending_starts, pending_ends)
        ]
        current["n_seizures"] = len(current["seizures"])
        pending_starts = []
        pending_ends = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        rate_match = _RATE_RE.match(line)
        if rate_match:
            sampling_rate = float(rate_match.group("rate"))
            continue
        channel_match = _CHANNEL_RE.match(line)
        if channel_match:
            name = channel_match.group("name").strip()
            channels[int(channel_match.group("index"))] = name
            continue
        file_match = _FILE_RE.match(line)
        if file_match:
            if current is not None:
                flush_seizures()
                files.append(current)
            current = {
                "file_name": file_match.group("name"),
                "n_seizures_declared": None,
                "seizures": [],
                "n_seizures": 0,
            }
            continue
        count_match = _N_SEIZURES_RE.match(line)
        if count_match and current is not None:
            current["n_seizures_declared"] = int(count_match.group("count"))
            continue
        start_match = _SEIZURE_START_RE.match(line)
        if start_match and current is not None:
            pending_starts.append(float(start_match.group("seconds")))
            continue
        end_match = _SEIZURE_END_RE.match(line)
        if end_match and current is not None:
            pending_ends.append(float(end_match.group("seconds")))
            continue

    if current is not None:
        flush_seizures()
        files.append(current)
    if sampling_rate is None:
        raise ValueError(f"No sampling rate found in {summary_path}")

    for item in files:
        declared = item["n_seizures_declared"]
        if declared is not None and declared != item["n_seizures"]:
            raise ValueError(
                f"Seizure count mismatch in {summary_path} for {item['file_name']}: "
                f"declared {declared}, parsed {item['n_seizures']}"
            )
        item["case_id"] = case_id

    return {
        "case_id": case_id,
        "patient_id": patient_id_for_case(case_id),
        "sampling_rate_hz": sampling_rate,
        "channels": channels,
        "files": files,
        "summary_path": str(summary_path.resolve()),
    }


def parse_sha256sums(path: str | Path) -> dict[str, str]:
    """Read a PhysioNet SHA256SUMS file into ``relative_path -> digest``."""

    result: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            result[parts[1].lstrip("* ")] = parts[0].lower()
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_chbmit_manifest(
    metadata_root: str | Path,
    output_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    records_path: str | Path | None = None,
    sha256_path: str | Path | None = None,
    verify_checksums: bool = False,
) -> pd.DataFrame:
    """Freeze all RECORDS files and annotation metadata into a CSV manifest.

    ``dataset_root`` is optional because metadata can be prepared before the
    multi-gigabyte EDF download.  ``source_exists`` and ``sha256_observed``
    therefore remain explicit rather than silently implying local data.
    """

    root = Path(metadata_root).resolve()
    records_file = Path(records_path) if records_path else root / "RECORDS"
    if not records_file.exists():
        raise FileNotFoundError(f"CHB-MIT RECORDS file not found: {records_file}")
    sha_map = parse_sha256sums(sha256_path) if sha256_path else {}
    data_root = Path(dataset_root).resolve() if dataset_root else None
    summary_by_case: dict[str, dict[str, Any]] = {}
    for summary_path in sorted(root.glob("chb*/chb*-summary.txt")):
        parsed = parse_chbmit_summary(summary_path)
        summary_by_case[parsed["case_id"]] = parsed

    rows: list[dict[str, Any]] = []
    for raw_relative in records_file.read_text(encoding="utf-8").splitlines():
        relative = raw_relative.strip().replace("\\", "/")
        if not relative or relative.startswith("#"):
            continue
        if not relative.lower().endswith(".edf"):
            continue
        case_id = relative.split("/", 1)[0].lower()
        file_name = relative.rsplit("/", 1)[-1]
        parsed = summary_by_case.get(case_id)
        if parsed is None:
            raise ValueError(f"No summary metadata found for {case_id}")
        file_info = next(
            (item for item in parsed["files"] if item["file_name"].lower() == file_name.lower()),
            None,
        )
        seizures = file_info["seizures"] if file_info else []
        local_path = data_root / Path(relative) if data_root else None
        expected_sha = sha_map.get(relative)
        observed_sha = (
            _sha256(local_path)
            if verify_checksums and local_path and local_path.is_file()
            else None
        )
        rows.append(
            {
                "relative_path": relative,
                "source_url": f"{CHBMIT_BASE_URL}/{relative}",
                "case_id": case_id,
                "patient_id": parsed["patient_id"],
                "sampling_rate_hz": parsed["sampling_rate_hz"],
                "n_seizures": int(len(seizures)),
                "seizure_intervals_json": json.dumps(seizures, sort_keys=True),
                "channel_names_json": json.dumps(
                    list(parsed["channels"].values()), ensure_ascii=False
                ),
                "expected_sha256": expected_sha,
                "observed_sha256": observed_sha,
                "source_exists": bool(local_path and local_path.is_file()),
                "label_has_seizure": int(bool(seizures)),
            }
        )
    if not rows:
        raise ValueError("No EDF records were found in RECORDS")
    frame = pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8-sig")
    return frame


def build_chbmit_patient_folds(
    manifest: str | Path | pd.DataFrame,
    output_dir: str | Path,
    *,
    validation_mode: str = "cyclic_patient",
) -> pd.DataFrame:
    """Create deterministic patient-disjoint folds for EDF-level evaluation.

    Each fold holds out one patient for test and the next patient (cyclically)
    for validation.  The remaining patient groups are training.  This is a
    transparent independent benchmark protocol, not a claim about the closed
    EEGWaveNet paper's undisclosed split.
    """

    if validation_mode != "cyclic_patient":
        raise ValueError("validation_mode must be 'cyclic_patient'")
    frame = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
    required = {"patient_id", "case_id", "relative_path", "label_has_seizure"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CHB-MIT manifest missing columns: {', '.join(sorted(missing))}")
    patients = sorted(frame["patient_id"].dropna().astype(str).unique())
    if len(patients) < 3:
        raise ValueError("At least three patient groups are required for train/val/test")

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for fold, test_patient in enumerate(patients, start=1):
        val_patient = patients[fold % len(patients)]
        split = pd.Series("train", index=frame.index, dtype="object")
        split.loc[frame["patient_id"].eq(val_patient)] = "val"
        split.loc[frame["patient_id"].eq(test_patient)] = "test"
        if (val_patient == test_patient) or split.eq("train").sum() == 0:
            raise AssertionError("invalid cyclic patient split")
        fold_frame = frame.copy()
        fold_frame["split"] = split.to_numpy()
        fold_frame["fold"] = fold
        path = destination / f"fold_{fold:02d}.csv"
        fold_frame.to_csv(path, index=False, encoding="utf-8-sig")
        rows.append(
            {
                "fold": fold,
                "manifest": path.name,
                "test_patient": test_patient,
                "validation_patient": val_patient,
                "n_train_files": int((split == "train").sum()),
                "n_val_files": int((split == "val").sum()),
                "n_test_files": int((split == "test").sum()),
                "patient_level_split": True,
                "same_patient_train_test": False,
            }
        )
    index = pd.DataFrame(rows)
    index.to_csv(destination / "fold_index.csv", index=False, encoding="utf-8-sig")
    (destination / "split_metadata.json").write_text(
        json.dumps(
            {
                "protocol": "cyclic_patient_holdout",
                "validation_mode": validation_mode,
                "n_folds": len(patients),
                "patient_groups": patients,
                "patient_level_split": True,
                "paper_performance_claim_allowed": False,
                "status": "independent_chbmit_benchmark_protocol",
                "note": "The EEGWaveNet paper's original CHB-MIT patient list and split are not public in the released code.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return index


def chbmit_manifest_summary(manifest: str | Path | pd.DataFrame) -> dict[str, Any]:
    """Return compact counts suitable for audit logs and CLI output."""

    frame = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest
    return {
        "files": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()),
        "cases": int(frame["case_id"].nunique()),
        "files_with_seizure": int(frame["label_has_seizure"].sum()),
        "seizure_events": int(frame["n_seizures"].sum()),
        "sampling_rates_hz": sorted(frame["sampling_rate_hz"].dropna().unique().tolist()),
        "source_exists": int(frame["source_exists"].sum()) if "source_exists" in frame else None,
        "manifest_sha256": hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest(),
    }


__all__ = [
    "CHBMIT_BASE_URL",
    "CHBMIT_EPOCH_SAMPLES",
    "CHBMIT_SAMPLING_RATE",
    "DEFAULT_BIPOLAR_CHANNELS_18",
    "build_chbmit_manifest",
    "build_chbmit_patient_folds",
    "chbmit_manifest_summary",
    "parse_chbmit_summary",
    "parse_sha256sums",
    "patient_id_for_case",
]
