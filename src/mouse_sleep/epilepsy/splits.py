from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def audit_animal_level_split(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate that recording segments cannot cross animal-level splits.

    A ``session_id`` identifies one exported recording segment, while
    ``animal_id`` identifies the biological subject.  An animal may therefore
    own several sessions, but all of them must receive the same split.
    """

    required = {"animal_id", "session_id", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Split manifest is missing required columns: {missing}")

    checked = frame.loc[:, sorted(required)].copy()
    for column in required:
        checked[column] = checked[column].astype("string").str.strip()
        if checked[column].isna().any() or checked[column].eq("").any():
            raise ValueError(f"Split manifest contains an empty {column}")

    unknown_splits = sorted(set(checked["split"]) - {"train", "val", "test"})
    if unknown_splits:
        raise ValueError(f"Split manifest contains unknown split values: {unknown_splits}")

    session_animals = checked.groupby("session_id")["animal_id"].nunique()
    ambiguous_sessions = sorted(session_animals.loc[session_animals.ne(1)].index.tolist())
    if ambiguous_sessions:
        raise ValueError(
            "A recording segment maps to more than one animal: "
            f"{ambiguous_sessions}"
        )

    animal_splits = checked.groupby("animal_id")["split"].nunique()
    leaked_animals = sorted(animal_splits.loc[animal_splits.ne(1)].index.tolist())
    if leaked_animals:
        raise ValueError(
            "Animal leakage detected; recording segments cross splits for: "
            f"{leaked_animals}"
        )

    sessions_by_animal = (
        checked.drop_duplicates(["animal_id", "session_id"])
        .groupby("animal_id")["session_id"]
        .apply(lambda values: sorted(values.astype(str).tolist()))
        .to_dict()
    )
    animals_by_split = {
        split: sorted(
            checked.loc[checked["split"].eq(split), "animal_id"].unique().tolist()
        )
        for split in ("train", "val", "test")
    }
    return {
        "n_animals": int(checked["animal_id"].nunique()),
        "n_recording_segments": int(checked["session_id"].nunique()),
        "animals_by_split": animals_by_split,
        "recording_segments_by_animal": sessions_by_animal,
    }


def build_loso_splits(cache_manifest: str | Path | pd.DataFrame, output_dir: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(cache_manifest) if not isinstance(cache_manifest, pd.DataFrame) else cache_manifest.copy()
    frame["animal_id"] = frame["animal_id"].astype(str)
    animals = sorted(frame["animal_id"].unique())
    rows = []
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    for fold, test_animal in enumerate(animals, 1):
        others = [a for a in animals if a != test_animal]
        val_animal = others[(fold - 1) % len(others)] if others else test_animal
        assigned = frame.copy()
        assigned["fold"] = fold
        assigned["split"] = "train"
        assigned.loc[assigned.animal_id.eq(val_animal), "split"] = "val"
        assigned.loc[assigned.animal_id.eq(test_animal), "split"] = "test"
        audit_animal_level_split(assigned)
        path = output_dir / f"loso_fold{fold}_{test_animal}.csv"
        assigned.to_csv(path, index=False, encoding="utf-8-sig")
        rows.append({"fold": fold, "test_animal": test_animal, "val_animal": val_animal, "manifest": str(path)})
    index = pd.DataFrame(rows); index.to_csv(output_dir / "fold_index.csv", index=False, encoding="utf-8-sig"); return index
