#!/usr/bin/env python
"""Bridge an audited continuous protocol into the PG36 morphology tables.

This stage is intentionally small and explicit.  It does not assign PG labels
or train a classifier; it only extracts independent event x channel features
and writes the window table consumed by the pooled clustering/freeze stages.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from epilepsy_eeg.morphology_adapter import build_pg36_input_tables  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--exclude-animal-channel",
        action="append",
        default=None,
        metavar="ANIMAL:CHANNEL",
    )
    parser.add_argument("--baseline-samples-per-stream", type=int, default=256)
    args = parser.parse_args(argv)
    excluded: list[tuple[str, str]] = []
    excluded_values = args.exclude_animal_channel or ["E07:CH7", "E77:CH7"]
    for item in excluded_values:
        parts = item.replace("/", ":").split(":")
        if len(parts) != 2:
            raise ValueError(f"invalid --exclude-animal-channel: {item}")
        excluded.append((parts[0].upper(), parts[1].upper()))
    manifest = build_pg36_input_tables(
        args.protocol_dir,
        args.output_root,
        excluded_animal_channels=excluded,
        baseline_samples_per_stream=args.baseline_samples_per_stream,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
