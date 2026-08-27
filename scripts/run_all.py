#!/usr/bin/env python
"""Convenience launcher for the two complete research workflows.

Examples::

    python scripts/run_all.py task1 --input data/raw --labels data/labels.csv \
        --output runs/task1_retrain --config configs/annotation_detection_final.yaml
    python scripts/run_all.py pg36 --taxonomy-root results/task2_morphology_pg36 \
        --output runs/pg36_retrain --skip-training

The implementation lives in :mod:`epilepsy_eeg.cli`; this file exists so a
paper-style repository also has one obvious script entry point without
duplicating orchestration logic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from epilepsy_eeg.cli import main as cli_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", choices=("task1", "pg36"), help="完整任务流水线")
    args, remainder = parser.parse_known_args(argv)
    command = "run-task1" if args.workflow == "task1" else "run-pg36"
    return int(cli_main([command, *remainder]))


if __name__ == "__main__":
    raise SystemExit(main())
