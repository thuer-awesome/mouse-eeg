"""Allow the research CLI to be run with ``python -m epilepsy_eeg``.

The console script installed by ``pip install -e .`` remains the preferred
entry point.  Keeping this module in the source distribution also makes a
fresh clone easy to smoke-test (with ``PYTHONPATH=src``) before installing an
editable environment; it dispatches to the same CLI and never loads a frozen
checkpoint by itself.
"""

from __future__ import annotations

from .cli import main


if __name__ == "__main__":  # pragma: no cover - exercised by the interpreter
    raise SystemExit(main())
