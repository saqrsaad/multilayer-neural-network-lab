"""Application entry point for Neural Network Lab.

Run with ``python -m neural_network_lab``. The startup implementation
lives in :mod:`neural_network_lab.main` so the console script declared
in ``pyproject.toml`` and ``python -m`` share one code path.
"""

from __future__ import annotations

from neural_network_lab.main import main

if __name__ == "__main__":
    raise SystemExit(main())
