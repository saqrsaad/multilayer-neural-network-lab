"""Application startup for Neural Network Lab.

This module owns :func:`main`, the entry point declared in
``pyproject.toml`` (``neural-network-lab = "neural_network_lab.main:main"``).
Startup creates (or reuses) the shared QApplication, shows the main
window, and returns the event loop exit code; shutdown follows the
standard Qt path with no background threads involved.
"""

from __future__ import annotations

import sys

from neural_network_lab.gui.main_window import create_application
from neural_network_lab.gui.main_window import create_main_window


def main(argv: list[str] | None = None) -> int:
    """Launch the application and run the event loop."""
    app = create_application(sys.argv if argv is None else argv)
    window = create_main_window()
    window.register_default_views()
    window.show()
    return int(app.exec())
