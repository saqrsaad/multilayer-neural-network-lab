"""GUI tests for the main window and navigation (offscreen)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QWidget

from neural_network_lab.gui.main_window import create_application
from neural_network_lab.gui.main_window import create_main_window
from neural_network_lab.gui.main_window import MainWindow
from neural_network_lab.gui.main_window import PlaceholderView
from neural_network_lab.gui.main_window import VIEW_SPECS


def _window() -> MainWindow:
    create_application()
    return MainWindow()


def test_navigation_covers_documented_sections():
    window = _window()
    assert [spec.key for spec in VIEW_SPECS] == [
        "dashboard",
        "datasets",
        "network",
        "training",
        "testing",
        "parameters",
        "experiments",
    ]
    assert window._navigation.count() == len(VIEW_SPECS)
    assert window.current_view_key() == "dashboard"


def test_unimplemented_views_show_placeholders_without_crashing():
    window = _window()
    for key in ("dashboard", "network", "training", "testing", "parameters", "experiments"):
        assert window.request_view(key) is True
        assert isinstance(window._pages[key], PlaceholderView)
    assert window.current_view_key() == "experiments"


def test_dataset_view_registers_as_integration_point():
    window = _window()
    assert window._pages.get("datasets") is None
    window.register_default_views()
    assert window.request_view("datasets") is True
    assert type(window._pages["datasets"]).__name__ == "DatasetView"


def test_unknown_view_key_never_crashes():
    window = _window()
    assert window.request_view("no-such-view") is False
    assert window.current_view_key() == "dashboard"


def test_failing_factory_shows_error_placeholder():
    window = _window()

    def broken() -> QWidget:
        raise RuntimeError("boom")

    window.register_view("broken", "Broken", broken)
    assert window.request_view("broken") is True
    assert isinstance(window._pages["broken"], PlaceholderView)


class DirtyView(QWidget):
    """Test stand-in for a view carrying unsaved edits."""

    def __init__(self, dirty: bool = True) -> None:
        """Create the stand-in with a fixed dirty flag."""
        super().__init__()
        self._dirty = dirty

    def has_unsaved_changes(self) -> bool:
        """Report the fixed dirty flag."""
        return self._dirty


def test_unsaved_changes_guard_blocks_and_allows(monkeypatch):
    window = _window()
    window.register_view("first", "First", DirtyView)
    window.register_view("second", "Second", QWidget)
    assert window.request_view("first") is True
    monkeypatch.setattr(
        "neural_network_lab.gui.main_window.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.Cancel,
    )
    assert window.request_view("second") is False
    assert window.current_view_key() == "first"
    monkeypatch.setattr(
        "neural_network_lab.gui.main_window.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.Discard,
    )
    assert window.request_view("second") is True
    assert window.current_view_key() == "second"


def test_create_application_reuses_instance():
    assert create_application() is create_application()


def test_create_main_window_starts_predictably():
    window = create_main_window()
    assert window.windowTitle() == "Neural Network Lab"
    assert window._stack.count() >= 1
