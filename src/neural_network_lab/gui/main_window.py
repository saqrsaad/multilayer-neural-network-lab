"""Main desktop window and navigation for Neural Network Lab.

The window provides a navigation list (one entry per section from
``docs/PROJECT_SPEC.md`` §12) beside a stacked content area. Only views
whose tasks are implemented are registered with real widgets; every
other section shows a clearly labeled placeholder instead of crashing.
Later tasks integrate new views through :meth:`MainWindow.register_view`
without touching navigation logic.

No neural-network mathematics lives here: this module only arranges
widgets, routes navigation, presents errors, and guards unsaved edits.
Views carrying edits expose ``has_unsaved_changes()``; navigation and
window close request confirmation before discarding them.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QListWidget
from PySide6.QtWidgets import QListWidgetItem
from PySide6.QtWidgets import QMainWindow
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QStackedWidget
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget

#: Consistent content margins (pixels) for GUI pages.
CONTENT_MARGIN = 12

#: Consistent spacing (pixels) between stacked GUI elements.
CONTENT_SPACING = 8

#: Fixed width (pixels) of the navigation list.
NAVIGATION_WIDTH = 180

#: Fallback text for sections whose real view is not implemented yet.
PLACEHOLDER_NOTE = "This view is not implemented yet."


@dataclass(frozen=True)
class ViewSpec:
    """Navigation entry: stable key plus user-facing title."""

    key: str
    title: str


#: Navigation order matching the documented GUI sections.
VIEW_SPECS: tuple[ViewSpec, ...] = (
    ViewSpec("dashboard", "Dashboard"),
    ViewSpec("datasets", "Dataset Management"),
    ViewSpec("network", "Network Configuration"),
    ViewSpec("training", "Training"),
    ViewSpec("testing", "Testing"),
    ViewSpec("parameters", "Weights and Biases"),
    ViewSpec("experiments", "Experiment History"),
)


class PlaceholderView(QWidget):
    """Safe stand-in for a feature view that is not implemented yet."""

    def __init__(self, title: str, note: str = PLACEHOLDER_NOTE) -> None:
        """Create a placeholder page with a title and note."""
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN
        )
        layout.setSpacing(CONTENT_SPACING)
        heading = QLabel(title)
        heading.setObjectName("placeholderTitle")
        layout.addWidget(heading)
        message = QLabel(note)
        message.setWordWrap(True)
        message.setObjectName("placeholderNote")
        layout.addWidget(message)
        layout.addStretch(1)


def create_application(argv: list[str] | None = None) -> QApplication:
    """Return the shared QApplication, creating it when necessary."""
    existing = QApplication.instance()
    if existing is not None:
        assert isinstance(existing, QApplication)
        return existing
    return QApplication(sys.argv if argv is None else argv)


def create_main_window() -> MainWindow:
    """Create the application main window with default views."""
    return MainWindow()


class MainWindow(QMainWindow):
    """Main window with navigation list and stacked content pages."""

    def __init__(self) -> None:
        """Create the window, navigation, and placeholder pages."""
        super().__init__()
        self.setWindowTitle("Neural Network Lab")
        self._factories: dict[str, Callable[[], QWidget]] = {}
        self._pages: dict[str, QWidget] = {}
        self._order: list[str] = []

        central = QWidget(self)
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(
            CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN
        )
        outer.setSpacing(CONTENT_SPACING)

        self._navigation = QListWidget(central)
        self._navigation.setFixedWidth(NAVIGATION_WIDTH)
        self._navigation.setObjectName("navigationList")
        outer.addWidget(self._navigation)

        self._stack = QStackedWidget(central)
        self._stack.setObjectName("contentStack")
        outer.addWidget(self._stack, 1)

        for spec in VIEW_SPECS:
            self._order.append(spec.key)
            item = QListWidgetItem(spec.title, self._navigation)
            item.setData(Qt.ItemDataRole.UserRole, spec.key)
            self._factories[spec.key] = self._placeholder_factory(spec.title)

        self._navigation.currentRowChanged.connect(self._on_navigation_row)
        self._navigation.setCurrentRow(0)

    @staticmethod
    def _placeholder_factory(title: str) -> Callable[[], QWidget]:
        """Build a factory producing a placeholder page."""

        def factory() -> QWidget:
            return PlaceholderView(title)

        return factory

    @staticmethod
    def _datasets_factory() -> QWidget:
        """Build the dataset editor view (lazy import avoids cycles)."""
        from neural_network_lab.gui.views.dataset_view import DatasetView

        return DatasetView()

    def _training_factory(self) -> QWidget:
        """Build the training view fed by the dataset editor selection."""
        from neural_network_lab.gui.views.training_view import TrainingView

        view = TrainingView()
        view.set_dataset_provider(self._datasets_current_dataset)
        return view

    def _testing_factory(self) -> QWidget:
        """Build the testing view fed by the training model and selection."""
        from neural_network_lab.gui.views.testing_view import TestingView

        view = TestingView()
        view.set_network_provider(self._training_current_network)
        view.set_dataset_provider(self._datasets_current_dataset)
        return view

    def _parameters_factory(self) -> QWidget:
        """Build the parameters view fed by the training model."""
        from neural_network_lab.gui.views.parameters_view import ParametersView

        view = ParametersView()
        view.set_network_provider(self._training_current_network)
        return view

    @staticmethod
    def _experiments_factory() -> QWidget:
        """Build the experiment history view (loads its own files)."""
        from neural_network_lab.gui.views.experiments_view import ExperimentsView

        return ExperimentsView()

    def _datasets_current_dataset(self):
        """Return the dataset editor's applied dataset, if any."""
        page = self._pages.get("datasets")
        getter = getattr(page, "current_dataset", None)
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception:
            return None

    def _training_current_network(self):
        """Return the training view's owned network, if any."""
        page = self._pages.get("training")
        getter = getattr(page, "current_network", None)
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception:
            return None

    def register_view(
        self, key: str, title: str, factory: Callable[[], QWidget]
    ) -> None:
        """Register (or replace) the factory for a navigation key.

        This is the integration point for feature views implemented by
        later tasks: registering a real view replaces its placeholder
        without changing navigation. Unknown keys extend the navigation
        list; re-registering a visible page rebuilds it on next visit.
        """
        if key not in self._factories:
            self._order.append(key)
            item = QListWidgetItem(title, self._navigation)
            item.setData(Qt.ItemDataRole.UserRole, key)
        self._factories[key] = factory
        existing = self._pages.pop(key, None)
        if existing is not None:
            self._stack.removeWidget(existing)
            existing.deleteLater()

    def register_default_views(self) -> None:
        """Register the real views implemented so far.

        Datasets, training, testing, parameters, and experiments are
        live; training consumes the dataset editor selection, testing
        and parameters consume the trained model plus the selection,
        and experiments loads its own saved files — all resolved
        lazily at click time.
        """
        self.register_view("datasets", "Dataset Management", self._datasets_factory)
        self.register_view("training", "Training", self._training_factory)
        self.register_view("testing", "Testing", self._testing_factory)
        self.register_view("parameters", "Weights and Biases", self._parameters_factory)
        self.register_view("experiments", "Experiment History", self._experiments_factory)

    def current_view_key(self) -> str | None:
        """Return the key of the currently visible page, if any."""
        row = self._navigation.currentRow()
        if 0 <= row < len(self._order):
            return self._order[row]
        return None

    def request_view(self, key: str) -> bool:
        """Navigate to a page, guarding unsaved edits.

        Returns True when the page is now visible. Returns False
        without changing anything when the key is unknown, the user
        cancels a discard confirmation, or navigation is otherwise
        refused — never raising for missing views.
        """
        if key not in self._factories:
            self.report_error(f"Unknown view: {key}")
            return False
        if not self._can_leave_current_view():
            return False
        page = self._pages.get(key)
        if page is None:
            try:
                page = self._factories[key]()
            except Exception as exc:
                page = PlaceholderView(
                    key, f"Could not load this view: {exc}"
                )
            self._pages[key] = page
            self._stack.addWidget(page)
        self._stack.setCurrentWidget(page)
        try:
            row = self._order.index(key)
        except ValueError:
            return True
        self._navigation.blockSignals(True)
        try:
            self._navigation.setCurrentRow(row)
        finally:
            self._navigation.blockSignals(False)
        return True

    def report_error(self, message: str, timeout_ms: int = 8000) -> None:
        """Present an error message in the status bar."""
        self.statusBar().showMessage(message, timeout_ms)

    def _current_page(self) -> QWidget | None:
        """Return the currently visible page widget, if any."""
        key = self.current_view_key()
        if key is None:
            return None
        return self._pages.get(key)

    def _can_leave_current_view(self) -> bool:
        """Check unsaved edits; ask before discarding them."""
        page = self._current_page()
        guard = getattr(page, "has_unsaved_changes", None)
        if callable(guard):
            try:
                dirty = bool(guard())
            except Exception:
                dirty = False
            if dirty:
                answer = QMessageBox.question(
                    self,
                    "Unsaved changes",
                    "The current view has unsaved edits. Discard them?",
                    QMessageBox.Discard | QMessageBox.Cancel,
                    QMessageBox.Cancel,
                )
                return answer == QMessageBox.Discard
        return True

    def _on_navigation_row(self, row: int) -> None:
        """Route navigation-list selection through the guarded request."""
        if 0 <= row < len(self._order):
            key = self._order[row]
            current = self._stack.currentWidget()
            if current is None or current is not self._pages.get(key):
                if not self.request_view(key):
                    current = self._stack.currentWidget()
                    for index, ordered_key in enumerate(self._order):
                        if self._pages.get(ordered_key) is current:
                            self._navigation.blockSignals(True)
                            try:
                                self._navigation.setCurrentRow(index)
                            finally:
                                self._navigation.blockSignals(False)
                            break

    def closeEvent(self, event: QCloseEvent) -> None:
        """Confirm unsaved edits and running training before closing."""
        if not self._can_leave_current_view():
            event.ignore()
            return
        busy = [
            page
            for page in self._pages.values()
            if callable(getattr(page, "is_training_active", None))
            and page.is_training_active()
        ]
        if busy:
            answer = QMessageBox.question(
                self,
                "Training in progress",
                "Training is still running. Quit and cancel training?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            for page in busy:
                cancel = getattr(page, "cancel_training", None)
                if callable(cancel):
                    try:
                        cancel()
                    except Exception:
                        pass
            for page in busy:
                wait = getattr(page, "wait_for_worker", None)
                if callable(wait):
                    try:
                        wait()
                    except Exception:
                        pass
        super().closeEvent(event)
