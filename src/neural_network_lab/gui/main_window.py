"""Main desktop window and navigation for Neural Network Lab.

The window shows a titled navigation panel (one entry per section from
``docs/PROJECT_SPEC.md`` §12) beside a resizable stacked content area;
the entry of the visible page stays highlighted. Only views whose
tasks are implemented are registered with real widgets; every other
section shows a clearly labeled placeholder instead of crashing. Later
tasks integrate new views through :meth:`MainWindow.register_view`
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
from PySide6.QtWidgets import QAbstractItemView
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QListWidget
from PySide6.QtWidgets import QListWidgetItem
from PySide6.QtWidgets import QMainWindow
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QSplitter
from PySide6.QtWidgets import QStackedWidget
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget

from neural_network_lab.gui.widgets.page_scaffold import make_selectable
from neural_network_lab.gui.widgets.page_scaffold import PageHeader
from neural_network_lab.gui.widgets.ui_theme import CONTENT_MARGIN
from neural_network_lab.gui.widgets.ui_theme import CONTENT_SPACING
from neural_network_lab.gui.widgets.ui_theme import DEFAULT_WINDOW_HEIGHT
from neural_network_lab.gui.widgets.ui_theme import DEFAULT_WINDOW_WIDTH
from neural_network_lab.gui.widgets.ui_theme import FIELD_SPACING
from neural_network_lab.gui.widgets.ui_theme import MIN_WINDOW_HEIGHT
from neural_network_lab.gui.widgets.ui_theme import MIN_WINDOW_WIDTH
from neural_network_lab.gui.widgets.ui_theme import NAVIGATION_MAX_WIDTH
from neural_network_lab.gui.widgets.ui_theme import NAVIGATION_MIN_WIDTH
from neural_network_lab.gui.widgets.ui_theme import apply_theme

#: Preferred width (pixels) of the navigation panel.
NAVIGATION_WIDTH = NAVIGATION_MIN_WIDTH

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
        header = PageHeader(title)
        layout.addWidget(header)
        message = make_selectable(QLabel(note))
        layout.addWidget(message)
        layout.addStretch(1)


def create_application(argv: list[str] | None = None) -> QApplication:
    """Return the shared QApplication, creating it when necessary.

    The shared visual language is installed here so that every window
    and page created afterwards renders identically, including windows
    built directly in tests.
    """
    existing = QApplication.instance()
    if existing is not None:
        assert isinstance(existing, QApplication)
        return apply_theme(existing)
    return apply_theme(
        QApplication(sys.argv if argv is None else argv)
    )


def create_main_window() -> MainWindow:
    """Create the application main window with default views."""
    return MainWindow()


class MainWindow(QMainWindow):
    """Main window with navigation panel and stacked content pages."""

    def __init__(self) -> None:
        """Create the window, navigation, and placeholder pages."""
        super().__init__()
        self.setWindowTitle("Neural Network Lab")
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        self._factories: dict[str, Callable[[], QWidget]] = {}
        self._pages: dict[str, QWidget] = {}
        self._order: list[str] = []

        central = QWidget(self)
        central.setObjectName("centralArea")
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(
            CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN
        )
        outer.setSpacing(CONTENT_SPACING)

        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        splitter.setObjectName("mainSplitter")
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        navigation_panel = QWidget(splitter)
        navigation_panel.setObjectName("navigationPanel")
        navigation_panel.setMinimumWidth(NAVIGATION_MIN_WIDTH)
        navigation_panel.setMaximumWidth(NAVIGATION_MAX_WIDTH)
        panel_layout = QVBoxLayout(navigation_panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(FIELD_SPACING)
        self._app_title = QLabel("Neural Network Lab", navigation_panel)
        self._app_title.setObjectName("appTitle")
        panel_layout.addWidget(self._app_title)
        self._app_subtitle = QLabel("Experiment workflow", navigation_panel)
        self._app_subtitle.setObjectName("appSubtitle")
        panel_layout.addWidget(self._app_subtitle)

        self._navigation = QListWidget(navigation_panel)
        self._navigation.setObjectName("navigationList")
        self._navigation.setUniformItemSizes(True)
        self._navigation.setSpacing(2)
        self._navigation.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self._navigation.setToolTip(
            "Jump to a section. The current section stays highlighted."
        )
        panel_layout.addWidget(self._navigation, 1)
        splitter.addWidget(navigation_panel)

        self._stack = QStackedWidget(splitter)
        self._stack.setObjectName("contentStack")
        splitter.addWidget(self._stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([NAVIGATION_WIDTH, DEFAULT_WINDOW_WIDTH])

        self._page_status = QLabel("", self)
        self.statusBar().addPermanentWidget(self._page_status)

        for spec in VIEW_SPECS:
            self._order.append(spec.key)
            item = QListWidgetItem(spec.title, self._navigation)
            item.setData(Qt.ItemDataRole.UserRole, spec.key)
            item.setToolTip(spec.title)
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

    def _dashboard_factory(self) -> QWidget:
        """Build the dashboard fed by the existing pages' state."""
        from neural_network_lab.gui.views.dashboard_view import DashboardView

        view = DashboardView()
        view.set_dataset_provider(self._datasets_current_dataset)
        view.set_network_provider(self._training_current_network)
        view.set_layer_sizes_provider(self._training_current_layer_sizes)
        view.set_training_active_provider(self._training_is_active)
        view.set_last_run_provider(self._training_last_run_details)
        view.set_experiment_state_provider(self._experiments_loaded_state)
        view.set_navigator(self.request_view)
        return view

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
        view.set_model_revision_provider(self._training_model_revision)
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

    def _visualization_factory(self) -> QWidget:
        """Build the visualization view fed by the model and dataset."""
        from neural_network_lab.gui.views.visualization_view import (
            VisualizationView,
        )

        view = VisualizationView()
        view.set_network_provider(self._training_current_network)
        view.set_dataset_provider(self._datasets_current_dataset)
        return view

    def _network_factory(self) -> QWidget:
        """Build the network configuration view fed by the active model."""
        from neural_network_lab.gui.views.network_configuration_view import (
            NetworkConfigurationView,
        )

        view = NetworkConfigurationView()
        view.set_current_sizes_provider(self._training_current_layer_sizes)
        view.set_dataset_provider(self._datasets_current_dataset)
        view.set_network_applier(self._training_apply_layer_sizes)
        return view

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

    def _training_current_layer_sizes(self):
        """Return the active network's layer sizes, if it exists."""
        page = self._pages.get("training")
        getter = getattr(page, "current_layer_sizes", None)
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception:
            return None

    def _training_is_active(self) -> bool:
        """Return whether the training view is running a training job."""
        page = self._pages.get("training")
        getter = getattr(page, "is_training_active", None)
        if not callable(getter):
            return False
        try:
            return bool(getter())
        except Exception:
            return False

    def _training_model_revision(self) -> int:
        """Return the active model's revision counter, 0 when unavailable."""
        page = self._pages.get("training")
        getter = getattr(page, "model_revision", None)
        if not callable(getter):
            return 0
        try:
            return int(getter())
        except Exception:
            return 0

    def _training_last_run_text(self) -> str | None:
        """Return the training view's one-line run summary, if any."""
        page = self._pages.get("training")
        getter = getattr(page, "last_run_summary", None)
        if not callable(getter):
            return None
        try:
            summary = getter()
        except Exception:
            return None
        return summary if isinstance(summary, str) and summary else None

    def _training_last_run_details(self) -> dict | None:
        """Return the training view's last finished-run fields, if any.

        The dashboard renders these as its results table, so a value the
        run did not record stays explicitly missing instead of being
        hidden inside one long sentence.
        """
        page = self._pages.get("training")
        getter = getattr(page, "last_run_details", None)
        if not callable(getter):
            return None
        try:
            details = getter()
        except Exception:
            return None
        return details if isinstance(details, dict) else None

    def _experiments_loaded_state(self) -> tuple[int, int, str | None]:
        """Return the experiment view's loaded-record summary.

        A zero-length directory string means the page was never opened,
        so the dashboard reports that instead of counting nothing.
        """
        page = self._pages.get("experiments")
        if page is None:
            return (0, 0, None)
        records_getter = getattr(page, "loaded_records", None)
        failures_getter = getattr(page, "load_failures", None)
        directory_getter = getattr(page, "current_directory", None)
        if not callable(records_getter):
            return (0, 0, None)
        try:
            records = len(records_getter())
        except Exception:
            records = 0
        skipped = 0
        if callable(failures_getter):
            try:
                skipped = len(failures_getter())
            except Exception:
                skipped = 0
        directory = ""
        if callable(directory_getter):
            try:
                directory = directory_getter() or ""
            except Exception:
                directory = ""
        return (records, skipped, directory)

    def _training_apply_layer_sizes(self, sizes) -> None:
        """Apply layer sizes to the training view's active network.

        The training page is materialized on demand so a configuration
        can be applied before Training is ever opened; the visible page
        and navigation order are unchanged.

        Raises:
            ValueError: When the training page cannot be built, the run
                is active, or the sizes are invalid — in every case the
                active network stays unchanged.
        """
        page = self._ensure_page("training")
        applier = getattr(page, "apply_layer_sizes", None)
        if not callable(applier):
            raise ValueError(
                "The training view is not available yet; open it once "
                "and try again."
            )
        applier(sizes)

    def _ensure_page(self, key: str) -> QWidget | None:
        """Build a registered page on demand without switching to it.

        Unlike :meth:`request_view` this never navigates and never
        validates unsaved edits; it only makes the page exist so other
        views can read or write through it. Returns None when the page
        is unknown or cannot be built.
        """
        page = self._pages.get(key)
        if page is not None:
            return page
        factory = self._factories.get(key)
        if factory is None:
            return None
        try:
            page = factory()
        except Exception:
            return None
        self._pages[key] = page
        self._stack.addWidget(page)
        return page

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
            item.setToolTip(title)
        self._factories[key] = factory
        existing = self._pages.pop(key, None)
        if existing is not None:
            self._stack.removeWidget(existing)
            existing.deleteLater()

    def register_default_views(self) -> None:
        """Register the real views implemented so far.

        The dashboard, datasets, network configuration, training,
        testing, parameters, experiments, and visualizations are live;
        training consumes the dataset editor selection, testing,
        parameters, and visualizations consume the trained model
        (visualizations also the selection), the dashboard reads the
        other pages' state, and experiments loads its own saved files —
        all resolved lazily at click time.
        """
        self.register_view("dashboard", "Dashboard", self._dashboard_factory)
        self.register_view("datasets", "Dataset Management", self._datasets_factory)
        self.register_view("network", "Network Configuration", self._network_factory)
        self.register_view("training", "Training", self._training_factory)
        self.register_view("testing", "Testing", self._testing_factory)
        self.register_view("parameters", "Weights and Biases", self._parameters_factory)
        self.register_view("experiments", "Experiment History", self._experiments_factory)
        self.register_view("visualization", "Visualization", self._visualization_factory)

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
        self._show_current_page_in_status_bar()
        return True

    def report_error(self, message: str, timeout_ms: int = 8000) -> None:
        """Present an error message in the status bar."""
        self.statusBar().showMessage(message, timeout_ms)

    def current_view_title(self) -> str:
        """Return the title of the visible page, or an empty string."""
        row = self._navigation.currentRow()
        if 0 <= row < len(self._order):
            item = self._navigation.item(row)
            if item is not None:
                return item.text()
        return ""

    def _show_current_page_in_status_bar(self) -> None:
        """Name the visible section in the window status bar."""
        title = self.current_view_title()
        self._page_status.setText(title)

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
                title = self.current_view_title() or "current section"
                answer = QMessageBox.question(
                    self,
                    "Unsaved changes",
                    f"{title} has unsaved edits. Leaving it now discards "
                    "them. Continue?",
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
                self._show_current_page_in_status_bar()

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
