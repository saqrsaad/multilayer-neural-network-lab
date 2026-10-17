"""Experiments view: browse saved records and compare compatible pairs.

The view loads saved experiment files through the existing persistence
layer, validates them through the existing
:class:`ExperimentRecord` interface, and compares pairs through the
existing :func:`compare_experiments` compatibility rules. Management
and comparison logic stay in the application/service layers; this view
only presents results and collects user actions. Records are never
recalculated, invented, overwritten, or deleted here.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView
from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import QLineEdit
from PySide6.QtWidgets import QPlainTextEdit
from PySide6.QtWidgets import QSplitter
from PySide6.QtWidgets import QTabWidget
from PySide6.QtWidgets import QTableWidget
from PySide6.QtWidgets import QTableWidgetItem
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget

from neural_network_lab.application.experiment_service import ExperimentRecord
from neural_network_lab.application.experiment_service import (
    ExperimentComparison,
)
from neural_network_lab.application.experiment_service import compare_experiments
from neural_network_lab.gui.widgets.page_scaffold import FeedbackArea
from neural_network_lab.gui.widgets.page_scaffold import SectionCard
from neural_network_lab.gui.widgets.page_scaffold import button_row
from neural_network_lab.gui.widgets.page_scaffold import install_page
from neural_network_lab.gui.widgets.page_scaffold import make_button
from neural_network_lab.gui.widgets.ui_theme import CONTENT_SPACING
from neural_network_lab.storage.experiment_storage import load_experiment_record

#: Placeholder shown wherever a metric was not recorded (distinct from 0).
MISSING_METRIC_TEXT = "— (not recorded)"

#: Project root of this checkout (…/src/neural_network_lab/gui/views/).
_PROJECT_ROOT = Path(__file__).resolve().parents[4]

#: Directory the directory-selection dialog opens at first.
DEFAULT_EXPERIMENTS_DIR = _PROJECT_ROOT / "data" / "experiments"

#: Columns of the saved-experiments table.
LIST_COLUMNS = (
    "Experiment",
    "Created",
    "Mode",
    "Layers",
    "LR",
    "Epochs",
    "Train loss",
    "Val loss",
    "Test MSE",
    "Test acc",
    "Train data",
    "Test data",
)


def format_metric(value: float | None) -> str:
    """Format a metric, marking absent values as not recorded."""
    if value is None:
        return MISSING_METRIC_TEXT
    return repr(float(value))


def format_name(value: str | None) -> str:
    """Format an optional dataset name."""
    return value if value is not None else "—"


def format_layers(sizes: tuple[int, ...]) -> str:
    """Format layer sizes as ``[n0, n1, ...]``."""
    return "[" + ", ".join(str(size) for size in sizes) + "]"


def format_epoch(value: int | None) -> str:
    """Format an optional epoch number."""
    return str(value) if value is not None else "—"


def format_record_detail(record: ExperimentRecord) -> str:
    """Render every field of a record as human-readable lines."""
    history = "\n".join(
        f"  epoch {index + 1}/{record.epochs}: {loss!r}"
        for index, loss in enumerate(record.loss_history)
    )
    best_line = (
        f"Best epoch: {record.best_epoch} (restored, validation MSE "
        f"{record.best_validation_loss!r})"
        if record.best_epoch is not None and record.best_validation_loss is not None
        else "Best epoch: — (no validation subset, so the final epoch is the model)"
    )
    return (
        f"Experiment: {record.experiment_id}\n"
        f"Created: {record.created_at}\n"
        f"Mode: {record.mode}\n"
        f"Layers: {format_layers(record.layer_sizes)}\n"
        f"Learning rate: {record.learning_rate!r}\n"
        f"Epochs: {record.epochs}\n"
        f"Train data: {format_name(record.train_dataset_name)} "
        f"({record.train_samples} samples, "
        f"{record.train_features} features, "
        f"{record.train_outputs} outputs)\n"
        f"Validation rows: "
        + (
            "— (no validation subset)"
            if record.validation_samples is None
            else f"{record.validation_samples}"
        )
        + "\n"
        f"Test data: {format_name(record.test_dataset_name)}"
        + (
            ""
            if record.test_samples is None
            else f" ({record.test_samples} samples)"
        )
        + "\n"
        f"Final train loss (MSE): {record.final_train_loss!r}\n"
        f"Validation MSE: {format_metric(record.validation_loss)}\n"
        f"{best_line}\n"
        f"Test MSE: {format_metric(record.test_mse)}\n"
        f"Test accuracy: {format_metric(record.test_accuracy)}\n"
        f"Model state: {record.model_path or '— (not saved)'}\n"
        f"Loss history (recorded training MSE per epoch):\n{history}"
    )


def _delta_row(
    label: str, change: float | None, better: str
) -> tuple[str, str]:
    """Format one comparison delta row as ``(label, text)``."""
    if change is None:
        return (label, MISSING_METRIC_TEXT)
    return (f"{label} (second − first; {better})", f"{change:+.6g}")


def comparison_rows(
    first: ExperimentRecord,
    second: ExperimentRecord,
    comparison: ExperimentComparison,
) -> list[tuple[str, str, str]]:
    """Build side-by-side ``(field, A, B)`` rows from two records.

    Every value comes straight from the stored records and the
    :class:`ExperimentComparison`; nothing is recalculated here.
    """
    rows = [
        ("Experiment", first.experiment_id, second.experiment_id),
        ("Created", first.created_at, second.created_at),
        ("Mode", first.mode, second.mode),
        (
            "Layers",
            format_layers(first.layer_sizes),
            format_layers(second.layer_sizes),
        ),
        (
            "Learning rate",
            repr(first.learning_rate),
            repr(second.learning_rate),
        ),
        ("Epochs", str(first.epochs), str(second.epochs)),
        (
            "Train data",
            format_name(first.train_dataset_name),
            format_name(second.train_dataset_name),
        ),
        (
            "Train samples",
            str(first.train_samples),
            str(second.train_samples),
        ),
        (
            "Test data",
            format_name(first.test_dataset_name),
            format_name(second.test_dataset_name),
        ),
        (
            "Final train loss (MSE)",
            repr(first.final_train_loss),
            repr(second.final_train_loss),
        ),
        (
            "Validation MSE",
            format_metric(first.validation_loss),
            format_metric(second.validation_loss),
        ),
        (
            "Best epoch",
            format_epoch(first.best_epoch),
            format_epoch(second.best_epoch),
        ),
        (
            "Test MSE",
            format_metric(first.test_mse),
            format_metric(second.test_mse),
        ),
        (
            "Test accuracy",
            format_metric(first.test_accuracy),
            format_metric(second.test_accuracy),
        ),
    ]
    for label, text in (
        _delta_row(
            "Final train loss change", comparison.train_loss_change,
            "negative is better",
        ),
        _delta_row(
            "Test MSE change", comparison.test_mse_change, "negative is better",
        ),
        _delta_row(
            "Test accuracy change", comparison.test_accuracy_change,
            "positive is better",
        ),
    ):
        rows.append((label, "", text))
    return rows


class ExperimentsView(QWidget):
    """Browse saved experiment records and compare compatible pairs."""

    def __init__(self, parent=None) -> None:
        """Create the directory controls, list, detail, and compare panes."""
        super().__init__(parent)
        self._records: list[ExperimentRecord] = []
        self._failures: list[tuple[str, str]] = []

        self._page = install_page(
            self,
            "Experiment History",
            "Open a folder of saved experiment records to review them and "
            "compare two runs. This page never writes, recalculates, or "
            "deletes a record.",
        )

        directory_card = SectionCard(
            "Records folder",
            description=(
                "Choose the folder holding saved experiment JSON files. "
                "Files that cannot be read are reported and skipped; the "
                "valid ones are still listed."
            ),
            parent=self._page,
        )
        self._directory_edit = QLineEdit(directory_card)
        self._directory_edit.setReadOnly(True)
        self._directory_edit.setPlaceholderText(
            "No directory selected — browse to a folder of saved records."
        )
        self._directory_edit.setObjectName("experimentsDirectory")
        directory_card.add(self._directory_edit)
        self._browse_button = make_button(
            "Browse…",
            kind="primary",
            tooltip="Choose a folder of saved experiment records.",
            parent=directory_card,
        )
        self._browse_button.setObjectName("browseButton")
        self._refresh_button = make_button(
            "Refresh",
            kind="secondary",
            tooltip="Reload the records of the current folder.",
            parent=directory_card,
        )
        self._refresh_button.setObjectName("refreshButton")
        directory_card.add_layout(
            button_row(self._browse_button, self._refresh_button)
        )
        self._page.add(directory_card)

        self._feedback = FeedbackArea(
            self._page, status_text="No records loaded."
        )
        self._status_label = self._feedback.status_label()
        self._error_label = self._feedback.error_label()
        self._page.add(self._feedback)

        panes = QSplitter(Qt.Orientation.Vertical, self._page)
        panes.setObjectName("experimentsSplitter")
        panes.setChildrenCollapsible(False)

        list_group = SectionCard(
            "Saved experiments",
            description="Newest first. Select one row for its detail, or "
            "two rows to compare them.",
            parent=panes,
        )
        self._records_table = QTableWidget(list_group)
        self._records_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._records_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._records_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._records_table.setAlternatingRowColors(True)
        self._records_table.setObjectName("experimentsTable")
        self._records_table.setMinimumHeight(200)
        list_group.add(self._records_table, 1)
        panes.addWidget(list_group)

        detail_panes = SectionCard(
            "Detail and comparison",
            description=(
                "The first selected record is shown in full; select "
                "exactly two records to compare them."
            ),
            parent=panes,
        )
        self._detail_tabs = QTabWidget(detail_panes)
        self._detail_tabs.setObjectName("experimentDetailTabs")
        self._detail_view = QPlainTextEdit(self._detail_tabs)
        self._detail_view.setReadOnly(True)
        self._detail_view.setObjectName("experimentDetail")
        self._detail_view.setPlaceholderText(
            "Select a record above to inspect its full configuration."
        )
        self._detail_tabs.addTab(self._detail_view, "Record detail")
        compare_pane = QWidget(self._detail_tabs)
        compare_layout = QVBoxLayout(compare_pane)
        compare_layout.setContentsMargins(0, 0, 0, 0)
        compare_layout.setSpacing(CONTENT_SPACING)
        self._compare_button = make_button(
            "Compare selected (needs exactly 2)",
            kind="primary",
            tooltip="Compare the two selected records side by side.",
            parent=compare_pane,
        )
        self._compare_button.setObjectName("compareButton")
        compare_layout.addLayout(button_row(self._compare_button))
        self._comparison_table = QTableWidget(compare_pane)
        self._comparison_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self._comparison_table.setAlternatingRowColors(True)
        self._comparison_table.setObjectName("comparisonTable")
        compare_layout.addWidget(self._comparison_table, 1)
        self._detail_tabs.addTab(compare_pane, "Comparison")
        detail_panes.add(self._detail_tabs, 1)
        panes.addWidget(detail_panes)
        panes.setStretchFactor(0, 3)
        panes.setStretchFactor(1, 2)
        self._page.add(panes, 1)

        self._browse_button.clicked.connect(self._on_browse)
        self._refresh_button.clicked.connect(self._on_refresh)
        self._records_table.itemSelectionChanged.connect(self._on_selection_changed)
        self._compare_button.clicked.connect(self._on_compare)

    def show_error(self, message: str) -> None:
        """Present an error inline."""
        self._error_label.setText(message)

    def clear_error(self) -> None:
        """Clear any inline error message."""
        self._error_label.setText("")

    def loaded_records(self) -> list[ExperimentRecord]:
        """Return the currently listed records in display order."""
        return list(self._records)

    def load_failures(self) -> list[tuple[str, str]]:
        """Return ``(filename, message)`` pairs skipped by the last load."""
        return list(self._failures)

    def current_directory(self) -> str:
        """Return the directory currently shown, or an empty string."""
        return self._directory_edit.text().strip()

    def load_directory(self, path: str | Path) -> bool:
        """Load every ``*.json`` record file from a directory.

        Files that fail to load are reported and skipped; valid ones
        are still listed. A missing or unreadable directory leaves the
        current list untouched.

        Returns True when the directory itself was usable.
        """
        directory = Path(path)
        if not directory.is_dir():
            self.show_error(f"Not a directory: {directory}")
            return False
        keep_ids = self._selected_ids_in_order()
        records: list[ExperimentRecord] = []
        failures: list[tuple[str, str]] = []
        for file_path in sorted(directory.glob("*.json")):
            try:
                payload = load_experiment_record(file_path)
                records.append(ExperimentRecord.from_dict(payload))
            except (TypeError, ValueError, OSError) as exc:
                failures.append((file_path.name, str(exc)))
        # Newest first, using each record's own recorded timestamp.
        records.sort(key=lambda record: record.created_at, reverse=True)
        self._records = records
        self._failures = failures
        self._directory_edit.setText(str(directory))
        self.clear_error()
        self._refresh_table()
        self._restore_selection(keep_ids)
        self._detail_view.clear()
        self._comparison_table.clear()
        self._comparison_table.setRowCount(0)
        self._comparison_table.setColumnCount(0)
        parts = [f"{len(records)} record(s) loaded"]
        if failures:
            names = ", ".join(name for name, _ in failures)
            parts.append(f"{len(failures)} file(s) skipped: {names}")
            self.show_error(
                "Some files could not be loaded: "
                + "; ".join(f"{name}: {message}" for name, message in failures)
            )
        self._status_label.setText(
            f"Directory {directory}: " + "; ".join(parts) + " "
            "(newest first)."
        )
        return True

    def _on_browse(self) -> None:
        """Ask for a directory, then load it (nothing stored on cancel)."""
        chosen = QFileDialog.getExistingDirectory(
            self, "Open experiments directory", str(DEFAULT_EXPERIMENTS_DIR)
        )
        if not chosen:
            return
        self.load_directory(chosen)

    def _on_refresh(self) -> None:
        """Reload the current directory, if one is selected."""
        current = self._directory_edit.text().strip()
        if not current:
            self.show_error(
                "No directory selected. Browse to a folder of saved records first."
            )
            return
        self.load_directory(current)

    def _record_row(self, record: ExperimentRecord) -> list[str]:
        """Format one table row for a record."""
        return [
            record.experiment_id[:8],
            record.created_at,
            record.mode,
            format_layers(record.layer_sizes),
            repr(record.learning_rate),
            str(record.epochs),
            repr(record.final_train_loss),
            format_metric(record.validation_loss),
            format_metric(record.test_mse),
            format_metric(record.test_accuracy),
            format_name(record.train_dataset_name),
            format_name(record.test_dataset_name),
        ]

    def _refresh_table(self) -> None:
        """Rebuild the records table from the loaded list."""
        table = self._records_table
        table.clear()
        table.setColumnCount(len(LIST_COLUMNS))
        table.setHorizontalHeaderLabels(list(LIST_COLUMNS))
        table.setRowCount(len(self._records))
        for row_index, record in enumerate(self._records):
            for col_index, text in enumerate(self._record_row(record)):
                item = QTableWidgetItem(text)
                if col_index == 0:
                    item.setData(Qt.ItemDataRole.UserRole, record.experiment_id)
                table.setItem(row_index, col_index, item)
        table.resizeColumnsToContents()

    def _selected_ids_in_order(self) -> list[str]:
        """Return selected experiment ids in table-row order."""
        rows = sorted(
            {index.row() for index in self._records_table.selectedIndexes()}
        )
        ids = []
        for row in rows:
            item = self._records_table.item(row, 0)
            if item is not None:
                stored = item.data(Qt.ItemDataRole.UserRole)
                ids.append(stored if stored is not None else "")
        return ids

    def _restore_selection(self, ids: list[str]) -> None:
        """Re-select records that are still listed after a reload.

        Selection is re-applied by row so the refresh never silently
        changes which records the detail and comparison panes show.
        """
        wanted = [value for value in ids if value]
        if not wanted:
            return
        by_id = {
            record.experiment_id: index
            for index, record in enumerate(self._records)
        }
        self._records_table.blockSignals(True)
        try:
            for experiment_id in wanted:
                row = by_id.get(experiment_id)
                if row is None:
                    continue
                for column in range(len(LIST_COLUMNS)):
                    item = self._records_table.item(row, column)
                    if item is not None:
                        item.setSelected(True)
        finally:
            self._records_table.blockSignals(False)
        self._on_selection_changed()

    def _on_selection_changed(self) -> None:
        """Show the full detail of the first selected record."""
        ids = self._selected_ids_in_order()
        if not ids:
            self._detail_view.clear()
            return
        by_id = {record.experiment_id: record for record in self._records}
        record = by_id.get(ids[0])
        if record is None:
            return
        self._detail_view.setPlainText(format_record_detail(record))

    def _on_compare(self) -> None:
        """Compare exactly two selected records side by side."""
        self.clear_error()
        ids = self._selected_ids_in_order()
        if len(ids) != 2:
            self.show_error(
                f"Select exactly two experiments to compare "
                f"(currently {len(ids)} selected)."
            )
            return
        by_id = {record.experiment_id: record for record in self._records}
        first = by_id.get(ids[0])
        second = by_id.get(ids[1])
        if first is None or second is None:
            self.show_error("Selected records are no longer listed; refresh first.")
            return
        try:
            comparison = compare_experiments(first, second)
        except (TypeError, ValueError) as exc:
            self.show_error(f"Cannot compare: {exc}")
            return
        rows = comparison_rows(first, second, comparison)
        table = self._comparison_table
        table.clear()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(
            ["Field", f"A: {first.experiment_id[:8]}", f"B: {second.experiment_id[:8]}"]
        )
        table.setRowCount(len(rows))
        for row_index, (field, left, right) in enumerate(rows):
            table.setItem(row_index, 0, QTableWidgetItem(field))
            table.setItem(row_index, 1, QTableWidgetItem(left))
            table.setItem(row_index, 2, QTableWidgetItem(right))
        table.resizeColumnsToContents()
        self._detail_tabs.setCurrentIndex(1)
