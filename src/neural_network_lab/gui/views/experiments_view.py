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
from PySide6.QtWidgets import QGroupBox
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QLineEdit
from PySide6.QtWidgets import QPlainTextEdit
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QTableWidget
from PySide6.QtWidgets import QTableWidgetItem
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget

from neural_network_lab.application.experiment_service import ExperimentRecord
from neural_network_lab.application.experiment_service import (
    ExperimentComparison,
)
from neural_network_lab.application.experiment_service import compare_experiments
from neural_network_lab.gui.main_window import CONTENT_MARGIN
from neural_network_lab.gui.main_window import CONTENT_SPACING
from neural_network_lab.storage.experiment_storage import load_experiment_record

#: Placeholder shown wherever a metric was not recorded (distinct from 0).
MISSING_METRIC_TEXT = "— (not recorded)"

#: Columns of the saved-experiments table.
LIST_COLUMNS = (
    "Experiment",
    "Created",
    "Mode",
    "Layers",
    "LR",
    "Epochs",
    "Train loss",
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


def format_record_detail(record: ExperimentRecord) -> str:
    """Render every field of a record as human-readable lines."""
    history = "\n".join(
        f"  epoch {index + 1}/{record.epochs}: {loss!r}"
        for index, loss in enumerate(record.loss_history)
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
        f"Test data: {format_name(record.test_dataset_name)}\n"
        f"Final train loss (MSE): {record.final_train_loss!r}\n"
        f"Test MSE: {format_metric(record.test_mse)}\n"
        f"Test accuracy: {format_metric(record.test_accuracy)}\n"
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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN
        )
        layout.setSpacing(CONTENT_SPACING)

        directory_row = QHBoxLayout()
        directory_row.setSpacing(CONTENT_SPACING)
        directory_row.addWidget(QLabel("Directory:", self))
        self._directory_edit = QLineEdit(self)
        self._directory_edit.setReadOnly(True)
        self._directory_edit.setPlaceholderText(
            "No directory selected — browse to a folder of saved records."
        )
        self._directory_edit.setObjectName("experimentsDirectory")
        directory_row.addWidget(self._directory_edit, 1)
        self._browse_button = QPushButton("Browse…", self)
        self._browse_button.setObjectName("browseButton")
        directory_row.addWidget(self._browse_button)
        self._refresh_button = QPushButton("Refresh", self)
        self._refresh_button.setObjectName("refreshButton")
        directory_row.addWidget(self._refresh_button)
        layout.addLayout(directory_row)

        self._status_label = QLabel("No records loaded.", self)
        self._status_label.setWordWrap(True)
        self._status_label.setObjectName("experimentsStatus")
        layout.addWidget(self._status_label)

        list_group = QGroupBox("Saved experiments", self)
        list_layout = QVBoxLayout(list_group)
        list_layout.setSpacing(CONTENT_SPACING)
        self._records_table = QTableWidget(list_group)
        self._records_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._records_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._records_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._records_table.setObjectName("experimentsTable")
        list_layout.addWidget(self._records_table)
        layout.addWidget(list_group, 1)

        detail_group = QGroupBox("Record detail", self)
        detail_layout = QVBoxLayout(detail_group)
        detail_layout.setSpacing(CONTENT_SPACING)
        self._detail_view = QPlainTextEdit(detail_group)
        self._detail_view.setReadOnly(True)
        self._detail_view.setObjectName("experimentDetail")
        self._detail_view.setPlaceholderText(
            "Select a record above to inspect its full configuration."
        )
        detail_layout.addWidget(self._detail_view)
        layout.addWidget(detail_group, 1)

        compare_group = QGroupBox("Side-by-side comparison", self)
        compare_layout = QVBoxLayout(compare_group)
        compare_layout.setSpacing(CONTENT_SPACING)
        self._compare_button = QPushButton("Compare selected (needs exactly 2)", self)
        self._compare_button.setObjectName("compareButton")
        compare_layout.addWidget(self._compare_button)
        self._comparison_table = QTableWidget(compare_group)
        self._comparison_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self._comparison_table.setObjectName("comparisonTable")
        compare_layout.addWidget(self._comparison_table, 1)
        layout.addWidget(compare_group, 1)

        self._error_label = QLabel("", self)
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #b00020;")
        self._error_label.setObjectName("experimentsError")
        layout.addWidget(self._error_label)

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
        records: list[ExperimentRecord] = []
        failures: list[tuple[str, str]] = []
        for file_path in sorted(directory.glob("*.json")):
            try:
                payload = load_experiment_record(file_path)
                records.append(ExperimentRecord.from_dict(payload))
            except (TypeError, ValueError, OSError) as exc:
                failures.append((file_path.name, str(exc)))
        self._records = records
        self._failures = failures
        self._directory_edit.setText(str(directory))
        self.clear_error()
        self._refresh_table()
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
            f"Directory {directory}: " + "; ".join(parts) + "."
        )
        return True

    def _on_browse(self) -> None:
        """Ask for a directory, then load it (nothing stored on cancel)."""
        chosen = QFileDialog.getExistingDirectory(self, "Open experiments directory")
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
