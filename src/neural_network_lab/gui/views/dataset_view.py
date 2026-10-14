"""Dataset editor view: enter, edit, import, and export datasets.

The view owns only presentation and workflow: a table of samples with
one row per sample, feature columns plus a single trailing binary
target column, and buttons wired to the existing data-layer services
(:class:`Dataset` validation and CSV import/export). Dataset rules and
file formats stay in the data layer; this view never trains a model.

Unsaved edits are never discarded silently: the table keeps them until
validation succeeds, and navigation away, layout changes, replacement
by import, or clearing all request confirmation while edits are
pending. Failed imports or exports leave the table untouched.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PySide6.QtWidgets import QCheckBox
from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QSpinBox
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget

from neural_network_lab.data.csv_handler import export_dataset_to_csv
from neural_network_lab.data.csv_handler import import_dataset_from_csv
from neural_network_lab.data.dataset import Dataset
from neural_network_lab.gui.main_window import CONTENT_MARGIN
from neural_network_lab.gui.main_window import CONTENT_SPACING
from neural_network_lab.gui.widgets.dataset_table import build_dataset_from_grid
from neural_network_lab.gui.widgets.dataset_table import DatasetTable

#: Bounds for the feature-count control.
MIN_FEATURES = 1
MAX_FEATURES = 32

#: Initial feature count for a fresh editor.
DEFAULT_FEATURES = 2


def peek_column_count(path: Path, *, has_header: bool) -> int:
    """Return the CSV column width without validating its contents.

    Only the width is read here so the caller can state the explicit
    target-column configuration; all value validation stays inside
    :func:`import_dataset_from_csv`.

    Raises:
        ValueError: If no usable row reveals a column count.
    """
    with open(path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if has_header and rows:
        rows = rows[1:]
    for row in rows:
        if row:
            return len(row)
    raise ValueError(f"Could not determine a column count for: {path}")


class DatasetView(QWidget):
    """Table-based editor for one labeled or unlabeled dataset."""

    def __init__(self, parent=None) -> None:
        """Create the editor controls, table, and status labels."""
        super().__init__(parent)
        self._current: Dataset | None = None
        self._applied_grid: list[list[str]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN
        )
        layout.setSpacing(CONTENT_SPACING)

        options = QHBoxLayout()
        options.setSpacing(CONTENT_SPACING)
        options.addWidget(QLabel("Features:"))
        self._feature_spin = QSpinBox(self)
        self._feature_spin.setRange(MIN_FEATURES, MAX_FEATURES)
        self._feature_spin.setValue(DEFAULT_FEATURES)
        self._feature_spin.setObjectName("featureSpin")
        options.addWidget(self._feature_spin)
        self._target_check = QCheckBox("Include binary target column", self)
        self._target_check.setChecked(True)
        self._target_check.setObjectName("targetCheck")
        options.addWidget(self._target_check)
        self._header_check = QCheckBox("CSV has header row", self)
        self._header_check.setChecked(True)
        self._header_check.setObjectName("headerCheck")
        options.addWidget(self._header_check)
        options.addStretch(1)
        layout.addLayout(options)

        self._table = DatasetTable(self)
        self._table.setObjectName("datasetTable")
        self._table.configure(DEFAULT_FEATURES, True)
        layout.addWidget(self._table, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(CONTENT_SPACING)
        self._add_button = QPushButton("Add Row", self)
        self._remove_button = QPushButton("Remove Selected", self)
        self._validate_button = QPushButton("Validate && Apply", self)
        self._clear_button = QPushButton("Clear", self)
        self._import_button = QPushButton("Import CSV…", self)
        self._export_button = QPushButton("Export CSV…", self)
        for button in (
            self._add_button,
            self._remove_button,
            self._validate_button,
            self._clear_button,
            self._import_button,
            self._export_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self._summary_label = QLabel("No dataset loaded.", self)
        self._summary_label.setObjectName("datasetSummary")
        layout.addWidget(self._summary_label)
        self._error_label = QLabel("", self)
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #b00020;")
        self._error_label.setObjectName("datasetError")
        layout.addWidget(self._error_label)

        self._add_button.clicked.connect(self._on_add_row)
        self._remove_button.clicked.connect(self._on_remove_rows)
        self._validate_button.clicked.connect(self._on_validate)
        self._clear_button.clicked.connect(self._on_clear)
        self._import_button.clicked.connect(self._on_import)
        self._export_button.clicked.connect(self._on_export)
        self._feature_spin.valueChanged.connect(self._on_layout_changed)
        self._target_check.toggled.connect(self._on_layout_changed)

    def current_dataset(self) -> Dataset | None:
        """Return the last validated dataset, if any."""
        return self._current

    def has_unsaved_changes(self) -> bool:
        """Return True when the table differs from the applied dataset."""
        return self._table.grid() != self._applied_grid

    def show_error(self, message: str) -> None:
        """Present a validation or operation error inline."""
        self._error_label.setText(message)

    def clear_error(self) -> None:
        """Clear any inline error message."""
        self._error_label.setText("")

    def _confirm_discard(self, action: str) -> bool:
        """Ask before an action that would drop pending table edits."""
        if not self.has_unsaved_changes():
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved edits",
            f"{action} will discard unsaved table edits. Continue?",
            QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return answer == QMessageBox.Discard

    def _update_summary(self) -> None:
        """Refresh the one-line description of the applied dataset."""
        if self._current is None:
            self._summary_label.setText("No dataset loaded.")
            return
        dataset = self._current
        target_text = "target present" if dataset.has_targets else "no target"
        self._summary_label.setText(
            f"{dataset.n_samples} samples · {dataset.n_features} "
            f"features · {target_text} — valid."
        )

    def _apply_grid(self, grid: list[list[str]], dataset: Dataset) -> None:
        """Record successfully applied table contents and dataset."""
        self._applied_grid = [list(row) for row in grid]
        self._current = dataset
        self.clear_error()
        self._update_summary()

    def validate_and_apply(self) -> Dataset | None:
        """Validate the table, apply it on success, report errors."""
        try:
            dataset = build_dataset_from_grid(
                self._table.grid(),
                n_features=self._feature_spin.value(),
                has_target=self._target_check.isChecked(),
            )
        except (TypeError, ValueError) as exc:
            self.show_error(str(exc))
            return None
        self._apply_grid(self._table.grid(), dataset)
        return dataset

    def _on_add_row(self) -> None:
        """Append one blank editable row."""
        self.clear_error()
        self._table.add_empty_row()

    def _on_remove_rows(self) -> None:
        """Remove the selected rows (an explicit user action)."""
        self.clear_error()
        removed = self._table.remove_selected_rows()
        if removed == 0:
            self.show_error("No rows selected.")

    def _on_validate(self) -> None:
        """Validate the table contents and apply them when valid."""
        self.validate_and_apply()

    def _on_clear(self) -> None:
        """Clear the table after confirmation while edits are pending."""
        if not self._confirm_discard("Clearing the table"):
            return
        self._table.load_grid([])
        self._current = None
        self._applied_grid = []
        self.clear_error()
        self._update_summary()

    def _on_layout_changed(self) -> None:
        """Rebuild columns, preserving fitting values after confirmation."""
        if not self._confirm_discard("Changing the column layout"):
            self._restore_layout_controls()
            return
        previous = self._table.grid()
        n_features = self._feature_spin.value()
        has_target = self._target_check.isChecked()
        width = n_features + (1 if has_target else 0)
        preserved = [row[:width] for row in previous]
        self._table.configure(n_features, has_target)
        if preserved:
            padded = [row + [""] * (width - len(row)) for row in preserved]
            self._table.load_grid(padded)
        self.clear_error()

    def _restore_layout_controls(self) -> None:
        """Reset layout controls to match the live table columns."""
        columns = self._table.columnCount()
        last_header = (
            self._table.horizontalHeaderItem(columns - 1) if columns > 0 else None
        )
        has_target = last_header is not None and last_header.text() == "target"
        n_features = columns - (1 if has_target else 0)
        self._feature_spin.blockSignals(True)
        self._target_check.blockSignals(True)
        try:
            self._feature_spin.setValue(max(MIN_FEATURES, n_features))
            self._target_check.setChecked(has_target)
        finally:
            self._feature_spin.blockSignals(False)
            self._target_check.blockSignals(False)

    def _on_import(self) -> None:
        """Import a CSV file, preserving the table when it fails."""
        dialog_path, _ = QFileDialog.getOpenFileName(
            self, "Import dataset CSV", "", "CSV files (*.csv)"
        )
        if not dialog_path:
            return
        path = Path(dialog_path)
        has_header = self._header_check.isChecked()
        want_target = self._target_check.isChecked()
        try:
            width = peek_column_count(path, has_header=has_header)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            self.show_error(f"Import failed: {exc}")
            return
        if want_target and width < 2:
            self.show_error(
                f"Import failed: file has {width} column(s); a target "
                "column needs at least 2 columns."
            )
            return
        target_columns: int | None = width - 1 if want_target else None
        if not self._confirm_discard("Importing a file"):
            return
        try:
            dataset = import_dataset_from_csv(
                path, target_columns=target_columns, has_header=has_header
            )
        except (TypeError, ValueError, OSError) as exc:
            self.show_error(f"Import failed: {exc}")
            return
        features, targets = dataset.to_arrays()
        if features.shape[1] > MAX_FEATURES:
            self.show_error(
                f"Import failed: file has {features.shape[1]} feature "
                f"columns (at most {MAX_FEATURES} supported)."
            )
            return
        self._feature_spin.blockSignals(True)
        self._target_check.blockSignals(True)
        try:
            self._feature_spin.setValue(features.shape[1])
            self._target_check.setChecked(targets is not None)
        finally:
            self._feature_spin.blockSignals(False)
            self._target_check.blockSignals(False)
        self._table.configure(features.shape[1], targets is not None)
        grid: list[list[str]] = []
        for index in range(features.shape[0]):
            row = [repr(float(value)) for value in features[index]]
            if targets is not None:
                row += [repr(float(value)) for value in targets[index]]
            grid.append(row)
        self._table.load_grid(grid)
        self._apply_grid(grid, dataset)

    def _on_export(self) -> None:
        """Validate, then export the table without touching it."""
        try:
            dataset = build_dataset_from_grid(
                self._table.grid(),
                n_features=self._feature_spin.value(),
                has_target=self._target_check.isChecked(),
            )
        except (TypeError, ValueError) as exc:
            self.show_error(f"Export stopped: {exc}")
            return
        dialog_path, _ = QFileDialog.getSaveFileName(
            self, "Export dataset CSV", "", "CSV files (*.csv)"
        )
        if not dialog_path:
            return
        path = Path(dialog_path)
        if path.exists():
            answer = QMessageBox.question(
                self,
                "Overwrite file",
                f"{path} already exists. Overwrite it?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Yes:
                return
        try:
            export_dataset_to_csv(dataset, path, overwrite=True)
        except OSError as exc:
            self.show_error(f"Export failed: {exc}")
            return
        self._apply_grid(self._table.grid(), dataset)
        self.clear_error()
