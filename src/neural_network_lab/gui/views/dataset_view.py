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

import numpy as np
from PySide6.QtWidgets import QCheckBox
from PySide6.QtWidgets import QComboBox
from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import QFormLayout
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QSpinBox
from PySide6.QtWidgets import QWidget

from neural_network_lab.data.csv_handler import export_dataset_to_csv
from neural_network_lab.data.csv_handler import import_dataset_from_csv
from neural_network_lab.data.dataset import BINARY_TARGETS
from neural_network_lab.data.dataset import CONTINUOUS_TARGETS
from neural_network_lab.data.dataset import Dataset
from neural_network_lab.data.dataset import TARGET_KIND_LABELS
from neural_network_lab.gui.widgets.dataset_table import MAX_TARGET_COLUMNS
from neural_network_lab.gui.widgets.dataset_table import build_dataset_from_grid
from neural_network_lab.gui.widgets.dataset_table import DatasetTable
from neural_network_lab.gui.widgets.page_scaffold import FeedbackArea
from neural_network_lab.gui.widgets.page_scaffold import SectionCard
from neural_network_lab.gui.widgets.page_scaffold import button_row
from neural_network_lab.gui.widgets.page_scaffold import install_page
from neural_network_lab.gui.widgets.page_scaffold import make_button
from neural_network_lab.gui.widgets.ui_theme import FIELD_SPACING

#: Bounds for the feature-count control.
MIN_FEATURES = 1
MAX_FEATURES = 32

#: Initial feature count for a fresh editor.
DEFAULT_FEATURES = 2

#: Bounds for the target-column count control. Zero means prediction-only.
MIN_TARGETS = 0
MAX_TARGETS = MAX_TARGET_COLUMNS

#: Initial target-column count for a fresh editor.
DEFAULT_TARGETS = 1

#: Built-in educational datasets for binary classification.
#:
#: Each row is ``f1, f2, target`` and every dataset has two input
#: features and one binary target, so it validates through the same
#: rules as a manually typed or imported dataset.
BUILTIN_DATASETS: dict[str, tuple[str, ...]] = {
    "XOR": ("0,0,0", "0,1,1", "1,0,1", "1,1,0"),
    "AND": ("0,0,0", "0,1,0", "1,0,0", "1,1,1"),
    "OR": ("0,0,0", "0,1,1", "1,0,1", "1,1,1"),
}


def builtin_dataset_grid(name: str) -> list[list[str]]:
    """Return the editable grid rows for a built-in dataset.

    Args:
        name: One of :data:`BUILTIN_DATASETS`.

    Returns:
        A list of ``[f1, f2, target]`` string rows.

    Raises:
        KeyError: If no built-in dataset is called ``name``.
    """
    return [row.split(",") for row in BUILTIN_DATASETS[name]]


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
        """Create the layout options, sample table, and file actions."""
        super().__init__(parent)
        self._current: Dataset | None = None
        self._applied_grid: list[list[str]] = []

        self._page = install_page(
            self,
            "Dataset Management",
            "Enter one sample per row, then validate and apply it. Every "
            "other section reads the applied dataset; edits stay here "
            "until validation succeeds.",
        )

        layout_card = SectionCard(
            "Table layout",
            description=(
                "Feature columns first, then one target column per "
                "network output. Setting targets to 0 removes them "
                "(prediction-only), and Task decides how the values are "
                "validated and reported. Changing the layout asks before "
                "discarding pending edits."
            ),
            parent=self._page,
        )
        layout_form = QFormLayout()
        layout_form.setContentsMargins(0, 0, 0, 0)
        layout_form.setSpacing(FIELD_SPACING)
        self._feature_spin = QSpinBox(self)
        self._feature_spin.setRange(MIN_FEATURES, MAX_FEATURES)
        self._feature_spin.setValue(DEFAULT_FEATURES)
        self._feature_spin.setObjectName("featureSpin")
        self._feature_spin.setToolTip(
            "Number of input feature columns in the table."
        )
        layout_form.addRow("Feature columns:", self._feature_spin)
        self._target_spin = QSpinBox(self)
        self._target_spin.setRange(MIN_TARGETS, MAX_TARGETS)
        self._target_spin.setValue(DEFAULT_TARGETS)
        self._target_spin.setSpecialValueText("None (prediction only)")
        self._target_spin.setObjectName("targetCountSpin")
        self._target_spin.setToolTip(
            f"Number of target columns (0 to {MAX_TARGETS}). Each target "
            "column is one network output, so the network's output size "
            "must match this number."
        )
        layout_form.addRow("Target columns:", self._target_spin)
        self._task_combo = QComboBox(self)
        self._task_combo.setObjectName("taskTypeCombo")
        self._task_combo.addItem(
            "Binary classification (targets 0 or 1)", BINARY_TARGETS
        )
        self._task_combo.addItem(
            "Regression (continuous targets)", CONTINUOUS_TARGETS
        )
        self._task_combo.setToolTip(
            "The declared task decides how targets are validated and which "
            "metrics are reported. Binary targets must be 0 or 1; continuous "
            "targets keep their exact values and are reported with MSE, MAE, "
            "and RMSE instead of a class accuracy."
        )
        layout_form.addRow("Task:", self._task_combo)
        self._header_check = QCheckBox("CSV file has a header row", self)
        self._header_check.setChecked(True)
        self._header_check.setObjectName("headerCheck")
        self._header_check.setToolTip(
            "Skip the first row of an imported CSV file when it holds "
            "column names instead of values."
        )
        layout_form.addRow("CSV header:", self._header_check)
        builtin_row = QWidget(self)
        builtin_layout = button_row(trailing_stretch=False)
        builtin_layout.setContentsMargins(0, 0, 0, 0)
        self._builtin_combo = QComboBox(builtin_row)
        self._builtin_combo.setObjectName("builtinDatasetCombo")
        self._builtin_combo.addItem("Select…", None)
        for name in BUILTIN_DATASETS:
            self._builtin_combo.addItem(name, name)
        builtin_layout.addWidget(self._builtin_combo, 1)
        self._load_builtin_button = make_button(
            "Load built-in",
            kind="secondary",
            tooltip="Replace the table with a built-in educational "
            "dataset, then validate it like typed data.",
            parent=builtin_row,
        )
        builtin_layout.addWidget(self._load_builtin_button)
        builtin_row.setLayout(builtin_layout)
        layout_form.addRow("Built-in dataset:", builtin_row)
        layout_card.add_layout(layout_form)
        self._page.add(layout_card)

        samples_card = SectionCard(
            "Samples",
            description=(
                "One sample per row: feature columns first, then the "
                "target columns. Training and testing use the last "
                "validated contents, not these pending edits."
            ),
            parent=self._page,
        )
        self._table = DatasetTable(samples_card)
        self._table.setObjectName("datasetTable")
        self._table.configure(DEFAULT_FEATURES, True, DEFAULT_TARGETS)
        samples_card.add(self._table, 1)
        self._add_button = make_button(
            "Add Row",
            kind="secondary",
            tooltip="Append one empty sample row to the table.",
            parent=samples_card,
        )
        self._remove_button = make_button(
            "Remove Selected",
            kind="secondary",
            tooltip="Delete every selected row from the table.",
            parent=samples_card,
        )
        self._clear_button = make_button(
            "Clear",
            kind="danger",
            tooltip="Empty the table (asks first while edits are pending).",
            parent=samples_card,
        )
        self._validate_button = make_button(
            "Validate && Apply",
            kind="primary",
            tooltip="Check every cell and make this the dataset the other "
            "sections use.",
            parent=samples_card,
        )
        samples_card.add_layout(
            button_row(
                self._add_button,
                self._remove_button,
                self._clear_button,
                self._validate_button,
            )
        )
        # The table takes the space the window has left over, so the
        # sample grid grows instead of leaving an empty column behind.
        self._page.add(samples_card, 1)

        files_card = SectionCard(
            "CSV files",
            description=(
                "Import reads a comma-separated UTF-8 file: the last "
                "*Target columns* columns become the target columns and "
                "everything before them stays a feature column. Export "
                "validates the table first and never overwrites a file "
                "without asking."
            ),
            parent=self._page,
        )
        self._import_button = make_button(
            "Import CSV…",
            kind="secondary",
            tooltip="Load samples from a CSV file into the table.",
            parent=files_card,
        )
        self._export_button = make_button(
            "Export CSV…",
            kind="secondary",
            tooltip="Write the validated table to a CSV file.",
            parent=files_card,
        )
        files_card.add_layout(button_row(self._import_button, self._export_button))
        self._page.add(files_card)

        self._feedback = FeedbackArea(
            self._page, status_text="No dataset loaded."
        )
        self._summary_label = self._feedback.status_label()
        self._error_label = self._feedback.error_label()
        self._page.add(self._feedback)

        self._add_button.clicked.connect(self._on_add_row)
        self._remove_button.clicked.connect(self._on_remove_rows)
        self._validate_button.clicked.connect(self._on_validate)
        self._clear_button.clicked.connect(self._on_clear)
        self._import_button.clicked.connect(self._on_import)
        self._export_button.clicked.connect(self._on_export)
        self._load_builtin_button.clicked.connect(self._on_load_builtin)
        self._feature_spin.valueChanged.connect(self._on_layout_changed)
        self._target_spin.valueChanged.connect(self._on_layout_changed)
        self._task_combo.currentIndexChanged.connect(self._on_task_changed)

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
        if dataset.has_targets:
            target_text = (
                f"{dataset.n_outputs} target column"
                f"{'' if dataset.n_outputs == 1 else 's'}"
            )
        else:
            target_text = "no target"
        task_text = (
            f" · {TARGET_KIND_LABELS.get(dataset.target_kind, dataset.target_kind)}"
            if dataset.has_targets
            else ""
        )
        self._summary_label.setText(
            f"{dataset.n_samples} samples · {dataset.n_features} "
            f"features · {target_text} — valid{task_text}."
        )

    def _apply_grid(self, grid: list[list[str]], dataset: Dataset) -> None:
        """Record successfully applied table contents and dataset."""
        self._applied_grid = [list(row) for row in grid]
        self._current = dataset
        self.clear_error()
        self._update_summary()

    def _requested_targets(self) -> int:
        """Return the number of target columns the layout controls ask for."""
        return int(self._target_spin.value())

    def _requested_has_target(self) -> bool:
        """Return whether the layout controls ask for any target column."""
        return self._requested_targets() > 0

    def _requested_task(self) -> str:
        """Return the task declared by the task selector."""
        kind = self._task_combo.currentData()
        return kind if isinstance(kind, str) and kind else BINARY_TARGETS

    def _on_task_changed(self) -> None:
        """Re-validate under the newly declared task."""
        self.clear_error()

    def validate_and_apply(self) -> Dataset | None:
        """Validate the table, apply it on success, report errors."""
        try:
            dataset = build_dataset_from_grid(
                self._table.grid(),
                n_features=self._feature_spin.value(),
                has_target=self._requested_has_target(),
                n_targets=self._requested_targets(),
                target_kind=self._requested_task(),
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
        n_targets = self._requested_targets()
        has_target = n_targets > 0
        width = n_features + n_targets
        preserved = [row[:width] for row in previous]
        self._table.configure(n_features, has_target, n_targets)
        if preserved:
            padded = [row + [""] * (width - len(row)) for row in preserved]
            self._table.load_grid(padded)
        self.clear_error()

    def _restore_layout_controls(self) -> None:
        """Reset layout controls to match the live table columns."""
        n_features = self._table.n_features()
        n_targets = self._table.n_targets()
        self._feature_spin.blockSignals(True)
        self._target_spin.blockSignals(True)
        try:
            self._feature_spin.setValue(max(MIN_FEATURES, n_features))
            self._target_spin.setValue(n_targets)
        finally:
            self._feature_spin.blockSignals(False)
            self._target_spin.blockSignals(False)

    def _on_import(self) -> None:
        """Import a CSV file, preserving the table when it fails."""
        dialog_path, _ = QFileDialog.getOpenFileName(
            self, "Import dataset CSV", "", "CSV files (*.csv)"
        )
        if not dialog_path:
            return
        path = Path(dialog_path)
        has_header = self._header_check.isChecked()
        wanted_targets = self._requested_targets()
        try:
            width = peek_column_count(path, has_header=has_header)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            self.show_error(f"Import failed: {exc}")
            return
        if wanted_targets > 0 and width <= wanted_targets:
            self.show_error(
                f"Import failed: file has {width} column(s); "
                f"{wanted_targets} target column(s) need at least "
                f"{wanted_targets + 1} columns."
            )
            return
        target_columns: list[int] | None = (
            list(range(width - wanted_targets, width))
            if wanted_targets > 0
            else None
        )
        if not self._confirm_discard("Importing a file"):
            return
        try:
            dataset = import_dataset_from_csv(
                path,
                target_columns=target_columns,
                has_header=has_header,
                target_kind=self._requested_task(),
            )
        except (TypeError, ValueError, OSError) as exc:
            self.show_error(f"Import failed: {exc}")
            return
        features, targets = dataset.to_arrays()
        if targets is not None and self._requested_task() == BINARY_TARGETS:
            if not np.all((targets == 0.0) | (targets == 1.0)):
                self.show_error(
                    "Import stopped: the file's target columns hold "
                    "continuous values, but the task is binary "
                    "classification. Switch Task to 'Regression "
                    "(continuous targets)' to import these targets."
                )
                return
        if features.shape[1] > MAX_FEATURES:
            self.show_error(
                f"Import failed: file has {features.shape[1]} feature "
                f"columns (at most {MAX_FEATURES} supported)."
            )
            return
        if targets is not None and targets.shape[1] > MAX_TARGETS:
            self.show_error(
                f"Import failed: file has {targets.shape[1]} target "
                f"columns (at most {MAX_TARGETS} supported)."
            )
            return
        self._feature_spin.blockSignals(True)
        self._target_spin.blockSignals(True)
        try:
            self._feature_spin.setValue(features.shape[1])
            self._target_spin.setValue(
                0 if targets is None else int(targets.shape[1])
            )
        finally:
            self._feature_spin.blockSignals(False)
            self._target_spin.blockSignals(False)
        self._table.configure(
            features.shape[1],
            targets is not None,
            DEFAULT_TARGETS if targets is None else int(targets.shape[1]),
        )
        grid: list[list[str]] = []
        for index in range(features.shape[0]):
            row = [repr(float(value)) for value in features[index]]
            if targets is not None:
                row += [repr(float(value)) for value in targets[index]]
            grid.append(row)
        self._table.load_grid(grid)
        self._apply_grid(grid, dataset)

    def _on_load_builtin(self) -> None:
        """Load a built-in dataset through the normal apply flow.

        The chosen dataset is validated by the same rules as typed or
        imported data; the editor is only replaced after validation, and
        pending edits are confirmed first. No model is touched here.
        """
        name = self._builtin_combo.currentData()
        if name is None:
            self.show_error("Select a built-in dataset to load first.")
            return
        try:
            grid = builtin_dataset_grid(str(name))
        except KeyError:
            self.show_error(f"Unknown built-in dataset: {name}")
            return
        if not self._confirm_discard(f"Loading the built-in {name} dataset"):
            return
        try:
            dataset = build_dataset_from_grid(
                grid,
                n_features=2,
                has_target=True,
                n_targets=DEFAULT_TARGETS,
                target_kind=self._requested_task(),
            )
        except (TypeError, ValueError) as exc:
            self.show_error(f"Could not load the {name} dataset: {exc}")
            return
        self._feature_spin.blockSignals(True)
        self._target_spin.blockSignals(True)
        try:
            self._feature_spin.setValue(dataset.n_features)
            self._target_spin.setValue(DEFAULT_TARGETS)
        finally:
            self._feature_spin.blockSignals(False)
            self._target_spin.blockSignals(False)
        self._table.configure(
            dataset.n_features, True, int(dataset.n_outputs or DEFAULT_TARGETS)
        )
        self._table.load_grid(grid)
        self._apply_grid(grid, dataset)

    def _on_export(self) -> None:
        """Validate, then export the table without touching it."""
        try:
            dataset = build_dataset_from_grid(
                self._table.grid(),
                n_features=self._feature_spin.value(),
                has_target=self._requested_has_target(),
                n_targets=self._requested_targets(),
                target_kind=self._requested_task(),
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
