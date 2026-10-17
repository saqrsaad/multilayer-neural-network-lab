"""Table widget and parsing helpers for the dataset editor.

Parsing and validation rules stay thin here on purpose: numeric
conversion errors are reported with their cell location, then the
existing data-layer :class:`Dataset` enforces shapes, finiteness, and
sample counts. No neural-network mathematics lives in this module.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from PySide6.QtWidgets import QTableWidget
from PySide6.QtWidgets import QTableWidgetItem

from neural_network_lab.data.dataset import DEFAULT_TASK

#: Base name of the target columns used by the dataset editor table.
TARGET_HEADER = "target"

#: Largest number of target columns the editor can hold.
MAX_TARGET_COLUMNS = 8


def feature_header(index: int) -> str:
    """Return the header label for a feature column."""
    return f"feature_{index}"


def target_header(index: int, count: int = 1) -> str:
    """Return the header label for one target column.

    A single target column keeps the plain ``target`` name; two or more
    are numbered ``target_0, target_1, ...`` so each column is
    identifiable in messages and in the table header.
    """
    if count <= 1:
        return TARGET_HEADER
    return f"{TARGET_HEADER}_{index}"


def parse_float_cell(text: str, *, row: int, column_name: str) -> float:
    """Parse one table cell, reporting its location on failure.

    Raises:
        ValueError: For empty, non-numeric, or non-finite cells, naming
            the 1-based row and column.
    """
    cleaned = text.strip()
    location = f"row {row + 1}, column '{column_name}'"
    if not cleaned:
        raise ValueError(f"Cell at {location} is empty.")
    try:
        value = float(cleaned)
    except ValueError as exc:
        raise ValueError(f"Cell at {location} is not a number: {text!r}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"Cell at {location} must be finite (no NaN or inf).")
    return value


def build_dataset_from_grid(
    grid: Sequence[Sequence[str]],
    *,
    n_features: int,
    has_target: bool,
    n_targets: int = 1,
    target_kind: str = DEFAULT_TASK,
):
    """Build a validated :class:`Dataset` from table text.

    Args:
        grid: Row-major cell text with ``n_features`` feature cells
            plus ``n_targets`` target cells per row when ``has_target``
            is set.
        n_features: Number of leading feature columns (at least 1).
        has_target: Whether the trailing columns hold targets.
        n_targets: Number of target columns. At least 1 when
            ``has_target`` is set; 0 selects a prediction-only dataset.
            Each target column becomes its own network output: target
            columns are never merged, dropped, duplicated, or filled in.
        target_kind: Declared task. Binary classification restricts the
            target cells to 0 or 1; continuous regression accepts any
            finite number and keeps its exact value.

    Returns:
        A validated ``Dataset`` (labeled when ``has_target`` is set).

    Raises:
        TypeError: For invalid ``n_features``/``n_targets``/``has_target``
            arguments or an unknown task.
        ValueError: For ragged rows, bad cells, out-of-range binary
            targets, or anything the data-layer validation rejects.
    """
    from neural_network_lab.data.dataset import BINARY_TARGETS
    from neural_network_lab.data.dataset import Dataset

    if isinstance(n_features, bool) or not isinstance(n_features, int):
        raise TypeError("n_features must be a positive integer")
    if n_features < 1:
        raise ValueError("n_features must be a positive integer")
    if isinstance(n_targets, bool) or not isinstance(n_targets, int):
        raise TypeError("n_targets must be a positive integer")
    if n_targets < 0:
        raise ValueError("n_targets must not be negative")
    if not isinstance(has_target, bool):
        raise TypeError("has_target must be a bool")
    if has_target and n_targets < 1:
        raise ValueError(
            "n_targets must be at least 1 when has_target is set; use "
            "has_target=False for a prediction-only dataset"
        )
    columns_of_targets = n_targets if has_target else 0
    rows = [list(row) for row in grid]
    if len(rows) == 0:
        raise ValueError("Enter at least one sample row before validating.")
    width = n_features + columns_of_targets
    features: list[list[float]] = []
    targets: list[list[float]] = []
    for row_index, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                f"Row {row_index + 1} has {len(row)} cells, expected {width}."
            )
        features.append(
            [
                parse_float_cell(
                    row[col], row=row_index, column_name=feature_header(col)
                )
                for col in range(n_features)
            ]
        )
        row_targets: list[float] = []
        for offset in range(columns_of_targets):
            column = n_features + offset
            name = target_header(offset, columns_of_targets)
            target = parse_float_cell(row[column], row=row_index, column_name=name)
            if target_kind == BINARY_TARGETS and target not in (0.0, 1.0):
                raise ValueError(
                    f"Target at row {row_index + 1}, column '{name}' must be "
                    f"0 or 1 for binary classification, got "
                    f"{row[column]!r}. Switch the task to continuous "
                    "regression to keep such values."
                )
            row_targets.append(target)
        if columns_of_targets:
            targets.append(row_targets)
    if columns_of_targets:
        return Dataset(features, targets, target_kind=target_kind)
    return Dataset(features)


class DatasetTable(QTableWidget):
    """Editable grid showing one sample per row."""

    def __init__(self, parent=None) -> None:
        """Create an empty table with editing enabled."""
        super().__init__(parent)
        self._n_features = 0
        self._n_targets = 0
        self._has_target = False
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        self.setMinimumHeight(220)
        self.horizontalHeader().setStretchLastSection(False)

    def configure(
        self, n_features: int, has_target: bool, n_targets: int = 1
    ) -> None:
        """Set the column layout (feature columns plus optional targets).

        Args:
            n_features: Number of feature columns (at least 1).
            has_target: Whether target columns follow the features.
            n_targets: Number of target columns. It must be at least 1
                when ``has_target`` is set; ``n_targets == 0`` is the
                supported prediction-only state and is only accepted
                together with ``has_target=False``.
        """
        if n_features < 1:
            raise ValueError("n_features must be a positive integer")
        if n_targets < 0:
            raise ValueError("n_targets must not be negative")
        if has_target and n_targets < 1:
            raise ValueError(
                "n_targets must be at least 1 when targets are present"
            )
        self._n_features = n_features
        self._has_target = has_target
        self._n_targets = n_targets if has_target else 0
        headers = [feature_header(i) for i in range(n_features)]
        headers += [
            target_header(index, self._n_targets)
            for index in range(self._n_targets)
        ]
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.resizeColumnsToContents()

    def n_features(self) -> int:
        """Number of feature columns currently configured."""
        return self._n_features

    def n_targets(self) -> int:
        """Number of target columns currently configured (0 when none)."""
        return self._n_targets

    def load_grid(self, rows: Sequence[Sequence[str]]) -> None:
        """Replace the table contents with the given cell text."""
        data = [list(row) for row in rows]
        self.setRowCount(len(data))
        for row_index, row in enumerate(data):
            for col_index in range(self.columnCount()):
                text = row[col_index] if col_index < len(row) else ""
                self.setItem(row_index, col_index, QTableWidgetItem(text))

    def grid(self) -> list[list[str]]:
        """Return the current cell text row by row."""
        rows: list[list[str]] = []
        for row_index in range(self.rowCount()):
            cells = []
            for col_index in range(self.columnCount()):
                item = self.item(row_index, col_index)
                cells.append("" if item is None else item.text())
            rows.append(cells)
        return rows

    def add_empty_row(self) -> None:
        """Append one blank editable row."""
        row = self.rowCount()
        self.insertRow(row)
        for col_index in range(self.columnCount()):
            self.setItem(row, col_index, QTableWidgetItem(""))

    def remove_selected_rows(self) -> int:
        """Remove rows intersecting the selection; return the count."""
        selected = sorted(
            {index.row() for index in self.selectedIndexes()},
            reverse=True,
        )
        for row in selected:
            self.removeRow(row)
        return len(selected)
