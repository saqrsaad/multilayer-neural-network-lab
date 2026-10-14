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

#: Target column header used by the dataset editor table.
TARGET_HEADER = "target"


def feature_header(index: int) -> str:
    """Return the header label for a feature column."""
    return f"feature_{index}"


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
    grid: Sequence[Sequence[str]], *, n_features: int, has_target: bool
):
    """Build a validated :class:`Dataset` from table text.

    Args:
        grid: Row-major cell text with ``n_features`` feature cells
            plus one target cell per row when ``has_target`` is set.
        n_features: Number of leading feature columns (at least 1).
        has_target: Whether the trailing column holds a binary target.

    Returns:
        A validated ``Dataset`` (labeled when ``has_target`` is set).

    Raises:
        TypeError: For invalid ``n_features``/``has_target`` arguments.
        ValueError: For ragged rows, bad cells, non-binary targets, or
            anything the data-layer validation rejects.
    """
    from neural_network_lab.data.dataset import Dataset

    if isinstance(n_features, bool) or not isinstance(n_features, int):
        raise TypeError("n_features must be a positive integer")
    if n_features < 1:
        raise ValueError("n_features must be a positive integer")
    if not isinstance(has_target, bool):
        raise TypeError("has_target must be a bool")
    rows = [list(row) for row in grid]
    if len(rows) == 0:
        raise ValueError("Enter at least one sample row before validating.")
    width = n_features + (1 if has_target else 0)
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
        if has_target:
            target = parse_float_cell(
                row[n_features], row=row_index, column_name=TARGET_HEADER
            )
            if target not in (0.0, 1.0):
                raise ValueError(
                    f"Target at row {row_index + 1} must be 0 or 1, "
                    f"got {row[n_features]!r}."
                )
            targets.append([target])
    if has_target:
        return Dataset(features, targets)
    return Dataset(features)


class DatasetTable(QTableWidget):
    """Editable grid showing one sample per row."""

    def __init__(self, parent=None) -> None:
        """Create an empty table with editing enabled."""
        super().__init__(parent)
        self._n_features = 0
        self._has_target = False
        self.setAlternatingRowColors(True)

    def configure(self, n_features: int, has_target: bool) -> None:
        """Set the column layout (feature columns plus optional target)."""
        if n_features < 1:
            raise ValueError("n_features must be a positive integer")
        self._n_features = n_features
        self._has_target = has_target
        headers = [feature_header(i) for i in range(n_features)]
        if has_target:
            headers.append(TARGET_HEADER)
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)

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
