"""CSV import and export for datasets.

Supported format (also produced by :func:`export_dataset_to_csv`):

- Plain comma-separated values, UTF-8 encoded.
- One header row followed by one sample per row, unless the caller
  passes ``has_header=False`` on import.
- Exported headers name feature columns ``feature_0, feature_1, ...``
  and target columns ``target_0, target_1, ...``. On import, header
  names are informational only: which columns are features and which
  are targets is determined solely by the explicit ``target_columns``
  argument, never inferred from names.
- All data cells must be finite decimal numbers parseable by
  ``float()``. Full float precision is written, so an exported file
  re-imported with the matching column configuration yields
  identical numerical values.

The data layer never trains the network and never imports PySide6.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from neural_network_lab.data.dataset import Dataset


def _coerce_path(path: str | os.PathLike[str], name: str) -> Path:
    """Convert a path argument to ``Path``, rejecting other types."""
    if isinstance(path, (str, os.PathLike)):
        return Path(os.fspath(path))
    raise TypeError(f"{name} must be a file path string, got {type(path).__name__}")


def _resolve_target_columns(
    target_columns: int | Sequence[int] | None, n_columns: int
) -> list[int]:
    """Validate the explicit target-column configuration.

    Args:
        target_columns: ``None`` for an unlabeled (features-only)
            dataset, a single non-negative column index, or a
            non-empty sequence of non-negative column indices.
        n_columns: Number of columns in the CSV file.

    Returns:
        Sorted list of target column indices (empty when ``None``).

    Raises:
        TypeError: For non-integer column selections.
        ValueError: For negative/out-of-range indices, duplicates, an
            empty selection, or a selection leaving no feature column.
    """
    if target_columns is None:
        return []
    if isinstance(target_columns, bool):
        raise TypeError("target_columns must be column indices, not bool")
    if isinstance(target_columns, (int, np.integer)):
        selected = [int(target_columns)]
    elif isinstance(target_columns, Sequence) and not isinstance(
        target_columns, (str, bytes)
    ):
        selected = list(target_columns)
    else:
        raise TypeError(
            "target_columns must be None, a column index, or a sequence "
            f"of column indices, got {type(target_columns).__name__}"
        )
    if len(selected) == 0:
        raise ValueError(
            "target_columns must select at least one column; "
            "pass None explicitly for an unlabeled dataset"
        )
    for index in selected:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError(
                "target_columns must contain only integer column indices"
            )
        index = int(index)
        if index < 0 or index >= n_columns:
            raise ValueError(
                f"target column index {index} out of range for "
                f"{n_columns} CSV columns; use non-negative indices"
            )
    if len(set(selected)) != len(selected):
        raise ValueError("target_columns must not contain duplicate indices")
    if len(selected) >= n_columns:
        raise ValueError(
            "target_columns must leave at least one feature column"
        )
    return sorted(selected)


def import_dataset_from_csv(
    path: str | os.PathLike[str],
    *,
    target_columns: int | Sequence[int] | None,
    has_header: bool = True,
) -> Dataset:
    """Import a numerical CSV file as a :class:`Dataset`.

    Args:
        path: CSV file to read. The parent directory must exist.
        target_columns: Explicit target-column configuration: ``None``
            for an unlabeled dataset (every column becomes a feature),
            a single 0-based column index, or a sequence of 0-based
            column indices. Required with no guessing: the caller must
            always state which columns are targets. Target columns are
            laid out in ascending index order; feature columns keep
            their file order.
        has_header: Whether the first row is a header to skip. The
            header must then have the same column count as the data.

    Returns:
        Validated :class:`Dataset` (existing validation rules apply:
        2D shapes, matching sample counts, finite values).

    Raises:
        TypeError: For non-path ``path`` or non-integer column config.
        ValueError: For empty files, ragged rows, blank rows,
            non-numeric or non-finite cells, bad column selections, or
            datasets failing validation.
        FileNotFoundError: If ``path`` does not exist.
    """
    file_path = _coerce_path(path, "path")
    if not isinstance(has_header, bool):
        raise TypeError("has_header must be a bool")
    try:
        with open(file_path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {file_path}") from None
    if len(rows) == 0:
        raise ValueError(f"CSV file is empty: {file_path}")
    header: list[str] | None = None
    if has_header:
        header = rows.pop(0)
        if len(rows) == 0:
            raise ValueError(
                f"CSV file has a header but no data rows: {file_path}"
            )
    width = len(rows[0])
    start_line = 2 if has_header else 1
    if width < 1:
        raise ValueError(f"CSV row {start_line} has no columns: {file_path}")
    if header is not None and len(header) != width:
        raise ValueError(
            f"CSV header has {len(header)} columns but data rows have "
            f"{width}: {file_path}"
        )
    table: list[list[float]] = []
    for offset, row in enumerate(rows):
        line_number = start_line + offset
        if len(row) == 0:
            raise ValueError(
                f"CSV row {line_number} is blank; blank rows are rejected, "
                "never silently discarded"
            )
        if len(row) != width:
            raise ValueError(
                f"CSV row {line_number} has {len(row)} columns, "
                f"expected {width}"
            )
        values: list[float] = []
        for column, cell in enumerate(row):
            try:
                values.append(float(cell))
            except ValueError as exc:
                raise ValueError(
                    f"CSV row {line_number}, column {column}: "
                    f"not a number ({cell!r})"
                ) from exc
        table.append(values)

    data_indices = list(range(width))
    target_index_list = _resolve_target_columns(target_columns, width)
    target_set = set(target_index_list)
    feature_indices = [i for i in data_indices if i not in target_set]

    matrix = np.array(table, dtype=float)
    features = matrix[:, feature_indices]
    if not target_index_list:
        return Dataset(features)
    targets = matrix[:, target_index_list]
    return Dataset(features, targets)


def export_dataset_to_csv(
    dataset: Dataset, path: str | os.PathLike[str], *, overwrite: bool = False
) -> Path:
    """Export a :class:`Dataset` to a CSV file in the supported format.

    Feature columns come first (``feature_0, ...``), then target
    columns (``target_0, ...``) when the dataset is labeled. Values are
    written at full float precision so re-importing reproduces them.

    Args:
        dataset: Dataset to export.
        path: Destination file. Its parent directory must already exist;
            it is never created implicitly.
        overwrite: Must be ``True`` to replace an existing file.

    Returns:
        The destination :class:`Path`.

    Raises:
        TypeError: If ``dataset`` is not a :class:`Dataset`, ``path``
            is not a file path, or ``overwrite`` is not a bool.
        FileExistsError: If the file exists and ``overwrite`` is False.
        FileNotFoundError: If the parent directory does not exist.
    """
    if not isinstance(dataset, Dataset):
        raise TypeError(
            f"dataset must be a Dataset, got {type(dataset).__name__}"
        )
    file_path = _coerce_path(path, "path")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")
    if file_path.exists() and not overwrite:
        raise FileExistsError(
            f"file already exists: {file_path}; pass overwrite=True "
            "to replace it explicitly"
        )
    features, targets = dataset.to_arrays()
    header = [f"feature_{i}" for i in range(features.shape[1])]
    if targets is not None:
        header += [f"target_{j}" for j in range(targets.shape[1])]
    with open(file_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for sample in range(features.shape[0]):
            row = [repr(float(value)) for value in features[sample]]
            if targets is not None:
                row += [repr(float(value)) for value in targets[sample]]
            writer.writerow(row)
    return file_path
