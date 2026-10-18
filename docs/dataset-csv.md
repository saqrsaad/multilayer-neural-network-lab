# Dataset and CSV Reference

This document describes how datasets are represented, validated, edited,
imported, and exported in `multilayer-neural-network-lab`. Every
statement is grounded in the current source code; no runtime behavior is
claimed.

## Source files

- `../src/neural_network_lab/data/dataset.py` — `Dataset` class and task
  constants (`BINARY_TARGETS`, `CONTINUOUS_TARGETS`, `NO_TASK`,
  `DEFAULT_TASK`).
- `../src/neural_network_lab/data/dataset_split.py` —
  `split_dataset`, `DatasetSplits`, `MIN_MEANINGFUL_SAMPLES`.
- `../src/neural_network_lab/data/csv_handler.py` —
  `import_dataset_from_csv`, `export_dataset_to_csv`,
  `_resolve_target_columns`.
- `../src/neural_network_lab/gui/views/dataset_view.py` — `DatasetView`,
  built-in datasets, import/export workflow.
- `../src/neural_network_lab/gui/widgets/dataset_table.py` —
  `DatasetTable`, `build_dataset_from_grid`, `parse_float_cell`,
  `MAX_TARGET_COLUMNS`.

Existing design context:
- `../docs/ARCHITECTURE.md` §2.4 (Data layer).
- `../docs/PROJECT_SPEC.md` §4 (Dataset Management).
- `../docs/USER_GUIDE.md` §5 (Datasets and CSV files).
- `../docs/algorithms.md` §10 (Dataset splitting).

## 1. Dataset data model

### `Dataset` class

`Dataset` (`data/dataset.py`) is the in-memory representation of a
numerical dataset. It holds:

- **Features**: a 2D float array of shape
  `(number_of_samples, number_of_features)`.
- **Targets**: an optional 2D float array of shape
  `(number_of_samples, number_of_outputs)`, or explicitly `None` when
  the dataset is unlabeled (prediction-only).

Target values are stored exactly as supplied — never rounded,
thresholded, or coerced.

### Array shapes and constraints

| Quantity | Shape | Constraint |
|---|---|---|
| Features | `(samples, features)` | ≥ 1 sample, ≥ 1 feature column, 2D only |
| Targets | `(samples, outputs)` | ≥ 1 sample, ≥ 1 output column, 2D only |
| Row count | — | Targets row count must match features row count |

Features and targets must be numeric, non-empty, and contain only finite
values (no NaN or infinity). Invalid rows are rejected; they are never
silently discarded. 1D inputs are not silently reshaped.

### Task types (declared, not guessed)

Every labeled dataset declares a **task**. The task is never inferred
from target values:

| Constant | Value | Meaning |
|---|---|---|
| `BINARY_TARGETS` | `"binary"` | Targets restricted to 0.0 or 1.0 where binary classification is used |
| `CONTINUOUS_TARGETS` | `"continuous"` | Any finite numeric target, preserved exactly |
| `NO_TASK` | `"none"` | Reported for unlabeled datasets |
| `DEFAULT_TASK` | `"binary"` | Task used when none is declared |

The default task is binary classification, which is the project's initial
scope. A target such as `0.61348` is never silently treated as binary or
rounded away; binary validation is applied only where binary
classification actually requires it.

Source:
- `../src/neural_network_lab/data/dataset.py` (`Dataset`,
  `validate_target_kind`, `BINARY_TARGETS`, `CONTINUOUS_TARGETS`,
  `NO_TASK`, `DEFAULT_TASK`)

### Dataset properties

| Property | Type | Meaning |
|---|---|---|
| `target_kind` | str | Declared task, or `"none"` when unlabeled |
| `is_binary_task` | bool | True for binary-declared and unlabeled datasets; False only for continuous |
| `has_targets` | bool | Whether the dataset carries target values |
| `n_samples` | int | Number of samples (rows) |
| `n_features` | int | Number of feature columns |
| `n_outputs` | int or None | Number of target columns, or `None` when unlabeled |

Source: `../src/neural_network_lab/data/dataset.py`

## 2. Feature and target validation

Core-level validation is performed by the data layer before any dataset
object is created:

- **Features**: must be 2D, at least one sample, at least one feature
  column, numeric, and finite. Empty, 1D, 3D, non-numeric, or
  non-finite inputs raise `TypeError` or `ValueError`.
- **Targets** (when present): must be 2D, at least one sample, at least
  one output column, row count matching features, numeric, and finite.
  Dimensions are never reshaped silently.

Binary targets are **not** validated at `Dataset` construction time.
The data layer records the task declaration without policing values.
Binary validation (targets must be 0.0 or 1.0) is applied where binary
classification actually requires it — in the GUI's `build_dataset_from_grid`
and on import, and in the evaluation service when a binary dataset is
evaluated.

### Compatibility with network sizes

`Dataset.validate_compatible_with(n_features, n_outputs)` checks that
the dataset's feature columns match the network input size and, when
`n_outputs` is given and the dataset is labeled, that its target
columns match the network output size. An unlabeled dataset passes a
features-only check but fails any supervised check requiring outputs.

Source:
- `../src/neural_network_lab/data/dataset.py`
  (`_as_feature_array`, `_as_target_array`, `Dataset.__init__`,
  `validate_compatible_with`, `is_compatible_with`)

## 3. GUI editing and validation

### Dataset editor (`DatasetView`)

The Dataset Management page (`gui/views/dataset_view.py`) is a table
editor: one sample per row, feature columns first, then target columns.

**Controls:**
- **Feature columns** spin box (1–32, default 2).
- **Target columns** spin box (0–8; 0 means prediction-only, default 1).
- **Task** combo: Binary classification (targets 0 or 1) or Regression
  (continuous targets).
- **CSV header** checkbox (whether the first row of an imported CSV is a
  header; checked by default).
- **Built-in dataset** selector and **Load built-in** button (XOR, AND,
  OR — each with two features and one binary target).
- Sample table: **Add Row**, **Remove Selected**, **Clear** (asks for
  confirmation while edits are pending).
- **Validate & Apply** (primary) — validates every cell and makes the
  table the dataset other sections use.
- **Import CSV…** and **Export CSV…** — file dialogs; failed imports or
  exports leave the table untouched.

**How changes are applied:**
- Edits to the table are pending until **Validate & Apply** succeeds.
- `has_unsaved_changes()` compares the current table grid to the
  applied grid.
- Navigation away, layout changes, import replacement, or clearing all
  request confirmation while edits are pending.
- On validation failure, the error is reported inline with row/column
  location and nothing is applied.

**Size limits:** `MAX_FEATURES = 32`, `MAX_TARGET_COLUMNS = 8` (the
target-column spin box is bounded accordingly). These are GUI limits,
not core limits.

### GUI-level grid validation (`build_dataset_from_grid`)

`gui/widgets/dataset_table.py` converts table text into a `Dataset`:

- Requires at least one sample row.
- Requires each row to have exactly `n_features` (+ `n_targets` target
  cells when targets are present); ragged rows are rejected with the
  row number.
- Parses each cell with `parse_float_cell`, reporting empty, non-numeric,
  or non-finite cells with their row and column location.
- For binary classification, rejects target cells that are not 0.0 or
  1.0, directing the user to switch to continuous regression.
- Delegates final shape/finiteness validation to `Dataset`.

Source:
- `../src/neural_network_lab/gui/views/dataset_view.py` (`DatasetView`)
- `../src/neural_network_lab/gui/widgets/dataset_table.py`
  (`DatasetTable`, `build_dataset_from_grid`, `parse_float_cell`,
  `MAX_TARGET_COLUMNS`)

## 4. CSV import behavior

### Format

- Plain comma-separated values, UTF-8 encoded.
- Optional header row (controlled by `has_header`; default `True`).
- One sample per row after the header.

### Explicit target-column selection

Target columns are determined **solely by the explicit**
`target_columns` argument. Header names are informational only and are
never used to infer which columns are targets.

`target_columns` accepts:
- `None` — unlabeled dataset; every column becomes a feature.
- A single 0-based column index.
- A sequence of 0-based column indices.

Target columns are laid out in ascending index order; feature columns
keep their file order. The selection must leave at least one feature
column and must not contain duplicates or out-of-range indices.

### GUI default target-column selection

Confirmed by the implementation: in `DatasetView._on_import`, the GUI
passes

```python
target_columns = list(range(width - wanted_targets, width))
```

when `wanted_targets > 0`, i.e. the **last N columns** become the
target columns, where N is the value of the **Target columns** spin box.
Everything before them stays a feature column. If `wanted_targets == 0`,
`target_columns` is `None` (unlabeled, prediction-only).

### Numeric parsing and invalid-data handling

- Every data cell is parsed with `float()`. Empty cells raise
  `ValueError` ("not a number").
- Blank rows are rejected ("never silently discarded").
- Ragged rows (wrong column count) are rejected with row and expected
  counts.
- Non-finite values (`nan`, `inf`) are rejected by the core `Dataset`
  validation after parsing.
- The header, when present, must have the same column count as the data
  rows; otherwise import fails.
- An empty file or a header-only file raises `ValueError`.
- A missing file raises `FileNotFoundError`.

On import failure, the editor's table is left untouched.

Source:
- `../src/neural_network_lab/data/csv_handler.py`
  (`import_dataset_from_csv`, `_resolve_target_columns`)
- `../src/neural_network_lab/gui/views/dataset_view.py` (`_on_import`,
  `peek_column_count`)

## 5. CSV export behavior

### Format and column naming

- Feature columns come first, named `feature_0, feature_1, …`.
- Target columns follow (when labeled), named `target_0, target_1, …`.
- One header row followed by one sample per row.
- UTF-8 encoded.

### Numeric precision

Values are written with `repr(float(value))`, i.e. full float precision,
so an exported file re-imported with the matching column configuration
yields identical numerical values.

### Overwrite protection

- `export_dataset_to_csv(dataset, path, overwrite=False)` raises
  `FileExistsError` if the file exists and `overwrite` is not `True`.
- In the GUI export flow, the user is shown an overwrite confirmation
  dialog before writing; canceling leaves the file untouched.
- Export validates the table first (via `build_dataset_from_grid`);
  invalid data never reaches the file dialog.

Source:
- `../src/neural_network_lab/data/csv_handler.py` (`export_dataset_to_csv`)
- `../src/neural_network_lab/gui/views/dataset_view.py` (`_on_export`)

## 6. Size limits and edge cases

| Limit | Value | Where enforced |
|---|---|---|
| Feature columns (GUI) | 1–32 | `DatasetView` (`MIN_FEATURES`, `MAX_FEATURES`) |
| Target columns (GUI) | 0–8 | `DatasetView` / `dataset_table.py` (`MIN_TARGETS`, `MAX_TARGET_COLUMNS`) |
| Built-in datasets | XOR, AND, OR | `DatasetView.BUILTIN_DATASETS` (2 features, 1 binary target each) |
| Samples | ≥ 1 | Core `Dataset` validation |
| Feature/target columns | ≥ 1 | Core `Dataset` validation |

Edge cases:
- **Unlabeled dataset** (`targets=None`): supports prediction but no
  supervised loss or accuracy; `target_kind` reports as `"none"`;
  `is_binary_task` is `True` (binary default applies).
- **Single-output vs multi-output**: each target column is its own
  network output; target columns are never merged, dropped, duplicated,
  or filled in.
- **Continuous targets in binary task**: rejected at grid validation
  time in the GUI and at evaluation time in `evaluate()`; the user must
  declare continuous regression to keep such values.
- **Split presets**: `split_dataset` offers train/validation/test
  presets including "train on all rows" (1.0/0.0/0.0). Small subsets
  (< 2 rows) are dropped into training with a warning.

Source:
- `../src/neural_network_lab/data/dataset_split.py`
  (`split_dataset`, `DatasetSplits`, `MIN_MEANINGFUL_SAMPLES`)

## 7. Core-level vs GUI-level validation (summary)

| Concern | Core (`data/`) | GUI (`gui/views/`, `gui/widgets/`) |
|---|---|---|
| Array shape (2D) | Enforced | Enforced via `build_dataset_from_grid` |
| Finite values | Enforced | Enforced via cell parsing + core |
| Row-count match | Enforced | Enforced via row-width check |
| Binary targets (0/1) | Not policed at construction | Enforced in grid builder and on import |
| Task declaration | Recorded, never guessed | Selected via Task combo |
| Numeric cell parsing | `np.array(..., dtype=float)` | `parse_float_cell` with row/column location |
| Ragged/blank rows | N/A (core takes arrays) | Rejected with row/column location |
| Overwrite protection | `FileExistsError` in storage | Confirmation dialog in export |

## 8. Limitations and behavior not implemented

- There is no GUI column-header editor; header names on import are
  informational only.
- The dataset editor does not persist datasets to disk directly; export
  writes CSV files chosen by the user, and the `data/datasets/` runtime
  directory exists but is not used by the GUI for auto-saving datasets.
- `TARGET_KINDS` contains only `binary` and `continuous`; no other task
  types are supported.

## 9. What cannot be established through static inspection

- Whether CSV import/export file dialogs render correctly under Qt.
- Whether `data/datasets/` exists or is writable in a given environment.
- Test-suite results for dataset/CSV behavior (`tests/test_dataset.py`,
  `tests/test_csv_handler.py`, `tests/test_dataset_editor.py` were
  reviewed as code, not executed).
