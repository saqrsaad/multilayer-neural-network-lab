# Test Coverage Reference

This document describes the existing automated test suite based on
static inspection of the test files and project configuration. It does
not claim that any test was executed or passed; runtime verification is
outside the scope of this documentation phase.

## Source files consulted

- `pyproject.toml` — pytest configuration (`testpaths`, `pythonpath`,
  `addopts`).
- `tests/` — all 18 test files (reviewed as code only).

## 1. Test framework and discovery settings

`pyproject.toml` (`[tool.pytest.ini_options]`) declares:

| Setting | Value | Meaning |
|---|---|---|
| `testpaths` | `["tests"]` | Test discovery root directory |
| `pythonpath` | `["src"]` | Source tree added to `sys.path` for imports |
| `addopts` | `"-ra"` | Show summary of failed/skipped tests |

Optional development dependencies (`[project.optional-dependencies] dev`):
pytest, pytest-cov, ruff, mypy. These are not part of the runtime
requirements and are not installed by default.

Source: `pyproject.toml` lines 32–49.

## 2. Test directory organization

The `tests/` directory contains 18 test files. None use fixtures from a
shared conftest (no `conftest.py` was found in the tree). All tests
import from `neural_network_lab.*`, which resolves through the
`pythonpath = ["src"]` setting.

| Test file | Responsibility |
|---|---|
| `test_activations.py` | Sigmoid activation and its derivative |
| `test_losses.py` | Mean squared error |
| `test_network.py` | Forward propagation, parameter shapes, initialization |
| `test_backpropagation.py` | Gradient computation, parameter updates |
| `test_training.py` | Training modes (continued, from scratch), history |
| `test_evaluation.py` | Evaluation service, read-only invariant, metrics |
| `test_experiments.py` | Experiment records, store, comparison |
| `test_storage.py` | Model and experiment persistence (JSON round-trips) |
| `test_dataset.py` | Dataset representation and validation |
| `test_csv_handler.py` | CSV import and export |
| `test_dataset_editor.py` | Dataset editor GUI (offscreen) |
| `test_experiments_view.py` | Experiments view GUI (offscreen) |
| `test_main_window.py` | Main window and navigation GUI (offscreen) |
| `test_parameters_view.py` | Weights and biases editor GUI (offscreen) |
| `test_testing_view.py` | Testing view GUI (offscreen) |
| `test_training_view.py` | Training view GUI (offscreen) |
| `test_visualizations.py` | Chart widgets and validation helpers (offscreen) |
| `test_background_training.py` | Background training worker and view wiring |

Note: there is no `test_dataset_split.py` file in the repository, despite
the existence of `src/neural_network_lab/data/dataset_split.py`. The
splitting logic is exercised indirectly through the training-view tests
(which call `split_dataset` via the training workflow) but has no
dedicated test file.

## 3. Core algorithm coverage (explicit tests)

### Activations (`test_activations.py`)
- Sigmoid at representative values (0, ±1), output range (0, 1)
- Shape preservation for 2D input
- Acceptance of lists and scalars
- Extreme values (±1000) without overflow
- Derivative at zero (0.25) and matching formula `σ(z)·(1−σ(z))`
- Rejection of NaN/inf, empty input, non-numeric input

### Losses (`test_losses.py`)
- Perfect predictions → MSE 0
- Hand-calculated MSE for 1D and 2D arrays
- Returns Python float
- `mse` alias matches `mean_squared_error`
- Shape mismatch rejection
- Empty input rejection
- Non-finite (NaN/inf) rejection
- Non-numeric rejection

### Forward propagation (`test_network.py`)
- Parameter shapes follow conventions `(n_l, n_{l+1})` weights,
  `(1, n_{l+1})` biases
- Seed reproducibility
- Explicit parameters stored as copies
- Output shape `(samples, nL)`, range (0, 1)
- Forward matches an independent manual scalar oracle
- Single-sample forward
- `forward_with_cache` structure (activations/pre-activations lengths)
- Forward does not modify parameters
- Invalid layer-size rejection (value and type errors)
- Shape mismatch, partial parameters, non-finite parameters rejected
- Forward rejects wrong dimensions (1D, 3D, wrong feature count, zero
  samples) and non-finite/non-numeric input

### Backpropagation (`test_backpropagation.py`)
- Gradient shapes match parameter shapes
- Hand-computed single-neuron gradients match an independent scalar oracle
- Hidden gradients use pre-update weights (verified via snapshot)
- Parameters untouched by gradient computation
- Gradients match central finite differences (weight and bias, all layers)
- `apply_gradients` follows gradient descent exactly
- Bad target rejection (wrong rows, wrong cols, 1D, NaN, non-numeric)
- `validate_targets` direct behavior
- Learning rate validation (bad values and non-numeric types)
- Gradient shape mismatch rejection

### Training modes (`test_training.py`)
- Continued training records history (5 epochs, `mode="continued"`,
  finite losses, final loss equals last history entry, parameters move)
- Continued training starts from current parameters (matches one GD step)
- From-scratch reinitializes then trains (parameters replaced, deterministic
  with seed)
- From-scratch validates before resetting (bad epochs/lr leave parameters
  untouched)
- Bad epochs and learning rates rejected for both `train` and
  `train_from_scratch`
- Mismatched data rejection (row count, output count, feature count,
  NaN input)

### Evaluation (`test_evaluation.py`)
- Known example: predictions, targets, MSE, accuracy match a manual oracle
- Threshold boundary (0.5 counts as class 1)
- Evaluation never modifies parameters
- Rejects unlabeled dataset, incompatible datasets, non-binary targets,
  wrong argument types

## 4. Data validation and CSV coverage (explicit tests)

### Dataset (`test_dataset.py`)
- Labeled/unlabeled dataset representation
- Constructor copies inputs; properties return copies
- Bad feature shapes rejected (1D, scalar, 3D, empty, zero samples,
  zero features)
- Non-numeric and non-finite features rejected
- 1D targets not silently reshaped
- Target row-count mismatch rejected
- Empty/non-finite targets rejected
- Compatibility checks (labeled and unlabeled, bad feature/output sizes)

### CSV (`test_csv_handler.py`)
- Labeled CSV import with explicit target column
- Unlabeled CSV import (`target_columns=None`)
- Multiple and first-column target selection
- Import without header
- Export/import round-trip preserves values (labeled and unlabeled)
- Export overwrite protection
- Non-dataset rejection for export
- Malformed file rejection (empty, header-only, ragged, blank row,
  short row, non-numeric cell, empty cell)
- Non-finite value rejection
- Missing file rejection
- Bad target column selection rejection (out of range, duplicates,
  empty, wrong types)

### Dataset editor GUI (`test_dataset_editor.py`)
- Cell parsing with located errors (empty, non-numeric, non-finite)
- Dataset building from grid (labeled and unlabeled)
- Bad grid rejection (empty, ragged, out-of-range binary, non-numeric)
- Table add/remove rows and headers
- View validates and applies; invalid values leave view dirty
- Failed import preserves table state
- Successful import replaces table
- Export validates before opening dialog
- Export writes reimportable CSV
- Clear confirmation when dirty
- Layout change restores controls when cancelled

## 5. Evaluation, persistence, and experiment coverage (explicit tests)

### Evaluation (`test_evaluation.py`)
- Covered above in §3.

### Experiment management (`test_experiments.py`)
- `run_experiment` records all fields (id, timestamp, mode, layer sizes,
  lr, epochs, dataset info, history, final loss, test metrics)
- Unique experiment IDs
- From-scratch without test set (test metrics are `None`)
- Bad request rejection (unknown mode, unlabeled train set, orphan test
  name, seed with continued mode, wrong types)
- Record dict round-trip
- Record `from_dict` rejects invalid fields
- Store add/list/get, duplicate rejection, wrong type rejection
- Experiment comparison deltas (`second − first`)
- Comparison without test metrics (deltas are `None`)
- Comparison rejects incompatible pairs and wrong types

### Persistence (`test_storage.py`)
- Model round-trip preserves predictions exactly
- Saved model file is plain JSON with correct format/version
- Experiment round-trip
- Saved experiment file is plain JSON with correct format/version
- Save refuses overwrite without authorization
- Load missing files
- Load rejects malformed JSON, wrong format/version, missing payload,
  NaN literals, ragged/mismatched payloads, non-finite JSON numbers,
  pickle bytes
- Save rejects non-finite state
- Save rejects wrong types

## 6. GUI coverage (explicit tests, offscreen)

GUI tests set `QT_QPA_PLATFORM=offscreen` and import PySide6/Matplotlib
conditionally. They do not launch a visible window.

### Main window (`test_main_window.py`)
- Navigation covers documented sections
- Unimplemented views show placeholders without crashing
- Dataset view registers as integration point
- Unknown view key never crashes
- Failing factory shows error placeholder
- Unsaved-changes guard blocks and allows navigation
- `create_application` reuses instance
- `create_main_window` starts predictably

### Dataset editor (`test_dataset_editor.py`)
- Covered above in §4.

### Training view (`test_training_view.py`)
- `parse_hidden_sizes` and `parse_learning_rate` validation
- Layer sizes require dataset input size
- From-scratch runs and reports (history lines, status text)
- Continue without model reports error
- Continue training preserves starting point (parameters change)
- Architecture change requires explicit rebuild (continue refused)
- Invalid settings rejected before training
- Unlabeled and missing datasets rejected
- Incompatible output size rejected

### Testing view (`test_testing_view.py`)
- Missing model reported
- Single-sample prediction shows probability and class
- Invalid manual input keeps text
- Labeled evaluation shows metrics and rows; parameters unchanged
- Prediction-only input has no metrics
- Incompatible and unlabeled problems reported
- Predict never modifies parameters
- Results table is read-only

### Parameters view (`test_parameters_view.py`)
- Cell parsing and formatting
- Display maps layers, shapes, values
- Missing model disables editing
- Valid edit updates intended parameter only
- Invalid edit leaves model unchanged
- Architecture change requires refresh
- Pending edits survive layer switch
- Refresh confirm keeps or discards edits
- Refresh picks up trained model
- Notice when model changed with pending edits
- Registers in navigation

### Visualizations (`test_visualizations.py`)
- `validate_loss_history` validation and rejection
- `validate_prediction_inputs` validation and rejection
- Loss chart plots recorded epochs; replaces stale content
- Prediction chart distinguishes series (predicted, class, actual)
- Prediction chart without targets
- Prediction chart rejects bad data safely
- Training controls refresh chart from history
- Testing view refreshes chart on evaluation
- Testing view prediction-only plots without targets

### Background training (`test_background_training.py`)
- Worker scratch run reports progress and result
- Worker continued run starts from snapshot and leaves source untouched
- Worker rejects bad mode and epochs
- Worker reports training errors without crashing
- Worker cancel before and during run
- View refuses second run while active
- View progress/failed/cancelled slots
- View installs result preserving object identity
- Close prompts and cancels running training
- Close without training skips prompt

### Experiments view (`test_experiments_view.py`)
- Format helpers distinguish missing from zero
- Load directory lists saved records
- Load skips bad files and keeps valid ones
- Missing directory leaves list untouched
- Selecting row shows full detail
- Compare compatible pair shows labeled deltas
- Compare marks absent metrics without fabrication
- Compare rejects wrong selection count
- Compare rejects incompatible pair
- Comparison rows use recorded values only
- Registers in navigation

## 7. Edge cases and failure paths covered by tests

Confirmed by test code:

- Empty input arrays rejected across activations, losses, network, dataset
- NaN/inf rejection across activations, losses, network, dataset, CSV
- Non-numeric input rejection (`TypeError`)
- Wrong tensor shapes rejected (1D instead of 2D, mismatched rows/cols)
- Learning rate validation (zero, negative, NaN, inf, non-numeric)
- Epoch count validation (zero, negative, non-int, non-numeric)
- Atomic parameter updates (rejected edits leave model unchanged)
- Forward propagation does not modify parameters
- Evaluation does not modify parameters
- Continued training preserves parameters as starting point
- From-scratch validates before replacing parameters
- Overwrite protection for model, experiment, CSV, and dataset export
- Malformed CSV rejection with row/column location
- Failed import/export preserves table state
- Unsaved-changes confirmation dialogs
- Background worker error reporting without crashing the thread
- Worker cancellation (before and during run)
- Experiment comparison compatibility rejection
- Record `from_dict` validation of every field

## 8. Coverage gaps and untested areas (by inspection)

The following appear **untested or only indirectly covered**, based on
static inspection of test file names and contents:

- **`dataset_split.py` has no dedicated test file.** The splitting logic
  (`MIN_MEANINGFUL_SAMPLES`, largest-remainder allocation, warning
  emission, ratio validation) is not directly asserted by any test file.
  It is exercised indirectly through the training-view tests, which call
  `split_dataset` as part of the training workflow, but no test asserts
  split subset counts, warnings, or meaningful-samples fallback directly.
- **GUI chart rendering under QtAgg** — chart classes are instantiated
  and their axes inspected in `test_visualizations.py`, but this runs
  under the offscreen platform; actual pixel rendering is not verified.
- **Persistence file I/O with real directories** — `test_storage.py`
  uses `tmp_path`, so save/load round-trips are exercised at the unit
  level, but auto-save directory creation (`data/models/`,
  `data/experiments/`) and the GUI's `_save_completed_run` flow are not
  covered by a dedicated test.
- **Training worker threading and QThread signal/slot behavior** — the
  worker's `run()` is invoked directly in `test_background_training.py`
  rather than through an actual `QThread`, so thread-affinity and
  signal delivery across threads are not verified.
- **Main window close with running training** — partially covered
  (`test_close_prompts_and_cancels_running_training` uses a fake busy
  view, not a real TrainingView).
- **View providers and cross-page state sharing** — the provider
  callbacks injected by `MainWindow` are not tested for the actual
  wiring between Dataset Management, Training, Testing, Visualization,
  and Dashboard.

## 9. Categories of verification (distinction)

| Category | Status | Basis |
|---|---|---|
| Tests present in repository | 18 files present | Static inspection of `tests/` |
| Tests reviewed statically | All 18 files | Read as code in this audit |
| Tests actually executed | None | Execution was prohibited in this session |
| Runtime behavior verified | None | Application/Qt not launched |

No test was executed in this documentation phase, and no behavior is
claimed to pass at runtime. The test files define expectations, but
their outcomes are unknown without running `python -m pytest`.

## 10. What cannot be established through static inspection

- Whether all 18 test files pass when executed with `pytest`.
- Whether GUI tests behave correctly under the offscreen Qt platform in
  the actual environment.
- Whether `data/models/` and `data/experiments/` directories are
  created and writable at runtime.
- Whether auto-saved experiment/model files load correctly through the
  Experiments view.
- Whether the `dataset_split.py` splitting logic produces the expected
  subset counts and warnings for real datasets (no dedicated test file).
