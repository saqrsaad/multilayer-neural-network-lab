# Persistence Reference

This document describes how model states and experiment records are
saved, loaded, and validated in `multilayer-neural-network-lab`. Every
statement is grounded in the storage layer, the application services,
and the GUI views that invoke them. No pickle is used anywhere;
serialization is JSON only.

## Source files

- `../src/neural_network_lab/storage/_json_files.py` — shared JSON file
  helpers (paths, overwrite protection, finiteness sweeps, envelope
  checks).
- `../src/neural_network_lab/storage/model_storage.py` — model-state
  save/load (`save_model_state`, `load_model_state`).
- `../src/neural_network_lab/storage/experiment_storage.py` —
  experiment-record save/load (`save_experiment_record`,
  `load_experiment_record`).
- `../src/neural_network_lab/application/experiment_service.py` —
  `ExperimentRecord` (to_dict/from_dict), `ExperimentStore`,
  `compare_experiments`, `run_experiment`.
- `../src/neural_network_lab/core/network.py` —
  `NeuralNetwork.to_state_dict` / `NeuralNetwork.from_state_dict`.
- `../src/neural_network_lab/gui/views/training_view.py` — GUI Save
  Model / Load Model buttons and automatic save-on-training-completion.
- `../src/neural_network_lab/gui/views/experiments_view.py` — GUI
  browser for loading experiment records from a directory.

Existing design context:
- `../docs/ARCHITECTURE.md` §2.5 (Storage layer) and §9 (Persistence).
- `../docs/PROJECT_SPEC.md` §10 (Model and Experiment Persistence).
- `../docs/USER_GUIDE.md` §10 (Saving and loading models and experiments).

## 1. Format overview

Both model states and experiment records are versioned JSON objects
with a common envelope:

```json
{
  "format": "<format identifier>",
  "version": 1,
  "payload": { ... }
}
```

| Artifact | Format string | Version |
|---|---|---|
| Model state | `neural-network-lab/model-state` | 1 |
| Experiment record | `neural-network-lab/experiment-record` | 1 |

Only version 1 is read. Files with a different `format` or `version`
are rejected before the payload is examined.

Source:
- `../src/neural_network_lab/storage/model_storage.py`
  (`MODEL_FORMAT`, `MODEL_FORMAT_VERSION`)
- `../src/neural_network_lab/storage/experiment_storage.py`
  (`EXPERIMENT_FORMAT`, `EXPERIMENT_FORMAT_VERSION`)
- `../src/neural_network_lab/storage/_json_files.py` (`check_envelope`)

## 2. Model-state format

### Envelope

```json
{
  "format": "neural-network-lab/model-state",
  "version": 1,
  "payload": {
    "layer_sizes": [n0, n1, ..., nL],
    "weights": [ [row...], ... ],
    "biases": [ [row...], ... ]
  }
}
```

### Payload

The payload is produced by `NeuralNetwork.to_state_dict()`:

| Key | Type | Meaning |
|---|---|---|
| `layer_sizes` | list of int | At least two positive integers `[n0, n1, ..., nL]` |
| `weights` | list of matrices | One `(n_l, n_{l+1})` matrix per layer, as nested lists of floats |
| `biases` | list of matrices | One `(1, n_{l+1})` array per layer, as nested lists of floats |

All numbers are plain Python floats at full float precision (via
`tolist()`), so a loaded model reproduces the saved predictions exactly.

Source:
- `../src/neural_network_lab/core/network.py` (`to_state_dict`,
  `from_state_dict`)
- `../src/neural_network_lab/storage/model_storage.py`
  (`_check_model_payload`, `save_model_state`, `load_model_state`)

### Serialization / deserialization flow

Saving:
1. `NeuralNetwork.to_state_dict()` produces the payload dict.
2. `save_model_state(state, path, overwrite=False)` wraps it in the
   envelope, runs a finiteness sweep, and writes indented JSON.

Loading:
1. `load_model_state(path)` reads the JSON, rejects the file if it is
   missing, malformed, not an object, or has the wrong format/version.
2. `_check_model_payload(payload)` validates `layer_sizes`, `weights`,
   and `biases` structurally: correct key presence, matrix counts,
   rectangular non-empty rows, numeric cells, and finiteness.
3. The caller (typically `NeuralNetwork.from_state_dict(state)`)
   performs final shape/value validation while building the network.

No model object is constructed or modified by `load_model_state` alone;
validation happens before any model is built.

Source:
- `../src/neural_network_lab/storage/model_storage.py`
  (`save_model_state`, `load_model_state`, `_check_model_payload`)
- `../src/neural_network_lab/core/network.py` (`from_state_dict`)

## 3. Experiment-record format

### Envelope

```json
{
  "format": "neural-network-lab/experiment-record",
  "version": 1,
  "payload": { ... }
}
```

### Payload

The payload is produced by `ExperimentRecord.to_dict()` and mirrors the
`ExperimentRecord` dataclass fields:

| Key | Type | Notes |
|---|---|---|
| `experiment_id` | str | Non-empty unique identifier |
| `created_at` | str | ISO-8601 UTC timestamp |
| `mode` | str | `"continued"` or `"from_scratch"` |
| `layer_sizes` | list of int | At least two positive integers |
| `learning_rate` | number | Finite, positive |
| `epochs` | int | Positive |
| `train_dataset_name` | str or null | |
| `train_samples` | int | Positive |
| `train_features` | int | Positive |
| `train_outputs` | int | Positive |
| `test_dataset_name` | str or null | |
| `loss_history` | list of number | Non-empty, finite, length == `epochs` |
| `final_train_loss` | number | Finite; must equal last `loss_history` entry |
| `test_mse` | number or null | Finite, non-negative |
| `test_accuracy` | number or null | In `[0, 1]` |
| `validation_loss` | number or null | Finite (optional field) |
| `validation_samples` | int or null | Positive (optional field) |
| `best_epoch` | int or null | Positive, ≤ `epochs` (optional field) |
| `best_validation_loss` | number or null | Finite (optional field) |
| `test_samples` | int or null | Positive (optional field) |
| `model_path` | str or null | Path to the saved model state (optional field) |

`to_dict` and `from_dict` are inverses: `from_dict(to_dict(record))`
reproduces the record exactly. `from_dict` performs full semantic
validation of every field, including cross-field consistency
(`final_train_loss` vs. last `loss_history` entry, `best_epoch` vs.
`epochs`, ranges).

Source:
- `../src/neural_network_lab/application/experiment_service.py`
  (`ExperimentRecord`, `to_dict`, `from_dict`)
- `../src/neural_network_lab/storage/experiment_storage.py`
  (`REQUIRED_RECORD_KEYS`, `_check_record_payload`,
  `save_experiment_record`, `load_experiment_record`)

### Serialization / deserialization flow

Saving:
1. `ExperimentRecord.to_dict()` produces the payload dict.
2. `save_experiment_record(record, path, overwrite=False)` checks that
   all required keys are present, wraps the payload in the envelope,
   runs a finiteness sweep, and writes indented JSON.

Loading:
1. `load_experiment_record(path)` reads the JSON, rejects the file if it
   is missing, malformed, not an object, or has the wrong format/version,
   and checks that every required key is present.
2. The caller (`ExperimentRecord.from_dict(payload)`) performs full
   semantic validation before the record object is constructed.

No record object is constructed by `load_experiment_record` alone;
validation happens before any record is built.

Source:
- `../src/neural_network_lab/storage/experiment_storage.py`
  (`save_experiment_record`, `load_experiment_record`)
- `../src/neural_network_lab/application/experiment_service.py`
  (`ExperimentRecord.from_dict`)

## 4. Shared JSON helpers

`_json_files.py` provides the file-level concerns used by both formats:

- `coerce_path(path, name)` — converts a string or `os.PathLike` to
  `pathlib.Path`; rejects other types with `TypeError`.
- `check_overwrite_allowed(path, overwrite, what)` — raises
  `FileExistsError` if the file exists and `overwrite` is not `True`.
  Existing files are never overwritten silently.
- `_reject_non_finite(value, what)` — recursive sweep that rejects
  NaN/inf floats anywhere in a payload before saving. Booleans are
  exempted (they are not floats).
- `_reject_json_constants(value)` — `parse_constant` hook used by
  `json.load` to reject `NaN`, `Infinity`, and `-Infinity` literals
  while parsing; only finite numbers are allowed.
- `write_json_document(document, path, what)` — writes indented JSON
  (`indent=2`, trailing newline) after the finiteness sweep.
- `read_json_document(path, what)` — reads and parses JSON; rejects
  missing files (`FileNotFoundError`), malformed JSON (`ValueError`),
  and non-object content (`ValueError`).
- `check_envelope(document, path, what, expected_format, version)` —
  validates `format` and `version`, extracts and returns the `payload`
  dict, rejecting missing or non-object payloads.

Source:
- `../src/neural_network_lab/storage/_json_files.py`

## 5. File locations and naming behavior

Two directories are used for persisted artifacts:

| Directory | Purpose | Source constant |
|---|---|---|
| `data/models/` | Model-state files (`model_<uuid>.json`) | `MODELS_DIR` in `training_view.py` |
| `data/experiments/` | Experiment-record files (`experiment_<uuid>.json`) | `EXPERIMENTS_DIR` in `training_view.py` |

These directories are relative to the project root
(`src/neural_network_lab/gui/views/training_view.py` computes
`_PROJECT_ROOT` as four parent levels above the views directory, i.e.
the repository root).

### Naming

Auto-saved files use universally unique identifiers to avoid collisions:

- Model: `model_<uuid4 hex>.json` (optionally `model_<uuid>_<n>.json`
  if a collision occurs, up to 100 attempts).
- Experiment: `experiment_<uuid4 hex>.json` (optionally
  `experiment_<uuid>_<n>.json` on collision, up to 100 attempts).

If no unique filename is available within 100 attempts, the save is
reported as failed and the corresponding path is left `None`.

Source:
- `../src/neural_network_lab/gui/views/training_view.py`
  (`_save_completed_model`, `_save_completed_run`, `MODELS_DIR`,
  `EXPERIMENTS_DIR`, `_empty_run_reference`)

## 6. How the GUI saves and loads model states

### Save Model (manual)

The **Save Model** button in the Training view's *Model state* card
(`_on_save_model`):

1. Requires a current network; otherwise reports "No model to save."
2. Opens a save file dialog filtered to `JSON files (*.json)`.
3. If the chosen path already exists, asks for explicit overwrite
   confirmation (`QMessageBox.Yes | QMessageBox.Cancel`).
4. Calls `save_model_state(self._network.to_state_dict(), path,
   overwrite=overwrite)`.
5. Reports success (`Model saved to {path}.`) or failure
   (`Model could not be saved: {exc}`) in the model status line.

### Load Model (manual)

The **Load Model** button (`_on_load_model`):

1. Opens an open file dialog filtered to `JSON files (*.json)`.
2. Calls `load_model_state(dialog_path)`, then
   `NeuralNetwork.from_state_dict(state)` to rebuild the network.
3. On success, replaces the current network, increments the model
   revision counter, updates the network configuration controls to match
   the loaded architecture, clears any previous run results, and reports
   the load in the model status line.
4. On failure, reports `Model could not be loaded: {exc}` and leaves
   the current network untouched.

Loading a model must not silently alter its parameters or architecture;
validation occurs before the replacement.

Source:
- `../src/neural_network_lab/gui/views/training_view.py`
  (`_on_save_model`, `_on_load_model`)

## 7. How experiment records are created, stored, and accessed

### Creation

An experiment record is created automatically when a training run
finishes (`_present_run_results` → `_save_completed_run`):

1. The model state is saved first (`_save_completed_model`), producing
   `model_path`.
2. A fresh `experiment_id` (`uuid4().hex`) and UTC `created_at`
   timestamp are generated.
3. An `ExperimentRecord` is assembled from the run's measured values:
   mode, layer sizes, learning rate, epoch count, training loss history,
   final training loss, validation loss/samples/best epoch/best
   validation loss, and — if a test subset existed — test MSE, test
   accuracy, and test sample count.
4. The record is validated round-trip via
   `ExperimentRecord.from_dict(record.to_dict())`.
5. The `data/experiments/` directory is created if needed.
6. The record is saved with `save_experiment_record` under a unique
   filename (`experiment_<uuid>.json`).
7. The returned reference includes `record_path` and `model_path`; both
   are `None` if the corresponding artifact could not be written.

The trained model remains applied in memory regardless of whether the
artifacts are saved successfully. Save failures are reported on the
Training view but do not undo the training run.

Source:
- `../src/neural_network_lab/gui/views/training_view.py`
  (`_save_completed_run`, `_save_completed_model`,
  `_present_run_results`, `_empty_run_reference`, `_run_reference`)

### Storage and access

Experiment records are stored as individual JSON files in
`data/experiments/`. There is no central index file; the set of records
is discovered by scanning a directory for `*.json` files.

The Experiments view (`ExperimentsView.load_directory`):

1. Requires an existing directory; a missing or unreadable directory
   leaves the current list untouched and reports an error.
2. Iterates over `sorted(directory.glob("*.json"))`.
3. For each file, calls `load_experiment_record(file_path)` then
   `ExperimentRecord.from_dict(payload)`.
4. Files that fail to load are recorded as failures
   `(filename, reason)` and skipped; valid records are still listed.
5. Records are sorted newest first by their own `created_at` timestamp.
6. The table lists id, timestamp, mode, layers, learning rate, epochs,
   final train loss, validation loss, test MSE/accuracy, and dataset
   names. Missing metrics are shown as "not recorded," never as zero.

The view never writes, recalculates, overwrites, or deletes records.

Source:
- `../src/neural_network_lab/gui/views/experiments_view.py`
  (`load_directory`, `REQUIRED_RECORD_KEYS` usage via
  `load_experiment_record`)
- `../src/neural_network_lab/application/experiment_service.py`
  (`ExperimentStore` — in-memory registry; note: the GUI does not use
  `ExperimentStore` for file browsing, it loads files directly)

### Experiment comparison

Two compatible records can be compared side by side through
`compare_experiments(first, second)`. Compatibility requires identical
`layer_sizes`, identical training-data identity (dataset name and
sample/feature/output counts), and identical test dataset name (or both
absent). Deltas are `second − first` for training loss, test MSE, and
test accuracy; deltas are `None` when either record lacks the metric.

Source:
- `../src/neural_network_lab/application/experiment_service.py`
  (`compare_experiments`, `_compatibility_error`)
- `../src/neural_network_lab/gui/views/experiments_view.py`
  (`_on_compare`, `comparison_rows`)

## 7. Error handling

| Layer | What is checked | Errors raised |
|---|---|---|
| `_json_files` | Path type, file existence, overwrite permission, JSON parseability, envelope format/version, payload presence, finiteness | `TypeError`, `FileExistsError`, `FileNotFoundError`, `ValueError` |
| `model_storage` | Payload dict type, required keys, `layer_sizes` validity, matrix counts/shapes, rectangularity, numeric cells, finiteness | `TypeError`, `ValueError` |
| `experiment_storage` | Record dict type, required key presence | `TypeError`, `ValueError` |
| `ExperimentRecord.from_dict` | All field types, ranges, cross-field consistency, timestamp parseability | `TypeError`, `ValueError` |
| `NeuralNetwork.from_state_dict` | State dict type, required keys, architecture/parameter validity | `TypeError`, `ValueError` |
| GUI save/load | Dialog cancellation, `OSError` from file I/O | Reported in status lines; no crash |

Key behaviors:

- **No silent overwrite:** saving raises `FileExistsError` unless
  `overwrite=True` is explicitly passed (API) or confirmed via dialog
  (GUI).
- **No silent data corruption:** validation happens before any file is
  written or any model/record object is built.
- **No pickle:** `json` is used exclusively for deserialization.
- **Failure isolation:** a failed model or experiment save does not undo
  a completed training run; the in-memory model stays applied.
- **Skip-and-continue directory loading:** invalid experiment files are
  skipped and reported; valid records in the same directory are still
  listed.

## 8. Limitations and behavior not implemented

- **No GUI buttons for saving or loading experiment records.**
  Experiment records are saved automatically after a training run and
  can be browsed/loaded through the Experiments view, but there is no
  manual "Save Experiment Record" or "Load Experiment Record" button in
  the GUI. (Note: `docs/USER_GUIDE.md` §10 states this same
  limitation, although it also says "no GUI buttons for saving/loading
  models," which is outdated — Save Model and Load Model buttons do
  exist in the Training view.)
- **No central experiment index.** Records are discovered by directory
  scan; there is no catalog or database of all saved experiments.
- **No deletion or overwriting of experiment records through the GUI.**
  Records are never deleted by the application, and auto-saved files
  use unique names to avoid collisions.
- **`train_dataset_name` and `test_dataset_name` are `None` in
  auto-saved records.** The GUI's `_save_completed_run` passes
  `train_dataset_name=None` and `test_dataset_name=None`; the record
  stores sample/feature/output counts but not dataset names. Dataset
  names are only meaningful when supplied programmatically via
  `run_experiment(...)`.
- **No custom save path for auto-saved artifacts.** Auto-saved models
  and records always go to `data/models/` and `data/experiments/` with
  generated UUID filenames. Manual model save/load can use any path
  chosen through the file dialog.
- **`data/models/` and `data/experiments/` directories must be
  creatable/writable.** If directory creation fails (`OSError`), the
  corresponding save is reported as failed and the path is left `None`.
- **100-attempt unique-filename limit.** If 100 collision variants all
  exist, the save fails with "no unique filename available."
- **Records saved by older versions.** Only envelope version 1 is read;
  there is no migration path for other versions.
- **The experiments browser loads `*.json` indiscriminately.** Any
  `.json` file in the chosen directory is attempted as an experiment
  record; non-record JSON files are skipped and reported as failures.

## 9. Verification status

The persistence behavior described above was established through static
inspection of source code. The following were **not** executed in this
session and therefore cannot be confirmed as runtime behavior:

- Actual save/load round-trips through the GUI or API.
- Whether the `data/models/` and `data/experiments/` directories exist
  or are writable in a given environment.
- Whether auto-saved experiment files load correctly through the
  Experiments view at runtime.
- Test-suite results for persistence (`tests/test_storage.py` and
  related tests were reviewed as code, not executed).
