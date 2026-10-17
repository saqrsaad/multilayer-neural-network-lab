# Neural Network Lab — User and Developer Guide

This guide describes the application as implemented: setup and launch,
the test suite, the graphical workflow, the learning algorithm at an
educational level, and known limitations. Every feature below was
checked against the repository; anything not listed here should be
treated as unavailable.

## 1. Setup

- Requires **Python 3.12 or newer** (`requires-python` in
  `pyproject.toml`).
- Runtime dependencies declared in `pyproject.toml`: **NumPy**,
  **PySide6**, **Matplotlib**, **Pandas**. Development tools:
  **pytest**, **pytest-cov**, **ruff**, **mypy** (the `dev` extra).
- The environment is prepared manually by the project owner. A
  typical setup from the repository root is:

```text
python -m pip install -e .
python -m pip install -e ".[dev]"
```

- The core mathematics use NumPy directly; PyTorch and TensorFlow are
  not used anywhere.

## 2. Launching the application

Run either of these from the repository root:

```text
python -m neural_network_lab
neural-network-lab
```

Both invoke `main()` in `src/neural_network_lab/main.py` (also
declared as the `neural-network-lab` console script in
`pyproject.toml`): it creates or reuses the shared `QApplication`,
shows the main window with all implemented views registered, and runs
the Qt event loop.

## 3. Running the tests

From the repository root:

```text
python -m pytest
```

Test configuration lives in `pyproject.toml` (`testpaths = ["tests"]`,
`pythonpath = ["src"]`). The suite has 18 test files covering
activations, forward propagation, loss calculation, backpropagation,
parameter updates, training behavior (including background training),
evaluation behavior, dataset validation, CSV import/export, the
dataset editor, the main window, the training/testing/parameters
views, visualizations, model persistence, experiment persistence and
management, and the experiments view.

## 4. The interface at a glance

The main window shows a navigation panel beside the active page. The
entry of the visible section stays highlighted, the window status bar
names the current section, and the navigation panel can be resized with
the splitter. Sections and their status:

- **Dashboard** — live overview (section 4.1).
- **Dataset Management** — live dataset editor (section 5).
- **Network Configuration** — live; edits the layer sizes of the same
  model the Training view uses (section 6).
- **Training** — live (section 6).
- **Testing** — live (section 7).
- **Weights and Biases** — live parameter editor (section 8).
- **Experiment History** — live record browser (section 9).
- **Visualization** — live decision-boundary and architecture charts.

Every page has the same structure: a page title with a one-line
explanation, one or more titled section cards, and a feedback area at
the bottom holding a status line and a validation/error line. Both
feedback lines disappear while they are empty. Validation errors,
operation status, warnings, and saved-file confirmations appear in the
coloured strips next to the control they belong to. Each section has at
most one primary (highlighted) button; state-replacing or destructive
actions such as **Clear**, **Cancel run**, and model replacement are
styled as warnings and ask for confirmation first. Long pages scroll
instead of clipping controls, tables, or charts, and the window can be
resized down to 1000×640 pixels.

Views with pending edits ask for confirmation (Discard/Cancel) before
navigation discards them. Closing the window while training runs asks
whether to quit and cancel training.

### 4.1 Dashboard

The Dashboard reports only state that exists: the applied dataset, the
active network and its layer sizes, whether the two are compatible, the
training run state, the last completed run's summary, and the records
loaded by Experiment History. A missing value is reported as a clear
empty state that names the section that owns it — no metric is
estimated or reused from an older run. The **Next step** card names the
recommended action for the current state and offers buttons that open
the section that performs it.

## 5. Datasets and CSV files

Each dataset is a 2D feature array of shape
`(samples, features)` plus an optional 2D binary-target array of
shape `(samples, outputs)`. The editor shows one sample per row with
`feature_0, feature_1, …` columns and a single trailing binary-target
column (`0` or `1` only).

- **Add Row / Remove Selected** edit the table; **Clear** empties it
  (with confirmation when edits are pending).
- **Validate & Apply** checks every cell: values must be numeric and
  finite — empty, non-numeric, `NaN`, and infinite entries are
  rejected with the cell location, and nothing is applied on failure.
- The feature count spin box and the "Include binary target column"
  checkbox define the table layout; changing them preserves fitting
  values after confirmation.
- **Import CSV…** reads a comma-separated UTF-8 file. The dialog
  options state the format explicitly: whether the first row is a
  header, and whether the **last column** is the binary target
  (`target_columns`). Exported files use headers `feature_0, …` and
  `target_0, …` at full float precision, so re-importing reproduces
  the values exactly. Malformed files (ragged rows, blank rows,
  non-numeric cells, non-finite values) are rejected with row/column
  locations, and the table is left untouched.
- **Export CSV…** validates first, then writes the file. Existing
  files are never overwritten silently: the storage API raises
  `FileExistsError` unless overwriting is explicitly authorized.
- The `data/datasets`, `data/experiments`, and `data/models`
  directories exist for saved artifacts.

## 6. Training and network configuration

Open **Training** to validate a labeled dataset (the view reads the
Dataset Management selection) and configure the network:

- Hidden layers as comma/space-separated sizes, e.g. `4, 3` (blank
  means no hidden layer). Default: `4`.
- Output size, default `1` (allowed 1–32).
- Learning rate, a finite number greater than zero. Default: `0.1`.
- Epoch count, default `100` (allowed 1–100000).
- Data split: how the dataset is divided into training, validation, and
  test rows (validation rows only choose the best epoch; test rows only
  evaluate the finished model).
- Input size is display-only and always comes from the dataset.

Open **Network Configuration** to set the whole architecture
(`input, hidden…, output`, e.g. `2, 4, 1`) directly on the same model.
Applying a different architecture replaces the current model and
discards its trained parameters, so it asks for confirmation;
applying the architecture already in use changes nothing.

Training runs in a background worker (`gui/workers/training_worker.py`)
using the same full-batch gradient-descent step as the core engine,
so the interface stays responsive. A progress bar and the status line
report the measured per-epoch training loss (`epoch k/N`); results
appear in three tabs — per-epoch loss readout, loss curve, and test
predictions — so they never have to share one squeezed column.

Two explicit actions start a run and a third stops it:

- **Train from scratch** builds a fresh network from the configured
  sizes; previous parameters are replaced and the status line says so.
- **Continue training** keeps the current parameters as the starting
  state. If the configured architecture no longer matches the model,
  training is refused with an error telling you to train from scratch
  instead — parameters are never silently reshaped or reset.
- **Cancel run** asks the running job to stop after its current epoch.
  Partial epochs are discarded and the previous model is left unchanged.

Only one run executes at a time — starting another while one is active
is rejected with a message. Closing the window mid-run offers the same
cancellation. **Save Model** and **Load Model** in the *Model state*
card write and read a JSON model-state file; loading replaces the model
used by every other section and asks for confirmation first.

## 7. Testing and evaluation

Open **Testing**:

- **Single-sample prediction** needs no targets: type one value per
  feature and press **Predict**. The view shows the predicted
  probability and the binary class (threshold 0.5). Invalid input is
  reported and your typed values are preserved.
- **Evaluate labeled dataset** scores the selected dataset read-only:
  per-sample predicted probabilities, predicted classes, actual
  targets, and errors appear in a table; the metrics line reports
  test MSE and accuracy at threshold 0.5. Prediction-only (unlabeled)
  input shows predictions with an explicit "metrics unavailable"
  notice instead of fabricated numbers.
- A labeled evaluation is plotted as predicted-versus-actual values
  per sample index; without targets only probabilities and classes
  are plotted. Charts always reflect recorded results.
- Testing never modifies weights or biases and never triggers
  training. Binary targets other than `0.0`/`1.0` are rejected with a
  clear message.

## 8. Weights and biases

Open **Weights and Biases** to inspect the shared live model:

- A layer selector lists every layer as `Layer l: n_l → n_l+1` with
  weight shape `(n_l, n_(l+1))` and bias shape `(1, n_(l+1))`. Tables
  show each value with its input/output indices.
- Editing a cell and pressing **Apply edits** validates **all**
  layers first (numeric, finite, exact shapes); any failure rejects
  the whole edit so the model is never partially modified. Accepted
  edits report which layers changed.
- A persistent notice states that manual edits change predictions:
  evaluation and continued training will use the edited parameters.
- **Refresh from model** reloads the display after training or model
  loading (with confirmation while edits are pending); switching
  layers never loses typed values.

## 9. Experiment history

Open **Experiment History**, then **Browse…** to a folder of saved
experiment-record JSON files (or **Refresh** to reload):

- The table lists each record: id, timestamp, mode, layers, learning
  rate, epochs, final train loss, test MSE/accuracy (shown as "not
  recorded" when absent — never as zero), and dataset names.
- Selecting a row shows the complete record in the **Record detail**
  tab, including the per-epoch measured loss history.
- Files that fail to load are skipped with their filenames and
  reasons; valid records still appear.
- Select exactly two rows and press **Compare selected** for a
  side-by-side table of configuration, settings, metrics, and deltas
  in the **Comparison** tab (`second − first`, each labeled with which
  direction is better).
  Only compatible experiments compare: same layer sizes, same
  training-data identity, and same test dataset. Anything else
  produces the exact mismatch reason instead of a comparison.
- This view never writes, recalculates, or deletes records.

## 10. Saving and loading models and experiments

Persistence is provided by the storage layer as versioned JSON
objects in UTF-8:

- Model state: `{"format": "neural-network-lab/model-state",
  "version": 1, "payload": {"layer_sizes": [...], "weights": [...],
  "biases": [...]}}`, saved/loaded with
  `save_model_state` / `load_model_state`.
- Experiment record: `{"format":
  "neural-network-lab/experiment-record", "version": 1, "payload":
  {...}}`, saved/loaded with `save_experiment_record` /
  `load_experiment_record` and rebuilt via
  `ExperimentRecord.from_dict`, which fully validates every field.

Only format `.../model-state` / `.../experiment-record` version `1`
files are read; anything else is rejected with the reason. Payloads
are structurally validated (shapes, required keys, finite numbers)
before any model object is built, and a loaded model reproduces the
saved predictions exactly. Saving refuses to overwrite an existing
file unless `overwrite=True` is passed explicitly. There are
currently no GUI buttons for saving/loading models or records —
persistence is used programmatically and by the experiments browser.

## 11. How the learning works (educational summary)

The network is a feedforward multilayer perceptron for binary
classification, implemented directly in NumPy:

- **Forward propagation**: each layer computes `Z = A·W + b` and
  applies the sigmoid `σ(z) = 1 / (1 + e^(−z))`, whose derivative is
  `σ(z)·(1 − σ(z))`, layer by layer up to the output.
- **Loss**: mean squared error, `MSE = mean((targets − predictions)²)`.
- **Backpropagation**: output error is propagated backward through
  the pre-update weights (chain rule), producing one gradient per
  weight matrix and bias vector with matching shapes.
- **Full-batch gradient descent**: every epoch uses the whole
  dataset at once, updating `W −= learning_rate · dW` (and likewise
  biases) with a measured post-update MSE recorded per epoch.
- **Modes**: from-scratch re-initializes parameters explicitly
  before training; continued training starts from the current
  parameters and never resets them silently.
- **Evaluation** thresholds probabilities at 0.5 (`1` if ≥ 0.5 else
  `0`) for accuracy; testing only reads parameters.

## 12. Known limitations

- The Dashboard only summarizes state owned by other sections; it does
  not create, modify, or refresh anything by itself (it re-reads the
  state each time it becomes visible).
- Experiment records are read through the Experiment History browser.
  There is no GUI action to delete or overwrite a saved record, and the
  model/experiment files written automatically by a completed training
  run are never overwritten (unique filenames are used).
- No testing-loss-over-time chart: only the recorded per-epoch training
  MSE (and validation MSE, when a validation subset was used) is
  plotted over epochs.
- Binary classification only: targets must be `0`/`1`, single-output
  charts only, sigmoid + MSE + full-batch gradient descent only
  (no Adam, CNNs/RNNs, PyTorch/TensorFlow, cloud, or accounts).
- The decision-boundary chart needs exactly two input features and one
  output; it explains what is missing instead of drawing a partial
  picture.
- Training is one run at a time in a background thread; a run is
  cancelled with **Cancel run** or by closing the window.
- This project is for educational experimentation — its outputs are
  not evidence that a model suits high-stakes decisions.

## 13. Project layout for developers

- `src/neural_network_lab/core/` — math engine (`network.py`,
  `activations.py`, `losses.py`, `backpropagation.py`,
  `optimizers.py`); never imports PySide6, Matplotlib widgets, or
  application code.
- `src/neural_network_lab/application/` — workflows
  (`experiment_service.py` with `run_experiment`, comparison, and an
  in-memory store; `evaluation_service.py` with `evaluate`;
  `training_service.py` is currently an empty placeholder).
- `src/neural_network_lab/data/` — `Dataset` validation and CSV
  import/export; `src/neural_network_lab/storage/` — JSON
  persistence with validation.
- `src/neural_network_lab/gui/` — `main_window.py` (navigation,
  unsaved-edit and training-aware close guards), `views/`,
  `widgets/` (including `charts.py`), `workers/`
  (`training_worker.py`).
- `src/neural_network_lab/gui/widgets/ui_theme.py` — the shared design
  tokens and the application-wide Qt style sheet; `page_scaffold.py` —
  the reusable page, card, feedback, and button components every view is
  built from. Both are presentation-only.
- `tests/` — automated suite run with `python -m pytest`.
- Key invariants enforced in code and tests: evaluation never
  modifies parameters; training never touches test data; shapes are
  validated at every boundary; metrics are measured, never invented.
