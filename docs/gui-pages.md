# GUI Pages Reference

This document is a page-by-page reference to the eight implemented GUI
pages of `multilayer-neural-network-lab`. For each page it explains
purpose, controls and actions, required inputs, resulting outputs, and
data flow. Every statement is grounded in the current source code; no
runtime behavior is claimed.

## Source files

- `../src/neural_network_lab/gui/main_window.py` — `MainWindow`,
  navigation panel, stacked pages, unsaved-edit and close guards.
- `../src/neural_network_lab/gui/views/dashboard_view.py` — Dashboard.
- `../src/neural_network_lab/gui/views/dataset_view.py` — Dataset
  Management.
- `../src/neural_network_lab/gui/views/network_configuration_view.py` —
  Network Configuration.
- `../src/neural_network_lab/gui/views/training_view.py` — Training.
- `../src/neural_network_lab/gui/views/testing_view.py` — Testing.
- `../src/neural_network_lab/gui/views/parameters_view.py` — Weights &
  Biases.
- `../src/neural_network_lab/gui/views/experiments_view.py` — Experiment
  History.
- `../src/neural_network_lab/gui/views/visualization_view.py` —
  Visualization.
- `../src/neural_network_lab/gui/widgets/charts.py` — chart widgets used
  by several pages.
- `../src/neural_network_lab/gui/widgets/page_scaffold.py` — shared page,
  card, feedback, and button components.
- `../src/neural_network_lab/gui/widgets/ui_theme.py` — design tokens and
  application stylesheet.

Existing design context:
- `../docs/ARCHITECTURE.md` §2.1 (GUI layer) and §12 (User Interface).
- `../docs/PROJECT_SPEC.md` §12 (User Interface) and §15 (Acceptance
  Criteria).
- `../docs/USER_GUIDE.md` §4–§10 (the broader user guide; this document
  links to it rather than duplicating it).

## 1. Window and navigation

The application window (`MainWindow`) shows a navigation panel beside a
resizable stacked content area. The navigation entry of the visible page
stays highlighted, and the window status bar names the current section.
The window can be resized down to 1000×640 pixels; long pages scroll
instead of clipping content.

### Navigation sections

Seven sections are defined in `VIEW_SPECS` and match the documented GUI
sections:

| Key | Title |
|---|---|
| `dashboard` | Dashboard |
| `datasets` | Dataset Management |
| `network` | Network Configuration |
| `training` | Training |
| `testing` | Testing |
| `parameters` | Weights and Biases |
| `experiments` | Experiment History |

An eighth page, **Visualization** (key `visualization`), is registered
by `register_default_views()` but is not part of `VIEW_SPECS`; it is
appended to the navigation list when default views are registered.

### Navigation behavior

- Selecting a navigation entry routes through `request_view(key)`, which
  guards against discarding unsaved edits: if the current page exposes
  `has_unsaved_changes()` and returns `True`, a confirmation dialog is
  shown before leaving.
- Closing the window while a training run is active asks whether to quit
  and cancel training; cancelled training leaves the previous model
  unchanged.
- Views whose real widget is not implemented show a labeled placeholder
  instead of crashing.

Source:
- `../src/neural_network_lab/gui/main_window.py` (`MainWindow`,
  `VIEW_SPECS`, `register_view`, `register_default_views`,
  `PlaceholderView`)
- `../src/neural_network_lab/gui/widgets/page_scaffold.py`
  (`install_page`, `SectionCard`, `FeedbackArea`, `MessageLabel`,
  `make_button`)

## 2. Dashboard

**Purpose:** Read-only live overview of dataset, network, training run,
and saved experiment state. Every value is read from the section that
owns it; nothing is estimated or invented.

**Controls:**
- **Refresh overview** button — re-reads every state source and redraws
  the cards.
- **Go to section** button — opens the page recommended for the current
  state.
- Six secondary section buttons (Dataset Management, Network
  Configuration, Training, Testing, Visualization, Experiment History) —
  open the corresponding page directly.
- **Last training run** table — read-only table of the last completed
  run's recorded fields (training mode, samples, features, outputs,
  architecture, learning rate, epochs, losses, validation, test metrics,
  saved artifacts).
- **Next step** card — names the recommended action for the current
  state.

**Required inputs:** None directly; it reads state through providers
injected by `MainWindow` (dataset editor's dataset, training view's
network and run details, experiments view's loaded-record state).

**Resulting outputs:** Status cards describing dataset, network,
dataset/network compatibility, training state, last run details, and
experiment load state. Missing values are reported as "—" or "— (not
recorded)" rather than estimated.

**Data flow:** Providers are called on `showEvent` and on refresh. The
dashboard computes nothing and stores nothing.

**Constraints:**
- The Dashboard summarizes state owned by other sections; it does not
  create, modify, or refresh anything by itself.
- Anything a run did not measure is marked "not recorded"; no metric is
  reused from an older run.

Source:
- `../src/neural_network_lab/gui/views/dashboard_view.py` (`DashboardView`)

## 3. Dataset Management

**Purpose:** Enter, edit, import, and export datasets through a table
editor. One sample per row; feature columns plus optional target
columns.

**Controls:**
- **Feature columns** spin box (1–32, default 2) and **Target columns**
  spin box (0–8; 0 means prediction-only).
- **Task** combo: Binary classification (targets 0 or 1) or Regression
  (continuous targets).
- **CSV header** checkbox (whether the first row of an imported CSV is a
  header).
- **Built-in dataset** selector and **Load built-in** button (XOR, AND,
  OR).
- Sample table: **Add Row**, **Remove Selected**, **Clear** (asks for
  confirmation while edits are pending).
- **Validate & Apply** (primary) — validates every cell and makes the
  table the dataset other sections use.
- **Import CSV…** and **Export CSV…** — file dialogs; failed imports or
  exports leave the table untouched.

**Required inputs:** At least one sample row; every cell must be numeric
and finite. For binary classification, every target cell must be 0.0 or
1.0. For regression, any finite value is accepted and preserved exactly.

**Resulting outputs:** A validated `Dataset` object exposed to other
sections via `current_dataset()`. A one-line summary of samples,
features, target columns, and task.

**Data flow:** `build_dataset_from_grid()` converts table text to a
`Dataset`; validation failures are reported with row/column location and
nothing is applied. Import uses `import_dataset_from_csv()` with
explicit target columns (the last N columns by default in the GUI).
Export writes `feature_0, …` and `target_0, …` headers at full float
precision.

**Constraints:**
- `MAX_FEATURES = 32`, `MAX_TARGET_COLUMNS = 8`.
- Changing the column layout asks for confirmation before discarding
  pending edits.
- The dataset editor's applied dataset is the single source of truth
  read by Training, Testing, Visualization, and Dashboard.
- Target columns are never merged, dropped, duplicated, or filled in.

Source:
- `../src/neural_network_lab/gui/views/dataset_view.py` (`DatasetView`,
  `BUILTIN_DATASETS`)
- `../src/neural_network_lab/gui/widgets/dataset_table.py`
  (`DatasetTable`, `build_dataset_from_grid`, `parse_float_cell`)
- `../src/neural_network_lab/data/csv_handler.py`
  (`import_dataset_from_csv`, `export_dataset_to_csv`)
- `../src/neural_network_lab/data/dataset.py` (`Dataset`,
  `BINARY_TARGETS`, `CONTINUOUS_TARGETS`)

## 4. Network Configuration

**Purpose:** Edit the shared model's architecture (`input, hidden…,
output`) directly. Training, Testing, Weights & Biases, and
Visualization all read this one model.

**Controls:**
- **Input size** — display-only, set by the selected dataset.
- Hidden layer rows: add, edit, or remove (1–256 neurons each; maximum
  8 hidden layers).
- **Output neurons** spin box (1–32).
- **Apply configuration** (primary) — builds the active model with the
  entered layers.
- Current architecture display and preview of the configuration to
  apply.

**Required inputs:** A dataset must be selected to fix the input size.
Hidden sizes and output size must be positive integers within bounds.

**Resulting outputs:** A fresh `NeuralNetwork` with Xavier-uniform
weights and zero biases (applied architecture replaces the current model
and discards its trained parameters).

**Data flow:** The view produces a layer-size tuple and calls the
injected applier (`TrainingView.apply_layer_sizes`), which constructs
`NeuralNetwork(sizes)`. Applying the architecture already in use is a
no-op and does not discard parameters.

**Constraints:**
- Applying a different architecture replaces the model and discards
  trained parameters; a confirmation dialog is shown first.
- The input size always comes from the dataset, never from user typing.

Source:
- `../src/neural_network_lab/gui/views/network_configuration_view.py`
  (`NetworkConfigurationView`, `parse_layer_sizes`,
  `describe_architecture`)
- `../src/neural_network_lab/core/network.py` (`NeuralNetwork`)

## 5. Training

**Purpose:** Configure architecture and training settings, then run
training in either mode from a background worker thread so the interface
stays responsive.

**Controls:**
- **Train from scratch** (primary) — builds a fresh network from
  configured sizes; previous parameters are replaced.
- **Continue training** (secondary) — keeps current parameters as the
  starting state; refuses if the configured architecture does not match
  the model.
- **Cancel run** — asks the running job to stop after its current
  epoch; partial epochs are discarded and the previous model is left
  unchanged.
- **Save Model** / **Load Model** — manual model-state file dialogs.
- Progress bar and status line reporting per-epoch training loss.
- Three result tabs: per-epoch loss readout, loss curve, and test
  predictions.

**Required inputs:** A labeled dataset selected from Dataset Management;
valid learning rate (> 0, finite); positive epoch count; compatible
dataset/network sizes.

**Resulting outputs:** Trained network applied in memory; loss history;
validation history (when a validation subset was used); best-epoch
restoration; test-set evaluation from the held-out test subset;
automatic saving of model state and experiment record.

**Data flow:**
1. `_collect_training_inputs()` validates the dataset, settings, and
   split; calls `split_dataset()` to divide rows into train/validation/
   test subsets.
2. `TrainingWorker` runs `train_with_validation()` in a background
   `QThread`; progress is emitted per epoch and throttled to at most 100
   GUI updates per run.
3. On completion, the GUI thread installs the trained state, evaluates
   the test subset read-only, draws history and predictions, saves the
   model and experiment record automatically.
4. Cancellation leaves the network unchanged (the worker trains on a
   copy of the parameters).

**Constraints:**
- Only one run executes at a time; a second run while one is active is
  rejected.
- Training never uses the testing dataset to update parameters.
- Validation rows are scored read-only and used only for best-epoch
  selection; test rows are used only for final evaluation.
- Continued training refuses to run if the architecture changed since
  the model was built (train from scratch instead).
- The GUI worker never receives a `seed`, so from-scratch training via
  the GUI is non-deterministic (the core `train_from_scratch` supports
  `seed`, but the GUI does not expose it).
- `training_service.py` is an empty placeholder (documented in
  `USER_GUIDE.md` line 314); training workflow logic lives in
  `training_view.py` and the core engine.

Source:
- `../src/neural_network_lab/gui/views/training_view.py` (`TrainingView`)
- `../src/neural_network_lab/gui/widgets/training_controls.py`
  (`TrainingControls`)
- `../src/neural_network_lab/gui/workers/training_worker.py`
  (`TrainingWorker`)
- `../src/neural_network_lab/core/training.py` (`train_with_validation`,
  `TrainingOutcome`)
- `../src/neural_network_lab/core/network.py` (`NeuralNetwork.train`,
  `train_from_scratch`)
- `../src/neural_network_lab/data/dataset_split.py` (`split_dataset`)

## 6. Testing

**Purpose:** Predict single hand-entered samples and evaluate labeled
datasets read-only. Testing never modifies weights or biases and never
triggers training.

**Controls:**
- **Single-sample prediction:** one text input per feature; **Predict**
  button; shows predicted probability and binary class (threshold 0.5)
  for single-output models, or raw output values for multi-output models.
- **Labeled dataset evaluation:** **Evaluate labeled dataset** button;
  output combo (shown only when there is a choice); per-sample results
  table; `PredictionChart`.

**Required inputs:** A network and a dataset. For labeled evaluation,
the dataset must have targets compatible with the network output size.

**Resulting outputs:** Prediction label, metrics line (test MSE, MAE,
RMSE; accuracy at threshold 0.5 for binary targets only), per-sample
table (predicted value, predicted class for binary, actual value,
error), and prediction chart for the selected output.

**Data flow:** `evaluate()` runs a forward pass and computes metrics
without modifying parameters. Results are remembered so the output
selector can redraw the chart. If the model or dataset changes since
the last evaluation, results are cleared on `showEvent`.

**Constraints:**
- Binary targets must be exactly 0.0 or 1.0; any other value is
  rejected.
- Continuous targets keep exact values; accuracy is not computed
  (reported as not applicable).
- Multi-output models: the table lists every output; the chart draws
  one selected output at a time and is skipped for multi-output models
  in the Training view's test-predictions tab.
- Prediction-only (unlabeled) input shows predictions with an explicit
  "metrics unavailable" notice instead of fabricated numbers.

Source:
- `../src/neural_network_lab/gui/views/testing_view.py` (`TestingView`)
- `../src/neural_network_lab/application/evaluation_service.py`
  (`evaluate`, `EvaluationResult`, `CLASSIFICATION_THRESHOLD`)
- `../src/neural_network_lab/gui/widgets/charts.py` (`PredictionChart`)

## 7. Weights & Biases

**Purpose:** Inspect and edit the shared live model's parameters, one
layer at a time.

**Controls:**
- **Layer** combo listing every layer as `Layer l: n_l → n_l+1` with
  weight shape `(n_l, n_{l+1})` and bias shape `(1, n_{l+1})`.
- Weights table and biases table showing each value with input/output
  indices.
- **Apply edits** (primary) — validates every edited cell across all
  layers and applies them atomically; any failure rejects the whole
  edit.
- **Refresh from model** — reloads the display from the current model
  (asks for confirmation while edits are pending).
- Persistent notice that manual edits change predictions.

**Required inputs:** A network must be available. Cell text must be
numeric and finite; table shapes must match the model's parameter
shapes exactly.

**Resulting outputs:** Updated model parameters; report of which layers
changed. Pending edits are preserved across layer switches.

**Data flow:** Edits are parsed and validated in full before
`network.set_parameters()` is called; a rejected edit leaves the model
completely unchanged. The view operates on the shared model supplied by
a provider, never on a duplicate.

**Constraints:**
- If the model architecture changed since the display was loaded, the
  user must refresh before editing.
- A rejected edit leaves the model untouched; accepted edits mark the
  model as modified.

Source:
- `../src/neural_network_lab/gui/views/parameters_view.py`
  (`ParametersView`, `parse_parameter_cell`, `format_parameter`)
- `../src/neural_network_lab/core/network.py`
  (`NeuralNetwork.set_parameters`, `weights`, `biases`)

## 8. Experiment History

**Purpose:** Browse saved experiment records and compare compatible
pairs. This page never writes, recalculates, overwrites, or deletes
records.

**Controls:**
- **Browse…** — choose a folder of saved experiment JSON files.
- **Refresh** — reload the records of the current folder.
- Records table (newest first by timestamp): id, created, mode, layers,
  learning rate, epochs, train loss, validation loss, test MSE, test
  accuracy, train data, test data.
- **Record detail** tab — full record including per-epoch loss history.
- **Compare selected** (primary, requires exactly two rows) — side-by
  side table of configuration, settings, metrics, and deltas
  (`second − first`, each labeled with which direction is better).

**Required inputs:** A directory containing experiment-record JSON files.
Files that fail to load are skipped and reported; valid records are
still listed.

**Resulting outputs:** Loaded records, load failures, and comparison
deltas. Missing metrics are shown as "— (not recorded)", never as zero.

**Data flow:** `load_experiment_record()` checks the envelope and
required fields; `ExperimentRecord.from_dict()` performs full semantic
validation. Compatibility rules for comparison require identical
`layer_sizes`, training-data identity, and test dataset name.

**Constraints:**
- This view never writes, recalculates, overwrites, or deletes records.
- Only compatible experiments compare; incompatibility produces the
  exact mismatch reason.
- Select exactly two rows to compare.

Source:
- `../src/neural_network_lab/gui/views/experiments_view.py`
  (`ExperimentsView`, `comparison_rows`)
- `../src/neural_network_lab/application/experiment_service.py`
  (`ExperimentRecord`, `ExperimentStore`, `compare_experiments`)
- `../src/neural_network_lab/storage/experiment_storage.py`
  (`load_experiment_record`, `save_experiment_record`)

## 9. Visualization

**Purpose:** Show what the current model has learned — decision
boundary (binary classification) or predicted-value surface (continuous
regression) for a two-feature projection — plus the network
architecture diagram.

**Controls:**
- **Feature on x** / **Feature on y** combo boxes (require ≥ 2 input
  features; the two chosen features must be different).
- **Output neuron** combo (shown only when the model has > 1 output).
- **Refresh visualizations** button.
- Architecture chart with zoom/pan toolbar.
- Decision boundary or predicted-value surface panel (switched by task
  type).

**Required inputs:** A network and a labeled dataset. For the boundary
panel: exactly two input features and one output. For the surface panel:
exactly two input features; continuous targets. Additional features are
held at their column mean, and this is disclosed in the projection note.

**Resulting outputs:** Class regions and decision boundary line (binary)
or predicted value surface with color scale and samples colored by
actual value (continuous); architecture diagram of layers, neurons, and
connections.

**Data flow:** `network.forward()` is called on a regular grid built
from the dataset's feature ranges; predictions are reshaped to match the
grid and drawn by `DecisionBoundaryChart` or `RegressionSurfaceChart`.
`ArchitectureChart.plot_architecture()` draws the model's actual
`layer_sizes`.

**Constraints:**
- The decision boundary is used only for labeled binary datasets;
  continuous targets produce a surface, never a boundary.
- The boundary needs exactly two input features and one output;
  otherwise an explanatory message is shown.
- Large architecture columns are summarized (up to 12 neurons drawn per
  column) with a caption stating the true count.
- The chart axes are labeled "feature 1"/"feature 2"; actual feature
  names are disclosed in the projection note below the chart.

Source:
- `../src/neural_network_lab/gui/views/visualization_view.py`
  (`VisualizationView`)
- `../src/neural_network_lab/gui/widgets/charts.py`
  (`DecisionBoundaryChart`, `RegressionSurfaceChart`,
  `ArchitectureChart`)

## 10. Cross-page workflows

### Dataset → model → training

1. **Dataset Management:** enter or import a dataset and **Validate &
   Apply**. The applied dataset is exposed via `current_dataset()`.
2. **Network Configuration** or **Training:** the input size is taken
   from the dataset; hidden layers and output size are configured to
   match the dataset's target columns.
3. **Training:** the selected dataset is split into train/validation/
   test subsets; training runs in the background; the trained model is
   applied in memory and automatically saved along with an experiment
   record.

### Continued training

1. Train from scratch (or load a model) to establish a network.
2. **Continue training** keeps the current parameters as the starting
   state. If the architecture was changed in the meantime, continued
   training is refused and the user is directed to train from scratch.

### Evaluation

1. **Testing:** select a labeled dataset and **Evaluate labeled
   dataset**. The evaluation is read-only and reports per-sample results
   and metrics; it never modifies the model.

### Experiment history

1. A completed training run automatically saves its model state and
   experiment record.
2. **Experiment History:** **Browse…** to the folder of saved records;
   select a row for full detail or exactly two rows to compare.

### Visualization

1. **Visualization:** select a network and dataset; choose two features
   and (for multi-output models) an output; the boundary or surface is
   drawn from the model's forward pass on a grid, with held features at
   their means disclosed.

### Dataset or task changes

- If the training data changes after a run, the Training view clears the
  previous results on `showEvent`.
- If the model or dataset changes after an evaluation, the Testing view
  clears the previous results on `showEvent`.
- The Visualization view redraws both panels from the current model and
  dataset whenever it becomes visible or the projection changes.

### Binary versus continuous targets

- **Binary classification:** targets must be 0.0 or 1.0; accuracy at
  threshold 0.5 is reported; decision boundary shows class regions;
  testing table shows predicted class.
- **Continuous regression:** any finite target value is preserved
  exactly; accuracy is not computed; testing table shows actual value
  and error; visualization shows a predicted-value surface.
- The dataset's task is declared when the dataset is built and is never
  guessed from values.

### Multi-output display

- The core, data, and evaluation layers fully support multiple outputs
  (each target column is one network output).
- Charts display one selected output at a time; tables always list every
  output.
- Multi-output models are never truncated to their first column in
  charts; the output selector is shown only when there is a choice.

## 11. Shared UI components

All pages are built from `page_scaffold.py` (`install_page`,
`SectionCard`, `FeedbackArea`, `MessageLabel`, `make_button`,
`button_row`) and styled by `ui_theme.py` (design tokens and the
application stylesheet). Every page has a title, a one-line explanation,
titled section cards, and a feedback area at the bottom holding status
and error lines that disappear while empty.

Source:
- `../src/neural_network_lab/gui/widgets/page_scaffold.py`
- `../src/neural_network_lab/gui/widgets/ui_theme.py`

## 12. Limitations

- Experiment records can be browsed and loaded, but there are no GUI
  buttons to manually save or load an experiment record (model save/load
  buttons exist in the Training view).
- Records are never deleted or overwritten through the GUI.
- Auto-saved artifacts use generated UUID filenames in fixed
  directories (`data/models/`, `data/experiments/`).
- The dashboard summarizes state only; it does not create, modify, or
  refresh anything.
- GUI rendering, Qt signal/slot behavior, file dialogs, and chart
  rendering under the Matplotlib QtAgg backend were not executed in this
  documentation session and cannot be confirmed as runtime behavior.

For broader usage instructions, setup, and known limitations, see
`../docs/USER_GUIDE.md`. For functional requirements, see
`../docs/PROJECT_SPEC.md`. For architecture and layering, see
`../docs/ARCHITECTURE.md`. For algorithms, see `../docs/algorithms.md`.
For persistence formats, see `../docs/persistence.md`.
