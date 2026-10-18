# File Structure Reference

This document describes the actual repository and source-code structure of
`multilayer-neural-network-lab`. It focuses on making the project layout
and file relationships easy to navigate. Every statement is grounded in
the current source tree; no runtime behavior is claimed.

## Source files consulted

- `pyproject.toml` — project metadata, dependencies, console script, test
  and lint configuration.
- `src/neural_network_lab/main.py` — application entry point.
- `src/neural_network_lab/__main__.py` — `python -m neural_network_lab` shim.
- `src/neural_network_lab/core/` — neural-network math engine.
- `src/neural_network_lab/application/` — application workflows.
- `src/neural_network_lab/data/` — dataset representation and CSV I/O.
- `src/neural_network_lab/gui/` — desktop interface.
- `src/neural_network_lab/storage/` — JSON persistence.
- `tests/` — automated test suite.
- `docs/` — project documentation.
- `data/` — runtime artifact directories (datasets, experiments, models).

Existing design context:
- `docs/ARCHITECTURE.md` — intended layer responsibilities and dependency
  direction (§2, §3).
- `docs/PROJECT_SPEC.md` — functional requirements and acceptance criteria.

## 1. Repository root

| Path | Purpose |
|---|---|
| `pyproject.toml` | Project metadata, runtime/dev dependencies, console script entry point, pytest/ruff/mypy configuration. |
| `README.md` | Project overview, goals, technology stack, setup, running instructions, documentation references. |
| `AGENTS.md` | Mandatory instructions for coding agents (scope, environment, file management, testing, documentation). |
| `docs/` | Project documentation (spec, architecture, workflow, user guide, algorithms, persistence, GUI pages, task specs). |
| `src/` | Source tree, laid out under `src/` for `setuptools` package discovery (`where = ["src"]`). |
| `tests/` | Automated test suite (18 files), run with `python -m pytest` from the repository root. |
| `data/` | Runtime artifact directories: `datasets/`, `experiments/`, `models/`. These are created by the application at runtime, not tracked as source. |
| `.venv/` | Python virtual environment (managed manually by the project owner; not part of the source tree). |
| `notes.txt`, `notes1.txt` | User-authored notes in Arabic about dependency installation and documentation requests. Not project documentation. |

Note: `docs/TASKS_INDEX.md` exists but is empty (no content). Individual task specs live in `docs/tasks/` (T001–T029 and `E-ux-ui-refinement.md`).

## 2. Source package layout

All importable code lives under `src/neural_network_lab/`. The package is
discovered by setuptools from `where = ["src"]` in `pyproject.toml`.

```
src/neural_network_lab/
├── __init__.py          (empty)
├── __main__.py          (python -m neural_network_lab shim -> main.main)
├── main.py              (application entry point: main())
├── core/
│   ├── __init__.py      (empty)
│   ├── activations.py   (sigmoid, sigmoid_derivative)
│   ├── losses.py        (mean_squared_error, mse alias)
│   ├── network.py       (NeuralNetwork: forward, train, train_from_scratch, parameters, state)
│   ├── backpropagation.py (compute_gradients, validate_targets)
│   ├── optimizers.py    (validate_learning_rate, sgd_update)
│   └── training.py      (train_with_validation, TrainingOutcome)
├── application/
│   ├── __init__.py      (empty)
│   ├── evaluation_service.py (evaluate, EvaluationResult)
│   ├── experiment_service.py (ExperimentRecord, ExperimentStore, compare_experiments, run_experiment)
│   └── training_service.py (EMPTY — 0 lines; documented placeholder)
├── data/
│   ├── __init__.py      (empty)
│   ├── dataset.py       (Dataset, BINARY_TARGETS, CONTINUOUS_TARGETS)
│   ├── dataset_split.py (split_dataset, DatasetSplits)
│   └── csv_handler.py   (import_dataset_from_csv, export_dataset_to_csv)
├── gui/
│   ├── __init__.py      (empty)
│   ├── main_window.py   (MainWindow, navigation, view registration)
│   ├── views/
│   │   ├── __init__.py
│   │   ├── dashboard_view.py
│   │   ├── dataset_view.py
│   │   ├── experiments_view.py
│   │   ├── network_configuration_view.py
│   │   ├── parameters_view.py
│   │   ├── testing_view.py
│   │   ├── training_view.py
│   │   └── visualization_view.py
│   ├── widgets/
│   │   ├── __init__.py
│   │   ├── charts.py       (LossHistoryChart, PredictionChart, DecisionBoundaryChart, RegressionSurfaceChart, ArchitectureChart)
│   │   ├── dataset_table.py (DatasetTable, build_dataset_from_grid, parse_float_cell)
│   │   ├── network_config_widget.py (NetworkConfigWidget)
│   │   ├── page_scaffold.py (install_page, SectionCard, FeedbackArea, MessageLabel, make_button, button_row)
│   │   ├── training_controls.py (TrainingControls)
│   │   └── ui_theme.py     (design tokens, application stylesheet)
│   └── workers/
│       ├── __init__.py
│       └── training_worker.py (TrainingWorker: QObject, background training)
└── storage/
    ├── __init__.py      (empty)
    ├── _json_files.py   (coerce_path, check_overwrite_allowed, write/read_json_document, check_envelope)
    ├── model_storage.py (save_model_state, load_model_state)
    └── experiment_storage.py (save_experiment_record, load_experiment_record)
```

All `__init__.py` files in the package are empty (0 lines).

### Responsibilities by layer (as implemented)

**`core/`** — The neural-network math engine. Implements activations, loss,
forward propagation, backpropagation, parameter updates, training modes,
and initialization using NumPy only. It does not import PySide6,
Matplotlib GUI components, or application code, consistent with
`docs/ARCHITECTURE.md` §2.3.

**`application/`** — Workflows that coordinate core, data, and storage:
evaluation (`evaluate`), experiment management (`ExperimentRecord`,
`ExperimentStore`, `compare_experiments`, `run_experiment`). Note:
`training_service.py` exists as an empty file (0 lines); the training
workflow is currently implemented in `gui/views/training_view.py`
calling core directly, which deviates from the intended
`GUI -> Application -> Core` layering (see §6 below).

**`data/`** — Dataset representation (`Dataset`), train/validation/test
splitting (`split_dataset`), and CSV import/export
(`import_dataset_from_csv`, `export_dataset_to_csv`). It does not train
the network or update model parameters.

**`gui/`** — The desktop interface: main window with navigation, eight
view pages, reusable widgets (charts, dataset table, network config
widget, page scaffold, training controls, UI theme), and the background
training worker.

**`storage/`** — JSON persistence: model states and experiment records,
with envelope format/version validation, finiteness sweeps, and overwrite
protection.

## 3. Dependency direction

The intended dependency direction, per `docs/ARCHITECTURE.md` §3, is:

```
GUI -> Application -> Core
Application -> Data
Application -> Storage
GUI -> Visualization Widgets
GUI -> Shared Presentation Components
```

The core layer must not depend on the GUI, storage, or application
workflow, and there must be no circular imports. This is respected in
the current source: `core/` modules import only NumPy and standard
library; `data/` imports only NumPy; `storage/` imports only
`_json_files` (no domain imports).

## 4. Application entry point

There are two equivalent entry points that share one code path:

- **Console script** declared in `pyproject.toml`:
  `neural-network-lab = "neural_network_lab.main:main"`.
- **Module execution**: `python -m neural_network_lab`, which resolves to
  `src/neural_network_lab/__main__.py` -> `main.main()`.

`main()` in `src/neural_network_lab/main.py`:
1. Creates (or reuses) the shared `QApplication` via
   `create_application()`.
2. Creates the main window via `create_main_window()`.
3. Registers the default (implemented) views via
   `register_default_views()`.
4. Shows the window and runs `app.exec()`.

Source:
- `src/neural_network_lab/main.py`
- `src/neural_network_lab/__main__.py`

## 5. GUI navigation structure

`MainWindow` (`gui/main_window.py`) shows a navigation panel
(`QListWidget`) beside a resizable stacked content area (`QStackedWidget`).
The navigation entry of the visible page stays highlighted.

Seven sections are declared in `VIEW_SPECS`, matching the documented GUI
sections in `docs/PROJECT_SPEC.md` §12:

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

Views whose real widget is implemented are registered via
`register_view()`; sections without a real view show a clearly labeled
`PlaceholderView` instead of crashing. Navigation is guarded by
unsaved-edit confirmation (`has_unsaved_changes()`) and, on window close,
by a prompt when a training run is active.

## 6. Data flow between layers

### Training workflow

1. **GUI** (`training_view.py`): collects settings and the selected
   dataset from the dataset editor provider; validates inputs; calls
   `split_dataset()` (data layer) to divide rows into train/validation/
   test subsets.
2. **GUI** (`training_worker.py`): runs `train_with_validation()` (core)
   in a background `QThread` so the interface stays responsive; emits
   progress/completion/failure signals.
3. **Core** (`training.py`, `network.py`): performs full-batch gradient
   descent on training rows only; scores validation rows read-only after
   each epoch; restores the best-epoch snapshot on completion.
4. **GUI** (`training_view.py`): on the GUI thread, installs the trained
   state, evaluates the held-out test subset read-only via
   `evaluate()` (application layer), draws history/predictions, and
   automatically saves the model state and experiment record.

Training data never reaches the test subset for updates or model
selection. The core training code remains independent of Qt.

### Evaluation workflow

**GUI** (`testing_view.py`) -> **Application** (`evaluation_service.py`,
`evaluate()`) -> **Core** (`network.py`, `forward()` + `losses.py`,
`mean_squared_error()`). Evaluation is strictly read-only: forward
propagation only, no parameter updates.

### Persistence workflow

**GUI** (`training_view.py`): on run completion, saves model state via
`save_model_state()` and experiment record via
`save_experiment_record()` (storage layer) to `data/models/` and
`data/experiments/` with generated UUID filenames.

**GUI** (`experiments_view.py`): loads experiment records from a user-
chosen directory via `load_experiment_record()` + `ExperimentRecord
.from_dict()` (storage + application layers); compares compatible pairs
via `compare_experiments()` (application layer).

### Core model lifecycle

`NeuralNetwork` (`core/network.py`) exposes forward prediction, training
(`train`, `train_from_scratch`), parameter access, validated parameter
edits (`set_parameters`), and state export/restore
(`to_state_dict`/`from_state_dict`). The application layer decides
whether training starts from scratch or continues from current
parameters.

## 7. Differences between intended architecture and current implementation

The intended architecture (`docs/ARCHITECTURE.md` §3) specifies the
dependency direction `GUI -> Application -> Core`, with training
workflow logic owned by the Application layer. The current
implementation deviates in one concrete respect:

- **`application/training_service.py` is empty (0 lines).** It is
  documented as "currently an empty placeholder" in
  `docs/USER_GUIDE.md` line 314, but no application service hosts the
  training workflow. Instead, `gui/views/training_view.py` (GUI layer)
  imports and calls core functions directly
  (`NeuralNetwork`, `train_with_validation`, `evaluate`, storage
  functions), bypassing the Application layer. This is a layering
  deviation, not a correctness defect: the workflow functions correctly,
  but training logic is not where the architecture intends it to live.
  Correcting it would require moving training workflow logic from the
  GUI into `application/training_service.py`, which is beyond the scope
  of a documentation-only change.

All other layers respect the intended dependency direction: `core/`
depends only on NumPy; `data/` depends only on NumPy; `storage/` depends
only on `_json_files`; `application/` depends on `core/` and `data/`;
and `gui/` depends on `application/`, `core/`, `data/`, and `storage/`
through its views and worker.

## 8. Tests and documentation placement

- `tests/` — 18 test files mirroring the source modules (activations,
  losses, network, backpropagation, training, evaluation, experiments,
  storage, CSV handler, dataset, dataset editor, main window, training
  view, testing view, parameters view, visualizations, background
  training, experiments view). Configured in `pyproject.toml` with
  `testpaths = ["tests"]` and `pythonpath = ["src"]`.
- `docs/` — `PROJECT_SPEC.md`, `ARCHITECTURE.md`,
  `DEVELOPMENT_WORKFLOW.md`, `USER_GUIDE.md`, `TASKS_INDEX.md` (empty),
  `tasks/` (individual task specs T001–T029 and `E-ux-ui-refinement.md`),
  `algorithms.md`, `persistence.md`, `gui-pages.md`, and this file.

## 9. What cannot be established through static inspection

- Whether the application launches and the GUI renders under Qt/PySide6.
- Whether the `data/models/` and `data/experiments/` directories exist
  or are writable in a given environment.
- Whether auto-saved experiment/model files load correctly at runtime.
- Test-suite results (tests were reviewed as code, not executed).
- Runtime signal/slot behavior between GUI providers and page updates.
