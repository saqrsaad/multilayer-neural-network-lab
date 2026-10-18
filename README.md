Neural Network Lab

An interactive desktop application for learning, experimenting with, and visualizing multilayer neural networks implemented with NumPy.

The project focuses on understanding neural network fundamentals through practical experiments rather than relying on deep learning frameworks.

Project Goals

- Implement a multilayer perceptron (MLP) using NumPy.
- Demonstrate forward propagation, loss calculation, backpropagation, and gradient descent.
- Allow users to enter and edit training and testing datasets through a graphical interface.
- Train a model, test it on different datasets, and continue training without automatically resetting its parameters.
- Allow users to inspect and manually edit weights and biases.
- Visualize training loss, testing loss, predictions, and parameter changes.
- Save and restore model states and experiment records.
- Provide clear documentation for users, developers, and AI coding agents.

Technology Stack

- Python
- NumPy
- PySide6
- Matplotlib
- Pandas
- pytest

The neural network's core mathematical operations must be implemented directly with NumPy.

Project Scope

The initial version focuses on a feedforward multilayer perceptron. The default educational configuration is binary classification using:

- Sigmoid activation.
- Mean Squared Error (MSE).
- Backpropagation.
- Full-batch gradient descent.
- User-configurable learning rate and epoch count.

The implementation also supports continuous-target regression (arbitrary finite targets, evaluated with MSE/MAE/RMSE without class accuracy) and multi-output models (one output per target column). Charts and prediction surfaces display one selected output at a time where applicable. Configurable layer sizes are supported within the constraints defined in "docs/PROJECT_SPEC.md".

Advanced architectures and optimizers (e.g. CNNs/RNNs, Adam) are outside the initial scope.

Development Setup

Environment preparation and dependency installation are performed manually by the project owner.

AI coding agents must not install dependencies, extensions, or tools.

After the environment is prepared, follow the project's development instructions and task specifications.

Project Documentation

- `docs/PROJECT_SPEC.md`: Functional requirements and project boundaries.
- `docs/ARCHITECTURE.md`: Component responsibilities and dependencies.
- `docs/DEVELOPMENT_WORKFLOW.md`: Task execution and verification rules.
- `docs/USER_GUIDE.md`: Setup, GUI workflow, dataset formatting, training and evaluation, experiment history, persistence formats, and known limitations.
- `docs/algorithms.md`: Mathematical reference for the implemented algorithms (forward propagation, MSE, backpropagation, gradient descent, initialization, best-epoch selection, dataset splitting).
- `docs/persistence.md`: Model-state and experiment-record JSON formats, validation, save/load flow, and file locations.
- `docs/gui-pages.md`: Page-by-page reference to the GUI (Dashboard, Dataset Management, Network Configuration, Training, Testing, Weights & Biases, Experiment History, Visualization).
- `docs/file-structure.md`: Repository and source-code layout, module responsibilities, and dependency direction.
- `docs/dataset-csv.md`: Dataset data model, task types, validation, and CSV import/export behavior.
- `docs/testing-coverage.md`: Test suite organization and coverage, based on static inspection; tests have not been executed in this documentation phase.
- `AGENTS.md`: Mandatory instructions for coding agents.
- `docs/TASKS_INDEX.md`: Task list and implementation status (currently empty).

Running the Application

Requires Python 3.12 or newer with the runtime dependencies from
`pyproject.toml` installed (NumPy, PySide6, Matplotlib, Pandas).
Install the project and its development tools with:

```text
python -m pip install -e .
python -m pip install -e ".[dev]"
```

Launch the desktop application with either of these commands run from
the repository root:

```text
python -m neural_network_lab
neural-network-lab
```

Run the automated test suite from the repository root with:

```text
python -m pytest
```

The 18 test files under `tests/` have been reviewed statically in this documentation phase but have not been executed; runtime behavior and test results are unverified. See `docs/testing-coverage.md` for coverage details.

See "docs/USER_GUIDE.md" for setup details, the GUI workflow, dataset
formatting, training and evaluation, experiment history, and known
limitations.

Educational Disclaimer

This project is intended for educational experimentation and visualization.

Its outputs should not be treated as evidence that a model is suitable for high-stakes decisions.

License

The project license will be specified by the repository owner before public distribution.