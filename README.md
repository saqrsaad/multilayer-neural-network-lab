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

The initial version focuses on a feedforward multilayer perceptron for binary classification.

The initial educational configuration uses:

- Sigmoid activation.
- Mean Squared Error (MSE).
- Backpropagation.
- Full-batch gradient descent.
- User-configurable learning rate and epoch count.

The architecture must support configurable layer sizes within the constraints defined in "docs/PROJECT_SPEC.md".

Advanced architectures and optimizers are outside the initial scope.

Development Setup

Environment preparation and dependency installation are performed manually by the project owner.

AI coding agents must not install dependencies, extensions, or tools.

After the environment is prepared, follow the project's development instructions and task specifications.

Project Documentation

- "docs/PROJECT_SPEC.md": Functional requirements and project boundaries.
- "docs/ARCHITECTURE.md": Component responsibilities and dependencies.
- "AGENTS.md": Mandatory instructions for coding agents.
- "docs/DEVELOPMENT_WORKFLOW.md": Task execution and verification rules.
- "docs/TASKS_INDEX.md": Task list and implementation status.

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

See "docs/USER_GUIDE.md" for setup details, the GUI workflow, dataset
formatting, training and evaluation, experiment history, and known
limitations.

Educational Disclaimer

This project is intended for educational experimentation and visualization.

Its outputs should not be treated as evidence that a model is suitable for high-stakes decisions.

License

The project license will be specified by the repository owner before public distribution.