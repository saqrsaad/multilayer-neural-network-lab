System Architecture

1. Architectural Principles

The application must follow a modular architecture with clear separation of concerns.

Core mathematics must remain independent of the GUI.

Data handling, application workflows, persistence, and presentation must have separate responsibilities.

Avoid unnecessary abstractions and premature complexity.

2. Main Layers

2.1 GUI Layer

Location: "src/neural_network_lab/gui/"

Responsibilities:

- Display application pages and controls.
- Collect user input.
- Display validation messages.
- Display predictions, metrics, tables, and charts.
- Send user actions to the application layer.

The GUI must not implement backpropagation or directly update weights.

2.2 Application Layer

Location: "src/neural_network_lab/application/"

Responsibilities:

- Coordinate training and testing workflows.
- Validate operation requests.
- Manage model lifecycle operations.
- Coordinate experiment creation and comparison.
- Request persistence operations.
- Return structured results to the GUI.

This layer connects the GUI to the core engine.

2.3 Core Layer

Location: "src/neural_network_lab/core/"

Responsibilities:

- Represent the neural network.
- Implement activation functions.
- Implement loss calculations.
- Implement forward propagation.
- Implement backpropagation.
- Implement parameter updates.
- Manage model parameters and training configuration.

The core layer must not import PySide6, Matplotlib GUI components, or application-specific widgets.

Core operations should be independently testable.

2.4 Data Layer

Location: "src/neural_network_lab/data/"

Responsibilities:

- Represent datasets.
- Validate input features and target values.
- Import and export CSV files.
- Prepare arrays for the neural network.
- Report invalid data clearly.

The data layer must not train the network or update model parameters.

2.5 Storage Layer

Location: "src/neural_network_lab/storage/"

Responsibilities:

- Save and load model states.
- Save and load experiment records.
- Preserve relevant metadata.
- Validate persisted content before accepting it.

Loading a model must not silently alter its parameters or architecture.

2.6 Visualization Components

Location: "src/neural_network_lab/gui/widgets/"

Responsibilities:

- Display loss histories.
- Display prediction comparisons.
- Display weight and bias tables.
- Display experiment comparisons.

Visualization components must use recorded results rather than inventing or estimating missing metrics.

2.7 Shared Presentation Components

Location: "src/neural_network_lab/gui/widgets/ui_theme.py" and "src/neural_network_lab/gui/widgets/page_scaffold.py"

Responsibilities:

- Own the design tokens (spacing, margins, colours, window sizes) and the application-wide Qt style sheet.
- Provide the reusable page, section card, feedback, and button components that every view is built from, so presentation is defined once instead of separately per page.

These modules contain no application state and no mathematics. Views compose them; they never import views.

3. Dependency Direction

The intended dependency direction is:

GUI -> Application -> Core

Application -> Data

Application -> Storage

GUI -> Visualization Widgets

GUI -> Shared Presentation Components

Visualization widgets may receive structured result data from the GUI or application layer.

The core layer must not depend on the GUI, storage implementation, or application workflow.

Avoid circular imports.

4. Core Model Responsibilities

The model must expose clear operations for:

- Forward prediction.
- Training for a specified number of epochs.
- Accessing model parameters.
- Validating parameter shapes.
- Applying validated parameter edits.
- Exporting and restoring model state.

The application layer decides whether training starts from scratch or continues from current parameters.

The core training implementation must not silently reset parameters during continued training.

5. Training and Evaluation Isolation

Training is allowed to update weights and biases.

Evaluation is read-only with respect to model parameters.

The application should preserve a snapshot of model parameters before evaluation in tests to verify that evaluation does not change them.

Training and evaluation must use explicitly selected datasets.

The testing dataset must never be passed into the training update process.

6. Parameter Shape Conventions

For a network with layer sizes:

[n0, n1, ..., nL]

the weight matrix connecting layer l to layer l+1 has shape:

(n_l, n_(l+1))

The corresponding bias vector has shape:

(1, n_(l+1))

Inputs use shape:

(number_of_samples, number_of_features)

Targets use shape:

(number_of_samples, number_of_outputs)

Use consistent shapes throughout the application.

Validate shapes at component boundaries.

7. Training Results

Training operations should return structured results containing relevant information such as:

- Number of completed epochs.
- Training loss history.
- Training configuration.
- Final training metrics.
- Operation status.

Evaluation operations should return structured results containing:

- Predictions.
- Targets, when available.
- Loss metrics, when calculable.
- Classification metrics, when applicable.
- Dataset identification.

Avoid passing unstructured dictionaries across the entire application when a small, clearly defined result model is more appropriate.

8. Background Execution

Long-running training must not block GUI event processing.

Use an appropriate Qt background execution mechanism when integrating training with the GUI.

The core training code must remain independent of Qt.

The GUI must receive progress and completion information safely.

9. Persistence

Model state and experiment history must be stored separately.

Persisted model states must include enough metadata to validate architecture and parameter dimensions during loading.

Experiment records must identify the model state and datasets associated with their results.

Do not overwrite previous experiment records without explicit user intent.

10. Error Handling

Expected input errors should be represented clearly and shown to users in understandable language.

Unexpected errors must not be silently ignored.

Do not use broad exception handling as a substitute for validation.

Keep diagnostic information available to developers without exposing unnecessary internal details in the GUI.

11. Testing Boundaries

Core tests must run without launching the GUI.

Application tests should verify training, evaluation, and persistence workflows.

GUI tests should focus on user interaction and presentation behavior where practical.

Critical invariants must have explicit tests.

12. Extension Policy

New features must be added to the appropriate layer.

Do not introduce a new framework or architectural layer without a clear requirement.

Update this document when a meaningful architectural decision changes.