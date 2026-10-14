T003 — Implement Forward Propagation
Objective
Implement forward propagation for a configurable feedforward multilayer perceptron using NumPy.
Required Reading
- "AGENTS.md"- "docs/PROJECT_SPEC.md"- "docs/ARCHITECTURE.md"- The existing activation function implementation.
Scope
1. Implement a network representation based on layer sizes.2. Store weight matrices and bias arrays with consistent shapes.3. Implement parameter initialization.4. Implement forward propagation across all layers.5. Return the final predictions.6. Make intermediate layer activations available when needed for backpropagation.7. Validate input dimensions and parameter shapes.8. Add unit tests.
For layer sizes:
[n0, n1, ..., nL]
the weight matrix for layer l must have shape:
(n_l, n_(l+1))
The corresponding bias array must have shape:
(1, n_(l+1))
For a batch of inputs:
X.shape == (number_of_samples, n0)
The final output must have shape:
(number_of_samples, nL)
Use sigmoid activation for all layers in this initial implementation.
Constraints
- Do not implement backpropagation or training.- Do not introduce PyTorch or TensorFlow.- Do not depend on the GUI.- Use NumPy matrix operations.- Do not silently reshape incompatible input data.- Preserve a consistent layer and parameter ordering.
Acceptance Criteria
- A network can be initialized from valid layer sizes.- Weight and bias dimensions follow the documented conventions.- Forward propagation returns correctly shaped outputs.- A manually calculated small example matches the implementation.- Invalid layer sizes and input dimensions are rejected clearly.- Tests pass.
Completion Report
Report the network interface, parameter conventions, changed files, tests executed, and any unresolved issues.