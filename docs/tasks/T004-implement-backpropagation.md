T004 — Implement Backpropagation and Parameter Updates
Objective
Implement backpropagation and full-batch gradient descent for the existing NumPy network.
Required Reading
- "AGENTS.md"- "docs/PROJECT_SPEC.md"- "docs/ARCHITECTURE.md"- The existing network, activation, and loss implementations.
Scope
1. Calculate output-layer deltas.2. Propagate deltas backward through hidden layers.3. Calculate weight and bias gradients.4. Update parameters using full-batch gradient descent.5. Validate target dimensions and learning-rate values.6. Add tests for gradients and parameter updates.
Use sigmoid activation and MSE.
For output activations A and targets Y, the MSE derivative with respect to the output activation is:
dL/dA = 2 * (A - Y) / (N * output_size)
The sigmoid derivative is:
sigma_prime(z) = A * (1 - A)
Use the chain rule to calculate gradients for all layers.
For a weight matrix W_l, gradient descent uses:
W_l = W_l - learning_rate * dL/dW_l
Biases must be updated using their corresponding gradients.
The implementation must be mathematically consistent with the defined MSE reduction and gradient convention.
Constraints
- Do not implement the GUI.- Do not add new dependencies.- Do not reset parameters during an update.- Calculate all required gradients using the same pre-update parameter state.- Do not update parameters until the gradients for all layers have been calculated.- Keep training independent of GUI components.
Acceptance Criteria
- Gradients have the correct shapes.- A simple numerical example produces correct gradients.- Parameter updates follow gradient descent.- The implementation is consistent with the MSE definition.- Gradient checks against finite differences pass within a documented tolerance.- Tests verify that all layer gradients use the pre-update parameters.- Relevant tests pass.
Completion Report
Report the gradient convention, implementation details, tests executed, actual results, and unresolved issues.