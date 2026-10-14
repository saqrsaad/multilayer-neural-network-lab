T002 — Implement Activation Functions and Loss
Objective
Implement the initial activation function and loss calculation required by the neural network.
Required Reading
- "AGENTS.md"- "docs/PROJECT_SPEC.md"- "docs/ARCHITECTURE.md"
Scope
Implement the following in the appropriate core modules:
1. Sigmoid activation using NumPy.2. A numerically stable sigmoid implementation.3. Mean Squared Error (MSE).4. Input validation for supported array shapes and invalid numerical values.5. Unit tests for valid inputs and edge cases.
Use these definitions:
Sigmoid:
sigma(z) = 1 / (1 + exp(-z))
Sigmoid derivative:
sigma_prime(z) = sigma(z) * (1 - sigma(z))
MSE:
MSE = mean((targets - predictions) ** 2)
The derivative helper must clearly document whether it accepts pre-activation values or already computed sigmoid outputs.
Constraints
- Do not implement the network or training loop.- Do not introduce new dependencies.- Use NumPy for numerical operations.- Do not silently accept NaN or infinite values.- Follow the existing directory structure.
Acceptance Criteria
- Sigmoid returns correct values for representative inputs.- The sigmoid implementation avoids numerical overflow for extreme inputs.- The derivative produces correct results.- MSE matches hand-calculated examples.- Invalid shapes and non-finite inputs are handled consistently.- Relevant tests pass.
Completion Report
Report implemented functions, changed files, test commands, actual results, and unresolved issues.