T006 — Implement Dataset Representation and Validation
Objective
Create a consistent representation for numerical datasets and validate data before training or evaluation.
Required Reading
- "AGENTS.md"- "docs/PROJECT_SPEC.md"- "docs/ARCHITECTURE.md"
Scope
1. Represent input features and optional target values.2. Store numerical data using NumPy arrays.3. Validate feature and target dimensions.4. Validate that input and target sample counts match.5. Reject empty datasets when an operation requires samples.6. Reject NaN and infinite values.7. Validate compatibility with a specified network input and output size.8. Add unit tests.
The dataset representation must support labeled datasets for supervised training and evaluation.
If targets are absent, prediction may still be possible, but supervised loss and accuracy cannot be calculated.
Constraints
- Do not implement the GUI.- Do not implement CSV operations in this task.- Do not train the network.- Do not silently discard invalid rows.- Do not silently change feature or target dimensions.- Keep validation independent of PySide6.
Acceptance Criteria
- Valid datasets can be represented consistently.- Invalid sample counts and dimensions are rejected.- Non-finite values are rejected.- Missing targets are handled explicitly.- Dataset compatibility can be checked against a network configuration.- Relevant tests pass.
Completion Report
Report the dataset interface, validation rules, changed files, tests executed, and unresolved issues.