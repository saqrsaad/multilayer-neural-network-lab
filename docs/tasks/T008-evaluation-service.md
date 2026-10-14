T008 — Evaluation Service
Objective
Implement evaluation utilities for the neural network without modifying model parameters.
Read First
- "AGENTS.md"- "docs/PROJECT_SPEC.md"- "docs/ARCHITECTURE.md"- "docs/DEVELOPMENT_WORKFLOW.md"- "docs/TASKS_INDEX.md"
Scope
- Evaluate predictions against labeled binary-classification targets.- Calculate loss using the project's existing loss implementation.- Calculate accuracy using a documented binary threshold.- Return predictions and metrics in a consistent, testable format.- Validate input shapes and reject incompatible data.- Ensure evaluation never updates weights or biases.
Constraints
- Do not implement GUI components.- Do not add training behavior to evaluation.- Reuse existing core and dataset implementations.- Do not install dependencies or create directories.
Acceptance Criteria
- Evaluation returns correct predictions and metrics on known examples.- Invalid inputs produce clear errors.- Model parameters remain unchanged after evaluation.- Automated tests cover normal cases and invalid inputs.- Relevant available tests pass.