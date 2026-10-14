T013 — Network Configuration and Training Controls
Objective
Implement the GUI controls for configuring and training the multilayer perceptron.
Read First
- "AGENTS.md"- "docs/PROJECT_SPEC.md"- "docs/ARCHITECTURE.md"- "docs/DEVELOPMENT_WORKFLOW.md"- "docs/TASKS_INDEX.md"
Dependencies
T005 and T011.
Scope
- Allow users to configure input size, hidden-layer structure, and output size where compatible with the selected dataset.- Provide editable learning-rate and epoch settings.- Provide separate actions for training from scratch and continuing training.- Display training progress and loss history as it becomes available.- Validate settings before training begins.- Display clear success, failure, and validation messages.
Constraints
- Do not implement training mathematics in GUI code.- Do not allow testing operations to update model parameters.- Do not freeze the GUI during long-running training; coordinate background execution in T018.- Do not install dependencies or create directories.
Acceptance Criteria
- Controls expose the required network and training settings.- Invalid settings are rejected before training.- The two training modes call the appropriate application/core APIs.- Training status and errors are communicated clearly.- Relevant available tests pass.