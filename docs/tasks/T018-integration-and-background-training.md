T018 — Integration and Background Training
Objective
Integrate the application workflows and run lengthy training operations without blocking the GUI.
Read First
- "AGENTS.md"- "docs/PROJECT_SPEC.md"- "docs/ARCHITECTURE.md"- "docs/DEVELOPMENT_WORKFLOW.md"- "docs/TASKS_INDEX.md"
Dependencies
T011 through T017 and the core tasks they rely on.
Scope
- Connect dataset editing, model configuration, training, evaluation, persistence, visualization, and experiment history.- Run training in an appropriate background worker using the project's existing Qt architecture.- Communicate progress, completion, and errors safely to the GUI thread.- Prevent conflicting operations while training is active.- Ensure the interface remains responsive.- Handle window closure and worker cleanup safely.- Keep model access and UI updates coordinated to avoid race conditions.
Constraints
- Do not duplicate neural-network algorithms in the GUI layer.- Do not introduce unnecessary concurrency.- Do not install dependencies or create directories.- Do not change unrelated functionality.
Acceptance Criteria
- Main workflows operate together using the documented architecture.- Training does not freeze the interface.- Progress and completion are reported correctly.- Worker failures are handled without crashing the application.- Relevant available tests pass.