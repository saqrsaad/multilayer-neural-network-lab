T011 — Main Window and Navigation
Objective
Build the main desktop window and navigation structure for Neural Network Lab.
Read First
- "AGENTS.md"- "docs/PROJECT_SPEC.md"- "docs/ARCHITECTURE.md"- "docs/DEVELOPMENT_WORKFLOW.md"- "docs/TASKS_INDEX.md"
Scope
- Implement the main PySide6 window using the existing GUI architecture.- Provide clear navigation to dataset editing, training, testing, model parameters, visualizations, and experiment history.- Establish consistent layout, spacing, labels, and error-message presentation.- Keep application startup and shutdown behavior predictable.- Connect navigation to existing views or clearly defined integration points.
Constraints
- Do not implement all feature views in this task.- Do not place neural-network calculations inside GUI widgets.- Do not change project dependencies or environment settings.- Do not install dependencies or create directories.
Acceptance Criteria
- The application launches into the main window.- Navigation is clear and usable.- Missing feature views do not cause crashes.- GUI code follows the documented architecture.- Relevant available tests pass.