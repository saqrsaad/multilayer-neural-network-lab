T017 — Experiment History and Comparison
Objective
Implement a GUI for browsing and comparing training experiments.
Read First
- "AGENTS.md"- "docs/PROJECT_SPEC.md"- "docs/ARCHITECTURE.md"- "docs/DEVELOPMENT_WORKFLOW.md"- "docs/TASKS_INDEX.md"
Dependencies
T009, T010, and T011.
Scope
- Display saved experiment records in a clear table or list.- Show experiment configuration, training settings, history, and evaluation metrics.- Allow users to select experiments for side-by-side comparison.- Highlight differences in settings and reported results.- Provide clear feedback when records cannot be loaded.- Integrate with the existing experiment and persistence services.
Constraints
- Do not recalculate or invent missing historical metrics.- Do not overwrite saved experiments without explicit user action.- Keep experiment management outside GUI widgets.- Do not install dependencies or create directories.
Acceptance Criteria
- Users can browse and inspect saved experiments.- Compatible experiments can be compared.- Differences in configuration and metrics are understandable.- Missing or invalid records are handled gracefully.- Relevant available tests pass.