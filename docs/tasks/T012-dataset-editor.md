T012 — Dataset Editor
Objective
Implement a GUI for entering, editing, importing, and exporting datasets.
Read First
- "AGENTS.md"- "docs/PROJECT_SPEC.md"- "docs/ARCHITECTURE.md"- "docs/DEVELOPMENT_WORKFLOW.md"- "docs/TASKS_INDEX.md"
Dependencies
T006, T007, and T011.
Scope
- Provide a table-based interface for entering and editing feature values and binary targets.- Allow users to add and remove samples.- Support datasets with multiple input features and one binary target.- Integrate existing dataset validation and CSV import/export services.- Display validation and import errors clearly.- Make the currently loaded dataset visible and understandable.
Constraints
- Keep dataset rules in the data layer, not in GUI widgets.- Do not start model training from this view.- Do not silently discard unsaved edits.- Do not install dependencies or create directories.
Acceptance Criteria
- Users can enter and edit sample rows.- Invalid values are reported clearly.- CSV import/export uses existing data-layer services.- GUI state remains consistent after successful or failed operations.- Relevant available tests pass.