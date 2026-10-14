T007 — Implement CSV Import and Export

Objective

Implement CSV import and export for datasets.

Required Reading

- "AGENTS.md"
- "docs/PROJECT_SPEC.md"
- "docs/ARCHITECTURE.md"
- The existing dataset representation and validation implementation.

Scope

1. Import numerical CSV datasets.
2. Export datasets to CSV.
3. Support an explicitly defined target-column configuration.
4. Validate imported data using the existing validation rules.
5. Report malformed files and invalid values clearly.
6. Add tests using small temporary CSV files.

The import interface must make it clear which columns represent features and which column or columns represent targets.

Constraints

- Do not implement GUI file dialogs.
- Do not install packages.
- Do not silently infer an incorrect target column.
- Do not overwrite existing files without explicit caller authorization.
- Use the project's existing data-handling conventions.

Acceptance Criteria

- Valid CSV files can be imported.
- Valid datasets can be exported and imported again without changing their numerical values.
- Invalid files produce clear errors.
- Target-column selection is explicit.
- Tests cover valid files, malformed files, and invalid numerical values.
- Relevant tests pass.

Completion Report

Report the CSV interface, supported format, changed files, tests executed, and known limitations.