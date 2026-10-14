T001 — Verify Project Structure and Configuration
Objective
Inspect the existing project structure and verify that its configuration is consistent with the project documentation.
Required Reading
- "AGENTS.md"- "docs/PROJECT_SPEC.md"- "docs/ARCHITECTURE.md"- "docs/DEVELOPMENT_WORKFLOW.md"
Scope
1. Inspect the existing directories and files.2. Review "pyproject.toml" and "requirements.txt" if present.3. Check whether the declared package layout matches the documented architecture.4. Identify missing files or configuration inconsistencies.5. Report the minimum changes required to make the project ready for implementation.
Constraints
- Do not install packages or tools.- Do not create directories or placeholder files.- Do not modify the Python environment.- Do not implement application features.- Do not reorganize the project.- Do not modify files unless a specific configuration correction is necessary and clearly within scope.
Acceptance Criteria
- The existing structure has been reviewed.- Configuration inconsistencies are documented.- Missing prerequisites are listed explicitly.- Any changes made are minimal and justified.- No dependencies have been installed.- No directories or placeholder files have been created.
Completion Report
Report:
1. Structure review summary.2. Configuration issues found.3. Files changed, if any.4. Remaining manual actions.