T020 — Final Verification and GitHub Readiness
Objective
Verify the completed project and prepare it for review and publication on GitHub.
Read First
- "AGENTS.md"- "README.md"- "docs/PROJECT_SPEC.md"- "docs/ARCHITECTURE.md"- "docs/DEVELOPMENT_WORKFLOW.md"- "docs/TASKS_INDEX.md"
Dependencies
All implementation tasks, including T018 and T019.
Scope
- Review the project against the specification and architecture.- Run available automated tests and report actual results.- Check for broken imports, obvious runtime errors, and inconsistent interfaces.- Review error handling, dataset validation, persistence, and separation between GUI and core logic.- Verify that documentation matches implemented behavior.- Check repository hygiene, including secrets, temporary files, generated artifacts, and appropriate ignore rules.- Report remaining defects and incomplete features.
Constraints
- Do not install dependencies or change the environment.- Do not hide, skip, or misrepresent failing tests.- Do not delete user data or make unrelated changes.- Do not commit, push, publish, or modify remote repository settings without explicit authorization.- Do not claim verification steps were completed unless they were actually performed.
Acceptance Criteria
- Available tests have been run and their actual results reported.- Identified defects are fixed only when within scope and safe to address; others are documented.- Documentation and implementation are consistent.- Remaining limitations and release blockers are listed clearly.- The repository is ready for owner review before any publication.