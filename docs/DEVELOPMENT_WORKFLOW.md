Development Workflow

1. Task-Driven Development

All implementation work must be driven by an explicit task file.

Each task file must define:

- Task identifier and title.
- Objective.
- Scope.
- Required changes.
- Constraints.
- Acceptance criteria.
- Required verification.

Do not infer permission to implement unrelated features.

2. Standard Execution Cycle

For each task:

1. Read the general agent instructions.
2. Read the relevant project specification and architecture.
3. Read the assigned task file.
4. Inspect relevant existing files.
5. Identify the smallest implementation that satisfies the task.
6. Implement the required changes.
7. Run relevant available tests.
8. Review the changes against the acceptance criteria.
9. Update relevant documentation if needed.
10. Report the outcome.

3. Task Boundaries

Complete only one assigned task at a time.

Do not begin the next task automatically.

If a task depends on unfinished work, report the dependency instead of implementing unrelated prerequisite tasks without authorization.

4. Change Management

Keep changes focused and easy to review.

Avoid unrelated refactoring.

Do not reformat entire files unless required.

Do not change public interfaces without a clear reason.

If a task reveals a necessary architectural change, explain it before expanding the scope.

5. Verification

Use existing project tools and dependencies only.

Run the smallest relevant test set first.

Run broader tests when appropriate and feasible.

Report actual test commands and results.

Clearly distinguish between verified behavior, unverified behavior, and known limitations.

6. Acceptance Criteria

Review every acceptance criterion before declaring a task complete.

If a criterion cannot be satisfied, explain the blocker.

Do not claim completion when essential criteria remain unmet.

7. Task Status

Use these statuses in "docs/TASKS_INDEX.md":

- "PLANNED": Not started.
- "IN_PROGRESS": Currently being implemented.
- "BLOCKED": Waiting for a dependency or decision.
- "REVIEW": Implemented but awaiting owner review.
- "DONE": Accepted by the project owner.

Only the project owner may mark a task as "DONE".

8. Token Efficiency

Prefer focused inspection over reading the entire repository.

Read files relevant to the assigned task.

Keep implementation plans concise.

Avoid repeatedly restating project documentation.

Do not implement speculative improvements.

Report only meaningful changes and verification results.