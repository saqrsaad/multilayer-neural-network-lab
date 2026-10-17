# T021 — Audit Training State and Model Persistence

## Objective

Inspect the existing implementation to determine how training state, model parameters, evaluation metrics, and persistence behave. This is a read-only audit; no implementation changes are permitted.

## Scope

1. Trace the existing `Train from Scratch` and `Continue Training` flows.
2. Identify when the network is created, reinitialized, or reused, and whether continuation preserves the current weights and biases.
3. Trace loss calculation and metric reporting, including the source and meaning of displayed training results.
4. Determine what happens to model parameters when dataset rows are cleared, replaced, or imported.
5. Inspect existing model persistence and experiment storage. Determine whether the best model state, final model state, and training history are preserved, and under what conditions.
6. Identify how the GUI obtains and displays training results.
7. Identify confirmed defects, missing behavior, and relevant dependencies between existing modules.

## Constraints

- Do not modify, create, or delete application source files.
- Do not implement fixes or refactor code.
- Do not install dependencies or run tests.
- Do not modify README.md, documentation, task indexes, or status/report files.
- Do not split this task into subtasks or start another task.
- Inspect only relevant files and follow dependencies as needed; do not read the entire repository indiscriminately.
- Distinguish confirmed findings from assumptions. Support findings with file paths and relevant symbols or line references where available.

## Completion Criteria

Provide a concise response in the chat containing:

- The actual training and continuation flow.
- Whether clearing or replacing dataset rows changes model parameters.
- How model state and training results are currently saved and restored.
- The meaning and origin of the displayed loss and evaluation metrics.
- Confirmed issues, missing capabilities, and relevant dependencies.
- A prioritized recommendation for the next implementation task.

Stop after the audit. Wait for explicit authorization before making changes.