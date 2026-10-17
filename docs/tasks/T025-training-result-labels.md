# T025 — Improve Training Result Feedback

## Objective
Make the final training result clearly distinguish final training loss from evaluation accuracy.

## Allowed Changes
- `src/neural_network_lab/gui/views/training_view.py`

Modify no other files.

## Requirements
1. Locate the existing final training-result display.
2. Label the final loss explicitly as **Final Training MSE**.
3. If the view already has an evaluation accuracy value, label it **Evaluation Accuracy** and keep it separate from training loss.
4. Do not calculate or invent new metrics.
5. Preserve the existing training behavior and layout as much as possible.

## Constraints
- Do not modify any other file.
- Do not add dependencies, documentation, or tests.
- Do not run tests.
- If the existing display already satisfies all requirements, make no changes and report that.
- Stop after this task.

## Completion
Report the exact change briefly. Do not start another task.