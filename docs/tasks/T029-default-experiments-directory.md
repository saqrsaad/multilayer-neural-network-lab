# T029 — Default Experiments Directory

## Objective
Make the Experiments view open `data/experiments` as the initial directory in its existing directory-selection dialog.

## Allowed Changes
- `src/neural_network_lab/gui/views/experiments_view.py`

Modify no other files.

## Requirements
1. Locate the existing directory-selection dialog.
2. Set its initial directory to `data/experiments`.
3. Preserve the existing dialog behavior and JSON loading logic.
4. Do not change how experiment files are saved.

## Constraints
- Modify only the allowed file.
- Do not inspect unrelated files.
- Do not add dependencies, documentation, or tests.
- Do not run tests.
- If the dialog cannot use this initial directory without broader changes, make no changes and report the blocker.

## Completion
Report the one-line change or the specific blocker. Stop after this task.