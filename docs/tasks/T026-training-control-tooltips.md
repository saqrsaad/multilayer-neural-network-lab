# T026 — Add Tooltips to Training Controls

## Objective
Add concise tooltips to the existing training settings controls.

## Allowed Changes
- `src/neural_network_lab/gui/views/training_view.py`

Modify no other files.

## Requirements
Add tooltips to the existing controls for:
- Epochs
- Learning Rate
- Activation Function
- Hidden Layer Size
- Output Size
- Train from Scratch
- Continue Training
- Save Model
- Load Model

Each tooltip must briefly explain the control's purpose. For the two training buttons, explain the difference between starting with newly initialized weights and continuing from the current model.

## Constraints
- Modify only the allowed file.
- Do not change control behavior, labels, layout, or training logic.
- Do not add dependencies, documentation, or tests.
- Do not run tests.
- If a listed control is not defined in this file, skip it and report it.
- Stop after this task.

## Completion
Report the file changed and list any controls skipped. Stop and wait for approval.