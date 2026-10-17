# T027 — Complete Training Control Tooltips

## Objective
Add tooltips to the existing training settings and training buttons.

## Allowed Changes
- `src/neural_network_lab/gui/widgets/network_config_widget.py`
- `src/neural_network_lab/gui/widgets/training_controls.py`

## Requirements
1. Add tooltips to the existing controls for Epochs, Learning Rate, Activation Function, Hidden Layer Size, and Output Size in `network_config_widget.py`.
2. Add tooltips to the existing Train from Scratch and Continue Training buttons in `training_controls.py`.
3. Explain each control's purpose briefly.
4. Explain that Train from Scratch initializes a new model, while Continue Training starts from the current model parameters.
5. Do not change any existing labels, behavior, or layout.

## Constraints
- Modify only the two allowed files.
- Do not inspect unrelated files.
- Do not add dependencies, documentation, or tests.
- Do not run tests.
- Stop after this task.

## Completion
List the controls updated and report the two files changed. Stop and wait for approval.