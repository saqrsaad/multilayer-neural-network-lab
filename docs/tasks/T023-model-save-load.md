# T023 — Model Save and Load

## Objective
Add model save and load actions to the existing training view.

## Allowed Changes
- `src/neural_network_lab/gui/views/training_view.py`

Modify no other files.

## Requirements
1. Add a **Save Model** action that saves the current network using the existing `save_model_state` API.
2. Add a **Load Model** action that loads a selected model using the existing `load_model_state` API.
3. Use Qt file dialogs and the existing network state APIs.
4. Reject incompatible loaded models without changing the current network.
5. Show clear success and error messages.
6. Disable both actions while training is active.

## Constraints
- Do not change training logic, storage modules, other views, or project configuration.
- Do not add dependencies or documentation.
- Do not run tests.
- If the existing APIs cannot support this within the allowed file, make no changes and report the blocker.

## Completion
Report the single file changed and the implemented behavior. Stop and wait for approval.