# T024 — Implement Network Configuration View

## Objective
Replace the existing Network Configuration placeholder with a form that displays and edits the current network architecture.

## Allowed Changes
- `src/neural_network_lab/gui/views/network_configuration_view.py`

Modify no other files.

## Requirements
1. Inspect only the allowed file and its existing public interface.
2. Display the current network layer sizes when a network is available.
3. Allow the user to enter layer sizes as positive integers, with at least two layers.
4. Apply valid configuration by creating a new `NeuralNetwork` and making it the active network through the existing interface.
5. Reject invalid input without changing the active network.
6. If the existing file cannot support this behavior without changing another file, make no changes and report the blocker.

## Constraints
- Do not modify navigation, other views, core algorithms, storage, or application services.
- Do not add dependencies, documentation, or tests.
- Do not run tests.
- Do not create subtasks or implement unrelated UI improvements.

## Completion
Briefly report the file changed and behavior implemented. Stop and wait for approval.