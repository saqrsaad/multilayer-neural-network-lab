# T022 — Save Completed Training Runs

## Objective
Save one experiment record automatically after a training run completes successfully.

## Allowed Changes
Modify only:
- `src/neural_network_lab/gui/views/training_view.py`

Do not modify any other file.

## Required Behavior
1. Use the existing experiment-record creation and storage APIs.
2. After a training run completes successfully and its final model state has been applied, create and save one experiment record.
3. Record only values already available from the completed training run and its existing configuration.
4. Use a unique filename for each saved record. Never overwrite an existing record.
5. If saving fails, keep the trained model applied and display a clear error message.
6. Do not save an experiment when training fails or is cancelled.

## Restrictions
- Do not add model save/load functionality.
- Do not implement best-model tracking or checkpoints.
- Do not change training algorithms, metrics, data handling, or other GUI pages.
- Do not refactor unrelated code.
- Do not create subtasks.
- Do not install dependencies or run tests.
- Do not modify documentation, README files, task indexes, or status/report files.
- If the existing APIs cannot support the required behavior by editing the allowed file alone, make no changes and report the specific blocker.

## Completion Report
Report the exact change made in `training_view.py` and any blocker. State explicitly that tests were not run.

Stop after this task and wait for approval.