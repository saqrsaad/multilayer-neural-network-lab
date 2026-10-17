# T028 — Verify Saved Experiment Compatibility

## Objective
Perform a read-only compatibility check between experiment records saved by the training view and records loaded by the Experiments view.

## Allowed Scope
Inspect only:
- `src/neural_network_lab/gui/views/training_view.py`
- `src/neural_network_lab/gui/views/experiments_view.py`
- `src/neural_network_lab/application/experiment_service.py`
- `src/neural_network_lab/storage/experiment_storage.py`

## Questions to Answer
1. Does the save path used by `training_view.py` match the directory searched by `experiments_view.py`?
2. Can the Experiments view load the exact record format created by `training_view.py`?
3. Are any required record fields missing or incompatible?

## Constraints
- Read-only: do not modify, create, or delete files.
- Do not inspect unrelated files.
- Do not install dependencies or run tests.
- Do not propose unrelated improvements.

## Completion
Answer the three questions with brief evidence from the allowed files. If a mismatch exists, identify the exact field or path. Stop after the audit.