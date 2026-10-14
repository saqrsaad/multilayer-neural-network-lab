T014 — Testing UI
Objective
Implement a GUI for testing a trained model without changing its parameters.
Read First
- "AGENTS.md"- "docs/PROJECT_SPEC.md"- "docs/ARCHITECTURE.md"- "docs/DEVELOPMENT_WORKFLOW.md"- "docs/TASKS_INDEX.md"
Dependencies
T008 and T011.
Scope
- Allow users to enter feature values for individual samples or select a labeled dataset for evaluation.- Display predicted probabilities and binary predictions.- When labeled targets are available, display the evaluation metrics supported by the application.- Clearly distinguish prediction-only input from labeled evaluation data.- Report missing models, invalid inputs, and incompatible feature dimensions.
Constraints
- Use the existing evaluation service.- Never update model weights or biases during testing.- Do not implement evaluation formulas inside GUI widgets.- Do not install dependencies or create directories.
Acceptance Criteria
- Users can test individual samples and evaluate labeled data.- Predictions and available metrics are displayed clearly.- Prediction-only inputs do not require target labels.- Tests confirm that the testing workflow does not alter model parameters.- Relevant available tests pass.