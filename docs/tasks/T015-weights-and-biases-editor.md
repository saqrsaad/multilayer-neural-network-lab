T015 — Weights and Biases Editor
Objective
Provide an educational interface for inspecting and editing network weights and biases.
Read First
- "AGENTS.md"- "docs/PROJECT_SPEC.md"- "docs/ARCHITECTURE.md"- "docs/DEVELOPMENT_WORKFLOW.md"- "docs/TASKS_INDEX.md"
Dependencies
T003 and T011.
Scope
- Display weights and biases for each layer in readable tables.- Identify layer indices, parameter shapes, and individual parameter values.- Allow users to edit valid parameter values.- Validate edited values and reject non-finite numbers.- Apply accepted changes through the model's public API.- Provide a clear way to refresh the display after training or loading a model.
Constraints
- Do not directly manipulate undocumented internal state.- Do not silently apply invalid edits.- Make it clear that manual edits can affect model predictions.- Do not install dependencies or create directories.
Acceptance Criteria
- Current weights and biases are displayed accurately.- Valid edits update the intended parameters.- Invalid edits leave model parameters unchanged.- Parameter shapes and layer identification remain clear.- Relevant available tests pass.