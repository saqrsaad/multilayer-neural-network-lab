General Agent Instructions

1. Read Before Acting

Before implementing any task, read:

1. "AGENTS.md"
2. "docs/PROJECT_SPEC.md"
3. "docs/ARCHITECTURE.md"
4. "docs/DEVELOPMENT_WORKFLOW.md"
5. The specific task file provided by the project owner

Read additional documentation only when relevant to the assigned task.

Do not rely on previous conversations as the source of project requirements.

2. Scope Control

Implement only the requirements of the assigned task.

Do not implement future tasks, unrelated features, or speculative improvements.

Do not change public interfaces, architecture, dependencies, or project conventions without a clear need.

If requirements conflict or an essential detail is missing, explain the issue before making a consequential design decision.

3. Environment Restrictions

The project owner manages the development environment manually.

You must not:

- Install Python packages.
- Install IDE extensions.
- Install system software.
- Modify the Python environment.
- Create or modify virtual environments.
- Upgrade or downgrade dependencies.
- Run package installation commands.
- Change global tool configuration.

Do not install tools to make a task easier.

If a required dependency or tool is missing, report the exact requirement and wait for the project owner.

You may inspect existing configuration files and run already available tools when permitted by the task.

4. File Management

The project owner creates the initial directory structure and task files.

Do not create missing project directories or placeholder files unless the assigned task explicitly requires it.

Modify only files necessary for the assigned task.

Do not overwrite user-created content without checking its purpose.

Do not delete files or reorganize the project without explicit authorization.

5. Implementation Rules

- Use Python and NumPy for the neural network mathematics.
- Keep the core engine independent of the GUI.
- Follow the architecture documented in "docs/ARCHITECTURE.md".
- Prefer clear, readable, maintainable code.
- Use descriptive names and type hints where useful.
- Validate inputs at appropriate boundaries.
- Avoid duplicated logic.
- Avoid unnecessary abstractions and premature optimization.
- Do not introduce unrequested dependencies.
- Do not use PyTorch or TensorFlow to implement the core network.

6. Mathematical Correctness

- Follow the equations and conventions documented in the project.
- Keep tensor and matrix shapes consistent.
- Calculate all required backpropagation deltas using the appropriate pre-update weights.
- Do not silently reset parameters during continued training.
- Evaluation must not modify weights or biases.
- Distinguish training metrics from testing metrics.
- Never fabricate metrics, training history, or experiment results.

If a mathematical ambiguity is discovered, document it and request clarification when it materially affects correctness.

7. Testing

Read the existing tests before changing tested behavior.

Add or update tests for the functionality implemented by the task.

Run only tests supported by the existing environment and permitted by the task.

Never claim that a test passed unless it was actually executed and passed.

If a test cannot be run, state why and identify the remaining verification requirement.

Do not weaken or remove tests merely to make the implementation pass.

8. Documentation

Update documentation when the task changes behavior, interfaces, configuration, or architecture.

Keep documentation consistent with the implemented code.

Do not claim that unfinished functionality is available.

9. Git and External Actions

Do not commit, push, publish, or modify remote repositories unless explicitly instructed.

Do not access external services or transmit project data unless explicitly authorized.

Do not create secrets, credentials, or API keys.

10. Completion Report

At the end of each task, report:

1. What was implemented.
2. Which files were changed.
3. Which tests were executed and their actual results.
4. Any known limitations or unresolved issues.
5. Any manual action required from the project owner.

Keep the report concise and factual.

11. Task Completion

A task is complete only when its acceptance criteria have been addressed and the implementation has been checked to the extent permitted by the available environment.

Do not silently expand the task scope.

Do not begin the next task unless explicitly instructed.