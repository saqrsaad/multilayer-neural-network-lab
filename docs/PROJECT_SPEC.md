Project Specification

1. Project Overview

Neural Network Lab is a desktop educational application for experimenting with multilayer neural networks.

Users must be able to enter datasets, configure a network, train it, test it, inspect its parameters, and compare experiments through a graphical interface.

The application must make the relationship between mathematical operations and observed results understandable.

2. Primary Objectives

1. Implement neural network mathematics using NumPy.
2. Separate the neural network engine from the graphical interface.
3. Support user-entered training and testing data.
4. Preserve model parameters during testing.
5. Support training from scratch and continued training.
6. Support manual weight and bias editing.
7. Visualize training and evaluation results.
8. Save and restore model states and experiment records.
9. Provide automated tests and maintainable documentation.

3. Initial Scope

The first release supports a feedforward multilayer perceptron (MLP) for binary classification.

Initial mathematical features:

- Forward propagation.
- Sigmoid activation.
- Mean Squared Error (MSE).
- Backpropagation.
- Full-batch gradient descent.
- Configurable learning rate.
- Configurable epoch count.
- Configurable layer sizes, subject to input and output compatibility.

The neural network engine must use NumPy and must not depend on the GUI.

4. Dataset Management

Users must be able to:

- Create datasets through the GUI.
- Add, edit, and remove rows.
- Select training and testing datasets independently.
- Import and export CSV files.
- Validate dataset dimensions and values.
- Save datasets for later use.

The initial data format consists of numerical input features and target values.

The number of input columns must match the network input size.

For supervised evaluation, target dimensions must match the network output size.

Invalid data must produce understandable validation messages.

5. Model Configuration

Users must be able to configure:

- Input dimension, determined by the dataset.
- Hidden layer count and sizes.
- Output dimension, determined by the task.
- Learning rate.
- Epoch count.
- Initial weight configuration.

The initial implementation uses sigmoid activation and MSE.

Configuration changes that affect network dimensions must trigger appropriate model reconfiguration or validation. They must not silently corrupt or misinterpret existing parameters.

6. Training Behavior

The application must provide two distinct training modes.

6.1 Train from Scratch

- Initialize model parameters using the selected initialization method.
- Start training with the selected training dataset.
- Record training configuration and results.
- Clearly indicate that previous parameters have been replaced.

6.2 Continue Training

- Start from the current model parameters.
- Use the selected training dataset and current training settings.
- Preserve the model's existing parameters as the starting state.
- Record the new training session separately.
- Allow evaluation on previously saved test datasets.

Training must never use the testing dataset to update model parameters.

The initial optimizer is full-batch gradient descent.

7. Testing and Evaluation

Testing must:

- Use the current model parameters without modifying them.
- Produce predictions for the selected test dataset.
- Calculate MSE when valid target values are available.
- Calculate classification accuracy using a threshold of 0.5.
- Display per-sample predictions and errors when targets are available.
- Clearly indicate when metrics cannot be calculated.

Testing must not automatically trigger training.

8. Manual Parameter Editing

Users must be able to inspect and edit model weights and biases through the GUI.

The application must:

- Validate parameter dimensions and numerical values.
- Apply accepted edits to the current model.
- Mark the model as modified.
- Allow evaluation using the edited parameters.
- Allow continued training from the edited state.
- Provide a way to restore a previously saved model state.

Manual edits must not be silently discarded.

9. Visualization

The initial release should provide:

- Training loss curves.
- Testing loss curves when evaluation history is available.
- Actual-versus-predicted comparisons.
- Per-sample result tables.
- Weight and bias tables.
- Experiment comparison summaries.

Testing loss must be calculated without updating parameters.

Training and testing metrics must be labeled separately.

Visualizations must reflect actual recorded results and must not use fabricated metrics.

10. Model and Experiment Persistence

The application must distinguish between:

Model State

Contains the network architecture, weights, biases, and required model configuration.

Experiment Record

Contains the training configuration, dataset references or snapshots, evaluation metrics, training history, and relevant model-state references.

Users must be able to save and load model states and review previous experiment results.

Saved artifacts must be validated before loading.

11. Experiment Comparison

Users must be able to compare model performance before and after continued training.

The application should preserve previous evaluation records and support reevaluation of saved test datasets using the current model.

Historical results must not be overwritten silently.

Comparisons must identify the datasets and model states used.

12. User Interface

The GUI should be organized into clear sections:

- Dashboard.
- Dataset Management.
- Network Configuration.
- Training.
- Testing.
- Weights and Biases.
- Experiment History.

The interface must provide understandable validation errors, operation status, and training progress.

Long-running training must not freeze the interface. Use an appropriate background execution mechanism when needed.

13. Testing Requirements

Automated tests must cover:

- Activation functions.
- Forward propagation.
- Loss calculation.
- Backpropagation.
- Parameter updates.
- Training behavior.
- Evaluation behavior.
- Dataset validation.
- Model persistence.
- Experiment persistence.

A critical invariant is that evaluation must not change model parameters.

14. Non-Goals

The initial release does not include:

- Convolutional or recurrent neural networks.
- Deep learning frameworks such as PyTorch or TensorFlow.
- Advanced optimizers such as Adam.
- Distributed training.
- Cloud services or user accounts.
- High-stakes decision-making applications.
- Automatic dependency installation or environment modification.

15. Acceptance Criteria

The initial release is acceptable when:

1. The application launches in the prepared environment.
2. Users can enter and edit datasets through the GUI.
3. Users can configure and train an MLP.
4. Training can start from scratch or continue from current parameters.
5. Testing does not modify model parameters.
6. Users can inspect and edit weights and biases.
7. Training and testing results are clearly displayed.
8. Model states and experiment records can be saved and restored.
9. Core functionality is covered by automated tests.
10. Setup, usage, architecture, and algorithm documentation is available.