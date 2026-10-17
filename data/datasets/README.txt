Multilayer Neural Network Lab — CSV test datasets
====================================================

General CSV convention:
- The first columns are input features.
- The final column(s) are target(s).
- During import, explicitly choose the number of target columns where the app asks.
- Do not use 10_intentionally_invalid_rows.csv as a normal training dataset; it is specifically for testing validation/error handling.

Files:
01_binary_classification_2_features.csv
  240 rows, 2 features, 1 binary target. Good first import/training smoke test.

02_xor_nonlinear_classification.csv
  240 rows, 2 features, 1 binary target. Nonlinear XOR pattern; useful for testing multiple hidden layers.

03_three_features_binary_classification.csv
  300 rows, 3 features, 1 binary target. Useful for feature projection in Decision Boundary view.

04_multi_output_binary_2_targets.csv
  300 rows, 3 features, 2 binary targets. Checks multi-output training/testing/evaluation.

05_multi_output_continuous_targets.csv
  260 rows, 3 features, 2 continuous targets in [0, 1]. Checks multiple continuous outputs; classification metrics may not apply.

06_three_class_one_hot_3_outputs.csv
  300 rows, 2 features, 3 one-hot target columns. Tests a 3-output representation of three classes. Each row has exactly one target set to 1.

07_larger_5_features_2_targets.csv
  600 rows, 5 features, 2 binary targets. Larger dataset and feature projection test. For Decision Boundary, choose two features; remaining feature(s) should be held at their means if the app supports that behavior.

08_nonlinear_continuous_target.csv
  300 rows, 2 features, 1 continuous target in [0, 1]. Useful for loss curves and prediction plots.

09_zero_negative_and_boundary_values.csv
  16 rows, 2 features, 1 binary target. Includes zero and negative feature values and points near a decision boundary; valid data.

10_intentionally_invalid_rows.csv
  5 rows with NaN, infinity, and a target value outside binary {0,1}. Use only to verify import/data validation rejects or clearly reports invalid values.

Suggested manual test sequence:
1. Import file 01; set target columns = 1; train, test, and evaluate.
2. Import file 02; try a network with at least two hidden layers.
3. Import file 03; open Decision Boundary and select two features.
4. Import file 04; set target columns = 2 and configure two output neurons.
5. Import file 06; configure three output neurons; verify output columns are not silently truncated.
6. Import file 07; try five input neurons, hidden layers of your choice, and two outputs.
7. Import file 09; inspect plots and verify zero/negative values do not break axes.
8. Import file 10 only for validation behavior.

Important:
- The data is synthetic and intended for application testing, not scientific benchmarking.
- If the application uses a fixed training/test split, ensure the split preserves enough examples of each class.
- The current Decision Boundary implementation may support only binary targets for the selected output; use files 01, 02, or 03 for the clearest initial test.
