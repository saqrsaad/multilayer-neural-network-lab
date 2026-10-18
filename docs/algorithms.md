# Algorithms Reference

This document describes the mathematical algorithms implemented in the
`multilayer-neural-network-lab` core engine. Every equation below is
derived from the actual source implementation, not from a generic
reference. The core engine uses NumPy only and is independent of the GUI,
application, and storage layers.

## Source files

- `../src/neural_network_lab/core/activations.py` — sigmoid and its derivative.
- `../src/neural_network_lab/core/losses.py` — mean squared error.
- `../src/neural_network_lab/core/network.py` — forward propagation, training modes, initialization, parameter state.
- `../src/neural_network_lab/core/backpropagation.py` — gradient computation.
- `../src/neural_network_lab/core/optimizers.py` — parameter update rule.
- `../src/neural_network_lab/core/training.py` — validation-based best-epoch selection and restoration.
- `../src/neural_network_lab/data/dataset_split.py` — train/validation/test splitting.
- `../src/neural_network_lab/application/evaluation_service.py` — evaluation metrics.

## 1. Conventions: layer sizes and parameter shapes

A network is defined by a sequence of layer sizes

$$[n_0, n_1, \dots, n_L]$$

where $n_0$ is the input dimension, $n_L$ is the output dimension, and
$n_1, \dots, n_{L-1}$ are hidden-layer sizes. The network contains
$L$ trainable weight layers, indexed $l = 0, \dots, L-1$, where layer
$l$ transforms activations from layer $l$ to layer $l+1$.

The project uses **row-vector batch orientation**: each row of a batch
matrix is one sample.

| Quantity | Shape | Meaning |
|---|---|---|
| $W_l$ | $(n_l, n_{l+1})$ | Weight matrix from layer $l$ to layer $l+1$ |
| $b_l$ | $(1, n_{l+1})$ | Bias vector for layer $l+1$ (broadcast over the batch) |
| $X$ | $(N, n_0)$ | Batch input, $N$ samples |
| $A_l$ | $(N, n_l)$ | Activations at layer $l$; $A_0 = X$ |
| $Z_l$ | $(N, n_{l+1})$ | Pre-activations at layer $l$ |
| $Y$ | $(N, n_L)$ | Targets |

Source: `../src/neural_network_lab/core/network.py`, parameter-shape
convention documented in the module docstring and enforced by
`../src/neural_network_lab/docs/ARCHITECTURE.md` §6.

## 2. Sigmoid activation and its derivative

The sigmoid is applied elementwise at every layer:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### Derivative

The derivative with respect to the pre-activation $z$ is

$$\sigma'(z) = \sigma(z) \bigl(1 - \sigma(z)\bigr)$$

In backpropagation the derivative is evaluated at the already-computed
activation $A = \sigma(Z)$, giving the factor $A(1-A)$.

### Numerical stability

The implementation evaluates two branches so `exp` never overflows:

- For $z \ge 0$: $\sigma(z) = 1 / (1 + e^{-z})$, where $e^{-z} \in (0, 1]$.
- For $z < 0$: $\sigma(z) = e^z / (1 + e^z)$, where $e^z \in (0, 1)$.

Outputs lie strictly in $(0, 1)$; derivatives lie in $(0, 0.25]$.

Inputs are validated to be numeric, non-empty, and finite
(NaN/inf are rejected).

Source: `../src/neural_network_lab/core/activations.py`
(`sigmoid`, `sigmoid_derivative`).

## 3. Forward propagation

For each layer $l = 0, \dots, L-1$:

$$Z_l = A_l W_l + b_l$$

$$A_{l+1} = \sigma(Z_l)$$

with $A_0 = X$. The network output is $A_L$, shape $(N, n_L)$.

The implementation caches all activations $[A_0, \dots, A_L]$ and
pre-activations $[Z_0, \dots, Z_{L-1}]$ for use by backpropagation.
Forward propagation is read-only: it never modifies weights or biases.

Input validation requires $X$ to be 2D with at least one sample, the
correct feature count, and only finite values.

Source: `../src/neural_network_lab/core/network.py`
(`forward_with_cache`, `forward`, `predict`).

## 4. Mean squared error and its gradient

### Loss

Mean squared error is averaged over **all** $N \cdot K$ output elements,
where $N$ is the batch size and $K = n_L$ is the output size:

$$L = \frac{1}{N K} \sum_{i=1}^{N} \sum_{k=1}^{K} (Y_{ik} - A_{ik})^2$$

Equivalently, $L = \operatorname{mean}\bigl((Y - A)^2\bigr)$ over every
sample and every output. This averaging convention matters for the
gradient below and for multi-output models (see §9).

### Gradient

The gradient of the loss with respect to the output activations is

$$\frac{\partial L}{\partial A} = \frac{2 (A - Y)}{N K}$$

This matches the implementation exactly: the normalizer is
$N \cdot K$, not $N$ alone or $K$ alone.

Inputs are validated to have identical shapes, be non-empty, and contain
only finite values.

Source: `../src/neural_network_lab/core/losses.py` (`mean_squared_error`).

## 5. Backpropagation

Gradients are computed for all layers from the **same pre-update
parameter state**. The implementation snapshots the weights before the
backward pass (`network.weights`, which returns copies), so hidden-layer
deltas always use the original $W_{l+1}$, never weights that have been
partially updated.

### Output-layer delta

Let $A = A_L$ be the output activation and $Y$ the targets. The output
delta is

$$\delta_L = \frac{\partial L}{\partial A} \odot A \odot (1 - A)
= \frac{2 (A - Y)}{N K} \odot A \odot (1 - A)$$

where $\odot$ denotes elementwise multiplication.

### Hidden-layer deltas

For $l = L-2, \dots, 0$, propagating backward through the pre-update
weight $W_{l+1}$:

$$\delta_l = \bigl(\delta_{l+1} W_{l+1}^{T}\bigr) \odot A_{l+1} \odot (1 - A_{l+1})$$

Note that the activation factor uses $A_{l+1} = \sigma(Z_l)$ (the
post-activation of layer $l$'s output), and the weight used is
$W_{l+1}$ (the matrix from layer $l+1$ to $l+2$), i.e. the pre-update
weight.

### Weight and bias gradients

$$\frac{\partial L}{\partial W_l} = A_l^{T} \delta_l
\quad\text{with shape }(n_l, n_{l+1})$$

$$\frac{\partial L}{\partial b_l} = \sum_{i=1}^{N} \delta_{l,i}
\quad\text{with shape }(1, n_{l+1})$$

The bias gradient keeps `keepdims=True`, so it remains a 2D row vector.

`compute_gradients` is pure: it never modifies the network.

Source: `../src/neural_network_lab/core/backpropagation.py`
(`compute_gradients`, `validate_targets`).

## 6. Parameter updates: full-batch gradient descent

Each epoch performs one full-batch gradient descent step over the entire
training set:

$$W_l \leftarrow W_l - \eta \frac{\partial L}{\partial W_l}$$

$$b_l \leftarrow b_l - \eta \frac{\partial L}{\partial b_l}$$

where $\eta > 0$ is the learning rate. The update is applied identically
to weights and biases. The implementation computes new parameter arrays
without mutating the inputs (`sgd_update` is a pure function), then
validates that the result is finite before committing.

### Important terminology

This is **full-batch gradient descent**, not stochastic gradient descent
in the mini-batch sense: every epoch uses the complete training dataset
at once to compute a single gradient. There is no shuffling, no
mini-batching, and no per-sample update. This is the only optimizer
supported by the initial release; no Adam, RMSprop, or other adaptive
method is implemented.

The learning rate must be a finite number strictly greater than zero;
NaN, infinity, non-positive values, and non-numeric types are rejected.

Source: `../src/neural_network_lab/core/optimizers.py`
(`sgd_update`, `validate_learning_rate`);
`../src/neural_network_lab/core/network.py` (`apply_gradients`,
`train_step`).

## 7. Weight and bias initialization

### Xavier-uniform weights

For each layer $l$ with fan-in $n_l$ and fan-out $n_{l+1}$:

$$\text{limit} = \sqrt{\frac{6}{n_l + n_{l+1}}}$$

Each entry of $W_l$ is drawn independently from the uniform
distribution $\mathcal{U}(-\text{limit}, \text{limit})$.

### Zero biases

Every bias vector $b_l$ is initialized to zeros, shape $(1, n_{l+1})$.

When training from scratch (see §8), all inputs and settings are
validated **before** the current parameters are replaced, so an invalid
request leaves the network untouched.

Source: `../src/neural_network_lab/core/network.py`
(`_random_parameters`, `train_from_scratch`).

## 8. Training modes

### Continued training

`train(X, y, epochs, eta)` starts from the network's current parameters
and applies full-batch gradient descent for `epochs` epochs. Parameters
are never reinitialized or silently reset. The recorded loss history
holds one measured **post-update** training MSE per epoch (the MSE is
recomputed on the same batch after the parameter update).

### Training from scratch

`train_from_scratch(X, y, epochs, eta, seed)` explicitly reinitializes
the network with fresh Xavier-uniform weights and zero biases (using the
optional `seed` for reproducibility), then trains for `epochs` epochs.
All inputs are validated before any parameter is replaced, so a rejected
request leaves the existing model completely unchanged.

Source: `../src/neural_network_lab/core/network.py`
(`train`, `train_from_scratch`, `_train_loop`).

## 9. Training with validation: best-epoch selection and restoration

When validation data is supplied, each epoch performs:

1. A training step on the training rows only (gradient update).
2. A read-only forward pass on the validation rows, scoring MSE without
   producing any gradient update. Validation data never contributes to
   parameter updates.

The **best epoch** is the 1-based epoch with the lowest validation MSE.
At that moment the exact parameters are snapshotted via
`to_state_dict()`. After the final epoch, if a best snapshot exists, it
is restored onto the network via `set_parameters()`. The caller therefore
receives the selected best model, not the parameters of the last epoch.

The loss history records one measured post-update training MSE per
requested epoch. The validation history records one validation MSE per
epoch, aligned by epoch number.

### Cancellation

An optional cancellation predicate is checked between epochs. If
cancellation is requested, the run stops immediately, returns the epochs
completed so far, and — importantly — **does not restore** the best
snapshot. The network retains whatever parameters it had at the point of
cancellation (each epoch's update is applied in full before the next
cancellation check).

Validation inputs are checked before the first training step so that
invalid held-out data cannot leave the model partially trained.

Source: `../src/neural_network_lab/core/training.py`
(`train_with_validation`, `TrainingOutcome`).

## 10. Dataset splitting

A labeled dataset can be divided into disjoint train/validation/test
subsets by row. Every sample belongs to exactly one subset, so
feature-target alignment is preserved and no sample appears in more than
one subset.

Split proportions must be finite, non-negative, and sum to 1.0. Row
counts are allocated by the largest-remainder method so the counts sum
exactly to the total number of samples.

A validation or test subset is considered meaningful only if it contains
at least `MIN_MEANINGFUL_SAMPLES = 2` rows; a single row can only score
0% or 100%, so its metrics would be misleading. If a requested subset
would be smaller than this, it is dropped, its rows join the training
subset, and a warning is emitted instead of silently reporting unusable
numbers.

Source: `../src/neural_network_lab/data/dataset_split.py`
(`split_dataset`, `DatasetSplits`, `MIN_MEANINGFUL_SAMPLES`).

## 11. Evaluation metrics

Evaluation is strictly read-only: a forward pass produces predictions,
and metrics are computed without modifying any parameter.

### Regression metrics

For predictions $A$ and targets $Y$ over $N$ samples and $K$ outputs:

- MSE: $\displaystyle \frac{1}{NK} \sum_{i,k} (Y_{ik} - A_{ik})^2$
- MAE: $\displaystyle \frac{1}{NK} \sum_{i,k} |Y_{ik} - A_{ik}|$
- RMSE: $\displaystyle \sqrt{\mathrm{MSE}}$

All three are averaged over **every sample and every output**, so they
are well-defined for multi-output models.

### Binary classification accuracy

For datasets declared as binary classification (targets $0.0$ or $1.0$),
the predicted class is

$$\hat{y}_{ik} =
\begin{cases}
1 & \text{if } A_{ik} \ge 0.5 \\
0 & \text{otherwise}
\end{cases}$$

Accuracy is the fraction of outputs whose predicted class matches the
target:

$$\text{accuracy} = \frac{1}{NK} \sum_{i,k} \mathbf{1}\{\hat{y}_{ik} = Y_{ik}\}$$

The threshold is exactly $0.5$, and a value of $0.5$ counts as class $1$.

For continuous-regression targets, no class label is defined and
accuracy is reported as absent (not computed, not zero).

Multi-output metrics are not truncated: every output column is included
in the averages above. Charts may display one selected output at a time,
but the underlying metrics cover all outputs.

Binary targets must be exactly $0.0$ or $1.0$; any other value is
rejected with a clear error.

Source: `../src/neural_network_lab/application/evaluation_service.py`
(`evaluate`, `EvaluationResult`, `CLASSIFICATION_THRESHOLD`).

## 12. Capabilities not implemented

The initial release intentionally does not include:

- Activations other than sigmoid.
- Losses other than MSE.
- Optimizers other than full-batch gradient descent (no Adam, no
  momentum, no adaptive methods).
- Mini-batch or stochastic (per-sample) training.
- Convolutional or recurrent networks.
- Multi-class classification beyond binary targets and one-hot-style
  continuous targets.
- Deep learning frameworks such as PyTorch or TensorFlow.

These are not absent by accident; they are outside the documented initial
scope (`docs/PROJECT_SPEC.md` §3 and §14).
