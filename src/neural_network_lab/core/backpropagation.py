"""Backpropagation for the sigmoid MLP with Mean Squared Error.

Gradient convention (consistent with ``losses.mean_squared_error``,
which averages over all ``N * K`` output elements):

- ``dL/dA = 2 * (A - Y) / (N * K)`` for output activations ``A``,
  targets ``Y``, batch size ``N`` and output size ``K``.
- Output delta: ``delta_L = dL/dA * A * (1 - A)``.
- Hidden delta: ``delta_l = (delta_{l+1} @ W_{l+1}.T) * A_l * (1 - A_l)``
  using the **pre-update** weights ``W_{l+1}``.
- ``dL/dW_l = A_l.T @ delta_l`` with shape ``(n_l, n_(l+1))``.
- ``dL/db_l = sum(delta_l, axis=0, keepdims=True)``
  with shape ``(1, n_(l+1))``.

``compute_gradients`` is pure: it never modifies the network. All layer
gradients are derived from the same pre-update parameter state before
any update is applied elsewhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from neural_network_lab.core.network import NeuralNetwork


def validate_targets(
    y: ArrayLike, n_samples: int, n_outputs: int
) -> NDArray[np.float64]:
    """Validate supervised targets against a batch shape.

    Args:
        y: Target values, must be 2D ``(n_samples, n_outputs)``.
        n_samples: Expected row count (must match the input batch).
        n_outputs: Expected column count (network output size).

    Returns:
        Targets as a float array.

    Raises:
        TypeError: If ``y`` cannot be converted to a numeric array.
        ValueError: If ``y`` is not 2D, is empty, has a mismatched
            shape, or contains NaN/inf.
    """
    try:
        arr = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"y must be numeric array-like: {exc}") from exc
    if arr.ndim != 2:
        raise ValueError(
            f"y must be 2D with shape (number_of_samples, {n_outputs}), "
            f"got shape {arr.shape}"
        )
    if arr.shape[0] < 1:
        raise ValueError("y must contain at least one sample")
    if arr.shape[0] != n_samples:
        raise ValueError(
            f"y row count {arr.shape[0]} != input batch size {n_samples}"
        )
    if arr.shape[1] != n_outputs:
        raise ValueError(
            f"y column count {arr.shape[1]} != network output size {n_outputs}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("y must contain only finite values (no NaN or inf)")
    return arr


def compute_gradients(
    network: NeuralNetwork, X: ArrayLike, y: ArrayLike
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Compute full-batch weight and bias gradients without updating anything.

    Every layer's delta is calculated from the same pre-update
    parameter state: the weights are snapshotted before the backward
    pass, so hidden-layer deltas always use the original ``W_{l+1}``.

    Args:
        network: Network whose current parameters define the gradients.
        X: Batch input of shape ``(number_of_samples, n0)``.
        y: Targets of shape ``(number_of_samples, nL)``.

    Returns:
        Tuple ``(weight_grads, bias_grads)`` in layer order, with
        ``weight_grads[l].shape == (n_l, n_(l+1))`` and
        ``bias_grads[l].shape == (1, n_(l+1))``.

    Raises:
        TypeError: For non-numeric inputs or targets.
        ValueError: For dimension mismatches or non-finite values.
    """
    output, activations, _ = network.forward_with_cache(X)
    n_samples, n_outputs = output.shape
    targets = validate_targets(y, n_samples, n_outputs)

    normalizer = float(n_samples * n_outputs)
    dL_dA = 2.0 * (output - targets) / normalizer

    # Snapshot of the pre-update weights; hidden deltas must use these.
    weights = network.weights
    n_layers = len(weights)

    deltas: list[NDArray[np.float64] | None] = [None] * n_layers
    deltas[-1] = dL_dA * output * (1.0 - output)
    for layer in range(n_layers - 2, -1, -1):
        activation = activations[layer + 1]
        propagated = deltas[layer + 1] @ weights[layer + 1].T
        deltas[layer] = propagated * activation * (1.0 - activation)

    weight_grads = [activations[layer].T @ deltas[layer] for layer in range(n_layers)]
    bias_grads = [
        np.sum(deltas[layer], axis=0, keepdims=True) for layer in range(n_layers)
    ]
    return weight_grads, bias_grads
