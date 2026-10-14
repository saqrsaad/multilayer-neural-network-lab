"""Full-batch gradient descent updates (initial optimizer).

This module holds the only optimizer supported by the initial release:
plain full-batch gradient descent, ``W = W - learning_rate * dL/dW``
(applied identically to biases). No other optimizers are introduced
here; later tasks must not assume additional update rules exist.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def validate_learning_rate(learning_rate: float) -> float:
    """Validate a learning rate and return it as a float.

    Args:
        learning_rate: Must be a finite number greater than zero.
            Strings, ``None``, NaN/inf, and non-positive values are
            rejected.

    Returns:
        The learning rate as a Python float.

    Raises:
        TypeError: If the value is not numeric.
        ValueError: If the value is NaN/inf or not greater than zero.
    """
    if isinstance(learning_rate, bool) or not isinstance(
        learning_rate, (int, float, np.floating, np.integer)
    ):
        raise TypeError("learning_rate must be a finite number greater than zero")
    try:
        value = float(learning_rate)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"learning_rate must be a finite number greater than zero: {exc}"
        ) from exc
    if not np.isfinite(value):
        raise ValueError("learning_rate must be finite (no NaN or inf)")
    if value <= 0.0:
        raise ValueError("learning_rate must be greater than zero")
    return value


def sgd_update(
    parameters: list[NDArray[np.float64]],
    gradients: list[NDArray[np.float64]],
    learning_rate: float,
) -> list[NDArray[np.float64]]:
    """Apply one full-batch gradient descent step to parameter arrays.

    Computes ``new_param = param - learning_rate * grad`` elementwise
    for each layer. Pure function: inputs are not modified.

    Args:
        parameters: Current parameter arrays (weights or biases).
        gradients: Gradient arrays with shapes matching ``parameters``.
        learning_rate: Positive, finite step size.

    Returns:
        New parameter arrays in the same order.

    Raises:
        TypeError: For non-numeric learning rates.
        ValueError: For count/shape mismatches, non-finite gradients,
            or invalid learning rates.
    """
    rate = validate_learning_rate(learning_rate)
    if len(parameters) != len(gradients):
        raise ValueError(
            f"expected {len(parameters)} gradient arrays, "
            f"got {len(gradients)}"
        )
    updated: list[NDArray[np.float64]] = []
    for index, (param, grad) in enumerate(zip(parameters, gradients)):
        param_arr = np.asarray(param, dtype=float)
        try:
            grad_arr = np.asarray(grad, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"gradients[{index}] must be numeric: {exc}") from exc
        if param_arr.shape != grad_arr.shape:
            raise ValueError(
                f"gradients[{index}] shape {grad_arr.shape} != "
                f"parameter shape {param_arr.shape}"
            )
        if not np.all(np.isfinite(grad_arr)):
            raise ValueError(
                f"gradients[{index}] must contain only finite values"
            )
        updated.append(param_arr - rate * grad_arr)
    return updated
