"""Loss calculations for the neural network core.

Initial release supports Mean Squared Error (MSE) per
``docs/PROJECT_SPEC.md``. Core code uses NumPy and stays independent
of the GUI, application, and storage layers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _as_finite_float_array(values: ArrayLike, name: str) -> np.ndarray:
    """Convert input to a float array and reject non-finite values."""
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric array-like: {exc}") from exc
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values (no NaN or inf)")
    return arr


def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute Mean Squared Error.

    MSE = mean((targets - predictions) ** 2)

    Args:
        y_true: Target values, any non-empty shape.
        y_pred: Predicted values, must have the same shape as ``y_true``.

    Returns:
        The mean squared error as a Python float.

    Raises:
        TypeError: If either input cannot be converted to a numeric array.
        ValueError: If shapes differ, either input is empty,
            or any value is NaN/inf.
    """
    targets = _as_finite_float_array(y_true, "y_true")
    predictions = _as_finite_float_array(y_pred, "y_pred")
    if targets.shape != predictions.shape:
        raise ValueError(
            f"y_true shape {targets.shape} must match y_pred shape {predictions.shape}"
        )
    return float(np.mean((targets - predictions) ** 2))


# Short alias used across the codebase and tests.
mse = mean_squared_error
