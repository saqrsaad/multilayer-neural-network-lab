"""Sigmoid activation function and its derivative.

Conventions follow ``docs/PROJECT_SPEC.md`` and ``docs/ARCHITECTURE.md``:
core mathematics implemented with NumPy, independent of the GUI.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_finite_float_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
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


def sigmoid(z: ArrayLike) -> NDArray[np.float64]:
    """Compute the sigmoid activation elementwise.

    sigma(z) = 1 / (1 + exp(-z))

    Numerically stable: the positive and negative branches are evaluated
    separately so ``exp`` never overflows for extreme inputs
    (e.g. ``z = +/-1000`` returns ``1.0`` / ``0.0`` instead of warning).

    Args:
        z: Pre-activation values, any shape. Lists/tuples/scalars are
            converted to a float array.

    Returns:
        Array of the same shape as ``z`` with values in (0, 1).

    Raises:
        TypeError: If ``z`` cannot be converted to a numeric array.
        ValueError: If ``z`` is empty or contains NaN/inf.
    """
    arr = _as_finite_float_array(z, "z")

    flat = arr.ravel()
    out_flat = np.empty_like(flat, dtype=float)
    positive = flat >= 0
    # Safe branch for z >= 0: exp(-z) in (0, 1].
    out_flat[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))
    # Safe branch for z < 0: exp(z) in (0, 1), never overflows.
    neg = flat[~positive]
    exp_neg = np.exp(neg)
    out_flat[~positive] = exp_neg / (1.0 + exp_neg)
    return out_flat.reshape(arr.shape)


def sigmoid_derivative(z: ArrayLike) -> NDArray[np.float64]:
    """Compute the sigmoid derivative elementwise from pre-activation values.

    This helper accepts **pre-activation values ``z``** (the inputs to
    :func:`sigmoid`), NOT already-computed sigmoid outputs. It evaluates::

        sigma_prime(z) = sigma(z) * (1 - sigma(z))

    Args:
        z: Pre-activation values, any shape.

    Returns:
        Array of the same shape as ``z`` with values in (0, 0.25].

    Raises:
        TypeError: If ``z`` cannot be converted to a numeric array.
        ValueError: If ``z`` is empty or contains NaN/inf.
    """
    s = sigmoid(z)
    return s * (1.0 - s)
