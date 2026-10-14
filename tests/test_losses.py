"""Unit tests for Mean Squared Error."""

import numpy as np
import pytest

from neural_network_lab.core.losses import mean_squared_error, mse


def test_mse_perfect_predictions():
    assert mean_squared_error([0.0, 1.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_mse_hand_calculated():
    # ((-1)^2 + (1)^2) / 2 = 1.0
    assert mean_squared_error([0.0, 1.0], [1.0, 0.0]) == pytest.approx(1.0)
    # ((1-2)^2 + (2-4)^2 + (3-6)^2) / 3 = (1 + 4 + 9) / 3
    assert mean_squared_error([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(
        14.0 / 3.0
    )


def test_mse_2d_arrays():
    y_true = np.array([[0.0, 1.0], [1.0, 0.0]])
    y_pred = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert mean_squared_error(y_true, y_pred) == pytest.approx(0.25)


def test_mse_returns_python_float():
    result = mean_squared_error([0.0], [1.0])
    assert isinstance(result, float)
    assert result == pytest.approx(1.0)


def test_mse_alias_matches():
    assert mse([0.0, 1.0], [1.0, 0.0]) == pytest.approx(
        mean_squared_error([0.0, 1.0], [1.0, 0.0])
    )


def test_mse_shape_mismatch():
    with pytest.raises(ValueError):
        mean_squared_error([0.0, 1.0], [0.0])
    with pytest.raises(ValueError):
        mean_squared_error(np.zeros((2, 2)), np.zeros((2, 3)))


def test_mse_rejects_empty():
    with pytest.raises(ValueError):
        mean_squared_error([], [])
    with pytest.raises(ValueError):
        mean_squared_error(np.array([]), np.array([]))


@pytest.mark.parametrize(
    "y_true,y_pred",
    [
        ([0.0, float("nan")], [0.0, 1.0]),
        ([0.0, 1.0], [0.0, float("inf")]),
        ([float("-inf")], [0.0]),
    ],
)
def test_mse_rejects_non_finite(y_true, y_pred):
    with pytest.raises(ValueError):
        mean_squared_error(y_true, y_pred)


def test_mse_rejects_non_numeric():
    with pytest.raises(TypeError):
        mean_squared_error(["a"], ["b"])
