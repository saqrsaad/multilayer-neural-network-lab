"""Unit tests for sigmoid activation and its derivative."""

import math

import numpy as np
import pytest

from neural_network_lab.core.activations import sigmoid, sigmoid_derivative


def test_sigmoid_zero():
    assert sigmoid(0.0) == pytest.approx(0.5)


def test_sigmoid_representative_values():
    z = np.array([-1.0, 0.0, 1.0])
    expected = np.array(
        [1.0 / (1.0 + math.exp(1.0)), 0.5, 1.0 / (1.0 + math.exp(-1.0))]
    )
    np.testing.assert_allclose(sigmoid(z), expected, rtol=1e-12)


def test_sigmoid_output_range_and_shape():
    z = np.array([[-2.0, -0.5], [0.5, 2.0]])
    out = sigmoid(z)
    assert out.shape == (2, 2)
    assert np.all(out > 0.0) and np.all(out < 1.0)


def test_sigmoid_accepts_lists_and_scalars():
    np.testing.assert_allclose(sigmoid([0.0]), np.array([0.5]))
    out = sigmoid(-1000.0)
    assert np.asarray(out).shape == ()
    assert float(out) == pytest.approx(0.0)


def test_sigmoid_extreme_values_no_overflow():
    with np.errstate(over="raise"):
        out = sigmoid(np.array([-1000.0, 1000.0]))
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(1.0)
    assert np.all(np.isfinite(out))


def test_sigmoid_derivative_at_zero():
    assert sigmoid_derivative(0.0) == pytest.approx(0.25)


def test_sigmoid_derivative_matches_formula():
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    s = sigmoid(z)
    np.testing.assert_allclose(sigmoid_derivative(z), s * (1.0 - s), rtol=1e-12)


def test_sigmoid_derivative_extreme_values():
    out = sigmoid_derivative(np.array([-1000.0, 1000.0]))
    np.testing.assert_allclose(out, np.array([0.0, 0.0]), atol=1e-12)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_sigmoid_rejects_non_finite_scalar(bad):
    with pytest.raises(ValueError):
        sigmoid(bad)


def test_sigmoid_rejects_non_finite_array():
    with pytest.raises(ValueError):
        sigmoid(np.array([0.0, np.nan]))
    with pytest.raises(ValueError):
        sigmoid(np.array([0.0, np.inf]))


def test_sigmoid_derivative_rejects_non_finite():
    with pytest.raises(ValueError):
        sigmoid_derivative(np.array([1.0, np.inf]))


def test_sigmoid_rejects_empty():
    with pytest.raises(ValueError):
        sigmoid([])


def test_sigmoid_rejects_non_numeric():
    with pytest.raises(TypeError):
        sigmoid(["a", "b"])
