"""Unit tests for backpropagation gradients and parameter updates.

Gradient convention under test (see ``core/backpropagation.py``):
``MSE = mean((A - Y) ** 2)`` over all ``N * K`` elements,
``dL/dA = 2 * (A - Y) / (N * K)``, sigmoid ``A * (1 - A)`` factors, and
hidden deltas propagated through the pre-update weights.
"""

import math

import numpy as np
import pytest

from neural_network_lab.core.backpropagation import compute_gradients
from neural_network_lab.core.backpropagation import validate_targets
from neural_network_lab.core.losses import mean_squared_error
from neural_network_lab.core.network import NeuralNetwork


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@pytest.fixture
def small_network():
    W1 = np.array([[0.5, -0.5], [0.25, 0.75]])
    b1 = np.array([[0.0, 0.1]])
    W2 = np.array([[1.0], [-1.0]])
    b2 = np.array([[0.25]])
    net = NeuralNetwork([2, 2, 1], weights=[W1, W2], biases=[b1, b2])
    return net


def test_gradient_shapes_match_parameters():
    net = NeuralNetwork([3, 4, 2, 1], seed=0)
    X = np.ones((5, 3))
    Y = np.zeros((5, 1))
    weight_grads, bias_grads = compute_gradients(net, X, Y)
    assert [g.shape for g in weight_grads] == [(3, 4), (4, 2), (2, 1)]
    assert [g.shape for g in bias_grads] == [(1, 4), (1, 2), (1, 1)]


def test_single_neuron_hand_computed_gradients():
    net = NeuralNetwork(
        [1, 1],
        weights=[np.array([[0.5]])],
        biases=[np.array([[0.0]])],
    )
    X = np.array([[1.0]])
    Y = np.array([[0.8]])
    # Independent scalar oracle (no core math imports).
    a = 1.0 / (1.0 + math.exp(-0.5))
    dL_dA = 2.0 * (a - 0.8)
    delta = dL_dA * a * (1.0 - a)
    weight_grads, bias_grads = compute_gradients(net, X, Y)
    np.testing.assert_allclose(weight_grads[0], np.array([[delta]]), rtol=1e-12)
    np.testing.assert_allclose(bias_grads[0], np.array([[delta]]), rtol=1e-12)


def test_hidden_gradients_use_pre_update_weights(small_network):
    X = np.array([[1.0, 0.5], [0.0, 1.0]])
    Y = np.array([[1.0], [0.0]])
    before_w = [w.copy() for w in small_network.weights]
    before_b = [b.copy() for b in small_network.biases]

    weight_grads, bias_grads = compute_gradients(small_network, X, Y)

    # Parameters must be untouched by gradient computation.
    for old, current in zip(before_w, small_network.weights):
        np.testing.assert_array_equal(old, current)

    # Independent oracle built only from the pre-update snapshot.
    W1, W2 = before_w
    b1, b2 = before_b
    A0 = X
    A1 = _sigmoid(A0 @ W1 + b1)
    A2 = _sigmoid(A1 @ W2 + b2)
    dA = 2.0 * (A2 - Y) / (X.shape[0] * 1)
    d2 = dA * A2 * (1.0 - A2)
    d1 = (d2 @ W2.T) * A1 * (1.0 - A1)
    np.testing.assert_allclose(weight_grads[1], A1.T @ d2, rtol=1e-12)
    np.testing.assert_allclose(bias_grads[1], d2.sum(axis=0, keepdims=True), rtol=1e-12)
    np.testing.assert_allclose(weight_grads[0], A0.T @ d1, rtol=1e-12)
    np.testing.assert_allclose(bias_grads[0], d1.sum(axis=0, keepdims=True), rtol=1e-12)


def test_gradients_match_finite_differences():
    # Central differences with eps=1e-6; tolerance 1e-4 is far above the
    # expected ~1e-9 truncation+roundoff error and guards only against
    # convention errors (e.g. wrong MSE reduction or stale weights).
    net = NeuralNetwork([2, 3, 2], seed=3)
    X = np.array([[0.2, -0.4], [0.9, 0.1], [-0.5, 0.7]])
    Y = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    eps = 1e-6
    weight_grads, bias_grads = compute_gradients(net, X, Y)

    def loss_for(weights, biases):
        probe = NeuralNetwork(
            net.layer_sizes, weights=weights, biases=biases
        )
        return mean_squared_error(probe.forward(X), Y)

    for layer, (w, b) in enumerate(zip(net.weights, net.biases)):
        for idx in np.ndindex(w.shape):
            plus = [x.copy() for x in net.weights]
            minus = [x.copy() for x in net.weights]
            plus[layer][idx] += eps
            minus[layer][idx] -= eps
            numeric = (
                loss_for(plus, net.biases) - loss_for(minus, net.biases)
            ) / (2.0 * eps)
            assert weight_grads[layer][idx] == pytest.approx(numeric, rel=1e-4, abs=1e-6)
        for idx in np.ndindex(b.shape):
            plus = [x.copy() for x in net.biases]
            minus = [x.copy() for x in net.biases]
            plus[layer][idx] += eps
            minus[layer][idx] -= eps
            numeric = (
                loss_for(net.weights, plus) - loss_for(net.weights, minus)
            ) / (2.0 * eps)
            assert bias_grads[layer][idx] == pytest.approx(numeric, rel=1e-4, abs=1e-6)


def test_apply_gradients_follows_gradient_descent(small_network):
    X = np.array([[1.0, 0.5], [0.0, 1.0]])
    Y = np.array([[1.0], [0.0]])
    weight_grads, bias_grads = compute_gradients(small_network, X, Y)
    old_w = [w.copy() for w in small_network.weights]
    old_b = [b.copy() for b in small_network.biases]
    small_network.apply_gradients(weight_grads, bias_grads, 0.25)
    for old, grad, new in zip(old_w, weight_grads, small_network.weights):
        np.testing.assert_allclose(new, old - 0.25 * grad, rtol=1e-12)
    for old, grad, new in zip(old_b, bias_grads, small_network.biases):
        np.testing.assert_allclose(new, old - 0.25 * grad, rtol=1e-12)


def test_compute_gradients_rejects_bad_targets(small_network):
    X = np.array([[1.0, 0.5], [0.0, 1.0]])
    with pytest.raises(ValueError):
        compute_gradients(small_network, X, np.array([[1.0]]))  # wrong rows
    with pytest.raises(ValueError):
        compute_gradients(small_network, X, np.zeros((2, 2)))  # wrong cols
    with pytest.raises(ValueError):
        compute_gradients(small_network, X, np.array([1.0, 0.0]))  # 1D
    with pytest.raises(ValueError):
        compute_gradients(small_network, X, np.array([[np.nan], [0.0]]))
    with pytest.raises(TypeError):
        compute_gradients(small_network, X, [["a"], ["b"]])


def test_validate_targets_direct():
    assert validate_targets([[0.0], [1.0]], 2, 1).shape == (2, 1)
    with pytest.raises(ValueError):
        validate_targets(np.zeros((2, 2)), 2, 1)


@pytest.mark.parametrize("bad_lr", [0.0, -0.1, float("nan"), float("inf")])
def test_apply_gradients_rejects_bad_learning_rate(small_network, bad_lr):
    grads = ([np.zeros((2, 2)), np.zeros((2, 1))], [np.zeros((1, 2)), np.zeros((1, 1))])
    with pytest.raises(ValueError):
        small_network.apply_gradients(grads[0], grads[1], bad_lr)


@pytest.mark.parametrize("bad_lr", ["0.1", None, [0.1]])
def test_apply_gradients_rejects_non_numeric_learning_rate(small_network, bad_lr):
    grads = ([np.zeros((2, 2)), np.zeros((2, 1))], [np.zeros((1, 2)), np.zeros((1, 1))])
    with pytest.raises(TypeError):
        small_network.apply_gradients(grads[0], grads[1], bad_lr)


def test_apply_gradients_rejects_shape_mismatch(small_network):
    with pytest.raises(ValueError):
        small_network.apply_gradients(
            [np.zeros((2, 2))],
            [np.zeros((1, 2)), np.zeros((1, 1))],
            0.1,
        )
    with pytest.raises(ValueError):
        small_network.apply_gradients(
            [np.zeros((2, 3)), np.zeros((2, 1))],
            [np.zeros((1, 2)), np.zeros((1, 1))],
            0.1,
        )
