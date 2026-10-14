"""Unit tests for forward propagation (no backpropagation/training)."""

import math

import numpy as np
import pytest

from neural_network_lab.core.network import NeuralNetwork


def _scalar_sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _manual_forward_2_2_1(X, W1, b1, W2, b2):
    """Independent oracle using scalar math (no core imports)."""
    out = []
    for row in X:
        h = []
        for j in range(2):
            z = row[0] * W1[0][j] + row[1] * W1[1][j] + b1[0][j]
            h.append(_scalar_sigmoid(z))
        z_out = h[0] * W2[0][0] + h[1] * W2[1][0] + b2[0][0]
        out.append([_scalar_sigmoid(z_out)])
    return np.array(out, dtype=float)


@pytest.fixture
def small_network():
    W1 = np.array([[0.5, -0.5], [0.25, 0.75]])
    b1 = np.array([[0.0, 0.1]])
    W2 = np.array([[1.0], [-1.0]])
    b2 = np.array([[0.25]])
    net = NeuralNetwork([2, 2, 1], weights=[W1, W2], biases=[b1, b2])
    return net, (W1, b1, W2, b2)


def test_init_parameter_shapes_follow_conventions():
    net = NeuralNetwork([3, 4, 2, 1], seed=0)
    assert net.layer_sizes == (3, 4, 2, 1)
    assert net.n_layers == 3
    assert [w.shape for w in net.weights] == [(3, 4), (4, 2), (2, 1)]
    assert [b.shape for b in net.biases] == [(1, 4), (1, 2), (1, 1)]
    assert all(np.all(np.isfinite(w)) for w in net.weights)
    assert all(np.all(b == 0.0) for b in net.biases)


def test_init_seed_reproducible():
    net_a = NeuralNetwork([2, 3, 1], seed=42)
    net_b = NeuralNetwork([2, 3, 1], seed=42)
    for wa, wb in zip(net_a.weights, net_b.weights):
        np.testing.assert_array_equal(wa, wb)


def test_explicit_parameters_stored_as_copies():
    W1 = np.array([[0.5, -0.5], [0.25, 0.75]])
    b1 = np.array([[0.0, 0.1]])
    W2 = np.array([[1.0], [-1.0]])
    b2 = np.array([[0.25]])
    net = NeuralNetwork([2, 2, 1], weights=[W1, W2], biases=[b1, b2])
    W1[0, 0] = 999.0
    np.testing.assert_allclose(net.weights[0][0, 0], 0.5)


def test_forward_output_shape():
    net = NeuralNetwork([3, 5, 4, 2], seed=1)
    X = np.ones((7, 3))
    out = net.forward(X)
    assert out.shape == (7, 2)
    assert np.all(out > 0.0) and np.all(out < 1.0)


def test_forward_matches_manual_example(small_network):
    net, (W1, b1, W2, b2) = small_network
    X = np.array([[1.0, 0.5], [0.0, 0.0]])
    expected = _manual_forward_2_2_1(X.tolist(), W1.tolist(), b1.tolist(),
                                     W2.tolist(), b2.tolist())
    np.testing.assert_allclose(net.forward(X), expected, rtol=1e-12)


def test_forward_single_sample(small_network):
    net, params = small_network
    W1, b1, W2, b2 = params
    X = np.array([[1.0, 0.5]])
    expected = _manual_forward_2_2_1(X.tolist(), W1.tolist(), b1.tolist(),
                                     W2.tolist(), b2.tolist())
    np.testing.assert_allclose(net.forward(X), expected, rtol=1e-12)


def test_forward_with_cache_structure(small_network):
    net, _ = small_network
    X = np.array([[1.0, 0.5], [0.0, 1.0]])
    output, activations, pre_activations = net.forward_with_cache(X)
    assert len(activations) == 3  # A0, A1, A2
    assert len(pre_activations) == 2  # Z0, Z1
    np.testing.assert_array_equal(activations[0], X)
    assert pre_activations[0].shape == (2, 2)
    assert pre_activations[1].shape == (2, 1)
    assert activations[1].shape == (2, 2)
    np.testing.assert_allclose(output, activations[-1], rtol=0, atol=0)
    np.testing.assert_allclose(net.forward(X), output, rtol=0, atol=0)


def test_forward_does_not_modify_parameters(small_network):
    net, _ = small_network
    before_w = [w.copy() for w in net.weights]
    before_b = [b.copy() for b in net.biases]
    net.forward(np.array([[0.2, -0.3], [0.9, 0.1]]))
    for backend, after in zip(before_w, net.weights):
        np.testing.assert_array_equal(backend, after)
    for backend, after in zip(before_b, net.biases):
        np.testing.assert_array_equal(backend, after)


@pytest.mark.parametrize(
    "bad_sizes",
    [[3], [2], [], [2, 0], [2, -1], [0, 1], [2, 3, 0]],
)
def test_init_rejects_invalid_sizes_value(bad_sizes):
    with pytest.raises(ValueError):
        NeuralNetwork(bad_sizes)


@pytest.mark.parametrize("bad_sizes", [[2.5, 3], ["3", 2], [2, None], "32", 32])
def test_init_rejects_invalid_sizes_type(bad_sizes):
    with pytest.raises(TypeError):
        NeuralNetwork(bad_sizes)


def test_init_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        NeuralNetwork([2, 2, 1],
                      weights=[np.zeros((2, 2)), np.zeros((3, 1))],
                      biases=[np.zeros((1, 2)), np.zeros((1, 1))])
    with pytest.raises(ValueError):
        NeuralNetwork([2, 2, 1],
                      weights=[np.zeros((2, 2)), np.zeros((2, 1))],
                      biases=[np.zeros((1, 2)), np.zeros((2, 1))])
    with pytest.raises(ValueError):
        NeuralNetwork([2, 2, 1], weights=[np.zeros((2, 2))],
                      biases=[np.zeros((1, 2))])


def test_init_rejects_partial_and_non_finite_parameters():
    with pytest.raises(ValueError):
        NeuralNetwork([2, 2, 1], weights=[np.zeros((2, 2)), np.zeros((2, 1))])
    bad_w = [np.array([[np.nan, 0.0], [0.0, 0.0]]), np.zeros((2, 1))]
    with pytest.raises(ValueError):
        NeuralNetwork([2, 2, 1], weights=bad_w,
                      biases=[np.zeros((1, 2)), np.zeros((1, 1))])


def test_forward_rejects_wrong_dimensions(small_network):
    net, _ = small_network
    with pytest.raises(ValueError):
        net.forward(np.array([1.0, 0.5]))  # 1D: must not be silently reshaped
    with pytest.raises(ValueError):
        net.forward(np.zeros((2, 2, 2)))  # 3D
    with pytest.raises(ValueError):
        net.forward(np.zeros((4, 3)))  # wrong feature count
    with pytest.raises(ValueError):
        net.forward(np.zeros((0, 2)))  # no samples


def test_forward_rejects_non_finite_and_non_numeric(small_network):
    net, _ = small_network
    with pytest.raises(ValueError):
        net.forward(np.array([[np.nan, 0.0]]))
    with pytest.raises(ValueError):
        net.forward(np.array([[np.inf, 0.0]]))
    with pytest.raises(TypeError):
        net.forward([["a", "b"]])


def _parameter_snapshot(net):
    return ([w.copy() for w in net.weights], [b.copy() for b in net.biases])


def _assert_parameters_equal(net, snapshot):
    for current, expected in zip(net.weights, snapshot[0]):
        np.testing.assert_array_equal(current, expected)
    for current, expected in zip(net.biases, snapshot[1]):
        np.testing.assert_array_equal(current, expected)


def test_set_parameters_updates_all_layers(small_network):
    net, _ = small_network
    new_weights = [np.full_like(w, 0.25) for w in net.weights]
    new_biases = [np.full_like(b, -0.5) for b in net.biases]
    net.set_parameters(new_weights, new_biases)
    _assert_parameters_equal(net, (new_weights, new_biases))


def test_set_parameters_rejects_shape_mismatch_atomically(small_network):
    net, _ = small_network
    before = _parameter_snapshot(net)
    bad_weights = [w.copy() for w in before[0]]
    bad_weights[1] = np.zeros((5, 5))
    with pytest.raises(ValueError):
        net.set_parameters(bad_weights, before[1])
    _assert_parameters_equal(net, before)


def test_set_parameters_rejects_non_finite_atomically(small_network):
    net, _ = small_network
    before = _parameter_snapshot(net)
    bad_biases = [b.copy() for b in before[1]]
    bad_biases[0][0, 0] = float("inf")
    with pytest.raises(ValueError):
        net.set_parameters(before[0], bad_biases)
    _assert_parameters_equal(net, before)


def test_set_parameters_rejects_count_mismatch_and_non_numeric(small_network):
    net, _ = small_network
    before = _parameter_snapshot(net)
    with pytest.raises(ValueError):
        net.set_parameters(before[0][:-1], before[1])
    _assert_parameters_equal(net, before)
    with pytest.raises(TypeError):
        net.set_parameters([[["a", "a"], ["a", "a"]], before[0][1]], before[1])
    _assert_parameters_equal(net, before)
