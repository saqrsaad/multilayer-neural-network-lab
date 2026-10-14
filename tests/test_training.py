"""Unit tests for training modes and training history.

Covers continued training (preserve current parameters as the starting
point) and training from scratch (explicit reinitialization first).
No GUI, evaluation, or persistence behavior is tested here.
"""

import numpy as np
import pytest

from neural_network_lab.core.backpropagation import compute_gradients
from neural_network_lab.core.losses import mean_squared_error
from neural_network_lab.core.network import NeuralNetwork

X_DATA = np.array(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5], [1.0, 0.5]]
)
Y_DATA = np.array([[0.0], [1.0], [1.0], [0.0], [0.5], [1.0]])


def _explicit_network():
    W1 = np.array([[0.5, -0.5], [0.25, 0.75]])
    b1 = np.array([[0.0, 0.1]])
    W2 = np.array([[1.0], [-1.0]])
    b2 = np.array([[0.25]])
    return NeuralNetwork([2, 2, 1], weights=[W1, W2], biases=[b1, b2])


def test_continued_training_records_history():
    net = NeuralNetwork([2, 3, 1], seed=11)
    before = [w.copy() for w in net.weights]
    result = net.train(X_DATA, Y_DATA, epochs=5, learning_rate=0.5)
    assert result.epochs_completed == 5
    assert result.mode == "continued"
    assert result.learning_rate == pytest.approx(0.5)
    assert len(result.loss_history) == 5
    assert all(isinstance(v, float) for v in result.loss_history)
    assert all(np.isfinite(result.loss_history))
    assert result.final_loss == pytest.approx(result.loss_history[-1])
    # History holds actual measured values: last entry equals fresh MSE.
    assert result.loss_history[-1] == pytest.approx(
        mean_squared_error(net.forward(X_DATA), Y_DATA)
    )
    # Parameters actually moved from their starting point.
    assert any(
        not np.array_equal(old, new)
        for old, new in zip(before, net.weights)
    )


def test_continued_training_starts_from_current_parameters():
    net = _explicit_network()
    weight_grads, bias_grads = compute_gradients(net, X_DATA, Y_DATA)
    old_w = [w.copy() for w in net.weights]
    old_b = [b.copy() for b in net.biases]
    learning_rate = 0.1
    net.train(X_DATA, Y_DATA, epochs=1, learning_rate=learning_rate)
    for old, grad, new in zip(old_w, weight_grads, net.weights):
        np.testing.assert_allclose(new, old - learning_rate * grad, rtol=1e-12)
    for old, grad, new in zip(old_b, bias_grads, net.biases):
        np.testing.assert_allclose(new, old - learning_rate * grad, rtol=1e-12)


def test_from_scratch_reinitializes_then_trains():
    net = _explicit_network()
    snapshot_w = [w.copy() for w in net.weights]
    result = net.train_from_scratch(
        X_DATA, Y_DATA, epochs=2, learning_rate=0.5, seed=7
    )
    assert result.mode == "from_scratch"
    assert result.epochs_completed == 2
    assert len(result.loss_history) == 2
    assert all(np.isfinite(result.loss_history))
    # Explicit starting parameters were replaced, not continued.
    assert any(
        not np.array_equal(old, new)
        for old, new in zip(snapshot_w, net.weights)
    )
    assert result.loss_history[-1] == pytest.approx(
        mean_squared_error(net.forward(X_DATA), Y_DATA)
    )


def test_from_scratch_is_deterministic_for_seed():
    first = _explicit_network()
    second = _explicit_network()
    result_a = first.train_from_scratch(
        X_DATA, Y_DATA, epochs=3, learning_rate=0.5, seed=7
    )
    result_b = second.train_from_scratch(
        X_DATA, Y_DATA, epochs=3, learning_rate=0.5, seed=7
    )
    np.testing.assert_allclose(result_a.loss_history, result_b.loss_history, rtol=1e-12)
    for wa, wb in zip(first.weights, second.weights):
        np.testing.assert_array_equal(wa, wb)


def test_from_scratch_validates_before_resetting():
    net = _explicit_network()
    snapshot_w = [w.copy() for w in net.weights]
    snapshot_b = [b.copy() for b in net.biases]
    with pytest.raises(ValueError):
        net.train_from_scratch(X_DATA, Y_DATA, epochs=0, learning_rate=0.5)
    with pytest.raises(ValueError):
        net.train_from_scratch(X_DATA, Y_DATA, epochs=2, learning_rate=-1.0)
    for old, current in zip(snapshot_w, net.weights):
        np.testing.assert_array_equal(old, current)
    for old, current in zip(snapshot_b, net.biases):
        np.testing.assert_array_equal(old, current)


@pytest.mark.parametrize("bad_epochs", [0, -2, 2.5, "3", True, None])
def test_train_rejects_bad_epochs(bad_epochs):
    net = NeuralNetwork([2, 2, 1], seed=0)
    with pytest.raises((TypeError, ValueError)):
        net.train(X_DATA, Y_DATA, epochs=bad_epochs, learning_rate=0.1)
    with pytest.raises((TypeError, ValueError)):
        net.train_from_scratch(X_DATA, Y_DATA, epochs=bad_epochs, learning_rate=0.1)


@pytest.mark.parametrize("bad_lr", [0.0, -0.5, float("nan"), float("inf")])
def test_train_rejects_bad_learning_rate(bad_lr):
    net = NeuralNetwork([2, 2, 1], seed=0)
    with pytest.raises(ValueError):
        net.train(X_DATA, Y_DATA, epochs=1, learning_rate=bad_lr)


@pytest.mark.parametrize("bad_lr", ["0.1", None, [0.1]])
def test_train_rejects_non_numeric_learning_rate(bad_lr):
    net = NeuralNetwork([2, 2, 1], seed=0)
    with pytest.raises(TypeError):
        net.train(X_DATA, Y_DATA, epochs=1, learning_rate=bad_lr)


def test_train_rejects_mismatched_data():
    net = NeuralNetwork([2, 2, 1], seed=0)
    with pytest.raises(ValueError):
        net.train(X_DATA, Y_DATA[:3], epochs=1, learning_rate=0.1)
    with pytest.raises(ValueError):
        net.train(X_DATA, np.zeros((6, 2)), epochs=1, learning_rate=0.1)
    with pytest.raises(ValueError):
        net.train(np.zeros((6, 3)), Y_DATA, epochs=1, learning_rate=0.1)
    with pytest.raises(ValueError):
        net.train(
            np.array([[np.nan, 0.0]]), np.array([[0.0]]), epochs=1, learning_rate=0.1
        )
