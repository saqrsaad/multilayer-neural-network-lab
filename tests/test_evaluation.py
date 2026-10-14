"""Unit tests for the evaluation service (read-only, no training)."""

import math

import numpy as np
import pytest

from neural_network_lab.application.evaluation_service import (
    CLASSIFICATION_THRESHOLD,
)
from neural_network_lab.application.evaluation_service import evaluate
from neural_network_lab.core.network import NeuralNetwork
from neural_network_lab.data.dataset import Dataset


def _scalar_sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def _explicit_network():
    return NeuralNetwork(
        [2, 2, 1],
        weights=[np.array([[0.5, -0.5], [0.25, 0.75]]), np.array([[1.0], [-1.0]])],
        biases=[np.array([[0.0, 0.1]]), np.array([[0.25]])],
    )


def _manual_predictions(network, X):
    weights = network.weights
    biases = network.biases
    out = []
    for row in X:
        hidden = [
            _scalar_sigmoid(row[0] * weights[0][0][j] + row[1] * weights[0][1][j] + biases[0][0][j])
            for j in range(2)
        ]
        z = hidden[0] * weights[1][0][0] + hidden[1] * weights[1][1][0] + biases[1][0][0]
        out.append([_scalar_sigmoid(z)])
    return np.array(out, dtype=float)


def test_evaluate_known_example():
    network = _explicit_network()
    features = [[1.0, 0.5], [0.0, 0.0], [1.0, 1.0]]
    targets = [[1.0], [0.0], [1.0]]
    dataset = Dataset(features, targets)
    expected_predictions = _manual_predictions(network, features)
    expected_mse = float(np.mean((np.array(targets) - expected_predictions) ** 2))
    expected_accuracy = float(
        np.mean((expected_predictions >= 0.5).astype(float) == np.array(targets))
    )
    result = evaluate(network, dataset, dataset_name="test-set")
    np.testing.assert_allclose(result.predictions, expected_predictions, rtol=1e-12)
    np.testing.assert_allclose(result.targets, np.array(targets), rtol=0, atol=0)
    assert result.mse == pytest.approx(expected_mse)
    assert result.accuracy == pytest.approx(expected_accuracy)
    assert result.n_samples == 3
    assert result.dataset_name == "test-set"


def test_threshold_boundary_is_documented():
    assert CLASSIFICATION_THRESHOLD == 0.5
    # Zero weights/biases give sigmoid(0) == 0.5, which counts as class 1.
    network = NeuralNetwork(
        [1, 1], weights=[np.array([[0.0]])], biases=[np.array([[0.0]])]
    )
    dataset = Dataset([[0.0], [1.0]], [[1.0], [0.0]])
    result = evaluate(network, dataset)
    np.testing.assert_allclose(result.predictions, np.array([[0.5], [0.5]]), rtol=1e-12)
    assert result.mse == pytest.approx(0.25)
    assert result.accuracy == pytest.approx(0.5)
    assert result.dataset_name is None


def test_evaluate_never_modifies_parameters():
    network = _explicit_network()
    dataset = Dataset([[1.0, 0.5], [0.0, 1.0]], [[1.0], [0.0]])
    before_weights = [w.copy() for w in network.weights]
    before_biases = [b.copy() for b in network.biases]
    evaluate(network, dataset)
    for old, current in zip(before_weights, network.weights):
        np.testing.assert_array_equal(old, current)
    for old, current in zip(before_biases, network.biases):
        np.testing.assert_array_equal(old, current)


def test_evaluate_rejects_unlabeled_dataset():
    network = _explicit_network()
    with pytest.raises(ValueError):
        evaluate(network, Dataset([[1.0, 0.5]]))


def test_evaluate_rejects_incompatible_dataset():
    network = _explicit_network()
    with pytest.raises(ValueError):
        evaluate(network, Dataset(np.zeros((2, 3)), np.zeros((2, 1))))
    with pytest.raises(ValueError):
        evaluate(network, Dataset(np.zeros((2, 2)), np.zeros((2, 2))))


def test_evaluate_rejects_non_binary_targets():
    network = _explicit_network()
    with pytest.raises(ValueError):
        evaluate(network, Dataset([[1.0, 0.5]], [[0.5]]))


def test_evaluate_rejects_wrong_types():
    network = _explicit_network()
    dataset = Dataset([[1.0, 0.5]], [[1.0]])
    with pytest.raises(TypeError):
        evaluate("not-a-network", dataset)
    with pytest.raises(TypeError):
        evaluate(network, [[1.0]])
    with pytest.raises(TypeError):
        evaluate(network, dataset, dataset_name=123)
