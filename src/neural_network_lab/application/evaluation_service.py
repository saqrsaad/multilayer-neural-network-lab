"""Evaluation utilities for testing a network without training it.

Evaluation is strictly read-only: it runs forward propagation through
:mod:`neural_network_lab.core.network`, scores the predictions with
:mod:`neural_network_lab.core.losses`, and never updates weights or
biases. Only labeled datasets compatible with the network are
evaluated. This layer coordinates the workflow and returns structured
results; it contains no GUI components.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neural_network_lab.core.losses import mean_squared_error
from neural_network_lab.core.network import NeuralNetwork
from neural_network_lab.data.dataset import Dataset

#: Decision boundary for binary accuracy: a prediction is class 1 when
#: it is greater than or equal to this value, otherwise class 0.
CLASSIFICATION_THRESHOLD = 0.5


@dataclass(frozen=True, eq=False)
class EvaluationResult:
    """Structured outcome of evaluating a network on a labeled dataset.

    ``predictions`` and ``targets`` both have shape
    ``(number_of_samples, number_of_outputs)``. ``mse`` is the Mean
    Squared Error and ``accuracy`` the fraction of outputs whose
    predicted class (threshold :data:`CLASSIFICATION_THRESHOLD`)
    matches the binary target. ``eq=False`` because array fields do
    not support value equality.
    """

    predictions: NDArray[np.float64]
    targets: NDArray[np.float64]
    mse: float
    accuracy: float
    n_samples: int
    dataset_name: str | None = None


def _require_binary_targets(targets: NDArray[np.float64]) -> None:
    """Reject non-binary targets with a clear error."""
    if not np.all((targets == 0.0) | (targets == 1.0)):
        offending = int(np.sum(~((targets == 0.0) | (targets == 1.0))))
        raise ValueError(
            "evaluation accuracy requires binary targets (0.0 or 1.0) "
            f"for binary classification; found {offending} other value(s)"
        )


def evaluate(
    network: NeuralNetwork, dataset: Dataset, *, dataset_name: str | None = None
) -> EvaluationResult:
    """Evaluate a network on a labeled, compatible dataset.

    Args:
        network: Network to test; its parameters are never modified.
        dataset: Labeled dataset whose feature columns match the
            network input size and whose target columns match the
            network output size.
        dataset_name: Optional identifier recorded on the result
            (e.g. which test dataset was used).

    Returns:
        :class:`EvaluationResult` with predictions, targets, MSE, and
        binary accuracy at :data:`CLASSIFICATION_THRESHOLD`.

    Raises:
        TypeError: If ``network``/``dataset`` have the wrong types or
            ``dataset_name`` is neither ``None`` nor a string.
        ValueError: If the dataset is unlabeled, incompatible with the
            network, or has non-binary targets.
    """
    if not isinstance(network, NeuralNetwork):
        raise TypeError(
            f"network must be a NeuralNetwork, got {type(network).__name__}"
        )
    if not isinstance(dataset, Dataset):
        raise TypeError(f"dataset must be a Dataset, got {type(dataset).__name__}")
    if dataset_name is not None and not isinstance(dataset_name, str):
        raise TypeError("dataset_name must be a string or None")
    layer_sizes = network.layer_sizes
    dataset.validate_compatible_with(layer_sizes[0], layer_sizes[-1])
    features, targets = dataset.to_arrays()
    assert targets is not None  # narrowed by the compatibility check above
    _require_binary_targets(targets)
    predictions = network.forward(features)
    loss = mean_squared_error(targets, predictions)
    predicted_classes = (predictions >= CLASSIFICATION_THRESHOLD).astype(float)
    accuracy = float(np.mean(predicted_classes == targets))
    return EvaluationResult(
        predictions=predictions,
        targets=targets,
        mse=loss,
        accuracy=accuracy,
        n_samples=dataset.n_samples,
        dataset_name=dataset_name,
    )
