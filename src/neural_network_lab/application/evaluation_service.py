"""Evaluation utilities for testing a network without training it.

Evaluation is strictly read-only: it runs forward propagation through
:mod:`neural_network_lab.core.network`, scores the predictions with
:mod:`neural_network_lab.core.losses`, and never updates weights or
biases. Only labeled datasets compatible with the network are
evaluated. This layer coordinates the workflow and returns structured
results; it contains no GUI components.

The dataset declares its task (:mod:`neural_network_lab.data.dataset`):

- ``binary`` targets must be 0.0 or 1.0, and the result reports
  threshold accuracy at :data:`CLASSIFICATION_THRESHOLD` in addition to
  the error measures.
- ``continuous`` targets keep their exact values, no thresholding or
  rounding happens anywhere, and accuracy is reported as ``None``
  because a class label is not defined for them.

Mean Squared Error is the training loss of this project, so MSE is
always reported; MAE and RMSE are derived from the same residuals, and
all three are averaged over every sample **and** every output, which
keeps them well-defined for multi-output models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neural_network_lab.core.losses import mean_squared_error
from neural_network_lab.core.network import NeuralNetwork
from neural_network_lab.data.dataset import BINARY_TARGETS
from neural_network_lab.data.dataset import TARGET_KIND_LABELS
from neural_network_lab.data.dataset import Dataset

#: Decision boundary for binary accuracy: a prediction is class 1 when
#: it is greater than or equal to this value, otherwise class 0.
CLASSIFICATION_THRESHOLD = 0.5


@dataclass(frozen=True, eq=False)
class EvaluationResult:
    """Structured outcome of evaluating a network on a labeled dataset.

    ``predictions`` and ``targets`` both have shape
    ``(number_of_samples, number_of_outputs)``. ``mse`` is the Mean
    Squared Error used as the training loss, ``mae`` the Mean Absolute
    Error and ``rmse`` the Root Mean Squared Error, all averaged over
    every sample and output. ``accuracy`` is the fraction of outputs
    whose thresholded class matches a binary target, and is ``None``
    for continuous targets, which have no class. ``target_kind`` is the
    task declared by the dataset, which tells the GUI which metrics and
    labels apply. ``eq=False`` because array fields do not support
    value equality.
    """

    predictions: NDArray[np.float64]
    targets: NDArray[np.float64]
    mse: float
    mae: float
    rmse: float
    n_samples: int
    accuracy: float | None = None
    target_kind: str = BINARY_TARGETS
    dataset_name: str | None = None

    @property
    def task_label(self) -> str:
        """Human-readable name of the task these metrics belong to."""
        return TARGET_KIND_LABELS.get(self.target_kind, self.target_kind)

    @property
    def n_outputs(self) -> int:
        """Number of outputs the predictions and targets carry."""
        return int(self.predictions.shape[1]) if self.predictions.ndim == 2 else 0


def _require_binary_targets(targets: NDArray[np.float64]) -> None:
    """Reject non-binary targets with a clear error.

    Only called for datasets that declare binary classification; a
    continuous dataset never goes through this check.
    """
    if not np.all((targets == 0.0) | (targets == 1.0)):
        offending = int(np.sum(~((targets == 0.0) | (targets == 1.0))))
        raise ValueError(
            "binary classification requires targets of 0.0 or 1.0; found "
            f"{offending} other value(s). If these targets are continuous, "
            "declare the dataset's task as continuous regression."
        )


def evaluate(
    network: NeuralNetwork,
    dataset: Dataset,
    *,
    dataset_name: str | None = None,
) -> EvaluationResult:
    """Evaluate a network on a labeled, compatible dataset.

    Args:
        network: Network to test; its parameters are never modified.
        dataset: Labeled dataset whose feature columns match the
            network input size and whose target columns match the
            network output size. Its declared task selects the metrics.
        dataset_name: Optional identifier recorded on the result
            (e.g. which test dataset was used).

    Returns:
        :class:`EvaluationResult` with predictions, targets, MSE, MAE,
        RMSE, and — for binary targets only — accuracy at
        :data:`CLASSIFICATION_THRESHOLD`.

    Raises:
        TypeError: If ``network``/``dataset`` have the wrong types or
            ``dataset_name`` is neither ``None`` nor a string.
        ValueError: If the dataset is unlabeled, incompatible with the
            network, or declares binary targets that are not 0.0/1.0.
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
    kind = dataset.target_kind
    if kind == BINARY_TARGETS:
        _require_binary_targets(targets)
    predictions = network.forward(features)
    if predictions.shape != targets.shape:
        raise ValueError(
            f"prediction shape {predictions.shape} does not match the "
            f"target shape {targets.shape}"
        )
    loss = mean_squared_error(targets, predictions)
    mae = float(np.mean(np.abs(predictions - targets)))
    rmse = float(np.sqrt(loss))
    accuracy: float | None = None
    if kind == BINARY_TARGETS:
        predicted_classes = (predictions >= CLASSIFICATION_THRESHOLD).astype(float)
        accuracy = float(np.mean(predicted_classes == targets))
    return EvaluationResult(
        predictions=predictions,
        targets=targets,
        mse=loss,
        mae=mae,
        rmse=rmse,
        n_samples=dataset.n_samples,
        accuracy=accuracy,
        target_kind=kind,
        dataset_name=dataset_name,
    )