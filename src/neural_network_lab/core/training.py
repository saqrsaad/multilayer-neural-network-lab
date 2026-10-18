"""Full-batch training loop with optional validation-based selection.

The update rule is the existing algorithm, unchanged: one measured
full-batch gradient-descent epoch per step, exactly as
:meth:`NeuralNetwork.train_step`. This module only adds the
bookkeeping around it:

- validation rows are scored with read-only forward propagation
  (:meth:`NeuralNetwork.forward` + :func:`mean_squared_error`) after
  each epoch, so validation data never produces a gradient update;
- the best epoch is the one with the lowest validation loss, and its
  exact parameters are snapshotted at that moment through
  :meth:`NeuralNetwork.to_state_dict` — the final epoch is never
  assumed to be the best;
- after a successful run the best snapshot is restored onto the
  network, so callers receive the selected model rather than whatever
  the last epoch happened to produce.

Test data is deliberately not accepted here: evaluating a held-out
test subset belongs to the application layer
(:func:`neural_network_lab.application.evaluation_service.evaluate`),
which keeps test rows out of both training and model selection.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from neural_network_lab.core.losses import mean_squared_error
from neural_network_lab.core.network import NeuralNetwork
from neural_network_lab.core.optimizers import validate_learning_rate


def _validate_epochs(epochs: int) -> int:
    """Validate an epoch count and return it as an int."""
    if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)):
        raise TypeError("epochs must be a positive integer")
    if int(epochs) < 1:
        raise ValueError("epochs must be a positive integer")
    return int(epochs)


def _validation_loss(
    network: NeuralNetwork, features: ArrayLike, targets: ArrayLike
) -> float:
    """Score validation rows read-only; parameters are never touched."""
    predictions = network.forward(features)
    return float(mean_squared_error(targets, predictions))


@dataclass(frozen=True)
class TrainingOutcome:
    """Structured outcome of one training run.

    ``loss_history`` holds one measured post-update training MSE per
    epoch. ``validation_history`` is ``None`` when no validation data was
    supplied; otherwise it holds the matching per-epoch validation MSE.
    ``best_epoch`` (1-based) and ``best_validation_loss`` describe the
    epoch with the lowest validation loss, whose exact parameters were
    restored onto the trained network when ``best_state`` is not
    ``None``. ``cancelled`` runs keep their best snapshot for reporting
    but never restore it.
    """

    epochs_completed: int
    loss_history: tuple[float, ...]
    learning_rate: float
    mode: str
    validation_history: tuple[float, ...] | None = None
    best_epoch: int | None = None
    best_validation_loss: float | None = None
    final_validation_loss: float | None = None
    best_state: dict | None = None
    cancelled: bool = False

    @property
    def final_loss(self) -> float:
        """Measured training MSE after the final epoch."""
        return self.loss_history[-1]

    @property
    def used_validation(self) -> bool:
        """Whether validation data participated in this run."""
        return self.validation_history is not None


def train_with_validation(
    network: NeuralNetwork,
    train_features: ArrayLike,
    train_targets: ArrayLike,
    *,
    epochs: int,
    learning_rate: float,
    mode: str = "continued",
    validation_features: ArrayLike | None = None,
    validation_targets: ArrayLike | None = None,
    reinitialize: bool = False,
    seed: int | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_epoch: Callable[[int, int, float], None] | None = None,
) -> TrainingOutcome:
    """Train in place, tracking validation loss and the best epoch.

    Args:
        network: Network trained in place; continued runs keep its
            current parameters as the starting point.
        train_features/train_targets: Training rows; only these ever
            contribute gradient updates.
        epochs: Positive number of full-batch epochs.
        learning_rate: Positive, finite step size.
        mode: ``"continued"`` or ``"from_scratch"``, recorded on the
            outcome for reporting.
        validation_features/validation_targets: Optional validation
            rows scored read-only after each epoch.
        reinitialize: When true, fresh parameters are drawn first
            (equivalent to ``train_from_scratch``), honoring ``seed``.
        seed: Seed for the reinitialization; only meaningful together
            with ``reinitialize``.
        should_cancel: Optional callable checked between epochs; when
            it returns true the run stops without restoring the best
            epoch (the trial epochs are reported, not applied).
        on_epoch: Optional callable invoked after each epoch as
            ``on_epoch(epoch, total_epochs, train_loss)`` so callers can
            report progress without the core loop knowing about them.

    Returns:
        :class:`TrainingOutcome` describing the run. When validation
        data was supplied and the run completed, the network's
        parameters are those of the best validation epoch.

    Raises:
        TypeError/ValueError: For invalid epochs, learning rate,
            mismatched arrays, or non-finite values — raised before any
            parameter is changed when the inputs are the problem.
    """
    epochs_count = _validate_epochs(epochs)
    rate = validate_learning_rate(learning_rate)
    if (validation_features is None) != (validation_targets is None):
        raise ValueError(
            "validation features and targets must be provided together"
        )
    uses_validation = validation_features is not None

    if reinitialize:
        fresh = NeuralNetwork(list(network.layer_sizes), seed=seed)
        network.set_parameters(fresh.weights, fresh.biases)

    # Validation rows are validated before any training step so bad
    # held-out data can never leave the model half-trained.
    if uses_validation:
        _validation_loss(
            network, validation_features, validation_targets  # type: ignore[arg-type]
        )

    history: list[float] = []
    validation_history: list[float] = []
    best_epoch: int | None = None
    best_loss: float | None = None
    best_state: dict | None = None

    for epoch in range(1, epochs_count + 1):
        if should_cancel is not None and should_cancel():
            return TrainingOutcome(
                epochs_completed=len(history),
                loss_history=tuple(history),
                learning_rate=rate,
                mode=mode,
                validation_history=tuple(validation_history) if uses_validation else None,
                best_epoch=best_epoch,
                best_validation_loss=best_loss,
                final_validation_loss=validation_history[-1] if validation_history else None,
                best_state=best_state,
                cancelled=True,
            )
        history.append(
            float(network.train_step(train_features, train_targets, rate))
        )
        if on_epoch is not None:
            on_epoch(epoch, epochs_count, history[-1])
        if uses_validation:
            loss_value = _validation_loss(
                network, validation_features, validation_targets  # type: ignore[arg-type]
            )
            validation_history.append(loss_value)
            if best_loss is None or loss_value < best_loss:
                best_loss = loss_value
                best_epoch = epoch
                best_state = network.to_state_dict()

    outcome = TrainingOutcome(
        epochs_completed=len(history),
        loss_history=tuple(history),
        learning_rate=rate,
        mode=mode,
        validation_history=tuple(validation_history) if uses_validation else None,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        final_validation_loss=validation_history[-1] if validation_history else None,
        best_state=best_state,
    )
    if best_state is not None:
        # Select the best epoch, not the last one.
        network.set_parameters(best_state["weights"], best_state["biases"])
    return outcome
