"""Background training worker: run epochs off the GUI thread.

The worker owns its network copy outright and reports progress,
completion, and failure through Qt signals, so the GUI thread never
blocks and never touches model state mid-run. All mathematics reuses
the existing core APIs (:meth:`NeuralNetwork.train_step`,
:meth:`NeuralNetwork.from_state_dict`, :meth:`NeuralNetwork.to_state_dict`);
nothing is duplicated here.
"""

from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QObject
from PySide6.QtCore import Signal
from PySide6.QtCore import Slot

from neural_network_lab.core.network import NeuralNetwork
from neural_network_lab.core.training import train_with_validation


class TrainingWorker(QObject):
    """Run a full training job epoch by epoch in a worker thread.

    Constructed and configured on the GUI thread, then moved to a
    :class:`QThread` whose ``started`` signal invokes :meth:`run`.
    Results leave the thread only through the signals below, carrying
    plain JSON-like data (no NumPy arrays, no widgets, no networks).

    Validation rows, when supplied, are scored read-only after each
    epoch; the epoch with the lowest validation loss is snapshotted and
    restored at the end, so the emitted state is the selected best
    model. Training rows alone ever contribute gradient updates.
    """

    progress = Signal(int, int, float)
    finished = Signal(dict)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        *,
        mode: str,
        layer_sizes: list[int],
        epochs: int,
        learning_rate: float,
        features,
        targets,
        initial_state: dict | None = None,
        seed: int | None = None,
        validation_features=None,
        validation_targets=None,
        parent=None,
    ) -> None:
        """Capture validated plain inputs; no training happens here."""
        super().__init__(parent)
        self._mode = mode
        self._layer_sizes = [int(size) for size in layer_sizes]
        self._epochs = epochs
        self._learning_rate = float(learning_rate)
        self._features = np.asarray(features, dtype=float)
        self._targets = np.asarray(targets, dtype=float)
        self._initial_state = initial_state
        self._seed = seed
        self._validation_features = (
            None
            if validation_features is None
            else np.asarray(validation_features, dtype=float)
        )
        self._validation_targets = (
            None
            if validation_targets is None
            else np.asarray(validation_targets, dtype=float)
        )
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        """Ask the running loop to stop after the current epoch."""
        self._cancel_event.set()

    def _emit_progress(self, epoch: int, total: int, loss: float) -> None:
        """Report one completed epoch; called from the worker thread."""
        self.progress.emit(epoch, total, loss)

    @Slot()
    def run(self) -> None:
        """Execute the training loop; runs in the worker thread."""
        try:
            if self._mode not in ("continued", "from_scratch"):
                raise ValueError(f"unknown training mode: {self._mode!r}")
            if (
                isinstance(self._epochs, bool)
                or not isinstance(self._epochs, int)
                or self._epochs < 1
            ):
                raise ValueError("epochs must be a positive integer")
            if self._mode == "continued":
                if self._initial_state is None:
                    raise ValueError(
                        "continued training needs the current parameters"
                    )
                network = NeuralNetwork.from_state_dict(self._initial_state)
            else:
                network = NeuralNetwork(
                    list(self._layer_sizes), seed=self._seed
                )
            outcome = train_with_validation(
                network,
                self._features,
                self._targets,
                epochs=self._epochs,
                learning_rate=self._learning_rate,
                mode=self._mode,
                validation_features=self._validation_features,
                validation_targets=self._validation_targets,
                should_cancel=self._cancel_event.is_set,
                on_epoch=self._emit_progress,
            )
        except Exception as exc:  # report, never crash the thread host
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        if outcome.cancelled:
            self.cancelled.emit()
            return
        self.finished.emit(
            {
                "state": network.to_state_dict(),
                "history": list(outcome.loss_history),
                "mode": outcome.mode,
                "epochs": outcome.epochs_completed,
                "learning_rate": outcome.learning_rate,
                "validation_history": (
                    None
                    if outcome.validation_history is None
                    else list(outcome.validation_history)
                ),
                "best_epoch": outcome.best_epoch,
                "best_validation_loss": outcome.best_validation_loss,
                "final_validation_loss": outcome.final_validation_loss,
            }
        )
