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
from neural_network_lab.core.network import TrainingResult


class TrainingWorker(QObject):
    """Run a full training job epoch by epoch in a worker thread.

    Constructed and configured on the GUI thread, then moved to a
    :class:`QThread` whose ``started`` signal invokes :meth:`run`.
    Results leave the thread only through the signals below, carrying
    plain JSON-like data (no NumPy arrays, no widgets, no networks).
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
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        """Ask the running loop to stop after the current epoch."""
        self._cancel_event.set()

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
            history: list[float] = []
            for epoch in range(1, self._epochs + 1):
                if self._cancel_event.is_set():
                    self.cancelled.emit()
                    return
                loss = network.train_step(
                    self._features, self._targets, self._learning_rate
                )
                loss_value = float(loss)
                history.append(loss_value)
                self.progress.emit(epoch, self._epochs, loss_value)
            result = TrainingResult(
                epochs_completed=len(history),
                loss_history=tuple(history),
                learning_rate=self._learning_rate,
                mode=self._mode,
            )
        except Exception as exc:  # report, never crash the thread host
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(
            {
                "state": network.to_state_dict(),
                "history": list(result.loss_history),
                "mode": result.mode,
                "epochs": result.epochs_completed,
                "learning_rate": result.learning_rate,
            }
        )
