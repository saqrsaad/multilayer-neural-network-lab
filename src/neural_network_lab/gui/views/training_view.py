"""Training view: configure, run, and report supervised training.

The view composes :class:`NetworkConfigWidget` and
:class:`TrainingControls` with a dataset supplied by a provider (wired
to the dataset editor by the main window; injectable in tests). Both
actions delegate to the existing core APIs on an explicitly owned
:class:`NeuralNetwork`: training from scratch builds a fresh network
and calls ``train_from_scratch``, while continued training calls
``train`` on the current one. No training mathematics lives here.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QThread
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget

from neural_network_lab.core.network import NeuralNetwork
from neural_network_lab.data.dataset import Dataset
from neural_network_lab.gui.main_window import CONTENT_MARGIN
from neural_network_lab.gui.main_window import CONTENT_SPACING
from neural_network_lab.gui.widgets.network_config_widget import NetworkConfigWidget
from neural_network_lab.gui.widgets.training_controls import TrainingControls
from neural_network_lab.gui.workers.training_worker import TrainingWorker


class TrainingView(QWidget):
    """Configure architecture/settings and run both training modes."""

    training_finished = Signal(str)
    training_failed = Signal(str)

    def __init__(self, parent=None) -> None:
        """Create the configuration, dataset info, and control widgets."""
        super().__init__(parent)
        self._network: NeuralNetwork | None = None
        self._dataset_provider: Callable[[], Dataset | None] = lambda: None
        self._worker_thread: QThread | None = None
        self._worker: TrainingWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN
        )
        layout.setSpacing(CONTENT_SPACING)

        self._config = NetworkConfigWidget(self)
        self._config.setObjectName("networkConfig")
        layout.addWidget(self._config)

        self._dataset_label = QLabel("Training data: none selected.", self)
        self._dataset_label.setWordWrap(True)
        self._dataset_label.setObjectName("trainingDatasetInfo")
        layout.addWidget(self._dataset_label)

        self._controls = TrainingControls(self)
        self._controls.setObjectName("trainingControls")
        layout.addWidget(self._controls, 1)

        self._controls.train_scratch_requested.connect(self._on_train_scratch)
        self._controls.train_continued_requested.connect(self._on_train_continued)

    def set_dataset_provider(
        self, provider: Callable[[], Dataset | None]
    ) -> None:
        """Set the callable supplying the selected training dataset."""
        self._dataset_provider = provider

    def current_network(self) -> NeuralNetwork | None:
        """Return the owned network, if one has been trained/built."""
        return self._network

    def current_dataset(self) -> Dataset | None:
        """Return the currently supplied dataset, if any."""
        try:
            dataset = self._dataset_provider()
        except Exception:
            return None
        return dataset if isinstance(dataset, Dataset) else None

    def refresh_dataset_info(self) -> Dataset | None:
        """Update the input size and info line from the supplied dataset."""
        dataset = self.current_dataset()
        if dataset is None:
            self._config.set_input_size(None)
            self._dataset_label.setText("Training data: none selected.")
            return None
        self._config.set_input_size(dataset.n_features)
        labeled = "labeled" if dataset.has_targets else "unlabeled"
        self._dataset_label.setText(
            f"Training data: {dataset.n_samples} samples, "
            f"{dataset.n_features} features ({labeled})."
        )
        return dataset

    def _collect_training_inputs(self) -> tuple[Dataset, tuple[int, ...], float, int]:
        """Validate settings and data; return dataset, sizes, lr, epochs.

        Raises:
            ValueError: With a user-readable message; nothing is
                trained or rebuilt when validation fails.
        """
        dataset = self.refresh_dataset_info()
        if dataset is None:
            raise ValueError(
                "No training dataset selected. Validate one in "
                "Dataset Management first."
            )
        if not dataset.has_targets:
            raise ValueError("Training needs a labeled dataset with targets.")
        try:
            sizes = self._config.layer_sizes()
            learning_rate = self._config.learning_rate()
            epochs = self._config.epochs()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid settings: {exc}") from exc
        try:
            dataset.validate_compatible_with(sizes[0], sizes[-1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Incompatible dataset or output size: {exc}") from exc
        return dataset, sizes, learning_rate, epochs

    def _on_train_scratch(self) -> None:
        """Build a fresh network and train it from scratch."""
        self._controls.clear_error()
        try:
            dataset, sizes, learning_rate, epochs = self._collect_training_inputs()
        except (TypeError, ValueError) as exc:
            self._controls.show_error(str(exc))
            return
        features, targets = dataset.to_arrays()
        assert targets is not None  # narrowed by the labeled check above
        self._start_worker(
            mode="from_scratch",
            sizes=sizes,
            features=features,
            targets=targets,
            learning_rate=learning_rate,
            epochs=epochs,
            initial_state=None,
        )

    def _on_train_continued(self) -> None:
        """Continue training from the current parameters."""
        self._controls.clear_error()
        if self._network is None:
            self._controls.show_error(
                "No model yet. Train from scratch first, then continue "
                "training from its parameters."
            )
            return
        try:
            dataset, sizes, learning_rate, epochs = self._collect_training_inputs()
        except (TypeError, ValueError) as exc:
            self._controls.show_error(str(exc))
            return
        if tuple(self._network.layer_sizes) != tuple(sizes):
            self._controls.show_error(
                f"Architecture changed to {tuple(sizes)}; the current model "
                f"is {tuple(self._network.layer_sizes)}. Train from scratch "
                "to rebuild it explicitly."
            )
            return
        features, targets = dataset.to_arrays()
        assert targets is not None  # narrowed by the labeled check above
        self._start_worker(
            mode="continued",
            sizes=sizes,
            features=features,
            targets=targets,
            learning_rate=learning_rate,
            epochs=epochs,
            initial_state=self._network.to_state_dict(),
        )

    def is_training_active(self) -> bool:
        """Return True while a background training run is in flight."""
        thread = self._worker_thread
        return thread is not None and thread.isRunning()

    def cancel_training(self) -> bool:
        """Ask an active run to stop after the current epoch."""
        if self._worker is None or not self.is_training_active():
            return False
        self._worker.request_cancel()
        return True

    def wait_for_worker(self, timeout_ms: int = 5000) -> bool:
        """Block until the worker thread finishes; True when done."""
        thread = self._worker_thread
        if thread is None:
            return True
        return bool(thread.wait(timeout_ms))

    def _start_worker(
        self,
        *,
        mode: str,
        sizes: tuple[int, ...],
        features,
        targets,
        learning_rate: float,
        epochs: int,
        initial_state: dict | None,
    ) -> None:
        """Launch one background run; the GUI thread stays responsive."""
        if self.is_training_active():
            self._controls.show_error(
                "Training is already running; wait for it to finish."
            )
            return
        self._controls.clear_error()
        self._controls.set_running(True)
        self._controls.show_status(
            f"Training ({mode}) started in the background: 0/{epochs} epochs."
        )
        thread = QThread(self)
        worker = TrainingWorker(
            mode=mode,
            layer_sizes=list(sizes),
            epochs=epochs,
            learning_rate=learning_rate,
            features=features,
            targets=targets,
            initial_state=initial_state,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_worker_progress)
        worker.finished.connect(self._on_worker_finished)
        worker.failed.connect(self._on_worker_failed)
        worker.cancelled.connect(self._on_worker_cancelled)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(self._on_thread_finished)
        self._worker_thread = thread
        self._worker = worker
        thread.start()

    def _on_worker_progress(
        self, completed: int, total: int, loss: float
    ) -> None:
        """Report per-epoch progress without touching the model."""
        self._controls.show_status(
            f"Training in the background… epoch {completed}/{total} "
            f"(training loss (MSE) {loss!r})."
        )

    def _on_worker_finished(self, payload: dict) -> None:
        """Install the trained state on the GUI thread and report."""
        self._controls.set_running(False)
        try:
            state = payload["state"]
            history = tuple(payload["history"])
            mode = str(payload["mode"])
            epochs = int(payload["epochs"])
            if (
                self._network is not None
                and tuple(self._network.layer_sizes)
                == tuple(state["layer_sizes"])
            ):
                self._network.set_parameters(state["weights"], state["biases"])
            else:
                self._network = NeuralNetwork.from_state_dict(state)
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            self._controls.show_error(f"Training failed: {exc}")
            self.training_failed.emit(str(exc))
            return
        if mode == "from_scratch":
            label = "from scratch (previous parameters replaced)"
        else:
            label = "continued (parameters preserved as the starting point)"
        self._controls.display_history(
            mode=label,
            epochs=epochs,
            final_loss=history[-1],
            history=history,
        )
        self.training_finished.emit(mode)

    def _on_worker_failed(self, message: str) -> None:
        """Report a worker failure; the previous model is untouched."""
        self._controls.set_running(False)
        self._controls.show_error(f"Training failed: {message}")
        self.training_failed.emit(message)

    def _on_worker_cancelled(self) -> None:
        """Report cancellation; partial epochs are discarded."""
        self._controls.set_running(False)
        self._controls.show_status(
            "Training cancelled; the previous model is unchanged."
        )

    def _on_thread_finished(self) -> None:
        """Release worker-thread resources on the GUI thread."""
        worker, thread = self._worker, self._worker_thread
        self._worker = None
        self._worker_thread = None
        if worker is not None:
            worker.deleteLater()
        if thread is not None:
            thread.deleteLater()
