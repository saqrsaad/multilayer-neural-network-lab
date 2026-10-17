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

import uuid
from collections.abc import Callable
from datetime import datetime
from datetime import timezone
from pathlib import Path

from PySide6.QtCore import QThread
from PySide6.QtCore import Signal
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget

from neural_network_lab.application.experiment_service import ExperimentRecord
from neural_network_lab.application.evaluation_service import evaluate
from neural_network_lab.core.network import NeuralNetwork
from neural_network_lab.data.dataset import Dataset
from neural_network_lab.data.dataset import TARGET_KIND_LABELS
from neural_network_lab.data.dataset_split import DatasetSplits
from neural_network_lab.data.dataset_split import split_dataset
from neural_network_lab.gui.widgets.network_config_widget import NetworkConfigWidget
from neural_network_lab.gui.widgets.page_scaffold import MessageLabel
from neural_network_lab.gui.widgets.page_scaffold import SectionCard
from neural_network_lab.gui.widgets.page_scaffold import button_row
from neural_network_lab.gui.widgets.page_scaffold import install_page
from neural_network_lab.gui.widgets.page_scaffold import make_button
from neural_network_lab.gui.widgets.training_controls import TrainingControls
from neural_network_lab.gui.widgets.ui_theme import CONTENT_SPACING
from neural_network_lab.gui.workers.training_worker import TrainingWorker
from neural_network_lab.storage.experiment_storage import save_experiment_record
from neural_network_lab.storage.model_storage import load_model_state
from neural_network_lab.storage.model_storage import save_model_state

#: Project root of this checkout (…/src/neural_network_lab/gui/views/).
_PROJECT_ROOT = Path(__file__).resolve().parents[4]

#: Directory receiving automatically saved experiment records.
EXPERIMENTS_DIR = _PROJECT_ROOT / "data" / "experiments"

#: Directory receiving automatically saved best-model state files.
MODELS_DIR = _PROJECT_ROOT / "data" / "models"

#: Smallest width (pixels) of the setup column beside the results.
SETUP_COLUMN_WIDTH = 320

#: Largest number of GUI progress updates one training run may produce.
#: Every epoch reports progress from the worker thread; refreshing the
#: page (and its charts) once per epoch makes a long run freeze the
#: interface, so updates are spread over this many steps instead.
PROGRESS_REPORT_STEPS = 100


def _run_reference(
    experiment_id: str,
    created_at: str,
    record_path: Path | None,
    model_path: Path | None,
) -> dict:
    """Return the persistence references of a completed run."""
    return {
        "experiment_id": experiment_id,
        "created_at": created_at,
        "record_path": None if record_path is None else str(record_path),
        "model_path": None if model_path is None else str(model_path),
    }


def _empty_run_reference(
    experiment_id: str, created_at: str, model_path: Path | None
) -> dict:
    """Return the references of a run whose record could not be written."""
    return _run_reference(experiment_id, created_at, None, model_path)


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
        self._pending_run_info: dict | None = None
        self._last_splits: DatasetSplits | None = None
        self._test_metrics: dict | None = None
        self._last_run_summary: str | None = None
        self._run_dataset: Dataset | None = None
        self._last_run_dataset: Dataset | None = None
        self._model_revision = 0
        self._reported_epoch = 0
        self._last_run_details: dict | None = None

        self._page = install_page(
            self,
            "Training",
            "Configure the architecture and the training settings on the "
            "left, then run training from scratch or continue from the "
            "current parameters. Results appear on the right.",
        )
        columns = QHBoxLayout()
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(CONTENT_SPACING)

        setup = QWidget()
        setup.setMinimumWidth(SETUP_COLUMN_WIDTH)
        setup_layout = QVBoxLayout(setup)
        setup_layout.setContentsMargins(0, 0, 0, 0)
        setup_layout.setSpacing(CONTENT_SPACING)

        self._config = NetworkConfigWidget(setup)
        self._config.setObjectName("networkConfig")
        setup_layout.addWidget(self._config)

        data_card = SectionCard(
            "Training data",
            parent=setup,
        )
        self._dataset_label = MessageLabel("info", data_card)
        self._dataset_label.setText("Training data: none selected.")
        data_card.add(self._dataset_label)
        setup_layout.addWidget(data_card)

        model_card = SectionCard(
            "Model state",
            description=(
                "Save the current weights and biases to a JSON file, or "
                "load a saved model. Loading replaces the model used by "
                "every other section."
            ),
            parent=setup,
        )
        self._save_button = make_button(
            "Save Model",
            kind="secondary",
            tooltip=(
                "Save the current network (weights and biases) to a JSON "
                "model state file."
            ),
            parent=model_card,
        )
        self._save_button.setObjectName("saveModelButton")
        self._load_button = make_button(
            "Load Model",
            kind="secondary",
            tooltip=(
                "Load a saved network from a JSON model state file, "
                "replacing the current model."
            ),
            parent=model_card,
        )
        self._load_button.setObjectName("loadModelButton")
        model_card.add_layout(button_row(self._save_button, self._load_button))
        self._model_status = MessageLabel("status", model_card)
        model_card.add(self._model_status)
        setup_layout.addWidget(model_card)
        setup_layout.addStretch(1)
        columns.addWidget(setup, 0)

        results = QWidget()
        results_layout = QVBoxLayout(results)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(CONTENT_SPACING)
        self._controls = TrainingControls(results)
        self._controls.setObjectName("trainingControls")
        results_layout.addWidget(self._controls, 1)
        columns.addWidget(results, 1)
        # The result column takes the leftover height so the charts and
        # the per-epoch readout stay as large as the window allows.
        self._page.add_layout(columns, 1)

        self._controls.train_scratch_requested.connect(self._on_train_scratch)
        self._controls.train_continued_requested.connect(self._on_train_continued)
        self._controls.cancel_requested.connect(self._on_cancel_requested)
        self._save_button.clicked.connect(self._on_save_model)
        self._load_button.clicked.connect(self._on_load_model)

    def _on_cancel_requested(self) -> None:
        """Ask the running job to stop after its current epoch."""
        if self.cancel_training():
            self._controls.show_status(
                "Cancelling after the current epoch… the previous model is "
                "unchanged until then."
            )

    def set_dataset_provider(
        self, provider: Callable[[], Dataset | None]
    ) -> None:
        """Set the callable supplying the selected training dataset."""
        self._dataset_provider = provider

    def current_network(self) -> NeuralNetwork | None:
        """Return the owned network, if one has been trained/built."""
        return self._network

    def current_layer_sizes(self) -> tuple[int, ...] | None:
        """Return the active network's layer sizes, if it exists."""
        if self._network is None:
            return None
        return tuple(self._network.layer_sizes)

    def apply_layer_sizes(self, sizes) -> None:
        """Replace the active network with one using ``sizes``.

        The new architecture is validated by
        :class:`~neural_network_lab.core.network.NeuralNetwork`; on any
        rejection the active network is left unchanged. Applying the
        architecture the model already uses is a no-op, so a redundant
        configuration can never discard trained parameters.

        Raises:
            ValueError: When training is running or the sizes are
                invalid.
        """
        if self.is_training_active():
            raise ValueError(
                "Training is running; wait for it to finish before "
                "changing the network configuration."
            )
        requested = tuple(int(size) for size in sizes)
        if self._network is not None and tuple(self._network.layer_sizes) == (
            requested
        ):
            return
        had_model = self._network is not None
        had_results = self._last_run_summary is not None
        network = NeuralNetwork(sizes)
        self._network = network
        self._model_revision += 1
        # The active model stays the single source of truth: the settings
        # shown on this page follow whatever architecture was applied.
        self._config.set_hidden_sizes(tuple(int(size) for size in sizes[1:-1]))
        self._config.set_output_size(int(sizes[-1]))
        detail = (
            "the previous model was replaced"
            if had_model
            else "a model was created"
        )
        cleared = (
            " and its results were cleared"
            if had_results
            else ""
        )
        self._invalidate_results(
            f"Architecture changed to {self._format_sizes(requested)}: "
            f"{detail}{cleared}."
        )

    @staticmethod
    def _format_sizes(sizes: tuple) -> str:
        """Format layer sizes as ``[n0, n1, ...]``."""
        return "[" + ", ".join(str(int(size)) for size in sizes) + "]"

    def last_run_summary(self) -> str | None:
        """Return the last finished run's summary text, if there is one.

        Read-only mirror of what the training controls already display;
        nothing is recomputed here.
        """
        return self._last_run_summary

    def last_run_details(self) -> dict | None:
        """Return the last finished run's recorded fields, if any.

        The values come straight from the completed run: its payload, the
        dataset it used, and the metrics the evaluation service measured.
        A field the run did not produce is ``None`` rather than an
        estimate, so the Dashboard can show "not recorded" instead of
        inventing a number. Returns ``None`` when no run has finished or
        when its results were cleared.
        """
        return dict(self._last_run_details) if self._last_run_details else None

    def model_revision(self) -> int:
        """Return a counter that changes whenever the model is replaced.

        Other views compare it to notice that displayed results belong to
        an older model, including continued training that rewrites the
        parameters in place.
        """
        return self._model_revision

    def _invalidate_results(self, message: str) -> None:
        """Drop results that no longer describe the current state."""
        self._controls.clear_history()
        self._controls.clear_predictions()
        self._last_run_summary = None
        self._last_run_details = None
        self._last_run_dataset = None
        self._controls.show_status(message)

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        """Keep the data line and results aligned with the selection."""
        super().showEvent(event)
        self.refresh_dataset_info()
        if (
            self._last_run_summary is not None
            and self._last_run_dataset is not None
            and self.current_dataset() is not self._last_run_dataset
        ):
            self._invalidate_results(
                "Training data changed since the last run, so its results "
                "were cleared. Train again to record new ones."
            )

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
            self._config.set_output_hint(None)
            self._dataset_label.setText("Training data: none selected.")
            return None
        self._config.set_input_size(dataset.n_features)
        self._config.set_output_hint(dataset.n_outputs, dataset.target_kind)
        labeled = "labeled" if dataset.has_targets else "unlabeled"
        task_text = (
            ""
            if dataset.n_outputs is None
            else f", {TARGET_KIND_LABELS.get(dataset.target_kind, '')}"
        )
        target_text = (
            ""
            if dataset.n_outputs is None
            else f", {dataset.n_outputs} target column"
            f"{'' if dataset.n_outputs == 1 else 's'}{task_text}"
        )
        self._dataset_label.setText(
            f"Training data: {dataset.n_samples} samples, "
            f"{dataset.n_features} features ({labeled}{target_text})."
        )
        return dataset

    def _collect_training_inputs(
        self,
    ) -> tuple[Dataset, tuple[int, ...], float, int, DatasetSplits]:
        """Validate settings and data; return inputs plus the split.

        The selected dataset is divided into training, validation, and
        test subsets before anything is trained. Validation and test
        rows never contribute gradient updates: they are only used for
        model selection and evaluation respectively.

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
            train_ratio, validation_ratio, test_ratio = self._config.split_ratios()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid settings: {exc}") from exc
        try:
            dataset.validate_compatible_with(sizes[0], sizes[-1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Incompatible dataset or output size: {exc}") from exc
        try:
            splits = split_dataset(
                dataset,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                test_ratio=test_ratio,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Could not split the dataset: {exc}") from exc
        return dataset, sizes, learning_rate, epochs, splits

    def _describe_data_plan(
        self, dataset: Dataset, splits: DatasetSplits
    ) -> str:
        """Return an educational one-line description of the data plan."""
        parts = [
            f"training {splits.train.n_samples}",
            f"validation {splits.validation.n_samples if splits.validation else 0}",
            f"test {splits.test.n_samples if splits.test else 0}",
        ]
        note = (
            "Validation rows are used only to choose the best epoch; "
            "test rows only to evaluate the finished model."
        )
        if splits.warning is not None:
            note = f"{splits.warning}. {note}"
        return (
            f"Training data: {dataset.n_samples} samples · "
            + " / ".join(parts)
            + f" rows — {note}"
        )

    def _on_train_scratch(self) -> None:
        """Build a fresh network and train it from scratch."""
        self._controls.clear_error()
        try:
            dataset, sizes, learning_rate, epochs, splits = (
                self._collect_training_inputs()
            )
        except (TypeError, ValueError) as exc:
            self._controls.show_error(str(exc))
            return
        self._last_splits = splits
        self._dataset_label.setText(self._describe_data_plan(dataset, splits))
        features, targets = splits.train.to_arrays()
        assert targets is not None  # narrowed by the labeled check above
        self._pending_run_info = {
            "train_samples": splits.train.n_samples,
            "train_features": splits.train.n_features,
            "train_outputs": int(splits.train.n_outputs or 0),
            "validation_samples": (
                splits.validation.n_samples if splits.validation else None
            ),
            "test_samples": splits.test.n_samples if splits.test else None,
            "warning": splits.warning,
        }
        self._run_dataset = dataset
        self._start_worker(
            mode="from_scratch",
            sizes=sizes,
            features=features,
            targets=targets,
            learning_rate=learning_rate,
            epochs=epochs,
            initial_state=None,
            validation_features=(
                splits.validation.to_arrays()[0] if splits.validation else None
            ),
            validation_targets=(
                splits.validation.to_arrays()[1] if splits.validation else None
            ),
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
            dataset, sizes, learning_rate, epochs, splits = (
                self._collect_training_inputs()
            )
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
        self._last_splits = splits
        self._dataset_label.setText(self._describe_data_plan(dataset, splits))
        features, targets = splits.train.to_arrays()
        assert targets is not None  # narrowed by the labeled check above
        self._pending_run_info = {
            "train_samples": splits.train.n_samples,
            "train_features": splits.train.n_features,
            "train_outputs": int(splits.train.n_outputs or 0),
            "validation_samples": (
                splits.validation.n_samples if splits.validation else None
            ),
            "test_samples": splits.test.n_samples if splits.test else None,
            "warning": splits.warning,
        }
        self._run_dataset = dataset
        self._start_worker(
            mode="continued",
            sizes=sizes,
            features=features,
            targets=targets,
            learning_rate=learning_rate,
            epochs=epochs,
            initial_state=self._network.to_state_dict(),
            validation_features=(
                splits.validation.to_arrays()[0] if splits.validation else None
            ),
            validation_targets=(
                splits.validation.to_arrays()[1] if splits.validation else None
            ),
        )

    def is_training_active(self) -> bool:
        """Return True while a background training run is in flight."""
        thread = self._worker_thread
        return thread is not None and thread.isRunning()

    def cancel_training(self) -> bool:
        """Ask an active run to stop after the current epoch."""
        worker = self._worker
        if worker is None or not self.is_training_active():
            return False
        try:
            worker.request_cancel()
        except RuntimeError:
            # The worker deleted itself (see _start_worker) in the small
            # window before the thread reports that it finished.
            return False
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
        validation_features=None,
        validation_targets=None,
    ) -> None:
        """Launch one background run; the GUI thread stays responsive."""
        if self.is_training_active():
            self._controls.show_error(
                "Training is already running; wait for it to finish."
            )
            return
        self._controls.clear_error()
        self._controls.set_running(True)
        self._set_model_actions_enabled(False)
        self._controls.clear_predictions()
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
            validation_features=validation_features,
            validation_targets=validation_targets,
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
        # The worker must delete itself from *its own* thread: a
        # ``deleteLater()`` issued from the GUI thread after the thread
        # finished would post its deferred delete to a thread that is
        # no longer running, so the object would never be freed.
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._worker_thread = thread
        self._worker = worker
        self._reported_epoch = 0
        thread.start()

    def _progress_step(self, total: int) -> int:
        """Return how many epochs may pass between GUI progress updates.

        Every epoch of a long run emits a progress signal. Updating the
        page (and through it the charts inside it) once per epoch makes
        the interface unresponsive, so the interface is refreshed at
        most this many times per run.
        """
        return max(1, int(total) // PROGRESS_REPORT_STEPS)

    def _on_worker_progress(
        self, completed: int, total: int, loss: float
    ) -> None:
        """Report per-epoch progress without touching the model.

        The update is throttled to a fixed number of reports per run so
        a long run cannot flood the GUI thread with queued events.
        """
        step = self._progress_step(total)
        last_epoch = max(1, int(total))
        if not (
            completed <= step
            or completed == last_epoch
            or (completed - self._reported_epoch) >= step
        ):
            return
        self._reported_epoch = completed
        self._controls.set_progress(completed, total)
        self._controls.show_status(
            f"Training in the background… epoch {completed}/{total} "
            f"(training loss (MSE) {loss!r})."
        )

    def _on_worker_finished(self, payload: dict) -> None:
        """Install the trained state on the GUI thread and report."""
        self._controls.set_running(False)
        self._controls.reset_progress()
        self._set_model_actions_enabled(True)
        try:
            state = payload["state"]
            history = tuple(payload["history"])
            mode = str(payload["mode"])
            epochs = int(payload["epochs"])
            validation_history = payload.get("validation_history")
            if (
                self._network is not None
                and tuple(self._network.layer_sizes)
                == tuple(state["layer_sizes"])
            ):
                # The worker already restored the best epoch's
                # parameters, so this installs the selected model.
                self._network.set_parameters(state["weights"], state["biases"])
            else:
                self._network = NeuralNetwork.from_state_dict(state)
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            self._controls.show_error(f"Training failed: {exc}")
            self.training_failed.emit(str(exc))
            return
        self._model_revision += 1
        if self._network is None:  # defensive: never installed above
            self._controls.show_error(
                "Training finished but no model is available to report."
            )
            return
        network = self._network
        try:
            self._present_run_results(
                network=network,
                state=state,
                history=history,
                mode=mode,
                epochs=epochs,
                validation_history=validation_history,
                payload=payload,
            )
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            # Presenting results happens in a signal handler: an
            # exception escaping into Qt would terminate the process
            # without a usable message. The trained model stays applied
            # and the reason is reported on the page.
            self._controls.show_error(
                f"Training finished, but its results could not be "
                f"presented: {exc}"
            )
            self.training_failed.emit(str(exc))

    def _present_run_results(
        self,
        *,
        network: NeuralNetwork,
        state: dict,
        history: tuple,
        mode: str,
        epochs: int,
        validation_history,
        payload: dict,
    ) -> None:
        """Score, draw, record, and summarize one completed run.

        Split out of :meth:`_on_worker_finished` so a failure in the
        presentation step is reported on the page instead of escaping
        into the Qt event loop.
        """
        test_metrics = self._evaluate_test_subset(network)
        if test_metrics is not None and test_metrics["predictions"] is not None:
            # Alignment is guaranteed by the evaluation service: the
            # plotted predictions and targets come from the same rows.
            self._controls.show_predictions(
                test_metrics["predictions"],
                test_metrics["targets"],
                classify=test_metrics["binary"],
            )
        else:
            self._controls.clear_predictions()
        if mode == "from_scratch":
            label = "from scratch (previous parameters replaced)"
        else:
            label = "continued (parameters preserved as the starting point)"
        self._controls.display_history(
            mode=label,
            epochs=epochs,
            final_loss=history[-1],
            history=history,
            validation_history=validation_history,
            best_epoch=payload.get("best_epoch"),
        )
        record = self._save_completed_run(
            mode=mode,
            layer_sizes=tuple(state["layer_sizes"]),
            learning_rate=payload.get("learning_rate"),
            epochs=epochs,
            history=history,
            validation_history=validation_history,
            best_epoch=payload.get("best_epoch"),
            best_validation_loss=payload.get("best_validation_loss"),
            test_metrics=test_metrics,
        )
        self._controls.show_status(
            self._format_result_summary(
                label=label,
                epochs=epochs,
                history=history,
                validation_history=validation_history,
                best_epoch=payload.get("best_epoch"),
                best_validation_loss=payload.get("best_validation_loss"),
                test_metrics=test_metrics,
                record_path=record["record_path"],
            )
        )
        self._last_run_summary = self._controls._status_label.text()
        self._last_run_dataset = self._run_dataset
        self._last_run_details = self._build_run_details(
            mode=mode,
            label=label,
            layer_sizes=tuple(state["layer_sizes"]),
            epochs=epochs,
            history=history,
            validation_history=validation_history,
            payload=payload,
            test_metrics=test_metrics,
            record=record,
        )
        self.training_finished.emit(mode)

    def _build_run_details(
        self,
        *,
        mode: str,
        label: str,
        layer_sizes: tuple[int, ...],
        epochs: int,
        history: tuple,
        validation_history,
        payload: dict,
        test_metrics: dict | None,
        record: dict,
    ) -> dict:
        """Collect the finished run's fields for the Dashboard table.

        Every value is either measured by this run, taken from the
        dataset it used, or ``None`` because the run did not produce it.
        Nothing is recomputed or estimated here.
        """
        info = self._pending_run_info or {}
        dataset = self._run_dataset
        sizes = tuple(int(size) for size in layer_sizes)
        return {
            "experiment_id": record.get("experiment_id"),
            "created_at": record.get("created_at"),
            "mode": mode,
            "mode_label": label,
            "preserved_parameters": mode == "continued",
            "layer_sizes": sizes,
            "n_inputs": sizes[0] if sizes else None,
            "hidden_layers": sizes[1:-1],
            "n_outputs": sizes[-1] if sizes else None,
            "learning_rate": payload.get("learning_rate"),
            "epochs": epochs,
            "train_samples": info.get("train_samples"),
            "train_features": info.get("train_features"),
            "train_outputs": info.get("train_outputs"),
            "target_kind": (
                None if dataset is None else dataset.target_kind
            ),
            "task_label": (
                "prediction only"
                if dataset is None
                else TARGET_KIND_LABELS.get(
                    dataset.target_kind, dataset.target_kind
                )
            ),
            "final_train_mse": float(history[-1]) if history else None,
            "final_validation_mse": (
                float(validation_history[-1]) if validation_history else None
            ),
            "validation_samples": info.get("validation_samples"),
            "best_epoch": payload.get("best_epoch"),
            "best_validation_mse": payload.get("best_validation_loss"),
            "test_samples": (
                None if test_metrics is None else test_metrics["samples"]
            ),
            "test_mse": None if test_metrics is None else test_metrics["mse"],
            "test_mae": None if test_metrics is None else test_metrics["mae"],
            "test_rmse": None if test_metrics is None else test_metrics["rmse"],
            "test_accuracy": (
                None if test_metrics is None else test_metrics["accuracy"]
            ),
            "record_path": record.get("record_path"),
            "model_path": record.get("model_path"),
            "split_warning": info.get("warning"),
        }

    def _evaluate_test_subset(self, network: NeuralNetwork) -> dict | None:
        """Score the held-out test subset read-only after training.

        Uses the existing evaluation service only. Returns ``None``
        when no test subset exists, so the caller can say so instead of
        quoting a misleading number. Metrics always cover every output;
        the per-sample series used by the single-output chart are only
        filled in for a one-output model, so no extra target column is
        ever dropped from the reported result.
        """
        splits = self._last_splits
        if splits is None or splits.test is None:
            return None
        try:
            result = evaluate(network, splits.test, dataset_name="test split")
        except (TypeError, ValueError) as exc:
            self._controls.show_error(f"Test evaluation failed: {exc}")
            return None
        n_outputs = int(network.layer_sizes[-1])
        single_output = n_outputs == 1
        return {
            "mse": result.mse,
            "mae": result.mae,
            "rmse": result.rmse,
            "accuracy": result.accuracy,
            "task_label": result.task_label,
            "binary": result.accuracy is not None,
            "samples": result.n_samples,
            "n_outputs": n_outputs,
            "predictions": (
                [float(v) for v in result.predictions[:, 0]]
                if single_output
                else None
            ),
            "targets": (
                [float(v) for v in result.targets[:, 0]]
                if single_output
                else None
            ),
        }

    def _format_result_summary(
        self,
        *,
        label: str,
        epochs: int,
        history: tuple,
        validation_history,
        best_epoch,
        best_validation_loss,
        test_metrics: dict | None,
        record_path: Path | None,
    ) -> str:
        """Compose the educational result line for a finished run."""
        parts = [f"Training complete ({label}): {epochs} epochs"]
        parts.append(f"Final Training MSE {float(history[-1])!r}")
        if validation_history:
            best_text = (
                f" (best epoch {int(best_epoch)} restored, "
                f"validation MSE {float(best_validation_loss)!r})"
                if best_epoch is not None
                else ""
            )
            parts.append(
                f"Final Validation MSE {float(validation_history[-1])!r}{best_text}"
            )
        else:
            parts.append(
                "Validation MSE not recorded: no validation subset "
                "was used, so the final epoch is the model."
            )
        if test_metrics is None:
            parts.append(
                "Test metrics unavailable: the dataset is too small for a "
                "meaningful test subset, so generalization is not measured."
            )
        else:
            plural = "" if test_metrics["n_outputs"] == 1 else "s"
            measures = (
                f"Test MSE {test_metrics['mse']!r}, MAE "
                f"{test_metrics['mae']!r}, RMSE {test_metrics['rmse']!r}"
            )
            if test_metrics["binary"]:
                measures += (
                    f", Evaluation Accuracy {test_metrics['accuracy']!r}"
                )
            parts.append(
                f"Test metrics from the held-out test subset "
                f"({test_metrics['samples']} samples, never used for "
                f"training or model selection, task: "
                f"{test_metrics['task_label']}): {measures} over "
                f"{test_metrics['n_outputs']} output{plural}."
            )
            if test_metrics["predictions"] is None:
                parts.append(
                    "The prediction chart shows one output at a time, so it "
                    "was skipped for this multi-output model; Testing shows "
                    "every output in its table."
                )
        if record_path is not None:
            parts.append(f"Experiment record saved to {record_path}.")
        return " ".join(parts)

    def _save_completed_run(
        self,
        *,
        mode: str,
        layer_sizes: tuple,
        learning_rate: object,
        epochs: int,
        history: tuple,
        validation_history=None,
        best_epoch=None,
        best_validation_loss=None,
        test_metrics: dict | None = None,
    ) -> dict:
        """Save one experiment record (and the model state) if possible.

        Uses only values from the completed run and its validated
        configuration: validation and test metrics stay ``None`` when the
        corresponding subsets did not exist. The trained model stays
        applied regardless of the outcome; failures are reported and
        leave the returned reference ``None``. Nothing is ever
        overwritten: filenames are unique.

        Returns:
            ``{"experiment_id", "created_at", "record_path",
            "model_path"}``; every path is ``None`` when the artifact
            could not be written.
        """
        info = self._pending_run_info or {}
        validation_values = (
            tuple(float(value) for value in validation_history)
            if validation_history
            else None
        )
        model_path = self._save_completed_model(layer_sizes=layer_sizes)
        created_at = datetime.now(timezone.utc).isoformat()
        experiment_id = uuid.uuid4().hex
        try:
            record = ExperimentRecord(
                experiment_id=experiment_id,
                created_at=created_at,
                mode=mode,
                layer_sizes=tuple(int(size) for size in layer_sizes),
                learning_rate=float(learning_rate),  # type: ignore[arg-type]
                epochs=int(epochs),
                train_dataset_name=None,
                train_samples=int(info.get("train_samples", 0)),
                train_features=int(info.get("train_features", 0)),
                train_outputs=int(info.get("train_outputs", 0)),
                test_dataset_name=None,
                loss_history=tuple(float(value) for value in history),
                final_train_loss=float(history[-1]),
                test_mse=None if test_metrics is None else test_metrics["mse"],
                test_accuracy=(
                    None if test_metrics is None else test_metrics["accuracy"]
                ),
                validation_loss=(
                    None
                    if validation_values is None
                    else validation_values[-1]
                ),
                validation_samples=info.get("validation_samples"),
                best_epoch=None if best_epoch is None else int(best_epoch),
                best_validation_loss=(
                    None if best_validation_loss is None
                    else float(best_validation_loss)
                ),
                test_samples=None if test_metrics is None else info.get("test_samples"),
                model_path=str(model_path) if model_path is not None else None,
            )
            ExperimentRecord.from_dict(record.to_dict())
        except (TypeError, ValueError, IndexError) as exc:
            self._controls.show_error(
                f"Experiment record could not be saved: {exc}"
            )
            return _empty_run_reference(experiment_id, created_at, model_path)
        try:
            EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._controls.show_error(
                f"Experiment record could not be saved: {exc}"
            )
            return _empty_run_reference(experiment_id, created_at, model_path)
        for attempt in range(100):
            suffix = "" if attempt == 0 else f"_{attempt + 1}"
            path = EXPERIMENTS_DIR / f"experiment_{record.experiment_id}{suffix}.json"
            if path.exists():
                continue
            try:
                save_experiment_record(record.to_dict(), path)
            except FileExistsError:
                continue
            except (TypeError, ValueError, OSError) as exc:
                self._controls.show_error(
                    f"Experiment record could not be saved: {exc}"
                )
                return _empty_run_reference(
                    experiment_id, created_at, model_path
                )
            return _run_reference(
                experiment_id, created_at, path, model_path
            )
        self._controls.show_error(
            "Experiment record could not be saved: "
            "no unique filename available."
        )
        return _empty_run_reference(experiment_id, created_at, model_path)

    def _save_completed_model(self, *, layer_sizes: tuple) -> Path | None:
        """Save the selected (best) model state; never overwrite a file.

        The in-memory model stays applied whether or not the file is
        written: a failure is reported and returns None.
        """
        if self._network is None or tuple(self._network.layer_sizes) != tuple(
            int(size) for size in layer_sizes
        ):
            return None
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._controls.show_error(f"Model state could not be saved: {exc}")
            return None
        state = self._network.to_state_dict()
        for attempt in range(100):
            suffix = "" if attempt == 0 else f"_{attempt + 1}"
            name = f"model_{uuid.uuid4().hex}{suffix}.json"
            path = MODELS_DIR / name
            if path.exists():
                continue
            try:
                return save_model_state(state, path)
            except FileExistsError:
                continue
            except (TypeError, ValueError, OSError) as exc:
                self._controls.show_error(
                    f"Model state could not be saved: {exc}"
                )
                return None
        self._controls.show_error(
            "Model state could not be saved: no unique filename available."
        )
        return None

    def _on_worker_failed(self, message: str) -> None:
        """Report a worker failure; the previous model is untouched."""
        self._controls.set_running(False)
        self._controls.reset_progress()
        self._set_model_actions_enabled(True)
        self._controls.show_error(f"Training failed: {message}")
        self.training_failed.emit(message)

    def _on_worker_cancelled(self) -> None:
        """Report cancellation; partial epochs are discarded."""
        self._controls.set_running(False)
        self._controls.reset_progress()
        self._set_model_actions_enabled(True)
        self._controls.show_status(
            "Training cancelled; the previous model is unchanged."
        )

    def _on_thread_finished(self) -> None:
        """Release worker-thread references on the GUI thread.

        The worker deletes itself from its own thread (see
        :meth:`_start_worker`), so only the references and the thread
        object are released here; touching the deleted worker from this
        thread would raise.
        """
        thread = self._worker_thread
        self._worker = None
        self._worker_thread = None
        if thread is not None:
            thread.deleteLater()

    def _set_model_actions_enabled(self, enabled: bool) -> None:
        """Enable or disable the Save/Load Model actions."""
        self._save_button.setEnabled(enabled)
        self._load_button.setEnabled(enabled)

    def _on_save_model(self) -> None:
        """Save the current network via a file dialog."""
        self._controls.clear_error()
        self._model_status.setText("")
        if self._network is None:
            self._model_status.setText(
                "No model to save. Train a model first."
            )
            return
        dialog_path, _ = QFileDialog.getSaveFileName(
            self, "Save model state", "", "JSON files (*.json)"
        )
        if not dialog_path:
            return
        path = Path(dialog_path)
        overwrite = False
        if path.exists():
            answer = QMessageBox.question(
                self,
                "Overwrite file",
                f"{path} already exists. Overwrite it?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Yes:
                return
            overwrite = True
        try:
            saved = save_model_state(
                self._network.to_state_dict(), path, overwrite=overwrite
            )
        except (TypeError, ValueError, OSError) as exc:
            self._model_status.setText(f"Model could not be saved: {exc}")
            return
        self._model_status.setText(f"Model saved to {saved}.")

    def _on_load_model(self) -> None:
        """Load a model file, replacing the current network on success."""
        self._controls.clear_error()
        self._model_status.setText("")
        dialog_path, _ = QFileDialog.getOpenFileName(
            self, "Load model state", "", "JSON files (*.json)"
        )
        if not dialog_path:
            return
        try:
            state = load_model_state(dialog_path)
            network = NeuralNetwork.from_state_dict(state)
        except (TypeError, ValueError, OSError) as exc:
            self._model_status.setText(f"Model could not be loaded: {exc}")
            return
        had_model = self._network is not None
        had_results = self._last_run_summary is not None
        self._network = network
        self._model_revision += 1
        self._config.set_hidden_sizes(
            tuple(int(size) for size in network.layer_sizes[1:-1])
        )
        self._config.set_output_size(int(network.layer_sizes[-1]))
        detail = (
            "the previous model was replaced"
            if had_model
            else "a model was loaded"
        )
        cleared = " and its results were cleared" if had_results else ""
        self._model_status.setText(f"Model loaded from {dialog_path} — {detail}.")
        self._invalidate_results(
            f"Model loaded from {dialog_path}: layers "
            f"{tuple(network.layer_sizes)} — {detail}{cleared}."
        )
