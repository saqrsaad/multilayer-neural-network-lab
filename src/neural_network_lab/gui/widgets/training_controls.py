"""Training controls: mode buttons, progress, status, and result tabs.

Two explicit actions map onto the two existing training APIs: training
from scratch and continued training; a third action asks the owning
view to cancel a running job. The widget performs no mathematics
itself; it emits signals that the owning view handles by calling the
core training methods, then renders the returned status, progress, and
history.

Results are shown one view at a time in a tab group (per-epoch loss
readout, loss curve, prediction comparison) instead of stacking three
tall result panels on top of each other.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPlainTextEdit
from PySide6.QtWidgets import QProgressBar
from PySide6.QtWidgets import QTabWidget
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget

from neural_network_lab.gui.widgets.charts import LossHistoryChart
from neural_network_lab.gui.widgets.charts import PredictionChart
from neural_network_lab.gui.widgets.page_scaffold import FeedbackArea
from neural_network_lab.gui.widgets.page_scaffold import button_row
from neural_network_lab.gui.widgets.page_scaffold import hint_text
from neural_network_lab.gui.widgets.page_scaffold import make_button
from neural_network_lab.gui.widgets.ui_theme import CONTENT_SPACING


class TrainingControls(QWidget):
    """Buttons, progress, feedback, and the recorded result displays."""

    train_scratch_requested = Signal()
    train_continued_requested = Signal()
    cancel_requested = Signal()

    def __init__(self, parent=None) -> None:
        """Create the training action buttons and result displays."""
        super().__init__(parent)
        self._progress_range = 0
        self._progress_value = -1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(CONTENT_SPACING)

        self._scratch_button = make_button(
            "Train from scratch",
            kind="primary",
            tooltip=(
                "Train a new model from newly initialized weights, replacing "
                "any previous parameters."
            ),
            parent=self,
        )
        self._scratch_button.setObjectName("trainScratchButton")
        self._continued_button = make_button(
            "Continue training",
            kind="secondary",
            tooltip=(
                "Continue training from the current model parameters instead "
                "of newly initialized weights."
            ),
            parent=self,
        )
        self._continued_button.setObjectName("trainContinuedButton")
        self._cancel_button = make_button(
            "Cancel run",
            kind="danger",
            tooltip="Stop the running job after its current epoch; the "
            "previous model is kept.",
            parent=self,
        )
        self._cancel_button.setObjectName("cancelTrainingButton")
        layout.addLayout(
            button_row(
                self._scratch_button,
                self._continued_button,
                self._cancel_button,
            )
        )
        layout.addWidget(
            hint_text(
                "From scratch replaces the current parameters; continue "
                "training keeps them as the starting point.",
                self,
            )
        )

        self._progress = QProgressBar(self)
        self._progress.setObjectName("trainingProgress")
        self._progress.setTextVisible(True)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("Ready to train")
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._feedback = FeedbackArea(self, status_text="No training run yet.")
        self._status_label = self._feedback.status_label()
        self._error_label = self._feedback.error_label()
        layout.addWidget(self._feedback)

        self._results = QTabWidget(self)
        self._results.setObjectName("trainingResults")
        self._history_view = QPlainTextEdit(self._results)
        self._history_view.setReadOnly(True)
        self._history_view.setObjectName("lossHistory")
        self._history_view.setPlaceholderText(
            "Per-epoch training loss (MSE) appears here after a run."
        )
        self._results.addTab(self._history_view, "Loss history")
        self._loss_chart = LossHistoryChart(self._results)
        self._results.addTab(self._loss_chart, "Loss curve")
        self._prediction_chart = PredictionChart(self._results)
        self._prediction_chart.setObjectName("trainingPredictionChart")
        self._results.addTab(self._prediction_chart, "Test predictions")
        layout.addWidget(self._results, 1)

        self._scratch_button.clicked.connect(self.train_scratch_requested.emit)
        self._continued_button.clicked.connect(self.train_continued_requested.emit)
        self._cancel_button.clicked.connect(self.cancel_requested.emit)
        self._cancel_button.setEnabled(False)

    def set_running(self, running: bool) -> None:
        """Enable the actions that make sense while a run is in flight."""
        self._scratch_button.setEnabled(not running)
        self._continued_button.setEnabled(not running)
        self._cancel_button.setEnabled(running)
        if running:
            self._progress.setVisible(True)
            self._progress_range = 0
            self._progress_value = -1
            self._progress.setRange(0, 1)
            self._progress.setValue(0)
            self._progress.setFormat("Starting…")

    def set_progress(self, completed: int, total: int) -> None:
        """Show completed epochs against the requested epoch count.

        Repeated identical values are ignored: the widget is updated from
        the worker thread's progress signal, and each real update forces
        a relayout of the page and its charts.
        """
        epochs = max(1, int(total))
        value = min(max(0, int(completed)), epochs)
        if value == self._progress_value and epochs == self._progress_range:
            return
        self._progress_range = epochs
        self._progress_value = value
        self._progress.setVisible(True)
        self._progress.setRange(0, epochs)
        self._progress.setValue(value)
        self._progress.setFormat(f"Epoch %v of {epochs}")

    def reset_progress(self) -> None:
        """Hide the progress bar after a run ends."""
        self._progress.setVisible(False)
        self._progress.setValue(0)
        self._progress_range = 0
        self._progress_value = -1

    def select_results_tab(self, index: int) -> None:
        """Bring one result view forward (0 loss history, 1 curve, 2 predictions)."""
        self._results.setCurrentIndex(index)

    def show_status(self, message: str) -> None:
        """Present a training status message."""
        self._status_label.setText(message)

    def show_error(self, message: str) -> None:
        """Present a training failure or validation message."""
        self._error_label.setText(message)

    def clear_error(self) -> None:
        """Clear any error message."""
        self._error_label.setText("")

    def display_history(
        self,
        *,
        mode: str,
        epochs: int,
        final_loss: float,
        history: tuple[float, ...],
        validation_history=None,
        best_epoch: int | None = None,
    ) -> None:
        """Render one measured per-epoch loss value per line.

        Both the text readout and the chart are refreshed from the same
        recorded histories; the validation series appears only when the
        run actually recorded one.
        """
        lines = [
            f"epoch {index + 1}/{epochs}: {loss!r}"
            for index, loss in enumerate(history)
        ]
        if validation_history:
            lines.append("validation MSE per epoch:")
            lines.extend(
                f"epoch {index + 1}/{epochs}: {loss!r}"
                for index, loss in enumerate(validation_history)
            )
            if best_epoch is not None:
                lines.append(
                    f"best validation epoch {int(best_epoch)} "
                    "(restored after training)"
                )
        self._history_view.setPlainText("\n".join(lines))
        self._loss_chart.plot_history(
            tuple(history), validation_history, best_epoch
        )
        self.select_results_tab(1)
        self.show_status(
            f"Training complete ({mode}): {epochs} epochs, "
            f"final training loss (MSE) {final_loss!r}."
        )

    def show_predictions(
        self, probabilities, targets=None, *, classify: bool = True
    ) -> bool:
        """Refresh the prediction comparison chart from a finished run.

        ``classify`` is ``False`` for continuous targets, which are never
        thresholded into classes.

        Returns the chart's own verdict so callers can report when the
        recorded predictions could not be plotted.
        """
        drawn = self._prediction_chart.plot_predictions(
            probabilities, targets, classify=classify
        )
        if drawn:
            self.select_results_tab(2)
        return drawn

    def clear_history(self) -> None:
        """Reset the loss readout and chart to their empty states."""
        self._history_view.clear()
        self._loss_chart.show_message(
            "No training history yet. Train the model to record loss values."
        )
        self.select_results_tab(0)

    def clear_predictions(self) -> None:
        """Reset the prediction chart to its empty state."""
        self._prediction_chart.show_message(
            "No test evaluation yet. Train with a test split to compare "
            "predictions against targets."
        )