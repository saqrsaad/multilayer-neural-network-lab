"""Training controls: mode buttons, status, and loss history display.

Two explicit actions map onto the two existing training APIs: training
from scratch and continued training. The widget performs no mathematics
itself; it emits signals that the owning view handles by calling the
core training methods, then renders the returned status and history.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QPlainTextEdit
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget

from neural_network_lab.gui.main_window import CONTENT_MARGIN
from neural_network_lab.gui.main_window import CONTENT_SPACING
from neural_network_lab.gui.widgets.charts import LossHistoryChart


class TrainingControls(QWidget):
    """Buttons, status line, error line, and loss-history readout."""

    train_scratch_requested = Signal()
    train_continued_requested = Signal()

    def __init__(self, parent=None) -> None:
        """Create the training action buttons and result displays."""
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN
        )
        layout.setSpacing(CONTENT_SPACING)

        buttons = QHBoxLayout()
        buttons.setSpacing(CONTENT_SPACING)
        self._scratch_button = QPushButton("Train from scratch", self)
        self._scratch_button.setObjectName("trainScratchButton")
        self._continued_button = QPushButton("Continue training", self)
        self._continued_button.setObjectName("trainContinuedButton")
        buttons.addWidget(self._scratch_button)
        buttons.addWidget(self._continued_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self._status_label = QLabel("No training run yet.", self)
        self._status_label.setWordWrap(True)
        self._status_label.setObjectName("trainingStatus")
        layout.addWidget(self._status_label)

        self._history_view = QPlainTextEdit(self)
        self._history_view.setReadOnly(True)
        self._history_view.setObjectName("lossHistory")
        self._history_view.setPlaceholderText(
            "Per-epoch training loss (MSE) appears here after a run."
        )
        layout.addWidget(self._history_view, 1)

        self._loss_chart = LossHistoryChart(self)
        layout.addWidget(self._loss_chart, 1)

        self._error_label = QLabel("", self)
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #b00020;")
        self._error_label.setObjectName("trainingError")
        layout.addWidget(self._error_label)

        self._scratch_button.clicked.connect(self.train_scratch_requested.emit)
        self._continued_button.clicked.connect(self.train_continued_requested.emit)

    def set_running(self, running: bool) -> None:
        """Disable both actions while a (synchronous) run executes."""
        self._scratch_button.setEnabled(not running)
        self._continued_button.setEnabled(not running)

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
        self, *, mode: str, epochs: int, final_loss: float, history: tuple[float, ...]
    ) -> None:
        """Render one measured per-epoch loss value per line."""
        lines = [
            f"epoch {index + 1}/{epochs}: {loss!r}"
            for index, loss in enumerate(history)
        ]
        self._history_view.setPlainText("\n".join(lines))
        self._loss_chart.plot_history(tuple(history))
        self.show_status(
            f"Training complete ({mode}): {epochs} epochs, "
            f"final training loss (MSE) {final_loss!r}."
        )
