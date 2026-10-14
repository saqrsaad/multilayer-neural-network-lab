"""Network configuration widget: architecture and training settings.

Collects hidden-layer structure, output size, learning rate, and epoch
count from the user and validates them into plain values. The input
size is display-only: it always comes from the selected dataset, never
from user typing. No training mathematics lives here; validated values
are handed to the existing core/application APIs by the owning view.
"""

from __future__ import annotations

import math

from PySide6.QtWidgets import QFormLayout
from PySide6.QtWidgets import QGroupBox
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QLineEdit
from PySide6.QtWidgets import QSpinBox
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget

from neural_network_lab.gui.main_window import CONTENT_MARGIN
from neural_network_lab.gui.main_window import CONTENT_SPACING

#: Default hidden-layer structure text.
DEFAULT_HIDDEN_TEXT = "4"

#: Default learning rate text.
DEFAULT_LEARNING_RATE_TEXT = "0.1"

#: Default epoch count.
DEFAULT_EPOCHS = 100

#: Bounds for the epoch control.
MIN_EPOCHS = 1
MAX_EPOCHS = 100000

#: Bounds for the output-size control.
MIN_OUTPUTS = 1
MAX_OUTPUTS = 32


def parse_hidden_sizes(text: str) -> tuple[int, ...]:
    """Parse comma/space-separated hidden sizes; empty means no hidden layer.

    Raises:
        ValueError: With a message naming the offending entry.
    """
    cleaned = text.replace(",", " ").split()
    if not cleaned:
        return ()
    sizes: list[int] = []
    for entry in cleaned:
        try:
            size = int(entry)
        except ValueError as exc:
            raise ValueError(
                f"Hidden size {entry!r} is not an integer; "
                "use e.g. '4, 3' or leave blank for none."
            ) from exc
        if size < 1:
            raise ValueError(
                f"Hidden size {entry!r} must be a positive integer."
            )
        sizes.append(size)
    return tuple(sizes)


def parse_learning_rate(text: str) -> float:
    """Parse and validate a learning-rate string.

    Raises:
        ValueError: For empty, non-numeric, non-finite, or
            non-positive input.
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Learning rate is empty.")
    try:
        value = float(cleaned)
    except ValueError as exc:
        raise ValueError(
            f"Learning rate {text!r} is not a number."
        ) from exc
    if not math.isfinite(value):
        raise ValueError("Learning rate must be finite (no NaN or inf).")
    if value <= 0.0:
        raise ValueError("Learning rate must be greater than zero.")
    return value


class NetworkConfigWidget(QGroupBox):
    """Editable network architecture and training settings."""

    def __init__(self, parent=None) -> None:
        """Create the configuration controls with sane defaults."""
        super().__init__("Network configuration", parent)
        self._input_size: int | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN
        )
        layout.setSpacing(CONTENT_SPACING)
        form = QFormLayout()
        form.setSpacing(CONTENT_SPACING)

        self._input_label = QLabel("Input size: — (set by the dataset)", self)
        self._input_label.setObjectName("inputSizeLabel")
        form.addRow(self._input_label)

        self._hidden_edit = QLineEdit(DEFAULT_HIDDEN_TEXT, self)
        self._hidden_edit.setObjectName("hiddenEdit")
        self._hidden_edit.setPlaceholderText("e.g. 4, 3 (blank for none)")
        form.addRow("Hidden layers:", self._hidden_edit)

        self._output_spin = QSpinBox(self)
        self._output_spin.setRange(MIN_OUTPUTS, MAX_OUTPUTS)
        self._output_spin.setValue(1)
        self._output_spin.setObjectName("outputSpin")
        form.addRow("Output size:", self._output_spin)

        self._learning_edit = QLineEdit(DEFAULT_LEARNING_RATE_TEXT, self)
        self._learning_edit.setObjectName("learningEdit")
        form.addRow("Learning rate:", self._learning_edit)

        self._epochs_spin = QSpinBox(self)
        self._epochs_spin.setRange(MIN_EPOCHS, MAX_EPOCHS)
        self._epochs_spin.setValue(DEFAULT_EPOCHS)
        self._epochs_spin.setObjectName("epochsSpin")
        form.addRow("Epochs:", self._epochs_spin)

        layout.addLayout(form)

    def set_input_size(self, n_features: int | None) -> None:
        """Display the dataset-driven input size (None clears it)."""
        self._input_size = n_features
        text = str(n_features) if n_features is not None else "—"
        self._input_label.setText(f"Input size: {text} (set by the dataset)")

    def hidden_sizes(self) -> tuple[int, ...]:
        """Return the validated hidden-layer sizes."""
        return parse_hidden_sizes(self._hidden_edit.text())

    def output_size(self) -> int:
        """Return the configured output size."""
        return int(self._output_spin.value())

    def learning_rate(self) -> float:
        """Return the validated learning rate."""
        return parse_learning_rate(self._learning_edit.text())

    def epochs(self) -> int:
        """Return the configured epoch count."""
        return int(self._epochs_spin.value())

    def layer_sizes(self) -> tuple[int, ...]:
        """Return the full ``[input, *hidden, output]`` architecture.

        Raises:
            ValueError: When no dataset input size is set or any part
                of the configuration is invalid.
        """
        if self._input_size is None:
            raise ValueError(
                "No dataset selected: input size is unknown. "
                "Validate a dataset first."
            )
        return (self._input_size, *self.hidden_sizes(), self.output_size())
