"""Network configuration widget: architecture and training settings.

Collects hidden-layer structure, output size, learning rate, and epoch
count from the user and validates them into plain values. The input
size is display-only: it always comes from the selected dataset, never
from user typing. No training mathematics lives here; validated values
are handed to the existing core/application APIs by the owning view.
"""

from __future__ import annotations

import math

from PySide6.QtWidgets import QComboBox
from PySide6.QtWidgets import QFormLayout
from PySide6.QtWidgets import QGroupBox
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QLineEdit
from PySide6.QtWidgets import QSpinBox
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget

from neural_network_lab.data.dataset import NO_TASK
from neural_network_lab.data.dataset import TARGET_KIND_LABELS
from neural_network_lab.data.dataset_split import SPLIT_PRESETS
from neural_network_lab.gui.widgets.ui_theme import CARD_MARGIN
from neural_network_lab.gui.widgets.ui_theme import CONTENT_SPACING

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
            CARD_MARGIN, CARD_MARGIN, CARD_MARGIN, CARD_MARGIN
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
        self._hidden_edit.setToolTip(
            "Number of units in each hidden layer, comma- or "
            "space-separated (e.g. '4, 3'); leave blank for none."
        )
        form.addRow("Hidden layers:", self._hidden_edit)

        self._output_spin = QSpinBox(self)
        self._output_spin.setRange(MIN_OUTPUTS, MAX_OUTPUTS)
        self._output_spin.setValue(1)
        self._output_spin.setObjectName("outputSpin")
        self._output_spin.setToolTip(
            "Number of output units: one per target column of the dataset. "
            "A supervised run is refused when this does not match the "
            "dataset's target columns."
        )
        form.addRow("Output size:", self._output_spin)

        self._output_hint = QLabel("", self)
        self._output_hint.setObjectName("outputHint")
        self._output_hint.setWordWrap(True)
        form.addRow(self._output_hint)

        self._learning_edit = QLineEdit(DEFAULT_LEARNING_RATE_TEXT, self)
        self._learning_edit.setObjectName("learningEdit")
        self._learning_edit.setToolTip(
            "Step size of each weight update; must be greater than zero."
        )
        form.addRow("Learning rate:", self._learning_edit)

        self._epochs_spin = QSpinBox(self)
        self._epochs_spin.setRange(MIN_EPOCHS, MAX_EPOCHS)
        self._epochs_spin.setValue(DEFAULT_EPOCHS)
        self._epochs_spin.setObjectName("epochsSpin")
        self._epochs_spin.setToolTip(
            "How many full passes over the training dataset to run."
        )
        form.addRow("Epochs:", self._epochs_spin)

        self._split_combo = QComboBox(self)
        self._split_combo.setObjectName("datasetSplitCombo")
        for label, train, validation, test in SPLIT_PRESETS:
            self._split_combo.addItem(label, (train, validation, test))
        self._split_combo.setToolTip(
            "How the selected dataset is split into training, validation, "
            "and test rows. Validation rows are used only to pick the best "
            "epoch; test rows are used only to evaluate the finished model. "
            "Small datasets (e.g. XOR with 4 rows) fall back to training on "
            "all rows."
        )
        form.addRow("Data split:", self._split_combo)

        layout.addLayout(form)

    def set_input_size(self, n_features: int | None) -> None:
        """Display the dataset-driven input size (None clears it)."""
        self._input_size = n_features
        text = str(n_features) if n_features is not None else "—"
        self._input_label.setText(f"Input size: {text} (set by the dataset)")

    def set_output_hint(
        self, n_outputs: int | None, target_kind: str | None = None
    ) -> None:
        """State how many outputs the selected dataset provides.

        Args:
            n_outputs: Number of target columns, or ``None`` when no
                labeled dataset is selected.
            target_kind: The dataset's declared task, used to say whether
                the targets are class labels or continuous values.
        """
        if n_outputs is None:
            self._output_hint.setText("")
            self._output_spin.setToolTip(
                "Number of output units: one per target column."
            )
            return
        plural = "" if n_outputs == 1 else "s"
        task_text = (
            ""
            if not target_kind or target_kind == NO_TASK
            else (
                f" · {TARGET_KIND_LABELS[target_kind]}"
                if target_kind in TARGET_KIND_LABELS
                else ""
            )
        )
        self._output_hint.setText(
            f"Dataset target columns: {n_outputs} — set Output size to "
            f"{n_outputs} for supervised training{task_text}."
        )
        self._output_spin.setToolTip(
            f"Number of output units: one per target column of the "
            f"dataset ({n_outputs} selected{plural})."
        )

    def hidden_sizes(self) -> tuple[int, ...]:
        """Return the validated hidden-layer sizes."""
        return parse_hidden_sizes(self._hidden_edit.text())

    def set_hidden_sizes(self, sizes: tuple[int, ...]) -> None:
        """Show ``sizes`` as the hidden-layer configuration."""
        text = ", ".join(str(int(size)) for size in sizes)
        self._hidden_edit.setText(text)
        self._hidden_edit.setCursorPosition(0)

    def set_output_size(self, n_outputs: int) -> None:
        """Show ``n_outputs`` as the configured output size.

        Values outside the control's range are clamped by the spin box;
        the architecture itself stays authoritative in the model.
        """
        self._output_spin.setValue(int(n_outputs))

    def output_size(self) -> int:
        """Return the configured output size."""
        return int(self._output_spin.value())

    def learning_rate(self) -> float:
        """Return the validated learning rate."""
        return parse_learning_rate(self._learning_edit.text())

    def epochs(self) -> int:
        """Return the configured epoch count."""
        return int(self._epochs_spin.value())

    def split_ratios(self) -> tuple[float, float, float]:
        """Return the selected ``(train, validation, test)`` ratios."""
        data = self._split_combo.currentData()
        if not isinstance(data, tuple) or len(data) != 3:
            return (1.0, 0.0, 0.0)
        return tuple(float(value) for value in data)  # type: ignore[return-value]

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
