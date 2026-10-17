"""Network configuration view: edit the active model architecture.

The view builds the full layer list from separate, explicit controls —
a display-only input size taken from the dataset, one row per hidden
layer that can be added, edited, and removed, and an output-neuron
count — and applies it to the *active* network owned by the training
view, so training, testing, and the parameters view all see the same
model. No network mathematics lives here: the controls produce a plain
tuple of positive integers that the injected applier turns into a
fresh :class:`NeuralNetwork`, and any rejection leaves the active
network untouched.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QSpinBox
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget

from neural_network_lab.data.dataset import Dataset
from neural_network_lab.gui.widgets.page_scaffold import FeedbackArea
from neural_network_lab.gui.widgets.page_scaffold import MessageLabel
from neural_network_lab.gui.widgets.page_scaffold import SectionCard
from neural_network_lab.gui.widgets.page_scaffold import button_row
from neural_network_lab.gui.widgets.page_scaffold import install_page
from neural_network_lab.gui.widgets.page_scaffold import make_button
from neural_network_lab.gui.widgets.page_scaffold import selectable_label
from neural_network_lab.gui.widgets.ui_theme import FIELD_SPACING

#: Bounds for one layer's neuron count.
MIN_NEURONS = 1
MAX_NEURONS = 256

#: Bounds for the output-neuron count.
MIN_OUTPUTS = 1
MAX_OUTPUTS = 32

#: Largest number of hidden layers the editor offers.
MAX_HIDDEN_LAYERS = 8

#: Neuron count used for a newly added hidden layer.
DEFAULT_HIDDEN_NEURONS = 4


def parse_layer_sizes(text: str) -> tuple[int, ...]:
    """Parse a comma/space-separated layer-size list.

    Returns:
        The layer sizes as a tuple of positive integers.

    Raises:
        ValueError: With a message naming the offending entry, when any
            entry is not a positive integer or fewer than two layers
            are given.
    """
    cleaned = text.replace(",", " ").split()
    if not cleaned:
        raise ValueError(
            "Enter at least two layer sizes, e.g. '2, 4, 1'."
        )
    sizes: list[int] = []
    for entry in cleaned:
        try:
            size = int(entry)
        except ValueError as exc:
            raise ValueError(
                f"Layer size {entry!r} is not an integer; "
                "use e.g. '2, 4, 1'."
            ) from exc
        if size < 1:
            raise ValueError(
                f"Layer size {entry!r} must be a positive integer."
            )
        sizes.append(size)
    if len(sizes) < 2:
        raise ValueError(
            "Layer sizes must list at least two layers "
            "(input and output), e.g. '2, 4, 1'."
        )
    return tuple(sizes)


def describe_architecture(sizes: tuple[int, ...]) -> str:
    """Describe layer sizes as ``2 inputs → 4 → 3 → 1 output``."""
    if not sizes:
        return "—"
    parts = [f"{sizes[0]} input" + ("" if sizes[0] == 1 else "s")]
    parts.extend(str(size) for size in sizes[1:-1])
    parts.append(f"{sizes[-1]} output" + ("" if sizes[-1] == 1 else "s"))
    return " → ".join(parts)


class NetworkConfigurationView(QWidget):
    """Edit the active network's layers and output size."""

    def __init__(self, parent=None) -> None:
        """Create the layer controls and result displays."""
        super().__init__(parent)
        self._applier: Callable[[tuple[int, ...]], None] | None = None
        self._sizes_provider: Callable[
            [], tuple[int, ...] | None
        ] | None = None
        self._dataset_provider: Callable[[], Dataset | None] = lambda: None
        self._hidden_rows: list[QSpinBox] = []
        self._editor_dirty = False

        self._page = install_page(
            self,
            "Network Configuration",
            "Build the active model's layers, then apply them. Training, "
            "Testing, Weights and Biases, and Visualization all read this "
            "one model.",
        )

        current_card = SectionCard(
            "Current architecture",
            description=(
                "The layers of the model every other section uses. A "
                "changed architecture replaces it and its trained "
                "parameters are discarded."
            ),
            parent=self._page,
        )
        self._current_label = MessageLabel("info", current_card)
        self._current_label.setText(
            "Current architecture: no active network yet."
        )
        current_card.add(self._current_label)
        self._page.add(current_card)

        editor = SectionCard(
            "Layers",
            description=(
                "The input size always comes from the dataset. Add, "
                "edit, or remove hidden layers freely, and set the output "
                "count to the number of target columns you want to "
                "predict."
            ),
            parent=self._page,
        )

        self._input_label = selectable_label(
            "Input size: — (set by the dataset)", editor
        )
        self._input_label.setObjectName("inputSizeLabel")
        editor.add(self._input_label)

        self._hidden_box = QWidget(editor)
        self._hidden_layout = QVBoxLayout(self._hidden_box)
        self._hidden_layout.setContentsMargins(0, 0, 0, 0)
        self._hidden_layout.setSpacing(FIELD_SPACING)
        self._hidden_title = QLabel("Hidden layers:", self._hidden_box)
        self._hidden_title.setObjectName("hiddenLayersTitle")
        self._hidden_layout.addWidget(self._hidden_title)
        self._hidden_empty = MessageLabel("hint", self._hidden_box)
        self._hidden_empty.setText(
            "No hidden layer: the network maps inputs straight to outputs."
        )
        self._hidden_layout.addWidget(self._hidden_empty)
        editor.add(self._hidden_box)
        self._add_hidden_button = make_button(
            "Add hidden layer",
            kind="secondary",
            tooltip="Append one hidden layer to the architecture.",
            parent=editor,
        )
        self._add_hidden_button.setObjectName("addHiddenLayerButton")
        editor.add_layout(button_row(self._add_hidden_button))

        self._output_spin = QSpinBox(editor)
        self._output_spin.setRange(MIN_OUTPUTS, MAX_OUTPUTS)
        self._output_spin.setValue(1)
        self._output_spin.setObjectName("outputNeuronsSpin")
        self._output_spin.setToolTip(
            "Number of output neurons: one per target column of the "
            "dataset."
        )
        output_row = QWidget(editor)
        output_row_layout = QHBoxLayout(output_row)
        output_row_layout.setContentsMargins(0, 0, 0, 0)
        output_row_layout.setSpacing(FIELD_SPACING)
        output_row_layout.addWidget(QLabel("Output neurons:", output_row))
        output_row_layout.addWidget(self._output_spin, 1)
        editor.add(output_row)

        self._preview_label = MessageLabel("status", editor)
        self._preview_label.setText("")
        editor.add(self._preview_label)

        self._apply_button = make_button(
            "Apply configuration",
            kind="primary",
            tooltip="Build the active model with these layers.",
            parent=editor,
        )
        self._apply_button.setObjectName("applyNetworkConfigButton")
        editor.add_layout(button_row(self._apply_button))
        self._page.add(editor)

        self._feedback = FeedbackArea(parent=self._page)
        self._status_label = self._feedback.status_label()
        self._error_label = self._feedback.error_label()
        self._page.add(self._feedback)
        self._page.finish()

        self._add_hidden_button.clicked.connect(self._on_add_hidden_layer)
        self._output_spin.valueChanged.connect(self._on_control_changed)
        self._apply_button.clicked.connect(self._on_apply)
        self._apply_button.setDefault(True)
        self._update_hidden_state()

    # ------------------------------------------------------------------
    # Injected providers
    # ------------------------------------------------------------------

    def set_network_applier(
        self, applier: Callable[[tuple[int, ...]], None]
    ) -> None:
        """Set the callable that applies a layer-size configuration.

        The applier receives the validated layer sizes and is expected
        to raise ``ValueError`` with a user-readable message when it
        cannot apply them.
        """
        self._applier = applier

    def set_current_sizes_provider(
        self, provider: Callable[[], tuple[int, ...] | None]
    ) -> None:
        """Set the callable returning the active network's layer sizes."""
        self._sizes_provider = provider

    def set_dataset_provider(
        self, provider: Callable[[], Dataset | None]
    ) -> None:
        """Set the callable supplying the dataset that fixes the input size."""
        self._dataset_provider = provider

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        """Refresh the current architecture whenever the page is shown."""
        super().showEvent(event)
        self._refresh_current_architecture()

    # ------------------------------------------------------------------
    # Layer controls
    # ------------------------------------------------------------------

    def _on_control_changed(self) -> None:
        """Mark the editor as user-edited and refresh the preview."""
        self._editor_dirty = True
        self._update_preview()

    def _input_size(self) -> int | None:
        """Return the dataset's feature count, or None when unavailable."""
        try:
            dataset = self._dataset_provider()
        except Exception:
            return None
        if not isinstance(dataset, Dataset):
            return None
        return dataset.n_features

    def _add_hidden_row(self, neurons: int = DEFAULT_HIDDEN_NEURONS) -> None:
        """Append one hidden-layer row with a neuron count."""
        row = QWidget(self._hidden_box)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(FIELD_SPACING)
        position = len(self._hidden_rows) + 1
        row_layout.addWidget(QLabel(f"Hidden layer {position}:", row))
        spin = QSpinBox(row)
        spin.setRange(MIN_NEURONS, MAX_NEURONS)
        spin.setValue(int(neurons))
        spin.setObjectName(f"hiddenLayerSpin{position}")
        spin.setToolTip("Number of neurons in this hidden layer.")
        row_layout.addWidget(spin, 1)
        remove = make_button(
            "Remove",
            kind="danger",
            tooltip=f"Remove hidden layer {position} from the configuration.",
            parent=row,
        )
        remove.setObjectName(f"removeHiddenLayerButton{position}")
        row_layout.addWidget(remove)
        self._hidden_layout.addWidget(row)
        self._hidden_rows.append(spin)
        spin.valueChanged.connect(self._on_control_changed)
        remove.clicked.connect(
            lambda _checked=False, target=spin: self._remove_hidden_row(target)
        )

    def _remove_hidden_row(self, spin: QSpinBox) -> None:
        """Remove one hidden-layer row and renumber the rest."""
        if spin not in self._hidden_rows:
            return
        row = spin.parentWidget()
        self._hidden_rows.remove(spin)
        self._hidden_layout.removeWidget(row)
        row.setParent(None)
        row.deleteLater()
        self._editor_dirty = True
        self._renumber_hidden_rows()
        self._update_hidden_state()
        self._update_preview()

    def _renumber_hidden_rows(self) -> None:
        """Keep the layer captions and widget names sequential."""
        for position, spin in enumerate(self._hidden_rows, start=1):
            row = spin.parentWidget()
            if row is None:
                continue
            label = row.layout().itemAt(0).widget()
            if isinstance(label, QLabel):
                label.setText(f"Hidden layer {position}:")
            spin.setObjectName(f"hiddenLayerSpin{position}")

    def _on_add_hidden_layer(self) -> None:
        """Add one hidden layer, respecting the editor's layer limit."""
        if len(self._hidden_rows) >= MAX_HIDDEN_LAYERS:
            self._error_label.setText(
                f"This editor supports at most {MAX_HIDDEN_LAYERS} hidden "
                "layers."
            )
            return
        self._error_label.setText("")
        self._add_hidden_row()
        self._editor_dirty = True
        self._update_hidden_state()
        self._update_preview()

    def _set_hidden_rows(self, sizes: tuple[int, ...]) -> None:
        """Rebuild the hidden-layer rows from ``sizes`` without events."""
        for spin in list(self._hidden_rows):
            row = spin.parentWidget()
            self._hidden_layout.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self._hidden_rows = []
        previous = self._editor_dirty
        self._editor_dirty = False
        try:
            for size in sizes[:MAX_HIDDEN_LAYERS]:
                self._add_hidden_row(int(size))
        finally:
            self._editor_dirty = previous
        self._update_hidden_state()

    def _update_hidden_state(self) -> None:
        """Show the empty state and enable the add button as needed."""
        self._hidden_empty.setVisible(not self._hidden_rows)
        self._add_hidden_button.setEnabled(
            len(self._hidden_rows) < MAX_HIDDEN_LAYERS
        )

    def hidden_sizes(self) -> tuple[int, ...]:
        """Return the configured hidden-layer sizes in order."""
        return tuple(int(spin.value()) for spin in self._hidden_rows)

    def layer_sizes(self) -> tuple[int, ...]:
        """Return the full ``[input, *hidden, output]`` architecture.

        Raises:
            ValueError: When no dataset input size is available.
        """
        input_size = self._input_size()
        if input_size is None:
            raise ValueError(
                "No dataset selected, so the input size is unknown. "
                "Validate a dataset in Dataset Management first."
            )
        hidden = self.hidden_sizes()
        outputs = int(self._output_spin.value())
        if not hidden:
            return (input_size, outputs)
        return (input_size, *hidden, outputs)

    def _update_preview(self) -> None:
        """Describe the architecture the controls currently describe."""
        input_size = self._input_size()
        if input_size is None:
            self._preview_label.setText(
                "Select and validate a dataset to fix the input size."
            )
            return
        sizes = self.layer_sizes()
        self._preview_label.setText(
            "Configuration to apply: "
            f"{describe_architecture(sizes)} {self._format_layers(sizes)}"
        )

    # ------------------------------------------------------------------
    # Current architecture and applying
    # ------------------------------------------------------------------

    def _refresh_current_architecture(self) -> None:
        """Display the active network's layer sizes, when available."""
        input_size = self._input_size()
        if input_size is None:
            self._input_label.setText("Input size: — (set by the dataset)")
        else:
            self._input_label.setText(
                f"Input size: {input_size} (set by the dataset)"
            )
        sizes = self._current_sizes()
        if sizes:
            self._current_label.setText(
                f"Current architecture: {self._format_layers(sizes)} — "
                f"{describe_architecture(sizes)}"
            )
            if not self._editor_dirty:
                self._output_spin.blockSignals(True)
                try:
                    self._output_spin.setValue(int(sizes[-1]))
                finally:
                    self._output_spin.blockSignals(False)
                self._set_hidden_rows(tuple(int(size) for size in sizes[1:-1]))
        else:
            self._current_label.setText(
                "Current architecture: no active network yet."
            )
        self._update_preview()

    def _format_layers(self, sizes: tuple[int, ...]) -> str:
        """Format layer sizes as ``[n0, n1, ...]``."""
        return "[" + ", ".join(str(size) for size in sizes) + "]"

    def _on_apply(self) -> None:
        """Validate the entered sizes and apply them to the network."""
        self._status_label.setText("")
        self._error_label.setText("")
        try:
            sizes = self.layer_sizes()
        except ValueError as exc:
            self._error_label.setText(f"Invalid configuration: {exc}")
            return
        if self._applier is None:
            self._error_label.setText(
                "No active network to configure yet: train a model first."
            )
            return
        current = self._current_sizes()
        if current is not None and current == sizes:
            # Same architecture: applying would only discard the current
            # parameters, so the existing model is kept untouched.
            self._status_label.setText(
                f"Configuration unchanged: the model already uses "
                f"{self._format_layers(sizes)}."
            )
            self._refresh_current_architecture()
            return
        if current is not None and not self._confirm_replacement(sizes):
            return
        try:
            self._applier(sizes)
        except (TypeError, ValueError) as exc:
            self._error_label.setText(f"Configuration not applied: {exc}")
            return
        self._status_label.setText(
            f"Configuration applied: {self._format_layers(sizes)} — "
            f"{describe_architecture(sizes)}."
        )
        self._editor_dirty = False
        self._refresh_current_architecture()

    def _current_sizes(self) -> tuple[int, ...] | None:
        """Return the active network's layer sizes, if a model exists."""
        if self._sizes_provider is None:
            return None
        try:
            sizes = self._sizes_provider()
        except Exception:
            return None
        if not sizes:
            return None
        try:
            return tuple(int(size) for size in sizes)
        except (TypeError, ValueError):
            return None

    def _confirm_replacement(self, sizes: tuple[int, ...]) -> bool:
        """Ask before a configuration change that would drop parameters.

        Returns True when no model exists yet (nothing to lose) or the
        user confirmed; False keeps the current model untouched.
        """
        answer = QMessageBox.question(
            self,
            "Replace current model",
            f"Applying {self._format_layers(sizes)} replaces the current "
            "model and discards its trained parameters. Continue?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return answer == QMessageBox.Yes