"""Weights and biases editor: inspect and edit model parameters.

The view displays one layer at a time with explicit layer indices,
shapes, and per-cell row/column positions. It operates on the shared
model supplied by a provider (wired to the training view by the main
window; injectable in tests) — never on a duplicate instance. Edits are
parsed and validated in full before anything is applied through the
documented :meth:`NeuralNetwork.set_parameters` API, so a rejected
edit leaves the model completely unchanged. A persistent notice makes
clear that manual edits change predictions.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QComboBox
from PySide6.QtWidgets import QGroupBox
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QTableWidget
from PySide6.QtWidgets import QTableWidgetItem
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget

from neural_network_lab.core.network import NeuralNetwork
from neural_network_lab.gui.main_window import CONTENT_MARGIN
from neural_network_lab.gui.main_window import CONTENT_SPACING


def parse_parameter_cell(
    text: str, *, layer: int, kind: str, row: int, col: int
) -> float:
    """Parse one parameter cell, naming its layer/matrix/row/column.

    Args:
        text: Raw cell text.
        layer: Zero-based layer index.
        kind: ``"W"`` for weights or ``"b"`` for biases.
        row: Zero-based matrix row.
        col: Zero-based matrix column.

    Raises:
        ValueError: For empty, non-numeric, or non-finite cells.
    """
    location = f"layer {layer} {kind}[{row}, {col}]"
    cleaned = text.strip()
    if not cleaned:
        raise ValueError(f"Cell at {location} is empty.")
    try:
        value = float(cleaned)
    except ValueError as exc:
        raise ValueError(f"Cell at {location} is not a number: {text!r}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"Cell at {location} must be finite (no NaN or inf).")
    return value


def format_parameter(value: float) -> str:
    """Format a parameter value at full round-trip precision."""
    return repr(float(value))


def _format_grid(matrix: np.ndarray) -> list[list[str]]:
    """Render an array as text rows at full precision."""
    return [[format_parameter(cell) for cell in row] for row in np.asarray(matrix)]


class ParametersView(QWidget):
    """Inspect and edit the shared model's weights and biases."""

    def __init__(self, parent=None) -> None:
        """Create the layer selector, tables, and action buttons."""
        super().__init__(parent)
        self._network_provider: Callable[[], NeuralNetwork | None] = lambda: None
        self._snapshot: list[list[list[list[str]]]] = []
        self._pending: dict[int, list[list[list[str]]]] = {}
        self._snapshot_signature: tuple | None = None
        self._visible_layer = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN
        )
        layout.setSpacing(CONTENT_SPACING)

        self._info_label = QLabel("No model available.", self)
        self._info_label.setWordWrap(True)
        self._info_label.setObjectName("parametersInfo")
        layout.addWidget(self._info_label)

        self._warning_label = QLabel(
            "Manual edits change this model's predictions: evaluation and "
            "continued training will use the edited parameters.",
            self,
        )
        self._warning_label.setWordWrap(True)
        self._warning_label.setStyleSheet("color: #8a5a00;")
        self._warning_label.setObjectName("parametersWarning")
        layout.addWidget(self._warning_label)

        top_row = QHBoxLayout()
        top_row.setSpacing(CONTENT_SPACING)
        top_row.addWidget(QLabel("Layer:", self))
        self._layer_combo = QComboBox(self)
        self._layer_combo.setObjectName("layerCombo")
        top_row.addWidget(self._layer_combo, 1)
        self._refresh_button = QPushButton("Refresh from model", self)
        self._refresh_button.setObjectName("refreshButton")
        top_row.addWidget(self._refresh_button)
        layout.addLayout(top_row)

        self._weights_group = QGroupBox("Weights", self)
        weights_layout = QVBoxLayout(self._weights_group)
        self._weights_shape = QLabel("", self._weights_group)
        self._weights_shape.setObjectName("weightsShape")
        weights_layout.addWidget(self._weights_shape)
        self._weights_table = QTableWidget(self._weights_group)
        self._weights_table.setObjectName("weightsTable")
        self._weights_table.setAlternatingRowColors(True)
        weights_layout.addWidget(self._weights_table)
        layout.addWidget(self._weights_group, 1)

        self._biases_group = QGroupBox("Biases", self)
        biases_layout = QVBoxLayout(self._biases_group)
        self._biases_shape = QLabel("", self._biases_group)
        self._biases_shape.setObjectName("biasesShape")
        biases_layout.addWidget(self._biases_shape)
        self._biases_table = QTableWidget(self._biases_group)
        self._biases_table.setObjectName("biasesTable")
        self._biases_table.setAlternatingRowColors(True)
        biases_layout.addWidget(self._biases_table)
        layout.addWidget(self._biases_group)

        actions = QHBoxLayout()
        actions.setSpacing(CONTENT_SPACING)
        self._apply_button = QPushButton("Apply edits", self)
        self._apply_button.setObjectName("applyButton")
        actions.addWidget(self._apply_button)
        self._status_label = QLabel("", self)
        self._status_label.setWordWrap(True)
        self._status_label.setObjectName("parametersStatus")
        actions.addWidget(self._status_label, 1)
        layout.addLayout(actions)

        self._error_label = QLabel("", self)
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #b00020;")
        self._error_label.setObjectName("parametersError")
        layout.addWidget(self._error_label)

        self._layer_combo.currentIndexChanged.connect(self._on_layer_changed)
        self._apply_button.clicked.connect(self._on_apply)
        self._refresh_button.clicked.connect(self._on_refresh_clicked)
        self._set_model_loaded(False)

    def set_network_provider(
        self, provider: Callable[[], NeuralNetwork | None]
    ) -> None:
        """Set the callable supplying the shared model to inspect."""
        self._network_provider = provider

    def current_network(self) -> NeuralNetwork | None:
        """Return the supplied model, if any and of the right type."""
        try:
            network = self._network_provider()
        except Exception:
            return None
        return network if isinstance(network, NeuralNetwork) else None

    def show_error(self, message: str) -> None:
        """Present an error inline."""
        self._error_label.setText(message)

    def clear_error(self) -> None:
        """Clear any inline error message."""
        self._error_label.setText("")

    def showEvent(self, event: QShowEvent) -> None:
        """Auto-refresh a clean display when the page becomes visible."""
        super().showEvent(event)
        if not self.has_unsaved_changes():
            self.refresh_from_model()
        elif self._model_signature() != self._snapshot_signature:
            self._status_label.setText(
                "Model changed since this display was loaded; "
                "Refresh from model to reload (pending edits kept)."
            )

    def has_unsaved_changes(self) -> bool:
        """Return True when pending table text differs from the model."""
        if self._snapshot_signature is None:
            return False
        return self._merged_grids() != self._snapshot

    def refresh_from_model(self) -> bool:
        """Reload every table from the current model.

        Returns True when a model was loaded, False with an info
        message when none is available. Pending edits are replaced;
        callers confirm first when edits are pending.
        """
        network = self.current_network()
        if network is None:
            self._info_label.setText(
                "No model available. Configure and train one in "
                "the Training view first."
            )
            self._set_model_loaded(False)
            return False
        sizes = network.layer_sizes
        weights = network.weights
        biases = network.biases
        self._snapshot = [
            [_format_grid(weights[layer]), _format_grid(biases[layer])]
            for layer in range(len(weights))
        ]
        self._pending = {}
        self._snapshot_signature = self._model_signature()
        self._layer_combo.blockSignals(True)
        try:
            self._layer_combo.clear()
            for layer in range(len(weights)):
                self._layer_combo.addItem(
                    f"Layer {layer}: {sizes[layer]} → {sizes[layer + 1]} "
                    f"(W {tuple(weights[layer].shape)}, "
                    f"b {tuple(biases[layer].shape)})"
                )
        finally:
            self._layer_combo.blockSignals(False)
        self._info_label.setText(
            f"Model layers {tuple(sizes)}: {len(weights)} weight/bias pair(s). "
            "Values below are the live shared parameters."
        )
        self._set_model_loaded(True)
        self._visible_layer = 0
        self._render_layer(0)
        self.clear_error()
        self._status_label.setText("Display matches the model.")
        return True

    def _model_signature(self) -> tuple | None:
        """Identify the exact model object and architecture displayed."""
        network = self.current_network()
        if network is None:
            return None
        return (id(network), tuple(network.layer_sizes))

    def _set_model_loaded(self, loaded: bool) -> None:
        """Enable or disable editing controls based on model presence."""
        self._layer_combo.setEnabled(loaded)
        self._apply_button.setEnabled(loaded)
        self._weights_group.setEnabled(loaded)
        self._biases_group.setEnabled(loaded)
        if not loaded:
            self._snapshot = []
            self._pending = {}
            self._snapshot_signature = None

    def _merged_grids(self) -> list[list[list[list[str]]]]:
        """Return per-layer grids with pending edits overlaid on snapshot."""
        return [
            self._pending.get(layer, self._snapshot[layer])
            for layer in range(len(self._snapshot))
        ]

    def _stash_visible_layer(self) -> None:
        """Remember the visible layer's current table text as pending."""
        if self._snapshot_signature is None:
            return
        if 0 <= self._visible_layer < len(self._snapshot):
            self._pending[self._visible_layer] = [
                self._read_table(self._weights_table),
                self._read_table(self._biases_table),
            ]

    def _on_layer_changed(self) -> None:
        """Switch layers, stashing visible edits so none are lost."""
        if self._snapshot_signature is None:
            return
        self._stash_visible_layer()
        index = self._layer_combo.currentIndex()
        if 0 <= index < len(self._snapshot):
            self._visible_layer = index
            self._render_layer(index)

    def _render_layer(self, index: int) -> None:
        """Write one layer's merged grids into the tables and labels."""
        merged = self._merged_grids()[index]
        self._write_table(self._weights_table, merged[0], "in", "out")
        self._write_table(self._biases_table, merged[1], "row", "out")
        self._update_shape_labels(index)

    @staticmethod
    def _write_table(
        table: QTableWidget, grid: list[list[str]], row_name: str, col_name: str
    ) -> None:
        """Write text grids into a table with row/column headers."""
        table.setRowCount(len(grid))
        table.setColumnCount(len(grid[0]) if grid else 0)
        table.setVerticalHeaderLabels([f"{row_name} {i}" for i in range(len(grid))])
        table.setHorizontalHeaderLabels(
            [f"{col_name} {j}" for j in range(len(grid[0]) if grid else 0)]
        )
        for i, row in enumerate(grid):
            for j, text in enumerate(row):
                table.setItem(i, j, QTableWidgetItem(text))

    def _update_shape_labels(self, index: int) -> None:
        """Describe the displayed layer's indices and shapes."""
        weights_table = self._weights_table
        biases_table = self._biases_table
        self._weights_shape.setText(
            f"Layer {index} weights W{index} shape "
            f"({weights_table.rowCount()}, {weights_table.columnCount()}) "
            f"(rows: inputs, cols: outputs)."
        )
        self._biases_shape.setText(
            f"Layer {index} biases b{index} shape "
            f"({biases_table.rowCount()}, {biases_table.columnCount()})."
        )

    @staticmethod
    def _read_table(table: QTableWidget) -> list[list[str]]:
        """Return one table's cell text row by row."""
        rows: list[list[str]] = []
        for i in range(table.rowCount()):
            cells = []
            for j in range(table.columnCount()):
                item = table.item(i, j)
                cells.append("" if item is None else item.text())
            rows.append(cells)
        return rows

    def _on_refresh_clicked(self) -> None:
        """Reload from the model after confirmation while edits pending."""
        if self.has_unsaved_changes():
            answer = QMessageBox.question(
                self,
                "Unsaved edits",
                "Refreshing will discard unsaved table edits. Continue?",
                QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Discard:
                return
        self.refresh_from_model()

    def _on_apply(self) -> None:
        """Validate every edited cell, then apply all layers atomically."""
        self.clear_error()
        network = self.current_network()
        if network is None:
            self.show_error(
                "No model available. Configure and train one in "
                "the Training view first."
            )
            return
        if self._snapshot_signature is None:
            self.show_error("Refresh the display before editing.")
            return
        self._stash_visible_layer()
        old_weights = network.weights
        old_biases = network.biases
        if len(old_weights) != len(self._snapshot):
            self.show_error(
                "Model architecture changed since this display was loaded. "
                "Refresh from model before editing."
            )
            return
        try:
            new_weights = []
            new_biases = []
            for layer, (weight_grid, bias_grid) in enumerate(self._merged_grids()):
                expected_w = old_weights[layer].shape
                expected_b = old_biases[layer].shape
                if (len(weight_grid), len(weight_grid[0])) != (
                    expected_w[0], expected_w[1]):
                    raise ValueError(
                        f"Layer {layer} weights table shape changed; "
                        "refresh before editing."
                    )
                if (len(bias_grid), len(bias_grid[0])) != (
                    expected_b[0], expected_b[1]):
                    raise ValueError(
                        f"Layer {layer} biases table shape changed; "
                        "refresh before editing."
                    )
                new_weights.append([
                    [parse_parameter_cell(cell, layer=layer, kind="W", row=i, col=j)
                     for j, cell in enumerate(row)]
                    for i, row in enumerate(weight_grid)
                ])
                new_biases.append([
                    [parse_parameter_cell(cell, layer=layer, kind="b", row=i, col=j)
                     for j, cell in enumerate(row)]
                    for i, row in enumerate(bias_grid)
                ])
        except (TypeError, ValueError) as exc:
            self.show_error(str(exc))
            return
        try:
            network.set_parameters(new_weights, new_biases)
        except (TypeError, ValueError) as exc:
            self.show_error(f"Edit rejected: {exc}")
            return
        changed = [
            layer for layer in range(len(old_weights))
            if not (np.asarray(old_weights[layer])
                    == np.asarray(network.weights[layer])).all()
            or not (np.asarray(old_biases[layer])
                    == np.asarray(network.biases[layer])).all()
        ]
        self._pending = {}
        self._snapshot = [
            [_format_grid(network.weights[layer]), _format_grid(network.biases[layer])]
            for layer in range(len(old_weights))
        ]
        self._snapshot_signature = self._model_signature()
        self._render_layer(self._visible_layer)
        if changed:
            self._status_label.setText(
                f"Edits applied to layer(s) {changed}; predictions made with "
                "this model will now differ."
            )
        else:
            self._status_label.setText("Values unchanged; model untouched.")
