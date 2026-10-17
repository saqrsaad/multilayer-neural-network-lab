"""Visualization view: decision boundary, prediction surface, architecture.

All panels are display-only. The projection panel asks the existing
prediction API (:meth:`NeuralNetwork.forward`) for the model's output on
a regular grid built from the dataset's own feature ranges, then draws
either class regions and the classification boundary (binary targets)
or the predicted value surface with its colour scale (continuous
targets). Because both are two-dimensional pictures, the page lets the
user choose which two input features (and which output, for a
multi-output model) it shows; every other feature is held at its column
mean and that choice is stated next to the chart. The architecture
panel reads the model's actual layer sizes and draws its neurons and
connections, with matplotlib's standard zoom and pan toolbar so large
architectures stay navigable.

No training, loss, gradient, evaluation, or weight mathematics happens
here, and no model parameter is ever written: the view only calls
public read-only APIs and renders what they return.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QComboBox
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QSplitter
from PySide6.QtWidgets import QStackedWidget
from PySide6.QtWidgets import QWidget

from neural_network_lab.core.network import NeuralNetwork
from neural_network_lab.data.dataset import Dataset
from neural_network_lab.gui.widgets.charts import ArchitectureChart
from neural_network_lab.gui.widgets.charts import create_navigation_toolbar
from neural_network_lab.gui.widgets.charts import DecisionBoundaryChart
from neural_network_lab.gui.widgets.charts import RegressionSurfaceChart
from neural_network_lab.gui.widgets.page_scaffold import FeedbackArea
from neural_network_lab.gui.widgets.page_scaffold import MessageLabel
from neural_network_lab.gui.widgets.page_scaffold import SectionCard
from neural_network_lab.gui.widgets.page_scaffold import button_row
from neural_network_lab.gui.widgets.page_scaffold import install_page
from neural_network_lab.gui.widgets.page_scaffold import make_button
from neural_network_lab.gui.widgets.ui_theme import FIELD_SPACING

#: Grid points sampled along each feature axis for the boundary regions.
BOUNDARY_GRID_RESOLUTION = 60

#: Extra margin added around the dataset's own feature range.
BOUNDARY_MARGIN_RATIO = 0.15

#: Half-width used when a feature column holds a single value.
BOUNDARY_MIN_RANGE = 0.5


class VisualizationView(QWidget):
    """Show the decision boundary and the architecture of the model."""

    def __init__(self, parent=None) -> None:
        """Create both chart panels and the refresh controls."""
        super().__init__(parent)
        self._network_provider: Callable[[], NeuralNetwork | None] = lambda: None
        self._dataset_provider: Callable[[], Dataset | None] = lambda: None

        self._page = install_page(
            self,
            "Visualization",
            "What the current model has learned: where it changes class "
            "and how its layers are connected. Both panels redraw from "
            "the model and dataset; nothing is computed or stored here.",
        )

        panels = QSplitter(Qt.Orientation.Horizontal, self._page)
        panels.setObjectName("visualizationSplitter")
        panels.setChildrenCollapsible(False)

        projection_card = SectionCard(
            "Decision boundary projection",
            description=(
                "A boundary is a two-dimensional picture of one output. "
                "Choose the two input features to show; every other "
                "feature is held at its column mean, which is stated "
                "below the selectors."
            ),
            parent=self._page,
        )
        selectors = QWidget(projection_card)
        selectors_layout = QHBoxLayout(selectors)
        selectors_layout.setContentsMargins(0, 0, 0, 0)
        selectors_layout.setSpacing(FIELD_SPACING)
        selectors_layout.addWidget(QLabel("Feature on x:", selectors))
        self._feature_x_combo = QComboBox(selectors)
        self._feature_x_combo.setObjectName("boundaryFeatureXCombo")
        self._feature_x_combo.setToolTip(
            "Which input feature the horizontal axis shows."
        )
        selectors_layout.addWidget(self._feature_x_combo, 1)
        selectors_layout.addWidget(QLabel("Feature on y:", selectors))
        self._feature_y_combo = QComboBox(selectors)
        self._feature_y_combo.setObjectName("boundaryFeatureYCombo")
        self._feature_y_combo.setToolTip(
            "Which input feature the vertical axis shows."
        )
        selectors_layout.addWidget(self._feature_y_combo, 1)
        projection_card.add(selectors)
        self._projection_note = MessageLabel("hint", projection_card)
        projection_card.add(self._projection_note)
        self._page.add(projection_card)

        self._output_combo = QComboBox(self._page)
        self._output_combo.setObjectName("boundaryOutputCombo")
        self._output_combo.setToolTip(
            "Which output is drawn: its class regions for a binary task, "
            "or its predicted values for a regression task."
        )
        output_row_widget = QWidget(self._page)
        output_layout = QHBoxLayout(output_row_widget)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(FIELD_SPACING)
        output_layout.addWidget(QLabel("Output neuron:", output_row_widget))
        output_layout.addWidget(self._output_combo, 1)
        self._output_selector = output_row_widget
        self._output_selector.setVisible(False)
        self._page.add(self._output_selector)

        self._boundary_group = SectionCard(
            "Decision boundary",
            description=(
                "Class regions for the selected projection of a binary "
                "model. The boundary is where that output reaches 0.5."
            ),
            parent=panels,
        )
        self._boundary_stack = QStackedWidget(self._boundary_group)
        self._boundary_stack.setObjectName("boundaryStack")
        self._boundary_chart = DecisionBoundaryChart(self._boundary_stack)
        self._boundary_stack.addWidget(self._boundary_chart)
        self._surface_chart = RegressionSurfaceChart(self._boundary_stack)
        self._boundary_stack.addWidget(self._surface_chart)
        self._boundary_group.add(self._boundary_stack, 1)
        panels.addWidget(self._boundary_group)

        architecture_group = SectionCard(
            "Network architecture",
            description=(
                "The active model's layers and connections, drawn from its "
                "real sizes. Use the toolbar to zoom, pan, and reset."
            ),
            parent=panels,
        )
        self._architecture_chart = ArchitectureChart(architecture_group)
        self._architecture_toolbar = create_navigation_toolbar(
            self._architecture_chart, architecture_group
        )
        self._architecture_toolbar.setObjectName("architectureToolbar")
        architecture_group.add(self._architecture_toolbar)
        architecture_group.add(self._architecture_chart, 1)
        panels.addWidget(architecture_group)
        panels.setStretchFactor(0, 3)
        panels.setStretchFactor(1, 2)
        self._page.add(panels, 1)

        self._refresh_button = make_button(
            "Refresh visualizations",
            kind="primary",
            tooltip="Redraw both panels from the current model and dataset.",
            parent=self._page,
        )
        self._refresh_button.setObjectName("refreshVisualizationsButton")
        self._status_label = MessageLabel("info", self._page)
        self._status_label.setText("No model or dataset selected.")
        self._page.add_layout(button_row(self._refresh_button))
        self._page.add(self._status_label)

        self._feedback = FeedbackArea(parent=self._page)
        self._error_label = self._feedback.error_label()
        self._page.add(self._feedback)

        self._refresh_button.clicked.connect(self.refresh)
        self._feature_x_combo.currentIndexChanged.connect(
            self._on_projection_changed
        )
        self._feature_y_combo.currentIndexChanged.connect(
            self._on_projection_changed
        )
        self._output_combo.currentIndexChanged.connect(
            self._on_projection_changed
        )

    def set_network_provider(
        self, provider: Callable[[], NeuralNetwork | None]
    ) -> None:
        """Set the callable supplying the model to visualize."""
        self._network_provider = provider

    def set_dataset_provider(
        self, provider: Callable[[], Dataset | None]
    ) -> None:
        """Set the callable supplying the dataset to visualize."""
        self._dataset_provider = provider

    def current_network(self) -> NeuralNetwork | None:
        """Return the supplied model, if any and of the right type."""
        try:
            network = self._network_provider()
        except Exception:
            return None
        return network if isinstance(network, NeuralNetwork) else None

    def current_dataset(self) -> Dataset | None:
        """Return the supplied dataset, if any and of the right type."""
        try:
            dataset = self._dataset_provider()
        except Exception:
            return None
        return dataset if isinstance(dataset, Dataset) else None

    def show_error(self, message: str) -> None:
        """Present an error inline."""
        self._error_label.setText(message)

    def clear_error(self) -> None:
        """Clear any inline error message."""
        self._error_label.setText("")

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        """Refresh both panels whenever the page becomes visible."""
        super().showEvent(event)
        self.refresh()

    def refresh(self) -> None:
        """Redraw both panels from the current model and dataset."""
        self.clear_error()
        network = self.current_network()
        dataset = self.current_dataset()
        self._refresh_architecture(network)
        self._sync_selectors(network, dataset)
        self._refresh_boundary(network, dataset)
        self._status_label.setText(self._describe_state(network, dataset))

    def _feature_label(self, index: int) -> str:
        """Return the dataset's name for one input feature."""
        return f"feature_{index}"

    def _sync_selectors(
        self, network: NeuralNetwork | None, dataset: Dataset | None
    ) -> None:
        """Fill the feature and output selectors from the current state."""
        n_features = 0 if dataset is None else dataset.n_features
        n_outputs = 0 if network is None else int(network.layer_sizes[-1])
        for combo, previous in (
            (self._feature_x_combo, self._feature_x_combo.currentIndex()),
            (self._feature_y_combo, self._feature_y_combo.currentIndex()),
        ):
            if combo.count() != n_features:
                combo.blockSignals(True)
                try:
                    combo.clear()
                    combo.addItems(
                        [self._feature_label(i) for i in range(n_features)]
                    )
                finally:
                    combo.blockSignals(False)
            if combo.count() and 0 <= previous < combo.count():
                if combo.currentIndex() != previous:
                    combo.blockSignals(True)
                    try:
                        combo.setCurrentIndex(previous)
                    finally:
                        combo.blockSignals(False)
        usable = n_features >= 2
        self._feature_x_combo.setEnabled(usable)
        self._feature_y_combo.setEnabled(usable)
        if usable and (
            self._feature_x_combo.currentIndex()
            == self._feature_y_combo.currentIndex()
        ):
            # A projection needs two different axes; default the second
            # one to another feature on first fill.
            self._feature_y_combo.blockSignals(True)
            try:
                self._feature_y_combo.setCurrentIndex(
                    1 if self._feature_x_combo.currentIndex() == 0 else 0
                )
            finally:
                self._feature_y_combo.blockSignals(False)
        if self._output_combo.count() != n_outputs:
            self._output_combo.blockSignals(True)
            try:
                self._output_combo.clear()
                self._output_combo.addItems(
                    [f"output {index}" for index in range(n_outputs)]
                )
            finally:
                self._output_combo.blockSignals(False)
        self._output_selector.setVisible(n_outputs > 1)
        if n_features < 2:
            self._projection_note.setText(
                "A decision boundary needs a dataset with at least two "
                "input features."
            )
        else:
            self._projection_note.setText("")

    def _on_projection_changed(self) -> None:
        """Keep the two chosen features distinct, then redraw."""
        x_index = self._feature_x_combo.currentIndex()
        y_index = self._feature_y_combo.currentIndex()
        if x_index == y_index and self._feature_y_combo.count() > 1:
            other = 0 if x_index != 0 else 1
            self._feature_y_combo.blockSignals(True)
            try:
                self._feature_y_combo.setCurrentIndex(other)
            finally:
                self._feature_y_combo.blockSignals(False)
        self._refresh_boundary(self.current_network(), self.current_dataset())

    def _projection(self) -> tuple[int, int, int] | None:
        """Return the selected ``(x feature, y feature, output)``."""
        x_index = self._feature_x_combo.currentIndex()
        y_index = self._feature_y_combo.currentIndex()
        output_index = self._output_combo.currentIndex()
        if x_index < 0 or y_index < 0 or x_index == y_index:
            return None
        if output_index < 0:
            output_index = 0
        return x_index, y_index, output_index

    def _describe_projection(
        self,
        dataset: Dataset,
        x_index: int,
        y_index: int,
        output_index: int,
        n_features: int,
    ) -> str:
        """Describe exactly what the projection panel is showing."""
        parts = [
            f"{self._feature_label(x_index)} (x) against "
            f"{self._feature_label(y_index)} (y)"
        ]
        if n_features > 2:
            held = [
                self._feature_label(index)
                for index in range(n_features)
                if index not in (x_index, y_index)
            ]
            parts.append("held at their column mean: " + ", ".join(held))
        if dataset.n_outputs is not None and dataset.n_outputs > 1:
            if dataset.is_binary_task:
                parts.append(f"class regions of output {output_index}")
            else:
                parts.append(f"predicted values of output {output_index}")
        return "; ".join(parts) + "."

    def _describe_state(
        self, network: NeuralNetwork | None, dataset: Dataset | None
    ) -> str:
        """Summarize what is currently visualized on one line."""
        if network is None:
            return "No model available yet. Train or configure one first."
        model_text = (
            f"Model [{', '.join(str(size) for size in network.layer_sizes)}]"
        )
        if dataset is None:
            return f"{model_text}; no dataset selected."
        target_text = "labeled" if dataset.has_targets else "unlabeled"
        return (
            f"{model_text}; dataset {dataset.n_samples} samples, "
            f"{dataset.n_features} features ({target_text})."
        )

    def _refresh_architecture(self, network: NeuralNetwork | None) -> None:
        """Draw the model's actual layers, or explain their absence."""
        if network is None:
            self._architecture_chart.show_message(
                "No model available yet. Train or configure one to see "
                "its layers and connections."
            )
            return
        self._architecture_chart.plot_architecture(tuple(network.layer_sizes))

    def _refresh_boundary(
        self, network: NeuralNetwork | None, dataset: Dataset | None
    ) -> None:
        """Draw the selected projection for the dataset's declared task.

        Binary targets get class regions and a decision boundary;
        continuous targets get a predicted-value surface instead, never a
        boundary labelled as if it were one.
        """
        binary = dataset is not None and dataset.is_binary_task
        self._show_region_panel(binary)
        chart = self._boundary_chart if binary else self._surface_chart
        if network is None or dataset is None:
            chart.show_message(
                "No dataset and model available yet. Validate a labeled "
                "dataset and train or configure a model first."
            )
            return
        sizes = tuple(network.layer_sizes)
        if sizes[0] != dataset.n_features:
            chart.show_message(
                f"This model reads {sizes[0]} feature(s) but the selected "
                f"dataset has {dataset.n_features}; retrain or load a "
                "matching model."
            )
            return
        if dataset.n_features < 2:
            chart.show_message(
                "This view needs at least two input features; the selected "
                f"dataset has {dataset.n_features}."
            )
            return
        if not dataset.has_targets:
            chart.show_message(
                "This view needs labeled samples; the selected dataset has "
                "no targets."
            )
            return
        projection = self._projection()
        if projection is None:
            chart.show_message(
                "Choose two different input features for this projection."
            )
            return
        x_index, y_index, output_index = projection
        if output_index >= sizes[-1]:
            chart.show_message(
                f"Output neuron {output_index} does not exist; this model "
                f"has {sizes[-1]} output(s)."
            )
            return
        try:
            features, targets = dataset.to_arrays()
            assert targets is not None  # narrowed by the labeled check
            projected_features = features[:, [x_index, y_index]]
            projected_targets = targets[:, output_index].reshape(-1, 1)
            self._projection_note.setText(
                self._describe_projection(
                    dataset, x_index, y_index, output_index,
                    dataset.n_features,
                )
            )
            grid_x, grid_y, region = self._boundary_grid(
                network, features, x_index, y_index, output_index
            )
        except (TypeError, ValueError) as exc:
            chart.show_message(f"Cannot plot this projection: {exc}")
            return
        if binary:
            drawn = self._boundary_chart.plot_boundary(
                projected_features, projected_targets, grid_x, grid_y, region
            )
            failure = "The decision boundary could not be drawn."
        else:
            drawn = self._surface_chart.plot_surface(
                projected_features,
                [float(value) for value in projected_targets[:, 0]],
                grid_x,
                grid_y,
                region,
                output_index=output_index,
                x_label=self._feature_label(x_index),
                y_label=self._feature_label(y_index),
            )
            failure = "The predicted-value surface could not be drawn."
        if not drawn:
            self.show_error(f"{failure} See the chart for the reason.")

    def _show_region_panel(self, binary: bool) -> None:
        """Show the boundary or surface panel and name the mode."""
        self._boundary_stack.setCurrentIndex(0 if binary else 1)
        self._boundary_group.setTitle(
            "Decision boundary" if binary else "Predicted value surface"
        )
        self._boundary_group.set_description(
            "Class regions for the selected projection of a binary model; "
            "the boundary is where that output reaches 0.5."
            if binary
            else "Predicted values for the selected output and two input "
            "features, with the samples' actual values for comparison."
        )

    def _boundary_grid(
        self,
        network: NeuralNetwork,
        features: np.ndarray,
        x_index: int,
        y_index: int,
        output_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample the model's output over the dataset's feature ranges.

        Builds a regular grid over the two selected feature columns (with
        a margin). Every other input feature is held at its column mean,
        which the page states next to the chart, so the picture is an
        explicit slice of the input space rather than a silently
        truncated model. The grid axes and the selected output's
        predictions, reshaped to match, are returned. Predictions come
        from the existing forward-pass API; nothing is computed here.

        Raises:
            ValueError: If the feature matrix has fewer than two columns,
                the indices are out of range, or the model does not
                produce the requested output.
        """
        columns = np.asarray(features, dtype=float)
        if columns.ndim != 2 or columns.shape[1] < 2:
            raise ValueError("the boundary grid needs at least two feature columns")
        n_features = columns.shape[1]
        for index in (x_index, y_index):
            if not 0 <= index < n_features:
                raise ValueError(f"feature index {index} is out of range")
        if x_index == y_index:
            raise ValueError("the boundary grid needs two different features")
        axes_values: list[np.ndarray] = []
        for column_index in (x_index, y_index):
            column = columns[:, column_index]
            low, high = float(np.min(column)), float(np.max(column))
            span = high - low
            margin = span * BOUNDARY_MARGIN_RATIO
            if span <= 0.0:
                margin = BOUNDARY_MIN_RANGE
            axes_values.append(
                np.linspace(
                    low - margin, high + margin, BOUNDARY_GRID_RESOLUTION
                )
            )
        grid_x, grid_y = np.meshgrid(axes_values[0], axes_values[1])
        points = np.tile(columns.mean(axis=0), (grid_x.size, 1))
        points[:, x_index] = grid_x.ravel()
        points[:, y_index] = grid_y.ravel()
        predictions = network.forward(points)
        if predictions.ndim != 2 or not 0 <= output_index < predictions.shape[1]:
            raise ValueError(
                "the model must produce the selected output for every grid "
                "point"
            )
        region = predictions[:, output_index].reshape(grid_x.shape)
        return axes_values[0], axes_values[1], region
