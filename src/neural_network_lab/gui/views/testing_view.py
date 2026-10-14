"""Testing view: prediction-only input and labeled-dataset evaluation.

Single samples entered by hand are prediction-only (no targets needed);
a supplied labeled dataset is evaluated read-only through the existing
:func:`evaluate` service, which reports MSE and binary accuracy. Both
paths use only forward propagation, so weights and biases are never
modified. All formulas stay in core/application code; this view parses
input text, presents results, and reports errors.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import QGroupBox
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QLineEdit
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QTableWidget
from PySide6.QtWidgets import QTableWidgetItem
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget

from neural_network_lab.application.evaluation_service import (
    CLASSIFICATION_THRESHOLD,
)
from neural_network_lab.application.evaluation_service import evaluate
from neural_network_lab.core.network import NeuralNetwork
from neural_network_lab.data.dataset import Dataset
from neural_network_lab.gui.main_window import CONTENT_MARGIN
from neural_network_lab.gui.main_window import CONTENT_SPACING
from neural_network_lab.gui.widgets.charts import PredictionChart
from neural_network_lab.gui.widgets.dataset_table import parse_float_cell


class TestingView(QWidget):
    """Predict single samples and evaluate labeled datasets read-only."""

    def __init__(self, parent=None) -> None:
        """Create the prediction and evaluation sections."""
        super().__init__(parent)
        self._network_provider: Callable[[], NeuralNetwork | None] = lambda: None
        self._dataset_provider: Callable[[], Dataset | None] = lambda: None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN
        )
        layout.setSpacing(CONTENT_SPACING)

        manual = QGroupBox("Single-sample prediction (no targets needed)", self)
        manual_layout = QVBoxLayout(manual)
        manual_layout.setSpacing(CONTENT_SPACING)
        self._feature_row = QHBoxLayout()
        self._feature_row.setSpacing(CONTENT_SPACING)
        manual_layout.addLayout(self._feature_row)
        self._feature_edits: list[QLineEdit] = []
        predict_row = QHBoxLayout()
        predict_row.setSpacing(CONTENT_SPACING)
        self._predict_button = QPushButton("Predict", self)
        self._predict_button.setObjectName("predictButton")
        predict_row.addWidget(self._predict_button)
        self._prediction_label = QLabel("No prediction yet.", self)
        self._prediction_label.setWordWrap(True)
        self._prediction_label.setObjectName("predictionResult")
        predict_row.addWidget(self._prediction_label, 1)
        manual_layout.addLayout(predict_row)
        layout.addWidget(manual)

        labeled = QGroupBox("Labeled dataset evaluation (read-only)", self)
        labeled_layout = QVBoxLayout(labeled)
        labeled_layout.setSpacing(CONTENT_SPACING)
        self._dataset_info = QLabel("Evaluation data: none selected.", self)
        self._dataset_info.setWordWrap(True)
        self._dataset_info.setObjectName("evaluationDatasetInfo")
        labeled_layout.addWidget(self._dataset_info)
        run_row = QHBoxLayout()
        run_row.setSpacing(CONTENT_SPACING)
        self._evaluate_button = QPushButton("Evaluate labeled dataset", self)
        self._evaluate_button.setObjectName("evaluateButton")
        run_row.addWidget(self._evaluate_button)
        self._metrics_label = QLabel("Metrics: unavailable.", self)
        self._metrics_label.setWordWrap(True)
        self._metrics_label.setObjectName("evaluationMetrics")
        run_row.addWidget(self._metrics_label, 1)
        labeled_layout.addLayout(run_row)
        self._results_table = QTableWidget(self)
        self._results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._results_table.setObjectName("evaluationResults")
        labeled_layout.addWidget(self._results_table, 1)
        self._prediction_chart = PredictionChart(self)
        labeled_layout.addWidget(self._prediction_chart, 1)
        layout.addWidget(labeled, 1)

        self._error_label = QLabel("", self)
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #b00020;")
        self._error_label.setObjectName("testingError")
        layout.addWidget(self._error_label)

        self._predict_button.clicked.connect(self._on_predict)
        self._evaluate_button.clicked.connect(self._on_evaluate)

    def set_network_provider(
        self, provider: Callable[[], NeuralNetwork | None]
    ) -> None:
        """Set the callable supplying the model to test."""
        self._network_provider = provider

    def set_dataset_provider(
        self, provider: Callable[[], Dataset | None]
    ) -> None:
        """Set the callable supplying the labeled evaluation dataset."""
        self._dataset_provider = provider

    def current_network(self) -> NeuralNetwork | None:
        """Return the supplied model, if any and of the right type."""
        try:
            network = self._network_provider()
        except Exception:
            return None
        return network if isinstance(network, NeuralNetwork) else None

    def show_error(self, message: str) -> None:
        """Present an error inline without clearing user input."""
        self._error_label.setText(message)

    def clear_error(self) -> None:
        """Clear any inline error message."""
        self._error_label.setText("")

    def _require_network(self) -> NeuralNetwork:
        """Return the model or raise a user-readable error."""
        network = self.current_network()
        if network is None:
            raise ValueError(
                "No model available. Configure and train one in "
                "the Training view first."
            )
        return network

    def _sync_feature_inputs(self, n_features: int) -> None:
        """Rebuild manual inputs to match, preserving fitting values."""
        previous = [edit.text() for edit in self._feature_edits]
        while self._feature_row.count():
            item = self._feature_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._feature_edits = []
        for index in range(n_features):
            edit = QLineEdit(self)
            edit.setPlaceholderText(f"x{index}")
            edit.setObjectName(f"featureInput{index}")
            if index < len(previous):
                edit.setText(previous[index])
            self._feature_row.addWidget(edit)
            self._feature_edits.append(edit)

    def _parse_manual_sample(self, n_features: int) -> list[float]:
        """Parse the manual feature inputs with located errors."""
        if len(self._feature_edits) != n_features:
            self._sync_feature_inputs(n_features)
        values = []
        for index, edit in enumerate(self._feature_edits):
            values.append(
                parse_float_cell(
                    edit.text(), row=0, column_name=f"feature_{index}"
                )
            )
        return values

    def _on_predict(self) -> None:
        """Predict one hand-entered sample without touching parameters."""
        self.clear_error()
        try:
            network = self._require_network()
        except ValueError as exc:
            self.show_error(str(exc))
            return
        try:
            sample = self._parse_manual_sample(network.layer_sizes[0])
            probabilities = network.forward([sample])
        except (TypeError, ValueError) as exc:
            self.show_error(f"Invalid input: {exc}")
            return
        probability = float(probabilities[0][0]) if probabilities.shape[1] == 1 else None
        if probability is None:
            shown = ", ".join(f"{float(v)!r}" for v in probabilities[0])
            self._prediction_label.setText(
                f"Predicted probabilities (multi-output model): [{shown}]. "
                f"Binary class uses threshold {CLASSIFICATION_THRESHOLD} per output."
            )
            return
        predicted = 1 if probability >= CLASSIFICATION_THRESHOLD else 0
        self._prediction_label.setText(
            f"Predicted probability {probability!r} → class {predicted} "
            f"(threshold {CLASSIFICATION_THRESHOLD}). Parameters unchanged."
        )

    def _on_evaluate(self) -> None:
        """Evaluate the supplied labeled dataset read-only."""
        self.clear_error()
        try:
            network = self._require_network()
        except ValueError as exc:
            self.show_error(str(exc))
            return
        try:
            dataset = self._dataset_provider()
        except Exception as exc:
            self.show_error(f"Could not read the selected dataset: {exc}")
            return
        if not isinstance(dataset, Dataset):
            self.show_error(
                "No evaluation dataset selected. Validate one in "
                "Dataset Management first."
            )
            return
        self._dataset_info.setText(
            f"Evaluation data: {dataset.n_samples} samples, "
            f"{dataset.n_features} features"
            f"{', labeled' if dataset.has_targets else ', prediction-only (no targets)'}."
        )
        if not dataset.has_targets:
            self._show_predictions_only(network, dataset)
            return
        try:
            result = evaluate(network, dataset)
        except (TypeError, ValueError) as exc:
            self.show_error(f"Evaluation failed: {exc}")
            return
        self._show_evaluation(result.predictions, result.targets)
        self._metrics_label.setText(
            f"Test MSE {result.mse!r} · accuracy {result.accuracy!r} "
            f"(threshold {CLASSIFICATION_THRESHOLD}, {result.n_samples} samples). "
            "Parameters unchanged."
        )

    def _show_predictions_only(self, network: NeuralNetwork, dataset: Dataset) -> None:
        """Show per-sample predictions where targets (and metrics) are absent."""
        try:
            features, _ = dataset.to_arrays()
            probabilities = network.forward(features)
        except (TypeError, ValueError) as exc:
            self.show_error(f"Evaluation failed: {exc}")
            return
        rows = [
            (
                f"{float(probabilities[i][0])!r}" if probabilities.shape[1] == 1
                else "[" + ", ".join(f"{float(v)!r}" for v in probabilities[i]) + "]",
                (
                    str(int(probabilities[i][0] >= CLASSIFICATION_THRESHOLD))
                    if probabilities.shape[1] == 1
                    else "—"
                ),
            )
            for i in range(probabilities.shape[0])
        ]
        self._fill_results_table(
            ["sample", "predicted probability", "predicted class"], rows
        )
        self._metrics_label.setText(
            "Metrics unavailable: prediction-only input has no targets."
        )
        self._prediction_chart.plot_predictions(probabilities, None)

    def _show_evaluation(self, predictions, targets) -> None:
        """Show per-sample probabilities, classes, actuals, and errors."""
        threshold = CLASSIFICATION_THRESHOLD
        rows = []
        for index in range(predictions.shape[0]):
            probabilities = [float(v) for v in predictions[index]]
            actuals = [float(v) for v in targets[index]]
            shown_prob = (
                f"{probabilities[0]!r}"
                if len(probabilities) == 1
                else "[" + ", ".join(f"{v!r}" for v in probabilities) + "]"
            )
            shown_class = (
                str(int(probabilities[0] >= threshold))
                if len(probabilities) == 1
                else "[" + ", ".join(str(int(v >= threshold)) for v in probabilities) + "]"
            )
            shown_actual = (
                f"{actuals[0]!r}"
                if len(actuals) == 1
                else "[" + ", ".join(f"{v!r}" for v in actuals) + "]"
            )
            shown_error = (
                f"{(actuals[0] - probabilities[0])!r}"
                if len(actuals) == 1
                else "["
                + ", ".join(f"{(a - p)!r}" for a, p in zip(actuals, probabilities))
                + "]"
            )
            rows.append((shown_prob, shown_class, shown_actual, shown_error))
        self._fill_results_table(
            ["sample", "predicted probability", "predicted class", "actual", "error"],
            rows,
        )
        self._prediction_chart.plot_predictions(predictions, targets)

    def _fill_results_table(
        self, headers: list[str], rows: list[tuple[str, ...]]
    ) -> None:
        """Replace the per-sample results table contents."""
        table = self._results_table
        table.clear()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            table.setItem(row_index, 0, QTableWidgetItem(str(row_index + 1)))
            for col_index, text in enumerate(row, start=1):
                table.setItem(row_index, col_index, QTableWidgetItem(text))
        table.resizeColumnsToContents()
