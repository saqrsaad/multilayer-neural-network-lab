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

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QComboBox
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QLineEdit
from PySide6.QtWidgets import QSplitter
from PySide6.QtWidgets import QTableWidget
from PySide6.QtWidgets import QTableWidgetItem
from PySide6.QtWidgets import QWidget

from neural_network_lab.application.evaluation_service import (
    CLASSIFICATION_THRESHOLD,
)
from neural_network_lab.application.evaluation_service import evaluate
from neural_network_lab.core.network import NeuralNetwork
from neural_network_lab.data.dataset import Dataset
from neural_network_lab.data.dataset import TARGET_KIND_LABELS
from neural_network_lab.gui.widgets.charts import PredictionChart
from neural_network_lab.gui.widgets.dataset_table import parse_float_cell
from neural_network_lab.gui.widgets.page_scaffold import FeedbackArea
from neural_network_lab.gui.widgets.page_scaffold import MessageLabel
from neural_network_lab.gui.widgets.page_scaffold import SectionCard
from neural_network_lab.gui.widgets.page_scaffold import button_row
from neural_network_lab.gui.widgets.page_scaffold import install_page
from neural_network_lab.gui.widgets.page_scaffold import make_button


class TestingView(QWidget):
    """Predict single samples and evaluate labeled datasets read-only."""

    def __init__(self, parent=None) -> None:
        """Create the prediction and evaluation sections."""
        super().__init__(parent)
        self._network_provider: Callable[[], NeuralNetwork | None] = lambda: None
        self._dataset_provider: Callable[[], Dataset | None] = lambda: None
        self._model_revision_provider: Callable[[], int] = lambda: 0
        self._evaluated_network: NeuralNetwork | None = None
        self._evaluated_dataset: Dataset | None = None
        self._evaluated_revision: int | None = None
        self._last_predictions: np.ndarray | None = None
        self._last_targets: np.ndarray | None = None
        self._last_binary = True
        self._metrics_base_text = "Metrics: unavailable."

        self._page = install_page(
            self,
            "Testing",
            "Predict a single hand-entered sample, or evaluate the "
            "selected dataset. Testing only reads the model: weights and "
            "biases are never changed here.",
        )

        manual = SectionCard(
            "Single-sample prediction",
            description=(
                "One value per input feature. No targets are needed: this "
                "only runs a forward pass."
            ),
            parent=self._page,
        )
        self._feature_row = button_row(trailing_stretch=False)
        self._feature_edits: list[QLineEdit] = []
        self._predict_button = make_button(
            "Predict",
            kind="primary",
            tooltip="Run one forward pass for the values above.",
            parent=manual,
        )
        self._predict_button.setObjectName("predictButton")
        self._prediction_label = MessageLabel("status", manual)
        self._prediction_label.setText("No prediction yet.")
        manual.add_layout(self._feature_row)
        manual.add_layout(button_row(self._predict_button))
        manual.add(self._prediction_label)
        self._page.add(manual)

        labeled = SectionCard(
            "Labeled dataset evaluation",
            description=(
                "Scores the dataset validated in Dataset Management "
                "against the current model and reports per-sample results "
                "and metrics side by side."
            ),
            parent=self._page,
        )
        self._dataset_info = MessageLabel("info", labeled)
        self._dataset_info.setText("Evaluation data: none selected.")
        labeled.add(self._dataset_info)
        self._metrics_label = MessageLabel("status", labeled)
        self._metrics_label.setText("Metrics: unavailable.")
        labeled.add(self._metrics_label)
        self._evaluate_button = make_button(
            "Evaluate labeled dataset",
            kind="primary",
            tooltip="Evaluate the selected dataset without changing the "
            "model.",
            parent=labeled,
        )
        self._evaluate_button.setObjectName("evaluateButton")
        self._output_combo = QComboBox(labeled)
        self._output_combo.setObjectName("evaluationOutputCombo")
        self._output_combo.setToolTip(
            "Which output the chart draws. The table always lists every "
            "output."
        )
        self._output_combo.setVisible(False)
        labeled.add_layout(
            button_row(
                self._evaluate_button,
                QLabel("Chart output:", labeled),
                self._output_combo,
            )
        )

        results_splitter = QSplitter(Qt.Orientation.Horizontal, labeled)
        results_splitter.setObjectName("evaluationResultsSplitter")
        results_splitter.setChildrenCollapsible(False)
        self._results_table = QTableWidget(results_splitter)
        self._results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._results_table.setObjectName("evaluationResults")
        self._results_table.setAlternatingRowColors(True)
        self._results_table.setMinimumHeight(220)
        results_splitter.addWidget(self._results_table)
        self._prediction_chart = PredictionChart(results_splitter)
        results_splitter.addWidget(self._prediction_chart)
        results_splitter.setStretchFactor(0, 3)
        results_splitter.setStretchFactor(1, 2)
        labeled.add(results_splitter, 1)
        self._page.add(labeled, 1)

        self._feedback = FeedbackArea(parent=self._page)
        self._error_label = self._feedback.error_label()
        self._page.add(self._feedback)

        self._predict_button.clicked.connect(self._on_predict)
        self._evaluate_button.clicked.connect(self._on_evaluate)
        self._output_combo.currentIndexChanged.connect(
            self._on_output_selected
        )

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

    def set_model_revision_provider(
        self, provider: Callable[[], int]
    ) -> None:
        """Set the callable reporting the model's revision counter.

        The counter changes whenever the shared model's parameters are
        replaced, so displayed results can be recognised as outdated.
        """
        self._model_revision_provider = provider

    def _current_revision(self) -> int:
        """Return the model's revision counter, or 0 when unavailable."""
        if self._model_revision_provider is None:
            return 0
        try:
            return int(self._model_revision_provider())
        except Exception:
            return 0

    def _current_dataset(self) -> Dataset | None:
        """Return the supplied dataset, if any and of the right type."""
        try:
            dataset = self._dataset_provider()
        except Exception:
            return None
        return dataset if isinstance(dataset, Dataset) else None

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        """Clear results that no longer match the current model or data."""
        super().showEvent(event)
        if self._evaluated_network is None:
            return
        changed = self.current_network() is not self._evaluated_network
        if not changed:
            changed = self._current_revision() != self._evaluated_revision
        if not changed:
            changed = self._current_dataset() is not self._evaluated_dataset
        if changed:
            self._clear_evaluation_results()
            self.show_error(
                "The model or dataset changed since this evaluation, so its "
                "results were cleared. Evaluate again for current numbers."
            )

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

    def _clear_evaluation_results(self) -> None:
        """Reset the evaluation outputs to their empty states.

        Called whenever an evaluation cannot be produced, so the
        dataset line, the metrics label, the per-sample table, and the
        prediction chart never keep showing numbers from an earlier,
        now-invalid evaluation.
        """
        self._dataset_info.setText("Evaluation data: none selected.")
        self._metrics_label.setText("Metrics: unavailable.")
        self._metrics_base_text = "Metrics: unavailable."
        self._last_predictions = None
        self._last_targets = None
        self._evaluated_network = None
        self._evaluated_dataset = None
        self._evaluated_revision = None
        table = self._results_table
        table.clear()
        table.setRowCount(0)
        table.setColumnCount(0)
        self._output_combo.setVisible(False)
        self._prediction_chart.show_message(
            "No evaluation yet. Evaluate a labeled dataset to compare "
            "predictions against targets."
        )

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
            self._clear_evaluation_results()
            self.show_error(str(exc))
            return
        try:
            dataset = self._dataset_provider()
        except Exception as exc:
            self._clear_evaluation_results()
            self.show_error(f"Could not read the selected dataset: {exc}")
            return
        if not isinstance(dataset, Dataset):
            self._clear_evaluation_results()
            self.show_error(
                "No evaluation dataset selected. Validate one in "
                "Dataset Management first."
            )
            return
        task_text = (
            ""
            if not dataset.has_targets
            else ", " + TARGET_KIND_LABELS.get(dataset.target_kind, "")
        )
        self._dataset_info.setText(
            f"Evaluation data: {dataset.n_samples} samples, "
            f"{dataset.n_features} features"
            f"{', labeled' if dataset.has_targets else ', prediction-only (no targets)'}"
            f"{task_text}."
        )
        if not dataset.has_targets:
            self._show_predictions_only(network, dataset)
            return
        try:
            result = evaluate(network, dataset)
        except (TypeError, ValueError) as exc:
            self._clear_evaluation_results()
            self.show_error(f"Evaluation failed: {exc}")
            return
        # One successful evaluation refreshes every output together:
        # metrics, per-sample table, and the prediction chart all come
        # from the same EvaluationResult.
        self._metrics_label.setText(self._format_metrics(result))
        self._remember_result(
            result.predictions,
            result.targets,
            binary=result.accuracy is not None,
            metrics_text=self._metrics_label.text(),
        )
        self._show_evaluation(
            result.predictions, result.targets, binary=result.accuracy is not None
        )
        self._record_evaluated(network, dataset)

    @staticmethod
    def _format_metrics(result) -> str:
        """Describe the metrics that apply to the dataset's declared task.

        Binary datasets report threshold accuracy; continuous datasets
        report MSE, MAE, and RMSE over every output and are never
        thresholded into classes.
        """
        measures = (
            f"Test MSE {result.mse!r} · MAE {result.mae!r} · "
            f"RMSE {result.rmse!r} ({result.n_outputs} output"
            f"{'' if result.n_outputs == 1 else 's'}, "
            f"{result.n_samples} samples)"
        )
        if result.accuracy is None:
            return (
                f"{result.task_label}: {measures}. Parameters unchanged."
            )
        return (
            f"{result.task_label}: {measures} · accuracy "
            f"{result.accuracy!r} (threshold {CLASSIFICATION_THRESHOLD}). "
            "Parameters unchanged."
        )

    def _record_evaluated(
        self, network: NeuralNetwork, dataset: Dataset
    ) -> None:
        """Remember which model and data the displayed results describe."""
        self._evaluated_network = network
        self._evaluated_dataset = dataset
        self._evaluated_revision = self._current_revision()

    def _remember_result(
        self, predictions, targets, *, binary: bool, metrics_text: str
    ) -> None:
        """Keep the evaluated arrays so the output selector can redraw."""
        self._last_predictions = (
            None if predictions is None else np.asarray(predictions, dtype=float)
        )
        self._last_targets = (
            None if targets is None else np.asarray(targets, dtype=float)
        )
        self._last_binary = binary
        self._metrics_base_text = metrics_text

    def _show_predictions_only(self, network: NeuralNetwork, dataset: Dataset) -> None:
        """Show per-sample predictions where targets (and metrics) are absent."""
        try:
            features, _ = dataset.to_arrays()
            probabilities = network.forward(features)
        except (TypeError, ValueError) as exc:
            self._clear_evaluation_results()
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
        if dataset.is_binary_task:
            headers = ["sample", "predicted probability", "predicted class"]
        else:
            headers = ["sample", "predicted value"]
            rows = [(probability,) for probability, _ in rows]
        self._fill_results_table(headers, rows)
        self._metrics_label.setText(
            "Metrics unavailable: prediction-only input has no targets."
        )
        self._remember_result(
            probabilities,
            None,
            binary=dataset.is_binary_task,
            metrics_text=self._metrics_label.text(),
        )
        self._plot_or_explain(
            probabilities, None, classify=dataset.is_binary_task
        )
        self._record_evaluated(network, dataset)

    def _show_evaluation(self, predictions, targets, *, binary: bool) -> None:
        """Show per-sample predictions, actuals, errors, and classes.

        The predicted-class column only exists for a binary task; a
        continuous task has no class to report and its values are shown
        exactly as produced by the forward pass.
        """
        threshold = CLASSIFICATION_THRESHOLD
        headers = ["sample", "predicted value", "actual value", "error"]
        if binary:
            headers.insert(2, "predicted class")
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
            if binary:
                rows.append((shown_prob, shown_class, shown_actual, shown_error))
            else:
                rows.append((shown_prob, shown_actual, shown_error))
        self._fill_results_table(headers, rows)
        self._plot_or_explain(predictions, targets, classify=binary)

    def _plot_or_explain(self, predictions, targets, *, classify: bool) -> None:
        """Plot the selected output, or explain why the chart is skipped.

        A multi-output model is never truncated to its first column: the
        chart draws the output the selector points at (the selector is
        shown only when there is a choice), while the table above keeps
        every output. Continuous targets are never thresholded, so no
        predicted-class line is drawn for them.
        """
        if predictions is None:
            self._prediction_chart.show_message(
                "No evaluation yet. Evaluate a labeled dataset to compare "
                "predictions against targets."
            )
            return
        array = np.asarray(predictions, dtype=float)
        n_outputs = int(array.shape[1]) if array.ndim == 2 else 1
        self._sync_output_selector(n_outputs)
        index = self._selected_output_index(n_outputs)
        if index is None:
            self._prediction_chart.show_message(
                f"Select which of the {n_outputs} outputs to draw in the "
                "chart; the table lists them all."
            )
            return
        selected_predictions = array[:, index]
        selected_targets = None
        if targets is not None:
            target_array = np.asarray(targets, dtype=float)
            if target_array.ndim == 2 and index < target_array.shape[1]:
                selected_targets = target_array[:, index]
        drawn = self._prediction_chart.plot_predictions(
            selected_predictions, selected_targets, classify=classify
        )
        if not drawn and n_outputs <= 1:
            # Nothing more to explain: the chart already says why.
            return
        self._append_chart_note(
            n_outputs, index, classify, selected_targets is not None
        )

    def _sync_output_selector(self, n_outputs: int) -> None:
        """Show and fill the output selector when there is a choice."""
        visible = n_outputs > 1
        if visible and self._output_combo.count() != n_outputs:
            previous = self._output_combo.currentIndex()
            self._output_combo.blockSignals(True)
            try:
                self._output_combo.clear()
                self._output_combo.addItems(
                    [f"output {index}" for index in range(n_outputs)]
                )
                if 0 <= previous < n_outputs:
                    self._output_combo.setCurrentIndex(previous)
            finally:
                self._output_combo.blockSignals(False)
        self._output_combo.setVisible(visible)

    def _selected_output_index(self, n_outputs: int) -> int | None:
        """Return the output the chart should draw, or ``None``.

        With a single output there is nothing to choose, so the only
        output is drawn regardless of what the (hidden) selector holds.
        """
        if n_outputs <= 1:
            return 0
        index = self._output_combo.currentIndex()
        return index if 0 <= index < n_outputs else None

    def _append_chart_note(
        self, n_outputs: int, index: int, classify: bool, has_targets: bool
    ) -> None:
        """Say which output the chart shows and what it can show."""
        notes: list[str] = []
        if n_outputs > 1:
            notes.append(
                f"chart: output {index} of {n_outputs} (the table lists all "
                "outputs)"
            )
        if not has_targets:
            notes.append("chart: no actual values to compare against")
        if not classify:
            notes.append("chart: continuous targets, no thresholding applied")
        if notes:
            self._metrics_label.setText(
                f"{self._metrics_label.text()} {'; '.join(notes)}."
            )

    def _on_output_selected(self) -> None:
        """Redraw the chart for the newly selected output."""
        predictions = self._last_predictions
        if predictions is None:
            return
        self._metrics_label.setText(self._metrics_base_text)
        self._plot_or_explain(
            predictions, self._last_targets, classify=self._last_binary
        )

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
