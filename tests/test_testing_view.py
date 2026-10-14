"""GUI tests for the testing view (offscreen, read-only)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

import numpy as np
from PySide6.QtWidgets import QTableWidget
from PySide6.QtWidgets import QTableWidgetItem

from neural_network_lab.core.network import NeuralNetwork
from neural_network_lab.data.dataset import Dataset
from neural_network_lab.gui.main_window import create_application
from neural_network_lab.gui.views.testing_view import TestingView


def _half_network() -> NeuralNetwork:
    """Single-neuron zero network: every prediction is exactly 0.5."""
    return NeuralNetwork(
        [1, 1],
        weights=[np.array([[0.0]])],
        biases=[np.array([[0.0]])],
    )


def _view(network=None, dataset=None) -> TestingView:
    create_application()
    view = TestingView()
    view.set_network_provider(lambda: network)
    view.set_dataset_provider(lambda: dataset)
    return view


def _snapshot(network: NeuralNetwork):
    return (
        [w.copy() for w in network.weights],
        [b.copy() for b in network.biases],
    )


def test_missing_model_reported():
    view = _view()
    view._predict_button.click()
    assert "No model" in view._error_label.text()
    view.clear_error()
    view._evaluate_button.click()
    assert "No model" in view._error_label.text()


def test_single_sample_prediction_shows_probability_and_class():
    view = _view(network=_half_network())
    view._predict_button.click()  # syncs one input row, empty -> invalid
    assert "Invalid input" in view._error_label.text()
    view._feature_edits[0].setText("0.0")
    view._predict_button.click()
    assert view._error_label.text() == ""
    assert "0.5" in view._prediction_label.text()
    assert "class 1" in view._prediction_label.text()


def test_invalid_manual_input_keeps_text():
    view = _view(network=_half_network())
    view._sync_feature_inputs(1)
    view._feature_edits[0].setText("oops")
    view._predict_button.click()
    assert "Invalid input" in view._error_label.text()
    assert view._feature_edits[0].text() == "oops"


def test_labeled_evaluation_shows_metrics_and_rows():
    network = _half_network()
    before = _snapshot(network)
    dataset = Dataset([[0.0], [1.0]], [[1.0], [0.0]])
    view = _view(network=network, dataset=dataset)
    view._evaluate_button.click()
    assert view._error_label.text() == ""
    assert "Test MSE" in view._metrics_label.text()
    assert "accuracy" in view._metrics_label.text()
    assert view._results_table.rowCount() == 2
    assert view._results_table.columnCount() == 5
    after = _snapshot(network)
    for old, new in zip(before[0], after[0]):
        np.testing.assert_array_equal(old, new)
    for old, new in zip(before[1], after[1]):
        np.testing.assert_array_equal(old, new)


def test_prediction_only_input_has_no_metrics():
    network = _half_network()
    dataset = Dataset([[0.0], [1.0]])
    view = _view(network=network, dataset=dataset)
    view._evaluate_button.click()
    assert view._error_label.text() == ""
    assert "Metrics unavailable" in view._metrics_label.text()
    assert "no targets" in view._metrics_label.text()
    assert view._results_table.rowCount() == 2
    assert view._results_table.columnCount() == 3


def test_incompatible_and_unlabeled_problems_reported():
    view = _view(
        network=_half_network(),
        dataset=Dataset([[0.0, 0.0, 0.0]], [[1.0]]),
    )
    view._evaluate_button.click()
    assert "Evaluation failed" in view._error_label.text()

    view2 = _view(network=_half_network(), dataset=None)
    view2._evaluate_button.click()
    assert "No evaluation dataset" in view2._error_label.text()

    view3 = _view(
        network=_half_network(), dataset=Dataset([[0.0]], [[0.5]])
    )
    view3._evaluate_button.click()
    assert "Evaluation failed" in view3._error_label.text()


def test_predict_never_modifies_parameters():
    network = _half_network()
    before = _snapshot(network)
    view = _view(network=network)
    view._sync_feature_inputs(1)
    view._feature_edits[0].setText("1.0")
    view._predict_button.click()
    after = _snapshot(network)
    for old, new in zip(before[0], after[0]):
        np.testing.assert_array_equal(old, new)
    for old, new in zip(before[1], after[1]):
        np.testing.assert_array_equal(old, new)


def test_results_table_is_read_only():
    view = _view(
        network=_half_network(), dataset=Dataset([[0.0]], [[1.0]])
    )
    assert (
        view._results_table.editTriggers()
        == QTableWidget.EditTrigger.NoEditTriggers
    )
    view._evaluate_button.click()
    item = view._results_table.item(0, 1)
    assert isinstance(item, QTableWidgetItem)
    assert view._results_table.rowCount() == 1
