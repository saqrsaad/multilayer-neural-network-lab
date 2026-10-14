"""GUI tests for training and prediction charts (offscreen)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

import numpy as np

from neural_network_lab.core.network import NeuralNetwork
from neural_network_lab.data.dataset import Dataset
from neural_network_lab.gui.main_window import create_application
from neural_network_lab.gui.views.testing_view import TestingView
from neural_network_lab.gui.widgets.charts import LossHistoryChart
from neural_network_lab.gui.widgets.charts import PredictionChart
from neural_network_lab.gui.widgets.charts import validate_loss_history
from neural_network_lab.gui.widgets.charts import validate_prediction_inputs
from neural_network_lab.gui.widgets.training_controls import TrainingControls


def test_validate_loss_history():
    assert validate_loss_history([0.5, 0.25]) == [0.5, 0.25]
    assert validate_loss_history((1,)) == [1.0]
    assert validate_loss_history(np.array([0.5])) == [0.5]
    with pytest.raises(ValueError, match="empty"):
        validate_loss_history([])
    with pytest.raises(ValueError, match="not finite"):
        validate_loss_history([0.5, float("inf")])
    with pytest.raises(TypeError, match="not a number"):
        validate_loss_history([0.5, "x"])
    with pytest.raises(TypeError, match="not a number"):
        validate_loss_history([True])
    with pytest.raises(TypeError, match="sequence"):
        validate_loss_history(None)


def test_validate_prediction_inputs():
    probs, actual = validate_prediction_inputs([0.2, 0.8], [0.0, 1.0])
    assert probs == [0.2, 0.8]
    assert actual == [0.0, 1.0]
    probs, actual = validate_prediction_inputs(np.array([[0.2], [0.8]]), None)
    assert probs == [0.2, 0.8]
    assert actual is None
    with pytest.raises(ValueError, match="does not match"):
        validate_prediction_inputs([0.2, 0.8], [1.0])
    with pytest.raises(ValueError, match="single-output"):
        validate_prediction_inputs(np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="empty"):
        validate_prediction_inputs([], [])
    with pytest.raises(ValueError, match="not finite"):
        validate_prediction_inputs([0.2, float("nan")], [0.0, 1.0])
    with pytest.raises(TypeError, match="numeric"):
        validate_prediction_inputs(["a"], ["b"])


def test_loss_chart_plots_recorded_epochs():
    create_application()
    chart = LossHistoryChart()
    assert len(chart._axes.lines) == 0  # empty-state message only
    assert chart.plot_history([0.5, 0.25, 0.125]) is True
    assert len(chart._axes.lines) == 1
    assert chart._axes.lines[0].get_xdata().tolist() == [1, 2, 3]
    assert chart._axes.lines[0].get_ydata().tolist() == [0.5, 0.25, 0.125]
    assert chart._axes.get_title() == "Training loss (MSE) by epoch"
    assert chart._axes.get_xlabel() == "epoch"
    assert chart._axes.get_ylabel() == "training loss (MSE)"
    legends = [t.get_text() for t in chart._axes.get_legend().get_texts()]
    assert legends == ["training MSE"]


def test_loss_chart_replaces_stale_content():
    create_application()
    chart = LossHistoryChart()
    assert chart.plot_history([0.5, 0.25]) is True
    assert chart.plot_history([0.9]) is True
    assert len(chart._axes.lines) == 1
    assert chart._axes.lines[0].get_xdata().tolist() == [1]
    assert chart.plot_history([]) is False
    assert len(chart._axes.lines) == 0  # message shown, no stale line


def test_prediction_chart_distinguishes_series():
    create_application()
    chart = PredictionChart()
    assert chart.plot_predictions([0.2, 0.8], [0.0, 1.0]) is True
    assert len(chart._axes.lines) == 3
    assert chart._axes.lines[0].get_ydata().tolist() == [0.2, 0.8]
    assert chart._axes.lines[1].get_ydata().tolist() == [0.0, 1.0]
    assert chart._axes.lines[2].get_ydata().tolist() == [0.0, 1.0]
    legends = [t.get_text() for t in chart._axes.get_legend().get_texts()]
    assert legends[0] == "predicted probability"
    assert "threshold 0.5" in legends[1]
    assert legends[2] == "actual target"
    assert chart._axes.get_xlabel() == "sample index"


def test_prediction_chart_without_targets():
    create_application()
    chart = PredictionChart()
    assert chart.plot_predictions([0.2, 0.8], None) is True
    assert len(chart._axes.lines) == 2
    assert "no targets" in chart._axes.get_title()


def test_prediction_chart_rejects_bad_data_safely():
    create_application()
    chart = PredictionChart()
    assert chart.plot_predictions([0.2, 0.8], [1.0]) is False
    assert len(chart._axes.lines) == 0
    assert chart.plot_predictions(np.zeros((2, 2))) is False
    assert len(chart._axes.lines) == 0


def test_training_controls_refresh_chart_from_history():
    create_application()
    controls = TrainingControls()
    assert len(controls._loss_chart._axes.lines) == 0
    controls.display_history(
        mode="continued", epochs=2, final_loss=0.25, history=(0.5, 0.25)
    )
    assert controls._loss_chart._axes.lines[0].get_ydata().tolist() == [0.5, 0.25]
    controls.display_history(
        mode="continued", epochs=1, final_loss=0.1, history=(0.1,)
    )
    assert len(controls._loss_chart._axes.lines) == 1
    assert controls._loss_chart._axes.lines[0].get_xdata().tolist() == [1]


def test_testing_view_refreshes_chart_on_evaluation():
    create_application()
    network = NeuralNetwork(
        [1, 1], weights=[np.array([[0.0]])], biases=[np.array([[0.0]])]
    )
    dataset = Dataset([[0.0], [1.0]], [[1.0], [0.0]])
    view = TestingView()
    view.set_network_provider(lambda: network)
    view.set_dataset_provider(lambda: dataset)
    view._evaluate_button.click()
    assert view._error_label.text() == ""
    assert len(view._prediction_chart._axes.lines) == 3
    assert view._prediction_chart._axes.lines[0].get_ydata().tolist() == [0.5, 0.5]
    assert view._prediction_chart._axes.lines[2].get_ydata().tolist() == [1.0, 0.0]


def test_testing_view_prediction_only_plots_without_targets():
    create_application()
    network = NeuralNetwork(
        [1, 1], weights=[np.array([[0.0]])], biases=[np.array([[0.0]])]
    )
    view = TestingView()
    view.set_network_provider(lambda: network)
    view.set_dataset_provider(lambda: Dataset([[0.0], [1.0]]))
    view._evaluate_button.click()
    assert view._error_label.text() == ""
    assert len(view._prediction_chart._axes.lines) == 2
