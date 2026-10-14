"""GUI tests for network configuration and training controls (offscreen)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtTest import QSignalSpy

from neural_network_lab.data.dataset import Dataset
from neural_network_lab.gui.main_window import create_application
from neural_network_lab.gui.views.training_view import TrainingView
from neural_network_lab.gui.widgets.network_config_widget import parse_hidden_sizes
from neural_network_lab.gui.widgets.network_config_widget import parse_learning_rate
from neural_network_lab.gui.widgets.network_config_widget import NetworkConfigWidget


def _view(dataset=None) -> TrainingView:
    create_application()
    view = TrainingView()
    view.set_dataset_provider(lambda: dataset)
    return view


def _labeled_dataset():
    return Dataset(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        [[0.0], [1.0], [1.0], [0.0]],
    )


def test_parse_hidden_sizes():
    assert parse_hidden_sizes("4, 3") == (4, 3)
    assert parse_hidden_sizes("  ") == ()
    assert parse_hidden_sizes("5") == (5,)
    with pytest.raises(ValueError, match="not an integer"):
        parse_hidden_sizes("4, x")
    with pytest.raises(ValueError, match="positive integer"):
        parse_hidden_sizes("4, 0")


def test_parse_learning_rate():
    assert parse_learning_rate("0.1") == pytest.approx(0.1)
    with pytest.raises(ValueError, match="empty"):
        parse_learning_rate("   ")
    with pytest.raises(ValueError, match="not a number"):
        parse_learning_rate("abc")
    with pytest.raises(ValueError, match="finite"):
        parse_learning_rate("inf")
    with pytest.raises(ValueError, match="greater than zero"):
        parse_learning_rate("-0.5")


def test_layer_sizes_need_dataset_input_size():
    create_application()
    widget = NetworkConfigWidget()
    with pytest.raises(ValueError, match="No dataset"):
        widget.layer_sizes()
    widget.set_input_size(2)
    assert widget.layer_sizes() == (2, 4, 1)


def test_train_from_scratch_runs_and_reports():
    view = _view(_labeled_dataset())
    view._config._epochs_spin.setValue(2)
    spy = QSignalSpy(view.training_finished)
    view._controls._scratch_button.click()
    assert spy.wait(10000)
    network = view.current_network()
    assert network is not None
    assert tuple(network.layer_sizes) == (2, 4, 1)
    assert "from scratch" in view._controls._status_label.text()
    assert view._controls._error_label.text() == ""
    history_lines = view._controls._history_view.toPlainText().strip().splitlines()
    assert len(history_lines) == 2


def test_continue_without_model_reports_error():
    view = _view(_labeled_dataset())
    view._controls._continued_button.click()
    assert view.current_network() is None
    assert "No model yet" in view._controls._error_label.text()


def test_continue_training_preserves_starting_point():
    view = _view(_labeled_dataset())
    view._config._epochs_spin.setValue(1)
    spy = QSignalSpy(view.training_finished)
    view._controls._scratch_button.click()
    assert spy.wait(10000)
    before = [w.copy() for w in view.current_network().weights]
    continued_spy = QSignalSpy(view.training_finished)
    view._controls._continued_button.click()
    assert continued_spy.wait(10000)
    assert "continued" in view._controls._status_label.text()
    after = view.current_network().weights
    assert any(not (a == b).all() for a, b in zip(before, after))


def test_architecture_change_requires_explicit_rebuild():
    view = _view(_labeled_dataset())
    view._config._epochs_spin.setValue(1)
    view._controls._scratch_button.click()
    view._config._hidden_edit.setText("5")
    view._controls._continued_button.click()
    assert "Train from scratch" in view._controls._error_label.text()


def test_invalid_settings_rejected_before_training():
    view = _view(_labeled_dataset())
    view._config._learning_edit.setText("abc")
    view._controls._scratch_button.click()
    assert view.current_network() is None
    assert "Invalid settings" in view._controls._error_label.text()


def test_unlabeled_and_missing_datasets_rejected():
    unlabeled = Dataset([[0.0, 0.0], [1.0, 1.0]])
    view = _view(unlabeled)
    view._controls._scratch_button.click()
    assert view.current_network() is None
    assert "labeled" in view._controls._error_label.text()

    empty_view = _view(None)
    empty_view._controls._scratch_button.click()
    assert empty_view.current_network() is None
    assert "No training dataset" in empty_view._controls._error_label.text()


def test_incompatible_output_size_rejected():
    view = _view(_labeled_dataset())
    view._config._output_spin.setValue(2)
    view._controls._scratch_button.click()
    assert view.current_network() is None
    assert "Incompatible" in view._controls._error_label.text()
