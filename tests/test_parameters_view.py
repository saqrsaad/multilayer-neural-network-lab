"""GUI tests for the weights and biases editor (offscreen)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

import numpy as np
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QMessageBox

from neural_network_lab.core.network import NeuralNetwork
from neural_network_lab.gui.main_window import create_application
from neural_network_lab.gui.main_window import MainWindow
from neural_network_lab.gui.views.parameters_view import format_parameter
from neural_network_lab.gui.views.parameters_view import ParametersView
from neural_network_lab.gui.views.parameters_view import parse_parameter_cell


def _explicit_network() -> NeuralNetwork:
    """Known [2, 2, 1] network with hand-picked parameters."""
    return NeuralNetwork(
        [2, 2, 1],
        weights=[np.array([[0.5, -0.5], [0.25, 0.75]]), np.array([[1.0], [-1.0]])],
        biases=[np.array([[0.0, 0.1]]), np.array([[0.25]])],
    )


def _view(network=None) -> ParametersView:
    create_application()
    view = ParametersView()
    view.set_network_provider(lambda: network)
    return view


def _snapshot(network: NeuralNetwork):
    return ([w.copy() for w in network.weights], [b.copy() for b in network.biases])


def _assert_snapshot_equal(network: NeuralNetwork, snapshot) -> None:
    for current, expected in zip(network.weights, snapshot[0]):
        np.testing.assert_array_equal(current, expected)
    for current, expected in zip(network.biases, snapshot[1]):
        np.testing.assert_array_equal(current, expected)


def test_parse_and_format_parameter_cells():
    assert parse_parameter_cell(" 0.5 ", layer=0, kind="W", row=0, col=1) == 0.5
    with pytest.raises(ValueError, match=r"layer 1 b\[0, 1\]"):
        parse_parameter_cell("", layer=1, kind="b", row=0, col=1)
    with pytest.raises(ValueError, match="not a number"):
        parse_parameter_cell("oops", layer=0, kind="W", row=0, col=0)
    with pytest.raises(ValueError, match="finite"):
        parse_parameter_cell("inf", layer=0, kind="W", row=1, col=0)
    assert format_parameter(1.0 / 3.0) == repr(1.0 / 3.0)


def test_display_maps_layers_shapes_and_values():
    view = _view(_explicit_network())
    assert view.refresh_from_model() is True
    assert view._layer_combo.count() == 2
    assert "Layer 0" in view._layer_combo.itemText(0)
    assert "(2, 2)" in view._layer_combo.itemText(0)
    assert view._weights_table.rowCount() == 2
    assert view._weights_table.columnCount() == 2
    assert view._weights_table.item(0, 1).text() == repr(-0.5)
    assert view._biases_table.rowCount() == 1
    assert view._biases_table.columnCount() == 2
    assert view._biases_table.item(0, 1).text() == repr(0.1)
    assert "(2, 2)" in view._weights_shape.text()
    view._layer_combo.setCurrentIndex(1)
    assert view._weights_table.rowCount() == 2
    assert view._weights_table.columnCount() == 1
    assert view._weights_table.item(0, 0).text() == repr(1.0)
    assert view._biases_table.item(0, 0).text() == repr(0.25)
    assert "(2, 1)" in view._weights_shape.text()
    assert view.has_unsaved_changes() is False


def test_missing_model_disables_editing():
    view = _view(None)
    assert view.refresh_from_model() is False
    assert "Training view" in view._info_label.text()
    assert view._apply_button.isEnabled() is False
    view._on_apply()
    assert "No model" in view._error_label.text()


def test_valid_edit_updates_intended_parameter():
    network = _explicit_network()
    before_output = network.forward([[1.0, 0.5]])
    before = _snapshot(network)
    view = _view(network)
    assert view.refresh_from_model() is True
    view._weights_table.item(0, 1).setText("0.75")
    assert view.has_unsaved_changes() is True
    view._apply_button.click()
    assert view._error_label.text() == ""
    assert network.weights[0][0, 1] == pytest.approx(0.75)
    expected_w0 = before[0][0].copy()
    expected_w0[0, 1] = 0.75
    np.testing.assert_array_equal(network.weights[0], expected_w0)
    np.testing.assert_array_equal(network.weights[1], before[0][1])
    np.testing.assert_array_equal(network.biases[0], before[1][0])
    np.testing.assert_array_equal(network.biases[1], before[1][1])
    assert not np.array_equal(network.forward([[1.0, 0.5]]), before_output)
    assert "layer(s) [0]" in view._status_label.text()
    assert view.has_unsaved_changes() is False


def test_invalid_edit_leaves_model_unchanged():
    network = _explicit_network()
    before = _snapshot(network)
    view = _view(network)
    assert view.refresh_from_model() is True
    view._weights_table.item(0, 0).setText("oops")
    view._apply_button.click()
    assert "not a number" in view._error_label.text()
    _assert_snapshot_equal(network, before)
    assert view._weights_table.item(0, 0).text() == "oops"
    view._weights_table.item(0, 0).setText("0.5")
    view._biases_table.item(0, 0).setText("inf")
    view._apply_button.click()
    assert "finite" in view._error_label.text()
    _assert_snapshot_equal(network, before)


def test_architecture_change_requires_refresh():
    network = _explicit_network()
    view = _view(network)
    assert view.refresh_from_model() is True
    replacement = NeuralNetwork([2, 3, 1], seed=0)
    replacement_before = _snapshot(replacement)
    view.set_network_provider(lambda: replacement)
    view._weights_table.item(0, 0).setText("0.5")
    view._apply_button.click()
    assert "refresh" in view._error_label.text()
    _assert_snapshot_equal(replacement, replacement_before)
    _assert_snapshot_equal(network, _snapshot(network))


def test_pending_edits_survive_layer_switch():
    view = _view(_explicit_network())
    assert view.refresh_from_model() is True
    view._weights_table.item(1, 0).setText("0.99")
    view._layer_combo.setCurrentIndex(1)
    assert view._weights_table.item(1, 0).text() == repr(-1.0)
    view._layer_combo.setCurrentIndex(0)
    assert view._weights_table.item(1, 0).text() == "0.99"
    assert view.has_unsaved_changes() is True


def test_refresh_confirm_keeps_or_discards_edits(monkeypatch):
    network = _explicit_network()
    view = _view(network)
    assert view.refresh_from_model() is True
    view._weights_table.item(0, 0).setText("0.99")
    monkeypatch.setattr(
        "neural_network_lab.gui.views.parameters_view.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.Cancel,
    )
    view._refresh_button.click()
    assert view._weights_table.item(0, 0).text() == "0.99"
    assert view.has_unsaved_changes() is True
    monkeypatch.setattr(
        "neural_network_lab.gui.views.parameters_view.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.Discard,
    )
    view._refresh_button.click()
    assert view._weights_table.item(0, 0).text() == repr(0.5)
    assert view.has_unsaved_changes() is False


def test_refresh_picks_up_trained_model():
    network = _explicit_network()
    holder = {"network": network}
    view = _view(None)
    view.set_network_provider(lambda: holder["network"])
    assert view.refresh_from_model() is True
    assert view._weights_table.item(0, 0).text() == repr(0.5)
    retrained = _explicit_network()
    retrained.set_parameters(
        [np.full((2, 2), 0.125), np.full((2, 1), -0.25)],
        [np.zeros((1, 2)), np.zeros((1, 1))],
    )
    holder["network"] = retrained
    assert view.refresh_from_model() is True
    assert view._weights_table.item(0, 0).text() == repr(0.125)
    assert view.has_unsaved_changes() is False


def test_shownotice_when_model_changed_with_pending_edits():
    network = _explicit_network()
    holder = {"network": network}
    view = _view(None)
    view.set_network_provider(lambda: holder["network"])
    assert view.refresh_from_model() is True
    view._weights_table.item(0, 0).setText("0.99")
    holder["network"] = _explicit_network()
    view.showEvent(QShowEvent())
    assert "pending edits kept" in view._status_label.text()
    assert view._weights_table.item(0, 0).text() == "0.99"


def test_parameters_view_registers_in_navigation():
    create_application()
    window = MainWindow()
    window.register_default_views()
    assert window.request_view("parameters") is True
    assert type(window._pages["parameters"]).__name__ == "ParametersView"
