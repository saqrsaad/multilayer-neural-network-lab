"""GUI tests for the dataset editor (offscreen, service-backed)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

import numpy as np
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QTableWidgetItem

from neural_network_lab.data.csv_handler import import_dataset_from_csv
from neural_network_lab.data.dataset import Dataset
from neural_network_lab.gui.main_window import create_application
from neural_network_lab.gui.views.dataset_view import DatasetView
from neural_network_lab.gui.widgets.dataset_table import build_dataset_from_grid
from neural_network_lab.gui.widgets.dataset_table import DatasetTable
from neural_network_lab.gui.widgets.dataset_table import parse_float_cell


def _view() -> DatasetView:
    create_application()
    return DatasetView()


def _fill_row(view: DatasetView, row: int, values: list[str]) -> None:
    for col, text in enumerate(values):
        view._table.setItem(row, col, QTableWidgetItem(text))


def test_parse_float_cell_valid_and_located_errors():
    assert parse_float_cell("1.5", row=0, column_name="feature_0") == pytest.approx(1.5)
    assert parse_float_cell(" 2 ", row=1, column_name="feature_0") == pytest.approx(2.0)
    with pytest.raises(ValueError, match="row 1"):
        parse_float_cell("", row=0, column_name="feature_0")
    with pytest.raises(ValueError, match="feature_0"):
        parse_float_cell("abc", row=0, column_name="feature_0")
    with pytest.raises(ValueError, match="finite"):
        parse_float_cell("nan", row=2, column_name="target")


def test_build_dataset_from_grid_labeled_and_unlabeled():
    labeled = build_dataset_from_grid(
        [["0.0", "1.0", "0"], ["1.0", "0.0", "1"]],
        n_features=2,
        has_target=True,
    )
    assert isinstance(labeled, Dataset)
    assert (labeled.n_samples, labeled.n_features, labeled.n_outputs) == (2, 2, 1)
    unlabeled = build_dataset_from_grid(
        [["0.5", "0.25"]], n_features=2, has_target=False
    )
    assert unlabeled.has_targets is False


def test_build_dataset_from_grid_rejects_bad_grids():
    with pytest.raises(ValueError, match="at least one sample"):
        build_dataset_from_grid([], n_features=2, has_target=True)
    with pytest.raises(ValueError, match="Row 2"):
        build_dataset_from_grid(
            [["0.0", "1.0", "0"], ["1.0", "0"]], n_features=2, has_target=True
        )
    with pytest.raises(ValueError, match="must be 0 or 1"):
        build_dataset_from_grid([["0.0", "1.0", "2"]], n_features=2, has_target=True)
    with pytest.raises(ValueError, match="not a number"):
        build_dataset_from_grid([["0.0", "oops", "0"]], n_features=2, has_target=True)
    with pytest.raises(TypeError):
        build_dataset_from_grid([["0.0"]], n_features="2", has_target=False)


def test_dataset_table_add_remove_and_headers():
    create_application()
    table = DatasetTable()
    table.configure(2, True)
    assert table.columnCount() == 3
    table.add_empty_row()
    table.add_empty_row()
    assert table.rowCount() == 2
    assert table.grid() == [["", "", ""], ["", "", ""]]
    table.setItem(0, 0, QTableWidgetItem("1.0"))
    assert table.grid()[0][0] == "1.0"
    table.selectRow(0)
    assert table.remove_selected_rows() == 1
    assert table.rowCount() == 1


def test_view_validates_and_applies():
    view = _view()
    assert view.current_dataset() is None
    assert view.has_unsaved_changes() is False
    view._table.add_empty_row()
    _fill_row(view, 0, ["0.0", "1.0", "0"])
    assert view.has_unsaved_changes() is True
    dataset = view.validate_and_apply()
    assert isinstance(dataset, Dataset)
    assert view.has_unsaved_changes() is False
    assert "2 features" in view._summary_label.text()
    assert view._error_label.text() == ""


def test_view_reports_invalid_values_and_stays_dirty():
    view = _view()
    view._table.add_empty_row()
    _fill_row(view, 0, ["0.0", "oops", "0"])
    assert view.validate_and_apply() is None
    assert "not a number" in view._error_label.text()
    assert view.current_dataset() is None
    assert view.has_unsaved_changes() is True


def test_failed_import_preserves_table_state(tmp_path, monkeypatch):
    view = _view()
    view._table.add_empty_row()
    _fill_row(view, 0, ["0.0", "1.0", "0"])
    before = view._table.grid()
    bad_file = tmp_path / "bad.csv"
    bad_file.write_text("a,b\nx,1.0\n", encoding="utf-8")
    monkeypatch.setattr(
        "neural_network_lab.gui.views.dataset_view.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(bad_file), ""),
    )
    monkeypatch.setattr(
        "neural_network_lab.gui.views.dataset_view.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.Discard,
    )
    view._import_button.click()
    assert "Import failed" in view._error_label.text()
    assert view._table.grid() == before


def test_successful_import_replaces_table(tmp_path, monkeypatch):
    view = _view()
    good_file = tmp_path / "good.csv"
    good_file.write_text("feature_0,feature_1,target\n0.0,1.0,0\n", encoding="utf-8")
    monkeypatch.setattr(
        "neural_network_lab.gui.views.dataset_view.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(good_file), ""),
    )
    view._import_button.click()
    assert view._error_label.text() == ""
    assert view.has_unsaved_changes() is False
    dataset = view.current_dataset()
    assert dataset is not None
    np.testing.assert_allclose(dataset.features, np.array([[0.0, 1.0]]))


def test_export_validates_before_opening_dialog(monkeypatch):
    view = _view()
    view._table.add_empty_row()
    _fill_row(view, 0, ["bad", "1.0", "0"])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("dialog must not open for invalid data")

    monkeypatch.setattr(
        "neural_network_lab.gui.views.dataset_view.QFileDialog.getSaveFileName",
        fail_if_called,
    )
    view._export_button.click()
    assert "Export stopped" in view._error_label.text()


def test_export_writes_reimportable_csv(tmp_path, monkeypatch):
    view = _view()
    view._table.add_empty_row()
    _fill_row(view, 0, ["0.0", "1.0", "1"])
    assert view.validate_and_apply() is not None
    out_file = tmp_path / "out.csv"
    monkeypatch.setattr(
        "neural_network_lab.gui.views.dataset_view.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(out_file), ""),
    )
    view._export_button.click()
    assert view._error_label.text() == ""
    reloaded = import_dataset_from_csv(out_file, target_columns=2)
    np.testing.assert_allclose(reloaded.features, np.array([[0.0, 1.0]]))


def test_clear_requests_confirmation_when_dirty(monkeypatch):
    view = _view()
    view._table.add_empty_row()
    _fill_row(view, 0, ["0.0", "1.0", "0"])
    monkeypatch.setattr(
        "neural_network_lab.gui.views.dataset_view.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.Cancel,
    )
    view._clear_button.click()
    assert view._table.rowCount() == 1
    monkeypatch.setattr(
        "neural_network_lab.gui.views.dataset_view.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.Discard,
    )
    view._clear_button.click()
    assert view._table.rowCount() == 0
    assert view.current_dataset() is None


def test_layout_change_restores_controls_when_cancelled(monkeypatch):
    view = _view()
    view._table.add_empty_row()
    _fill_row(view, 0, ["0.0", "1.0", "0"])
    monkeypatch.setattr(
        "neural_network_lab.gui.views.dataset_view.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.Cancel,
    )
    view._feature_spin.setValue(3)
    assert view._feature_spin.value() == 2
    assert view._table.columnCount() == 3
    assert view._table.grid() == [["0.0", "1.0", "0"]]
