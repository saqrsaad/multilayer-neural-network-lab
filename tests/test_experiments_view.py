"""GUI tests for the experiment history view (offscreen)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QTableWidgetSelectionRange

from neural_network_lab.application.experiment_service import ExperimentRecord
from neural_network_lab.gui.main_window import create_application
from neural_network_lab.gui.main_window import MainWindow
from neural_network_lab.gui.views.experiments_view import comparison_rows
from neural_network_lab.gui.views.experiments_view import ExperimentsView
from neural_network_lab.gui.views.experiments_view import format_layers
from neural_network_lab.gui.views.experiments_view import format_metric
from neural_network_lab.gui.views.experiments_view import format_name
from neural_network_lab.gui.views.experiments_view import format_record_detail
from neural_network_lab.gui.views.experiments_view import MISSING_METRIC_TEXT
from neural_network_lab.storage.experiment_storage import save_experiment_record


def _record_dict(**overrides):
    base = {
        "experiment_id": "abc123def456",
        "created_at": "2026-01-01T00:00:00+00:00",
        "mode": "continued",
        "layer_sizes": [2, 3, 1],
        "learning_rate": 0.1,
        "epochs": 2,
        "train_dataset_name": "train-a",
        "train_samples": 4,
        "train_features": 2,
        "train_outputs": 1,
        "test_dataset_name": "test-a",
        "loss_history": [0.5, 0.25],
        "final_train_loss": 0.25,
        "test_mse": 0.2,
        "test_accuracy": 0.75,
    }
    base.update(overrides)
    return base


def _record(**overrides):
    return ExperimentRecord.from_dict(_record_dict(**overrides))


def _write(directory, name, record_dict):
    save_experiment_record(record_dict, directory / name)


def _view() -> ExperimentsView:
    create_application()
    return ExperimentsView()


def _select_rows(view: ExperimentsView, first: int, last: int) -> None:
    view._records_table.setRangeSelected(
        QTableWidgetSelectionRange(first, 0, last, 10), True
    )


def test_format_helpers_distinguish_missing_from_zero():
    assert format_metric(None) == MISSING_METRIC_TEXT
    assert "not recorded" in format_metric(None)
    assert format_metric(0.0) == "0.0"
    assert format_layers((2, 3, 1)) == "[2, 3, 1]"
    assert format_name(None) == "—"
    assert format_name("train-a") == "train-a"
    detail = format_record_detail(_record())
    assert "abc123def456" in detail
    assert "epoch 2/2" in detail
    assert "not recorded" not in detail


def test_load_directory_lists_saved_records(tmp_path):
    _write(tmp_path, "a.json", _record_dict(experiment_id="a" * 12))
    _write(tmp_path, "b.json", _record_dict(experiment_id="b" * 12))
    view = _view()
    assert view.load_directory(tmp_path) is True
    assert [r.experiment_id for r in view.loaded_records()] == ["a" * 12, "b" * 12]
    assert view._records_table.rowCount() == 2
    assert view._records_table.columnCount() == 11
    assert "2 record(s) loaded" in view._status_label.text()
    assert view._error_label.text() == ""
    assert view.load_failures() == []


def test_load_skips_bad_files_and_keeps_valid_ones(tmp_path):
    _write(tmp_path, "good.json", _record_dict(experiment_id="c" * 12))
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "wrong.json").write_text(
        '{"format": "other", "version": 1, "payload": {}}', encoding="utf-8"
    )
    view = _view()
    assert view.load_directory(tmp_path) is True
    assert [r.experiment_id for r in view.loaded_records()] == ["c" * 12]
    assert len(view.load_failures()) == 2
    assert "1 record(s) loaded" in view._status_label.text()
    assert "2 file(s) skipped" in view._status_label.text()
    assert "broken.json" in view._error_label.text()
    assert "wrong.json" in view._error_label.text()


def test_missing_directory_leaves_list_untouched(tmp_path):
    _write(tmp_path, "a.json", _record_dict(experiment_id="a" * 12))
    view = _view()
    assert view.load_directory(tmp_path) is True
    assert view.load_directory(tmp_path / "absent") is False
    assert "Not a directory" in view._error_label.text()
    assert [r.experiment_id for r in view.loaded_records()] == ["a" * 12]
    assert view._records_table.rowCount() == 1


def test_selecting_row_shows_full_detail(tmp_path):
    _write(tmp_path, "a.json", _record_dict(experiment_id="a" * 12))
    view = _view()
    assert view.load_directory(tmp_path) is True
    view._records_table.selectRow(0)
    detail = view._detail_view.toPlainText()
    assert "a" * 12 in detail
    assert "epoch 1/2" in detail
    assert "Test accuracy" in detail


def test_compare_compatible_pair_shows_labeled_deltas(tmp_path):
    _write(tmp_path, "a.json", _record_dict(
        experiment_id="a" * 12, final_train_loss=0.25,
        loss_history=[0.5, 0.25], test_mse=0.2, test_accuracy=0.75,
    ))
    _write(tmp_path, "b.json", _record_dict(
        experiment_id="b" * 12, final_train_loss=0.125,
        loss_history=[0.4, 0.125], test_mse=0.1, test_accuracy=1.0,
    ))
    view = _view()
    assert view.load_directory(tmp_path) is True
    before = sorted(path.name for path in tmp_path.iterdir())
    _select_rows(view, 0, 1)
    view._compare_button.click()
    assert view._error_label.text() == ""
    assert sorted(path.name for path in tmp_path.iterdir()) == before
    table = view._comparison_table
    assert table.columnCount() == 3
    assert table.item(0, 1).text() == "a" * 12
    assert table.item(0, 2).text() == "b" * 12
    labels = [table.item(row, 0).text() for row in range(table.rowCount())]
    train_row = next(
        row for row in range(table.rowCount()) if "train loss change" in labels[row]
    )
    assert table.item(train_row, 2).text().startswith("-0.125")
    assert "negative is better" in labels[train_row]
    acc_row = next(
        row for row in range(table.rowCount()) if "accuracy change" in labels[row]
    )
    assert table.item(acc_row, 2).text().startswith("+0.25")
    assert "positive is better" in labels[acc_row]
    mse_row = next(
        row for row in range(table.rowCount()) if row not in (train_row, acc_row)
        and "Test MSE change" in labels[row]
    )
    assert table.item(mse_row, 2).text().startswith("-0.1")


def test_compare_marks_absent_metrics_without_fabrication(tmp_path):
    _write(tmp_path, "a.json", _record_dict(
        experiment_id="a" * 12, test_mse=None, test_accuracy=None,
        test_dataset_name=None,
    ))
    _write(tmp_path, "b.json", _record_dict(
        experiment_id="b" * 12, test_mse=None, test_accuracy=None,
        test_dataset_name=None,
    ))
    view = _view()
    assert view.load_directory(tmp_path) is True
    _select_rows(view, 0, 1)
    view._compare_button.click()
    assert view._error_label.text() == ""
    table = view._comparison_table
    texts = [
        table.item(row, col).text()
        for row in range(table.rowCount())
        for col in (1, 2)
    ]
    assert MISSING_METRIC_TEXT in texts
    assert table.item(0, 1).text() == "a" * 12


def test_compare_rejects_wrong_selection_count(tmp_path):
    _write(tmp_path, "a.json", _record_dict(experiment_id="a" * 12))
    view = _view()
    assert view.load_directory(tmp_path) is True
    view._compare_button.click()
    assert "exactly two" in view._error_label.text()
    view._records_table.selectRow(0)
    view._compare_button.click()
    assert "exactly two" in view._error_label.text()


def test_compare_rejects_incompatible_pair(tmp_path):
    _write(tmp_path, "a.json", _record_dict(experiment_id="a" * 12))
    _write(tmp_path, "b.json", _record_dict(
        experiment_id="b" * 12, layer_sizes=[2, 5, 1],
    ))
    view = _view()
    assert view.load_directory(tmp_path) is True
    _select_rows(view, 0, 1)
    view._compare_button.click()
    assert "Cannot compare" in view._error_label.text()
    assert "layer_sizes" in view._error_label.text()


def test_comparison_rows_use_recorded_values_only():
    first = _record(experiment_id="a" * 12)
    second = _record(
        experiment_id="b" * 12, final_train_loss=0.125,
        loss_history=[0.4, 0.125], test_mse=0.1, test_accuracy=1.0,
    )
    from neural_network_lab.application.experiment_service import compare_experiments

    rows = dict(
        (field, (left, right))
        for field, left, right in comparison_rows(
            first, second, compare_experiments(first, second)
        )
    )
    assert rows["Final train loss (MSE)"] == ("0.25", "0.125")
    assert rows["Test accuracy"] == ("0.75", "1.0")


def test_experiments_view_registers_in_navigation():
    create_application()
    window = MainWindow()
    window.register_default_views()
    assert window.request_view("experiments") is True
    assert type(window._pages["experiments"]).__name__ == "ExperimentsView"
