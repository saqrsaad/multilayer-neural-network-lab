"""Tests for background training: worker runs, view wiring, cleanup."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

import numpy as np
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QWidget

from neural_network_lab.core.network import NeuralNetwork
from neural_network_lab.data.dataset import Dataset
from neural_network_lab.gui.main_window import create_application
from neural_network_lab.gui.main_window import MainWindow
from neural_network_lab.gui.views.training_view import TrainingView
from neural_network_lab.gui.workers.training_worker import TrainingWorker


def _dataset():
    return Dataset(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        [[0.0], [1.0], [1.0], [0.0]],
    )


def _arrays(dataset):
    features, targets = dataset.to_arrays()
    assert targets is not None
    return features, targets


def _run_direct(worker):
    """Execute the worker synchronously, capturing signal payloads."""
    seen = {"progress": [], "finished": [], "failed": [], "cancelled": []}
    worker.progress.connect(lambda a, b, c: seen["progress"].append((a, b, c)))
    worker.finished.connect(lambda payload: seen["finished"].append(payload))
    worker.failed.connect(lambda message: seen["failed"].append(message))
    worker.cancelled.connect(lambda: seen["cancelled"].append(True))
    worker.run()
    return seen


def test_worker_scratch_reports_progress_and_result():
    features, targets = _arrays(_dataset())
    worker = TrainingWorker(
        mode="from_scratch",
        layer_sizes=[2, 4, 1],
        epochs=2,
        learning_rate=0.1,
        features=features,
        targets=targets,
    )
    seen = _run_direct(worker)
    assert seen["failed"] == [] and seen["cancelled"] == []
    assert [(epoch, total) for epoch, total, _ in seen["progress"]] == [(1, 2), (2, 2)]
    assert len(seen["finished"]) == 1
    payload = seen["finished"][0]
    assert payload["mode"] == "from_scratch"
    assert payload["epochs"] == 2
    assert len(payload["history"]) == 2
    assert all(isinstance(loss, float) for loss in payload["history"])
    assert payload["state"]["layer_sizes"] == [2, 4, 1]


def test_worker_continued_starts_from_snapshot_and_leaves_source_untouched():
    source = NeuralNetwork([2, 4, 1], seed=1)
    snapshot = source.to_state_dict()
    before = [w.copy() for w in source.weights]
    features, targets = _arrays(_dataset())
    worker = TrainingWorker(
        mode="continued",
        layer_sizes=[2, 4, 1],
        epochs=2,
        learning_rate=0.1,
        features=features,
        targets=targets,
        initial_state=snapshot,
    )
    seen = _run_direct(worker)
    assert len(seen["finished"]) == 1
    assert seen["finished"][0]["mode"] == "continued"
    for current, expected in zip(source.weights, before):
        np.testing.assert_array_equal(current, expected)


def test_worker_rejects_bad_mode_and_epochs():
    features, targets = _arrays(_dataset())
    worker = TrainingWorker(
        mode="nope",
        layer_sizes=[2, 4, 1],
        epochs=2,
        learning_rate=0.1,
        features=features,
        targets=targets,
    )
    seen = _run_direct(worker)
    assert len(seen["failed"]) == 1 and seen["finished"] == []
    assert "unknown training mode" in seen["failed"][0]
    worker = TrainingWorker(
        mode="from_scratch",
        layer_sizes=[2, 4, 1],
        epochs=0,
        learning_rate=0.1,
        features=features,
        targets=targets,
    )
    seen = _run_direct(worker)
    assert len(seen["failed"]) == 1 and seen["finished"] == []


def test_worker_reports_training_errors_without_crashing():
    features, targets = _arrays(_dataset())
    worker = TrainingWorker(
        mode="from_scratch",
        layer_sizes=[2, 4, 1],
        epochs=2,
        learning_rate=float("inf"),
        features=features,
        targets=targets,
    )
    seen = _run_direct(worker)
    assert len(seen["failed"]) == 1 and seen["finished"] == []


def test_worker_cancel_before_and_during_run():
    features, targets = _arrays(_dataset())
    worker = TrainingWorker(
        mode="from_scratch",
        layer_sizes=[2, 4, 1],
        epochs=3,
        learning_rate=0.1,
        features=features,
        targets=targets,
    )
    worker.request_cancel()
    seen = _run_direct(worker)
    assert seen["cancelled"] != [] and seen["finished"] == []

    worker = TrainingWorker(
        mode="from_scratch",
        layer_sizes=[2, 4, 1],
        epochs=5,
        learning_rate=0.1,
        features=features,
        targets=targets,
    )
    worker.progress.connect(lambda *args: worker.request_cancel())
    seen = _run_direct(worker)
    assert seen["cancelled"] != [] and seen["finished"] == []
    assert len(seen["progress"]) == 1


def test_view_refuses_second_run_while_active(monkeypatch):
    create_application()
    view = TrainingView()
    view.set_dataset_provider(_dataset)
    monkeypatch.setattr(TrainingView, "is_training_active", lambda self: True)
    view._controls._scratch_button.click()
    assert "already running" in view._controls._error_label.text()
    assert view._worker_thread is None


def test_view_progress_failed_and_cancelled_slots():
    create_application()
    view = TrainingView()
    view.set_dataset_provider(_dataset)
    view._on_worker_progress(1, 4, 0.5)
    assert "epoch 1/4" in view._controls._status_label.text()
    view._on_worker_failed("boom")
    assert "Training failed: boom" in view._controls._error_label.text()
    assert view._controls._scratch_button.isEnabled()
    view._on_worker_cancelled()
    assert "cancelled" in view._controls._status_label.text().lower()
    assert view.cancel_training() is False
    assert view.wait_for_worker() is True


def test_view_installs_result_preserving_object_identity():
    create_application()
    view = TrainingView()
    view.set_dataset_provider(_dataset)
    view._config._epochs_spin.setValue(1)
    spy = QSignalSpy(view.training_finished)
    view._controls._scratch_button.click()
    assert spy.wait(10000)
    first = view.current_network()
    assert first is not None
    second_spy = QSignalSpy(view.training_finished)
    view._controls._scratch_button.click()
    assert second_spy.wait(10000)
    assert view.current_network() is first


def test_close_prompts_and_cancels_running_training(monkeypatch):
    create_application()
    window = MainWindow()
    calls = {"cancel": 0, "wait": 0}

    class BusyView(QWidget):
        def is_training_active(self):
            return True

        def cancel_training(self):
            calls["cancel"] += 1
            return True

        def wait_for_worker(self, timeout_ms=5000):
            calls["wait"] += 1
            return True

    window._pages["busy"] = BusyView()
    window.show()
    assert window.isVisible()
    monkeypatch.setattr(
        "neural_network_lab.gui.main_window.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.Yes,
    )
    window.close()
    assert calls == {"cancel": 1, "wait": 1}
    assert not window.isVisible()


def test_close_without_training_skips_prompt(monkeypatch):
    create_application()
    window = MainWindow()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("no prompt expected without active training")

    monkeypatch.setattr(
        "neural_network_lab.gui.main_window.QMessageBox.question",
        fail_if_called,
    )
    window.show()
    assert window.isVisible()
    window.close()
    assert not window.isVisible()
