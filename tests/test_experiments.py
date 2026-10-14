"""Unit tests for experiment management (in-memory only, no files)."""

from datetime import datetime

import numpy as np
import pytest

from neural_network_lab.application.experiment_service import ExperimentRecord
from neural_network_lab.application.experiment_service import ExperimentStore
from neural_network_lab.application.experiment_service import compare_experiments
from neural_network_lab.application.experiment_service import run_experiment
from neural_network_lab.core.network import NeuralNetwork
from neural_network_lab.data.dataset import Dataset

TRAIN_FEATURES = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
TRAIN_TARGETS = [[0.0], [1.0], [1.0], [0.0]]
TEST_FEATURES = [[0.5, 0.5], [0.25, 0.75]]
TEST_TARGETS = [[1.0], [0.0]]


def _network():
    return NeuralNetwork([2, 3, 1], seed=5)


def _train_dataset():
    return Dataset(TRAIN_FEATURES, TRAIN_TARGETS)


def _test_dataset():
    return Dataset(TEST_FEATURES, TEST_TARGETS)


def test_run_experiment_continued_records_everything():
    network = _network()
    record = run_experiment(
        network,
        _train_dataset(),
        epochs=3,
        learning_rate=0.5,
        mode="continued",
        train_dataset_name="train-a",
        test_dataset=_test_dataset(),
        test_dataset_name="test-a",
    )
    assert isinstance(record.experiment_id, str) and record.experiment_id
    datetime.fromisoformat(record.created_at)  # must be a valid timestamp
    assert record.mode == "continued"
    assert record.layer_sizes == (2, 3, 1)
    assert record.learning_rate == pytest.approx(0.5)
    assert record.epochs == 3
    assert record.train_dataset_name == "train-a"
    assert record.train_samples == 4
    assert record.train_features == 2
    assert record.train_outputs == 1
    assert record.test_dataset_name == "test-a"
    assert len(record.loss_history) == 3
    assert all(isinstance(value, float) for value in record.loss_history)
    assert record.final_train_loss == pytest.approx(record.loss_history[-1])
    assert isinstance(record.test_mse, float)
    assert isinstance(record.test_accuracy, float)
    assert 0.0 <= record.test_accuracy <= 1.0


def test_run_experiment_ids_are_unique():
    first = run_experiment(_network(), _train_dataset(), epochs=1, learning_rate=0.1)
    second = run_experiment(_network(), _train_dataset(), epochs=1, learning_rate=0.1)
    assert first.experiment_id != second.experiment_id


def test_run_experiment_from_scratch_without_test_set():
    network = _network()
    record = run_experiment(
        network, _train_dataset(), epochs=2, learning_rate=0.5,
        mode="from_scratch", seed=9,
    )
    assert record.mode == "from_scratch"
    assert len(record.loss_history) == 2
    assert record.test_mse is None
    assert record.test_accuracy is None
    assert record.test_dataset_name is None


def test_run_experiment_rejects_bad_requests():
    with pytest.raises(ValueError):
        run_experiment(_network(), _train_dataset(), epochs=1,
                       learning_rate=0.1, mode="unknown")
    with pytest.raises(ValueError):
        run_experiment(_network(), Dataset([[0.0, 1.0]]), epochs=1,
                       learning_rate=0.1)
    with pytest.raises(ValueError):
        run_experiment(_network(), _train_dataset(), epochs=1,
                       learning_rate=0.1, test_dataset_name="orphan")
    with pytest.raises(ValueError):
        run_experiment(_network(), _train_dataset(), epochs=1,
                       learning_rate=0.1, seed=3)
    with pytest.raises(TypeError):
        run_experiment("nope", _train_dataset(), epochs=1, learning_rate=0.1)
    with pytest.raises(TypeError):
        run_experiment(_network(), [[0.0]], epochs=1, learning_rate=0.1)


def test_record_dict_round_trip():
    record = run_experiment(
        _network(), _train_dataset(), epochs=2, learning_rate=0.5,
        train_dataset_name="t", test_dataset=_test_dataset(),
        test_dataset_name="e",
    )
    rebuilt = ExperimentRecord.from_dict(record.to_dict())
    assert rebuilt == record


def test_record_from_dict_rejects_invalid():
    record = run_experiment(_network(), _train_dataset(), epochs=1, learning_rate=0.1)
    good = record.to_dict()
    with pytest.raises(ValueError):
        ExperimentRecord.from_dict({k: v for k, v in good.items() if k != "mode"})
    for key, value in [
        ("mode", "nope"), ("epochs", 0), ("learning_rate", -1.0),
        ("layer_sizes", [2]), ("loss_history", []),
        ("final_train_loss", 999.0), ("test_accuracy", 1.5),
        ("test_mse", -0.5), ("train_samples", 0),
    ]:
        bad = dict(good)
        bad[key] = value
        with pytest.raises(ValueError):
            ExperimentRecord.from_dict(bad)
    with pytest.raises(ValueError):
        ExperimentRecord.from_dict(dict(good, created_at="not-a-time"))
    with pytest.raises(TypeError):
        ExperimentRecord.from_dict([1, 2, 3])


def test_store_add_list_get():
    store = ExperimentStore()
    assert len(store) == 0
    first = run_experiment(_network(), _train_dataset(), epochs=1, learning_rate=0.1)
    second = run_experiment(_network(), _train_dataset(), epochs=1, learning_rate=0.1)
    store.add(first)
    store.add(second)
    assert len(store) == 2
    assert store.list_experiments() == [first, second]
    assert store.get_experiment(first.experiment_id) == first
    with pytest.raises(KeyError):
        store.get_experiment("missing-id")
    with pytest.raises(ValueError):
        store.add(first)  # duplicates never overwrite
    with pytest.raises(TypeError):
        store.add("not-a-record")


def test_compare_compatible_experiments():
    network = _network()
    kwargs = dict(
        train_dataset_name="train-a", test_dataset=_test_dataset(),
        test_dataset_name="test-a",
    )
    first = run_experiment(network, _train_dataset(), epochs=1,
                           learning_rate=0.1, **kwargs)
    second = run_experiment(network, _train_dataset(), epochs=2,
                            learning_rate=0.1, **kwargs)
    comparison = compare_experiments(first, second)
    assert comparison.first_id == first.experiment_id
    assert comparison.second_id == second.experiment_id
    assert comparison.train_loss_change == pytest.approx(
        second.final_train_loss - first.final_train_loss
    )
    assert comparison.test_mse_change == pytest.approx(
        second.test_mse - first.test_mse
    )
    assert comparison.test_accuracy_change == pytest.approx(
        second.test_accuracy - first.test_accuracy
    )


def test_compare_without_test_metrics():
    first = run_experiment(_network(), _train_dataset(), epochs=1, learning_rate=0.1)
    second = run_experiment(_network(), _train_dataset(), epochs=1, learning_rate=0.1)
    comparison = compare_experiments(first, second)
    assert comparison.test_mse_change is None
    assert comparison.test_accuracy_change is None
    assert isinstance(comparison.train_loss_change, float)


def test_compare_rejects_incompatible_and_wrong_types():
    base = run_experiment(
        _network(), _train_dataset(), epochs=1, learning_rate=0.1,
        train_dataset_name="t", test_dataset=_test_dataset(),
        test_dataset_name="e",
    )
    other_arch = run_experiment(
        NeuralNetwork([2, 2, 1], seed=5), _train_dataset(), epochs=1,
        learning_rate=0.1, train_dataset_name="t",
        test_dataset=_test_dataset(), test_dataset_name="e",
    )
    with pytest.raises(ValueError):
        compare_experiments(base, other_arch)
    other_test = run_experiment(
        _network(), _train_dataset(), epochs=1, learning_rate=0.1,
        train_dataset_name="t", test_dataset=_test_dataset(),
        test_dataset_name="other",
    )
    with pytest.raises(ValueError):
        compare_experiments(base, other_test)
    other_train = run_experiment(
        _network(), _train_dataset(), epochs=1, learning_rate=0.1,
        train_dataset_name="different",
        test_dataset=_test_dataset(), test_dataset_name="e",
    )
    with pytest.raises(ValueError):
        compare_experiments(base, other_train)
    with pytest.raises(TypeError):
        compare_experiments(base, "not-a-record")
