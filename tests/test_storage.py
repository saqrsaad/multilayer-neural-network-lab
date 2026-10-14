"""Unit tests for model and experiment persistence (tmp files only).

Covers save/load round trips, numerical precision, overwrite
protection, and failure cases. No pickle is used anywhere: saved
files must be plain JSON.
"""

import json

import numpy as np
import pytest

from neural_network_lab.application.experiment_service import ExperimentRecord
from neural_network_lab.application.experiment_service import run_experiment
from neural_network_lab.core.network import NeuralNetwork
from neural_network_lab.data.dataset import Dataset
from neural_network_lab.storage.experiment_storage import load_experiment_record
from neural_network_lab.storage.experiment_storage import save_experiment_record
from neural_network_lab.storage.model_storage import load_model_state
from neural_network_lab.storage.model_storage import save_model_state


def _explicit_network():
    return NeuralNetwork(
        [2, 2, 1],
        weights=[np.array([[0.5, -0.5], [0.25, 0.75]]), np.array([[1.0], [-1.0]])],
        biases=[np.array([[0.0, 0.1]]), np.array([[0.25]])],
    )


def test_model_round_trip_preserves_predictions(tmp_path):
    network = _explicit_network()
    path = tmp_path / "model.json"
    save_model_state(network.to_state_dict(), path)
    rebuilt = NeuralNetwork.from_state_dict(load_model_state(path))
    assert rebuilt.layer_sizes == network.layer_sizes
    for old, new in zip(network.weights, rebuilt.weights):
        np.testing.assert_array_equal(old, new)
    sample = np.array([[1.0, 0.5], [0.0, 0.0]])
    np.testing.assert_array_equal(rebuilt.forward(sample), network.forward(sample))


def test_saved_model_file_is_plain_json(tmp_path):
    path = tmp_path / "model.json"
    save_model_state(_explicit_network().to_state_dict(), path)
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    assert document["format"] == "neural-network-lab/model-state"
    assert document["version"] == 1
    assert document["payload"]["layer_sizes"] == [2, 2, 1]


def test_experiment_round_trip(tmp_path):
    record = run_experiment(
        NeuralNetwork([2, 3, 1], seed=5),
        Dataset([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], [[0.0], [1.0], [1.0]]),
        epochs=2,
        learning_rate=0.5,
        train_dataset_name="train-a",
        test_dataset=Dataset([[1.0, 1.0]], [[0.0]]),
        test_dataset_name="test-a",
    )
    path = tmp_path / "experiment.json"
    save_experiment_record(record.to_dict(), path)
    rebuilt = ExperimentRecord.from_dict(load_experiment_record(path))
    assert rebuilt == record


def test_saved_experiment_file_is_plain_json(tmp_path):
    record = run_experiment(
        NeuralNetwork([2, 2, 1], seed=5),
        Dataset([[0.0, 0.0], [1.0, 1.0]], [[0.0], [1.0]]),
        epochs=1,
        learning_rate=0.1,
    )
    path = tmp_path / "experiment.json"
    save_experiment_record(record.to_dict(), path)
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    assert document["format"] == "neural-network-lab/experiment-record"
    assert document["payload"]["experiment_id"] == record.experiment_id


def test_save_refuses_overwrite_without_authorization(tmp_path):
    state = _explicit_network().to_state_dict()
    path = tmp_path / "model.json"
    save_model_state(state, path)
    with pytest.raises(FileExistsError):
        save_model_state(state, path)
    save_model_state(state, path, overwrite=True)  # must not raise
    record = run_experiment(
        NeuralNetwork([2, 2, 1], seed=5),
        Dataset([[0.0, 0.0], [1.0, 1.0]], [[0.0], [1.0]]),
        epochs=1,
        learning_rate=0.1,
    ).to_dict()
    exp_path = tmp_path / "experiment.json"
    save_experiment_record(record, exp_path)
    with pytest.raises(FileExistsError):
        save_experiment_record(record, exp_path)
    save_experiment_record(record, exp_path, overwrite=True)


def test_load_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_state(tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError):
        load_experiment_record(tmp_path / "absent.json")


@pytest.mark.parametrize(
    "text",
    [
        "{not json",  # malformed JSON
        "[1, 2, 3]",  # valid JSON but not an object
        '{"format": "other", "version": 1, "payload": {}}',  # wrong format
        '{"format": "neural-network-lab/model-state", "version": 999,'
        ' "payload": {}}',  # unsupported version
        '{"format": "neural-network-lab/model-state", "version": 1}',  # no payload
        '{"format": "neural-network-lab/model-state", "version": 1,'
        ' "payload": {"layer_sizes": [2, 1]}}',  # missing keys
        '{"format": "neural-network-lab/model-state", "version": 1,'
        ' "payload": {"layer_sizes": [2, 1], "weights": [[[0.0]]],'
        ' "biases": [[[NaN]]]}}',  # NaN literal rejected
    ],
)
def test_load_model_rejects_bad_files(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        load_model_state(path)


def test_load_model_rejects_ragged_and_mismatched_payload(tmp_path):
    state = _explicit_network().to_state_dict()
    state["weights"][0] = [[0.5, -0.5], [0.25]]  # ragged rows
    path = tmp_path / "ragged.json"
    document = {
        "format": "neural-network-lab/model-state",
        "version": 1,
        "payload": state,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle)
    with pytest.raises(ValueError):
        load_model_state(path)
    tampered = _explicit_network().to_state_dict()
    tampered["layer_sizes"] = [2, 3, 1]  # no longer matches the matrices
    with pytest.raises(ValueError):
        NeuralNetwork.from_state_dict(tampered)
    with pytest.raises(TypeError):
        NeuralNetwork.from_state_dict([1, 2, 3])
    with pytest.raises(ValueError):
        NeuralNetwork.from_state_dict({"layer_sizes": [2, 1]})


def test_load_model_rejects_non_finite_json_numbers(tmp_path):
    path = tmp_path / "huge.json"
    path.write_text(
        '{"format": "neural-network-lab/model-state", "version": 1,'
        ' "payload": {"layer_sizes": [1, 1], "weights": [[[1e999]]],'
        ' "biases": [[[0.0]]]}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_model_state(path)


def test_load_model_rejects_pickle_bytes(tmp_path):
    path = tmp_path / "evil.pkl"
    path.write_bytes(b"\x80\x04\x95\x00\x00\x00\x00\x00\x00\x00\x00.")
    with pytest.raises(ValueError):
        load_model_state(path)


def test_save_rejects_non_finite_state(tmp_path):
    state = _explicit_network().to_state_dict()
    state["weights"][0][0][0] = float("inf")
    with pytest.raises(ValueError):
        save_model_state(state, tmp_path / "bad.json")


def test_load_experiment_rejects_bad_files(tmp_path):
    record = run_experiment(
        NeuralNetwork([2, 2, 1], seed=5),
        Dataset([[0.0, 0.0], [1.0, 1.0]], [[0.0], [1.0]]),
        epochs=1,
        learning_rate=0.1,
    ).to_dict()
    good_path = tmp_path / "good.json"
    save_experiment_record(record, good_path)
    with open(good_path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    assert isinstance(document["payload"]["loss_history"], list)
    incomplete = dict(document)
    incomplete["payload"] = {k: v for k, v in record.items() if k != "epochs"}
    bad_path = tmp_path / "incomplete.json"
    with open(bad_path, "w", encoding="utf-8") as handle:
        json.dump(incomplete, handle)
    with pytest.raises(ValueError):
        load_experiment_record(bad_path)
    wrong_format = dict(document, format="something-else")
    wrong_path = tmp_path / "wrong.json"
    with open(wrong_path, "w", encoding="utf-8") as handle:
        json.dump(wrong_format, handle)
    with pytest.raises(ValueError):
        load_experiment_record(wrong_path)
    text_path = tmp_path / "text.json"
    text_path.write_text("just some text", encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment_record(text_path)


def test_save_rejects_wrong_types(tmp_path):
    with pytest.raises(TypeError):
        save_model_state([1, 2, 3], tmp_path / "bad.json")
    with pytest.raises(TypeError):
        save_experiment_record("nope", tmp_path / "bad.json")
    with pytest.raises(TypeError):
        load_model_state(123)
