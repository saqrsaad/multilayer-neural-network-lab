"""Unit tests for dataset representation and validation (no CSV/GUI)."""

import numpy as np
import pytest

from neural_network_lab.data.dataset import Dataset


def test_labeled_dataset_representation():
    features = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
    targets = [[0.0], [1.0], [0.5]]
    dataset = Dataset(features, targets)
    assert dataset.has_targets is True
    assert dataset.n_samples == 3
    assert dataset.n_features == 2
    assert dataset.n_outputs == 1
    assert len(dataset) == 3
    np.testing.assert_allclose(
        dataset.features, np.array(features, dtype=float)
    )
    np.testing.assert_allclose(
        dataset.targets, np.array(targets, dtype=float)
    )
    assert "features=(3, 2)" in repr(dataset)


def test_unlabeled_dataset_targets_explicitly_none():
    dataset = Dataset([[0.0, 1.0], [1.0, 0.0]])
    assert dataset.has_targets is False
    assert dataset.targets is None
    assert dataset.n_outputs is None
    assert dataset.n_samples == 2
    features, targets = dataset.to_arrays()
    np.testing.assert_allclose(features, np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert targets is None


def test_constructor_copies_inputs():
    features = np.array([[0.0, 1.0]])
    targets = np.array([[1.0]])
    dataset = Dataset(features, targets)
    features[0, 0] = 999.0
    targets[0, 0] = 999.0
    np.testing.assert_allclose(dataset.features, np.array([[0.0, 1.0]]))
    np.testing.assert_allclose(dataset.targets, np.array([[1.0]]))


def test_properties_return_copies():
    dataset = Dataset([[0.0, 1.0]], [[1.0]])
    dataset.features[0, 0] = 999.0
    assert dataset.features[0, 0] == pytest.approx(0.0)
    dataset.targets[0, 0] = 999.0
    assert dataset.targets[0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "bad_features",
    [
        [1.0, 2.0, 3.0],  # 1D: must not be silently reshaped
        1.5,  # scalar
        np.zeros((2, 2, 2)),  # 3D
        [],  # empty
        np.zeros((0, 2)),  # zero samples
        np.zeros((3, 0)),  # zero feature columns
    ],
)
def test_rejects_bad_feature_shapes(bad_features):
    with pytest.raises(ValueError):
        Dataset(bad_features)


def test_rejects_non_numeric_features():
    with pytest.raises(TypeError):
        Dataset([["a", "b"]])


@pytest.mark.parametrize(
    "bad_features",
    [
        [[np.nan, 0.0]],
        [[np.inf, 0.0]],
        [[0.0, float("-inf")]],
    ],
)
def test_rejects_non_finite_features(bad_features):
    with pytest.raises(ValueError):
        Dataset(bad_features)


def test_targets_1d_not_silently_reshaped():
    with pytest.raises(ValueError):
        Dataset([[0.0, 1.0], [1.0, 0.0]], [0.0, 1.0])


def test_rejects_target_row_mismatch():
    with pytest.raises(ValueError):
        Dataset([[0.0, 1.0], [1.0, 0.0]], [[0.0]])


def test_rejects_empty_and_non_finite_targets():
    with pytest.raises(ValueError):
        Dataset([[0.0, 1.0]], np.zeros((0, 1)))
    with pytest.raises(ValueError):
        Dataset([[0.0, 1.0], [1.0, 0.0]], [[0.0], [np.nan]])
    with pytest.raises(ValueError):
        Dataset([[0.0, 1.0]], [[np.inf]])
    with pytest.raises(TypeError):
        Dataset([[0.0, 1.0]], [["a"]])


def test_compatibility_labeled():
    dataset = Dataset(
        [[0.0, 1.0], [1.0, 0.0]], [[0.0], [1.0]]
    )
    dataset.validate_compatible_with(2, 1)  # must not raise
    assert dataset.is_compatible_with(2, 1) is True
    with pytest.raises(ValueError):
        dataset.validate_compatible_with(3, 1)
    with pytest.raises(ValueError):
        dataset.validate_compatible_with(2, 2)
    assert dataset.is_compatible_with(3, 1) is False
    assert dataset.is_compatible_with(2, 2) is False


def test_compatibility_unlabeled():
    dataset = Dataset([[0.0, 1.0], [1.0, 0.0]])
    dataset.validate_compatible_with(2)  # prediction-only: must not raise
    dataset.validate_compatible_with(2, None)
    assert dataset.is_compatible_with(2) is True
    # Supervised use explicitly requires targets.
    with pytest.raises(ValueError):
        dataset.validate_compatible_with(2, 1)
    assert dataset.is_compatible_with(2, 1) is False
    with pytest.raises(ValueError):
        dataset.validate_compatible_with(5)


@pytest.mark.parametrize("bad_size", [0, -2, True, 2.5, "2", None])
def test_compatibility_rejects_bad_feature_size(bad_size):
    dataset = Dataset([[0.0, 1.0]], [[1.0]])
    with pytest.raises((TypeError, ValueError)):
        dataset.validate_compatible_with(bad_size, 1)
    assert dataset.is_compatible_with(bad_size, 1) is False


@pytest.mark.parametrize("bad_size", [0, -2, True, 2.5, "2"])
def test_compatibility_rejects_bad_output_size(bad_size):
    dataset = Dataset([[0.0, 1.0]], [[1.0]])
    with pytest.raises((TypeError, ValueError)):
        dataset.validate_compatible_with(2, bad_size)
    assert dataset.is_compatible_with(2, bad_size) is False


def test_compatibility_none_output_size_means_features_only():
    dataset = Dataset([[0.0, 1.0]], [[1.0]])
    dataset.validate_compatible_with(2, None)  # must not raise
    assert dataset.is_compatible_with(2, None) is True
