"""Unit tests for CSV import and export (small temporary files only)."""

import numpy as np
import pytest

from neural_network_lab.data.csv_handler import export_dataset_to_csv
from neural_network_lab.data.csv_handler import import_dataset_from_csv
from neural_network_lab.data.dataset import Dataset


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_import_labeled_csv_with_explicit_target_column(tmp_path):
    csv_file = _write(
        tmp_path / "labeled.csv",
        "feature_0,feature_1,target_0\n0.0,1.0,0.0\n1.0,0.0,1.0\n",
    )
    dataset = import_dataset_from_csv(csv_file, target_columns=2)
    assert dataset.n_samples == 2
    assert dataset.n_features == 2
    assert dataset.n_outputs == 1
    np.testing.assert_allclose(
        dataset.features, np.array([[0.0, 1.0], [1.0, 0.0]])
    )
    np.testing.assert_allclose(dataset.targets, np.array([[0.0], [1.0]]))


def test_import_unlabeled_csv_requires_explicit_none(tmp_path):
    csv_file = _write(
        tmp_path / "features.csv", "feature_0,feature_1\n0.5,0.25\n"
    )
    dataset = import_dataset_from_csv(csv_file, target_columns=None)
    assert dataset.has_targets is False
    assert dataset.n_features == 2


def test_import_multiple_and_first_column_targets(tmp_path):
    csv_file = _write(
        tmp_path / "multi.csv", "h0,h1,h2\n1.0,0.0,0.5\n0.0,1.0,0.25\n"
    )
    dataset = import_dataset_from_csv(csv_file, target_columns=[0, 2])
    np.testing.assert_allclose(dataset.features, np.array([[0.0], [1.0]]))
    np.testing.assert_allclose(
        dataset.targets, np.array([[1.0, 0.5], [0.0, 0.25]])
    )


def test_import_without_header(tmp_path):
    csv_file = _write(tmp_path / "noheader.csv", "0.0,1.0,0.0\n1.0,0.0,1.0\n")
    dataset = import_dataset_from_csv(
        csv_file, target_columns=2, has_header=False
    )
    assert dataset.n_samples == 2
    assert dataset.n_features == 2


def test_export_import_round_trip_preserves_values(tmp_path):
    original = Dataset(
        [[0.1, 1.0 / 3.0], [1.0, 0.0], [123.456, -78.9]],
        [[0.0], [1.0], [0.0]],
    )
    csv_file = tmp_path / "roundtrip.csv"
    export_dataset_to_csv(original, csv_file)
    reloaded = import_dataset_from_csv(csv_file, target_columns=2)
    np.testing.assert_array_equal(reloaded.features, original.features)
    np.testing.assert_array_equal(reloaded.targets, original.targets)


def test_export_import_round_trip_unlabeled(tmp_path):
    original = Dataset([[0.1, 0.2], [0.3, 0.4]])
    csv_file = tmp_path / "unlabeled.csv"
    export_dataset_to_csv(original, csv_file)
    reloaded = import_dataset_from_csv(csv_file, target_columns=None)
    np.testing.assert_array_equal(reloaded.features, original.features)
    assert reloaded.has_targets is False


def test_export_refuses_overwrite_without_authorization(tmp_path):
    dataset = Dataset([[0.0, 1.0]], [[1.0]])
    csv_file = tmp_path / "exists.csv"
    export_dataset_to_csv(dataset, csv_file)
    with pytest.raises(FileExistsError):
        export_dataset_to_csv(dataset, csv_file)
    export_dataset_to_csv(dataset, csv_file, overwrite=True)  # must not raise


def test_export_rejects_non_dataset(tmp_path):
    with pytest.raises(TypeError):
        export_dataset_to_csv([[0.0]], tmp_path / "bad.csv")


@pytest.mark.parametrize(
    "text",
    [
        "",  # empty file
        "a,b\n",  # header only
        "a,b\n1.0\n2.0,3.0\n",  # ragged rows
        "a,b\n\n1.0,2.0\n",  # blank row
        "a,b\n1.0,2.0\n3.0\n",  # short row
        "a,b\nx,2.0\n",  # non-numeric cell
        "a,b\n1.0,\n",  # empty cell
    ],
)
def test_import_rejects_malformed_files(tmp_path, text):
    csv_file = _write(tmp_path / "bad.csv", text)
    with pytest.raises(ValueError):
        import_dataset_from_csv(csv_file, target_columns=1)


def test_import_rejects_non_finite_values(tmp_path):
    csv_file = _write(tmp_path / "nonfinite.csv", "a,b\nnan,1.0\n")
    with pytest.raises(ValueError):
        import_dataset_from_csv(csv_file, target_columns=1)
    csv_file = _write(tmp_path / "inf.csv", "a,b\n1.0,inf\n")
    with pytest.raises(ValueError):
        import_dataset_from_csv(csv_file, target_columns=0)


def test_import_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        import_dataset_from_csv(tmp_path / "absent.csv", target_columns=0)


@pytest.mark.parametrize(
    "target_columns",
    [5, -1, [0, 0], [], [0, 1], "1", 1.5, True],
)
def test_import_rejects_bad_target_columns(tmp_path, target_columns):
    csv_file = _write(tmp_path / "cols.csv", "a,b\n1.0,2.0\n")
    with pytest.raises((TypeError, ValueError)):
        import_dataset_from_csv(csv_file, target_columns=target_columns)
