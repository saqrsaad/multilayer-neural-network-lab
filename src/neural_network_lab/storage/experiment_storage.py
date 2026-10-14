"""Saving and loading of experiment records.

Serialization format (JSON object, UTF-8):

- ``format``: ``"neural-network-lab/experiment-record"``.
- ``version``: ``1`` (only version 1 is read).
- ``payload``: the dictionary produced by
  ``ExperimentRecord.to_dict`` (identifier, timestamp, mode, layer
  sizes, learning rate, epochs, dataset information, loss history,
  final training loss, and test metrics).

Only ``json`` is used for deserialization; pickle is never used.
Full domain validation happens in ``ExperimentRecord.from_dict`` when
the caller rebuilds the record — loading here checks the envelope
and the presence of every required field. Compose::

    save_experiment_record(record.to_dict(), path)
    record = ExperimentRecord.from_dict(load_experiment_record(path))

No GUI imports here.
"""

from __future__ import annotations

import os
from pathlib import Path

from neural_network_lab.storage import _json_files

#: File envelope identifier for experiment records.
EXPERIMENT_FORMAT = "neural-network-lab/experiment-record"

#: The only experiment-record version this reader supports.
EXPERIMENT_FORMAT_VERSION = 1

#: Required payload keys, mirroring ``ExperimentRecord.to_dict``.
REQUIRED_RECORD_KEYS = (
    "experiment_id",
    "created_at",
    "mode",
    "layer_sizes",
    "learning_rate",
    "epochs",
    "train_dataset_name",
    "train_samples",
    "train_features",
    "train_outputs",
    "test_dataset_name",
    "loss_history",
    "final_train_loss",
    "test_mse",
    "test_accuracy",
)


def _check_record_payload(payload: dict) -> dict:
    """Check that all required record fields are present."""
    for key in REQUIRED_RECORD_KEYS:
        if key not in payload:
            raise ValueError(f"experiment record is missing {key!r}")
    return payload


def save_experiment_record(
    record: dict, path: str | os.PathLike[str], *, overwrite: bool = False
) -> Path:
    """Save an experiment record dict to a JSON file.

    Args:
        record: Dictionary as produced by ``ExperimentRecord.to_dict``.
        path: Destination file (parent directory must already exist).
        overwrite: Must be ``True`` to replace an existing file.

    Returns:
        The destination :class:`Path`.

    Raises:
        TypeError: If ``record`` is not a dict or arguments are invalid.
        ValueError: If required fields are missing or the payload
            holds non-finite numbers.
        FileExistsError: If the file exists and ``overwrite`` is False.
    """
    if not isinstance(record, dict):
        raise TypeError(
            f"experiment record must be a dict, got {type(record).__name__}"
        )
    _check_record_payload(record)
    file_path = _json_files.coerce_path(path, "path")
    _json_files.check_overwrite_allowed(file_path, overwrite, "experiment file")
    document = {
        "format": EXPERIMENT_FORMAT,
        "version": EXPERIMENT_FORMAT_VERSION,
        "payload": record,
    }
    return _json_files.write_json_document(document, file_path, "experiment record")


def load_experiment_record(path: str | os.PathLike[str]) -> dict:
    """Load an experiment record file and check its envelope and fields.

    The returned dict is ready for ``ExperimentRecord.from_dict``,
    which performs full semantic validation while rebuilding the
    record — no record object is constructed here.

    Raises:
        TypeError: If ``path`` is not a file path.
        FileNotFoundError: If the file does not exist.
        ValueError: For malformed JSON, wrong format/version, or
            missing record fields.
    """
    file_path = _json_files.coerce_path(path, "path")
    document = _json_files.read_json_document(file_path, "experiment record")
    payload = _json_files.check_envelope(
        document, file_path, "experiment record", EXPERIMENT_FORMAT, EXPERIMENT_FORMAT_VERSION
    )
    return _check_record_payload(payload)
