"""Saving and loading of model states (architecture + parameters).

Serialization format (JSON object, UTF-8):

- ``format``: ``"neural-network-lab/model-state"``.
- ``version``: ``1`` (only version 1 is read).
- ``payload``: ``{"layer_sizes": [...], "weights": [...], "biases": [...]}``
  as produced by ``NeuralNetwork.to_state_dict`` — plain nested lists
  of finite numbers at full float precision, so a loaded model
  produces predictions identical to the saved one.

Only ``json`` is used for deserialization; pickle is never used.
Compose with the core API::

    save_model_state(network.to_state_dict(), path)
    network = NeuralNetwork.from_state_dict(load_model_state(path))

No GUI imports here.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from neural_network_lab.storage import _json_files

#: File envelope identifier for model states.
MODEL_FORMAT = "neural-network-lab/model-state"

#: The only model-state version this reader supports.
MODEL_FORMAT_VERSION = 1


def _check_number_matrix(value: object, name: str) -> list:
    """Validate one parameter list as a rectangular 2D finite-number matrix."""
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError(f"model state {name!r} must be a non-empty list of rows")
    width: int | None = None
    for row in value:
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError(f"model state {name!r} rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"model state {name!r} rows have inconsistent lengths")
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, (int, float)):
                raise ValueError(
                    f"model state {name!r} must contain only numbers"
                )
            if isinstance(cell, float) and not math.isfinite(cell):
                raise ValueError(
                    f"model state {name!r} must contain only finite numbers"
                )
    return value


def _check_model_payload(payload: dict) -> dict:
    """Validate a model payload's structure before any model is built."""
    for key in ("layer_sizes", "weights", "biases"):
        if key not in payload:
            raise ValueError(f"model state is missing {key!r}")
    layer_sizes = payload["layer_sizes"]
    if (
        not isinstance(layer_sizes, list)
        or len(layer_sizes) < 2
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in layer_sizes
        )
    ):
        raise ValueError(
            "model state needs 'layer_sizes' with at least two positive integers"
        )
    expected = len(layer_sizes) - 1
    for key in ("weights", "biases"):
        matrices = payload[key]
        if not isinstance(matrices, list) or len(matrices) != expected:
            raise ValueError(
                f"model state {key!r} must hold {expected} matrices "
                f"for {len(layer_sizes)} layer sizes"
            )
        for matrix in matrices:
            _check_number_matrix(matrix, f"{key}[...]")
    return payload


def save_model_state(
    state: dict, path: str | os.PathLike[str], *, overwrite: bool = False
) -> Path:
    """Save a model state dict to a JSON file.

    Args:
        state: Dictionary as produced by ``NeuralNetwork.to_state_dict``.
        path: Destination file (parent directory must already exist).
        overwrite: Must be ``True`` to replace an existing file.

    Returns:
        The destination :class:`Path`.

    Raises:
        TypeError: If ``state`` is not a dict or arguments are invalid.
        ValueError: If the payload is structurally invalid or holds
            non-finite numbers.
        FileExistsError: If the file exists and ``overwrite`` is False.
    """
    if not isinstance(state, dict):
        raise TypeError(f"model state must be a dict, got {type(state).__name__}")
    _check_model_payload(state)
    file_path = _json_files.coerce_path(path, "path")
    _json_files.check_overwrite_allowed(file_path, overwrite, "model file")
    document = {
        "format": MODEL_FORMAT,
        "version": MODEL_FORMAT_VERSION,
        "payload": state,
    }
    return _json_files.write_json_document(document, file_path, "model state")


def load_model_state(path: str | os.PathLike[str]) -> dict:
    """Load and structurally validate a model state file.

    The returned dict is ready for ``NeuralNetwork.from_state_dict``,
    which performs the final shape/value validation while building the
    model — no model object is constructed or modified here.

    Raises:
        TypeError: If ``path`` is not a file path.
        FileNotFoundError: If the file does not exist.
        ValueError: For malformed JSON, wrong format/version, or an
            invalid model payload.
    """
    file_path = _json_files.coerce_path(path, "path")
    document = _json_files.read_json_document(file_path, "model state")
    payload = _json_files.check_envelope(
        document, file_path, "model state", MODEL_FORMAT, MODEL_FORMAT_VERSION
    )
    return _check_model_payload(payload)
