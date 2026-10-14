"""Shared JSON file helpers for the storage layer.

File-level concerns only (paths, overwrite protection, JSON parsing):
domain validation lives in the model, experiment, and storage modules
that own each format. No domain imports here, and never pickle.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path


def coerce_path(path: str | os.PathLike[str], name: str) -> Path:
    """Convert a path argument to ``Path``, rejecting other types."""
    if isinstance(path, (str, os.PathLike)):
        return Path(os.fspath(path))
    raise TypeError(f"{name} must be a file path string, got {type(path).__name__}")


def check_overwrite_allowed(path: Path, overwrite: bool, what: str) -> None:
    """Enforce explicit authorization before replacing an existing file."""
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{what} already exists: {path}; pass overwrite=True "
            "to replace it explicitly"
        )


def _reject_non_finite(value: object, what: str) -> None:
    """Reject NaN/inf floats anywhere in a payload before saving."""
    if isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{what} contains a non-finite number")
        return
    if isinstance(value, dict):
        for item in value.values():
            _reject_non_finite(item, what)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_non_finite(item, what)


def _reject_json_constants(value: str) -> None:
    """Reject NaN/Infinity literals while parsing JSON."""
    raise ValueError(f"unsupported JSON constant {value!r}; only finite numbers allowed")


def write_json_document(document: dict, path: Path, what: str) -> Path:
    """Write a dict as indented JSON after a finiteness sweep."""
    if not isinstance(document, dict):
        raise TypeError(f"{what} must be a dict, got {type(document).__name__}")
    _reject_non_finite(document, what)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    return path


def read_json_document(path: Path, what: str) -> dict:
    """Read a JSON file, rejecting missing/malformed/non-object content."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle, parse_constant=_reject_json_constants)
    except FileNotFoundError:
        raise FileNotFoundError(f"{what} file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{what} file is malformed JSON: {path} ({exc})") from exc
    if not isinstance(document, dict):
        raise ValueError(
            f"{what} file must contain a JSON object, got "
            f"{type(document).__name__}: {path}"
        )
    return document


def check_envelope(
    document: dict, path: Path, what: str, expected_format: str, version: int
) -> dict:
    """Validate a file envelope and return the inner payload dict."""
    actual_format = document.get("format")
    if actual_format != expected_format:
        raise ValueError(
            f"unsupported {what} format {actual_format!r} in {path}; "
            f"expected {expected_format!r}"
        )
    if document.get("version") != version:
        raise ValueError(
            f"unsupported {what} version {document.get('version')!r} in {path}; "
            f"this reader supports version {version}"
        )
    payload = document.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"{what} file has no payload object: {path}")
    return payload
