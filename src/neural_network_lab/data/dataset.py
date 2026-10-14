"""Numerical dataset representation and validation.

Conventions follow ``docs/PROJECT_SPEC.md`` and ``docs/ARCHITECTURE.md``:
the data layer represents datasets, validates features and targets, and
prepares arrays for the neural network. It never trains the network and
never imports PySide6.

Labeled datasets (features plus targets) support supervised training
and evaluation. Unlabeled datasets (features only, targets explicitly
``None``) still allow prediction, but supervised loss and accuracy
cannot be calculated from them.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_feature_array(features: ArrayLike) -> NDArray[np.float64]:
    """Validate input features and return them as a float array copy."""
    try:
        arr = np.array(features, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"features must be numeric array-like: {exc}") from exc
    if arr.ndim != 2:
        raise ValueError(
            "features must be 2D with shape (number_of_samples, "
            f"number_of_features), got shape {arr.shape}"
        )
    if arr.shape[0] < 1:
        raise ValueError("features must contain at least one sample")
    if arr.shape[1] < 1:
        raise ValueError("features must contain at least one feature column")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "features must contain only finite values (no NaN or inf); "
            "invalid rows are rejected, never silently discarded"
        )
    return arr


def _as_target_array(
    targets: ArrayLike, n_samples: int
) -> NDArray[np.float64]:
    """Validate target values against a feature sample count."""
    try:
        arr = np.array(targets, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"targets must be numeric array-like: {exc}") from exc
    if arr.ndim != 2:
        raise ValueError(
            "targets must be 2D with shape (number_of_samples, "
            f"number_of_outputs), got shape {arr.shape}; dimensions are "
            "never reshaped silently"
        )
    if arr.shape[0] < 1:
        raise ValueError("targets must contain at least one sample")
    if arr.shape[0] != n_samples:
        raise ValueError(
            f"targets row count {arr.shape[0]} != features row count {n_samples}"
        )
    if arr.shape[1] < 1:
        raise ValueError("targets must contain at least one output column")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "targets must contain only finite values (no NaN or inf); "
            "invalid rows are rejected, never silently discarded"
        )
    return arr


def _validate_positive_int(value: int, name: str) -> int:
    """Validate a network dimension and return it as an int."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    if int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class Dataset:
    """In-memory numerical dataset with optional targets.

    Features are stored as a ``(number_of_samples, number_of_features)``
    float array. Targets, when present, are stored as a
    ``(number_of_samples, number_of_outputs)`` float array; when absent
    they are represented explicitly as ``None``.
    """

    def __init__(
        self, features: ArrayLike, targets: ArrayLike | None = None
    ) -> None:
        """Create a dataset from features and optional targets.

        Args:
            features: 2D numeric array-like with at least one sample
                and one feature column.
            targets: Optional 2D numeric array-like whose row count
                matches ``features``. ``None`` means the dataset is
                unlabeled (prediction-only).

        Raises:
            TypeError: For non-numeric features/targets or a non-array
                ``targets`` that is not ``None``.
            ValueError: For wrong dimensions, row-count mismatch,
                empty data, or NaN/inf values.
        """
        feature_arr = _as_feature_array(features)
        if targets is None:
            target_arr: NDArray[np.float64] | None = None
        else:
            target_arr = _as_target_array(targets, feature_arr.shape[0])
        self._features = feature_arr
        self._targets = target_arr

    @property
    def features(self) -> NDArray[np.float64]:
        """Copy of the feature array ``(number_of_samples, number_of_features)``."""
        return self._features.copy()

    @property
    def targets(self) -> NDArray[np.float64] | None:
        """Copy of the target array, or ``None`` when unlabeled."""
        return None if self._targets is None else self._targets.copy()

    @property
    def has_targets(self) -> bool:
        """Whether this dataset carries target values."""
        return self._targets is not None

    @property
    def n_samples(self) -> int:
        """Number of samples (rows)."""
        return int(self._features.shape[0])

    @property
    def n_features(self) -> int:
        """Number of feature columns."""
        return int(self._features.shape[1])

    @property
    def n_outputs(self) -> int | None:
        """Number of target columns, or ``None`` when unlabeled."""
        return None if self._targets is None else int(self._targets.shape[1])

    def __len__(self) -> int:
        """Number of samples (rows)."""
        return self.n_samples

    def __repr__(self) -> str:
        """Concise summary without dumping array contents."""
        targets = f", targets={self._targets.shape}" if self.has_targets else ""
        return (
            f"Dataset(features={self._features.shape}{targets})"
        )

    def to_arrays(self) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        """Return ``(features, targets)`` copies for network use.

        Targets are ``None`` for unlabeled datasets; callers requiring
        supervision must check :attr:`has_targets` first.
        """
        return self.features, self.targets

    def validate_compatible_with(
        self, n_features: int, n_outputs: int | None = None
    ) -> None:
        """Check compatibility with network input/output sizes.

        Args:
            n_features: Required network input size.
            n_outputs: Required network output size, or ``None`` to
                check features only (e.g. prediction-only use).

        Raises:
            TypeError: For non-integer sizes.
            ValueError: If feature columns mismatch ``n_features``; if
                ``n_outputs`` is given but the dataset is unlabeled; or
                if target columns mismatch ``n_outputs``.
        """
        expected_features = _validate_positive_int(n_features, "n_features")
        if self.n_features != expected_features:
            raise ValueError(
                f"dataset has {self.n_features} feature columns, "
                f"but the network expects {expected_features}"
            )
        if n_outputs is None:
            return
        expected_outputs = _validate_positive_int(n_outputs, "n_outputs")
        if not self.has_targets:
            raise ValueError(
                "dataset has no targets, so supervised use requiring "
                f"{expected_outputs} output(s) is not possible"
            )
        assert self.n_outputs is not None  # narrowed by has_targets
        if self.n_outputs != expected_outputs:
            raise ValueError(
                f"dataset has {self.n_outputs} target columns, "
                f"but the network expects {expected_outputs}"
            )

    def is_compatible_with(
        self, n_features: int, n_outputs: int | None = None
    ) -> bool:
        """Return whether :meth:`validate_compatible_with` would pass."""
        try:
            self.validate_compatible_with(n_features, n_outputs)
        except (TypeError, ValueError):
            return False
        return True
