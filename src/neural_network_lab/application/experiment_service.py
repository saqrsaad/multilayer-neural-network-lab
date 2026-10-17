"""Application-level service for creating and managing experiments.

An experiment captures one training run: a unique identifier,
timestamp, model configuration, training settings, dataset
information, the measured training history, and final evaluation
metrics. This module is persistence-free (saving/loading belongs to
the storage layer) and contains no GUI code. Training and evaluation
reuse :mod:`neural_network_lab.core.network`,
:mod:`neural_network_lab.data.dataset`, and
:mod:`neural_network_lab.application.evaluation_service`.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone

from neural_network_lab.application.evaluation_service import evaluate
from neural_network_lab.core.network import NeuralNetwork
from neural_network_lab.core.training import train_with_validation
from neural_network_lab.data.dataset import Dataset

#: Valid values for the ``mode`` of an experiment / training run.
TRAINING_MODES = ("continued", "from_scratch")


def _new_experiment_id() -> str:
    """Generate a unique experiment identifier."""
    return uuid.uuid4().hex


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ExperimentRecord:
    """Immutable record of a single training experiment.

    ``loss_history`` holds one measured post-update training MSE per
    epoch. ``test_mse``/``test_accuracy`` are ``None`` when no test
    dataset was evaluated. Dataset fields describe the data used, not
    the data itself.
    """

    experiment_id: str
    created_at: str
    mode: str
    layer_sizes: tuple[int, ...]
    learning_rate: float
    epochs: int
    train_dataset_name: str | None
    train_samples: int
    train_features: int
    train_outputs: int
    test_dataset_name: str | None
    loss_history: tuple[float, ...]
    final_train_loss: float
    test_mse: float | None
    test_accuracy: float | None
    validation_loss: float | None = None
    validation_samples: int | None = None
    best_epoch: int | None = None
    best_validation_loss: float | None = None
    test_samples: int | None = None
    model_path: str | None = None

    def to_dict(self) -> dict:
        """Return a JSON-compatible dictionary of this record."""
        return {
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "mode": self.mode,
            "layer_sizes": list(self.layer_sizes),
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "train_dataset_name": self.train_dataset_name,
            "train_samples": self.train_samples,
            "train_features": self.train_features,
            "train_outputs": self.train_outputs,
            "test_dataset_name": self.test_dataset_name,
            "loss_history": list(self.loss_history),
            "final_train_loss": self.final_train_loss,
            "test_mse": self.test_mse,
            "test_accuracy": self.test_accuracy,
            "validation_loss": self.validation_loss,
            "validation_samples": self.validation_samples,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "test_samples": self.test_samples,
            "model_path": self.model_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentRecord:
        """Rebuild a record from a plain dictionary, validating it fully.

        Args:
            data: Dictionary as produced by :meth:`to_dict`.

        Raises:
            TypeError: If ``data`` is not a dictionary or fields have
                the wrong types.
            ValueError: If required fields are missing or invalid
                (bad mode, dimensions, empty history, inconsistent
                final loss, non-finite metrics, ...).
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"experiment record must be a dict, got {type(data).__name__}"
            )

        def require(key: str):
            if key not in data:
                raise ValueError(f"experiment record is missing {key!r}")
            return data[key]

        experiment_id = require("experiment_id")
        created_at = require("created_at")
        mode = require("mode")
        layer_sizes = require("layer_sizes")
        learning_rate = require("learning_rate")
        epochs = require("epochs")
        train_dataset_name = require("train_dataset_name")
        train_samples = require("train_samples")
        train_features = require("train_features")
        train_outputs = require("train_outputs")
        test_dataset_name = require("test_dataset_name")
        loss_history = require("loss_history")
        final_train_loss = require("final_train_loss")
        test_mse = require("test_mse")
        test_accuracy = require("test_accuracy")

        if not isinstance(experiment_id, str) or not experiment_id:
            raise ValueError("experiment record needs a non-empty 'experiment_id'")
        if not isinstance(created_at, str) or not created_at:
            raise ValueError("experiment record needs a non-empty 'created_at'")
        try:
            datetime.fromisoformat(created_at)
        except ValueError as exc:
            raise ValueError(
                f"experiment record has invalid ISO timestamp: {created_at!r}"
            ) from exc
        if mode not in TRAINING_MODES:
            raise ValueError(
                f"experiment record has invalid mode {mode!r}; "
                f"expected one of {TRAINING_MODES}"
            )
        if (
            not isinstance(layer_sizes, (list, tuple))
            or len(layer_sizes) < 2
            or any(
                isinstance(size, bool)
                or not isinstance(size, int)
                or size < 1
                for size in layer_sizes
            )
        ):
            raise ValueError(
                "experiment record needs 'layer_sizes' with at least "
                "two positive integers"
            )
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(float(learning_rate))
            or float(learning_rate) <= 0.0
        ):
            raise ValueError(
                "experiment record needs a finite positive 'learning_rate'"
            )
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("experiment record needs a positive 'epochs' count")
        for key, value in (
            ("train_samples", train_samples),
            ("train_features", train_features),
            ("train_outputs", train_outputs),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"experiment record needs a positive {key!r}")
        for key, value in (
            ("train_dataset_name", train_dataset_name),
            ("test_dataset_name", test_dataset_name),
        ):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"experiment record {key!r} must be a string or None")
        if (
            not isinstance(loss_history, (list, tuple))
            or len(loss_history) == 0
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in loss_history
            )
        ):
            raise ValueError(
                "experiment record needs a non-empty 'loss_history' "
                "of finite numbers"
            )
        history = tuple(float(value) for value in loss_history)
        if len(history) != epochs:
            raise ValueError(
                f"experiment record 'epochs' ({epochs}) does not match "
                f"'loss_history' length ({len(history)})"
            )
        if (
            isinstance(final_train_loss, bool)
            or not isinstance(final_train_loss, (int, float))
            or not math.isfinite(float(final_train_loss))
        ):
            raise ValueError("experiment record needs a finite 'final_train_loss'")
        if not math.isclose(
            float(final_train_loss), history[-1], rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(
                "experiment record 'final_train_loss' does not match "
                "the last 'loss_history' entry"
            )
        for key, value in (("test_mse", test_mse), ("test_accuracy", test_accuracy)):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"experiment record {key!r} must be finite or None")
        if test_mse is not None and float(test_mse) < 0.0:
            raise ValueError("experiment record 'test_mse' must not be negative")
        if test_accuracy is not None and not 0.0 <= float(test_accuracy) <= 1.0:
            raise ValueError(
                "experiment record 'test_accuracy' must lie in [0, 1]"
            )
        # Optional fields introduced with the split-training workflow:
        # records saved before it simply omit them and load as None.
        def optional_metric(key: str) -> float | None:
            if key not in data or data[key] is None:
                return None
            value = data[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"experiment record {key!r} must be finite or None")
            if not math.isfinite(float(value)):
                raise ValueError(f"experiment record {key!r} must be finite or None")
            return float(value)

        def optional_samples(key: str) -> int | None:
            if key not in data or data[key] is None:
                return None
            value = data[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(
                    f"experiment record {key!r} must be a positive integer or None"
                )
            return int(value)

        best_epoch = optional_samples("best_epoch")
        if best_epoch is not None and epochs > 0 and best_epoch > epochs:
            raise ValueError(
                "experiment record 'best_epoch' exceeds its 'epochs' count"
            )
        model_path = data.get("model_path")
        if model_path is not None and not isinstance(model_path, str):
            raise ValueError("experiment record 'model_path' must be a string or None")
        return cls(
            experiment_id=experiment_id,
            created_at=created_at,
            mode=mode,
            layer_sizes=tuple(int(size) for size in layer_sizes),
            learning_rate=float(learning_rate),
            epochs=epochs,
            train_dataset_name=train_dataset_name,
            train_samples=train_samples,
            train_features=train_features,
            train_outputs=train_outputs,
            test_dataset_name=test_dataset_name,
            loss_history=history,
            final_train_loss=float(final_train_loss),
            test_mse=None if test_mse is None else float(test_mse),
            test_accuracy=None if test_accuracy is None else float(test_accuracy),
            validation_loss=optional_metric("validation_loss"),
            validation_samples=optional_samples("validation_samples"),
            best_epoch=best_epoch,
            best_validation_loss=optional_metric("best_validation_loss"),
            test_samples=optional_samples("test_samples"),
            model_path=model_path,
        )


@dataclass(frozen=True)
class ExperimentComparison:
    """Outcome of comparing two compatible experiment records.

    Deltas are ``second - first``: a positive ``train_loss_change``
    means the second run ended with higher training loss, while a
    positive ``test_accuracy_change`` means higher accuracy. Test
    deltas are ``None`` when either record has no test metrics.
    """

    first_id: str
    second_id: str
    train_loss_change: float
    test_mse_change: float | None
    test_accuracy_change: float | None


def _compatibility_error(first: ExperimentRecord, second: ExperimentRecord) -> str | None:
    """Return the reason two records are incompatible, or ``None``."""
    if first.layer_sizes != second.layer_sizes:
        return (
            f"layer_sizes differ: {first.layer_sizes} vs {second.layer_sizes}"
        )
    for key in ("train_dataset_name", "train_samples", "train_features", "train_outputs"):
        if getattr(first, key) != getattr(second, key):
            return f"training data differs ({key}: {getattr(first, key)!r} vs {getattr(second, key)!r})"
    if first.test_dataset_name != second.test_dataset_name:
        return (
            "test datasets differ: "
            f"{first.test_dataset_name!r} vs {second.test_dataset_name!r}"
        )
    return None


def compare_experiments(
    first: ExperimentRecord, second: ExperimentRecord
) -> ExperimentComparison:
    """Compare two compatible experiment records.

    Compatibility rule (explicit): both records must share the same
    ``layer_sizes``, the same training-data identity (dataset name,
    sample/feature/output counts), and the same test dataset name (or
    both have none). Otherwise a :exc:`ValueError` names the mismatch.

    Args:
        first: Baseline record.
        second: Record to compare against the baseline.

    Returns:
        :class:`ExperimentComparison` with ``second - first`` deltas.

    Raises:
        TypeError: If either argument is not an :class:`ExperimentRecord`.
        ValueError: If the records are incompatible.
    """
    if not isinstance(first, ExperimentRecord):
        raise TypeError(
            f"first must be an ExperimentRecord, got {type(first).__name__}"
        )
    if not isinstance(second, ExperimentRecord):
        raise TypeError(
            f"second must be an ExperimentRecord, got {type(second).__name__}"
        )
    reason = _compatibility_error(first, second)
    if reason is not None:
        raise ValueError(f"experiments are not comparable: {reason}")
    if first.test_mse is not None and second.test_mse is not None:
        test_mse_change: float | None = second.test_mse - first.test_mse
    else:
        test_mse_change = None
    if first.test_accuracy is not None and second.test_accuracy is not None:
        test_accuracy_change: float | None = (
            second.test_accuracy - first.test_accuracy
        )
    else:
        test_accuracy_change = None
    return ExperimentComparison(
        first_id=first.experiment_id,
        second_id=second.experiment_id,
        train_loss_change=second.final_train_loss - first.final_train_loss,
        test_mse_change=test_mse_change,
        test_accuracy_change=test_accuracy_change,
    )


class ExperimentStore:
    """In-memory registry of experiment records (no file I/O here)."""

    def __init__(self) -> None:
        """Create an empty store."""
        self._records: dict[str, ExperimentRecord] = {}

    def __len__(self) -> int:
        """Number of stored records."""
        return len(self._records)

    def add(self, record: ExperimentRecord) -> None:
        """Store a record; duplicate ids are rejected, never overwritten.

        Raises:
            TypeError: If ``record`` is not an :class:`ExperimentRecord`.
            ValueError: If the id is already stored.
        """
        if not isinstance(record, ExperimentRecord):
            raise TypeError(
                f"record must be an ExperimentRecord, got {type(record).__name__}"
            )
        if record.experiment_id in self._records:
            raise ValueError(
                f"experiment {record.experiment_id!r} is already stored"
            )
        self._records[record.experiment_id] = record

    def list_experiments(self) -> list[ExperimentRecord]:
        """Return all records in insertion order."""
        return list(self._records.values())

    def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        """Return the record with an id, or raise :exc:`KeyError`.

        Raises:
            KeyError: If no stored record has ``experiment_id``.
        """
        try:
            return self._records[experiment_id]
        except KeyError:
            raise KeyError(f"unknown experiment id: {experiment_id!r}") from None


def run_experiment(
    network: NeuralNetwork,
    train_dataset: Dataset,
    *,
    epochs: int,
    learning_rate: float,
    mode: str = "continued",
    seed: int | None = None,
    train_dataset_name: str | None = None,
    test_dataset: Dataset | None = None,
    test_dataset_name: str | None = None,
    validation_dataset: Dataset | None = None,
    validation_dataset_name: str | None = None,
) -> ExperimentRecord:
    """Train a network and record the experiment.

    The network is trained in place with the existing core APIs: the
    ``continued`` mode preserves current parameters, ``from_scratch``
    explicitly reinitializes them first. When a validation dataset is
    given, its rows are scored read-only after each epoch and the epoch
    with the lowest validation loss is restored on the network — the
    final epoch is never assumed to be the best. The test dataset, when
    given, is evaluated read-only afterwards and never influences
    training or model selection.

    Args:
        network: Network to train in place.
        train_dataset: Labeled training dataset.
        epochs: Positive number of full-batch epochs.
        learning_rate: Positive, finite step size.
        mode: ``"continued"`` or ``"from_scratch"``.
        seed: Seed for ``"from_scratch"`` reinitialization; must be
            ``None`` for ``"continued"``.
        train_dataset_name: Optional label identifying the training data.
        test_dataset: Optional labeled test dataset (never used to
            update parameters).
        test_dataset_name: Optional label identifying the test data;
            only meaningful together with ``test_dataset``.

    Returns:
        The new :class:`ExperimentRecord` (with a fresh unique id and
        UTC timestamp).

    Raises:
        TypeError: For wrong argument types.
        ValueError: For an unknown mode, an unlabeled training set, a
            test name without a test dataset, a seed with
            ``"continued"``, or any training/evaluation validation
            failure (dimensions, binary targets, ...).
    """
    if not isinstance(network, NeuralNetwork):
        raise TypeError(
            f"network must be a NeuralNetwork, got {type(network).__name__}"
        )
    if not isinstance(train_dataset, Dataset):
        raise TypeError(
            f"train_dataset must be a Dataset, got {type(train_dataset).__name__}"
        )
    if mode not in TRAINING_MODES:
        raise ValueError(
            f"mode must be one of {TRAINING_MODES}, got {mode!r}"
        )
    if train_dataset_name is not None and not isinstance(train_dataset_name, str):
        raise TypeError("train_dataset_name must be a string or None")
    if test_dataset is not None and not isinstance(test_dataset, Dataset):
        raise TypeError(
            f"test_dataset must be a Dataset or None, "
            f"got {type(test_dataset).__name__}"
        )
    if test_dataset is None and test_dataset_name is not None:
        raise ValueError(
            "test_dataset_name requires a test_dataset to identify"
        )
    if test_dataset_name is not None and not isinstance(test_dataset_name, str):
        raise TypeError("test_dataset_name must be a string or None")
    if mode == "continued" and seed is not None:
        raise ValueError("seed applies only to mode 'from_scratch'")
    if not train_dataset.has_targets:
        raise ValueError("training requires a labeled dataset with targets")
    validation_features = None
    validation_targets = None
    validation_samples: int | None = None
    if validation_dataset is not None:
        if not isinstance(validation_dataset, Dataset):
            raise TypeError(
                "validation_dataset must be a Dataset or None, "
                f"got {type(validation_dataset).__name__}"
            )
        if not validation_dataset.has_targets:
            raise ValueError(
                "validation requires a labeled dataset with targets"
            )
        if validation_dataset_name is None:
            raise ValueError(
                "validation_dataset_name requires a validation_dataset to identify"
            )
        validation_features, validation_targets = validation_dataset.to_arrays()
        validation_samples = validation_dataset.n_samples
    if validation_dataset is None and validation_dataset_name is not None:
        raise ValueError(
            "validation_dataset_name requires a validation_dataset to identify"
        )
    if validation_dataset_name is not None and not isinstance(
        validation_dataset_name, str
    ):
        raise TypeError("validation_dataset_name must be a string or None")
    features, targets = train_dataset.to_arrays()
    assert targets is not None  # narrowed by has_targets above
    outcome = train_with_validation(
        network,
        features,
        targets,
        epochs=epochs,
        learning_rate=learning_rate,
        mode=mode,
        validation_features=validation_features,
        validation_targets=validation_targets,
        reinitialize=(mode == "from_scratch"),
        seed=seed if mode == "from_scratch" else None,
    )
    test_mse: float | None = None
    test_accuracy: float | None = None
    test_samples: int | None = None
    if test_dataset is not None:
        evaluation = evaluate(
            network, test_dataset, dataset_name=test_dataset_name
        )
        test_mse = evaluation.mse
        test_accuracy = evaluation.accuracy
        test_samples = test_dataset.n_samples
    return ExperimentRecord(
        experiment_id=_new_experiment_id(),
        created_at=_utc_now_iso(),
        mode=mode,
        layer_sizes=network.layer_sizes,
        learning_rate=outcome.learning_rate,
        epochs=outcome.epochs_completed,
        train_dataset_name=train_dataset_name,
        train_samples=train_dataset.n_samples,
        train_features=train_dataset.n_features,
        train_outputs=int(train_dataset.n_outputs or 0),
        test_dataset_name=test_dataset_name,
        loss_history=tuple(outcome.loss_history),
        final_train_loss=outcome.final_loss,
        test_mse=test_mse,
        test_accuracy=test_accuracy,
        validation_loss=outcome.final_validation_loss,
        validation_samples=validation_samples,
        best_epoch=outcome.best_epoch,
        best_validation_loss=outcome.best_validation_loss,
        test_samples=test_samples,
    )
