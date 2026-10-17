"""Deterministic train/validation/test splitting of a labeled dataset.

Subsets are disjoint row ranges of one dataset, so every feature-target
row stays aligned and each sample lives in exactly one subset. A
validation or test row is therefore never seen by gradient updates:
the splitter hands the subsets to the caller, and the training loop in
:mod:`neural_network_lab.core.training` only ever runs forward passes
on validation data.

Small educational datasets (XOR with four rows, for instance) cannot
support meaningful validation and test subsets: a one-row subset can
only score 0% or 100%, so reporting metrics from it would mislead. When
a requested subset would be smaller than
:data:`MIN_MEANINGFUL_SAMPLES`, that subset is dropped, its rows join
the training subset, and the returned warning names what happened
instead of silently reporting unusable numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_network_lab.data.dataset import Dataset

#: Smallest subset still worth scoring: a single row can only score
#: 0% or 100%, so evaluation metrics from it would be misleading.
MIN_MEANINGFUL_SAMPLES = 2

#: Relative tolerance when checking that proportions sum to one.
RATIO_TOLERANCE = 1e-6

#: Selectable split presets as ``(label, train, validation, test)``.
#:
#: The first entry trains on every row: no held-out data exists, so no
#: validation or test metrics are reported for that configuration.
SPLIT_PRESETS: tuple[tuple[str, float, float, float], ...] = (
    ("No validation/test split (train on all rows)", 1.0, 0.0, 0.0),
    ("Train/validation/test = 50 / 25 / 25", 0.5, 0.25, 0.25),
    ("Train/validation/test = 60 / 20 / 20", 0.6, 0.2, 0.2),
    ("Train/validation/test = 80 / 10 / 10", 0.8, 0.1, 0.1),
)

#: Names of the three subsets, in split order.
SUBSET_NAMES = ("train", "validation", "test")

#: Slot holding the training rows; held-out subsets never precede it.
TRAIN_SLOT = 0


@dataclass(frozen=True)
class DatasetSplits:
    """One labeled dataset divided into train/validation/test subsets.

    ``validation`` and ``test`` are ``None`` when no meaningful subset
    of that kind exists; ``warning`` then explains the fallback so the
    UI can tell the user instead of implying measured metrics.
    """

    train: Dataset
    validation: Dataset | None
    test: Dataset | None
    warning: str | None = None

    def describe(self) -> str:
        """Return a one-line description of the resulting subsets."""
        parts = [f"training {self.train.n_samples}"]
        if self.validation is not None:
            parts.append(f"validation {self.validation.n_samples}")
        if self.test is not None:
            parts.append(f"test {self.test.n_samples}")
        return " / ".join(parts)


def validate_ratios(
    train_ratio: float, validation_ratio: float, test_ratio: float
) -> tuple[float, float, float]:
    """Validate split proportions and return them as floats.

    Raises:
        TypeError: If any proportion is not a number.
        ValueError: If a proportion is negative, not finite, or the
            three proportions do not sum to one.
    """
    ratios: list[float] = []
    for name, value in zip(
        SUBSET_NAMES, (train_ratio, validation_ratio, test_ratio)
    ):
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} proportion must be a number")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{name} proportion must be finite")
        if number < 0.0:
            raise ValueError(f"{name} proportion must not be negative")
        ratios.append(number)
    total = sum(ratios)
    if abs(total - 1.0) > RATIO_TOLERANCE:
        raise ValueError(
            "split proportions must sum to 1.0, got "
            f"{total!r} (training {ratios[0]!r}, validation {ratios[1]!r}, "
            f"test {ratios[2]!r})"
        )
    return tuple(ratios)


def _allocate_counts(sample_count: int, ratios: list[float]) -> list[int]:
    """Split ``sample_count`` rows into counts matching ``ratios``.

    Rows are allocated by largest fractional remainder, so the counts
    always sum to ``sample_count`` exactly.
    """
    raw = [sample_count * ratio for ratio in ratios]
    counts = [int(entry) for entry in raw]
    remaining = sample_count - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (raw[index] - counts[index], -index),
        reverse=True,
    )
    position = 0
    while remaining > 0 and order:
        index = order[position % len(order)]
        counts[index] += 1
        remaining -= 1
        position += 1
    return counts


def _subset_dataset(dataset: Dataset, row_indices: list[int]) -> Dataset:
    """Build a subset dataset from row indices, preserving row order.

    The declared task travels with the data, so a regression dataset
    stays a regression dataset in every subset.
    """
    features, targets = dataset.to_arrays()
    assert targets is not None  # labeled by the caller's check
    return Dataset(
        features[row_indices],
        targets[row_indices],
        target_kind=dataset.target_kind,
    )


def split_dataset(
    dataset: Dataset,
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int | None = None,
) -> DatasetSplits:
    """Split a labeled dataset into training, validation, test subsets.

    Args:
        dataset: Labeled dataset to divide; its rows are never modified.
        train_ratio: Proportion of rows used for gradient updates.
        validation_ratio: Proportion reserved for per-epoch model
            selection (forward passes only, never updates).
        test_ratio: Proportion reserved for evaluation after training
            (never used for updates or model selection).
        seed: Optional seed making the row-to-subset assignment
            reproducible across runs.

    Returns:
        :class:`DatasetSplits` whose ``validation``/``test`` members are
        ``None`` when those subsets could not be made meaningful; the
        ``warning`` field then names the reason.

    Raises:
        TypeError: If ``dataset`` is not a :class:`Dataset` or a
            proportion is not a number.
        ValueError: If the dataset is unlabeled, has too few rows for
            any training data, or the proportions are invalid.
    """
    if not isinstance(dataset, Dataset):
        raise TypeError(f"dataset must be a Dataset, got {type(dataset).__name__}")
    if not dataset.has_targets:
        raise ValueError(
            "splitting requires a labeled dataset with targets; "
            "validation and test metrics cannot be computed otherwise"
        )
    ratios = validate_ratios(train_ratio, validation_ratio, test_ratio)
    total_rows = dataset.n_samples
    if total_rows < 1:
        raise ValueError("dataset must contain at least one row")

    requested = [ratio > 0.0 for ratio in ratios]
    if not any(requested[1:]):
        # Training only: every row is used for gradient updates.
        return DatasetSplits(train=dataset, validation=None, test=None)

    row_order = list(range(total_rows))
    if seed is not None:
        row_order = [int(index) for index in np.random.default_rng(seed).permutation(total_rows)]

    warnings: list[str] = []
    while True:
        active = [ratio for ratio, want in zip(ratios, requested) if want]
        allocated = _allocate_counts(total_rows, active)
        counts = [0] * len(ratios)
        slot = 0
        for index, wanted in enumerate(requested):
            if wanted:
                counts[index] = allocated[slot]
                slot += 1
        if counts[TRAIN_SLOT] < 1:
            raise ValueError(
                f"only {total_rows} row(s): too few for the requested split"
            )
        oversized = next(
            (
                index
                for index, wanted in enumerate(requested)
                if index > TRAIN_SLOT
                and wanted
                and counts[index] < MIN_MEANINGFUL_SAMPLES
            ),
            None,
        )
        if oversized is None:
            break
        requested[oversized] = False
        warnings.append(
            f"the {SUBSET_NAMES[oversized]} subset would hold "
            f"{counts[oversized]} of {total_rows} rows, which cannot give "
            f"meaningful metrics (at least {MIN_MEANINGFUL_SAMPLES} are "
            "needed), so it was dropped and its rows were used for training"
        )
        if not any(requested[1:]):
            return DatasetSplits(
                train=dataset,
                validation=None,
                test=None,
                warning="; ".join(warnings),
            )
        remaining = [ratio for ratio, want in zip(ratios, requested) if want]
        remaining_total = sum(remaining)
        scaled = list(ratios)
        for index, wanted in enumerate(requested):
            scaled[index] = ratios[index] / remaining_total if wanted else 0.0
        ratios = tuple(scaled)

    subsets: dict[str, Dataset] = {}
    cursor = 0
    for index, wanted in enumerate(requested):
        if not wanted:
            continue
        selected = sorted(row_order[cursor : cursor + counts[index]])
        cursor += counts[index]
        subsets[SUBSET_NAMES[index]] = _subset_dataset(dataset, selected)
    return DatasetSplits(
        train=subsets[SUBSET_NAMES[TRAIN_SLOT]],
        validation=subsets.get("validation"),
        test=subsets.get("test"),
        warning="; ".join(warnings) if warnings else None,
    )
