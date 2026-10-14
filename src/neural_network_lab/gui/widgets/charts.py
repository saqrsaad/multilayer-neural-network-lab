"""Matplotlib chart widgets for recorded training and evaluation results.

Display-only: every plotted value comes from recorded
``TrainingResult`` loss histories or ``EvaluationResult``
predictions/targets supplied by the owning views. No loss, gradient,
accuracy, or threshold mathematics beyond reusing the documented
classification threshold for labeling lives here.
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from neural_network_lab.application.evaluation_service import (
    CLASSIFICATION_THRESHOLD,
)


def _is_number(value: object) -> bool:
    """Return True for real numeric scalars, excluding booleans."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def validate_loss_history(history: object) -> list[float]:
    """Validate a recorded per-epoch loss history.

    Args:
        history: Non-empty sequence of finite loss values, one per epoch.

    Returns:
        The values as plain Python floats.

    Raises:
        TypeError: For non-sequence input or non-numeric entries.
        ValueError: For empty histories or non-finite values.
    """
    if isinstance(history, (str, bytes)) or not isinstance(history, (list, tuple)):
        try:
            items = list(history)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(
                "loss history must be a sequence of numbers, "
                f"got {type(history).__name__}"
            ) from exc
    else:
        items = list(history)
    if len(items) == 0:
        raise ValueError("loss history is empty; nothing to plot")
    values: list[float] = []
    for index, entry in enumerate(items):
        if not _is_number(entry):
            raise TypeError(
                f"loss history entry {index} is not a number: {entry!r}"
            )
        value = float(entry)
        if not math.isfinite(value):
            raise ValueError(
                f"loss history entry {index} is not finite: {entry!r}"
            )
        values.append(value)
    return values


def _as_single_output_series(values: object, name: str) -> list[float]:
    """Validate one per-sample series as a flat finite float list.

    Accepts flat sequences or ``(n, 1)`` arrays; anything else is
    rejected so multi-output or ragged data can never be misplotted.
    """
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence, got text")
    try:
        items = list(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"{name} must be a numeric sequence, got {type(values).__name__}"
        ) from exc
    try:
        array = np.asarray(items, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain only numbers: {exc}") from exc
    if array.ndim == 2:
        if array.shape[1] != 1:
            raise ValueError(
                f"{name} has {array.shape[1]} outputs per sample; "
                "charts support single-output models only"
            )
        array = array[:, 0]
    elif array.ndim != 1:
        raise ValueError(f"{name} must be one value per sample, got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} is empty; nothing to plot")
    bad = np.where(~np.isfinite(array))[0]
    if len(bad) > 0:
        raise ValueError(f"{name} entry {int(bad[0])} is not finite")
    return [float(value) for value in array]


def validate_prediction_inputs(
    probabilities: object, targets: object = None
) -> tuple[list[float], list[float] | None]:
    """Validate per-sample prediction (and optional target) series.

    Returns:
        ``(probabilities, targets)`` as plain float lists, with targets
        ``None`` when not supplied.

    Raises:
        TypeError: For non-sequence or non-numeric input.
        ValueError: For empty, ragged, multi-output, mismatched-length,
            or non-finite input.
    """
    probs = _as_single_output_series(probabilities, "probabilities")
    if targets is None:
        return probs, None
    actual = _as_single_output_series(targets, "targets")
    if len(actual) != len(probs):
        raise ValueError(
            f"targets length {len(actual)} does not match "
            f"probabilities length {len(probs)}"
        )
    return probs, actual


def _base_figure() -> tuple[Figure, object]:
    """Create a figure with one axes for chart widgets."""
    figure = Figure(figsize=(5, 3))
    return figure, figure.add_subplot(111)


class LossHistoryChart(FigureCanvas):
    """Line chart of recorded per-epoch training loss (MSE)."""

    def __init__(self, parent=None) -> None:
        """Create the chart showing its empty-state message."""
        figure, axes = _base_figure()
        self._figure = figure
        self._axes = axes
        super().__init__(self._figure)
        self.setObjectName("lossHistoryChart")
        self.show_message(
            "No training history yet. Train the model to record loss values."
        )

    def show_message(self, text: str) -> None:
        """Replace the chart with a centered informational message."""
        axes = self._axes
        axes.clear()
        axes.set_title("Training loss (MSE) by epoch")
        axes.text(
            0.5, 0.5, text, ha="center", va="center", wrap=True,
            transform=axes.transAxes,
        )
        axes.set_xticks([])
        axes.set_yticks([])
        self._figure.tight_layout()
        self.draw()

    def plot_history(self, history: object) -> bool:
        """Plot recorded losses at epochs 1..N, replacing prior content.

        Returns True when drawn, False with a message when the history
        is missing or invalid.
        """
        try:
            values = validate_loss_history(history)
        except (TypeError, ValueError) as exc:
            self.show_message(f"Cannot plot training loss: {exc}")
            return False
        epochs = list(range(1, len(values) + 1))
        axes = self._axes
        axes.clear()
        axes.plot(epochs, values, marker="o", label="training MSE")
        axes.set_title("Training loss (MSE) by epoch")
        axes.set_xlabel("epoch")
        axes.set_ylabel("training loss (MSE)")
        axes.legend()
        self._figure.tight_layout()
        self.draw()
        return True


class PredictionChart(FigureCanvas):
    """Per-sample predicted probabilities, classes, and actual targets."""

    def __init__(self, parent=None) -> None:
        """Create the chart showing its empty-state message."""
        figure, axes = _base_figure()
        self._figure = figure
        self._axes = axes
        super().__init__(self._figure)
        self.setObjectName("predictionChart")
        self.show_message(
            "No evaluation yet. Evaluate a labeled dataset to compare "
            "predictions against targets."
        )

    def show_message(self, text: str) -> None:
        """Replace the chart with a centered informational message."""
        axes = self._axes
        axes.clear()
        axes.set_title("Predicted vs actual targets")
        axes.text(
            0.5, 0.5, text, ha="center", va="center", wrap=True,
            transform=axes.transAxes,
        )
        axes.set_xticks([])
        axes.set_yticks([])
        self._figure.tight_layout()
        self.draw()

    def plot_predictions(
        self,
        probabilities: object,
        targets: object = None,
        *,
        threshold: float = CLASSIFICATION_THRESHOLD,
    ) -> bool:
        """Plot per-sample probabilities, classes, and optional targets.

        Returns True when drawn, False with a message when the inputs
        are missing, mismatched, or otherwise invalid.
        """
        try:
            probs, actual = validate_prediction_inputs(probabilities, targets)
            if not _is_number(threshold) or not math.isfinite(float(threshold)):
                raise ValueError(f"threshold is not finite: {threshold!r}")
            limit = float(threshold)
        except (TypeError, ValueError) as exc:
            self.show_message(f"Cannot plot predictions: {exc}")
            return False
        classes = [1.0 if value >= limit else 0.0 for value in probs]
        samples = list(range(1, len(probs) + 1))
        axes = self._axes
        axes.clear()
        axes.plot(samples, probs, marker="o", label="predicted probability")
        axes.plot(
            samples,
            classes,
            marker="s",
            linestyle="--",
            label=f"predicted class (threshold {limit})",
        )
        if actual is not None:
            axes.plot(samples, actual, marker="^", linestyle="",
                      label="actual target")
            axes.set_title("Predicted vs actual targets (labeled evaluation)")
        else:
            axes.set_title("Predicted probabilities and classes (no targets)")
        axes.set_xlabel("sample index")
        axes.set_ylabel("output value")
        axes.legend()
        self._figure.tight_layout()
        self.draw()
        return True
